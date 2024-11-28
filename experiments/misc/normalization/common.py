import os
from glob import glob

import h5py
import z5py
import numpy as np
import imageio.v3 as imageio

from torchvision import transforms

from torch_em.model import UNet2d, UNet3d
from torch_em.data import MinTwoInstanceSampler, datasets
from torch_em.transform import raw as fetch_transforms

from micro_sam.evaluation.livecell import _get_livecell_paths


if os.path.exists("/scratch/share/cidas"):
    ROOT = "/scratch/share/cidas/cca/data"
    SAVE_DIR = "/scratch/share/cidas/cca/test/verify_normalization"
else:
    ROOT = "/media/anwai/ANWAI/data/"
    SAVE_DIR = "/media/anwai/ANWAI/test/verify_normalization"


def get_model(dataset, task, norm):
    out_chans = 1 if task == "binary" or dataset == "plantseg" else 2
    _model_class = UNet2d if dataset == "livecell" else UNet3d

    model = _model_class(
        in_channels=1,
        out_channels=out_chans,
        initial_features=64,
        depth=4,
        final_activation="Sigmoid",
        norm=norm,
    )
    return model


def get_experiment_name(dataset, task, norm, model_choice):
    cfg = "2d" if dataset == "livecell" else "3d"
    name = f"{dataset}_{model_choice}{cfg}_{norm}_{task}"
    return name


class MultipleRawTransforms:
    def __init__(
        self, p=0.3, norm=None, blur_kwargs={}, gaussian_kwargs={},
        poisson_kwargs={}, additive_poisson_kwargs={}, contrast_kwargs={}
    ):
        self.norm = fetch_transforms.normalize_percentile if norm is None else norm
        augs = [self.norm]

        if gaussian_kwargs is not None:
            augs.append(transforms.RandomApply([fetch_transforms.GaussianBlur(**blur_kwargs)], p=p))
        if poisson_kwargs is not None:
            augs.append(transforms.RandomApply([fetch_transforms.PoissonNoise(**poisson_kwargs)], p=p/2))
        if additive_poisson_kwargs is not None:
            augs.append(
                transforms.RandomApply([fetch_transforms.AdditivePoissonNoise(**additive_poisson_kwargs)], p=p/2)
            )
        if gaussian_kwargs is not None:
            augs.append(transforms.RandomApply([fetch_transforms.AdditiveGaussianNoise(**gaussian_kwargs)], p=p/2))
        if contrast_kwargs is not None:
            aug2 = transforms.RandomApply([fetch_transforms.RandomContrast(**contrast_kwargs)], p)

        self.raw_transform = fetch_transforms.get_raw_transform(
            normalizer=self.norm,
            augmentation1=transforms.Compose(augs),
            augmentation2=aug2
        )

    def __call__(self, raw):
        raw = raw[None]  # NOTE: reason for doing this is to add an empty dimension to work with torch transforms.
        raw = self.raw_transform(raw)
        return raw


def get_dataloaders(dataset, task):
    assert task in ["binary", "boundaries"]
    sampler = MinTwoInstanceSampler()

    loader_kwargs = {
        "num_workers": 16, "download": True, "sampler": sampler,
        "raw_transform": MultipleRawTransforms(
            p=0.3,
            poisson_kwargs=None if dataset == "livecell" else {},
            additive_poisson_kwargs={"lam": (0.0, 0.2)} if dataset == "livecell" else {},
        ),
    }

    if dataset == "livecell":
        train_loader = datasets.get_livecell_loader(
            path=os.path.join(ROOT, "livecell"),
            split="train",
            patch_shape=(512, 512),
            batch_size=2,
            binary=True if task == "binary" else False,
            boundaries=True if task == "boundaries" else False,
            **loader_kwargs
        )
        val_loader = datasets.get_livecell_loader(
            path=os.path.join(ROOT, "livecell"),
            split="val",
            patch_shape=(512, 512),
            batch_size=1,
            binary=True if task == "binary" else False,
            boundaries=True if task == "boundaries" else False,
            **loader_kwargs
        )
    elif dataset == "plantseg":
        train_loader = datasets.get_plantseg_loader(
            path=os.path.join(ROOT, "plantseg"),
            name="root",
            split="train",
            patch_shape=(32, 512, 512),
            batch_size=2,
            boundaries=True if task == "boundaries" else False,
            **loader_kwargs
        )

        val_loader = datasets.get_plantseg_loader(
            path=os.path.join(ROOT, "plantseg"),
            name="root",
            split="val",
            patch_shape=(32, 512, 512),
            batch_size=1,
            boundaries=True if task == "boundaries" else False,
            **loader_kwargs
        )
    elif dataset == "mitoem":
        train_loader = datasets.get_mitoem_loader(
            path=os.path.join(ROOT, "mitoem"),
            splits="train",
            patch_shape=(32, 512, 512),
            batch_size=2,
            binary=True if task == "binary" else False,
            boundaries=True if task == "boundaries" else False,
            **loader_kwargs
        )

        val_loader = datasets.get_mitoem_loader(
            path=os.path.join(ROOT, "mitoem"),
            splits="val",
            patch_shape=(32, 512, 512),
            batch_size=1,
            binary=True if task == "binary" else False,
            boundaries=True if task == "boundaries" else False,
            **loader_kwargs
        )
    elif dataset == "gonuclear":
        train_loader = datasets.get_gonuclear_loader(
            path=os.path.join(ROOT, "gonuclear"),
            patch_shape=(32, 512, 512),
            batch_size=2,
            segmentation_task="nuclei",
            binary=True if task == "binary" else False,
            boundaries=True if task == "boundaries" else False,
            sample_ids=[1135, 1136, 1137],
            **loader_kwargs
        )
        val_loader = datasets.get_gonuclear_loader(
            path=os.path.join(ROOT, "gonuclear"),
            patch_shape=(32, 512, 512),
            batch_size=1,
            segmentation_task="nuclei",
            binary=True if task == "binary" else False,
            boundaries=True if task == "boundaries" else False,
            sample_ids=[1139],
            **loader_kwargs
        )
    else:
        raise ValueError(f"{dataset} is not a valid dataset choice for this experiment.")

    return train_loader, val_loader


def dice_score(gt, seg, eps=1e-7):
    nom = 2 * np.sum(gt * seg)
    denom = np.sum(gt) + np.sum(seg)
    score = float(nom) / float(denom + eps)
    return score


def get_test_images(dataset):
    if dataset == "livecell":
        image_paths, gt_paths = _get_livecell_paths(input_folder=os.path.join(ROOT, "livecell"), split="test")
        return image_paths, gt_paths
    else:
        if dataset == "gonuclear":
            datasets.gonuclear.get_gonuclear_data(path=os.path.join(ROOT, "gonuclear"), download=True)
            volume_paths = [os.path.join(ROOT, "gonuclear", "gonuclear_datasets", "1170.h5")]
        elif dataset == "plantseg":
            datasets.plantseg.get_plantseg_data(
                path=os.path.join(ROOT, "plantseg"), download=True, name="root", split="test"
            )
            volume_paths = sorted(glob(os.path.join(ROOT, "plantseg", "root_test", "*.h5")))
        elif dataset == "mitoem":
            volume_paths = sorted(glob(os.path.join(ROOT, "mitoem", "*_val.n5")))

        return volume_paths, volume_paths


def _load_image(input_path, key=None):
    if key is None:
        image = imageio.imread(input_path)

    else:
        if input_path.endswith(".h5"):
            with h5py.File(input_path, "r") as f:
                image = f[key][:]
        else:
            with z5py.File(input_path, "r") as f:
                image = f[key][:]

    return image
