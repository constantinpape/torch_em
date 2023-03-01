import os
from glob import glob

import numpy as np
import torch
import torch_em
import torch_em.shallow2deep as shallow2deep
from elf.io import open_file
from torch_em.model import UNet3d
from torch_em.data.datasets.lucchi import _require_lucchi_data
from torch_em.data.datasets.uro_cell import _require_urocell_data


DATA_ROOT = "/scratch/pape/s2d-mitochondria"
DATASETS = ["lucchi", "platy", "urocell"]


def normalize_datasets(datasets):
    wrong_ds = list(set(datasets) - set(DATASETS))
    if wrong_ds:
        raise ValueError(f"Unkown datasets: {wrong_ds}. Only {DATASETS} are supported")
    datasets = list(sorted(datasets))
    return datasets


def require_ds(dataset):
    os.makedirs(DATA_ROOT, exist_ok=True)
    data_path = os.path.join(DATA_ROOT, dataset)
    if dataset == "lucchi":
        _require_lucchi_data(data_path, download=True)
        paths = [
            os.path.join(data_path, "lucchi_train.h5"),
        ]
        assert all(os.path.exists(pp) for pp in paths), f"{paths}"
        raw_key, label_key = "raw", "labels"
    elif dataset == "platy":
        raise NotImplementedError
    elif dataset == "urocell":
        _require_urocell_data(data_path, download=True)
        paths = glob(os.path.join(data_path, "*.h5"))
        paths.sort()
        paths = [pp for pp in paths if "labels/mito" in open_file(pp, "r")]
        paths = paths[:-1]
        raw_key, label_key = "raw", "labels/mito"
    return paths, raw_key, label_key


def require_rfs_ds(dataset, n_rfs, sampling_strategy):
    out_folder = os.path.join(DATA_ROOT, f"rfs3d-{sampling_strategy}", dataset)
    os.makedirs(out_folder, exist_ok=True)
    if len(glob(os.path.join(out_folder, "*.pkl"))) == n_rfs:
        return

    patch_shape_min = [64, 64, 64]
    patch_shape_max = [96, 128, 128]

    raw_transform = torch_em.transform.raw.normalize
    label_transform = shallow2deep.ForegroundTransform(ndim=3)

    paths, raw_key, label_key = require_ds(dataset)
    if sampling_strategy == "vanilla":
        shallow2deep.prepare_shallow2deep(
            raw_paths=paths, raw_key=raw_key, label_paths=paths, label_key=label_key,
            patch_shape_min=patch_shape_min, patch_shape_max=patch_shape_max,
            n_forests=args.n_rfs, n_threads=args.n_threads,
            output_folder=out_folder, ndim=3,
            raw_transform=raw_transform, label_transform=label_transform,
            is_seg_dataset=True
        )
    else:
        sampling_kwargs = {}
        if sampling_strategy == "worst_tiles":
            sampling_kwargs["tile_shape"] = [16, 16, 16]
        shallow2deep.prepare_shallow2deep_advanced(
            raw_paths=paths, raw_key=raw_key, label_paths=paths, label_key=label_key,
            patch_shape_min=patch_shape_min, patch_shape_max=patch_shape_max,
            n_forests=args.n_rfs, n_threads=args.n_threads,
            forests_per_stage=25, sample_fraction_per_stage=0.05,
            output_folder=out_folder, ndim=3,
            raw_transform=raw_transform, label_transform=label_transform,
            is_seg_dataset=True, sampling_strategy=sampling_strategy,
            sampling_kwargs=sampling_kwargs,
        )


def require_rfs(datasets, n_rfs, sampling_strategy):
    for ds in datasets:
        require_rfs_ds(ds, n_rfs, sampling_strategy)


def get_ds(file_pattern, rf_pattern, n_samples, label_key="labels",
           path_selection=None, check_labels=False):
    raw_transform = torch_em.transform.raw.normalize
    label_transform = torch_em.transform.BoundaryTransform(ndim=3, add_binary_target=True)
    patch_shape = (64, 256, 256)

    paths = glob(file_pattern)
    paths.sort()
    if check_labels:
        paths = [pp for pp in paths if label_key in open_file(pp, "r")]
    if path_selection:
        paths = paths[path_selection]
    assert len(paths) > 0

    rf_paths = glob(rf_pattern)
    rf_paths.sort()
    assert len(rf_paths) > 0

    raw_key = "raw"
    return shallow2deep.shallow2deep_dataset.get_shallow2deep_dataset(
        paths, raw_key, paths, label_key, rf_paths,
        patch_shape=patch_shape,
        raw_transform=raw_transform,
        label_transform=label_transform,
        n_samples=n_samples, ndim=3,
    )


def get_loader(args, split, dataset_names):
    datasets = []
    n_samples = 500 if split == "train" else 25
    if "lucchi" in dataset_names:
        ds_name = "kasthuri"
        # we need to use the test split here, because val is too small in z
        split_ = "test" if split == "val" else split
        file_pattern = os.path.join(DATA_ROOT, ds_name, f"*_{split_}.h5")
        rf_pattern = os.path.join(DATA_ROOT, f"rfs3d-{args.sampling_strategy}", ds_name, "*.pkl")
        datasets.append(get_ds(file_pattern, rf_pattern, n_samples))
    if "urocell" in dataset_names:
        ds_name = "urocell"
        file_pattern = os.path.join(DATA_ROOT, ds_name, "*.h5")
        rf_pattern = os.path.join(DATA_ROOT, f"rfs3d-{args.sampling_strategy}", ds_name, "*.pkl")
        path_selection = np.s_[:-1] if split == "train" else np.s_[-1:]
        datasets.append(get_ds(
            file_pattern, rf_pattern, n_samples,
            label_key="labels/mito", path_selection=path_selection, check_labels=True
        ))
    ds = torch_em.data.concat_dataset.ConcatDataset(*datasets) if len(datasets) > 1 else datasets[0]
    loader = torch.utils.data.DataLoader(
        ds, shuffle=True, batch_size=args.batch_size, num_workers=24
    )
    loader.shuffle = True
    return loader


def train_shallow2deep(args):
    datasets = normalize_datasets(args.datasets)
    name = f"s2d-em-mitos-{'_'.join(datasets)}-3d-{args.sampling_strategy}"
    require_rfs(datasets, args.n_rfs, args.sampling_strategy)

    model = UNet3d(in_channels=1, out_channels=2, final_activation="Sigmoid",
                   depth=4, initial_features=32)

    train_loader = get_loader(args, "train", datasets)
    val_loader = get_loader(args, "val", datasets)
    loss = torch_em.loss.DiceLoss()
    if "kasthuri" in datasets:
        loss = torch_em.loss.wrapper.LossWrapper(
            loss, torch_em.loss.wrapper.MaskIgnoreLabel()
        )

    trainer = torch_em.default_segmentation_trainer(
        name, model, train_loader, val_loader, learning_rate=1.0e-4,
        loss=loss, device=args.device, log_image_interval=50
    )
    trainer.fit(args.n_iterations)


def check(args, train=True, val=True, n_images=2):
    from torch_em.util.debug import check_loader
    datasets = normalize_datasets(args.datasets)
    if train:
        print("Check train loader")
        loader = get_loader(args, "train", datasets)
        check_loader(loader, n_images)
    if val:
        print("Check val loader")
        loader = get_loader(args, "val", datasets)
        check_loader(loader, n_images)


if __name__ == "__main__":
    parser = torch_em.util.parser_helper(require_input=False)
    parser.add_argument("--datasets", "-d", nargs="+", default=DATASETS)
    parser.add_argument("--n_rfs", type=int, default=500)
    parser.add_argument("--n_threads", type=int, default=32)
    parser.add_argument("--sampling_strategy", "-s", default="worst_tiles")
    args = parser.parse_args()
    if args.check:
        check(args, n_images=5, val=False)
    else:
        train_shallow2deep(args)
