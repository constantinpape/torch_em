import os
from glob import glob
from typing import Union, Tuple

import torch_em

from .. import util
from ... import ImageCollectionDataset


KAGGLE_DATASET_NAME = "vbookshelf/pneumothorax-chest-xray-images-and-masks"
CHECKSUM = "1ade68d31adb996c531bb686fb9d02fe11876ddf6f25594ab725e18c69d81538"


def get_siim_acr_data(path, download):
    os.makedirs(path, exist_ok=True)

    data_dir = os.path.join(path, "siim-acr-pneumothorax")
    if os.path.exists(data_dir):
        return data_dir

    util.download_source_kaggle(path=path, dataset_name=KAGGLE_DATASET_NAME, download=download)

    zip_path = os.path.join(path, "pneumothorax-chest-xray-images-and-masks.zip")
    util._check_checksum(path=zip_path, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def _get_siim_acr_paths(path, split, download):
    data_dir = get_siim_acr_data(path=path, download=download)

    assert split in ["train", "test"], f"'{split}' is not a valid split."

    image_paths = sorted(glob(os.path.join(data_dir, "png_images", f"*_{split}_*.png")))
    gt_paths = sorted(glob(os.path.join(data_dir, "png_masks", f"*_{split}_*.png")))

    return image_paths, gt_paths


def get_siim_acr_dataset(
    path: Union[os.PathLike, str],
    split: str,
    patch_shape: Tuple[int, int],
    download: bool = False,
    **kwargs
):
    image_paths, gt_paths = _get_siim_acr_paths(path=path, split=split, download=download)

    dataset = ImageCollectionDataset(
        raw_image_paths=image_paths, label_image_paths=gt_paths, patch_shape=patch_shape, **kwargs
    )

    return dataset


def get_siim_acr_loader(
    path: Union[os.PathLike, str],
    split: str,
    patch_shape: Tuple[int, int],
    batch_size: int,
    download: bool = False,
    **kwargs
):
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_siim_acr_dataset(
        path=path, split=split, patch_shape=patch_shape, download=download, **ds_kwargs
    )
    loader = torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
    return loader
