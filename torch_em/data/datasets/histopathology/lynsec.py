"""
"""

import os
from glob import glob
from tqdm import tqdm
from pathlib import Path
from natsort import natsorted
from typing import Union, Tuple, List, Optional, Literal

import numpy as np
import imageio.v3 as imageio

import torch_em

from torch.utils.data import Dataset, DataLoader

from .. import util


URL = "https://zenodo.org/records/8065174/files/lynsec.zip"
CHECKSUM = "14b9b5a9c39cb41afc7f31de5a995cefff0947c215e14ab9c7a463f32fbbf4b6"


def _preprocess_dataset(data_dir):
    data_dirs = natsorted(glob(os.path.join(data_dir, "lynsec*")))
    for _dir in data_dirs:
        if os.path.basename(_dir) == "lynsec 1":
            target_dir = "ihc"
        else:
            target_dir = "h&e"

        image_dir = os.path.join(data_dir, target_dir, "images")
        label_dir = os.path.join(data_dir, target_dir, "labels")
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)

        paths = natsorted(glob(os.path.join(_dir, "*.npy")))
        for fpath in tqdm(paths, desc="Preprocessing inputs"):
            fname = Path(fpath).stem
            darray = np.load(fpath)

            raw = darray[..., :3]
            labels = darray[..., 3]

            if target_dir == "h&e" and fname in [f"{i}_l2" for i in range(35)]:  # set of images have mismatching labels
                continue

            imageio.imwrite(os.path.join(image_dir, f"{fname}.tif"), raw, compression="zlib")
            imageio.imwrite(os.path.join(label_dir, f"{fname}.tif"), labels, compression="zlib")


def get_lynsec_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """
    """
    data_dir = os.path.join(path, "data")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(data_dir, exist_ok=True)

    zip_path = os.path.join(path, "lynsec.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=data_dir)

    _preprocess_dataset(data_dir)

    return data_dir


def get_lynsec_paths(
    path: Union[os.PathLike, str], choice: Optional[Literal['ihc', 'h&e']] = None, download: bool = False
) -> Tuple[List[str], List[str]]:
    """
    """
    data_dir = get_lynsec_data(path, download)

    if choice is None:
        choice = "*"

    raw_paths = natsorted(glob(os.path.join(data_dir, choice, "images", "*.tif")))
    label_paths = natsorted(glob(os.path.join(data_dir, choice, "labels", "*.tif")))

    return raw_paths, label_paths


def get_lynsec_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    choice: Optional[Literal['ihc', 'h&e']] = None,
    download: bool = False,
    **kwargs
) -> Dataset:
    """
    """
    raw_paths, label_paths = get_lynsec_paths(path, choice, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        **kwargs
    )


def get_lynsec_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    choice: Optional[Literal['ihc', 'h&e']] = None,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_lynsec_dataset(path, patch_shape, choice, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
