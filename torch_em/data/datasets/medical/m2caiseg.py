import os
import shutil
from glob import glob
from tqdm import tqdm
from pathlib import Path
from natsort import natsorted
from typing import Union, Tuple

import numpy as np
import imageio.v3 as imageio

import torch_em

from .. import util


LABEL_MAPS = {
    (0, 0, 0): 0,  # out of frame
    (0, 85, 170): 1,  # instrument
    (85, 170, 0): 2,  # liver
    (85, 170, 255): 3,  # gall bladder
    (85, 255, 0): 4,  # fat
    (85, 255, 170): 5,  # upper wall
    (255, 0, 255): 6,  # intestine
    (170, 0, 255): 7,  # artery
    (170, 0, 85): 8,  # unknown
}


def get_m2caiseg_data(path, download):
    os.makedirs(path, exist_ok=True)

    data_dir = os.path.join(path, r"m2caiSeg dataset")
    if os.path.exists(data_dir):
        return data_dir

    util.download_source_kaggle(path=path, dataset_name="salmanmaq/m2caiseg", download=download)
    zip_path = os.path.join(path, "m2caiseg.zip")
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def _get_m2caiseg_paths(path, split, download):
    data_dir = get_m2caiseg_data(path=path, download=download)

    if split == "val":
        impaths = natsorted(glob(os.path.join(data_dir, "train", "images", "*.jpg")))
        gpaths = natsorted(glob(os.path.join(data_dir, "train", "groundtruth", "*.png")))

        imids = [os.path.split(_p)[-1] for _p in impaths]
        gids = [os.path.split(_p)[-1] for _p in gpaths]

        image_paths = [
            _p for _p in natsorted(
                glob(os.path.join(data_dir, "trainval", "images", "*.jpg"))
            ) if os.path.split(_p)[-1] not in imids
        ]
        gt_paths = [
            _p for _p in natsorted(
                glob(os.path.join(data_dir, "trainval", "groundtruth", "*.png"))
            ) if os.path.split(_p)[-1] not in gids
        ]

    else:
        image_paths = natsorted(glob(os.path.join(data_dir, split, "images", "*.jpg")))
        gt_paths = natsorted(glob(os.path.join(data_dir, split, "groundtruth", "*.png")))

    images_dir = os.path.join(data_dir, "preprocessed", "images")
    mask_dir = os.path.join(data_dir, "preprocessed_masks")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    fimage_paths, fgt_paths = [], []
    for image_path, gt_path in tqdm(zip(image_paths, gt_paths), total=len(image_paths)):
        image = imageio.imread(image_path)
        gt = imageio.imread(gt_path)

        image_id = Path(image_path).stem
        gt_id = Path(gt_path).stem

        if image.shape != gt.shape:
            print("This pair of image and labels mismatch.")
            continue

        dst_image_path = os.path.join(images_dir, f"{image_id}.tif")
        dst_gt_path = os.path.join(mask_dir, f"{gt_id}.tif")

        fimage_paths.append(image_path)
        fgt_paths.append(dst_gt_path)
        if os.path.exists(dst_gt_path) and os.path.exists(dst_image_path):
            continue

        instances = np.zeros(gt.shape[:2])
        for lmap in LABEL_MAPS:
            binary_map = (gt == lmap).all(axis=2)
            instances[binary_map > 0] = LABEL_MAPS[lmap]

        imageio.imwrite(dst_image_path, image, compression="zlib")
        imageio.imwrite(dst_gt_path, instances, compression="zlib")

    return fimage_paths, fgt_paths


def get_m2caiseg_dataset(
    path: Union[os.PathLike, str],
    split: str,
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    assert split in ["train", "val", "test"]

    image_paths, gt_paths = _get_m2caiseg_paths(path=path, split=split, download=download)

    print(len(image_paths), len(gt_paths))

    breakpoint()

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": True}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    dataset = torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=gt_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        **kwargs
    )

    return dataset


def get_m2caiseg_loader(
    path: Union[os.PathLike, str],
    split: str,
    patch_shape: Tuple[int, int],
    batch_size: int,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_m2caiseg_dataset(
        path=path, split=split, patch_shape=patch_shape, resize_inputs=resize_inputs, download=download, **ds_kwargs
    )
    loader = torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
    return loader
