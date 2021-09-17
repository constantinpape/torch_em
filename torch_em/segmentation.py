import os
from glob import glob

import torch
import torch.utils.data
from elf.io import open_file

from .data import ConcatDataset, ImageCollectionDataset, SegmentationDataset
from .loss import DiceLoss
from .trainer import DefaultTrainer
from .trainer.tensorboard_logger import TensorboardLogger
from .transform import get_augmentations, get_raw_transform


# TODO add a heuristic to estimate this from the number of epochs
DEFAULT_SCHEDULER_KWARGS = {
    'mode': 'min',
    'factor': .5,
    'patience': 5
}


#
# convenience functions for segmentation loaders
#

# TODO implement balanced and make it the default
# def samples_to_datasets(n_samples, raw_paths, raw_key, split='balanced'):
def samples_to_datasets(n_samples, raw_paths, raw_key, split='uniform'):
    assert split in ('balanced', 'uniform')
    n_datasets = len(raw_paths)
    if split == 'uniform':
        # even distribution of samples to datasets
        samples_per_ds = n_samples // n_datasets
        divider = n_samples % n_datasets
        return [samples_per_ds + 1 if ii < divider else samples_per_ds
                for ii in range(n_datasets)]
    else:
        # distribution of samples to dataset based on the dataset lens
        raise NotImplementedError


def check_paths(raw_paths, label_paths):
    if type(raw_paths) != type(label_paths):
        raise ValueError(f"Expect raw and label paths of same type, got {type(raw_paths)}, {type(label_paths)}")

    def _check_path(path):
        if not os.path.exists(path):
            raise ValueError(f"Could not find path {path}")

    if isinstance(raw_paths, str):
        _check_path(raw_paths)
        _check_path(label_paths)
    else:
        if len(raw_paths) != len(label_paths):
            raise ValueError(f"Expect same number of raw and label paths, got {len(raw_paths)}, {len(label_paths)}")
        for rp, lp in zip(raw_paths, label_paths):
            _check_path(rp)
            _check_path(lp)


def is_segmentation_dataset(raw_paths, raw_key,
                            label_paths, label_key):
    """ Check if we can load the data as SegmentationDataset
    """

    def _can_open(path, key):
        try:
            open_file(path, mode='r')[key]
            return True
        except Exception:
            return False

    if isinstance(raw_paths, str):
        can_open_raw = _can_open(raw_paths, raw_key)
        can_open_label = _can_open(label_paths, label_key)
    else:
        can_open_raw = [_can_open(rp, raw_key) for rp in raw_paths]
        if not can_open_raw.count(can_open_raw[0]) == len(can_open_raw):
            raise ValueError("Inconsistent raw data")
        can_open_raw = can_open_raw[0]

        can_open_label = [_can_open(lp, label_key) for lp in label_paths]
        if not can_open_label.count(can_open_label[0]) == len(can_open_label):
            raise ValueError("Inconsistent label data")
        can_open_label = can_open_label[0]

    if can_open_raw != can_open_label:
        raise ValueError("Inconsistent raw and label data")

    return can_open_raw


def _load_segmentation_dataset(raw_paths, raw_key, label_paths, label_key,
                               **kwargs):
    rois = kwargs.pop('rois', None)
    if isinstance(raw_paths, str):
        if rois is not None:
            assert len(rois) == 3 and all(isinstance(roi, slice) for roi in rois)
        ds = SegmentationDataset(raw_paths, raw_key,
                                 label_paths, label_key,
                                 roi=rois, **kwargs)
    else:
        assert len(raw_paths) > 0
        if rois is not None:
            assert len(rois) == len(label_paths)
            assert all(isinstance(roi, tuple) for roi in rois)
        n_samples = kwargs.pop('n_samples', None)

        samples_per_ds = [None] * len(raw_paths) if n_samples is None else samples_to_datasets(n_samples,
                                                                                               raw_paths,
                                                                                               raw_key)
        ds = []
        for i, (raw_path, label_path) in enumerate(zip(raw_paths, label_paths)):
            roi = None if rois is None else rois[i]
            dset = SegmentationDataset(raw_path, raw_key,
                                       label_path, label_key,
                                       roi=roi, n_samples=samples_per_ds[i],
                                       **kwargs)
            ds.append(dset)
        ds = ConcatDataset(*ds)
    return ds


def _load_image_collection_dataset(raw_paths, raw_key, label_paths, label_key, roi,
                                   **kwargs):

    def _get_paths(rpath, rkey, lpath, lkey, this_roi):
        rpath = glob(os.path.join(rpath, rkey))
        rpath.sort()
        if len(rpath) == 0:
            raise ValueError(f"Could not find any images for pattern {os.path.join(rpath, rkey)}")
        lpath = glob(os.path.join(lpath, lkey))
        lpath.sort()
        if len(rpath) != len(lpath):
            raise ValueError(f"Expect same number of raw and label images, got {len(rpath)}, {len(lpath)}")

        if this_roi is not None:
            rpath, lpath = rpath[roi], lpath[roi]

        return rpath, lpath

    patch_shape = kwargs.pop('patch_shape')
    if len(patch_shape) == 3:
        if patch_shape[0] != 1:
            raise ValueError(f"Image collection dataset expects 2d patch shape, got {patch_shape}")
        patch_shape = patch_shape[1:]
    assert len(patch_shape) == 2

    if isinstance(raw_paths, str):
        raw_paths, label_paths = _get_paths(raw_paths, raw_key, label_paths, label_key, roi)
        ds = ImageCollectionDataset(raw_paths, label_paths, patch_shape=patch_shape, **kwargs)
    else:
        ds = []
        n_samples = kwargs.pop('n_samples', None)
        samples_per_ds = [None] * len(raw_paths) if n_samples is None else samples_to_datasets(n_samples,
                                                                                               raw_paths,
                                                                                               raw_key)
        if roi is None:
            roi = len(raw_paths) * [None]
        assert len(roi) == len(raw_paths)
        for i, (raw_path, label_path, this_roi) in enumerate(zip(raw_paths, label_paths, roi)):
            rpath, lpath = _get_paths(raw_path, raw_key, label_path, label_key, this_roi)
            dset = ImageCollectionDataset(rpath, lpath, patch_shape=patch_shape,
                                          n_samples=samples_per_ds[i], **kwargs)
            ds.append(dset)
        ds = ConcatDataset(*ds)
    return ds


def _get_default_transform(path, key, is_seg_dataset):
    if is_seg_dataset:
        with open_file(path, mode='r') as f:
            shape = f[key].shape
            if len(shape) == 2:
                ndim = 2
            else:
                # heuristics to figure out whether to use default 3d
                # or default anisotropic augmentations
                ndim = 'anisotropic' if shape[0] < shape[1] // 2 else 3
    else:
        ndim = 2
    return get_augmentations(ndim)


def default_segmentation_loader(
    raw_paths,
    raw_key,
    label_paths,
    label_key,
    batch_size,
    patch_shape,
    label_transform=None,
    label_transform2=None,
    raw_transform=None,
    transform=None,
    dtype=torch.float32,
    label_dtype=torch.float32,
    rois=None,
    n_samples=None,
    sampler=None,
    ndim=None,
    is_seg_dataset=None,
    **loader_kwargs
):
    check_paths(raw_paths, label_paths)
    if is_seg_dataset is None:
        is_seg_dataset = is_segmentation_dataset(raw_paths, raw_key,
                                                 label_paths, label_key)

    # we always use a raw transform in the convenience function
    if raw_transform is None:
        raw_transform = get_raw_transform()

    # we always use augmentations in the convenience function
    if transform is None:
        transform = _get_default_transform(raw_paths if isinstance(raw_paths, str) else raw_paths[0],
                                           raw_key, is_seg_dataset)

    if is_seg_dataset:
        ds = _load_segmentation_dataset(raw_paths, raw_key, label_paths, label_key,
                                        patch_shape=patch_shape,
                                        raw_transform=raw_transform, label_transform=label_transform,
                                        label_transform2=label_transform2, transform=transform,
                                        rois=rois, n_samples=n_samples, sampler=sampler,
                                        ndim=ndim, dtype=dtype, label_dtype=label_dtype)
    else:
        ds = _load_image_collection_dataset(raw_paths, raw_key, label_paths, label_key, roi=rois,
                                            patch_shape=patch_shape, label_transform=label_transform,
                                            raw_transform=raw_transform,
                                            label_transform2=label_transform2, transform=transform,
                                            n_samples=n_samples, sampler=sampler)

    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, **loader_kwargs)
    # monkey patch shuffle attribute to the loader
    loader.shuffle = loader_kwargs.get('shuffle', False)
    return loader


#
# convenience functions for segmentation trainers
#


def default_segmentation_trainer(
    name,
    model,
    train_loader,
    val_loader,
    loss=None,
    metric=None,
    learning_rate=1e-3,
    device=None,
    log_image_interval=100,
    mixed_precision=True,
    early_stopping=None,
    logger=TensorboardLogger,
    scheduler_kwargs=DEFAULT_SCHEDULER_KWARGS,
    optimizer_kwargs={},
    trainer_class=DefaultTrainer
):
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate,
                                 **optimizer_kwargs)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        **scheduler_kwargs
    )

    loss = DiceLoss() if loss is None else loss
    metric = DiceLoss() if metric is None else metric

    if device is None:
        device = torch.device('cuda')
    else:
        device = torch.device(device)

    trainer = trainer_class(
        name=name,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss=loss,
        metric=metric,
        optimizer=optimizer,
        device=device,
        lr_scheduler=scheduler,
        mixed_precision=mixed_precision,
        early_stopping=early_stopping,
        log_image_interval=log_image_interval,
        logger=logger
    )
    return trainer
