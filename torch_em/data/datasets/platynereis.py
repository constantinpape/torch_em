import os
from glob import glob

import numpy as np
import torch_em
from .util import download_source, update_kwargs, unzip

URLS = {
    "cells": "https://zenodo.org/record/3675220/files/membrane.zip?download=1",
    "nuclei": "https://zenodo.org/record/3675220/files/nuclei.zip?download=1",
    "cilia": "https://zenodo.org/record/3675220/files/cuticle.zip?download=1",
    "cuticle": "https://zenodo.org/record/3675220/files/cuticle.zip?download=1"
}

CHECKSUMS = {
    "cells": "30eb50c39e7e9883e1cd96e0df689fac37a56abb11e8ed088907c94a5980d6a3",
    "nuclei": "a05033c5fbc6a3069479ac6595b0a430070f83f5281f5b5c8913125743cf5510",
    "cilia": None,
    "cuticle": None
}


#
# TODO data-loader for more classes:
# - cilia
# - cuticle
# - mitos
#


def _require_platy_data(path, name, download):
    os.makedirs(path, exist_ok=True)
    url = URLS[name]
    checksum = CHECKSUMS[name]

    zip_path = os.path.join(path, 'data.zip')
    download_source(zip_path, url, download=download, checksum=checksum)
    unzip(zip_path, path, remove=False)


def _check_data(path, prefix, extension, n_files):
    if not os.path.exists(path):
        return False
    files = glob(os.path.join(path, f"{prefix}*{extension}"))
    return len(files) == n_files


def get_platynereis_cell_loader(path, patch_shape,
                                sample_ids=None, rois={},
                                offsets=None, boundaries=False,
                                download=False, **kwargs):
    cell_root = os.path.join(path, "membrane")

    prefix = "train_data_membrane_"
    ext = ".n5"
    n_files = 9
    data_is_complete = _check_data(cell_root, prefix, ext, n_files)
    if not data_is_complete:
        _require_platy_data(path, "cells", download)

    if sample_ids is None:
        sample_ids = list(range(1, n_files + 1))
    assert min(sample_ids) >= 1 and max(sample_ids) <= n_files
    sample_ids.sort()

    data_paths = []
    data_rois = []
    for sample in sample_ids:
        data_paths.append(
            os.path.join(cell_root, f"{prefix}{sample:02}{ext}")
        )
        data_rois.append(rois.get(sample, np.s_[:, :, :]))

    kwargs = update_kwargs(kwargs, 'patch_shape', patch_shape)
    kwargs = update_kwargs(kwargs, 'rois', data_rois)
    # set ndim to 3 if not otherwise specified
    if "ndim" not in kwargs:
        kwargs["ndim"] = 3

    assert not ((offsets is not None) and boundaries)
    if offsets is not None:
        # we add a binary target channel for foreground background segmentation
        label_transform = torch_em.transform.label.AffinityTransform(offsets=offsets,
                                                                     ignore_label=0,
                                                                     add_binary_target=False,
                                                                     add_mask=True)
        msg = "Offsets are passed, but 'label_transform2' is in the kwargs. It will be over-ridden."
        kwargs = update_kwargs(kwargs, 'label_transform2', label_transform, msg=msg)
    elif boundaries:
        label_transform = torch_em.transform.label.BoundaryTransform()
        msg = "Boundaries is set to true, but 'label_transform' is in the kwargs. It will be over-ridden."
        kwargs = update_kwargs(kwargs, 'label_transform', label_transform, msg=msg)

    raw_key = "volumes/raw/s1"
    label_key = "volumes/labels/segmentation/s1"
    return torch_em.default_segmentation_loader(
        data_paths, raw_key,
        data_paths, label_key,
        **kwargs
    )


def get_platynereis_nuclei_loader(path, patch_shape,
                                  sample_ids=None, rois={},
                                  offsets=None, boundaries=False, binary=False,
                                  download=False, **kwargs):
    nuc_root = os.path.join(path, "nuclei")
    prefix = "train_data_nuclei_"
    ext = ".h5"
    n_files = 12
    data_is_complete = _check_data(nuc_root, "train_data_nuclei", ".h5", 12)
    if not data_is_complete:
        _require_platy_data(path, "nuclei", download)

    if sample_ids is None:
        sample_ids = list(range(1, n_files + 1))
    assert min(sample_ids) >= 1 and max(sample_ids) <= n_files
    sample_ids.sort()

    data_paths = []
    data_rois = []
    for sample in sample_ids:
        data_paths.append(
            os.path.join(nuc_root, f"{prefix}{sample:02}{ext}")
        )
        data_rois.append(rois.get(sample, np.s_[:, :, :]))

    kwargs = update_kwargs(kwargs, 'patch_shape', patch_shape)
    kwargs = update_kwargs(kwargs, 'rois', data_rois)
    # set ndim to 3 if not otherwise specified
    if "ndim" not in kwargs:
        kwargs["ndim"] = 3

    assert sum((offsets is None, boundaries, binary)) <= 1
    if offsets is not None:
        # we add a binary target channel for foreground background segmentation
        label_transform = torch_em.transform.label.AffinityTransform(offsets=offsets,
                                                                     ignore_label=-1,
                                                                     add_binary_target=True,
                                                                     add_mask=True)
        msg = "Offsets are passed, but 'label_transform2' is in the kwargs. It will be over-ridden."
        kwargs = update_kwargs(kwargs, 'label_transform2', label_transform, msg=msg)
    elif boundaries:
        label_transform = torch_em.transform.label.BoundaryTransform(add_binary_target=True)
        msg = "Boundaries is set to true, but 'label_transform' is in the kwargs. It will be over-ridden."
        kwargs = update_kwargs(kwargs, 'label_transform', label_transform, msg=msg)
    elif binary:
        label_transform = torch_em.transform.label.labels_to_binary
        msg = "Binary is set to true, but 'label_transform' is in the kwargs. It will be over-ridden."
        kwargs = update_kwargs(kwargs, 'label_transform', label_transform, msg=msg)

    raw_key = "volumes/raw"
    label_key = "volumes/labels/nucleus_instance_labels"
    return torch_em.default_segmentation_loader(
        data_paths, raw_key,
        data_paths, label_key,
        **kwargs
    )
