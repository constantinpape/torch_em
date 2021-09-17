import os
from glob import glob

import torch_em
from .util import download_source, unzip, update_kwargs

COVID_IF_URL = "https://zenodo.org/record/5092850/files/covid-if-groundtruth.zip?download=1"
CHECKSUM = "d9cd6c85a19b802c771fb4ff928894b19a8fab0e0af269c49235fdac3f7a60e1"


def _download_covid_if(path, download):
    url = COVID_IF_URL
    checksum = CHECKSUM

    if os.path.exists(path):
        return

    os.makedirs(path, exist_ok=True)
    zip_path = os.path.join(path, "covid-if.zip")
    download_source(zip_path, url, download, checksum)
    unzip(zip_path, path, True)


def get_covid_if_loader(path, patch_shape, sample_range=None,
                        target="cells", download=False,
                        offsets=None, boundaries=False, binary=False,
                        **kwargs):

    available_targets = ("cells", "nuclei")
    # TODO support all of these
    # available_targets = ("cells", "nuclei", "infected_cells")
    assert target in available_targets, f"{target} not found in {available_targets}"

    if target == "cells":
        raw_key = "raw/serum_IgG/s0"
        label_key = "labels/cells/s0"
    elif target == "nuclei":
        raw_key = "raw/nuclei/s0"
        label_key = "labels/nuclei/s0"
    # elif target == "infected_cells":

    _download_covid_if(path, download)

    file_paths = glob(os.path.join(path, "*.h5"))
    file_paths.sort()
    if sample_range is not None:
        start, stop = sample_range
        if start is None:
            start = 0
        if stop is None:
            stop = len(file_paths)
        file_paths = [os.path.join(path, f'gt_image_{idx:03}.h5') for idx in range(start, stop)]
        assert all(os.path.exists(fp) for fp in file_paths), f"Invalid sample range {sample_range}"

    assert sum((offsets is not None, boundaries, binary)) <= 1
    if offsets is not None:
        # we add a binary target channel for foreground background segmentation
        label_transform = torch_em.transform.label.AffinityTransform(offsets=offsets,
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

    kwargs = update_kwargs(kwargs, 'patch_shape', patch_shape)
    kwargs = update_kwargs(kwargs, 'ndim', 2)

    return torch_em.default_segmentation_loader(
        file_paths, raw_key,
        file_paths, label_key,
        **kwargs
    )
