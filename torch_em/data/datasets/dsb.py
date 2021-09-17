import os
from shutil import move

import torch_em
from .util import download_source, unzip, update_kwargs

DSB_URLS = {
    "full": "",  # TODO
    "reduced": "https://github.com/stardist/stardist/releases/download/0.1.0/dsb2018.zip"
}
CHECKSUMS = {
    "full": None,
    "reduced": "e44921950edce378063aa4457e625581ba35b4c2dbd9a07c19d48900129f386f"
}


def _download_dsb(path, source, download):
    os.makedirs(path, exist_ok=True)
    url = DSB_URLS[source]
    checksum = CHECKSUMS[source]

    train_out_path = os.path.join(path, "train")
    test_out_path = os.path.join(path, "test")

    if os.path.exists(train_out_path) and os.path.exists(test_out_path):
        return

    zip_path = os.path.join(path, "dsb.zip")
    download_source(zip_path, url, download, checksum)
    unzip(zip_path, path, True)

    move(os.path.join(path, "dsb2018", "train"), train_out_path)
    move(os.path.join(path, "dsb2018", "test"), test_out_path)


def get_dsb_loader(path, patch_shape, split, download=False,
                   offsets=None, boundaries=False, binary=False,
                   source="reduced", **kwargs):
    assert split in ("test", "train"), split
    _download_dsb(path, source, download)

    image_path = os.path.join(path, split, "images")
    label_path = os.path.join(path, split, "masks")

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
        image_path, "*.tif",
        label_path, "*.tif",
        **kwargs
    )
