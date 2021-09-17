import os
import torch_em
from .util import update_kwargs


# TODO separate via organ
def _download_monuseg(path, download):
    os.makedirs(path, exist_ok=True)

    im_path = os.path.join(path, "images")
    label_path = os.path.join(path, "labels")

    if os.path.exists(im_path) and os.path.exists(label_path):
        return
    raise NotImplementedError


# TODO
def _process_monuseg():
    pass


# TODO implement selecting organ
def get_monuseg_loader(path, patch_shape, download=False, roi=None,
                       offsets=None, boundaries=False, binary=False,
                       **kwargs):
    _download_monuseg(path, download)

    image_path = os.path.join(path, "images")
    label_path = os.path.join(path, "labels")

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
        kwargs = update_kwargs(kwargs, "label_transform", label_transform, msg=msg)
    elif binary:
        label_transform = torch_em.transform.label.labels_to_binary
        msg = "Binary is set to true, but 'label_transform' is in the kwargs. It will be over-ridden."
        kwargs = update_kwargs(kwargs, "label_transform", label_transform, msg=msg)

    kwargs = update_kwargs(kwargs, "patch_shape", patch_shape)
    kwargs = update_kwargs(kwargs, "ndim", 2)

    return torch_em.default_segmentation_loader(
        image_path, "*.tif",
        label_path, "*.tif",
        is_seg_dataset=False,
        rois=roi, **kwargs
    )
