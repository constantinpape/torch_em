import os
from shutil import copyfileobj

import imageio
import numpy as np
import requests
from tqdm import tqdm

import torch_em
import torch.utils.data
from .util import download_source, unzip, update_kwargs

URLS = {
    "images": "http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/images.zip",
    "train": ("http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/"
              "LIVECell/livecell_coco_train.json"),
    "val": ("http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/"
            "LIVECell/livecell_coco_val.json"),
    "test": ("http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/",
             "LIVECell/livecell_coco_test.json")
}
# TODO
CHECKSUM = None


def _download_livecell_images(path, download):
    os.makedirs(path, exist_ok=True)
    image_path = os.path.join(path, "images")

    if os.path.exists(image_path):
        return

    url = URLS["images"]
    checksum = CHECKSUM
    zip_path = os.path.join(path, "livecell.zip")
    download_source(zip_path, url, download, checksum)
    unzip(zip_path, path, True)


# TODO use download flag
def _download_annotation_file(path, split, download):
    annotation_file = os.path.join(path, f"{split}.json")
    if not os.path.exists(annotation_file):
        url = URLS[split]
        print("Downloading livecell annotation file from", url)
        with requests.get(url, stream=True) as r:
            with open(annotation_file, 'wb') as f:
                copyfileobj(r.raw, f)
    return annotation_file


def _create_segmentations_from_annotations(annotation_file, image_folder, seg_folder):
    # TODO try except to explain how to install this
    from pycocotools.coco import COCO

    coco = COCO(annotation_file)
    category_ids = coco.getCatIds(catNms=["cell"])
    image_ids = coco.getImgIds(catIds=category_ids)

    image_paths, seg_paths = [], []
    for image_id in tqdm(image_ids, desc="creating livecell segmentations from coco-style annotations"):
        # get the path for the image data and make sure the corresponding image exists
        image_metadata = coco.loadImgs(image_id)[0]
        file_name = image_metadata["file_name"]
        sub_folder = file_name.split("_")[0]
        image_path = os.path.join(image_folder, sub_folder, file_name)
        assert os.path.exists(image_path), image_path
        image_paths.append(image_path)

        # get the output path
        out_folder = os.path.join(seg_folder, sub_folder)
        os.makedirs(out_folder, exist_ok=True)
        seg_path = os.path.join(out_folder, file_name)
        seg_paths.append(seg_path)
        if os.path.exists(seg_path):
            continue

        # create and save the segmentation
        annotation_ids = coco.getAnnIds(imgIds=image_metadata["id"], catIds=category_ids)
        annotations = coco.loadAnns(annotation_ids)
        assert len(annotations) <= np.iinfo("uint16").max
        shape = (image_metadata["height"], image_metadata["width"])
        seg = np.zeros(shape, dtype="uint16")
        for seg_id, annotation in enumerate(annotations, 1):
            mask = coco.annToMask(annotation).astype("bool")
            assert mask.shape == seg.shape
            seg[mask] = seg_id

        imageio.imwrite(seg_path, seg)

    return image_paths, seg_paths


def _download_livecell_annotations(path, split, download):
    annotation_file = _download_annotation_file(path, split, download)
    if split == "test":
        split_name = "livecell_test_images"
        image_folder = os.path.join(path, "images", split_name)
        seg_folder = os.path.join(path, "annotations", split_name)
    else:
        split_name = "livecell_train_val_images"
        image_folder = os.path.join(path, "images", split_name)
        seg_folder = os.path.join(path, "annotations", split_name)
    assert os.path.exists(image_folder), image_folder

    return _create_segmentations_from_annotations(annotation_file, image_folder, seg_folder)


def _livecell_segmentation_loader(
    image_paths, label_paths,
    batch_size, patch_shape,
    label_transform=None,
    label_transform2=None,
    raw_transform=None,
    transform=None,
    dtype=torch.float32,
    n_samples=None,
    **loader_kwargs
):

    # we always use a raw transform in the convenience function
    if raw_transform is None:
        raw_transform = torch_em.transform.get_raw_transform()

    # we always use augmentations in the convenience function
    if transform is None:
        transform = torch_em.transform.get_augmentations(ndim=2)

    ds = torch_em.data.ImageCollectionDataset(image_paths, label_paths,
                                              patch_shape=patch_shape,
                                              raw_transform=raw_transform,
                                              label_transform=label_transform,
                                              label_transform2=label_transform2,
                                              transform=transform,
                                              n_samples=n_samples)

    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, **loader_kwargs)
    # monkey patch shuffle attribute to the loader
    loader.shuffle = loader_kwargs.get('shuffle', False)
    return loader


# TODO at support for the different benchmark tasks
def get_livecell_loader(path, patch_shape, split, download=False,
                        offsets=None, boundaries=False, binary=False,
                        **kwargs):
    assert split in ("train", "val", "test")
    _download_livecell_images(path, download)
    image_paths, seg_paths = _download_livecell_annotations(path, split, download)

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

    kwargs.update({"patch_shape": patch_shape})
    return _livecell_segmentation_loader(image_paths, seg_paths, **kwargs)
