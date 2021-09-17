import os
import sys
from glob import glob
from shutil import copyfile


import torch_em
from torch_em.data.datasets.covid_if import _download_covid_if
from resize_unet import ResizeUNet
from skimage.transform import rescale

DATA_FOLDER = "./data"


def rescale_labels(labels):
    assert labels.ndim == 2, labels.ndim
    return rescale(labels, scale=(0.5, 0.5))


def get_loader(patch_shape, batch_size):
    _download_covid_if(DATA_FOLDER, True)
    file_paths = glob(os.path.join(DATA_FOLDER, "*.h5"))
    file_paths.sort()
    raw_key = "raw/serum_IgG/s0"
    label_key = "labels/cells/s0"
    label_transform = torch_em.transform.label.labels_to_binary
    return torch_em.default_segmentation_loader(file_paths, raw_key,
                                                file_paths, label_key,
                                                batch_size=batch_size,
                                                patch_shape=patch_shape,
                                                label_transform=label_transform,
                                                label_transform2=rescale_labels)


def train_model():
    patch_shape = (512, 512)
    batch_size = 4
    loader = get_loader(patch_shape, batch_size)
    model = ResizeUNet(in_channels=1, out_channels=1, depth=3, initial_features=16)
    name = "diff-output-shape"
    trainer = torch_em.default_segmentation_trainer(name, model, loader, loader, logger=None)
    iterations = 5000
    trainer.fit(iterations)


def export_model():
    import imageio
    import h5py
    from torch_em.util import export_biomageio_model, get_default_citations
    from bioimageio.spec.shared import yaml

    with h5py.File("./data/gt_image_000.h5", "r") as f:
        input_data = f["raw/serum_IgG/s0"][:256, :256]
    imageio.imwrite("./cover.jpg", input_data)

    doc = "Example Model: Different Output Shape"
    cite = get_default_citations(model="UNet2d")

    export_biomageio_model(
        "./checkpoints/diff-output-shape",
        "./exported",
        input_data=input_data,
        authors=[{"name": "Constantin Pape; @constantinpape"}],
        tags=["segmentation"],
        license="CC-BY-4.0",
        documentation=doc,
        git_repo="https://github.com/constantinpape/torch-em.git",
        cite=cite,
        covers=["./cover.jpg"],
        input_optional_parameters=False
    )

    rdf_path = "./exported/rdf.yaml"
    with open(rdf_path, "r") as f:
        rdf = yaml.load(f)

    # update the shape descriptions
    rdf["inputs"][0]["shape"] = {"min": [1, 1, 32, 32], "step": [0, 0, 16, 16]}
    rdf["outputs"][0]["shape"] = {"reference_input": "input", "offset": [0, 0, 0, 0], "scale": [1, 1, 0.5, 0.5]}

    # update the network description
    rdf["source"] = "./resize_unet.py:ResizeUNet"
    rdf["kwargs"] = dict(in_channels=1, out_channels=1, depth=3, initial_features=16)
    copyfile("./resize_unet.py", "./exported/resize_unet.py")

    with open(rdf_path, "w") as f:
        yaml.dump(rdf, f)


if __name__ == "__main__":
    train = bool(int(sys.argv[1]))
    if train:
        train_model()
    else:
        export_model()
