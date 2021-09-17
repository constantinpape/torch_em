import sys
import torch_em
from torch_em.data.datasets import get_covid_if_loader
from torch_em.model import UNet2d

DATA_FOLDER = "./data"


def train_model():
    patch_shape = (512, 512)
    batch_size = 4
    loader = get_covid_if_loader(DATA_FOLDER, patch_shape,
                                 batch_size=batch_size, download=True,
                                 binary=True)
    model = UNet2d(in_channels=1, out_channels=1, depth=3, initial_features=16)
    name = "diff-output-shape"
    trainer = torch_em.default_segmentation_trainer(name, model, loader, loader, logger=None)
    iterations = 5000
    trainer.fit(iterations)


def export_model():
    import h5py
    from torch_em.util import export_biomageio_model, get_default_citations
    from bioimageio.spec.shared import yaml

    with h5py.File("./data/gt_image_000.h5", "r") as f:
        input_data = f["raw/serum_IgG/s0"][:256, :256]

    doc = "Example Model: Fixed Shape"
    cite = get_default_citations(model="UNet2d")

    export_biomageio_model(
        "./checkpoints/fixed-shape",
        "./exported",
        input_data=input_data,
        authors=[{"name": "Constantin Pape; @constantinpape"}],
        tags=["segmentation"],
        license="CC-BY-4.0",
        documentation=doc,
        git_repo="https://github.com/constantinpape/torch-em.git",
        cite=cite,
        input_optional_parameters=False
    )

    shape = (1, 1) + input_data.shape
    assert len(shape) == 4

    # replace the shape
    rdf_path = "./exported/rdf.yaml"
    with open(rdf_path, "r") as f:
        rdf = yaml.load(f)
    rdf["inputs"][0]["shape"] = shape
    rdf["outputs"][0]["shape"] = shape
    with open(rdf_path, "w") as f:
        yaml.dump(rdf, f)


if __name__ == "__main__":
    train = bool(int(sys.argv[1]))
    if train:
        train_model()
    else:
        export_model()
