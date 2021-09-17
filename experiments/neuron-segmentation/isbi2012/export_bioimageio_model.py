import os

import numpy as np
from elf.io import open_file
from torch_em.data.datasets import get_bioimageio_dataset_id
from torch_em.util import (add_weight_formats, export_biomageio_model,
                           get_default_citations, export_parser_helper)


def _load_data(input_, ndim):
    with open_file(input_, 'r') as f:
        ds = f['volumes/raw'] if 'volumes/raw' in f else f['raw']
        shape = ds.shape
        if ndim == 2:
            s0, s1 = shape[0] - 1, shape[0]
            bb = np.s_[s0:s1, :, :]
        else:
            assert False, "3d not supported yet"
        raw = ds[bb]
    return raw


def _get_name(is_aff, ndim):
    name = "ISBI2012"
    name += "-2D" if ndim == 2 else "-3D"
    if is_aff:
        name += "-AffinityModel"
    else:
        name += "-BoundaryModel"
    return name


def _get_doc(is_aff_model, ndim):
    training_url = "https://github.com/constantinpape/torch-em/tree/main/experiments/neuron-segmentation/isbi2012"
    if is_aff_model:
        title = f"{ndim}D U-Net for Affinity Prediction"
        output_name = "affinity maps"
        seg = "with the mutex watershed "
    else:
        title = f"{ndim}D U-Net for Boundary Prediction"
        output_name = "boundary maps"
        seg = "multicut segmentation"

    doc = f"""
## {title}

This model was trained on the data of the ISBI2012 neuron segmentation challenge.
It predicts {output_name} that can be processed {seg} to obtain an instance segmentation.
For more details, check out [the training scripts]({training_url})."""

    return doc


def export_to_bioimageio(checkpoint, input_, output, affs_to_bd, additional_formats):

    ckpt_name = os.path.split(checkpoint)[1]

    ndim = 3 if '3d' in ckpt_name else 2
    input_data = _load_data(input_, ndim)

    is_aff_model = 'affinity' in ckpt_name
    if is_aff_model and affs_to_bd:
        postprocessing = f'affinities_to_boundaries{ndim}d'
    else:
        postprocessing = None
    if is_aff_model and affs_to_bd:
        is_aff_model = False

    if is_aff_model:
        offsets = [
            [-1, 0], [0, -1],
            [-3, 0], [0, -3],
            [-9, 0], [0, -9],
            [-27, 0], [0, -27]
        ]
        config = {"mws": {"offsets": offsets}}
    else:
        config = {}

    name = _get_name(is_aff_model, ndim)
    tags = ["u-net", "neuron-segmentation", "segmentation", "volume-em", "isbi2012-challenge"]
    tags += ["boundary-prediction"] if is_aff_model else ["affinity-prediction"]

    cite = get_default_citations(
        model="UNet2d" if ndim == 2 else "AnisotropicUNet",
        model_output="affinities" if is_aff_model else "boundaries"
    )
    cite["data"] = "https://doi.org/10.3389/fnana.2015.00142"
    doc = _get_doc(is_aff_model, ndim)

    export_biomageio_model(
        checkpoint, output, input_data,
        name=name,
        authors=[{"name": "Constantin Pape; @constantinpape"}],
        tags=tags,
        license='CC-BY-4.0',
        documentation=doc,
        git_repo='https://github.com/constantinpape/torch-em.git',
        cite=cite,
        model_postprocessing=postprocessing,
        input_optional_parameters=False,
        for_deepimagej="torchscript" in additional_formats,
        links=[get_bioimageio_dataset_id("isbi2012")],
        config=config
    )
    add_weight_formats(output, additional_formats)


if __name__ == '__main__':
    parser = export_parser_helper()
    args = parser.parse_args()
    export_to_bioimageio(args.checkpoint, args.input, args.output,
                         bool(args.affs_to_bd), args.additional_formats)
