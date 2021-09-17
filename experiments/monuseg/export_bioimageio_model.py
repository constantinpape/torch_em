import os

import imageio
# from torch_em.data.datasets import get_bioimageio_dataset_id
from torch_em.util import (add_weight_formats, export_biomageio_model,
                           get_default_citations, export_parser_helper)


def _get_name(is_aff):
    name = "MoNuSeg-Nuclei"
    if is_aff:
        name += "-AffinityModel"
    else:
        name += "-BoundaryModel"
    return name


def _get_doc(is_aff_model):
    ndim = 2
    if is_aff_model:
        doc = f"""
## {ndim}D U-Net for Affinity Prediction

This model was trained on data from the MoNuSeg segmentation challenge.
It predicts affinity maps and foreground probabilities for nucleus segmentation in different
organs with H&E staining.
The affinities can be processed with the mutex watershed to obtain an instance segmentation.
        """
    else:
        doc = f"""
## {ndim}D U-Net for Boundary Prediction

This model was trained on data from the MoNuSeg segmentation challenge.
It predicts boundary maps and foreground probabilities for nucleus segmentation in different
organs with H&E staining.
The boundaries can be processed with multicut segmentation to obtain an instance segmentation.
        """
    return doc


# TODO write offsets and other mws params into the config if this is a affinity model
def export_to_bioimageio(checkpoint, output, input_, affs_to_bd, additional_formats):

    ckpt_name = os.path.split(checkpoint)[1]
    if input_ is None:
        input_data = None
    else:
        input_data = imageio.imread(input_)
        input_data = input_data[:512, :512]
        input_data = input_data.transpose((2, 0, 1))
        assert input_data.shape[0] == 3, f"{input_data.shape}"

    is_aff_model = 'affinity' in ckpt_name
    if is_aff_model and affs_to_bd:
        postprocessing = 'affinities_with_foreground_to_boundaries2d'
    else:
        postprocessing = None

    if is_aff_model and affs_to_bd:
        is_aff_model = False
    name = _get_name(is_aff_model)
    tags = ["u-net", "nucleus-segmentation", "segmentation", "volume-em", "platynereis", "nuclei"]
    tags += ["boundary-prediction"] if is_aff_model else ["affinity-prediction"]

    # eventually we should refactor the citation logic
    cite = get_default_citations(
        model="UNet2d", model_output="affinities" if is_aff_model else "boundaries"
    )
    cite["data"] = "https://ieeexplore.ieee.org/document/8880654"

    doc = _get_doc(is_aff_model)

    export_biomageio_model(
        checkpoint, output,
        input_data=input_data,
        name=name,
        authors=[{"name": "Constantin Pape; @constantinpape"}],
        tags=tags,
        license='CC-BY-4.0',
        documentation=doc,
        git_repo='https://github.com/constantinpape/torch-em.git',
        cite=cite,
        model_postprocessing=postprocessing,
        input_optional_parameters=False,
        # need custom deepimagej fields if we have torchscript export
        for_deepimagej="torchscript" in additional_formats,
        # TODO
        # links=[get_bioimageio_dataset_id("monuseg")]
    )
    add_weight_formats(output, additional_formats)


if __name__ == '__main__':
    parser = export_parser_helper()
    args = parser.parse_args()
    export_to_bioimageio(args.checkpoint, args.output, args.input,
                         bool(args.affs_to_bd), args.additional_formats)
