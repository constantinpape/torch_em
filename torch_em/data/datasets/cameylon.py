import os
import warnings
import numpy as np
from glob import glob

import openslide


def _download_cameylon(path):
    is_cam16 = os.path.exists(os.path.join(path, "CAMELYON16"))
    is_cam17 = os.path.exists(os.path.join(path, "CAMELYON17"))
    if is_cam16 and is_cam17 is True:
        return

    try:
        import awscli
    except ModuleNotFoundError:
        os.system("pip install awscli")

    warnings.warn("The CAMELYON dataset could take a couple of hours to download the dataset.")

    os.system(f"aws s3 cp --no-sign-request s3://camelyon-dataset/ {path} --recursive")


def get_cameylon_dataset(path):
    """Take a look at two things for histopathology WSI reading:
        - tiatoolbox - https://tia-toolbox.readthedocs.io/
        - openslide - (example: https://github.com/computationalpathologygroup/Camelyon16/blob/master/Python/Evaluation_FROC.py)
    """
    all_paths = sorted(glob(os.path.join(path, "CAMELYON16", "images", "*")))
    print(all_paths[-1])

    level = 5  # Image level at which the evaluation is done

    slide = openslide.open_slide(all_paths[-1])
    dims = slide.level_dimensions[level]
    pixelarray = np.array(slide.read_region((0, 0), level, dims))


def get_cameylon_loader(path):
    # TODO: get a dataset for creating the dataloader
    _download_cameylon(path)
    get_cameylon_dataset(path)
