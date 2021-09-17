import numpy as np

from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.morphology import binary_dilation
from skimage.draw import line
from skimage.filters import gaussian
from skimage.measure import label

from .augmentation import get_augmentations
from .raw import standardize
from ..data import SegmentationDataset, MinForegroundSampler


#
# defect augmentations
#
# TODO
# - alignment jitter


def get_artifact_source(artifact_path, patch_shape, min_mask_fraction,
                        normalizer=standardize,
                        raw_key="artifacts", mask_key="alpha_mask"):
    augmentation = get_augmentations(ndim=2)
    sampler = MinForegroundSampler(min_mask_fraction)
    return SegmentationDataset(
        artifact_path, raw_key,
        artifact_path, mask_key,
        patch_shape=patch_shape,
        raw_transform=standardize,
        transform=augmentation,
        sampler=sampler
    )


class EMDefectAugmentation:
    """Augment raw data with transformations similar to defects common in EM data.

    Arguments:
        p_drop_slice: probability for a missing slice
        p_low_contrast: probability for a low contrast slice
        p_deform_slice: probaboloty for a deformed slice
        p_paste_artifact: probability for inserting an artifact from data source
        contrast_scale: scale of low contrast transformation
        deformation_mode: deformation mode that should be used
        deformation_strength: deformation strength in pixel
        artifact_source: data source for additional artifacts
        mean_val: mean value for artifact normalization
        std_val: std value for artifact normalization
    """
    def __init__(
        self,
        p_drop_slice,
        p_low_contrast,
        p_deform_slice,
        p_paste_artifact=0.0,
        contrast_scale=0.1,
        deformation_mode='undirected',
        deformation_strength=10,
        artifact_source=None,
        mean_val=None,
        std_val=None
    ):
        if p_paste_artifact > 0.0:
            assert artifact_source is not None
        self.artifact_source = artifact_source

        # use cumulative probabilities
        self.p_drop_slice = p_drop_slice
        self.p_low_contrast = self.p_drop_slice + p_low_contrast
        self.p_deform_slice = self.p_low_contrast + p_deform_slice
        self.p_paste_artifact = self.p_deform_slice + p_paste_artifact
        assert self.p_paste_artifact < 1.0

        self.contrast_scale = contrast_scale
        self.mean_val = mean_val
        self.std_val = std_val

        # set the parameters for deformation augments
        if isinstance(deformation_mode, str):
            assert deformation_mode in ('all', 'undirected', 'compress')
            self.deformation_mode = deformation_mode
        elif isinstance(deformation_mode, (list, tuple)):
            assert len(deformation_mode) == 2
            assert 'undirected' in deformation_mode
            assert 'compress' in deformation_mode
            self.deformation_mode = 'all'
        self.deformation_strength = deformation_strength

    def drop_slice(self, raw):
        raw[:] = 0
        return raw

    def low_contrast(self, raw):
        mean = raw.mean()
        raw -= mean
        raw *= self.contrast_scale
        raw += mean
        return raw

    # this simulates a typical defect:
    # missing line of data with rest of data compressed towards the line
    def compress_slice(self, raw):
        shape = raw.shape
        # randomly choose fixed x or fixed y with p = 1/2
        fixed_x = np.random.rand() < .5
        if fixed_x:
            x0, y0 = 0, np.random.randint(1, shape[1] - 2)
            x1, y1 = shape[0] - 1, np.random.randint(1, shape[1] - 2)
        else:
            x0, y0 = np.random.randint(1, shape[0] - 2), 0
            x1, y1 = np.random.randint(1, shape[0] - 2), shape[1] - 1

        # generate the mask of the line that should be blacked out
        line_mask = np.zeros_like(raw, dtype='bool')
        rr, cc = line(x0, y0, x1, y1)
        line_mask[rr, cc] = 1

        # generate vectorfield pointing towards the line to compress the image
        # first we get the unit vector representing the line
        line_vector = np.array([x1 - x0, y1 - y0], dtype='float32')
        line_vector /= np.linalg.norm(line_vector)
        # next, we generate the normal to the line
        normal_vector = np.zeros_like(line_vector)
        normal_vector[0] = - line_vector[1]
        normal_vector[1] = line_vector[0]

        # make meshgrid
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        # generate the vector field
        flow_x, flow_y = np.zeros_like(raw), np.zeros_like(raw)

        # find the 2 components where coordinates are bigger / smaller than the line
        # to apply normal vector in the correct direction
        components = label(np.logical_not(line_mask))
        assert len(np.unique(components)) == 3, "%i" % len(np.unique(components))
        neg_val = components[0, 0] if fixed_x else components[-1, -1]
        pos_val = components[-1, -1] if fixed_x else components[0, 0]

        flow_x[components == pos_val] = self.deformation_strength * normal_vector[1]
        flow_y[components == pos_val] = self.deformation_strength * normal_vector[0]
        flow_x[components == neg_val] = - self.deformation_strength * normal_vector[1]
        flow_y[components == neg_val] = - self.deformation_strength * normal_vector[0]

        # add small random noise
        flow_x += np.random.uniform(-1, 1, shape) * (self.deformation_strength / 8.)
        flow_y += np.random.uniform(-1, 1, shape) * (self.deformation_strength / 8.)

        # apply the flow fields
        flow_x, flow_y = (x + flow_x).reshape(-1, 1), (y + flow_y).reshape(-1, 1)
        cval = 0.0 if self.mean_val is None else self.mean_val
        raw = map_coordinates(
            raw, (flow_y, flow_x), mode='constant', order=3, cval=cval
        ).reshape(shape)

        # dilate the line mask and zero out the raw below it
        line_mask = binary_dilation(line_mask, iterations=10)
        raw[line_mask] = 0.
        return raw

    def undirected_deformation(self, raw):
        shape = raw.shape

        # make meshgrid
        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))

        # generate random vector field and smooth it
        flow_x = np.random.uniform(-1, 1, shape) * self.deformation_strength
        flow_y = np.random.uniform(-1, 1, shape) * self.deformation_strength
        flow_x = gaussian(flow_x, sigma=3.)  # sigma is hard-coded for now
        flow_y = gaussian(flow_y, sigma=3.)  # sigma is hard-coded for now

        # apply the flow fields
        flow_x, flow_y = (x + flow_x).reshape(-1, 1), (y + flow_y).reshape(-1, 1)
        raw = map_coordinates(raw, (flow_y, flow_x), mode='constant').reshape(shape)
        return raw

    def deform_slice(self, raw):
        if self.deformation_mode in ('undirected', 'compress'):
            mode = self.deformation_mode
        else:
            mode = 'undireccted' if np.random.rand() < .5 else 'compress'
        if mode == 'compress':
            raw = self.compress_slice(raw)
        else:
            raw = self.undirected_deformation(raw)
        return raw

    def paste_artifact(self, raw):
        # draw a random artifact location
        artifact_index = np.random.randint(len(self.artifact_source))
        artifact, alpha_mask = self.artifact_source[artifact_index]
        artifact = artifact.numpy().squeeze()
        alpha_mask = alpha_mask.numpy().squeeze()
        assert artifact.shape == raw.shape, f"{artifact.shape}, {raw.shape}"
        assert alpha_mask.shape == raw.shape
        assert alpha_mask.min() >= 0., f"{alpha_mask.min()}"
        assert alpha_mask.max() <= 1., f"{alpha_mask.max()}"

        # blend the raw raw data and the artifact according to the alpha mask
        raw = raw * (1. - alpha_mask) + artifact * alpha_mask
        return raw

    def __call__(self, raw):
        for z in range(raw.shape[0]):
            r = np.random.rand()
            if r < self.p_drop_slice:
                # print("Drop slice", z)
                raw[z] = self.drop_slice(raw[z])
            elif r < self.p_low_contrast:
                # print("Low contrast", z)
                raw[z] = self.low_contrast(raw[z])
            elif r < self.p_deform_slice:
                # print("Deform slice", z)
                raw[z] = self.deform_slice(raw[z])
            elif r < self.p_paste_artifact:
                # print("Paste artifact", z)
                raw[z] = self.paste_artifact(raw[z])
        return raw
