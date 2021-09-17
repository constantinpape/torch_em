import numpy as np
import torch
from ..util import (ensure_spatial_array, ensure_tensor_with_channels,
                    load_image, supports_memmap)


# TODO pad images that are too small for the patch shape
class ImageCollectionDataset(torch.utils.data.Dataset):

    def _check_inputs(self, raw_images, label_images):
        if len(raw_images) != len(label_images):
            raise ValueError(f"Expect same number of  and label images, got {len(raw_images)} and {len(label_images)}")

        is_multichan = None
        for raw_im, label_im in zip(raw_images, label_images):

            # we only check for compatible shapes if both images support memmap, because
            # we don't want to load everything into ram
            if supports_memmap(raw_im) and supports_memmap(label_im):
                shape = load_image(raw_im).shape
                assert len(shape) in (2, 3)

                multichan = len(shape) == 3
                if is_multichan is None:
                    is_multichan = multichan
                else:
                    assert is_multichan == multichan

                # we assume axis last
                if is_multichan:
                    shape = shape[:-1]

                label_shape = load_image(label_im).shape
                if shape != label_shape:
                    msg = f"Expect raw and labels of same shape, got {shape}, {label_shape} for {raw_im}, {label_im}"
                    raise ValueError(msg)

    def __init__(
        self,
        raw_image_paths,
        label_image_paths,
        patch_shape,
        raw_transform=None,
        label_transform=None,
        label_transform2=None,
        transform=None,
        dtype=torch.float32,
        label_dtype=torch.float32,
        n_samples=None,
        sampler=None
    ):
        self._check_inputs(raw_image_paths, label_image_paths)
        self.raw_images = raw_image_paths
        self.label_images = label_image_paths
        self._ndim = 2

        assert len(patch_shape) == self._ndim
        self.patch_shape = patch_shape

        self.raw_transform = raw_transform
        self.label_transform = label_transform
        self.label_transform2 = label_transform2
        self.transform = transform
        self.sampler = sampler

        self.dtype = dtype
        self.label_dtype = label_dtype

        if n_samples is None:
            self._len = len(self.raw_images)
            self.sample_random_index = False
        else:
            self._len = n_samples
            self.sample_random_index = True

    def __len__(self):
        return self._len

    @property
    def ndim(self):
        return self._ndim

    def _sample_bounding_box(self, shape):
        if any(sh < psh for sh, psh in zip(shape, self.patch_shape)):
            raise NotImplementedError("Image padding is not supported yet.")
        bb_start = [
            np.random.randint(0, sh - psh) if sh - psh > 0 else 0
            for sh, psh in zip(shape, self.patch_shape)
        ]
        return tuple(slice(start, start + psh) for start, psh in zip(bb_start, self.patch_shape))

    def _get_sample(self, index):
        if self.sample_random_index:
            index = np.random.randint(0, len(self.raw_images))
        # these are just the file paths
        raw, label = self.raw_images[index], self.label_images[index]
        raw = load_image(raw)
        label = load_image(label)

        have_raw_channels = raw.ndim == 3
        have_label_channels = label.ndim == 3
        if have_label_channels:
            raise NotImplementedError("Multi-channel labels are not supported.")

        shape = raw.shape
        # we assume images are loaded with channel last!
        if have_raw_channels:
            shape = shape[:-1]

        # sample random bounding box for this image
        bb = self._sample_bounding_box(shape)

        raw = np.array(raw[bb])
        label = np.array(label[bb])

        # to channel first
        if have_raw_channels:
            raw = raw.transpose((2, 0, 1))

        return raw, label

    def __getitem__(self, index):
        raw, labels = self._get_sample(index)
        initial_label_dtype = labels.dtype

        if self.raw_transform is not None:
            raw = self.raw_transform(raw)

        if self.label_transform is not None:
            labels = self.label_transform(labels)

        if self.transform is not None:
            raw, labels = self.transform(raw, labels)
            # if self.trafo_halo is not None:
            #     raw = self.crop(raw)
            #     labels = self.crop(labels)

        # support enlarging bounding box here as well (for affinity transform) ?
        if self.label_transform2 is not None:
            labels = ensure_spatial_array(labels, self.ndim, dtype=initial_label_dtype)
            labels = self.label_transform2(labels)

        raw = ensure_tensor_with_channels(raw, ndim=self._ndim, dtype=self.dtype)
        labels = ensure_tensor_with_channels(labels, ndim=self._ndim, dtype=self.label_dtype)
        return raw, labels
