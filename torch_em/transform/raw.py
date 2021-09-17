import numpy as np


#
# normalization functions
#


def standardize(raw, mean=None, std=None, axis=None, eps=1e-7):
    raw = raw.astype('float32')

    mean = raw.mean(axis=axis, keepdims=True) if mean is None else mean
    raw -= mean

    std = raw.std(axis=axis, keepdims=True) if std is None else std
    raw /= (std + eps)

    return raw


def normalize(raw, minval=None, maxval=None, axis=None, eps=1e-7):
    raw = raw.astype('float32')

    minval = raw.min(axis=axis, keepdims=True) if minval is None else minval
    raw -= minval

    maxval = raw.max(axis=axis, keepdims=True) if maxval is None else maxval
    raw /= (maxval + eps)

    return raw


def normalize_percentile(raw, lower=1.0, upper=99.0, axis=None, eps=1e-7):
    v_lower = np.percentile(raw, lower, axis=axis, keepdims=True)
    v_upper = np.percentile(raw, upper, axis=axis, keepdims=True) - v_lower
    return normalize(raw, v_lower, v_upper, eps=eps)


# TODO
#
# intensity augmentations / noise augmentations
#


#
# default transformation:
# apply intensity augmentations and normalize
#

class RawTransform:
    def __init__(self, normalizer, augmentation1=None, augmentation2=None):
        self.normalizer = normalizer
        self.augmentation1 = augmentation1
        self.augmentation2 = augmentation2

    def __call__(self, raw):
        if self.augmentation1 is not None:
            raw = self.augmentation1(raw)
        raw = self.normalizer(raw)
        if self.augmentation2 is not None:
            raw = self.augmentation2(raw)
        return raw


def get_raw_transform(normalizer=standardize, augmentation1=None, augmentation2=None):
    return RawTransform(normalizer,
                        augmentation1=augmentation1,
                        augmentation2=augmentation2)
