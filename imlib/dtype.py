from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def to_range(images, min_value=0.0, max_value=1.0, dtype=None):
    """Transform images from [-1.0, 1.0] to [min_value, max_value] of dtype."""
    assert np.min(images) >= -1.0 - 1e-5 and np.max(images) <= 1.0 + 1e-5 \
        and (images.dtype == np.float32 or images.dtype == np.float64), \
        ('The input images should be float64(32) '
         'and in the range of [-1.0, 1.0]!')
    if dtype is None:
        dtype = images.dtype
    return ((images + 1.) / 2. * (max_value - min_value) +
            min_value).astype(dtype)


def uint2im(images):
    """Transform images from uint8 to [-1.0, 1.0] of float64."""
    assert images.dtype == np.uint8, 'The input images type should be uint8!'
    return images / 127.5 - 1.0


def float2im(images):
    """Transform images from [0, 1.0] to [-1.0, 1.0]."""
    assert np.min(images) >= 0.0 - 1e-5 and np.max(images) <= 1.0 + 1e-5 \
        and (images.dtype == np.float32 or images.dtype == np.float64), \
        'The input images should be float64(32) and in the range of [0.0, 1.0]!'
    return images * 2 - 1.0


def im2uint(images):
    """Transform images from [-1.0, 1.0] to uint8."""
    return to_range(images, 0, 255, np.uint8)


def im2float(images):
    """Transform images from [-1.0, 1.0] to [0.0, 1.0]."""
    return to_range(images, 0.0, 1.0)


def float2uint(images):
    """Transform images from [0, 1.0] to uint8."""
    assert np.min(images) >= 0.0 - 1e-5 and np.max(images) <= 1.0 + 1e-5 \
        and (images.dtype == np.float32 or images.dtype == np.float64), \
        'The input images should be float64(32) and in the range of [0.0, 1.0]!'
    return (images * 255).astype(np.uint8)


def uint2float(images):
    """Transform images from uint8 to [0.0, 1.0] of float64."""
    assert images.dtype == np.uint8, 'The input images type should be uint8!'
    return images / 255.0
