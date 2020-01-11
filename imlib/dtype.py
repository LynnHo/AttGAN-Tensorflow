import numpy as np


def _check(images, dtypes, min_value=-np.inf, max_value=np.inf):
    # check type
    assert isinstance(images, np.ndarray), '`images` should be np.ndarray!'

    # check dtype
    dtypes = dtypes if isinstance(dtypes, (list, tuple)) else [dtypes]
    assert images.dtype in dtypes, 'dtype of `images` shoud be one of %s!' % dtypes

    # check nan and inf
    assert np.all(np.isfinite(images)), '`images` contains NaN or Inf!'

    # check value
    if min_value not in [None, -np.inf]:
        l = '[' + str(min_value)
    else:
        l = '(-inf'
        min_value = -np.inf
    if max_value not in [None, np.inf]:
        r = str(max_value) + ']'
    else:
        r = 'inf)'
        max_value = np.inf
    assert np.min(images) >= min_value and np.max(images) <= max_value, \
        '`images` should be in the range of %s!' % (l + ',' + r)


def to_range(images, min_value=0.0, max_value=1.0, dtype=None):
    """Transform images from [-1.0, 1.0] to [min_value, max_value] of dtype."""
    _check(images, [np.float32, np.float64], -1.0, 1.0)
    dtype = dtype if dtype else images.dtype
    return ((images + 1.) / 2. * (max_value - min_value) + min_value).astype(dtype)


def float2im(images):
    """Transform images from [0, 1.0] to [-1.0, 1.0]."""
    _check(images, [np.float32, np.float64], 0.0, 1.0)
    return images * 2 - 1.0


def float2uint(images):
    """Transform images from [0, 1.0] to uint8."""
    _check(images, [np.float32, np.float64], -0.0, 1.0)
    return (images * 255).astype(np.uint8)


def im2uint(images):
    """Transform images from [-1.0, 1.0] to uint8."""
    return to_range(images, 0, 255, np.uint8)


def im2float(images):
    """Transform images from [-1.0, 1.0] to [0.0, 1.0]."""
    return to_range(images, 0.0, 1.0)


def uint2im(images):
    """Transform images from uint8 to [-1.0, 1.0] of float64."""
    _check(images, np.uint8)
    return images / 127.5 - 1.0


def uint2float(images):
    """Transform images from uint8 to [0.0, 1.0] of float64."""
    _check(images, np.uint8)
    return images / 255.0


def cv2im(images):
    """Transform opencv images to [-1.0, 1.0]."""
    images = uint2im(images)
    return images[..., ::-1]


def im2cv(images):
    """Transform images from [-1.0, 1.0] to opencv images."""
    images = im2uint(images)
    return images[..., ::-1]
