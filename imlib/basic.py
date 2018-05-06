from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from imlib.dtype import *
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc


def imread(paths, mode='RGB'):
    """Read image(s).

    if `paths` is a list or tuple, then read a list of images into [-1.0, 1.0]
    of float and return the numpy array batch in shape of N * H * W (* C)
    if `paths` is a single str, then read an image into [-1.0, 1.0] of float

    Args:
        mode: It can be one of the following strings:
            * 'L' (8 - bit pixels, black and white)
            * 'P' (8 - bit pixels, mapped to any other mode using a color palette)
            * 'RGB' (3x8 - bit pixels, true color)
            * 'RGBA' (4x8 - bit pixels, true color with transparency mask)
            * 'CMYK' (4x8 - bit pixels, color separation)
            * 'YCbCr' (3x8 - bit pixels, color video format)
            * 'I' (32 - bit signed integer pixels)
            * 'F' (32 - bit floating point pixels)

    Returns:
        Float64 image in [-1.0, 1.0].
    """
    def _imread(path, mode='RGB'):
        return scipy.misc.imread(path, mode=mode) / 127.5 - 1

    if isinstance(paths, (list, tuple)):
        images = [_imread(path, mode) for path in paths]
        return np.array(images)
    else:
        return _imread(paths, mode)


def imwrite(image, path):
    """Save an [-1.0, 1.0] image."""
    if image.ndim == 3 and image.shape[2] == 1:  # for gray image
        image = np.array(image, copy=True)
        image.shape = image.shape[0:2]
    return scipy.misc.imsave(path, to_range(image, 0, 255, np.uint8))


def imshow(image):
    """Show a [-1.0, 1.0] image."""
    if image.ndim == 3 and image.shape[2] == 1:  # for gray image
        image = np.array(image, copy=True)
        image.shape = image.shape[0:2]
    plt.imshow(to_range(image), cmap=plt.gray())


show = plt.show
