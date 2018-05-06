from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from imlib.dtype import *
import numpy as np
import scipy.misc


def rgb2gray(images):
    if images.ndim == 4 or images.ndim == 3:
        assert images.shape[-1] == 3, 'Channel size should be 3!'
    else:
        raise Exception('Wrong dimensions!')

    return (images[..., 0] * 0.299 + images[..., 1] * 0.587 + images[..., 2] * 0.114).astype(images.dtype)


def gray2rgb(images):
    assert images.ndim == 2 or images.ndim == 3, 'Wrong dimensions!'
    rgb_imgs = np.zeros(images.shape + (3,), dtype=images.dtype)
    rgb_imgs[..., 0] = images
    rgb_imgs[..., 1] = images
    rgb_imgs[..., 2] = images
    return rgb_imgs


def imresize(image, size, interp='bilinear'):
    """Resize an [-1.0, 1.0] image.

    Args:
        size : int, float or tuple
            * int   - Percentage of current size.
            * float - Fraction of current size.
            * tuple - Size of the output image.

        interp : str, optional
            Interpolation to use for re-sizing ('nearest', 'lanczos',
            'bilinear', 'bicubic' or 'cubic').
    """
    # scipy.misc.imresize should deal with uint8 image, or it would cause some
    # problem (scale the image to [0, 255])
    return (scipy.misc.imresize(im2uint(image), size, interp=interp) / 127.5 - 1).astype(image.dtype)


def resize_images(images, size, interp='bilinear'):
    """Resize batch [-1.0, 1.0] images of shape (N * H * W (* 3)).

    Args:
        size : int, float or tuple
            * int   - Percentage of current size.
            * float - Fraction of current size.
            * tuple - Size of the output image.

        interp : str, optional
            Interpolation to use for re-sizing ('nearest', 'lanczos',
            'bilinear', 'bicubic' or 'cubic').
    """
    rs_imgs = []
    for img in images:
        rs_imgs.append(imresize(img, size, interp))
    return np.array(rs_imgs)


def immerge(images, row, col):
    """Merge images into an image with (row * h) * (col * w).

    `images` is in shape of N * H * W(* C=1 or 3)
    """
    h, w = images.shape[1], images.shape[2]
    if images.ndim == 4:
        img = np.zeros((h * row, w * col, images.shape[3]))
    elif images.ndim == 3:
        img = np.zeros((h * row, w * col))
    for idx, image in enumerate(images):
        i = idx % col
        j = idx // col
        img[j * h:j * h + h, i * w:i * w + w, ...] = image

    return img
