import numpy as np


def align_crop_opencv(img,
                      src_landmarks,
                      standard_landmarks,
                      crop_size=512,
                      face_factor=0.7,
                      align_type='similarity',
                      order=3,
                      mode='edge'):
    """Align and crop a face image by landmarks.

    Arguments:
        img                : Face image to be aligned and cropped.
        src_landmarks      : [[x_1, y_1], ..., [x_n, y_n]].
        standard_landmarks : Standard shape, should be normalized.
        crop_size          : Output image size, should be 1. int for (crop_size, crop_size)
                             or 2. (int, int) for (crop_size_h, crop_size_w).
        face_factor        : The factor of face area relative to the output image.
        align_type         : 'similarity' or 'affine'.
        order              : The order of interpolation. The order has to be in the range 0-5:
                                 - 0: INTER_NEAREST
                                 - 1: INTER_LINEAR
                                 - 2: INTER_AREA
                                 - 3: INTER_CUBIC
                                 - 4: INTER_LANCZOS4
                                 - 5: INTER_LANCZOS4
        mode               : One of ['constant', 'edge', 'symmetric', 'reflect', 'wrap'].
                             Points outside the boundaries of the input are filled according
                             to the given mode.
    """
    # set OpenCV
    import cv2
    inter = {0: cv2.INTER_NEAREST, 1: cv2.INTER_LINEAR, 2: cv2.INTER_AREA,
             3: cv2.INTER_CUBIC, 4: cv2.INTER_LANCZOS4, 5: cv2.INTER_LANCZOS4}
    border = {'constant': cv2.BORDER_CONSTANT, 'edge': cv2.BORDER_REPLICATE,
              'symmetric': cv2.BORDER_REFLECT, 'reflect': cv2.BORDER_REFLECT101,
              'wrap': cv2.BORDER_WRAP}

    # check
    assert align_type in ['affine', 'similarity'], 'Invalid `align_type`! Allowed: %s!' % ['affine', 'similarity']
    assert order in [0, 1, 2, 3, 4, 5], 'Invalid `order`! Allowed: %s!' % [0, 1, 2, 3, 4, 5]
    assert mode in ['constant', 'edge', 'symmetric', 'reflect', 'wrap'], 'Invalid `mode`! Allowed: %s!' % ['constant', 'edge', 'symmetric', 'reflect', 'wrap']

    # crop size
    if isinstance(crop_size, (list, tuple)) and len(crop_size) == 2:
        crop_size_h = crop_size[0]
        crop_size_w = crop_size[1]
    elif isinstance(crop_size, int):
        crop_size_h = crop_size_w = crop_size
    else:
        raise Exception('Invalid `crop_size`! `crop_size` should be 1. int for (crop_size, crop_size) or 2. (int, int) for (crop_size_h, crop_size_w)!')

    # estimate transform matrix
    trg_landmarks = standard_landmarks * max(crop_size_h, crop_size_w) * face_factor + np.array([crop_size_w // 2, crop_size_h // 2])
    if align_type == 'affine':
        tform = cv2.estimateAffine2D(trg_landmarks, src_landmarks, ransacReprojThreshold=np.Inf)[0]
    else:
        tform = cv2.estimateAffinePartial2D(trg_landmarks, src_landmarks, ransacReprojThreshold=np.Inf)[0]

    # warp image by given transform
    output_shape = (crop_size_h, crop_size_w)
    img_crop = cv2.warpAffine(img, tform, output_shape[::-1], flags=cv2.WARP_INVERSE_MAP + inter[order], borderMode=border[mode])

    # get transformed landmarks
    tformed_landmarks = cv2.transform(np.expand_dims(src_landmarks, axis=0), cv2.invertAffineTransform(tform))[0]

    return img_crop, tformed_landmarks


def align_crop_skimage(img,
                       src_landmarks,
                       standard_landmarks,
                       crop_size=512,
                       face_factor=0.7,
                       align_type='similarity',
                       order=3,
                       mode='edge'):
    """Align and crop a face image by landmarks.

    Arguments:
        img                : Face image to be aligned and cropped.
        src_landmarks      : [[x_1, y_1], ..., [x_n, y_n]].
        standard_landmarks : Standard shape, should be normalized.
        crop_size          : Output image size, should be 1. int for (crop_size, crop_size)
                             or 2. (int, int) for (crop_size_h, crop_size_w).
        face_factor        : The factor of face area relative to the output image.
        align_type         : 'similarity' or 'affine'.
        order              : The order of interpolation. The order has to be in the range 0-5:
                                 - 0: INTER_NEAREST
                                 - 1: INTER_LINEAR
                                 - 2: INTER_AREA
                                 - 3: INTER_CUBIC
                                 - 4: INTER_LANCZOS4
                                 - 5: INTER_LANCZOS4
        mode               : One of ['constant', 'edge', 'symmetric', 'reflect', 'wrap'].
                             Points outside the boundaries of the input are filled according
                             to the given mode.
    """
    raise NotImplementedError("'align_crop_skimage' is not implemented!")
