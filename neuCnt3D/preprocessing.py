import numpy as np
from skimage.transform import resize


def correct_image_anisotropy(img, px_rsz_ratio, pad, anti_aliasing=True, preserve_range=True):
    """
    Downsample the original microscopy image in the XY plane
    in order to obtain a uniform 3D pixel size.

    Parameters
    ----------
    img: numpy.ndarray (shape=(Z,Y,X))
        microscopy volume image

    px_rsz_ratio: numpy.ndarray (shape=(3,), dtype=float)
        pixel resize ratio

    anti_aliasing: bool
        apply an anti-aliasing filter when downsampling the XY plane

    preserve_range: bool
        keep the original intensity range

    pad: numpy.ndarray (shape=(3,2), dtype=int)
        padding range array

    Returns
    -------
    iso_img: numpy.ndarray (shape=(Z,Y,X))
        padded isotropic microscopy volume image

    iso_crop_img: numpy.ndarray (shape=(Z,Y,X))
        isotropic microscopy volume image
    """
    # lateral downsampling
    iso_shape = np.ceil(np.multiply(np.asarray(img.shape), px_rsz_ratio)).astype(int)
    iso_img = np.zeros(shape=iso_shape, dtype=img.dtype)
    for z in range(iso_shape[0]):
        iso_img[z, ...] = \
            resize(img[z, ...], output_shape=tuple(iso_shape[1:]),
                   anti_aliasing=anti_aliasing, preserve_range=preserve_range)

    # resize padding matrix
    rsz_pad = (np.floor(np.multiply(np.array([px_rsz_ratio, px_rsz_ratio]).transpose(), pad))).astype(int)

    # delete resized padded boundaries
    iso_crop_img = iso_img.copy()
    if np.count_nonzero(rsz_pad) > 0:
        iso_crop_img = iso_img[rsz_pad[0, 0]:iso_img.shape[0] - rsz_pad[0, 1],
                               rsz_pad[1, 0]:iso_img.shape[1] - rsz_pad[1, 1],
                               rsz_pad[2, 0]:iso_img.shape[2] - rsz_pad[2, 1]]

    # estimate resized border to be neglected at the blob detection stage
    # (cast to native Python integer type)
    rsz_border = tuple(np.max(rsz_pad, axis=1))
    rsz_border_int = list()
    for b in rsz_border:
        rsz_border_int.append(b.item())
    rsz_border_int = tuple(rsz_border_int)

    return iso_img, iso_crop_img, rsz_pad, rsz_border_int
