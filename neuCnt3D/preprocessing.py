import numpy as np
from skimage.transform import resize


def correct_image_anisotropy(img, px_size, anti_aliasing=True, preserve_range=True):
    """
    Downsample the original microscopy image in the XY plane
    in order to obtain a uniform 3D pixel size.

    Parameters
    ----------
    img: numpy.ndarray (shape=(Z,Y,X))
        microscopy volume image

    px_size: numpy.ndarray (shape=(3,), dtype=float)
        pixel size [μm]

    anti_aliasing: bool
        apply an anti-aliasing filter when downsampling the XY plane

    preserve_range: bool
        keep the original intensity range

    Returns
    -------
    iso_img: numpy.ndarray (shape=(Z,Y,X))
        isotropic microscopy volume image
    """
    # new isotropic pixel size
    px_size_iso = np.max(px_size) * np.ones(shape=px_size.shape)
    rsz_ratio = np.divide(px_size, px_size_iso)

    # lateral downsampling
    iso_shape = np.ceil(np.multiply(np.asarray(img.shape), rsz_ratio)).astype(int)
    iso_img = np.zeros(shape=iso_shape, dtype=img.dtype)
    for z in range(iso_shape[0]):
        iso_img[z, ...] = \
            resize(img[z, ...], output_shape=tuple(iso_shape[1:]),
                   anti_aliasing=anti_aliasing, preserve_range=preserve_range)

    return iso_img
