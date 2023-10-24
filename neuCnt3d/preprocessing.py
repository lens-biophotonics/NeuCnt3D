import numpy as np
from skimage.transform import resize


def correct_anisotropy(img, px_rsz_ratio, pad, slice_ovlp, anti_aliasing=True, preserve_range=True):
    """
    Resize the original microscopy image in the XY plane
    in order to obtain a uniform 3D pixel size.

    Parameters
    ----------
    img: numpy.ndarray (axis order: (Z,Y,X))
        microscopy volume image

    px_rsz_ratio: numpy.ndarray (shape=(3,), dtype=float)
        pixel resize ratio

    pad: numpy.ndarray (shape=(3,2), dtype=int)
        padding range array

    slice_ovlp: int
        image slice lateral overlap

    anti_aliasing: bool
        apply an anti-aliasing filter when resizing the XY plane

    preserve_range: bool
        keep the original intensity range

    Returns
    -------
    iso_img: numpy.ndarray (shape=(Z,Y,X))
        padded isotropic microscopy volume image

    rsz_pad: numpy.ndarray (shape=(3,2), dtype=int)
        resized padding range array

    rsz_border_int: tuple
        each element of the tuple will exclude peaks from within exclude_border-pixels
        of the border of the image along that dimension
    """
    # lateral resizing
    iso_shape = np.ceil(np.multiply(np.asarray(img.shape), px_rsz_ratio)).astype(int)
    iso_img = np.zeros(shape=iso_shape, dtype=img.dtype)
    for z in range(iso_shape[0]):
        iso_img[z, ...] = \
            resize(img[z, ...], output_shape=tuple(iso_shape[1:]),
                   anti_aliasing=anti_aliasing, preserve_range=preserve_range)

    # get resized image padding matrix
    rsz_pad = (np.floor(np.multiply(np.array([px_rsz_ratio, px_rsz_ratio]).transpose(), pad))).astype(int)

    # get resized slice lateral overlap
    rsz_slice_ovlp = np.multiply(np.array([slice_ovlp, slice_ovlp, slice_ovlp]), px_rsz_ratio).astype(int)

    # estimate resized border to be neglected at the blob detection stage
    # (cast to native Python integer type)
    rsz_border = tuple(np.max(rsz_pad, axis=1))
    rsz_border_int = list()
    for b in rsz_border:
        rsz_border_int.append(b.item())
    rsz_border_int = tuple(rsz_border_int)

    return iso_img, rsz_pad, rsz_border_int, rsz_slice_ovlp
