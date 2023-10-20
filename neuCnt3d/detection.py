import numpy as np
from skimage.feature import blob_dog, blob_log


def config_detection_scales(diam_um, px_size):
    """
    Compute the minimum and maximum standard deviation
    for the Gaussian kernel used by the blob detection algorithm.

    Parameters
    ----------
    diam_um: tuple
        soma diameter (minimum, maximum, step size) [μm]

    px_size: numpy.ndarray (shape=(3,), dtype=float)
        pixel size [μm]

    Returns
    -------
    sigma_px: numpy.ndarray (shape=(2,), dtype=int)
        minimum and maximum spatial scales [px]

    sigma_num: int
        number of spatial scales analyzed
    """
    min_diam_um, max_diam_um, stp_diam_um = diam_um
    sigma_num = len(np.arange(min_diam_um, max_diam_um, stp_diam_um))
    sigma_px = np.array([min_diam_um, max_diam_um]) / (2 * np.sqrt(3) * np.max(px_size))

    # one scale at least
    if sigma_num <= 0:
        sigma_num = 1

    return sigma_px, sigma_num


def correct_blob_coord(blobs, slice_rng, slice_ovlp, z_sel):
    """
    Correct the original soma coordinates with respect to
    the relative image slice position and delete detections
    out of the requested depth range.

    Parameters
    ----------
    blobs: numpy.ndarray (shape=(N,4))
        2D array with each row representing 3 coordinate values for a 3D image,
        plus the best sigma of the Gaussian kernel which detected the blob

    slice_rng: NumPy slice object
        image slice index range

    slice_ovlp: int

    z_sel: NumPy slice object
        selected z-depth range

    Returns
    -------
    blobs: numpy.ndarray (shape=(N,4))
        2D array with each row representing 3 coordinate values for a 3D image,
        plus the best sigma of the Gaussian kernel which detected the blob
    """
    # correct X, Y coordinates wrt x0, y0 offset
    blobs[:, 1] += slice_rng[1].start - slice_ovlp[1]
    blobs[:, 2] += slice_rng[2].start - slice_ovlp[2]

    # exclude detections out of z_sel
    z_msk = blobs[:, 0] >= z_sel.start
    if z_sel.stop is not None:
        z_msk = np.logical_and(z_msk, blobs[:, 0] <= z_sel.stop)

    blobs = blobs[z_msk, :]

    return blobs


def detect_soma(img, min_sigma=1, max_sigma=50, num_sigma=10, sigma_ratio=1.6, approach='log',
                thresh=None, thresh_rel=None, overlap=0.5, border=0, dark=False):
    """
    Apply 3D soma segmentation filter to input volume image.

    Parameters
    ----------
    img: numpy.ndarray or memory-mapped file (axis order: (Z,Y,X))
        soma fluorescence volume image

    min_sigma: int
        minimum spatial scale [px]

    max_sigma: int
        maximum spatial scale [px]

    num_sigma: int
        number of spatial scales analyzed

    sigma_ratio: float
        the ratio between the standard deviation of Gaussian kernels
        used for computing the Difference of Gaussian

    approach: str
        blob detection approach
        (Laplacian of Gaussian or Difference of Gaussian)

    thresh: float
        minimum intensity of peaks in the filtered image

    thresh_rel: float
        minimum percentage intensity of peaks in the filtered image relative to maximum [%]

    overlap: float
        maximum blob overlap percentage [%]

    border: tuple
        each element of the tuple will exclude peaks from within exclude_border-pixels
        of the border of the image along that dimension

    dark: bool
        if True, detect dark 3D blob-like structures
        (i.e., negative contrast polarity)

    Returns
    -------
    blobs: numpy.ndarray (shape=(N,4))
        2D array with each row representing 3 coordinate values for a 3D image,
        plus the best sigma of the Gaussian kernel which detected the blob

    NOTE:
    modify skimage.feature.blob (line 205)
    as follows:

    return exclude_border >>> return tuple(list(exclude_border) + [0])
    """
    # invert image
    if dark:
        img = np.invert(img)

    # detect blobs
    if approach == 'log':
        blobs = blob_log(img, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma, threshold=thresh,
                         threshold_rel=thresh_rel, overlap=overlap, exclude_border=border)
    elif approach == 'dog':
        blobs = blob_dog(img, min_sigma=min_sigma, max_sigma=max_sigma, sigma_ratio=sigma_ratio, threshold=thresh,
                         threshold_rel=thresh_rel, overlap=overlap, exclude_border=border)
    else:
        raise ValueError('Unrecognized blob detection approach! '
                         'This must be either "log" (Laplacian of Gaussian) or "dog" (Difference of Gaussian)...')

    # estimate blob radii
    blobs[..., 3] = blobs[..., 3] * np.sqrt(3)

    return blobs


def merge_parallel_blobs(par_blobs, inv=-1):
    """
    Merge blobs detected by parallel processes or threads,
    filtering out invalid background slices.

    Parameters
    ----------
    par_blobs: list of numpy.ndarray
        list of blob coordinates extracted in parallel
        from separate basic image slices

    inv: int or float
        invalid value assigned to skipped
        background slices

    Returns
    -------
    blobs: numpy.ndarray (shape=(N,4))
        2D array with each row representing 3 coordinate values for a 3D image,
        plus the best sigma of the Gaussian kernel which detected the blob

    """
    inv = inv * np.ones((4,))
    par_blobs = [x for x in par_blobs if not (x == inv).all()]
    blobs = np.vstack(par_blobs)

    return blobs
