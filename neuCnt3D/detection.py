import numpy as np

from skimage.feature import blob_dog, blob_log


def config_detection_scales(diam_um, px_size):
    """
    Description

    Parameters
    ----------
    diam_um

    px_size

    Returns
    -------
    sigma_px

    sigma_num

    """
    min_diam_um, max_diam_um, stp_diam_um = diam_um
    sigma_num = len(np.arange(min_diam_um, max_diam_um, stp_diam_um))
    sigma_px = np.array([min_diam_um, max_diam_um]) / (2 * np.sqrt(3) * np.max(px_size))

    # one scale at least
    if sigma_num <= 0:
        sigma_num = 1

    return sigma_px, sigma_num


def correct_blob_coord(blobs, slice_rng, rsz_pad, z_sel):
    """
    Description

    Parameters
    ----------
    blobs

    slice_rng

    z_sel

    Returns
    -------
    blobs

    """
    # correct X, Y coordinates wrt x0, y0 offset
    blobs[:, 1] += slice_rng[1].start - rsz_pad[1, 0]
    blobs[:, 2] += slice_rng[2].start - rsz_pad[2, 0]

    # exclude detections out of z_sel
    z_msk = blobs[:, 0] >= z_sel.start
    if z_sel.stop is not None:
        z_msk = np.logical_and(z_msk, blobs[:, 0] <= z_sel.stop)

    blobs = blobs[z_msk, :]

    return blobs


def detect_soma(img, min_sigma=1, max_sigma=50, num_sigma=10, sigma_ratio=1.6, method='log',
                threshold=None, overlap=0.5, threshold_rel=None, border=0, dark=False):
    """
    Apply 3D soma segmentation filter to input volume image.

    Parameters
    ----------
    img

    min_sigma

    max_sigma

    num_sigma

    sigma_ratio

    method

    threshold

    overlap

    threshold_rel

    border

    dark

    Returns
    -------
    blobs

    NOTE:
    modify skimage.feature.blob (line 205)
    as follows:

    return exclude_border >>> return tuple(list(exclude_border) + [0])
    """
    # invert image
    if dark:
        img = np.invert(img)

    # detect blobs
    if method == 'log':
        blobs = blob_log(img, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma, threshold=threshold,
                         overlap=overlap, threshold_rel=threshold_rel, exclude_border=border)
    elif method == 'dog':
        blobs = blob_dog(img, min_sigma=min_sigma, max_sigma=max_sigma, sigma_ratio=sigma_ratio, threshold=threshold,
                         overlap=overlap, threshold_rel=threshold_rel, exclude_border=border)

    # estimate blob radii
    blobs[..., 3] = blobs[..., 3] * np.sqrt(3)

    return blobs
