import numpy as np

from skimage.feature import blob_dog, blob_log


def detect_soma(img, min_sigma=1, max_sigma=50, num_sigma=10, sigma_ratio=1.6, method='log',
                threshold=None, overlap=0.5, threshold_rel=None, exclude_border=False, dark=False):
    """
    Apply 3D soma segmentation filter to input volume image.

    Parameters
    ----------

    Returns
    -------

    """
    # invert image
    if dark:
        img = np.invert(img)

    # detect blobs
    if method == 'log':
        blobs = blob_log(img, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma, threshold=threshold,
                         overlap=overlap, threshold_rel=threshold_rel, exclude_border=exclude_border)
    elif method == 'dog':
        blobs = blob_dog(img, min_sigma=min_sigma, max_sigma=max_sigma, sigma_ratio=sigma_ratio, threshold=threshold,
                         overlap=overlap, threshold_rel=threshold_rel, exclude_border=exclude_border)

    # estimate blob radii
    blobs[..., 3] = blobs[..., 3] * np.sqrt(3)

    return blobs
