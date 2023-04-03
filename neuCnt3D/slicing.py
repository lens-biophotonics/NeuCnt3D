from itertools import product

import numpy as np
import psutil
from neuCnt3D.utils import get_available_cores


def adjust_slice_coord(axis_iter, pad_rng, slice_shape, img_shape, slice_per_dim, axis, flip=False, min_edge=-1):
    """
    Adjust image slice coordinates at boundaries.

    Parameters
    ----------
    axis_iter: int
        iteration counter along axis (see pipeline.iterate_frangi_on_slices)

    pad_rng: int
        patch padding range

    slice_shape: numpy.ndarray (shape=(3,), dtype=int)
        shape of the basic image slices analyzed iteratively [px]

    img_shape: numpy.ndarray (shape=(3,), dtype=int)
        total image shape [px]

    slice_per_dim: int
        number of image slices along axis

    axis: int
        axis index

    flip: bool
        if True, flip axes coordinates

    min_edge: int
        threshold on the minimum side of boundary image slices [px]

    Returns
    -------
    start: int
        adjusted start index

    stop: int
        adjusted stop index

    axis_pad_array: numpy.ndarray (shape=(2,), dtype=int)
        axis pad range [\'left\', \'right\']
    """
    # initialize axis pad array
    axis_pad_array = np.zeros(shape=(2,), dtype=np.uint8)

    # compute start and stop coordinates
    start = axis_iter * slice_shape[axis] - pad_rng
    stop = axis_iter * slice_shape[axis] + slice_shape[axis] + pad_rng

    # flip coordinates
    if flip:
        start_tmp = start
        start = img_shape[axis] - stop
        stop = img_shape[axis] - start_tmp

    # adjust start coordinate
    if start < 0:
        start = 0
    else:
        axis_pad_array[0] = pad_rng

    # adjust stop coordinate
    if stop > img_shape[axis]:
        stop = img_shape[axis]
    else:
        axis_pad_array[1] = pad_rng

    # handle image boundaries for very dense image parcellations
    if min_edge > 0:
        if axis_iter == slice_per_dim - 2:
            if img_shape[axis] - (stop - axis_pad_array[1]) < min_edge:
                stop = img_shape[axis]
        elif axis_iter == slice_per_dim - 1:
            if (stop - axis_pad_array[1]) - (start + axis_pad_array[0]) < min_edge:
                start = None

    return start, stop, axis_pad_array


def compute_slice_range(z, y, x, slice_shape, img_shape, slice_per_dim, pad_rng=0, flip=False, min_edge=-1):
    """
    Compute basic slice coordinates from microscopy volume image.

    Parameters
    ----------
    z: int
        z-depth index

    y: int
        row index

    x: int
        column index

    slice_shape: numpy.ndarray (shape=(3,), dtype=int)
        shape of the basic image slices analyzed iteratively [px]

    img_shape: numpy.ndarray (shape=(3,), dtype=int)
        total image shape [px]

    slice_per_dim: numpy.ndarray (shape=(3,), dtype=int)
        number of image slices along each separate axis

    pad_rng: int
        slice padding range

    flip: bool
        if True, flip axes coordinates

    Returns
    -------
    rng: tuple
        3D slice index ranges

    pad_mat: numpy.ndarray
        3D padding range array
    """
    # adjust original image patch coordinates
    # and generate padding range matrix
    pad_mat = np.zeros(shape=(3, 2), dtype=np.uint8)
    z_start, z_stop, pad_mat[0, :] = \
        adjust_slice_coord(z, pad_rng, slice_shape, img_shape, slice_per_dim[0], axis=0, flip=flip, min_edge=min_edge)
    y_start, y_stop, pad_mat[1, :] = \
        adjust_slice_coord(y, pad_rng, slice_shape, img_shape, slice_per_dim[1], axis=1, flip=flip, min_edge=min_edge)
    x_start, x_stop, pad_mat[2, :] = \
        adjust_slice_coord(x, pad_rng, slice_shape, img_shape, slice_per_dim[2], axis=2, flip=flip, min_edge=min_edge)

    # generate index ranges
    z_rng = slice(z_start, z_stop, 1)
    y_rng = slice(y_start, y_stop, 1)
    x_rng = slice(x_start, x_stop, 1)
    rng = np.index_exp[z_rng, y_rng, x_rng]
    for r in rng:
        if r.start is None:
            return None, pad_mat

    return rng, pad_mat


def compute_slice_shape(img_shape, item_size, px_size=None, max_size=100, pad_rng=0):
    """
    Compute basic image chunk shape depending on its maximum size (in bytes).

    Parameters
    ----------
    img_shape: numpy.ndarray (shape=(3,))
        total image shape [px]

    item_size: int
        image item size (in bytes)

    px_size: numpy.ndarray (shape=(3,), dtype=float)
        pixel size [μm]

    max_slice_size: float
        maximum memory size (in bytes) of the basic slices analyzed iteratively

    pad_rng: int
        slice padding range

    Returns
    -------
    slice_shape: numpy.ndarray (shape=(3,), dtype=int)
        shape of the basic image slices analyzed in parallel [px]

    slice_shape_um: numpy.ndarray (shape=(3,), dtype=float)
        shape of the basic image slices analyzed in parallel [μm]
        (if px_size is provided)
    """
    if len(img_shape) == 4:
        item_size = item_size * 3

    slice_depth = img_shape[0]
    slice_side = np.round(1024 * np.sqrt((max_size / (slice_depth * item_size))) - 2 * pad_rng)
    slice_shape = np.array([slice_depth, slice_side, slice_side]).astype(int)
    slice_shape = np.min(np.stack((img_shape[:3], slice_shape)), axis=0)

    if px_size is not None:
        slice_shape_um = np.multiply(slice_shape, px_size)
        return slice_shape, slice_shape_um
    else:
        return slice_shape


def config_image_slicing(img_shape, item_size, px_size, batch_size, slice_size):
    """
    Slicing configuration for the parallel analysis of basic chunks of the input microscopy volume.

    Parameters
    ----------
    image_shape: numpy.ndarray (shape=(3,), dtype=int)
        total image shape [px]

    item_size: int
        image item size (in bytes)

    px_size: numpy.ndarray (shape=(3,), dtype=float)
        pixel size [μm]

    px_size_iso: numpy.ndarray (shape=(3,), dtype=float)
        adjusted isotropic pixel size [μm]

    smooth_sigma: numpy.ndarray (shape=(3,), dtype=int)
        3D standard deviation of low-pass Gaussian filter [px]
        (resolution anisotropy correction)

    scale_sigma: numpy.ndarray (or int)
        spatial scales of interest [px]

    batch_size: int
        slice batch size

    slice_size: float
        maximum memory size (in megabytes) of the basic image slices
        analyzed in parallel

    Returns
    -------
    rng_in_lst: list
        list of analyzed fiber channel slice ranges

    pad_mat_lst: list
        list of slice padding ranges

    in_slice_shape_um: numpy.ndarray (shape=(3,), dtype=float)
        shape of the basic image slices analyzed iteratively [μm]

    tot_slice_num: int
        total number of analyzed image slices

    batch_size: int
        adjusted slice batch size
    """
    # shape of processed TPFM slices
    in_slice_shape, in_slice_shape_um = compute_slice_shape(img_shape, item_size,
                                                            px_size=px_size, max_size=slice_size)

    # iteratively define input/output slice 3D ranges
    slice_per_dim = np.ceil(np.divide(img_shape, in_slice_shape)).astype(int)
    tot_slice_num = int(np.prod(slice_per_dim))

    # initialize empty range lists
    rng_in_lst = list()
    for z, y, x in product(range(slice_per_dim[0]), range(slice_per_dim[1]), range(slice_per_dim[2])):

        # index ranges of analyzed fiber image slices (with padding)
        rng_in, _ = \
            compute_slice_range(z, y, x, in_slice_shape, img_shape, slice_per_dim)
        if rng_in is not None:
            rng_in_lst.append(rng_in)
        else:
            tot_slice_num -= 1

    # adjust slice batch size
    if batch_size > tot_slice_num:
        batch_size = tot_slice_num

    return rng_in_lst, in_slice_shape, in_slice_shape_um, tot_slice_num, batch_size


def config_slice_batch(frangi_scales, mem_growth_factor=149.7, mem_fudge_factor=1.0,                                    # NOTE: default values need to be adjusted!
                       min_slice_size_mb=-1, jobs_to_cores=0.8, max_ram_mb=None):
    """
    Compute size and number of the batches of basic microscopy image slices
    analyzed in parallel.

    Parameters
    ----------
    frangi_scales: list (dtype=float)
        analyzed spatial scales in [μm]

    mem_growth_factor: float
        empirical memory growth factor
        of the Frangi filtering stage

    mem_fudge_factor: float
        memory fudge factor

    min_slice_size_mb: float
        minimum slice size in [MB]

    jobs_to_cores: float
        max number of jobs relative to the available CPU cores
        (default: 80%)

    max_ram_mb: float
        maximum RAM available to the Frangi filtering stage [MB]

    Returns
    -------
    slice_batch_size: int
        slice batch size

    slice_size_mb: float
        memory size (in megabytes) of the basic image slices
        fed to the Frangi filter
    """
    # maximum RAM not provided: use all
    if max_ram_mb is None:
        max_ram_mb = psutil.virtual_memory()[1] / 1e6

    # number of logical cores
    num_cpu = get_available_cores()

    # number of spatial scales
    num_scales = len(frangi_scales)

    # initialize slice batch size
    slice_batch_size = int(jobs_to_cores * num_cpu // num_scales)

    # get image slice size
    slice_size_mb = get_slice_size(max_ram_mb, mem_growth_factor, mem_fudge_factor, slice_batch_size, num_scales)
    while slice_size_mb < min_slice_size_mb:
        slice_batch_size -= 1
        slice_size_mb = get_slice_size(max_ram_mb, mem_growth_factor, mem_fudge_factor, slice_batch_size, num_scales)

    return slice_batch_size, slice_size_mb


def crop_slice(img_slice, rng, flipped=()):
    """
    Shrink image slice at volume boundaries, for overall shape consistency.

    Parameters
    ----------
    img_slice: numpy.ndarray
        image slice

    rng: tuple
        3D index range

    flipped: tuple
        flipped axes

    Returns
    -------
    cropped_slice: numpy.ndarray
        cropped image slice
    """
    # check slice shape and output index ranges
    out_slice_shape = img_slice.shape
    crop_rng = np.zeros(shape=(3,), dtype=int)
    for s in range(3):
        crop_rng[s] = out_slice_shape[s] - np.arange(rng[s].start, rng[s].stop, rng[s].step).size

    # crop slice if required
    if 0 in flipped:
        cropped_slice = img_slice[crop_rng[0] or None:, ...]
    else:
        cropped_slice = img_slice[:-crop_rng[0] or None, ...]
    if 1 in flipped:
        cropped_slice = cropped_slice[:, crop_rng[1] or None:, ...]
    else:
        cropped_slice = cropped_slice[:, :-crop_rng[1] or None, ...]
    if 2 in flipped:
        cropped_slice = cropped_slice[:, :, crop_rng[2] or None:, ...]
    else:
        cropped_slice = cropped_slice[:, :, :-crop_rng[2] or None, ...]

    return cropped_slice


def get_slice_size(max_ram, mem_growth_factor, mem_fudge_factor, slice_batch_size, num_scales=1):
    """
    Compute the size of the basic microscopy image slices fed to the Frangi filtering stage.

    Parameters
    ----------
    max_ram: float
        available RAM

    mem_growth_factor: float
        empirical memory growth factor
        of the pipeline stage (Frangi filter or ODF estimation)

    mem_fudge_factor: float
        memory fudge factor

    slice_batch_size: int
        slice batch size

    num_scales: int
        number of spatial scales analyzed in parallel

    Returns
    -------
    slice_size: float
        memory size (in megabytes) of the basic image slices
        fed to the pipeline stage
    """
    slice_size = max_ram / (slice_batch_size * mem_growth_factor * mem_fudge_factor * num_scales)

    return slice_size


def slice_channel(img, rng, channel, mosaic=False):
    """
    Slice desired channel from input image volume.

    Parameters
    ----------
    image: numpy.ndarray
        microscopy volume image

    rng: tuple (dtype=int)
        3D index ranges

    channel: int
        image channel axis

    mosaic: bool
        True for tiled reconstructions aligned using ZetaStitcher

    Returns
    -------
    image_slice: numpy.ndarray
        sliced image patch
    """
    z_rng, r_rng, c_rng = rng

    if channel is None:
        image_slice = img[z_rng, r_rng, c_rng]
    else:
        if mosaic:
            image_slice = img[z_rng, channel, r_rng, c_rng]
        else:
            image_slice = img[z_rng, r_rng, c_rng, channel]

    return image_slice
