from itertools import product

import numpy as np
import psutil
from neuCnt3d.utils import get_available_cores


def adjust_slice_coord(axis_iter, pad_rng, slice_shape, img_shape, axis):
    """
    Adjust image slice coordinates at boundaries.

    Parameters
    ----------
    axis_iter: int
        iteration counter along axis

    pad_rng: int
        patch padding range [px]

    slice_shape: numpy.ndarray (shape=(3,), dtype=int)
        shape of the basic image slices analyzed iteratively [px]

    img_shape: numpy.ndarray (shape=(3,), dtype=int)
        total image shape [px]

    axis: int
        axis index

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

    return start, stop, axis_pad_array


def compute_slice_padding(sigma_px, px_size, pad_factor=1.0):
    """
    Compute lateral image padding range
    for coping with blob detection boundary artifacts.

    Parameters
    ----------
    sigma_px: numpy.ndarray (shape=(2,), dtype=int)
        minimum and maximum spatial scales [px]

    px_size: numpy.ndarray (shape=(3,), dtype=float)
        pixel size [μm]

    pad_factor: float
        soma diameter multiplication factor
        (pad_rng = diameter * pad_factor)

    Returns
    -------
    pad_rng: int
        slice padding range [px]
    """
    pad_rng = int(np.ceil(2 * np.sqrt(3) * sigma_px[1] * pad_factor / px_size[-1]))

    return pad_rng


def compute_slice_range(z, y, x, slice_shape, img_shape, pad_rng=0):
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

    pad_rng: int
        slice padding range

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
        adjust_slice_coord(z, pad_rng, slice_shape, img_shape, axis=0)
    y_start, y_stop, pad_mat[1, :] = \
        adjust_slice_coord(y, pad_rng, slice_shape, img_shape, axis=1)
    x_start, x_stop, pad_mat[2, :] = \
        adjust_slice_coord(x, pad_rng, slice_shape, img_shape, axis=2)

    # generate index ranges
    z_rng = slice(z_start, z_stop, 1)
    y_rng = slice(y_start, y_stop, 1)
    x_rng = slice(x_start, x_stop, 1)
    rng = np.index_exp[z_rng, y_rng, x_rng]
    for r in rng:
        if r.start is None:
            return None

    return rng, pad_mat


def config_image_slicing(sigma_px, img_shape, item_size, px_size, batch_size, slice_size):
    """
    Slicing configuration for the parallel analysis of basic chunks of the input microscopy volume.

    Parameters
    ----------
    sigma_px: numpy.ndarray (shape=(2,), dtype=int)
        minimum and maximum spatial scales [px]

    img_shape: numpy.ndarray (shape=(3,), dtype=int)
        total image shape [px]

    item_size: int
        image item size (in bytes)

    px_size: numpy.ndarray (shape=(3,), dtype=float)
        pixel size [μm]

    batch_size: int
        slice batch size

    slice_size: float
        maximum memory size (in megabytes) of the basic image slices
        analyzed in parallel

    Returns
    -------
    rng_in_lst: list
        list of analyzed fiber channel slice ranges

    rng_out_lst: list
        list of output slice ranges

    pad_mat_lst: list
        list of slice padding ranges

    slice_shape_um: numpy.ndarray (shape=(3,), dtype=float)
        shape of the basic image slices analyzed iteratively [μm]

    px_rsz_ratio: numpy.ndarray (shape=(3,), dtype=float)
        3D image resize ratio

    slice_num: int
        total number of analyzed image slices

    batch_size: int
        adjusted slice batch size
    """
    # compute input patch padding range
    border = compute_slice_padding(sigma_px, px_size)

    # shape of the image slices processed in parallel
    slice_shape, slice_shape_um = \
        compute_slice_shape(img_shape, item_size, slice_size, px_sz=px_size, pad_rng=border)

    # adjust output shapes according to the anisotropic pixel size correction
    px_size_iso = px_size[0] * np.ones(shape=px_size.shape)
    px_rsz_ratio = np.divide(px_size, px_size_iso)
    out_img_shape = np.ceil(np.multiply(px_rsz_ratio, img_shape)).astype(int)
    out_slice_shape = np.ceil(np.multiply(px_rsz_ratio, slice_shape)).astype(int)

    # iteratively define the input/output 3D slice ranges
    slice_per_dim = np.ceil(np.divide(img_shape, slice_shape)).astype(int)
    slice_num = int(np.prod(slice_per_dim))

    # initialize empty range lists
    rng_in_lst = list()
    rng_out_lst = list()
    pad_mat_lst = list()
    for z, y, x in product(range(slice_per_dim[0]), range(slice_per_dim[1]), range(slice_per_dim[2])):

        # index ranges of analyzed neuron image slices (with padding)
        rng_in, pad_mat = \
            compute_slice_range(z, y, x, slice_shape, img_shape, pad_rng=border)
        if rng_in is not None:
            rng_in_lst.append(rng_in)
            pad_mat_lst.append(pad_mat)

            # output index ranges
            rng_out, _ = \
                compute_slice_range(z, y, x, out_slice_shape, out_img_shape)
            rng_out_lst.append(rng_out)

        # invalid image slice
        else:
            slice_num -= 1

    # adjust slice batch size
    if batch_size > slice_num:
        batch_size = slice_num

    return rng_in_lst, rng_out_lst, pad_mat_lst, slice_shape_um, px_rsz_ratio, slice_num, batch_size


def config_slice_batch(blob_method, sigma_num, mem_fudge_factor=1.0,
                       min_slice_size=-1, jobs=0.8, max_ram=None):
    """
    Compute the size of the basic microscopy image slices
    and the size of the slice batches analyzed in parallel.

    Parameters
    ----------
    blob_method: str
        blob detection approach
        (Laplacian of Gaussian or Difference of Gaussian)

    sigma_num: int
        number of spatial scales

    mem_fudge_factor: float
        memory fudge factor

    min_slice_size: float
        minimum slice size [B]

    jobs: int
        number of parallel jobs

    max_ram: float
        maximum RAM available to the blob detection stage [B]

    Returns
    -------
    batch_sz: int
        slice batch size

    slice_sz: float
        memory size of the basic image slices
        fed to the blob detection function [B]
    """
    # maximum RAM not provided: use all
    if max_ram is None:
        max_ram = psutil.virtual_memory()[1]

    # number of logical cores
    num_cpu = get_available_cores()

    # initialize slice batch size
    batch_sz = min(jobs, num_cpu)

    # select memory growth factor
    if blob_method == 'log':
        mem_growth_factor = 5.0
    elif blob_method == 'dog':
        mem_growth_factor = 1.0

    # get image slice size
    slice_sz = get_slice_size(max_ram, mem_growth_factor, mem_fudge_factor, batch_sz, sigma_num)
    while slice_sz < min_slice_size:
        batch_sz -= 1
        slice_sz = get_slice_size(max_ram, mem_growth_factor, mem_fudge_factor, batch_sz, sigma_num)

    return batch_sz, slice_sz


def compute_slice_shape(img_shape, item_sz, max_slice_sz, px_sz=None, pad_rng=0):
    """
    Compute basic image chunk shape depending on its maximum size (in bytes).

    Parameters
    ----------
    img_shape: numpy.ndarray (shape=(3,), dtype=int)
        total image shape [px]

    item_sz: int
        image item size [B]

    max_slice_size: float
        maximum memory size of the basic slices analyzed iteratively [B]

    px_sz: numpy.ndarray (shape=(3,), dtype=float)
        pixel size [μm]

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
    slice_depth = img_shape[0]
    slice_side = np.round(np.sqrt((max_slice_sz / (slice_depth * item_sz))) - 2 * pad_rng)
    slice_shape = np.array([slice_depth, slice_side, slice_side]).astype(int)
    slice_shape = np.min(np.stack((img_shape[:3], slice_shape)), axis=0)

    if px_sz is not None:
        slice_shape_um = np.multiply(slice_shape, px_sz)
        return slice_shape, slice_shape_um
    else:
        return slice_shape


def crop_slice(img_slice, rng):
    """
    Shrink image slice at volume boundaries, for overall shape consistency.

    Parameters
    ----------
    img_slice: numpy.ndarray (axis order: (Z,Y,X))
        image slice

    rng: tuple
        3D index range

    Returns
    -------
    cropped_slice: numpy.ndarray
        cropped image slice
    """
    # check slice shape and output index ranges
    out_slice_shape = img_slice.shape
    crop_rng = np.zeros(shape=(3,), dtype=int)
    for s in range(3):
        rsz = np.arange(rng[s].start, rng[s].stop, rng[s].step).size
        crop_rng[s] = out_slice_shape[s] - rsz

    # crop slice if required
    cropped_slice = img_slice[:-crop_rng[0] or None, ...]
    cropped_slice = cropped_slice[:, :-crop_rng[1] or None, :]
    cropped_slice = cropped_slice[:, :, :-crop_rng[2] or None]

    return cropped_slice


def get_slice_size(max_ram, mem_growth_factor, mem_fudge_factor, batch_sz, num_scales):
    """
    Compute the size of the basic microscopy image slices fed to the blob detection function.

    Parameters
    ----------
    max_ram: float
        available RAM [B]

    mem_growth_factor: float
        empirical memory growth factor
        of the blob detection stage

    mem_fudge_factor: float
        memory fudge factor

    batch_sz: int
        slice batch size

    num_scales: int
        number of spatial scales
        of interest

    Returns
    -------
    slice_sz: float
        memory size of the basic image slices
        fed to the blob detector [B]
    """
    slice_sz = max_ram / (batch_sz * mem_growth_factor * mem_fudge_factor * num_scales)

    return slice_sz


def slice_channel(img, rng, ch, mosaic=False):
    """
    Slice desired channel from input image volume.

    Parameters
    ----------
    img: numpy.ndarray or memory-mapped file (axis order: (Z,Y,X))
        microscopy volume image

    rng: tuple (dtype=int)
        3D index ranges

    ch: int
        neuronal body channel

    mosaic: bool
        True for tiled reconstructions aligned using ZetaStitcher

    Returns
    -------
    img_slice: numpy.ndarray (axis order: (Z,Y,X))
        sliced image patch
    """
    z_rng, r_rng, c_rng = rng

    if ch is None:
        img_slice = img[z_rng, r_rng, c_rng]
    else:
        if mosaic:
            img_slice = img[z_rng, ch, r_rng, c_rng]
        else:
            img_slice = img[z_rng, r_rng, c_rng, ch]

    return img_slice
