from itertools import product

import numpy as np
import psutil
from neuCnt3d.utils import get_available_cores


def compute_axis_range(ax_iter, slice_shape, img_shape, slice_per_dim, ax, ovlp_rng=0):
    """
    Adjust image slice coordinates at boundaries.

    Parameters
    ----------
    ax_iter: tuple
        iteration counters along axes

    slice_shape: numpy.ndarray (shape=(3,), dtype=int)
        shape of the basic image slices analyzed iteratively [px]

    img_shape: numpy.ndarray (shape=(3,), dtype=int)
        total image shape [px]

    slice_per_dim: numpy.ndarray (shape=(3,), dtype=int)
        number of slices per dimension

    ax: int
        axis index

    ovlp_rng: int
        slicing range
        extension along each axis (on each side)

    Returns
    -------
    start: int
        adjusted slice start coordinate

    stop: int
        adjusted slice stop coordinate

    pad: numpy.ndarray (shape=(2,), dtype=int)
        lower and upper padding ranges
    """
    # initialize axis pad array
    pad = np.zeros(shape=(2,), dtype=int)

    # compute start and stop coordinates
    start = ax_iter[ax] * slice_shape[ax]
    stop = start + slice_shape[ax]

    # adjust start coordinate
    start -= ovlp_rng
    if start < 0:
        pad[0] = -start
        start = 0

    # adjust stop coordinate
    stop += ovlp_rng
    if stop > img_shape[ax]:
        pad[1] = stop - img_shape[ax]
        stop = img_shape[ax]

    # handle image shape residuals at boundaries
    if ax_iter[ax] == slice_per_dim[ax] - 1:
        if np.remainder(img_shape[ax], slice_shape[ax]) > 0:
            stop = img_shape[ax]
            pad[1] = ovlp_rng

    return start, stop, pad


def compute_slice_overlap(sigma_px, truncate=4):
    """
    Compute lateral image padding range
    for coping with blob detection boundary artifacts.

    Parameters
    ----------
    sigma_px: numpy.ndarray (shape=(2,), dtype=int)
        minimum and maximum spatial scales [px]

    truncate: int
        soma diameter multiplication factor
        (pad_rng = diameter * pad_factor)

    Returns
    -------
    ovlp: int
        image slice lateral overlap [px]
    """
    ovlp = int(np.ceil(2 * truncate * sigma_px[-1]) // 2)

    return ovlp


def compute_slice_range(ax_iter, slice_shape, img_shape, slice_per_dim, slice_ovlp=0):
    """
    Compute basic slice coordinates from microscopy volume image.

    Parameters
    ----------
    ax_iter: tuple
        iteration counters along axes

    slice_shape: numpy.ndarray (shape=(3,), dtype=int)
        shape of the basic image slices analyzed iteratively [px]

    img_shape: numpy.ndarray (shape=(3,), dtype=int)
        total image shape [px]

    slice_per_dim: numpy.ndarray (shape=(3,), dtype=int)
        number of slices per dimension

    slice_ovlp: int
        image slice lateral overlap [px]

    Returns
    -------
    rng: tuple
        3D slice index ranges

    pad: np.ndarray (shape=(3,2), dtype=int)
        padded boundaries
    """
    # adjust original image patch coordinates
    # and generate padding range matrix
    dims = len(ax_iter)
    start = np.zeros((dims,), dtype=int)
    stop = np.zeros((dims,), dtype=int)
    pad = np.zeros((dims, 2), dtype=int)
    slc = tuple()
    for ax in range(dims):
        start[ax], stop[ax], pad[ax] = \
            compute_axis_range(ax_iter, slice_shape, img_shape, slice_per_dim, ax=ax, ovlp_rng=slice_ovlp)
        slc += (slice(start[ax], stop[ax], 1),)

    # generate tuple of slice index ranges
    rng = np.index_exp[slc]

    return rng, pad


def config_image_slicing(sigma_px, img_shape, item_sz, px_sz, batch_sz, slice_sz):
    """
    Slicing configuration for the parallel analysis of basic chunks of the input microscopy volume.

    Parameters
    ----------
    sigma_px: numpy.ndarray (shape=(2,), dtype=int)
        minimum and maximum spatial scales [px]

    img_shape: numpy.ndarray (shape=(3,), dtype=int)
        total image shape [px]

    item_sz: int
        image item size (in bytes)

    px_sz: numpy.ndarray (shape=(3,), dtype=float)
        pixel size [μm]

    batch_sz: int
        slice batch size

    slice_sz: float
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

    slice_ovlp: int
        image slice lateral overlap [px]
    """
    # compute input patch padding range
    slice_ovlp = compute_slice_overlap(sigma_px)

    # shape of the image slices processed in parallel
    slice_shape, slice_shape_um = \
        compute_slice_shape(img_shape, item_sz, slice_sz, px_sz=px_sz, ovlp=slice_ovlp)

    # adjust output shapes according to the anisotropic pixel size correction
    px_sz_iso = px_sz[0] * np.ones(shape=px_sz.shape)
    px_rsz_ratio = np.divide(px_sz, px_sz_iso)
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
        in_rng, pad = \
            compute_slice_range((z, y, x), slice_shape, img_shape, slice_per_dim, slice_ovlp=slice_ovlp)
        if in_rng is not None:
            rng_in_lst.append(in_rng)
            pad_mat_lst.append(pad)

            # output index ranges
            out_rng, _ = \
                compute_slice_range((z, y, x), out_slice_shape, out_img_shape, slice_per_dim)
            rng_out_lst.append(out_rng)

        # invalid image slice
        else:
            slice_num -= 1

    # adjust slice batch size
    if batch_sz > slice_num:
        batch_sz = slice_num

    return rng_in_lst, rng_out_lst, pad_mat_lst, slice_shape_um, px_rsz_ratio, slice_num, batch_sz, slice_ovlp


def config_slice_batch(blob_method, sigma_num, mem_fudge_factor=1.0,
                       min_slice_sz=-1, jobs=0.8, max_ram=None):
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

    min_slice_sz: float
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
    batch_sz = min(jobs // sigma_num, num_cpu)
    if batch_sz == 0:
        batch_sz = 1

    # select memory growth factor
    if blob_method == 'log':
        mem_growth_factor = 5.0
    elif blob_method == 'dog':
        mem_growth_factor = 3.0

    # get image slice size
    slice_sz = get_slice_size(max_ram, mem_growth_factor, mem_fudge_factor, batch_sz, sigma_num)
    while slice_sz < min_slice_sz:
        batch_sz -= 1
        slice_sz = get_slice_size(max_ram, mem_growth_factor, mem_fudge_factor, batch_sz, sigma_num)

    return batch_sz, slice_sz


def compute_slice_shape(img_shape, item_sz, max_slice_sz, px_sz=None, ovlp=0):
    """
    Compute basic image chunk shape depending on its maximum size (in bytes).

    Parameters
    ----------
    img_shape: numpy.ndarray (shape=(3,), dtype=int)
        total image shape [px]

    item_sz: int
        image item size [B]

    max_slice_sz: float
        maximum memory size of the basic slices analyzed iteratively [B]

    px_sz: numpy.ndarray (shape=(3,), dtype=float)
        pixel size [μm]

    ovlp: int
        image slice lateral overlap [px]

    Returns
    -------
    slice_shape: numpy.ndarray (shape=(3,), dtype=int)
        shape of the basic image slices analyzed in parallel [px]

    slice_shape_um: numpy.ndarray (shape=(3,), dtype=float)
        shape of the basic image slices analyzed in parallel [μm]
        (if px_size is provided)
    """

    tot_ovlp = 2 * ovlp
    slice_depth = img_shape[0] + tot_ovlp
    slice_side = np.round(np.sqrt((max_slice_sz / (slice_depth * item_sz))) - tot_ovlp)
    slice_shape = np.array([slice_depth, slice_side, slice_side]).astype(int)
    slice_shape = np.min(np.stack((img_shape[:3], slice_shape)), axis=0)

    if px_sz is not None:
        slice_shape_um = np.multiply(slice_shape, px_sz)
        return slice_shape, slice_shape_um
    else:
        return slice_shape


def crop_slice(img_slice, rng, slice_ovlp):
    """
    Shrink image slice at volume boundaries, for overall shape consistency.

    Parameters
    ----------
    img_slice: numpy.ndarray (axis order: (Z,Y,X))
        image slice

    rng: tuple
        3D index range

    slice_ovlp: numpy.ndarray (shape=(3,), dtype=int)
        image slice lateral overlap [px]

    Returns
    -------
    cropped_slice: numpy.ndarray
        cropped image slice
    """

    # delete overlapping slice boundaries
    img_slice = img_slice[slice_ovlp[0]:img_slice.shape[0] - slice_ovlp[0],
                          slice_ovlp[1]:img_slice.shape[1] - slice_ovlp[1],
                          slice_ovlp[2]:img_slice.shape[2] - slice_ovlp[2]]

    # check slice shape and output index ranges
    out_slice_shape = img_slice.shape
    crop_rng = np.zeros(shape=(3,), dtype=int)
    for s in range(3):
        crop_rng[s] = out_slice_shape[s] - np.arange(rng[s].start, rng[s].stop, rng[s].step).size

    # crop image slice if required
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


def slice_channel(img, rng, ch, is_tiled=False):
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

    is_tiled: bool
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
        img_slice = img[z_rng, ch, r_rng, c_rng] if is_tiled else img[z_rng, r_rng, c_rng, ch]

    return img_slice
