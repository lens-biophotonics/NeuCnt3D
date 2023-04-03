import gc
from multiprocessing import cpu_count
from os import environ, path, remove, unlink
from shutil import rmtree
from time import perf_counter

import numpy as np
from joblib import dump, load
from skimage.filters import (threshold_li, threshold_niblack,
                             threshold_sauvola, threshold_triangle,
                             threshold_yen)


def convert_spatial_scales(scales_um, px_size):
    """
    Compute the Frangi filter scales in pixel.

    Parameters
    ----------
    scales_um: list (dtype=float)
        Frangi filter scales [μm]

    px_size: int
        isotropic pixel size [μm]

    Returns
    -------
    scales_px: numpy.ndarray (dtype=int)
        Frangi filter scales [px]
    """
    scales_um = np.asarray(scales_um)
    scales_px = scales_um / px_size

    return scales_px


def create_memory_map(file_path, shape, dtype, arr=None, mmap_mode='r+'):
    """
    Create a memory-map to an array stored in a binary file on disk.

    Parameters
    ----------
    file_path: str
        path to file object to be used as the array data buffer

    shape: tuple
        shape of the store array

    dtype:
        data-type used to interpret the file contents

    arr: numpy.ndarray
        array to be mapped

    mmap_mode: str
        file opening mode

    Returns
    -------
    mmap: NumPy memory map
        memory-mapped array
    """
    if path.exists(file_path):
        unlink(file_path)
    if arr is None:
        arr = np.zeros(tuple(shape), dtype=dtype)
    _ = dump(arr, file_path)
    mmap = load(file_path, mmap_mode=mmap_mode)
    del arr
    _ = gc.collect()

    return mmap


def get_available_cores():
    """
    Return the number of available logical cores.

    Parameters
    ----------
    None

    Returns
    -------
    num_cpu: int
        number of available cores
    """
    num_cpu = environ.pop('OMP_NUM_THREADS', default=None)
    if num_cpu is None:
        num_cpu = cpu_count()
    else:
        num_cpu = int(num_cpu)

    return num_cpu


def delete_tmp_folder(tmp_dir):
    """
    Delete temporary folder.

    Parameters
    ----------
    tmp_dir: str
        path to temporary folder to be removed

    Returns
    -------
    None
    """
    try:
        rmtree(tmp_dir)
    except OSError:
        pass


def delete_tmp_files(file_lst):
    """
    Close and remove temporary files.

    Parameters
    ----------
    file_lst: list
        list of temporary file dictionaries
        ('path': file path; 'obj': file object)

    Returns
    -------
    None
    """
    if type(file_lst) is not list:
        file_lst = [file_lst]

    for file in file_lst:
        file['obj'].close()
        remove(file['path'])


def divide_nonzero(nd_array1, nd_array2, new_value=1e-10):
    """
    Divide two arrays handling zero denominator values.

    Parameters
    ----------
    nd_array1: numpy.ndarray
        dividend array

    nd_array2: numpy.ndarray
        divisor array

    new_value: float
        substituted value

    Returns
    -------
    divided: numpy.ndarray
        divided array
    """
    denominator = np.copy(nd_array2)
    denominator[denominator == 0] = new_value
    divided = np.divide(nd_array1, denominator)

    return divided


def elapsed_time(start_time):
    """
    Compute elapsed time from input start reference.

    Parameters
    ----------
    start_time: float
        start time reference

    Returns
    -------
    total: float
        total time [s]

    mins: float
        minutes

    secs: float
        seconds
    """
    stop_time = perf_counter()
    total = stop_time - start_time
    mins = total // 60
    secs = total % 60

    return total, mins, secs


def fwhm_to_sigma(fwhm):
    """
    Compute the standard deviation of a Gaussian distribution
    from its FWHM value.

    Parameters
    ----------
    fwhm: float
        full width at half maximum

    Returns
    -------
    sigma: float
        standard deviation
    """
    sigma = np.sqrt(np.square(fwhm) / (8 * np.log(2)))

    return sigma


def get_item_bytes(data):
    """
    Retrieve data item size in bytes.

    Parameters
    ----------
    data: numpy.ndarray or HDF5 dataset
        input data

    Returns
    -------
    bytes: int
        item size in bytes
    """
    # get data type
    data_type = data.dtype

    # type byte size
    try:
        bytes = int(np.iinfo(data_type).bits / 8)
    except ValueError:
        bytes = int(np.finfo(data_type).bits / 8)

    return bytes


def add_output_prefix(img_name, min_diam_um, max_diam_um, method):
    """
    Generate the output filename including
    information on the blob detection configuration.

    Parameters
    ----------
    img_name

    min_diam_um

    max_diam_um

    method

    Returns
    -------
    out_name: str
        extended filename
    """
    pfx = method + '_minD' + str(min_diam_um) + 'um_maxD' + str(max_diam_um) + 'um_'
    out_name = pfx + 'img' + img_name

    return out_name


def mask_background(img, thresh_method='yen'):
    """
    Compute background mask.

    Parameters
    ----------
    img: numpy.ndarray (shape=(Z,Y,X))
        microscopy volume image

    thresh_method: str
        image thresholding method

    Returns
    -------
    background_mask: numpy.ndarray (shape=(Z,Y,X), dtype=bool)
        boolean background mask
    """
    # select thresholding method
    if thresh_method == 'li':
        initial_li_guess = np.mean(img[img != 0])
        thresh = threshold_li(img, initial_guess=initial_li_guess)
    elif thresh_method == 'niblack':
        thresh = threshold_niblack(img, window_size=15, k=0.2)
    elif thresh_method == 'sauvola':
        thresh = threshold_sauvola(img, window_size=15, k=0.2, r=None)
    elif thresh_method == 'triangle':
        thresh = threshold_triangle(img, nbins=256)
    elif thresh_method == 'yen':
        thresh = threshold_yen(img, nbins=256)
    else:
        raise ValueError("  Unsupported thresholding method!!!")

    # compute mask
    background_mask = img < thresh

    return background_mask


def normalize_image(img, max_out_val=255.0, dtype=np.uint8):
    """
    Normalize image data.

    Parameters
    ----------
    img: numpy.ndarray
        input image

    max_out_val: float
        maximum output value

    dtype:
        output data type

    Returns
    -------
    norm_img: numpy.ndarray
        normalized image
    """
    # get min and max values
    min_val = np.min(img)
    max_val = np.max(img)

    # normalization
    if max_val != 0:
        if max_val != min_val:
            norm_img = (((img - min_val) / (max_val - min_val)) * max_out_val).astype(dtype)
        else:
            norm_img = ((img / max_val) * max_out_val).astype(dtype)
    else:
        norm_img = img.astype(dtype)

    return norm_img


def ceil_to_multiple(number, multiple):
    """
    Round up number to the nearest multiple.

    Parameters
    ----------
    number:
        number to be rounded

    multiple:
        the input number will be rounded
        to the nearest multiple higher than this value

    Returns
    -------
    rounded:
        rounded up number
    """
    rounded = multiple * np.ceil(number / multiple)

    return rounded


def transform_axes(nd_array, flipped=None, swapped=None, expand=None):
    """
    Manipulate axes and dimensions of the input array.
    The transformation sequence is:
    axes flip >>> axes swap >>> dimensions expansion.

    Parameters
    ----------
    nd_array: numpy.ndarray
        input data array

    swapped: tuple (dtype=int)
        axes to be swapped

    flipped: tuple (dtype=int)
        axes to be flipped

    expand: int
        insert new axis at this position

    Returns
    -------
    nd_array: numpy.ndarray
        transformed data array
    """
    if flipped is not None:
        nd_array = np.flip(nd_array, axis=flipped)

    if swapped is not None:
        swap_src, swap_dest = swapped
        nd_array = np.swapaxes(nd_array, swap_src, swap_dest)

    if expand is not None:
        nd_array = np.expand_dims(nd_array, axis=expand)

    return nd_array
