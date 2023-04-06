import gc
from multiprocessing import cpu_count
from os import environ, path, unlink
from shutil import rmtree

import numpy as np
from joblib import dump, load


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


def delete_tmp_dir(tmp_dir):
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


def get_item_bytes(data):
    """
    Retrieve data item size in bytes.

    Parameters
    ----------
    data: numpy.ndarray
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
    img_name: str
        name of the input microscopy volume image

    min_diam_um: float
        minimum soma diameter of interest [μm]

    max_diam_um: float
        maximum soma diameter of interest [μm]

    method: str
        blob detection approach
        (Laplacian of Gaussian or Difference of Gaussian)

    Returns
    -------
    out_name: str
        extended filename
    """
    pfx = method + '_minD' + str(min_diam_um) + 'um_maxD' + str(max_diam_um) + 'um_'
    out_name = pfx + 'img' + img_name

    return out_name
