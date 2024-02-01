import gc
import tempfile
from multiprocessing import cpu_count
from os import environ, path, unlink
from shutil import rmtree

import numpy as np
from joblib import dump, load


def create_memory_map(shape, dtype, name='tmp', tmp=None, arr=None, mmap_mode='r+'):
    """
    Create a memory-map to an array stored in a binary file on disk.

    Parameters
    ----------
    shape: tuple
        shape of the store array

    dtype:
        data-type used to interpret the file contents

    name: str
        optional temporary filename

    tmp: str
        temporary file directory

    arr: numpy.ndarray
        array to be mapped

    mmap_mode: str
        file opening mode

    Returns
    -------
    mmap: NumPy memory map
        memory-mapped array
    """
    if tmp is None:
        tmp = tempfile.mkdtemp()
    mmap_path = path.join(tmp, name + '.mmap')

    if path.exists(mmap_path):
        unlink(mmap_path)
    if arr is None:
        arr = np.zeros(tuple(shape), dtype=dtype)
    _ = dump(arr, mmap_path)
    mmap = load(mmap_path, mmap_mode=mmap_mode)
    del arr
    _ = gc.collect()

    return mmap


def get_available_cores():
    """
    Return the number of available logical cores.

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
    item_bytes: int
        item size in bytes
    """
    # get data type
    data_type = data.dtype

    # type byte size
    try:
        item_bytes = int(np.iinfo(data_type).bits / 8)
    except ValueError:
        item_bytes = int(np.finfo(data_type).bits / 8)

    return item_bytes


def add_output_prefix(img_name, min_diam_um, max_diam_um, method, rel_loc_thr, rel_glob_thr):
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
        (log: Laplacian of Gaussian; or dog: Difference of Gaussian)

    rel_loc_thr: float
        minimum intensity of peaks in the filtered image
        relative to local slice maximum

    rel_glob_thr: float
        minimum intensity of peaks in the filtered image
        relative to global maximum

    Returns
    -------
    out_name: str
        extended filename
    """
    out_name = '{}_minD{}um_maxD{}um_globthr{}_locthr{}_img{}' \
        .format(method, min_diam_um, max_diam_um, rel_glob_thr, rel_loc_thr, img_name)

    return out_name
