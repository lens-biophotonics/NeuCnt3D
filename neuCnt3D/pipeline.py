from os import path
from time import perf_counter

import numpy as np
from joblib import Parallel, delayed
from neuCnt3D.detection import detect_soma
from neuCnt3D.input import get_image_info
from neuCnt3D.output import save_array
from neuCnt3D.printing import print_analysis_info, print_analysis_time
from neuCnt3D.slicing import (config_image_slicing, config_slice_batch,
                              crop_slice, slice_channel)
from neuCnt3D.utils import convert_spatial_scales, create_memory_map


def init_output_volume(img_shape, slice_shape, tmp_dir, img_name, z_min=0, z_max=None):
    """
    Initialize the output datasets of the Frangi filtering stage.

    Parameters
    ----------
    img_shape: numpy.ndarray (shape=(3,), dtype=int)
        volume image shape [px]

    slice_shape: numpy.ndarray (shape=(3,), dtype=int)
        shape of the basic image slices analyzed iteratively [px]

    resize_ratio: numpy.ndarray (shape=(3,), dtype=float)
        3D resize ratio

    tmp_dir: str
        temporary file directory

    img_name: str
        name of the input volume image

    z_min: int
        minimum output z-depth in [px]

    z_max: int
        maximum output z-depth in [px]

    Returns
    -------
    neuron_msk: NumPy memory-map object (shape=(Z,Y,X), dtype=uint8)
        initialized neuron mask image

    z_sel: NumPy slice object
        selected z-depth range
    """
    # shape copies
    img_shape = img_shape.copy()
    slice_shape = slice_shape.copy()

    # adapt output z-axis shape if required
    if z_min != 0 or z_max is not None:
        if z_max is None:
            z_max = slice_shape[0]
        img_shape[0] = z_max - z_min
    z_sel = slice(z_min, z_max, 1)

    # neuron mask memory map
    neuron_msk_path = path.join(tmp_dir, 'neuron_msk_' + img_name + '.mmap')
    neuron_msk = create_memory_map(neuron_msk_path, img_shape, dtype='uint8')

    return neuron_msk, z_sel


def neuron_analysis(img, rng_in, ch_neuron=0, dark=False, mosaic=False):
    """
    Conduct a Frangi-based fiber orientation analysis on basic slices selected from the whole microscopy volume image.

    Parameters
    ----------
    img: numpy.ndarray (shape=(Z,Y,X))
        fiber fluorescence volume image

    rng_in: NumPy slice object
        input image range

    rng_out: NumPy slice object
        output range

    pad: numpy.ndarray (shape=(Z,Y,X))
        3D image padding range

    smooth_sigma: numpy.ndarray (shape=(3,), dtype=int)
        3D standard deviation of the low-pass Gaussian filter [px]
        (applied to the XY plane)

    scales_px: numpy.ndarray (dtype=int)
        spatial scales [px]

    px_rsz_ratio: numpy.ndarray (shape=(3,), dtype=float)
        3D image resize ratio

    z_sel: NumPy slice object
        selected z-depth range

    neuron_msk: NumPy memory map (shape=(Z,Y,X), dtype=uint8)
        neuron mask image

    ch_neuron: int
        neuronal bodies channel

    dark: bool
        if True, enhance black 3D tubular structures
        (i.e., negative contrast polarity)

    mosaic: bool
        must be True for tiled reconstructions aligned using ZetaStitcher

    Returns
    -------
    None
    """
    # slice neuron image
    neuron_slice = slice_channel(img, rng_in, channel=ch_neuron, mosaic=mosaic)

    # skip background slice
    if np.max(neuron_slice) != 0:

        # TODO!
        blobs = detect_soma(neuron_slice, dark=dark)

        return blobs

    else:
        return []


def parallel_neuron_count_on_slices(img, px_size, save_dir, tmp_dir, img_name, sigma_um,
                                    ch_neuron=0, dark=False, z_min=0, z_max=None, mosaic=False,
                                    max_ram_mb=None, jobs_to_cores=0.8, backend='threading'):
    """
    Perform unsupervised neuronal body enhancement and counting to basic TPFM image slices using parallel threads.

    Parameters
    ----------
    img: NumPy memory-map object (shape=(Z,Y,X))
        microscopy volume image

    px_size: numpy.ndarray (shape=(3,), dtype=float)
        pixel size [μm]

    px_size_iso: numpy.ndarray (shape=(3,), dtype=float)
        adjusted isotropic pixel size [μm]

    smooth_sigma: numpy.ndarray (shape=(3,), dtype=int)
        3D standard deviation of the low-pass Gaussian filter [px]
        (applied to the XY plane)

    save_dir: str
        saving directory string path

    tmp_dir: str
        temporary file directory

    img_name: str
        name of the input volume image

    sigma_um: list (dtype=float)
        spatial scales of interest in [μm]

    ch_neuron: int
        neuronal bodies channel

    dark: bool
        if True, enhance black 3D tubular structures
        (i.e., negative contrast polarity)

    z_min: int
        minimum output z-depth in [px]

    z_max: int
        maximum output z-depth in [px]

    mosaic: bool
        must be True for tiled reconstructions aligned using ZetaStitcher

    max_ram_mb: float
        maximum RAM available to the Frangi filtering stage [MB]

    jobs_to_cores: float
        max number of jobs relative to the available CPU cores
        (default: 80%)

    backend: str
        backend module employed by joblib.Parallel

    Returns
    -------
    None
    """
    # get info on the input volume image
    img_shape, img_shape_um, img_item_size, ch_neuron = get_image_info(img, px_size, ch_neuron, mosaic=mosaic)

    # configure batch of basic image slices to be analyzed in parallel
    batch_size, max_slice_size = config_slice_batch(sigma_um, max_ram_mb=max_ram_mb, jobs_to_cores=jobs_to_cores)

    # configure the overall microscopy volume slicing
    rng_in_lst, in_slice_shape, in_slice_shape_um, tot_slice_num, batch_size = \
        config_image_slicing(img_shape, img_item_size, px_size, batch_size, slice_size=max_slice_size)

    # print analysis configuration
    print_analysis_info(sigma_um, img_shape_um, in_slice_shape_um, tot_slice_num, px_size, img_item_size)

    # parallel unsupervised neuron enhancement, segmentation and counting of microscopy image sub-volumes
    start_time = perf_counter()
    with Parallel(n_jobs=batch_size, backend=backend, verbose=100, max_nbytes=None) as parallel:
        par_blobs = parallel(
            delayed(neuron_analysis)(
                img, rng_in_lst[i], ch_neuron=ch_neuron, dark=dark, mosaic=mosaic) for i in range(tot_slice_num))

    # concatenate parallel results
    blobs = [item for sublist in par_blobs for item in sublist]

    # print analysis time
    print_analysis_time(start_time)

    return blobs


def save_output_volume(neuron_msk, px_size, save_dir, img_name):
    """
    Save the output arrays of the neuron analysis stage to TIF files.

    Parameters
    ----------
    neuron_msk: NumPy memory map (shape=(Z,Y,X), dtype=uint8)
        neuron mask image

    px_size: numpy.ndarray (shape=(3,), dtype=float)
        pixel size (Z,Y,X) [μm]

    save_dir: str
        saving directory string path

    img_name: str
        name of the input microscopy volume image

    Returns
    -------
    None
    """
    # save neuron channel volumes to TIF
    save_array('neuron_msk_' + img_name, save_dir, neuron_msk, px_size)
