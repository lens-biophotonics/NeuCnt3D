from os import path

import numpy as np
from joblib import Parallel, delayed
from neuCnt3D.detection import (config_detection_scales, correct_blob_coord,
                                detect_soma)
from neuCnt3D.input import get_image_info
from neuCnt3D.preprocessing import correct_image_anisotropy
from neuCnt3D.printing import print_analysis_info, print_results
from neuCnt3D.slicing import (config_image_slicing, config_slice_batch,
                              crop_slice, slice_channel)
from neuCnt3D.utils import create_memory_map, delete_tmp_dir


def init_napari_volume(img_shape, px_rsz_ratio, tmp_dir, z_min=0, z_max=None):
    """
    Initialize the memory-mapped image for the Napari viewer.

    Parameters
    ----------
    img_shape: numpy.ndarray (shape=(3,), dtype=int)
        volume image shape [px]

    px_rsz_ratio: numpy.ndarray (shape=(3,), dtype=float)
        3D resize ratio

    tmp_dir: str
        temporary file directory

    z_min: int
        minimum output z-depth in [px]

    z_max: int
        maximum output z-depth in [px]

    Returns
    -------
    neu_img: NumPy memory-map object (shape=(Z,Y,X), dtype=uint8)
        initialized neuron channel array

    z_sel: NumPy slice object
        selected z-depth range
    """
    # adapt output z-axis shape if required
    if not z_min == 0 or z_max is not None:
        if z_max is None:
            z_max = img_shape[0]
        img_shape[0] = z_max - z_min
    z_sel = slice(z_min, z_max, 1)

    # get resized output shape
    out_shape = np.ceil(np.multiply(px_rsz_ratio, img_shape)).astype(int)

    # neuron channel memory map
    neu_tmp_path = path.join(tmp_dir, 'tmp_neu_img.mmap')
    neu_img = create_memory_map(neu_tmp_path, out_shape, dtype='uint8')

    return neu_img, z_sel


def neuron_analysis(img, rng_in, rng_out, pad, method, sigma_px, sigma_num, overlap, rel_thresh, px_rsz_ratio,
                    neu_img, z_sel, ch_neu=0, dark=False, mosaic=False):
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

    method:

    sigma_px: numpy.ndarray (shape=(2,), dtype=int)
        minimum and maximum spatial scales [px]

    sigma_num

    overlap

    rel_thresh

    px_rsz_ratio: numpy.ndarray (shape=(3,), dtype=float)
        3D image resize ratio

    neu_img: NumPy memory map (shape=(Z,Y,X), dtype=uint8)
        neuron channel image

    z_sel: NumPy slice object
        selected z-depth range

    ch_neu: int
        neuronal bodies channel

    dark: bool
        if True, enhance dark 3D blob-like structures
        (i.e., negative contrast polarity)

    mosaic: bool
        must be True for tiled reconstructions aligned using ZetaStitcher

    Returns
    -------
    None
    """
    # slice neuron image
    neu_slice = slice_channel(img, rng_in, channel=ch_neu, mosaic=mosaic)

    # skip background
    if np.max(neu_slice) != 0:

        # correct pixel size anisotropy (resize XY plane)
        iso_neu_slice, iso_neu_slice_crop, rsz_pad, rsz_border = correct_image_anisotropy(neu_slice, px_rsz_ratio, pad)

        # perform unsupervised neuron count analysis
        blobs = detect_soma(iso_neu_slice, method=method, min_sigma=sigma_px[0], max_sigma=sigma_px[1],
                            num_sigma=sigma_num, overlap=overlap, threshold_rel=rel_thresh,
                            dark=dark, border=rsz_border)

        # add slice offset to coordinates and select z-range
        blobs = correct_blob_coord(blobs, rng_out, rsz_pad, z_sel)

        # crop iso_neu_slice
        iso_neu_slice_crop = crop_slice(iso_neu_slice_crop, rng_out)

        # fill memory-mapped output array
        neu_img[rng_out] = iso_neu_slice_crop[z_sel, ...].astype(np.uint8)

        return blobs

    else:
        return []


def parallel_neuron_detection_on_slices(img, px_size, method, diam_um, overlap, rel_thresh, tmp_dir,
                                        ch_neu=0, z_min=0, z_max=None, mosaic=False, max_ram_mb=None, jobs_to_cores=0.8,
                                        dark=False, backend='threading'):
    """
    Perform unsupervised neuronal body enhancement and counting on batches of
    basic microscopy image slices using parallel threads.

    Parameters
    ----------
    img: NumPy memory-map object (shape=(Z,Y,X))
        microscopy volume image

    px_size: numpy.ndarray (shape=(3,), dtype=float)
        pixel size [μm]

    method

    diam_um

    overlap

    rel_thresh

    tmp_dir: str
        temporary file directory

    ch_neu: int
        neuronal bodies channel

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

    dark: bool
        if True, enhance black 3D blob-like structures
        (i.e., negative contrast polarity)

    backend: str
        backend module employed by joblib.Parallel

    Returns
    -------
    None
    """
    # get info on the input volume image
    img_shape, img_shape_um, img_item_size, ch_neu = get_image_info(img, px_size, ch_neu, mosaic=mosaic)

    # compute the scale range adopted by the blob detection stage
    sigma_px, sigma_num = config_detection_scales(diam_um, px_size)

    # configure batch of basic image slices to be analyzed in parallel
    batch_size, max_slice_size = config_slice_batch(sigma_num, max_ram_mb=max_ram_mb, jobs_to_cores=jobs_to_cores)

    # configure the overall microscopy volume slicing
    rng_in_lst, rng_out_lst, pad_mat_lst, slice_shape_um, px_rsz_ratio, slice_num, batch_size = \
        config_image_slicing(sigma_px, img_shape, img_item_size, px_size, batch_size, max_slice_size)

    # initialize resized neuron channel image
    neu_img, z_sel = init_napari_volume(img_shape, px_rsz_ratio, tmp_dir, z_min=z_min, z_max=z_max)

    # print analysis configuration
    print_analysis_info(method, diam_um, sigma_num, img_shape_um, slice_shape_um, slice_num, px_size, img_item_size)

    # parallel unsupervised neuron enhancement, segmentation and counting of microscopy image sub-volumes
    with Parallel(n_jobs=batch_size, backend=backend, verbose=100, max_nbytes=None) as parallel:
        par_blobs = parallel(
            delayed(neuron_analysis)(
                img, rng_in_lst[i], rng_out_lst[i], pad_mat_lst[i], method, sigma_px, sigma_num, overlap, rel_thresh,
                px_rsz_ratio, neu_img, z_sel, ch_neu=ch_neu, dark=dark, mosaic=mosaic)
            for i in range(slice_num))

    # concatenate parallel results
    blobs = np.vstack(par_blobs)

    # delete temporary folder
    delete_tmp_dir(tmp_dir)

    # print results to terminal
    print_results(blobs, img_shape)

    return blobs, neu_img
