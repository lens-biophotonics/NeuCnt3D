import numpy as np
from joblib import Parallel, delayed
from neuCnt3d.detection import (config_detection_scales, correct_blob_coord,
                                detect_soma, merge_parallel_blobs)
from neuCnt3d.input import get_image_info
from neuCnt3d.preprocessing import correct_anisotropy
from neuCnt3d.printing import print_analysis_info, print_results
from neuCnt3d.slicing import (config_image_slicing, config_slice_batch,
                              crop_slice, slice_channel)
from neuCnt3d.utils import create_memory_map, delete_tmp_dir


def init_napari_image(img_shape, px_rsz_ratio, tmp_dir=None, z_rng=(0, None), view=False):
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

    z_rng: int
        output z-range in [px]

    view: bool
        visualize point cloud in the Napari viewer

    Returns
    -------
    neu_img: memory-mapped file (axis order: (Z,Y,X), dtype=uint8)
        initialized neuron channel array

    z_sel: NumPy slice object
        selected z-depth range

    tmp_dir: str
        temporary file directory
    """
    # adapt output z-axis shape if required
    z_min, z_max = z_rng
    if not z_min == 0 or z_max is not None:
        if z_max is None:
            z_max = img_shape[0]
        img_shape[0] = z_max - z_min
    z_sel = slice(z_min, z_max, 1)

    # get resized output shape
    out_shape = np.ceil(np.multiply(px_rsz_ratio, img_shape)).astype(int)

    # neuron channel memory map
    if view:

        # TODO: initialize an HDF5 dataset for too large arrays (compared to RAM), as done in the Foa3D tool
        neu_img = create_memory_map(out_shape, dtype='uint8', name='tmp_neu_img', tmp=tmp_dir)

    else:
        neu_img = None

    return neu_img, z_sel, tmp_dir


def neuron_analysis(img, rng_in, rng_out, pad, method, sigma_px, sigma_num, blob_ovlp, slice_ovlp, abs_thr, rel_thr,
                    px_rsz_ratio, neu_img, z_sel, ch_neu=0, dark=False, is_tiled=False,
                    inv=-1, zero_thr=10, bg_thr=0.01, pad_mode='reflect'):
    """
    Conduct an unsupervised neuronal body enhancement and counting on a basic slice
    selected from the whole microscopy volume image.

    Parameters
    ----------
    img: numpy.ndarray or memory-mapped file (axis order: (Z,Y,X))
        soma fluorescence volume image

    rng_in: NumPy slice object
        input image range

    rng_out: NumPy slice object
        output range

    pad: numpy.ndarray (shape=(Z,Y,X))
        3D image padding range

    method: str
        blob detection approach
        (log: Laplacian of Gaussian; or dog: Difference of Gaussian)

    sigma_px: numpy.ndarray (shape=(2,), dtype=int)
        minimum and maximum spatial scales [px]

    sigma_num: int
        number of spatial scales analyzed

    blob_ovlp: float
        maximum blob percentage overlap [%]

    slice_ovlp: int
        image slice lateral overlap [px]

    abs_thr: float
        absolute blob intensity threshold

    rel_thr: float
        minimum percentage intensity of peaks in the filtered image relative to maximum [%]

    px_rsz_ratio: numpy.ndarray (shape=(3,), dtype=float)
        3D image resize ratio

    neu_img: memory-mapped file (axis order: (Z,Y,X), dtype=uint8)
        neuron channel image

    z_sel: NumPy slice object
        selected z-depth range

    ch_neu: int
        neuronal bodies channel

    dark: bool
        if True, detect dark 3D blob-like structures
        (i.e., negative contrast polarity)

    is_tiled: bool
        must be True for tiled reconstructions aligned using ZetaStitcher

    inv: int or float
        invalid value assigned to skipped
        background slices

    zero_thr: int
        zero-threshold value

    bg_thr: float
        maximum relative threshold of zero pixels

    pad_mode: str
        image padding mode

    Returns
    -------
    blobs: numpy.ndarray (shape=(N,4))
        2D array with each row representing 3 coordinate values for a 3D image,
        plus the best sigma of the Gaussian kernel which detected the blob
    """
    # slice neuron image
    neu_slice = slice_channel(img, rng_in, ch=ch_neu, is_tiled=is_tiled)

    # skip background
    tot = np.prod(neu_slice.shape)
    nonzero = np.count_nonzero(neu_slice > zero_thr)
    if nonzero / tot > bg_thr:

        # correct pixel size anisotropy (resize XY-plane)
        iso_neu_slice, rsz_pad, rsz_border, rsz_ovlp = correct_anisotropy(neu_slice, px_rsz_ratio, pad, slice_ovlp)

        # pad soma slice if required
        iso_neu_slice = np.pad(iso_neu_slice, rsz_pad, mode=pad_mode)

        # perform unsupervised neuron count analysis
        blobs = detect_soma(iso_neu_slice.astype(float), approach=method, min_sigma=sigma_px[0], max_sigma=sigma_px[1],
                            num_sigma=sigma_num, blob_ovlp=blob_ovlp, thresh_rel=rel_thr, thresh=abs_thr,
                            dark=dark, border=rsz_border)

        # add slice offset to coordinates and select z-range
        blobs = correct_blob_coord(blobs, rng_out, rsz_ovlp, z_sel)

        # crop iso_neu_slice
        iso_neu_slice_crop = crop_slice(iso_neu_slice, rng_out, rsz_ovlp)

        # fill memory-mapped output array
        if neu_img is not None:
            neu_img[rng_out] = iso_neu_slice_crop[z_sel, ...].astype(np.uint8)

        return blobs

    else:
        return inv * np.ones((4,))


def parallel_neuron_detection_on_slices(img, px_sz, method, diam_um, blob_ovlp, rel_glob_thr, rel_loc_thr,
                                        ch_neu=0, dark=False, z_rng=(0, None), is_tiled=False, max_ram=None,
                                        jobs=0.8, backend='threading', tmp_dir=None, inv=-1, verbose=10,
                                        view=False):
    """
    Perform unsupervised neuronal body enhancement and counting on batches of
    basic microscopy image slices using parallel processes or threads.

    Parameters
    ----------
    img: numpy.ndarray or memory-mapped file (axis order: (Z,Y,X))
        microscopy volume image

    px_sz: numpy.ndarray (shape=(3,), dtype=float)
        pixel size [μm]

    method: str
        blob detection approach
        (log: Laplacian of Gaussian; or dog: Difference of Gaussian)

    diam_um: tuple
        soma diameter (minimum, maximum, step size) [μm]

    blob_ovlp: float
        maximum blob overlap percentage [%]

    rel_glob_thr: float
        minimum intensity of peaks in the filtered image
        relative to global maximum

    rel_thr: float
        minimum intensity of peaks in the filtered image
        relative to local slice maximum

    ch_neu: int
        neuronal bodies channel

    dark: bool
        if True, detect black 3D blob-like structures
        (i.e., negative contrast polarity)

    z_rng: int
        output z-range in [px]

    is_tiled: bool
        True for tiled reconstructions aligned using ZetaStitcher

    max_ram: float
        maximum RAM available to the blob detection stage [B]

    jobs: int
        number of parallel jobs

    backend: str
        backend module employed by joblib.Parallel

    tmp_dir: str
        temporary file directory

    inv: int or float
        invalid value assigned to skipped
        background slices

    verbose: int
        verbosity level

    view: bool
        visualize soma point cloud in the Napari viewer

    Returns
    -------
    blobs: numpy.ndarray (shape=(N,4))
        2D array with each row representing 3 coordinate values for a 3D image,
        plus the best sigma of the Gaussian kernel which detected the blob

    neu_img: memory-mapped file (axis order: (Z,Y,X), dtype=uint8)
        resized soma channel image with isotropic pixel size
    """
    # get info on the input volume image
    img_shape, img_shape_um, img_item_size, img_max, ch_neu = get_image_info(img, px_sz, ch_neu, is_tiled=is_tiled)

    # compute the scale range adopted by the blob detection stage
    sigma_px, sigma_num = config_detection_scales(diam_um, px_sz)

    # configure batch of basic image slices to be analyzed in parallel
    batch_sz, max_slice_sz = config_slice_batch(method, sigma_num, max_ram=max_ram, jobs=jobs)

    # configure the overall microscopy volume slicing
    rng_in_lst, rng_out_lst, pad_mat_lst, slice_shape_um, px_rsz_ratio, slice_num, batch_sz, slice_ovlp = \
        config_image_slicing(sigma_px, img_shape, img_item_size, px_sz, batch_sz, max_slice_sz)

    # initialize resized neuron channel image
    neu_img, z_sel, tmp_dir = init_napari_image(img_shape, px_rsz_ratio, tmp_dir=tmp_dir, z_rng=z_rng, view=view)

    # parallel unsupervised neuron localization and counting on microscopy image sub-volumes
    # adapt blob thresholds
    if rel_glob_thr is not None:
        abs_thr = rel_glob_thr * img_max

    # print analysis configuration
    print_analysis_info(method, diam_um, sigma_num, blob_ovlp, abs_thr, rel_loc_thr,
                        img_shape_um, slice_shape_um, slice_num, px_sz, img_item_size)

    with Parallel(n_jobs=batch_sz, backend=backend, verbose=verbose, max_nbytes=None) as parallel:
        par_blobs = parallel(
            delayed(neuron_analysis)(
                img, rng_in_lst[i], rng_out_lst[i], pad_mat_lst[i], method, sigma_px, sigma_num, blob_ovlp,
                slice_ovlp, abs_thr, rel_loc_thr, px_rsz_ratio, neu_img, z_sel,
                ch_neu=ch_neu, dark=dark, is_tiled=is_tiled, inv=inv)
            for i in range(slice_num))

    # concatenate parallel results
    blobs = merge_parallel_blobs(par_blobs, inv=inv)

    # delete temporary folder
    if tmp_dir is not None:
        delete_tmp_dir(tmp_dir)

    # print results to terminal
    print_results(blobs, img_shape)

    return blobs, neu_img
