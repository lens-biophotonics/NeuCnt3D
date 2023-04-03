import os
from platform import system

import numpy as np

from neuCnt3D.utils import elapsed_time

# adjust ANSI escape sequence
# decoding to Windows OS
if system == 'Windows':
    os.system("color")


def color_text(r, g, b, text):
    """
    Get colored text string.

    Parameters
    ----------
    r: int
        red channel value

    g: int
        green channel value

    b: int
        blue channel value

    text: str
        text string

    Returns
    -------
    clr_text: str
        colored text
    """
    clr_text = "\033[38;2;{};{};{}m{} \033[38;2;255;255;255m".format(r, g, b, text)

    return clr_text


def print_analysis_info(scales_um, image_shape_um, in_slice_shape_um, tot_slice_num, px_size, image_item_size):
    """
    Print analysis configuration.

    Parameters
    ----------
    scales_um: list (dtype=float)
        analyzed spatial scales [μm]

    image_shape_um: numpy.ndarray (shape=(3,), dtype=float)
        volume image shape [μm]

    in_slice_shape_um: numpy.ndarray (shape=(3,), dtype=float)
        shape of the analyzed image slices [μm]

    tot_slice_num: int
        total number of analyzed image slices

    px_size: numpy.ndarray (shape=(3,), dtype=float)
        pixel size [μm]

    image_item_size: int
        image item size (in bytes)

    Returns
    -------
    None
    """
    print(color_text(0, 191, 255, "\n3D Neuronal Body Enhancement"))
    print("\nScales     [\u03BCm]: {}".format(np.asarray(scales_um)))
    print_slicing_info(image_shape_um, in_slice_shape_um, tot_slice_num, px_size, image_item_size)


def print_analysis_time(start_time):
    """
    Print volume image analysis time.

    Parameters
    ----------
    start_time: float
        analysis start time

    Returns
    -------
    None
    """
    _, mins, secs = elapsed_time(start_time)
    print("\nVolume image analyzed in: {0} min {1:3.1f} s\n".format(mins, secs))


def print_import_time(start_time):
    """
    Print volume image import time.

    Parameters
    ----------
    start_time: float
        import start time

    Returns
    -------
    None
    """
    _, mins, secs = elapsed_time(start_time)
    print("Volume image loaded in: {0} min {1:3.1f} s".format(mins, secs))


def print_pipeline_heading():
    """
    Print NeuCnt3D pipeline heading.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    print(color_text(0, 250, 154, "\n3D Unsupervised Neuron Segmentation and Counting"))


def print_prepro_heading():
    """
    Print preprocessing heading.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    print(color_text(0, 191, 255, "\n\nMicroscopy Volume Image Preprocessing"), end='\r')


def print_resolution(px_size, psf_fwhm):
    """
    Print pixel and optical resolution of the microscopy system.

    Parameters
    ----------
    px_size: numpy.ndarray (shape=(3,), dtype=float)
        pixel size [μm]

    psf_fwhm: numpy.ndarray (shape=(3,), dtype=float)
        PSF 3D FWHM [μm]

    Returns
    -------
    None
    """
    print("Pixel size           [μm]: ({0:.3f}, {1:.3f}, {2:.3f})".format(px_size[0], px_size[1], px_size[2]))
    print("PSF FWHM             [μm]: ({0:.3f}, {1:.3f}, {2:.3f})".format(psf_fwhm[0], psf_fwhm[1], psf_fwhm[2]))


def print_slicing_info(image_shape_um, slice_shape_um, tot_slice_num, px_size, image_item_size):
    """
    Print information on the slicing of the basic image sub-volumes
    iteratively processed by the Foa3D pipeline.

    Parameters
    ----------
    image_shape_um: numpy.ndarray (shape=(3,), dtype=float)
        volume image shape [μm]

    slice_shape_um: numpy.ndarray (shape=(3,), dtype=float)
        shape of the analyzed image slices [μm]

    tot_slice_num: int
        total number of analyzed image slices

    px_size: numpy.ndarray (shape=(3,), dtype=float)
        pixel size [μm]

    image_item_size: int
        image item size (in bytes)

    Returns
    -------
    None
    """
    # adjust slice shape
    if np.any(image_shape_um < slice_shape_um):
        slice_shape_um = image_shape_um

    # get image memory size
    image_size = image_item_size * np.prod(np.divide(image_shape_um, px_size))

    # get slice memory size
    max_slice_size = image_item_size * np.prod(np.divide(slice_shape_um, px_size))

    # print info
    print("\n                              Z      Y      X")
    print("Total image shape    [μm]: ({0:.1f}, {1:.1f}, {2:.1f})"
          .format(image_shape_um[0], image_shape_um[1], image_shape_um[2]))
    print("Total image size     [MB]: {0}\n"
          .format(np.ceil(image_size / 1024**2).astype(int)))
    print("Basic slice shape    [μm]: ({0:.1f}, {1:.1f}, {2:.1f})"
          .format(slice_shape_um[0], slice_shape_um[1], slice_shape_um[2]))
    print("Basic slice size     [MB]: {0}"
          .format(np.ceil(max_slice_size / 1024**2).astype(int)))
    print("Basic slice number:        {0}\n"
          .format(tot_slice_num))


def print_image_shape(cli_args, image, mosaic, ch_axis=None):
    """
    Print volume image shape.

    Parameters
    ----------
    cli_args: see ArgumentParser.parse_args
        populated namespace of command line arguments

    image: numpy.ndarray (shape=(Z,Y,X))
        microscopy volume image

    mosaic: bool
        True for tiled reconstructions aligned using ZetaStitcher

    ch_axis: int
        channel axis (if ndim == 4)

    Returns
    -------
    None
    """
    # get pixel size
    px_size_z = cli_args.px_size_z
    px_size_xy = cli_args.px_size_xy

    # adapt axis order
    image_shape = image.shape
    if len(image_shape) == 4:
        if mosaic:
            ch_axis = 1
        else:
            ch_axis = -1

    # get image shape
    if ch_axis is not None:
        image_shape = np.delete(image_shape, ch_axis)

    print("\n                     Z      Y      X")
    print("Image shape [μm]: ({0:.1f}, {1:.1f}, {2:.1f})"
          .format(image_shape[0] * px_size_z, image_shape[1] * px_size_xy, image_shape[2] * px_size_xy))
