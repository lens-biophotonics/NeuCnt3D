import os
from platform import system

import numpy as np


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


def print_analysis_info(method, diam_um, sigma_num, blob_ovlp, abs_thr, rel_thr,
                        img_shape_um, slice_shape_um, slice_num, px_sz, img_item_sz):
    """
    Print analysis configuration.

    Parameters
    ----------
    method: str
        blob detection approach
        (Laplacian of Gaussian or Difference of Gaussian)

    diam_um: tuple
        soma diameter (minimum, maximum, step size) [μm]

    sigma_num: int
        number of spatial scales

    blob_ovlp: float
        maximum blob overlap percentage [%]

    abs_thr: float
        absolute blob intensity threshold

    rel_thr: float
        minimum peak intensity
        in the filtered image relative to maximum

    img_shape_um: numpy.ndarray (shape=(3,), dtype=float)
        volume image shape [μm]

    slice_shape_um: numpy.ndarray (shape=(3,), dtype=float)
        shape of the analyzed image slices [μm]

    slice_num: int
        total number of analyzed image slices

    px_sz: numpy.ndarray (shape=(3,), dtype=float)
        pixel size [μm]

    img_item_sz: int
        image item size (in bytes)

    Returns
    -------
    None
    """
    print(color_text(0, 191, 255, "\n\n3D Neuronal Body Localization"))

    print_blob_info(method, diam_um, sigma_num, blob_ovlp, abs_thr, rel_thr)

    print_slicing_info(img_shape_um, slice_shape_um, slice_num, px_sz, img_item_sz)


def print_pipeline_heading():
    """
    Print NeuCnt3D pipeline heading.

    Returns
    -------
    None
    """
    print(color_text(0, 250, 154, "\n3D Unsupervised Neuron Segmentation and Counting"))


def print_blob_info(method, diam_um, sigma_num, blob_ovlp, abs_thr, rel_thr):
    """
    Print blob detection information.

    Parameters
    ----------
    method: str
        blob detection approach
        (Laplacian of Gaussian or Difference of Gaussian)

    diam_um: tuple
        soma diameter (minimum, maximum, step size) [μm]

    sigma_num: int
        number of spatial scales

    blob_ovlp: float
        maximum blob overlap percentage [%]

    abs_thr: float
        absolute blob intensity threshold

    rel_thr: float
        minimum percentage peak intensity
        in the filtered image relative to maximum [%]

    Returns
    -------
    None
    """
    min_diam_um, max_diam_um, stp_diam_um = diam_um

    if method == 'log':
        print("\nMethod: " + color_text(0, 255, 84, "Laplacian of Gaussian"))
    elif method == 'dog':
        print("\nMethod: " + color_text(255, 126, 0, "Difference of Gaussian"))

    if abs_thr is not None:
        print("Absolute blob threshold: {0:.1f}".format(abs_thr))
    if rel_thr is not None:
        print("Relative blob threshold: {0:.1f}%".format(100 * rel_thr))

    print("Maximum blob overlap:    {0:.1f}%".format(100 * blob_ovlp))
    print("Minimum diameter   [μm]: {0:.1f}".format(min_diam_um))
    print("Maximum diameter   [μm]: {0:.1f}".format(max_diam_um))
    print("Diameter step      [μm]: {0:.1f}".format(stp_diam_um))
    print("Spatial scales:          {0}".format(sigma_num))


def print_results(blobs, img_shape):
    """
    Print soma detection results.

    Parameters
    ----------
    blobs: numpy.ndarray (shape=(N,4))
        2D array with each row representing 3 coordinate values for a 3D image,
        plus the best sigma of the Gaussian kernel which detected the blob

    img_shape: numpy.ndarray (shape=(3,), dtype=float)
        volume image shape [μm]

    Returns
    -------
    None
    """
    num_cell = len(blobs)
    print("\nTotal cell count: {0}\t({1} cell/mm\u00b3)\n"
          .format(num_cell, np.floor(1e9 * num_cell / np.prod(img_shape)).astype(int)))


def print_slicing_info(img_shape_um, slice_shape_um, slice_num, px_sz, img_item_sz):
    """
    Print information on the slicing of the basic image sub-volumes
    iteratively processed by the Foa3D pipeline.

    Parameters
    ----------
    img_shape_um: numpy.ndarray (shape=(3,), dtype=float)
        volume image shape [μm]

    slice_shape_um: numpy.ndarray (shape=(3,), dtype=float)
        shape of the analyzed image slices [μm]

    slice_num: int
        total number of analyzed image slices

    px_sz: numpy.ndarray (shape=(3,), dtype=float)
        pixel size [μm]

    img_item_sz: int
        image item size (in bytes)

    Returns
    -------
    None
    """
    # adjust slice shape
    if np.any(img_shape_um < slice_shape_um):
        slice_shape_um = img_shape_um

    # get image memory size
    img_sz = img_item_sz * np.prod(np.divide(img_shape_um, px_sz))

    # get slice memory size
    max_slice_sz = img_item_sz * np.prod(np.divide(slice_shape_um, px_sz))

    # print info
    print("\n                           Z      Y      X")
    print("Total image shape  [μm]: ({0:.1f}, {1:.1f}, {2:.1f})"
          .format(img_shape_um[0], img_shape_um[1], img_shape_um[2]))
    print("Total image size   [MB]: {0}\n"
          .format(np.ceil(img_sz / 1024**2).astype(int)))
    print("Basic slice shape  [μm]: ({0:.1f}, {1:.1f}, {2:.1f})"
          .format(slice_shape_um[0], slice_shape_um[1], slice_shape_um[2]))
    print("Basic slice size   [MB]: {0}"
          .format(np.ceil(max_slice_sz / 1024**2).astype(int)))
    print("Basic slice number:      {0}\n"
          .format(slice_num))
