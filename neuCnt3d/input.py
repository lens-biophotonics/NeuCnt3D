import argparse
import tempfile

import numpy as np
import tifffile as tiff

try:
    from zetastitcher import VirtualFusedVolume
except ImportError:
    pass

from os import path

from neuCnt3d.output import create_save_dir
from neuCnt3d.printing import color_text
from neuCnt3d.utils import add_output_prefix, create_memory_map, get_item_bytes


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    pass


def get_cli_parser():
    """
    Parse command line arguments.

    Returns
    -------
    cli_args: see ArgumentParser.parse_args
        populated namespace of command line arguments
    """
    # configure parser object
    cli_parser = argparse.ArgumentParser(
        description='NeuCnt3D: An Unsupervised 3D Neuron Counting Tool\n'
                    'author:   Michele Sorelli (2023)\n\n',
        formatter_class=CustomFormatter)
    cli_parser.add_argument(dest='image_path',
                            help='path to input microscopy volume image\n'
                                 '* supported formats: .tif, .yml (ZetaStitcher stitch file)\n')
    cli_parser.add_argument('-a', '--approach', default='log',
                            help='blob detection approach:\n'
                                 u'log \u2023 Laplacian of Gaussian\n'
                                 u'dog \u2023 Difference of Gaussian')
    cli_parser.add_argument('-o', '--overlap', type=float, default=33.0,
                            help='maximum blob overlap percentage [%%]')
    cli_parser.add_argument('-t', '--threshold', type=float, default=10.0,
                            help='minimum percentage intensity of peaks in the filtered image relative to maximum [%%]')
    cli_parser.add_argument('-j', '--jobs-prc', type=float, default=80.0,
                            help='maximum parallel jobs relative to the number of available CPU cores [%%]')
    cli_parser.add_argument('-r', '--ram', type=float, default=None,
                            help='maximum RAM available [GB]: use all if None')
    cli_parser.add_argument('-b', '--backend', default='threading',
                            help='Joblib parallelization backend implementation '
                            '(either loky, multiprocessing or threading)')
    cli_parser.add_argument('-v', '--view', action='store_true', default=False,
                            help='visualize detected blobs')
    cli_parser.add_argument('-m', '--mmap', action='store_true', default=False,
                            help='create a memory-mapped array of the microscopy volume image')
    cli_parser.add_argument('--min-diam', type=float, default=10.0,
                            help='minimum soma diameter of interest [μm]')
    cli_parser.add_argument('--max-diam', type=float, default=40.0,
                            help='maximum soma diameter of interest [μm]')
    cli_parser.add_argument('--stp-diam', type=float, default=5.0,
                            help='diameter step size [μm]')
    cli_parser.add_argument('--px-size-xy', type=float, default=0.878, help='lateral pixel size [μm]')
    cli_parser.add_argument('--px-size-z', type=float, default=1.0, help='longitudinal pixel size [μm]')
    cli_parser.add_argument('--ch-neuron', type=int, default=0, help='neuronal soma channel (RGB image)')
    cli_parser.add_argument('--z-min', type=float, default=0, help='forced minimum output z-depth [μm]')
    cli_parser.add_argument('--z-max', type=float, default=None, help='forced maximum output z-depth [μm]')

    # parse arguments
    cli_args = cli_parser.parse_args()

    return cli_args


def get_image_file(cli_args):
    """
    Get microscopy image file path and format.

    Parameters
    ----------
    cli_args: see ArgumentParser.parse_args
        populated namespace of command line arguments

    Returns
    -------
    img_path: str
        path to the input microscopy volume image

    img_name: str
        name of the input microscopy volume image

    mosaic: bool
        True for tiled reconstructions aligned using ZetaStitcher

    in_mmap: bool
        create a memory-mapped array of the microscopy volume image,
        increasing the parallel processing performance
        (the image will be preliminarily loaded to RAM)
    """
    in_mmap = cli_args.mmap
    img_path = cli_args.image_path
    img_name = path.basename(img_path)
    split_name = img_name.split('.')

    if len(split_name) == 1:
        raise ValueError('Format must be specified for input volume images!')
    else:
        mosaic = False
        img_fmt = split_name[-1]
        img_name = img_name.replace('.' + split_name[-1], '')
        if img_fmt == 'yml':
            mosaic = True

    return img_path, img_name, mosaic, in_mmap


def get_image_info(img, px_size, ch_neu, mosaic=False, ch_axis=None):
    """
    Get information on the input microscopy volume image.

    Parameters
    ----------
    img: numpy.ndarray
        microscopy volume image

    px_size: numpy.ndarray (shape=(3,), dtype=float)
        pixel size [μm]

    ch_neu: int
        neuronal soma channel

    mosaic: bool
        True for tiled reconstructions aligned using ZetaStitcher

    ch_axis: int
        channel axis

    Returns
    -------
    img_shape: numpy.ndarray (shape=(3,), dtype=int)
        volume image shape [px]

    img_shape_um: numpy.ndarray (shape=(3,), dtype=float)
        volume image shape [μm]

    img_item_size: int
        array item size (in bytes)
    """
    # adapt channel axis
    img_shape = np.asarray(img.shape)
    ndim = len(img_shape)
    if ndim == 4:
        ch_axis = 1 if mosaic else -1
    elif ndim == 3:
        ch_neu = None

    # get info on microscopy volume image
    if ch_axis is not None:
        img_shape = np.delete(img_shape, ch_axis)
    img_shape_um = np.multiply(img_shape, px_size)
    img_item_size = get_item_bytes(img)

    return img_shape, img_shape_um, img_item_size, ch_neu


def get_detection_config(cli_args, img_name):
    """
    Retrieve the NeuCnt3D tool configuration.

    Parameters
    ----------
    cli_args: see ArgumentParser.parse_args
        populated namespace of command line arguments

    img_name: str
        name of the input microscopy volume image

    Returns
    -------
    approach: str
        blob detection approach
        (Laplacian of Gaussian or Difference of Gaussian)

    diam_um: tuple
        soma diameter (minimum, maximum, step size) [μm]

    overlap: float
        maximum blob overlap percentage [%]

    rel_thresh: float
        minimum percentage intensity of peaks in the filtered image relative to maximum [%]

    px_size: numpy.ndarray (shape=(3,), dtype=float)
        pixel size [μm]

    z_min: int
        minimum output z-depth [px]

    z_max: int
        maximum output z-depth [px]

    ch_neuron: int
        neuronal bodies channel (RGB image)

    backend: str
        supported parallelization backend implementations:
        loky, multiprocessing or threading

    max_ram_mb: float
        maximum RAM available to the blob detection stage [MB]

    jobs_to_cores: float
        max number of jobs relative to the available CPU cores
        (default: 80%)

    img_name: str
        microscopy image filename
    """
    # pipeline configuration
    ch_neuron = cli_args.ch_neuron
    backend = cli_args.backend
    max_ram = cli_args.ram
    max_ram_mb = None if max_ram is None else max_ram * 1000
    jobs_prc = cli_args.jobs_prc
    jobs_to_cores = 1 if jobs_prc >= 100 else 0.01 * jobs_prc
    approach = cli_args.approach
    view_blobs = cli_args.view

    # image pixel size
    px_size_z = cli_args.px_size_z
    px_size_xy = cli_args.px_size_xy
    px_size = np.array([px_size_z, px_size_xy, px_size_xy])

    # spatial scales of interest
    min_diam_um = cli_args.min_diam
    max_diam_um = cli_args.max_diam
    stp_diam_um = cli_args.stp_diam
    diam_um = (min_diam_um, max_diam_um, stp_diam_um)

    # other detection parameters
    overlap = 0.01 * cli_args.overlap
    rel_thresh = 0.01 * cli_args.threshold

    # forced output z-range
    z_min = cli_args.z_min
    z_max = cli_args.z_max
    z_min = int(np.floor(z_min / px_size[0]))
    if z_max is not None:
        z_max = int(np.ceil(z_max / px_size[0]))

    # add configuration prefix to output filenames
    img_name = add_output_prefix(img_name, min_diam_um, max_diam_um, approach)

    return approach, diam_um, overlap, rel_thresh, px_size, \
        z_min, z_max, ch_neuron, backend, max_ram_mb, jobs_to_cores, img_name, view_blobs


def load_microscopy_image(cli_args):
    """
    Load microscopy volume image from TIFF, NumPy or ZetaStitcher .yml file.

    Parameters
    ----------
    cli_args: see ArgumentParser.parse_args
        populated namespace of command line arguments

    Returns
    -------
    img: NumPy memory map
        microscopy volume image

    mosaic: bool
        True for tiled microscopy reconstructions aligned using ZetaStitcher

    cli_args: see ArgumentParser.parse_args
        updated namespace of command line arguments

    save_dir: str
        saving directory

    tmp: str
        temporary file directory

    img_name: str
        microscopy image name
    """
    # print heading
    print(color_text(0, 191, 255, "\nMicroscopy Volume Image Import\n"))

    # retrieve microscopy image path and name
    img_path, img_name, mosaic, in_mmap = get_image_file(cli_args)

    # load microscopy tiled reconstruction (aligned using ZetaStitcher)
    if mosaic:
        print("Loading " + img_name + " tiled reconstruction...")
        img = VirtualFusedVolume(img_path)

    # load microscopy z-stack
    else:
        print("Loading " + img_name + " z-stack...")
        img = tiff.imread(img_path)

    # create image memory map
    tmp_dir = None
    if in_mmap:
        tmp_dir = tempfile.mkdtemp()
        img = create_memory_map(img.shape, dtype=img.dtype, name=img_name, tmp=tmp_dir, arr=img[:], mmap_mode='r')

    # create saving directory
    save_dir = create_save_dir(img_path, img_name)

    return img, mosaic, cli_args, save_dir, tmp_dir, img_name