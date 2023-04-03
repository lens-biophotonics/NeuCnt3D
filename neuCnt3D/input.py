import argparse

import numpy as np
import tifffile as tiff

try:
    from zetastitcher import VirtualFusedVolume
except ImportError:
    pass

import tempfile
from os import path

from neuCnt3D.output import create_save_dir
from neuCnt3D.printing import color_text, print_image_shape
from neuCnt3D.utils import add_output_prefix, create_memory_map, get_item_bytes


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    pass


def cli_parser():
    """
    Parse command line arguments.

    Parameters
    ----------
    None

    Returns
    -------
    cli_args: see ArgumentParser.parse_args
        populated namespace of command line arguments
    """
    # configure parser object
    cli_parser = argparse.ArgumentParser(
        description='NeuCnt3D: An Unsupervised 3D Neuron Count Pipeline\n'
                    'author:   Michele Sorelli (2023)\n\n',
        formatter_class=CustomFormatter)
    cli_parser.add_argument(dest='image_path',
                            help='path to input microscopy volume image\n'
                                 '* supported formats: .tif, .yml (ZetaStitcher stitch file)\n')
    cli_parser.add_argument('-m', '--method', default='log',
                            help='blob detection approach')
    cli_parser.add_argument('-o', '--overlap', type=float, default=33.0,
                            help='maximum blob overlap percentage [%%]')
    cli_parser.add_argument('-t', '--threshold', type=float, default=10.0,
                            help='minimum percentage intensity of peaks in the filtered image [%%]')
    cli_parser.add_argument('--min-diam', type=float, default=0,
                            help='minimum soma diameter of interest [μm]')
    cli_parser.add_argument('--max-diam', type=float, default=30,
                            help='maximum soma diameter of interest [μm]')
    cli_parser.add_argument('--stp-diam', type=float, default=5,
                            help='diameter step size [μm]')
    cli_parser.add_argument('-j', '--jobs-prc', type=float, default=80.0,
                            help='maximum parallel jobs relative to the number of available CPU cores [%%]')
    cli_parser.add_argument('-r', '--ram', type=float, default=None,
                            help='maximum RAM available [GB] (default: use all)')
    cli_parser.add_argument('-v', '--view', action='store_true', default=False,
                            help='visualize detected blobs')
    cli_parser.add_argument('--px-size-xy', type=float, default=0.878, help='lateral pixel size [μm]')
    cli_parser.add_argument('--px-size-z', type=float, default=1.0, help='longitudinal pixel size [μm]')
    cli_parser.add_argument('--ch-neuron', type=int, default=0, help='neuronal soma channel')
    cli_parser.add_argument('--z-min', type=float, default=0, help='forced minimum output z-depth [μm]')
    cli_parser.add_argument('--z-max', type=float, default=None, help='forced maximum output z-depth [μm]')

    # parse arguments
    cli_args = cli_parser.parse_args()

    return cli_args


def get_image_file(cli_args, mosaic=False):
    """
    Description

    Parameters
    ----------

    Returns
    -------

    """
    img_path = cli_args.image_path
    img_fname = path.basename(img_path)
    split_name = img_fname.split('.')
    img_name = img_fname.replace('.' + split_name[-1], '')

    if len(split_name) == 1:
        raise ValueError('Format must be specified for input volume images!')
    else:
        img_fmt = split_name[-1]
        if img_fmt == 'yml':
            mosaic = True

    return img_path, img_name, mosaic


def get_image_info(img, px_size, ch_neuron, mosaic=False, ch_axis=None):
    """
    Get information on the input microscopy volume image.

    Parameters
    ----------
    img: numpy.ndarray
        microscopy volume image

    px_size: numpy.ndarray (shape=(3,), dtype=float)
        pixel size [μm]

    ch_neuron: int
        neuronal bodies channel

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
        if mosaic:
            ch_axis = 1
        else:
            ch_axis = -1
    elif ndim == 3:
        ch_neuron = None

    # get info on microscopy volume image
    if ch_axis is not None:
        img_shape = np.delete(img_shape, ch_axis)
    img_shape_um = np.multiply(img_shape, px_size)
    img_item_size = get_item_bytes(img)

    return img_shape, img_shape_um, img_item_size, ch_neuron


def get_detection_config(cli_args, img_name):
    """
    Retrieve the NeuCnt3D pipeline configuration.

    Parameters
    ----------
    cli_args: see ArgumentParser.parse_args
        populated namespace of command line arguments

    img_name: str
        name of the input volume image

    Returns
    -------
    min_sigma_px

    max_sigma_px

    num_sigma

    px_size: numpy.ndarray (shape=(3,), dtype=float)
        pixel size [μm]

    z_min: int
        minimum output z-depth [px]

    z_max: int
        maximum output z-depth [px]

    ch_neuron: int
        neuronal bodies channel

    max_ram_mb: float
        maximum RAM available to the Frangi filtering stage [MB]

    jobs_to_cores: float
        max number of jobs relative to the available CPU cores
        (default: 80%)

    img_name: str
        microscopy image filename
    """
    # pipeline configuration
    ch_neuron = cli_args.ch_neuron
    max_ram = cli_args.ram
    max_ram_mb = None if max_ram is None else max_ram * 1000
    jobs_prc = cli_args.jobs_prc
    jobs_to_cores = 1 if jobs_prc >= 100 else 0.01 * jobs_prc
    method = cli_args.method
    view_blobs = cli_args.view

    # image pixel size
    px_size_z = cli_args.px_size_z
    px_size_xy = cli_args.px_size_xy
    px_size = np.array([px_size_z, px_size_xy, px_size_xy])
    px_size_max = np.max(px_size)

    # spatial scales of interest
    min_diam_um = cli_args.min_diam
    max_diam_um = cli_args.max_diam
    stp_diam_um = cli_args.stp_diam
    num_sigma = len(np.arange(min_diam_um, max_diam_um, stp_diam_um)) - 1
    sigma_px = np.array([min_diam_um, max_diam_um]) / (2 * np.sqrt(3) * px_size_max)

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
    img_name = add_output_prefix(img_name, min_diam_um, max_diam_um, method)

    return method, sigma_px, num_sigma, overlap, rel_thresh, px_size, \
        z_min, z_max, ch_neuron, max_ram_mb, jobs_to_cores, img_name, view_blobs


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
        microscopy volume image or dataset of fiber orientation vectors

    mosaic: bool
        True for tiled microscopy reconstructions aligned using ZetaStitcher

    cli_args: see ArgumentParser.parse_args
        updated namespace of command line arguments

    save_dir: str
        saving directory

    tmp_dir: str
        temporary file directory

    img_name: str
        microscopy image name
    """
    # print heading
    print(color_text(0, 191, 255, "\nMicroscopy Volume Image Import\n"))

    # retrieve microscopy image path and name
    img_path, img_name, mosaic = get_image_file(cli_args)

    # load microscopy tiled reconstruction (aligned using ZetaStitcher)
    if mosaic:
        print("Loading " + img_name + " tiled reconstruction...")
        img = VirtualFusedVolume(img_path)

    # load microscopy z-stack
    else:
        print("Loading " + img_name + " z-stack...")
        img = tiff.imread(img_path)

    # create saving directory
    save_dir = create_save_dir(img_path, img_name)

    # create temporary file directory
    tmp_dir = tempfile.mkdtemp()

    # create image memory map
    mmap_path = path.join(tmp_dir, 'tmp_' + img_name + '.mmap')
    img = create_memory_map(mmap_path, img.shape, dtype=img.dtype, arr=img[:], mmap_mode='r')

    # print microscopy image shape
    print_image_shape(cli_args, img, mosaic)

    return img, mosaic, cli_args, save_dir, tmp_dir, img_name
