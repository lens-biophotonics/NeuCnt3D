import argparse
import tempfile

import numpy as np
import tifffile as tiff

try:
    from zetastitcher import VirtualFusedVolume
except ImportError:
    pass

from os import getcwd, path

from neuCnt3d.output import create_save_dir
from neuCnt3d.printing import color_text
from neuCnt3d.utils import add_output_prefix, create_memory_map, get_item_bytes


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    pass


def get_cli_args():
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
    cli_parser.add_argument(dest='img_path',
                            help='path to input microscopy volume image\n'
                                 '* supported formats: .tif, .tiff, .yml (ZetaStitcher stitch file)\n')
    cli_parser.add_argument('-a', '--approach', default='log',
                            help='blob detection approach:\n'
                                 u'log \u2023 Laplacian of Gaussian\n'
                                 u'dog \u2023 Difference of Gaussian')
    cli_parser.add_argument('-o', '--ovlp', type=float, default=33.0,
                            help='maximum blob overlap percentage [%%]')
    cli_parser.add_argument('-j', '--jobs', type=int, default=16,
                            help='number of parallel jobs')
    cli_parser.add_argument('-r', '--ram', type=float, default=None,
                            help='maximum RAM available [GB]: use all if None')
    cli_parser.add_argument('-b', '--backend', default='threading',
                            help='Joblib parallelization backend implementation '
                            '(either loky, multiprocessing or threading)')
    cli_parser.add_argument('-c', '--ch', type=int, default=0, help='neuronal soma channel (RGB image)')
    cli_parser.add_argument('-d', '--dark', action='store_true', default=False,
                            help='detect dark blobs on a bright background')
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
    cli_parser.add_argument('--rel-thr', type=float, default=None,
                            help='minimum percentage intensity of peaks in the filtered image relative to maximum [%%]')
    cli_parser.add_argument('--abs-thr', type=float, default=None,
                            help='minimum intensity of peaks in the filtered image')
    cli_parser.add_argument('--px-size-xy', type=float, default=0.878, help='lateral pixel size [μm]')
    cli_parser.add_argument('--px-size-z', type=float, default=1.0, help='longitudinal pixel size [μm]')
    cli_parser.add_argument('--z-min', type=float, default=0, help='forced minimum output z-depth [μm]')
    cli_parser.add_argument('--z-max', type=float, default=None, help='forced maximum output z-depth [μm]')

    # parse arguments
    cli_args = cli_parser.parse_args()

    return cli_args


def get_file_info(cli_args):
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

    is_tiled: bool
        True for tiled reconstructions aligned using ZetaStitcher

    is_mmap: bool
        create a memory-mapped array of the microscopy volume image,
        increasing the parallel processing performance
        (the image will be preliminarily loaded to RAM)
    """
    is_mmap = cli_args.mmap
    img_path = cli_args.img_path
    img_name = path.basename(img_path)
    split_name = img_name.split('.')

    if len(split_name) == 1:
        raise ValueError('Format must be specified for input volume images!')
    else:
        is_tiled = False
        img_fmt = split_name[-1]
        img_name = img_name.replace('.{}'.format(img_fmt), '')
        if img_fmt == 'yml':
            is_tiled = True

    return img_path, img_name, is_tiled, is_mmap


def get_image_info(img, px_size, ch_neu, is_tiled=False, ch_axis=None):
    """
    Get information on the input microscopy volume image.

    Parameters
    ----------
    img: numpy.ndarray or memory-mapped file (axis order: (Z,Y,X))
        microscopy volume image

    px_size: numpy.ndarray (shape=(3,), dtype=float)
        pixel size [μm]

    ch_neu: int
        neuronal soma channel

    is_tiled: bool
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

    img_max: float
        max image value

    ch_neu: int
        neuronal body channel (RGB image)
    """
    # adapt channel axis
    img_shape = np.asarray(img.shape)
    ndim = len(img_shape)
    if ndim == 4:
        ch_axis = 1 if is_tiled else -1
    elif ndim == 3:
        ch_neu = None

    # get info on microscopy volume image
    if ch_axis is not None:
        img_shape = np.delete(img_shape, ch_axis)
    img_shape_um = np.multiply(img_shape, px_size)
    img_item_size = get_item_bytes(img)
    img_max = np.iinfo(img.dtype).max

    return img_shape, img_shape_um, img_item_size, img_max, ch_neu


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
    blob_method: str
        blob detection approach
        (Laplacian of Gaussian or Difference of Gaussian)

    diam_um: tuple
        soma diameter (minimum, maximum, step size) [μm]

    ovlp: float
        maximum blob overlap percentage [%]

    abs_thr: float
        absolute blob intensity threshold

    rel_thr: float
        minimum percentage intensity of peaks in the filtered image relative to maximum [%]

    px_sz: numpy.ndarray (shape=(3,), dtype=float)
        pixel size [μm]

    z_rng: int
        output z-range in [px]

    ch_neu: int
        neuronal body channel (RGB image)

    dark: bool
        if True, detect dark 3D blob-like structures
        (i.e., negative contrast polarity)

    backend: str
        supported parallelization backend implementations:
        loky, multiprocessing or threading

    max_ram: float
        maximum RAM available to the blob detection stage [B]

    jobs: int
        number of parallel jobs

    img_name: str
        microscopy image filename

    view_blobs: bool
        visualize point cloud in the Napari viewer
    """
    # pipeline configuration
    ch_neu = cli_args.ch
    backend = cli_args.backend
    jobs = cli_args.jobs
    max_ram = cli_args.ram
    if max_ram is not None:
        max_ram *= 1024**3

    blob_method = cli_args.approach
    view_blobs = cli_args.view

    # image pixel size
    px_sz = np.array([cli_args.px_size_z, cli_args.px_size_xy, cli_args.px_size_xy])

    # spatial scales of interest
    min_diam_um = cli_args.min_diam
    max_diam_um = cli_args.max_diam
    stp_diam_um = cli_args.stp_diam
    diam_um = (min_diam_um, max_diam_um, stp_diam_um)

    # other detection parameters
    dark = cli_args.dark
    ovlp = 0.01 * cli_args.ovlp
    abs_thr = cli_args.abs_thr
    rel_thr = cli_args.rel_thr
    if rel_thr is not None:
        rel_thr *= 0.01

    # forced output z-range
    z_min = int(np.floor(cli_args.z_min / px_sz[0]))
    z_max = int(np.ceil(cli_args.z_max / px_sz[0])) if cli_args.z_max is not None else cli_args.z_max
    z_rng = (z_min, z_max)

    # add configuration prefix to output filenames
    img_name = add_output_prefix(img_name, min_diam_um, max_diam_um, blob_method)

    return blob_method, diam_um, ovlp, abs_thr, rel_thr, px_sz, \
        z_rng, ch_neu, dark, backend, max_ram, jobs, img_name, view_blobs


def load_microscopy_image(cli_args):
    """
    Load microscopy volume image from TIFF, NumPy or ZetaStitcher .yml file.

    Parameters
    ----------
    cli_args: see ArgumentParser.parse_args
        populated namespace of command line arguments

    Returns
    -------
    img: numpy.ndarray or NumPy memory-map object (axis order: (Z,Y,X))
        microscopy volume image

    is_tiled: bool
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
    img_path, img_name, is_tiled, is_mmap = get_file_info(cli_args)

    # import microscopy tiled reconstruction (aligned using ZetaStitcher)
    if is_tiled:
        print("Loading " + img_name + " tiled reconstruction...")
        img = VirtualFusedVolume(img_path)

    # import microscopy z-stack
    else:
        print("Loading " + img_name + " z-stack...")
        img = tiff.imread(img_path)

    # create image memory map
    tmp_dir = tempfile.mkdtemp(dir=getcwd())
    if is_mmap:
        img = create_memory_map(img.shape, dtype=img.dtype, name=img_name, tmp=tmp_dir, arr=img[:], mmap_mode='r')

    # create saving directory
    save_dir = create_save_dir(img_path, img_name)

    return img, is_tiled, cli_args, save_dir, tmp_dir, img_name
