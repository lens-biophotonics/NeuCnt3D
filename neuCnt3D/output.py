import warnings
from datetime import datetime
from os import mkdir, path

import pandas as pd

warnings.simplefilter(action='ignore')

import napari # noqa


def create_save_dir(img_path, img_name):
    """
    Create saving directory.

    Parameters
    ----------
    img_path: str
        path to input microscopy volume image

    img_name: str
        name of the input microscopy volume image

    Returns
    -------
    save_dir: str
        saving directory
    """
    # get current time
    time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # get base path
    base_path = path.dirname(img_path)

    # create saving directory
    save_dir = path.join(base_path, time_stamp + '_' + img_name)
    if not path.isdir(save_dir):
        mkdir(save_dir)

    return save_dir


def save_soma(blobs, px_size, save_dir, save_fname):
    """
    Save detected soma coordinates and radii to log file.

    Parameters
    ----------
    blobs: numpy.ndarray (shape=(N,4))
        2D array with each row representing 3 coordinate values for a 3D image,
        plus the best sigma of the Gaussian kernel which detected the blob

    px_size: numpy.ndarray (shape=(3,), dtype=float)
        pixel size [μm]

    save_dir: str
        saving directory

    save_fname str
        saved filename

    Returns
    -------
    None
    """
    # convert to [μm]
    blobs = blobs * px_size[0]

    # save to .csv
    df = pd.DataFrame({'z [μm]': blobs[:, 0], 'y [μm]': blobs[:, 1], 'x [μm]': blobs[:, 2], 'rad [μm]': blobs[:, 3]})
    df.to_csv(path.join(save_dir, save_fname + '.csv'), mode='a', sep=';', index=False, header=True)


def view_soma(blobs, neu_img, method, edge_width=0.2):
    """
    Display the detected soma in the Napari viewer.

    Parameters
    ----------
    blobs: numpy.ndarray (shape=(N,4))
        2D array with each row representing 3 coordinate values for a 3D image,
        plus the best sigma of the Gaussian kernel which detected the blob

    neu_img: NumPy memory-map object (shape=(Z,Y,X), dtype=uint8)
        soma channel image

    method: str
        blob detection approach
        (Laplacian of Gaussian or Difference of Gaussian)

    edge_width: float
        width of the blob symbol edge

    Returns
    -------
    None
    """
    if method == 'log':
        edge_color = 'lime'
    elif method == 'dog':
        edge_color = 'orangered'

    viewer = napari.view_image(neu_img, rgb=False)
    viewer.add_points(blobs[:, :-1], size=blobs[:, -1], name='points',
                      edge_color=edge_color, edge_width=edge_width, face_color=[0] * 4)
    napari.run()
