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
        name of the input volume image

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
    Description

    Parameters
    ----------

    Returns
    -------

    """
    # convert to [μm]
    blobs = blobs * px_size[0]

    # save to .csv
    df = pd.DataFrame({'z [μm]': blobs[:, 0], 'y [μm]': blobs[:, 1], 'x [μm]': blobs[:, 2], 'rad [μm]': blobs[:, 3]})
    df.to_csv(path.join(save_dir, save_fname + '.csv'), mode='a', sep=';', index=False, header=True)


def view_soma(blobs, neu_img, method, edge_width=0.2):
    """
    Description

    Parameters
    ----------

    Returns
    -------

    """
    if method == 'log':
        edge_color = 'lime'
    elif method == 'dog':
        edge_color = 'orangered'

    viewer = napari.view_image(neu_img, rgb=False)
    viewer.add_points(blobs[:, :-1], size=blobs[:, -1], name='points',
                      edge_color=edge_color, edge_width=edge_width, face_color=[0] * 4)
    napari.run()
