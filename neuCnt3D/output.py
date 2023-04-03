from datetime import datetime
from os import mkdir, path

import napari
import pandas as pd


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


def save_soma(blobs, save_dir, save_fname):
    """
    """
    df = pd.DataFrame({'z': blobs[:, 0], 'y': blobs[:, 1], 'x': blobs[:, 2], 'rad': blobs[:, 3]})
    df.to_csv(path.join(save_dir, save_fname + '.csv'), mode='a', sep=';', index=False, header=True)


def view_soma(neu_img, blobs, face_color='orangered'):
    """
    """
    viewer = napari.view_image(neu_img, rgb=False)
    viewer.add_points(blobs[:, :-1], size=blobs[:, -1], name='points', face_color=face_color)
    napari.run()
