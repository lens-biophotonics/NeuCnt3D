from neuCnt3d.input import (get_cli_args, get_detection_config,
                            load_microscopy_image)
from neuCnt3d.output import save_soma, view_soma
from neuCnt3d.pipeline import parallel_neuron_detection_on_slices
from neuCnt3d.printing import print_pipeline_heading
from neuCnt3d.utils import delete_tmp_dir


def neuCnt3D(cli_args):

    # load microscopy volume image
    img, is_tiled, cli_args, save_dir, tmp_dir, img_name = load_microscopy_image(cli_args)

    # get analysis configuration
    blob_method, diam_um, ovlp, abs_thr, rel_thr, px_sz, z_rng, ch_neu, dark, \
        backend, max_ram, jobs, out_name, view = get_detection_config(cli_args, img_name)

    # perform parallel unsupervised blob detection on batches of basic image slices
    blobs, neu_img = \
        parallel_neuron_detection_on_slices(img, px_sz, blob_method, diam_um, ovlp, abs_thr, rel_thr,
                                            ch_neu=ch_neu, dark=dark, z_rng=z_rng, is_tiled=is_tiled,
                                            max_ram=max_ram, jobs=jobs, backend=backend, tmp_dir=tmp_dir, view=view)

    # save blob coordinates and radii to .csv log
    save_soma(blobs, px_sz, save_dir, out_name)
    if view:
        view_soma(blobs, neu_img, blob_method)

    # delete temporary folder
    delete_tmp_dir(tmp_dir)


def main():

    # start NeuCnt3D pipeline by terminal
    print_pipeline_heading()
    neuCnt3D(cli_args=get_cli_args())


if __name__ == '__main__':
    main()
