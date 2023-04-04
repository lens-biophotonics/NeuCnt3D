from neuCnt3D.input import (cli_parser, get_detection_config,
                            load_microscopy_image)
from neuCnt3D.output import save_soma, view_soma
from neuCnt3D.pipeline import parallel_neuron_detection_on_slices
from neuCnt3D.printing import print_pipeline_heading


def neuCnt3D(cli_args):

    # load microscopy volume image
    img, mosaic, cli_args, save_dir, tmp_dir, img_name = load_microscopy_image(cli_args)

    # get analysis configuration
    method, diam_um, overlap, rel_thresh, px_size, z_min, z_max, \
        ch_neu, backend, max_ram_mb, jobs_to_cores, out_name, view = get_detection_config(cli_args, img_name)

    # perform parallel unsupervised blob detection on batches of basic image slices
    blobs, neu_img = \
        parallel_neuron_detection_on_slices(img, px_size, method, diam_um, overlap, rel_thresh, tmp_dir,
                                            ch_neu=ch_neu, z_min=z_min, z_max=z_max, mosaic=mosaic,
                                            max_ram_mb=max_ram_mb, jobs_to_cores=jobs_to_cores,
                                            backend=backend)

    # save blob coordinates and radii to .csv log
    save_soma(blobs, px_size, save_dir, out_name)
    if view:
        view_soma(blobs, neu_img, method)


def main():

    # start NeuCnt3D pipeline by terminal
    print_pipeline_heading()
    neuCnt3D(cli_args=cli_parser())


if __name__ == '__main__':
    main()
