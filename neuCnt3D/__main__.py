from neuCnt3D.detection import detect_soma
from neuCnt3D.input import (cli_parser, get_detection_config,
                            load_microscopy_image)
from neuCnt3D.output import save_soma, view_soma
from neuCnt3D.preprocessing import correct_image_anisotropy
from neuCnt3D.printing import print_pipeline_heading
from neuCnt3D.utils import delete_tmp_folder


def neuCnt3D(cli_args):

    # load microscopy volume image
    img, mosaic, cli_args, save_dir, tmp_dir, img_name = load_microscopy_image(cli_args)

    # get analysis configuration
    method, sigma_px, num_sigma, overlap, rel_thresh, px_size, z_min, z_max, \
        ch_neuron, max_ram_mb, jobs_to_cores, out_fname, view = get_detection_config(cli_args, img_name)

    # preprocessing: correct anisotropic pixel size
    neu_img = img[:, :, :, ch_neuron]
    neu_img = correct_image_anisotropy(neu_img, px_size)

    # perform unsupervised neuron count analysis
    blobs = detect_soma(neu_img, method=method, min_sigma=sigma_px[0], max_sigma=sigma_px[1],
                        num_sigma=num_sigma, overlap=overlap, threshold_rel=rel_thresh)

    # save and view blobs
    save_soma(blobs, save_dir, out_fname)
    if view:
        view_soma(neu_img, blobs)

    # delete temporary folder
    delete_tmp_folder(tmp_dir)


def main():
    # start NeuCnt3D pipeline by terminal
    print_pipeline_heading()
    neuCnt3D(cli_args=cli_parser())


if __name__ == '__main__':
    main()
