# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

"""
Implement behaviors (process models) of the processes defined in processes.py
For further documentation please refer to processes.py
"""
import glob
from os.path import isfile
import numpy as np
import time
import cv2 as cv
from matplotlib import pyplot as plt

from lava.lib.demos.object_tracking.matching_methods import (SQDIFF,
                                                             CCORR,
                                                             CCOEFF)
from lava.lib.demos.object_tracking.util import (grayscale_conversion,
                                                 generate_animation_compare,
                                                 scale_image,
                                                 draw_rectangule,
                                                 get_answers,
                                                 write_results,
                                                 crop_template_from_scaled_frame,
                                                 determine_scale_factor)
def main():
    # LaSOT filenames
    filename = r"images/lasot-protocol3-test/atv/atv-1/img/"
    sequence_id = "atv-1_CCOEFF"
    gt_path = filename + "../groundtruth.txt"

    # UAV123 filenames
    #filename = r"images/UAV123_10fps/data_seq/UAV123_10fps/bike1/"
    #sequence_id = "uav/bike1_CCORR_01"
    #gt_path = filename + "images/UAV123_10fps/anno/UAV123_10fps/bike1.txt"

    # calculate if scaling down is necessary
    scale_factor = determine_scale_factor(filename, gt_path)
    print(scale_factor)
    # read the frames, convert if not given in grayscale, downsample in
    # order to overcome the memory issue
    frames = [cv.imread(file) for file in sorted(glob.glob(filename + "*.jpg"))]
    frame_shape = frames[0].shape
    frames_orig = np.copy(frames)
    frames_orig_om = np.copy(frames)
    frames_orig2 = np.copy(frames)
    frames = grayscale_conversion(frames)
    frames = scale_image(frames, scale_factor=scale_factor)
    frames2 = np.copy(frames)
    convolution_type = 'valid'

    # for lasot
    template, original_template_shape = crop_template_from_scaled_frame(frames2[0], scale_factor, filename, gt_path)

    # for UAV
    #template, original_template_shape = crop_template_from_scaled_frame(frames[0], scale_factor, filename, gt_path)
    template2 = np.copy(template)

    valid_sm_shape = tuple((frame_shape[0]-original_template_shape[0]+1, frame_shape[1]-original_template_shape[1]+1))


    ### Calculate the template matching with Lava implementation
    sm, om = CCOEFF(frames=frames, template=template, scale_factor=scale_factor, convolution_type=convolution_type)
    sm = scale_image(sm, dimension=(valid_sm_shape[1],valid_sm_shape[0]))
    om = scale_image(om, dimension=(valid_sm_shape[1],valid_sm_shape[0]))

    rf = draw_rectangule(frames_orig, sm, original_template_shape, convolution_type=convolution_type)
    #rf_om = draw_rectangule(frames_orig_om, om, original_template_shape, convolution_type=convolution_type)

    """for smm in sm:
        plt.imshow(smm)
        plt.axis('off')
        plt.show()
    for rff in rf:
        plt.imshow(rff)
        plt.axis('off')
        plt.show()

    for omm in om:
        plt.imshow(omm)
        plt.axis('off')
        plt.show()
    for rff_om in rf_om:
        plt.imshow(rff_om)
        plt.axis('off')
        plt.show()"""

    answers_lava = get_answers(sm, convolution_type=convolution_type, template_shape=original_template_shape)
    write_results(answers_lava, './results/' + sequence_id + '_lava.txt')

    ### Calculate the matching with OpenCV TemplateMatching algorithm
    sm2 = []
    c =0
    for frame2 in frames2:

        saliency_map2 = cv.matchTemplate(frame2, template2, cv.TM_CCOEFF)
        sm2.append(saliency_map2)
        if np.mod(c, 20) == 0:
            """plt.imshow(saliency_map2)
            plt.axis('off')
            plt.show()"""
        c = c+1

    sm2 = scale_image(sm2, dimension=(valid_sm_shape[1],valid_sm_shape[0]))
    rf2 = draw_rectangule(frames_orig2, sm2, original_template_shape, convolution_type='valid')

    answers_opencv = get_answers(sm2, convolution_type='valid', template_shape=original_template_shape)
    write_results(answers_opencv, './results/' + sequence_id + '_opencv.txt')

    #generate_animation_compare(sm, rf, sm2, rf2, r"./images/videos/lasot_" + sequence_id + ".mp4")


if __name__ == "__main__":
    start = time.perf_counter()
    main()
    print(time.perf_counter() - start)