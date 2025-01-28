# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

"""
Implement behaviors (process models) of the processes defined in processes.py
For further documentation please refer to processes.py
"""
import glob
from os.path import isfile
import os
import numpy as np
import time
import cv2 as cv
from matplotlib import pyplot as plt

from lava.lib.demos.object_tracking.matching_methods import (CCORR,
                                                             CCOEFF)
from lava.lib.demos.object_tracking.util import (grayscale_conversion,
                                                 generate_animation_compare,
                                                 scale_images,
                                                 draw_rectangule,
                                                 get_answers,
                                                 write_results,
                                                 crop_template_from_scaled_frame,
                                                 crop_template_from_scaled_frame_DAVIS,
                                                 determine_scale_factor)
def main():
    """Executes the object tracking with template matching algorithm both in Lava and in
       OpenCV. Provide the preferences and details for the operation down below. This includes
       giving the pathways, setting up the flags that controls saving the saliency maps on the
       RAM, convolution type, generate video that compares two algortihms, and scaling factor.
       If enabled, a video that compares two algorithms will be generated. Also, the maximum
       macth for every frame will be written to a txt file.

       LaSOT dataset can be found here http://vision.cs.stonybrook.edu/~lasot/
       UAV123 dataset can be found here https://cemse.kaust.edu.sa/ivul/uav123
       DAVIS 2016 dataset can be found here https://davischallenge.org/davis2016/code.html
       """

    save_saliency_maps = True
    convolution_type = 'valid'
    generate_video = True
    scaling_down = False

    # LaSOT filenames
    """filename = r"images/lasot-protocol3-test/cosplay/cosplay/cosplay-1/img/"
    sequence_id = "lasot/cosplay-1_CCOEFF_dnf"
    gt_path = filename + "../groundtruth.txt"
    first_frame_name = '00000001.jpg'

    # UAV123 filenames
    filename = r"images/UAV123_10fps/data_seq/UAV123_10fps/person1/"
    sequence_id = "uav/person1_CCOEFF"
    gt_path = "images/UAV123_10fps/anno/UAV123_10fps/person1.txt"
    first_frame_name = '000001.jpg'"""

    # DAVIS 2016 filenames

    gt_path = r"C:\Users\AYCA\Desktop\intel_backup\DAVIS\Annotations\480p"
    objects = os.listdir(gt_path + r'\\')

    for object in objects:
        filename = r"C:\Users\AYCA\Desktop\intel_backup\DAVIS\JPEGImages\480p" + r"\\" + object + r"\\"
        sequence_id = object + "_CCOEFF_sf02"
        gt_complete_path = r"C:\Users\AYCA\Desktop\intel_backup\DAVIS\Annotations\480p" + r"\\" + object + r"\bounding_box.txt"
        first_frame_name = '00000.jpg'
        result_path = r".\media\DAVIS16\results" + r"\\"

        # calculate if scaling down is necessary
        if scaling_down:
            scale_factor = determine_scale_factor(filename, gt_complete_path, first_frame_name)
            print("Scaling factor for" + object + " is %f." %scale_factor)
        else:
            scale_factor = 0.2

        # read the frames, convert if not given in grayscale, downscale in order to overcome the memory issue
        frames = [cv.imread(file) for file in sorted(glob.glob(filename + "*.jpg"))]
        frame_shape = frames[0].shape
        frames_orig = np.copy(frames)
        frames_orig2 = np.copy(frames)
        frames = grayscale_conversion(frames)
        frames = scale_images(frames, scale_factor=scale_factor)
        frames2 = np.copy(frames)

        # crop the template given the groundtruth information
        template, original_template_shape = crop_template_from_scaled_frame_DAVIS(frames[0], scale_factor, filename, gt_complete_path)
        template2 = np.copy(template)

        # calculate the shape of the result of the convolution if it is valid convolution
        valid_sm_shape = tuple((frame_shape[0]-original_template_shape[0]+1, frame_shape[1]-original_template_shape[1]+1))

        ### Calculate the template matching with Lava implementation
        sm = []
        answers_lava = []
        om = []
        sm, answers_lava, om = CCOEFF(frames=frames, template=template, scale_factor=scale_factor,
                                      convolution_type=convolution_type, original_template_size=original_template_shape,
                                      conv_shape=valid_sm_shape, save_saliency_maps=save_saliency_maps)

        # write the maximum of the resulting saliency map to a txt file
        write_results(answers_lava, result_path + sequence_id + '_lava.txt')

        # draw rectangle on the original frames given the saliency maps
        rf = draw_rectangule(frames_orig, sm, original_template_shape, convolution_type=convolution_type)
        cv.imwrite(result_path + sequence_id + "_lava1.png", rf[0])
        cv.imwrite(result_path + sequence_id + "_lava10.png", rf[9])

        ### Calculate the matching with OpenCV TemplateMatching algorithm
        sm2 = []
        answers_opencv = []
        for frame2 in frames2:
            saliency_map2 = cv.matchTemplate(frame2, template2, cv.TM_CCOEFF)
            saliency_map2 = scale_images([saliency_map2], dimension=(valid_sm_shape[1], valid_sm_shape[0]))[0]
            if save_saliency_maps:
                sm2.append(saliency_map2)
            _, _, _, max_loc = cv.minMaxLoc(saliency_map2, None)
            answers_opencv.append((np.array((max_loc[0], max_loc[1])) + (
                int((original_template_shape[0] + 1) / 2), int((original_template_shape[1] + 1) / 2))))

        rf2 = draw_rectangule(frames_orig2, sm2, original_template_shape, convolution_type='valid')
        cv.imwrite(result_path + sequence_id + "_opencv1.png", rf2[0])
        cv.imwrite(result_path + sequence_id + "_opencv10.png", rf2[9])

        answers_opencv = get_answers(sm2, convolution_type='valid', template_shape=original_template_shape)
        write_results(answers_opencv, result_path + sequence_id + '_opencv.txt')

        if generate_video:
            generate_animation_compare(sm, rf, sm2, rf2, r".\media\DAVIS16\results\videos" + r"\\" + sequence_id + ".gif")

if __name__ == "__main__":
    main()
