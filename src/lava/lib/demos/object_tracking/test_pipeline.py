# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

"""
Implement behaviors (process models) of the processes defined in processes.py
For further documentation please refer to processes.py
"""
import glob
import numpy as np
import time
import cv2 as cv

from lava.lib.demos.object_tracking.matching_methods import (CCORR,
                                                             CCOEFF)
from lava.lib.demos.object_tracking.util import (grayscale_conversion,
                                                 scale_images,
                                                 get_answers,
                                                 write_results,
                                                 crop_template_from_scaled_frame,
                                                 determine_scale_factor)
def main():
    ### UAV123 filenames
    filenames = [r"images/UAV123_10fps/data_seq/UAV123_10fps/boat1/",
                 r"images/UAV123_10fps/data_seq/UAV123_10fps/building1/",
                 r"images/UAV123_10fps/data_seq/UAV123_10fps/bike1/"]
    gt_paths = [r"images/UAV123_10fps/anno/UAV123_10fps/boat1.txt",
                r"images/UAV123_10fps/anno/UAV123_10fps/building1.txt",
                r"images/UAV123_10fps/anno/UAV123_10fps/bike1.txt"]
    sequence_ids = ["boat1_CCOEFF",
                    "building_CCOEFF",
                    "bike1_CCOEFF"]
    for filename, sequence_id, gt_path in zip(filenames, sequence_ids, gt_paths):
        #gt_path = filename + "../groundtruth.txt"


        # calculate if scaling down is necessary
        scale_factor = determine_scale_factor(filename, gt_path)
        print('Scale factor is ', scale_factor)

        # read the frames, convert if not given in grayscale, downsample in
        # order to overcome the memory issue
        frames = [cv.imread(file) for file in sorted(glob.glob(filename + "*.jpg"))]
        frame_shape = frames[0].shape

        frames = grayscale_conversion(frames)
        frames = scale_images(frames, scale_factor=scale_factor)
        frames2 = np.copy(frames)
        convolution_type = 'valid'

        template, original_template_shape = crop_template_from_scaled_frame(frames[0], scale_factor, filename, gt_path)
        template2 = np.copy(template)

        valid_sm_shape = tuple((frame_shape[0]-original_template_shape[0]+1, frame_shape[1]-original_template_shape[1]+1))

        sm, answers_lava, om = CCOEFF(frames=frames, template=template, scale_factor=scale_factor, convolution_type=convolution_type,
                       original_template_size= original_template_shape)

        write_results(answers_lava, './results/uav/' + sequence_id + '_lava.txt')

        sm2 = []
        for frame2 in frames2:
            saliency_map2 = cv.matchTemplate(frame2, template2, cv.TM_CCOEFF)
            sm2.append(saliency_map2)

        sm2 = scale_images(sm2, dimension=(valid_sm_shape[1], valid_sm_shape[0]))
        answers_opencv = get_answers(sm2, convolution_type='valid', template_shape=original_template_shape)
        write_results(answers_opencv, './results/uav/' + sequence_id + '_opencv.txt')
        print("END OF " + sequence_id)


if __name__ == "__main__":
    start = time.perf_counter()
    main()
    print(time.perf_counter() - start)