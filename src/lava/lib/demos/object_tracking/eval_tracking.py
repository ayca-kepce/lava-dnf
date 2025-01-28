# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

"""
Implement behaviors (process models) of the processes defined in processes.py
For further documentation please refer to processes.py
"""
import time
from lava.lib.demos.object_tracking.util import read_results, precision, write_precision
import os

def main():
    """Calculates the L2 norm of the prediction to the ground truth for every
       frame, and calculates the precision score at the end.

       The data is stored in a txt file. The first entry at every line
       resembles the distance of the prediction to the ground truth for the
       given frame. The second entry compares it with threshold and returns
       true if the distance is below than the threshold for every frame. The
       third entry is the precision score for the whole sequence.
       """

    """
    # Sequence id and path to ground truth for LaSOT
    sequence_id = "lasot/cosplay-1_CCOEFF"
    gt_path = r"./images/lasot-protocol3-test/cosplay/cosplay-1/groundtruth.txt"

    # Sequence id and path to ground truth for UAV123
    sequence_id = "uav/person1_CCOEFF"
    gt_path = r"images/UAV123_10fps/anno/UAV123_10fps/person1.txt"
    """

    # Threshold for precision score (given as 20 UAV123 and LASOT)
    threshold = 100

    # Sequence id and path to ground truth for DAVIS 2016
    sequence_id = "_CCORR"
    scale_factor = "_sf02"
    av_precision_lava = 0
    av_precision_opencv = 0

    gt_path = r"C:\Users\AYCA\Desktop\intel_backup\DAVIS\Annotations\480p"
    objects = os.listdir(gt_path + r'\\')
    for object in objects:
        result_path = r".\media\DAVIS16\results\answers" + r"\\"
        lava_path =   result_path + object + sequence_id + scale_factor + "_lava.txt"
        opencv_path = result_path + object + sequence_id + scale_factor + "_opencv.txt"
        gt_complete_path = gt_path + r"\\" + object + r"\bounding_box.txt"

        # Read the results of Lava and OpenCV template matching algorithms and the ground truths
        results_lava = read_results(lava_path)
        results_opencv = read_results(opencv_path)
        results_gt = read_results(gt_complete_path)

        # Calculate the L2 norms and precision scores
        frame_scores_lava, PRE_lava = precision(results_lava, results_gt, threshold)
        frame_scores_opencv, PRE_opencv = precision(results_opencv, results_gt, threshold)

        lava_PRE_path =   result_path + r"..\\precisions\\th" + str(threshold) + r"\\" + object + sequence_id + "_lava_PRE.txt"
        opencv_PRE_path = result_path + r"..\\precisions\\th" + str(threshold) + r"\\" + object + sequence_id + "_opencv_PRE.txt"

        # Write the L2 norms and precision scores
        write_precision(frame_scores_lava, PRE_lava, lava_PRE_path)
        write_precision(frame_scores_opencv, PRE_opencv, opencv_PRE_path)

        av_precision_lava = av_precision_lava + PRE_lava
        av_precision_opencv = av_precision_opencv + PRE_opencv

    print("Average precision of Lava tracker is: ", av_precision_lava/50)
    print("Average precision of OpenCV tracker is: ", av_precision_opencv/50)



if __name__ == "__main__":
    main()
