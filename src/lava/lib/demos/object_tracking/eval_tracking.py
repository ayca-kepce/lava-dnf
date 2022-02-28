# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

"""
Implement behaviors (process models) of the processes defined in processes.py
For further documentation please refer to processes.py
"""
import time
from lava.lib.demos.object_tracking.util import read_results, precision, write_precision


def main():
    # Change these two paths only.
    #sequence_id = "atv-1_CCOEFF"
    #gt_path = r"./images/lasot-protocol3-test/atv/atv-1/groundtruth.txt"
    sequence_id = "uav/boat1_CCOEFF"
    gt_path = r"images/UAV123_10fps/anno/UAV123_10fps/boat1.txt"

    lava_path = r"./results/" + sequence_id + "_lava.txt"
    opencv_path = r"./results/" + sequence_id + "_opencv.txt"

    results_lava = read_results(lava_path)
    results_opencv = read_results(opencv_path)
    results_gt = read_results(gt_path)

    frame_scores_lava, PRE_lava = precision(results_lava, results_gt, 20)
    frame_scores_opencv, PRE_opencv = precision(results_opencv, results_gt, 20)

    lava_PRE_path = r"./results/" + sequence_id + "_lava_PRE.txt"
    opencv_PRE_path = r"./results/" + sequence_id + "_opencv_PRE.txt"

    write_precision(frame_scores_lava, PRE_lava, lava_PRE_path)
    write_precision(frame_scores_opencv, PRE_opencv, opencv_PRE_path)

if __name__ == "__main__":
    start = time.perf_counter()
    main()
    print(time.perf_counter() - start)