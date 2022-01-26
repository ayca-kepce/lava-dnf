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
from matplotlib import pyplot as plt

from lava.lib.demos.object_tracking.matching_methods import (SQDIFF,
                                                             CCORR,
                                                             CCOEFF)
from lava.lib.demos.object_tracking.util import generate_animation
from lava.proc.lif.process import LIF
from lava.lib.dnf.connect.connect import connect
from lava.lib.dnf.operations.operations import Convolution, Weights
from lava.lib.dnf.kernels.kernels import ConvolutionKernel, SelectiveKernel
from lava.lib.demos.object_tracking.processes import (TemplateMatching,
                                                      TemplateNormalization,
                                                      FrameInput,
                                                      FrameNormalization,
                                                      OutputDNF)
from lava.lib.demos.object_tracking.neurons.processes import one_input_neuron, two_input_neuron
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi1SimCfg

def mainkukj():
    filename = "./images/"
    frames = [cv.imread(file) for file in sorted(glob.glob(r"./images/*.jpg"))]
    template = cv.imread(filename + "fish_template.png")

    #for frame in frames:
        #saliency_map2 = cv.matchTemplate(frame, template, cv.TM_CCOEFF)
        #saliency_map2 = cv.resize(saliency_map2, (frame.shape[1],frame.shape[0]), interpolation=cv.INTER_AREA)

    saliency_maps, output_maps, frames_rect = CCOEFF(frames=frames,template=template, scale_factor=0.3)
    generate_animation(frames_rect, output_maps, saliency_maps, "lava-fish-CCOEFF.mp4")

def main():
    filename = "./images/"
    frame = [cv.imread(filename + "mock.png")]
    template = cv.imread(filename + "mock_template.png")
    frames_rect =[]
    saliency_maps =[]

    sm, om= CCORR(frames=frame,template=template, scale_factor=1)
    #frames_rect.append(fm_rect)
    saliency_maps.append(sm)

    saliency_map2 = cv.matchTemplate(frame[0], template, cv.TM_CCORR)
    #saliency_map2 = cv.resize(saliency_map2, (frame.shape[1],frame.shape[0]), interpolation=cv.INTER_AREA)
    plt.imshow(saliency_map2)
    plt.show()

def mainrht():
    filename = "./images/"
    frame = []
    frame.append(np.array([[[1,2,3],[4,5,6],[7,8,9]],[[1,2,3],[4,5,6],[7,8,9]],[[1,2,3],[4,5,6],[7,8,9]]]))
    template = np.array([[[1,2,1],[0,0,0],[1,2,1]],[[1,2,1],[0,0,0],[1,2,1]],[[1,2,1],[0,0,0],[1,2,1]]])*2
    frames_rect =[]
    saliency_maps =[]

    sm, om= CCORR(frames=frame,template=template*2, scale_factor=1)
    #frames_rect.append(fm_rect)
    saliency_maps.append(sm)
    saliency_map2 = cv.matchTemplate(frame[0].astype('float32'), template, cv.TM_CCORR)
    #saliency_map2 = cv.resize(saliency_map2, (frame.shape[1],frame.shape[0]), interpolation=cv.INTER_AREA)
    plt.imshow(saliency_map2)
    plt.show()

if __name__ == "__main__":
    start = time.perf_counter()
    main()
    print(time.perf_counter() - start)