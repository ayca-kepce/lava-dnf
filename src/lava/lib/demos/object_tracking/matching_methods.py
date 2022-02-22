# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

"""
Implement behaviors (process models) of the processes defined in processes.py
For further documentation please refer to processes.py
"""
import numpy as np
import cv2 as cv
import time
import typing as ty
from matplotlib import pyplot as plt

from lava.proc.monitor.process import Monitor

from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.sub.model import AbstractSubProcessModel
from lava.magma.core.run_configs import Loihi1SimCfg
from lava.magma.core.run_conditions import RunSteps
from lava.proc.lif.process import LIF
from lava.lib.dnf.connect.connect import connect
from lava.lib.dnf.operations.operations import Convolution, Weights
from lava.lib.dnf.kernels.kernels import ConvolutionKernel
from lava.lib.demos.object_tracking.processes import (TemplateMatching,
                                                      TemplateNormalization,
                                                      FrameInput,
                                                      FrameNormalization,
                                                      OutputDNF)
from lava.lib.demos.object_tracking.neurons.processes import one_input_neuron, two_input_neuron, \
    two_input_neuron_squares
from lava.lib.demos.object_tracking.util import get_answers


def SQDIFF(frame, template):
    #
    frame_input = FrameInput(frame=frame)

    # neuron population to subtract the mean from frame input
    subtraction = two_input_neuron_squares(shape=tuple(frame.shape))
    frame_input.s_out.connect(subtraction.a_in1)

    # create the kernel to be convolved
    kernel = ConvolutionKernel(template=-template)

    # creating the convoluion to calculate the mean of each patch
    conn2, sp12, sp22 = connect(frame_input.s_out, subtraction.a_in2,
                                ops=[Convolution(kernel=kernel)], sign_mode=1)

    # create the output DNF which is a SelectiveDNF
    output_population = LIF(shape=tuple(frame.shape), vth=2)

    """conn3, sp13, sp23 = connect(subtraction.s_out, output_population.a_in,
                             ops=[Weights(1)], sign_mode=2)"""
    subtraction.s_out.connect(output_population.a_in)

    frame_input.run(condition=RunSteps(num_steps=8),
                    run_cfg=Loihi1SimCfg(select_tag='graded', select_sub_proc_model=True))
    # print('SALIENCY',saliency_map.saliency_map.get())

    frame_input.stop()


def CCORR(frames, template, **kwargs):
    scale_factor = kwargs.pop("scale_factor", 1)
    convolution_type = kwargs.pop("convolution_type", "valid")

    # frame is given as input
    frame_input = FrameInput(frame=frames[0])

    saliency_map = TemplateMatching(template=template,
                                    frame_shape=frames[0].shape,
                                    convolution_type=convolution_type)

    # connect the input frame to the neuron population of same size
    frame_input.s_out.connect(saliency_map.a_in)

    # create the output DNF which is a SelectiveDNF
    sm_shape = tuple((frames[0].shape[0]-template.shape[0]+1, frames[0].shape[1]-template.shape[1]+1))

    """output_map = OutputDNF(sm_shape=np.array(frames[0].shape)-np.array(template.shape)+1, vth=200, amp_exc=2, width_exc=template.shape, global_inh=-250)
    saliency_map.s_out.connect(output_map.a_in)"""

    fm = []
    sm = []
    om = []

    saliency_map.run(condition=RunSteps(num_steps=3),
                     run_cfg=Loihi1SimCfg(select_tag='graded', select_sub_proc_model=True))
    sal_map = saliency_map.saliency_map.get()

    sm.append(sal_map)
    """plt.imshow(sal_map)
    plt.axis('off')
    plt.show()"""
    #om.append(output_map.output_map.get())

    # weigggtt = saliency_map.wee.get()
    # print("weeee", weigggtt)
    # print("weeeeminn", np.min(weigggtt))
    # print("weeeemaaxx", np.max(weigggtt))

    # print('FRAME', frame_input.frame.get())
    # print("frameeeeeminn", np.min(frame_input.frame.get()))
    # print("frameeeeemaaxx", np.max(frame_input.frame.get()))
    # print('SALIENCY', saliency_map.saliency_map.get())
    # print("saliencyyyyyyminn", np.min(saliency_map.saliency_map.get()))
    # print("saliencyyyyyymaaxx", np.max(saliency_map.saliency_map.get()))
    # print('OUTPUT', output_map.output_map.get())

    frames.pop(0)
    count = 0
    for frame in frames:
        print(count)
        frame_input.frame.set(frame)
        saliency_map.run(condition=RunSteps(num_steps=3),
                         run_cfg=Loihi1SimCfg(select_tag='graded', select_sub_proc_model=True))
        sal_map = saliency_map.saliency_map.get()

        sm.append(sal_map)
        count = count + 1
        """if np.mod(count,20) == 0:
            print("saliencyyyyyyminn", np.min(sal_map))
            print("saliencyyyyyymaaxx", np.max(sal_map))
            plt.imshow(sal_map)
            plt.axis('off')
            plt.show()"""
        #om.append(output_map.output_map.get())

    saliency_map.stop()

    return sm, om


def CCOEFF(frames, template, **kwargs):
    convolution_type = kwargs.pop("convolution_type", "valid")
    template_shape = template.shape

    # template is normalized
    template_normalized = TemplateNormalization(template=template)
    template_normalized.run(condition=RunSteps(num_steps=2),
                            run_cfg=Loihi1SimCfg(select_tag='graded'))
    tem_nor = template_normalized.normalized_template.get().astype(np.int16)
    template_normalized.stop()

    # frame is given as input
    frame_input = FrameInput(frame=frames[0])
    frame_normalized = FrameNormalization(frame_shape=np.array(frames[0].shape),
                                          template_shape=template_shape)

    frame_input.s_out.connect(frame_normalized.a_in)

    saliency_map = TemplateMatching(template=tem_nor,
                                    frame_shape=frames[0].shape,
                                    convolution_type=convolution_type)

    # connect the input frame to the neuron population of same size
    frame_normalized.s_out.connect(saliency_map.a_in)

    # create the output DNF which is a SelectiveDNF
    sm_shape = tuple((frames[0].shape[0]-template.shape[0]+1, frames[0].shape[1]-template.shape[1]+1))
    """output_map = OutputDNF(sm_shape=sm_shape, vth=200, amp_exc=2, width_exc=[3,3], global_inh=-250)
    saliency_map.s_out.connect(output_map.a_in)"""

    fm = []
    sm = []
    om = []

    saliency_map.run(condition=RunSteps(num_steps=3),
                     run_cfg=Loihi1SimCfg(select_tag='graded', select_sub_proc_model=True))

    sal_map = saliency_map.saliency_map.get()
    sm.append(sal_map)
    """plt.imshow(sal_map)
    plt.axis('off')
    plt.show()"""
    #om.append(output_map.output_map.get())

    # print('NORM_TEMP', tem_nor)
    # print("max of temnor", np.max(tem_nor))
    # print('FRAME', frame_input.frame.get())
    # print('NORM_FRAME', frame_normalized.frame_normalized.get())
    # print('SALIENCY', saliency_map.saliency_map.get())
    # print('OUTPUT', output_map.output_map.get())

    #frames.pop(0)
    count = 0
    for frame in frames:
        frame_input.frame.set(frame)
        saliency_map.run(condition=RunSteps(num_steps=3),
                         run_cfg=Loihi1SimCfg(select_tag='graded', select_sub_proc_model=True))

        sal_map = saliency_map.saliency_map.get()
        sm.append(sal_map)
        #om.append(output_map.output_map.get())
        count = count + 1
        print(count)
        if np.mod(count, 500) == 1:
            """print("saliencyyyyyyminn", np.min(sal_map))
            print("saliencyyyyyymaaxx", np.max(sal_map))
            plt.imshow(sal_map)
            plt.axis('off')
            plt.show()"""

    saliency_map.stop()

    return sm, om