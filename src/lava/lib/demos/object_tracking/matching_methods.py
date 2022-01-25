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
from lava.lib.demos.object_tracking.neurons.processes import one_input_neuron, two_input_neuron, two_input_neuron_squares
from lava.lib.demos.object_tracking.util import grayscale_conversion, scale_image, draw_rectangule, normalize_image

def SQDIFF(frame, template ):
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

def CCORR(frames, template, scale_factor):

    # hold rgb frame and template
    frames_orig = frames
    template_orig = template
    # check if the frame and the template are provided in grayscale, convert if not
    frames = grayscale_conversion(frames)
    template = grayscale_conversion([template])
    # downsample the frame and the template in order to overcome the memory issue
    frames = scale_image(frames, scale_factor)
    template = scale_image(template, scale_factor)

    """template = []
    template.append(templates)"""

    # frame is given as input
    frame_input = FrameInput(frame=frames[0])

    saliency_map = TemplateMatching(template=template[0],
                                    frame_shape=frames[0].shape)

    # connect the input frame to the neuron population of same size
    frame_input.s_out.connect(saliency_map.a_in)

    # create the output DNF which is a SelectiveDNF
    """output_map = OutputDNF(frame_shape=np.array(frames[0].shape), vth=200, amp_exc=2, width_exc=[3,3], global_inh=-250)
    saliency_map.s_out.connect(output_map.a_in)"""

    """monitor_FN = Monitor()
    monitor_FN.probe(target=frame_normalized.s_out, num_steps=8)

    monitor_TM = Monitor()
    monitor_TM.probe(target=saliency_map.a_in, num_steps=8)"""
    fm = []
    sm = []
    om = []

    saliency_map.run(condition=RunSteps(num_steps=3),
                     run_cfg=Loihi1SimCfg(select_tag='graded', select_sub_proc_model=True))

    sm.append(saliency_map.saliency_map.get())
    # om.append(output_map.output_map.get())
    print('FRAME', frame_input.frame.get())
    print('SALIENCY', saliency_map.saliency_map.get())
    # print('OUTPUT', output_map.output_map.get())

    plt.imshow(saliency_map.saliency_map.get())
    plt.show()
    frames.pop(0)

    """for frame in frames:
        frame_input.frame.set(frame)
        saliency_map.run(condition=RunSteps(num_steps=3),
                         run_cfg=Loihi1SimCfg(select_tag='graded', select_sub_proc_model=True))

        sm.append(saliency_map.saliency_map.get())
        om.append(output_map.output_map.get())

        print('NORM_FRAME', frame_normalized.frame_normalized.get())
        print('SALIENCY', saliency_map.saliency_map.get())
        print('OUTPUT', output_map.output_map.get())

        plt.imshow(output_map.output_map.get())
        plt.show()"""

    saliency_map.stop()

    sm = scale_image(sm, 1 / scale_factor)
    # om = scale_image(om, 1/scale_factor)
    #fm = draw_rectangule(frames_orig, sm, template_orig.shape)

    """
    plt.imshow(sm)
    plt.show()

    plt.imshow(fm_rect)
    plt.show()"""

    return sm, om #, fm

def CCOEFF(frames, template, scale_factor):
    # hold rgb frame and template
    frames_orig = frames
    template_orig = template
    # check if the frame and the template are provided in grayscale, convert if not
    frames = grayscale_conversion(frames)
    template = grayscale_conversion([template])
    # downsample the frame and the template in order to overcome the memory issue
    frames = scale_image(frames, scale_factor)
    template = scale_image(template, scale_factor)

    # template is normalized
    template_normalized = TemplateNormalization(template=template[0])
    template_normalized.run(condition=RunSteps(num_steps=2),
                            run_cfg=Loihi1SimCfg(select_tag='graded'))
    tem_nor = template_normalized.normalized_template.get()
    template_normalized.stop()
    # frame is given as input
    frame_input = FrameInput(frame=frames[0])

    frame_normalized = FrameNormalization(frame_shape=np.array(frames[0].shape),
                                          template_shape=np.array(template[0].shape))

    frame_input.s_out.connect(frame_normalized.a_in)

    saliency_map = TemplateMatching(template=tem_nor,
                                    frame_shape=frames[0].shape)

    # connect the input frame to the neuron population of same size
    frame_normalized.s_out.connect(saliency_map.a_in)

    # create the output DNF which is a SelectiveDNF
    """output_map = OutputDNF(frame_shape=np.array(frames[0].shape), vth=200, amp_exc=2, width_exc=[3,3], global_inh=-250)
    saliency_map.s_out.connect(output_map.a_in)"""

    
    """monitor_FN = Monitor()
    monitor_FN.probe(target=frame_normalized.s_out, num_steps=8)

    monitor_TM = Monitor()
    monitor_TM.probe(target=saliency_map.a_in, num_steps=8)"""
    fm = []
    sm = []
    om = []

    saliency_map.run(condition=RunSteps(num_steps=3),
                         run_cfg=Loihi1SimCfg(select_tag='graded', select_sub_proc_model=True))

    sm.append(saliency_map.saliency_map.get())
    #om.append(output_map.output_map.get())
    print('NORM_TEMP', tem_nor)
    #print("max of temnor", np.max(tem_nor))
    print('FRAME', frame_input.frame.get())
    print('NORM_FRAME', frame_normalized.frame_normalized.get())
    print('SALIENCY', saliency_map.saliency_map.get())
    #print('OUTPUT', output_map.output_map.get())

    plt.imshow(saliency_map.saliency_map.get())
    plt.show()
    frames.pop(0)

    """for frame in frames:
        frame_input.frame.set(frame)
        saliency_map.run(condition=RunSteps(num_steps=3),
                         run_cfg=Loihi1SimCfg(select_tag='graded', select_sub_proc_model=True))

        sm.append(saliency_map.saliency_map.get())
        om.append(output_map.output_map.get())

        print('NORM_FRAME', frame_normalized.frame_normalized.get())
        print('SALIENCY', saliency_map.saliency_map.get())
        print('OUTPUT', output_map.output_map.get())

        plt.imshow(output_map.output_map.get())
        plt.show()"""

    #data_FN = monitor_FN.get_data()[frame_normalized.name][frame_normalized.s_out.name]
    #data_TM = monitor_TM.get_data()[saliency_map.name][saliency_map.a_in.name]

    #print("FNNNNNNN", data_FN)
    #print("TMMMMMMM", data_TM)
    saliency_map.stop()

    sm = scale_image(sm, 1/scale_factor)
    #om = scale_image(om, 1/scale_factor)
    fm = draw_rectangule(frames_orig, sm, template_orig.shape)

    """
    plt.imshow(sm)
    plt.show()

    plt.imshow(fm_rect)
    plt.show()"""

    return sm, om, fm