# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

"""
Implement behaviors (process models) of the processes defined in processes.py
For further documentation please refer to processes.py
"""
import numpy as np
import cv2 as cv
from lava.magma.core.run_configs import Loihi1SimCfg
from lava.magma.core.run_conditions import RunSteps
from lava.lib.demos.object_tracking.processes import (TemplateMatching,
                                                      TemplateNormalization,
                                                      FrameInput,
                                                      FrameNormalization,
                                                      OutputDNF)

from lava.lib.demos.object_tracking.util import scale_images


def CCORR(frames, template, **kwargs):
    convolution_type = kwargs.pop("convolution_type", "valid")
    original_template_size = kwargs.pop("original_template_size")
    conv_shape = kwargs.pop("conv_shape")
    save_saliency_maps = kwargs.pop("save_saliency_maps", False)

    # frame is given as input
    frame_input = FrameInput(frame=frames[0])

    saliency_map = TemplateMatching(template=template,
                                    frame_shape=frames[0].shape,
                                    convolution_type=convolution_type)

    # connect the input frame to the neuron population of same size
    frame_input.s_out.connect(saliency_map.a_in)

    # create the output DNF which is a SelectiveDNF
    """output_map = OutputDNF(sm_shape=np.array(frames[0].shape)-np.array(template.shape)+1, vth=200, amp_exc=2, width_exc=template.shape, global_inh=-250)
    saliency_map.s_out.connect(output_map.a_in)"""

    sm = []
    answers = []
    om = []

    saliency_map.run(condition=RunSteps(num_steps=3),
                     run_cfg=Loihi1SimCfg(select_tag='graded', select_sub_proc_model=True))

    sal_map = saliency_map.saliency_map.get()
    sal_map = scale_images([sal_map], dimension=(conv_shape[1], conv_shape[0]))[0]

    if save_saliency_maps:
        sm.append(sal_map)

    _, _, _, max_loc = cv.minMaxLoc(sal_map)
    if convolution_type == "valid":
        answers.append((np.array((max_loc[0], max_loc[1])) + (
            int((original_template_size[0] + 1) / 2), int((original_template_size[1] + 1) / 2))))
    elif convolution_type == "same":
        answers.append(max_loc)

    """om_map = output_map.output_map.get()
    om_map = scale_image([om_map], dimension=(conv_shape[1], conv_shape[0]))[0]
    _, _, _, max_loc = cv.minMaxLoc(om_map)
    om.append((np.array((max_loc[0], max_loc[1])) + (
    int((original_template_size[0] + 1) / 2), int((original_template_size[1] + 1) / 2))))"""

    frames.pop(0)
    count = 0
    for frame in frames:
        print("Frame number %d is being processed." %count)
        frame_input.frame.set(frame)
        saliency_map.run(condition=RunSteps(num_steps=3),
                         run_cfg=Loihi1SimCfg(select_tag='graded', select_sub_proc_model=True))
        sal_map = saliency_map.saliency_map.get()
        sal_map = scale_images([sal_map], dimension=(conv_shape[1], conv_shape[0]))[0]

        if save_saliency_maps:
            sm.append(sal_map)

        _, _, _, max_loc = cv.minMaxLoc(sal_map)
        if convolution_type == "valid":
            answers.append((np.array((max_loc[0], max_loc[1])) + (
            int((original_template_size[0] + 1) / 2), int((original_template_size[1] + 1) / 2))))
        elif convolution_type == "same":
            answers.append(max_loc)

        """om_map = output_map.output_map.get()
        om_map = scale_image([om_map], dimension=(conv_shape[1], conv_shape[0]))[0]
        _, _, _, max_loc = cv.minMaxLoc(om_map)
        om.append((np.array((max_loc[0], max_loc[1])) + (
        int((original_template_size[0] + 1) / 2), int((original_template_size[1] + 1) / 2))))"""

        count = count + 1

    saliency_map.stop()

    return sm, answers, om


def CCOEFF(frames, template, **kwargs):
    convolution_type = kwargs.pop("convolution_type", "valid")
    original_template_size = kwargs.pop("original_template_size")
    conv_shape = kwargs.pop("conv_shape")
    save_saliency_maps = kwargs.pop("save_saliency_maps", False)
    template_shape = template.shape

    # template is normalized
    template_normalized = TemplateNormalization(template=template)
    template_normalized.run(condition=RunSteps(num_steps=3),
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
    """output_map = OutputDNF(sm_shape=sm_shape, vth=200, amp_exc=2, width_exc=[3,3], global_inh=-250)
    saliency_map.s_out.connect(output_map.a_in)"""

    sm = []
    answers = []
    om = []

    saliency_map.run(condition=RunSteps(num_steps=3),
                     run_cfg=Loihi1SimCfg(select_tag='graded', select_sub_proc_model=True))

    sal_map = saliency_map.saliency_map.get()
    sal_map = scale_images([sal_map], dimension=(conv_shape[1], conv_shape[0]))[0]
    if save_saliency_maps:
        sm.append(sal_map)

    _, _, _, max_loc = cv.minMaxLoc(sal_map)
    if convolution_type == "valid":
        answers.append((np.array((max_loc[0], max_loc[1])) + (
            int((original_template_size[0] + 1) / 2), int((original_template_size[1] + 1) / 2))))
    elif convolution_type == "same":
        answers.append(max_loc)
    """om_map = output_map.output_map.get()
    om_map = scale_image([om_map], dimension=(conv_shape[1], conv_shape[0]))[0]
    _, _, _, max_loc = cv.minMaxLoc(om_map)
    om.append((np.array((max_loc[0], max_loc[1])) + (
    int((original_template_size[0] + 1) / 2), int((original_template_size[1] + 1) / 2))))"""

    frames.pop(0)
    count = 0
    for frame in frames:
        print("Frame number %d is being processed." %count)
        frame_input.frame.set(frame)
        saliency_map.run(condition=RunSteps(num_steps=3),
                         run_cfg=Loihi1SimCfg(select_tag='graded', select_sub_proc_model=True))

        sal_map = saliency_map.saliency_map.get()
        sal_map = scale_images([sal_map], dimension=(conv_shape[1], conv_shape[0]))[0]
        if save_saliency_maps:
            sm.append(sal_map)

        _, _, _, max_loc = cv.minMaxLoc(sal_map)
        if convolution_type == "valid":
            answers.append((np.array((max_loc[0], max_loc[1])) + (
                int((original_template_size[0] + 1) / 2), int((original_template_size[1] + 1) / 2))))
        elif convolution_type == "same":
            answers.append(max_loc)

        """om_map = output_map.output_map.get()
        om_map = scale_image([om_map], dimension=(conv_shape[1], conv_shape[0]))[0]
        _, _, _, max_loc = cv.minMaxLoc(om_map)
        om.append((np.array((max_loc[0], max_loc[1])) + (
        int((original_template_size[0] + 1) / 2), int((original_template_size[1] + 1) / 2))))"""
        count = count + 1


    saliency_map.stop()

    return sm, answers, om