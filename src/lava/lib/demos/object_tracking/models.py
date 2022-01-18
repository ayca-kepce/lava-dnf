# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

"""
Implement behaviors (process models) of the processes defined in processes.py
For further documentation please refer to processes.py
"""
import numpy as np
import cv2
import time

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
                                                      FrameNormalization)
from lava.lib.demos.object_tracking.neurons.processes import one_input_neuron, two_input_neuron
from lava.lib.demos.object_tracking.util import determine_scale_factor


@implements(proc=FrameInput, protocol=LoihiProtocol)
@requires(CPU) #
class FrameInputPyModel(PyLoihiProcessModel):
    """PyLoihiProcessModel for FrameInputProcess.

    Receives the frame as an argument, validates, and sends its to Outport."""

    frame: np.ndarray = LavaPyType(np.ndarray, np.float64)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32, precision=24)


    def validate_frame(self):
        # check if the frame is a numpy array, convert if not
        if not np.ndarray == type(np.ndarray):
            self.frame = np.array(self.frame)

        # check if the frame is in gray scale, convert if not
        if len(self.frame.shape) == 3:
            if self.frame.shape[-1] == 3 or self.frame.shape[-1] == 4:
                self.frame = cv2.cvtColor(self.frame.astype('float32'), cv2.COLOR_BGR2GRAY)
            else:
                raise ValueError("Please provide the frame either RGB or grayscale.")
        elif len(self.frame.shape) > 3:
            raise ValueError("Please provide the frame either RGB or grayscale.")

    def run_spk(self):
        # check if the frame is provided correct, convert grayscale if not
        self.validate_frame()

        # send the normalized template through the PyOutPort
        self.s_out.send(self.frame)


@implements(proc=TemplateNormalization, protocol=LoihiProtocol)
@requires(CPU) # LMT
class TemplateNormalizationPyModel(PyLoihiProcessModel):
    """PyLoihiProcessModel for TemplateNormalizationProcess.

    Normalizes the template."""

    template: np.ndarray = LavaPyType(np.ndarray, np.int32)
    normalized_template: np.ndarray = LavaPyType(np.ndarray, np.int32)

    def validate_template(self):
        # check if the template is a numpy array, convert if not
        if not np.ndarray == type(self.template):
            self.template = np.array(self.template)

        # check if the template is in gray scale, convert if not
        if len(self.template.shape) == 3:
            if self.template.shape[-1] == 3 or self.template.shape[-1] == 4:
                self.template = cv2.cvtColor(self.template.astype('float32'), cv2.COLOR_BGR2GRAY)
            else:
                raise ValueError("Please provide the template either RGB or grayscale.")
        elif len(self.template.shape) > 3:
            raise ValueError("Please provide the template either RGB or grayscale.")

    def run_spk(self):
        # check if the template is provided correct, convert grayscale if not
        self.validate_template()
        # normalize the template
        self.normalized_template = self.template - np.ones_like(self.template) * np.mean(self.template)


@implements(proc=FrameNormalization, protocol=LoihiProtocol)
@requires(CPU)
@tag('graded')
class FrameNormalizationSubModel(AbstractSubProcessModel):
    """PyLoihiProcessModel for FrameNormalizationProcess.

       Normalizes the frame."""

    def __init__(self, proc):
        self.frame_shape = proc.init_args.get("frame_shape")
        self.template_shape = proc.init_args.get("template_shape")

        # neuron population to subtract the mean from frame input
        self.subtraction = two_input_neuron(shape=tuple(self.frame_shape), template_size=self.template_shape[0]*self.template_shape[1])
        proc.in_ports.a_in.connect(self.subtraction.in_ports.a_in1)

        # create the kernel to be convolved
        kernel = ConvolutionKernel(template= -np.ones(self.template_shape))

        # creating the convoluion to calculate the mean of each patch
        conn, sp1, sp2 = connect(proc.in_ports.a_in, self.subtraction.a_in2,
                                 ops=[Convolution(kernel=kernel)], sign_mode=3)

        self.conn = conn
        self.sp1 = sp1
        self.sp2 = sp2
        proc.vars.frame_normalized.alias(self.subtraction.vars.v)
        self.subtraction.out_ports.s_out.connect(proc.out_ports.s_out)


@implements(proc=TemplateMatching, protocol=LoihiProtocol)
@requires(CPU)
@tag('graded')
class TemplateMatchingSubModel(AbstractSubProcessModel):
    """PyLoihiProcessModel for TemplateNormalizationProcess.

       Implements template matching algorithm."""

    def __init__(self, proc):
        self.frame_shape = proc.init_args.get("frame_shape")
        self.template = proc.init_args.get("template")

        # connect the input frame to the neuron population of same size
        self.template_matching_population = one_input_neuron(shape=tuple(self.frame_shape))

        # template matching
        kernel = ConvolutionKernel(template=self.template)
        conn3, sp13, sp23 = connect(proc.in_ports.a_in, self.template_matching_population.a_in,
                                    ops=[Convolution(kernel=kernel)], sign_mode=1)
        self.conn3 = conn3
        self.sp13 = sp13
        self.sp23 = sp23
        proc.vars.saliency_map.alias(self.template_matching_population.vars.v)
        self.template_matching_population.out_ports.s_out.connect(proc.out_ports.s_out)