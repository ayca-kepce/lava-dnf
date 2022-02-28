# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

"""
Implement behaviors (process models) of the processes defined in processes.py
For further documentation please refer to processes.py
"""
import numpy as np
import cv2

from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.sub.model import AbstractSubProcessModel
from lava.proc.conv.process import Conv
from lava.lib.dnf.kernels.kernels import SelectiveKernel
from lava.lib.demos.object_tracking.processes import (TemplateMatching,
                                                      TemplateNormalization,
                                                      FrameInput,
                                                      FrameNormalization,
                                                      OutputDNF)
from lava.lib.demos.object_tracking.neurons.processes import one_input_neuron, two_input_neuron
from lava.lib.demos.object_tracking.util import template_resize



@implements(proc=FrameInput, protocol=LoihiProtocol)
@requires(CPU) #
class FrameInputPyModel(PyLoihiProcessModel):
    """PyLoihiProcessModel for FrameInputProcess.

    Receives the frame as an argument, validates, and sends its to Outport."""

    frame: np.ndarray = LavaPyType(np.ndarray, np.int16, precision=8)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int16, precision=8)


    def validate_frame(self):
        # check if the frame is a numpy array, convert if not
        if not np.ndarray == type(np.ndarray):
            self.frame = np.array(self.frame)

        # check if the frame is in gray scale, convert if not
        if len(self.frame.shape) == 3:
            if self.frame.shape[-1] == 3 or self.frame.shape[-1] == 4:
                self.frame = cv2.cvtColor(self.frame.astype('int16'), cv2.COLOR_BGR2GRAY)
            if self.frame.shape[-1] == 1:
                pass
            else:
                raise ValueError("Please provide the frame either RGB or grayscale.")
        elif len(self.frame.shape) > 3:
            raise ValueError("Please provide the frame either RGB or grayscale.")
        self.frame = np.reshape(self.frame.astype('int16'), (self.frame.shape[0], self.frame.shape[1], 1))


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

    template: np.ndarray = LavaPyType(np.ndarray, np.int16, precision=8)
    normalized_template: np.ndarray = LavaPyType(np.ndarray, np.int16, precision=8)

    def validate_template(self):
        # check if the template is a numpy array, convert if not
        if not np.ndarray == type(self.template):
            self.template = np.array(self.template)

        # check if the template is in gray scale, convert if not
        if len(self.template.shape) == 3:
            if self.template.shape[-1] == 3 or self.template.shape[-1] == 4:
                self.template = cv2.cvtColor(self.template.astype('int16'), cv2.COLOR_BGR2GRAY)
            else:
                raise ValueError("Please provide the template either RGB or grayscale.")
        elif len(self.template.shape) > 3:
            raise ValueError("Please provide the template either RGB or grayscale.")

    def run_spk(self):
        # check if the template is provided correct, convert grayscale if not
        self.validate_template()
        # normalize the template
        self.normalized_template = self.template - np.ones_like(self.template) * np.mean(self.template)
        self.normalized_template = np.sign(self.normalized_template) * np.right_shift(np.abs(self.normalized_template).astype(np.int32), 5)



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
        self.subtraction = two_input_neuron(shape=tuple((self.frame_shape[0], self.frame_shape[1], 1)),
                                            template_size=self.template_shape[0]*self.template_shape[1])
        proc.in_ports.a_in.connect(self.subtraction.in_ports.a_in1)

        # create the kernel to be convolved
        self.template= -np.ones(self.template_shape)
        self.template = template_resize(self.template)
        self.template = np.reshape(self.template, (1,self.template.shape[0], self.template.shape[1],1))

        # creating the convoluion to calculate the mean of each patch
        self.conv0 = Conv(input_shape=(self.frame_shape[0], self.frame_shape[1], 1),
                          weight=self.template,
                          padding=(int(self.template.shape[1] / 2), int(self.template.shape[2] / 2)))

        proc.in_ports.a_in.connect(self.conv0.in_ports.s_in)
        self.conv0.out_ports.a_out.connect(self.subtraction.a_in2)

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
        self.convolution_type = proc.init_args.get("convolution_type", 'valid')

        # template matching
        if self.convolution_type == "same":
            # zero pad the template if necessary. this is needed to calculate the shape of the padding in the convolution
            self.template = template_resize(self.template)
            self.template = np.reshape(self.template, (1, self.template.shape[0], self.template.shape[1], 1))
            # connect the result of the convolution to the neuron population of same size with input frame
            self.template_matching_population = one_input_neuron(shape=tuple((self.frame_shape[0], self.frame_shape[1], 1)))
            self.conv = Conv(input_shape=(self.frame_shape[0], self.frame_shape[1], 1),
                         weight=self.template, padding = (int(self.template.shape[1]/2), int(self.template.shape[2]/2)))
        elif self.convolution_type == "valid":
            # connect the result of the convolution to the neuron population of the size of the convolution result
            conv_shape = np.array(self.frame_shape) - np.array(self.template.shape) + 1
            self.template = np.reshape(self.template, (1, self.template.shape[0], self.template.shape[1], 1))
            self.template_matching_population = one_input_neuron(shape=tuple((conv_shape[0], conv_shape[1], 1)))
            self.conv = Conv(input_shape=(self.frame_shape[0], self.frame_shape[1], 1), weight=self.template)

        else:
            raise ValueError("The convolution type provided is unvalid. Choose either 'same' or 'valid'.")

        proc.in_ports.a_in.connect(self.conv.in_ports.s_in)
        self.conv.out_ports.a_out.connect(self.template_matching_population.a_in)
        proc.vars.saliency_map.alias(self.template_matching_population.vars.v)
        self.template_matching_population.out_ports.s_out.connect(proc.out_ports.s_out)


@implements(proc=OutputDNF, protocol=LoihiProtocol)
@requires(CPU)
@tag('graded')
class OutputDNFSubModel(AbstractSubProcessModel):
    """PyLoihiProcessModel for TemplateNormalizationProcess.

       Implements template matching algorithm."""

    def __init__(self, proc):
        self.sm_shape = proc.init_args.get("sm_shape")
        self.amp_exc = proc.init_args.get("amp_exc",5)
        self.width_exc = proc.init_args.get("width_exc", 4)
        self.global_inh = proc.init_args.get("global_inh", -5)
        self.vth = proc.init_args.get("vth", 5)

        # create the output DNF which is a SelectiveDNF
        self.output_population = one_input_neuron(shape=tuple((self.sm_shape[0], self.sm_shape[1], 1)), vth=self.vth)
        proc.in_ports.a_in.connect(self.output_population.in_ports.a_in)

        kernel = SelectiveKernel(amp_exc=self.amp_exc, width_exc=self.width_exc, global_inh=self.global_inh)
        self.template = template_resize(kernel.weights)
        self.template = np.reshape(self.template, (1, self.template.shape[0], self.template.shape[1], 1))
        self.conv2 = Conv(input_shape=(self.sm_shape[0], self.sm_shape[1], 1),
                          weight=self.template,
                          padding = (int(self.template.shape[1]/2), int(self.template.shape[2]/2)))
        self.output_population.s_out.connect(self.conv2.in_ports.s_in)
        self.conv2.out_ports.a_out.connect(self.output_population.a_in)

        self.monitor_population = one_input_neuron(shape=tuple((self.sm_shape[0], self.sm_shape[1], 1)), vth=0)
        self.output_population.out_ports.s_out.connect(self.monitor_population.in_ports.a_in)
        proc.vars.output_map.alias(self.monitor_population.vars.v)
