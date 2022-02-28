# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

"""
Creates processes for template matching. Behaviors of these processes are
defined in models.py
For further documentation please refer to models.py
"""

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort
import numpy as np
import typing as ty


class FrameInput(AbstractProcess):
    """ Receives the frame as an argument, validates, and sends its to Outport.

        Intializes the FrameInput.

        Parameters
        ------
        frame : 2D or 3D array
            Frame used in the template matching, can be RGB or grayscale
    """

    def __init__(self, **kwargs: ty.Any):
        super().__init__(**kwargs)
        frame = kwargs.get("frame")
        if len(frame.shape) > 2:
            frame_shape = np.array([frame.shape[0], frame.shape[1]])
        else:
            frame_shape = frame.shape
        self.frame = Var(shape=tuple(frame_shape), init=frame)
        self.s_out = OutPort(shape=tuple((frame_shape[0], frame_shape[1], 1)))


class TemplateNormalization(AbstractProcess):
    """ Receives the frame as an argument, validates, and sends its to Outport.

        Intializes the TemplateNormalization.

        Parameters
        ------
        template : 2D or 3D array
            Template to be found in the frame, can be RGB or grayscale
        """

    def __init__(self, **kwargs: ty.Any):
        super().__init__(**kwargs)
        template = kwargs.get("template")
        self.template = Var(shape=tuple(template.shape), init=template)
        if len(template.shape) > 2:
            template_shape = np.array([template.shape[0], template.shape[1]])
        else:
            template_shape = template.shape
        self.normalized_template = Var(shape=tuple(template_shape), init=np.zeros(template_shape))


class FrameNormalization(AbstractProcess):
    """Normalizes the frame input.

       Intialize the FrameNormalizationProcess

        Parameters
        ------
        frame_shape : ndarray
             Shape of the frame, used while defining the size of the InPorts and OutPorts
        """

    def __init__(self, **kwargs: ty.Any):
        super().__init__(**kwargs)
        frame_shape = kwargs.get("frame_shape")
        self.a_in = InPort(shape=tuple((frame_shape[0], frame_shape[1], 1)))
        self.s_out = OutPort(shape=tuple((frame_shape[0], frame_shape[1], 1)))
        self.frame_normalized = Var(shape=tuple((frame_shape[0], frame_shape[1], 1)), init=np.zeros((frame_shape[0], frame_shape[1], 1)))


class TemplateMatching(AbstractProcess):
    """Executes the template matching algorithm.

    intialize the TemplateMatchingProcess

        Parameters
        ------
        frame_shape : ndarray
            Shape of the frame, used while defining the size of the InPorts and OutPorts

    """

    def __init__(self, **kwargs: ty.Any):
        super().__init__(**kwargs)
        frame_shape = kwargs.get("frame_shape")
        template = kwargs.get("template")
        template_reshaped = np.reshape(kwargs.get("template"), (1, kwargs.get("template").shape[0], kwargs.get("template").shape[1], 1))
        conv_type = kwargs.get("convolution_type", 'valid')
        if conv_type == 'same':
            conv_shape = (frame_shape[0], frame_shape[1], 1)
        elif conv_type == 'valid':
            conv_shape = np.array(frame_shape) - np.array(template.shape) + 1
            conv_shape = (conv_shape[0], conv_shape[1], 1)
        else:
            print("The convolution type provided is unvalid. Choose either 'same' or 'valid'.")

        self.saliency_map = Var(shape=tuple(conv_shape), init=np.ones(conv_shape))
        self.a_in = InPort(shape=tuple((frame_shape[0], frame_shape[1], 1)))
        self.a_in_recv = Var(shape=tuple(frame_shape), init=np.ones(frame_shape))
        self.s_out = OutPort(shape=tuple(conv_shape))


class OutputDNF(AbstractProcess):
    """Executes the template matching algorithm.

    intialize the TemplateMatchingProcess

        Parameters
        ------
        frame_shape : ndarray
            Shape of the frame, used while defining the size of the InPorts and OutPorts

    """

    def __init__(self, **kwargs: ty.Any):
        super().__init__(**kwargs)
        sm_shape = kwargs.get("sm_shape")
        self.output_map = Var(shape=tuple((sm_shape[0], sm_shape[1], 1)), init=np.ones((sm_shape[0], sm_shape[1], 1)))
        self.a_in = InPort(shape=tuple((sm_shape[0], sm_shape[1], 1)))
