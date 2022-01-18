# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort


class one_input_neuron(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (1,))

        self.shape = shape
        self.a_in = InPort(shape=shape)
        self.s_out = OutPort(shape=shape)
        self.v = Var(shape=shape, init=0)


class two_input_neuron(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (1,))
        template_size = kwargs.get("template_size")

        self.shape = shape
        self.a_in1 = InPort(shape=shape)
        self.a_in2 = InPort(shape=shape)
        self.s_out = OutPort(shape=shape)
        self.v = Var(shape=shape, init=0)
        self.template_size = Var(shape=tuple((1,)), init=template_size)

class two_input_neuron_squares(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (1,))

        self.shape = shape
        self.a_in1 = InPort(shape=shape)
        self.a_in2 = InPort(shape=shape)
        self.s_out = OutPort(shape=shape)
        self.v = Var(shape=shape, init=0)
