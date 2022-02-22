# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import matplotlib.pyplot as plt
import numpy as np
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.lib.demos.object_tracking.neurons.processes import (one_input_neuron,
                                                              two_input_neuron,
                                                              two_input_neuron_squares)

@implements(proc=one_input_neuron, protocol=LoihiProtocol)
@requires(CPU)
@tag('graded')
class PyLifModel_one_input_neuron(PyLoihiProcessModel):

    a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int32, precision=24)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE,np.int32, precision=24)
    v: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
    vth: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)

    def __init__(self):
        super(PyLifModel_one_input_neuron, self).__init__()
        # Let's define some bit-widths from Loihi
        # State variables u and v are 24-bits wide
        self.uv_bitwidth = 24
        self.max_uv_val = 2 ** (self.uv_bitwidth - 1)
        # Threshold and incoming activation are MSB-aligned using 6-bits
        self.timestep_counter = 0

    def run_spk(self):
        self.timestep_counter = self.timestep_counter + 1
        #print("timestep counter", self.timestep_counter)

        # Receive synaptic input
        a_in_data = self.a_in.recv()
        """print("a_in_data min",np.min(a_in_data), "\n")
        print("a_in_data max",np.max(a_in_data), "\n")"""

        # Update voltage
        # --------------
        neg_voltage_limit = -np.int32(self.max_uv_val) + 1
        pos_voltage_limit = np.int32(self.max_uv_val) - 1

        self.v[:] = np.clip(a_in_data, neg_voltage_limit, pos_voltage_limit)

        #print("a_in_data min after clipped", np.min(self.v), "\n")
        #print("a_in_data max after clipped", np.max(self.v), "\n")

        # send binary spikes if the threshold is exceeded
        s_out = np.left_shift(self.v >= self.vth, 6)
        self.s_out.send(s_out)


@implements(proc=two_input_neuron, protocol=LoihiProtocol)
@requires(CPU)
@tag('graded')
class PyLifModel_two_input_neuron(PyLoihiProcessModel):
    a_in1: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int32, precision=16)
    a_in2: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int32, precision=24)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE,np.int32, precision=24)
    v: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
    template_size: np.ndarray = LavaPyType(np.ndarray, np.int16)

    def __init__(self):
        super(PyLifModel_two_input_neuron, self).__init__()
        # Let's define some bit-widths from Loihi
        # State variables u and v are 24-bits wide
        self.uv_bitwidth = 24
        self.max_uv_val = 2 ** (self.uv_bitwidth - 1)
        # Threshold and incoming activation are MSB-aligned using 6-bits
        self.act_shift = 0
        self.timestep_counter = 0

    def run_spk(self):
        self.timestep_counter = self.timestep_counter + 1
        #print("timestep counter", self.timestep_counter)
        # Receive synaptic input
        a_in1_data = self.a_in1.recv()
        a_in2_data = self.a_in2.recv()
        #print('ain data 1 in two input neuron', a_in1_data, '\n')

        # Necessary shift to make frame normalization in the right scale
        x = np.log2(self.template_size).astype(int)
        a_in2_data = np.sign(a_in2_data) * np.right_shift(np.abs(a_in2_data), x)
        #a_in2_data = a_in2_data/self.template_size
        #print('ain data 2 in two input neuron', a_in2_data, '\n')

        a_in_data = a_in1_data + a_in2_data
        #print('normalized frame\n', a_in_data)
        neg_voltage_limit = -np.int32(self.max_uv_val) + 1
        pos_voltage_limit = np.int32(self.max_uv_val) - 1

        self.v[:] = np.clip(a_in_data, neg_voltage_limit, pos_voltage_limit)

        #print("v2", self.v, "\n")
        #print("s_out2", s_out, "\n")
        self.s_out.send(self.v)


@implements(proc=two_input_neuron_squares, protocol=LoihiProtocol)
@requires(CPU)
@tag('graded')
class PyLifModel_two_input_neuron_squares(PyLoihiProcessModel):
    a_in1: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int32, precision=16)
    a_in2: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int32, precision=16)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE,np.int32, precision=16)
    v: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)

    def __init__(self):
        super(PyLifModel_two_input_neuron, self).__init__()
        # Let's define some bit-widths from Loihi
        # State variables u and v are 24-bits wide
        self.uv_bitwidth = 24
        self.max_uv_val = 2 ** (self.uv_bitwidth - 1)
        # Threshold and incoming activation are MSB-aligned using 6-bits
        self.act_shift = 0
        self.timestep_counter = 0

    def run_spk(self):
        self.timestep_counter = self.timestep_counter + 1
        print("timestep counter", self.timestep_counter)
        # Receive synaptic input
        a_in1_data = self.a_in1.recv()
        a_in2_data = self.a_in2.recv()

        # Hardware left-shifts synpatic input for MSB alignment
        a_in1_data = np.left_shift(a_in1_data, self.act_shift)

        a_in_data = a_in1_data + a_in2_data

        neg_voltage_limit = -np.int32(self.max_uv_val) + 1
        pos_voltage_limit = np.int32(self.max_uv_val) - 1

        self.v[:] = np.clip(a_in_data, neg_voltage_limit, pos_voltage_limit)

        #print("v2", self.v, "\n")

        s_out = self.v*self.v
        #print("s_out2", s_out, "\n")
        self.s_out.send(s_out)