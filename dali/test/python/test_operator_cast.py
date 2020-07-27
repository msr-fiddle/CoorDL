# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import nvidia.dali as dali
from nvidia.dali.backend_impl import TensorListGPU
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

from test_utils import check_batch
from test_utils import compare_pipelines
from test_utils import RandomDataIterator
from test_utils import get_dali_extra_path

class CastPipeline(Pipeline):
    def __init__(self, device, batch_size, iterator, cast_dtypes, num_threads=1, device_id=0):
        super(CastPipeline, self).__init__(batch_size, num_threads, device_id)
        self.layout = "HWC"
        self.device = device
        self.iterator = iterator
        self.inputs = ops.ExternalSource()
        self.cast = [ops.Cast(device=device, dtype=dtype) for dtype in cast_dtypes]

    def define_graph(self):
        self.data = self.inputs()
        out = self.data.gpu() if self.device == 'gpu' else self.data
        for k in range(len(self.cast)):
            out = self.cast[k](out)
        return out

    def iter_setup(self):
        data = self.iterator.next()
        self.feed_input(self.data, data, layout=self.layout)

def check_cast_operator_float16(device, batch_size):
    input_shape=(300, 400, 3)
    eii1 = RandomDataIterator(batch_size, shape=input_shape)
    eii2 = RandomDataIterator(batch_size, shape=input_shape)
    compare_pipelines(
        CastPipeline(device, batch_size, iter(eii1), [types.FLOAT16, types.FLOAT]),
        CastPipeline(device, batch_size, iter(eii2), [types.FLOAT]),
        batch_size=batch_size, N_iterations=5)

def test_cast_operator_float16():
    for device in ['cpu', 'gpu']:
        for batch_size in [3]:
            yield check_cast_operator_float16, device, batch_size
