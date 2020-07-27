# Copyright (c) 2017-2019, NVIDIA CORPORATION. All rights reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
from nvidia.dali import types
import torch
import torch.utils.dlpack as torch_dlpack
import ctypes
import logging
import functools
from nvidia.dali.backend import TensorListGPU 
import math
import numpy as np
import threading
#import Queue
from torch._six import queue, container_abcs, string_classes
import multiprocessing 
import os
import pin_util
import batch_util
import sys
import time

to_torch_type = {
    np.dtype(np.float32) : torch.float32,
    np.dtype(np.float64) : torch.float64,
    np.dtype(np.float16) : torch.float16,
    np.dtype(np.uint8)   : torch.uint8,
    np.dtype(np.int8)    : torch.int8,
    np.dtype(np.int16)   : torch.int16,
    np.dtype(np.int32)   : torch.int32,
    np.dtype(np.int64)   : torch.int64
}



def feed_ndarray(dali_tensor, arr):
    """
    Copy contents of DALI tensor to PyTorch's Tensor.

    Parameters
    ----------
    `dali_tensor` : nvidia.dali.backend.TensorCPU or nvidia.dali.backend.TensorGPU
                    Tensor from which to copy
    `arr` : torch.Tensor
            Destination of the copy
    """
    assert dali_tensor.shape() == list(arr.size()), \
            ("Shapes do not match: DALI tensor has size {0}"
            ", but PyTorch Tensor has size {1}".format(dali_tensor.shape(), list(arr.size())))
    #turn raw int to a c void pointer
    c_type_pointer = ctypes.c_void_p(arr.data_ptr())
    dali_tensor.copy_to_external(c_type_pointer)
    return arr

class DALISupportingClassificationIterator(object):
    """
    General DALI iterator for PyTorch. It can return any number of
    outputs from the DALI pipeline in the form of PyTorch's Tensors.

    Please keep in mind that Tensors returned by the iterator are
    still owned by DALI. They are valid till the next iterator call.
    If the content needs to be preserved please copy it to another tensor.

    Parameters
    ----------
    pipelines : list of nvidia.dali.pipeline.Pipeline
                List of pipelines to use
    output_map : list of str
                 List of strings which maps consecutive outputs
                 of DALI pipelines to user specified name.
                 Outputs will be returned from iterator as dictionary
                 of those names.
                 Each name should be distinct
    size : int
           Number of samples in the epoch (Usually the size of the dataset).
           Providing -1 means that the iterator will work until StopIteration is raised
           from the inside of iter_setup(). The options `fill_last_batch`, `last_batch_padded` and
           `auto_reset` don't work in such case. It works with only one pipeline inside
           the iterator.
    auto_reset : bool, optional, default = False
                 Whether the iterator resets itself for the next epoch
                 or it requires reset() to be called separately.
    fill_last_batch : bool, optional, default = True
                 Whether to fill the last batch with data up to 'self.batch_size'.
                 The iterator would return the first integer multiple
                 of self._num_gpus * self.batch_size entries which exceeds 'size'.
                 Setting this flag to False will cause the iterator to return
                 exactly 'size' entries.
    dynamic_shape: bool, optional, default = False
                 Whether the shape of the output of the DALI pipeline can
                 change during execution. If True, the pytorch tensor will be resized accordingly
                 if the shape of DALI returned tensors changes during execution.
                 If False, the iterator will fail in case of change.
    last_batch_padded : bool, optional, default = False
                 Whether the last batch provided by DALI is padded with the last sample
                 or it just wraps up. In the conjunction with `fill_last_batch` it tells
                 if the iterator returning last batch with data only partially filled with
                 data from the current epoch is dropping padding samples or samples from
                 the next epoch. If set to False next epoch will end sooner as data from
                 it was consumed but dropped. If set to True next epoch would be the
                 same length as the first one. For this happen, the option `pad_last_batch`
                 in the reader need to be set to `True` as well.

    Example
    -------
    With the data set ``[1,2,3,4,5,6,7]`` and the batch size 2:

    fill_last_batch = False, last_batch_padded = True  -> last batch = ``[7]``, next iteration will return ``[1, 2]``

    fill_last_batch = False, last_batch_padded = False -> last batch = ``[7]``, next iteration will return ``[2, 3]``

    fill_last_batch = True, last_batch_padded = True   -> last batch = ``[7, 7]``, next iteration will return ``[1, 2]``

    fill_last_batch = True, last_batch_padded = False  -> last batch = ``[7, 1]``, next iteration will return ``[2, 3]``
    """
    def __init__(self,
                 pipelines,
                 size,
                 global_size,
                 output_map = ["data", "label"],
                 auto_reset=False,
                 fill_last_batch=True,
                 dynamic_shape=False,
                 last_batch_padded = False,
                 batch_maps=[],
                 counter_maps=[],
                 locks=[],
                 iter_id=0,
                 global_batch_required=0,
                 must_pin=True,
                 dead_proc_q=None,
                 dead_lock= None):
        if not isinstance(pipelines, list):
            pipelines = [pipelines]
        self._num_gpus = len(pipelines)
        print("INIT batch map len = {}".format(len(batch_maps)))
        assert pipelines is not None, "Number of provided pipelines has to be at least 1"
        self.batch_size = pipelines[0].batch_size
        self._size = int(size)
        self.pin_memory_thread = None
        self.batch_prep_thread = None
        self._global_size = int(global_size)
        self._iter_id = iter_id
        self._auto_reset = auto_reset
        self._dynamic_shape = dynamic_shape
        self._fill_last_batch = fill_last_batch
        self._last_batch_padded = last_batch_padded
        self._dl_to_fetch_from = self._iter_id
        self._outstanding_batches = 0
        self._must_pin = must_pin
        self._threads = []
        self.batch_maps = batch_maps
        self.counter_maps = counter_maps
        self.locks = locks
        self.done_event = multiprocessing.Event()
        self.dead_map = dead_proc_q
        self.dead_lock = dead_lock
        #self._dl_to_fetch_from = self._iter_id
        self._total_dl = len(batch_maps)  #Count from 1
        self._global_batch_id = global_batch_required 
        self._local_batch_id = int(global_batch_required/self._total_dl) - 1
        self.my_last_global_batch = self._global_batch_id - self._total_dl
        print("last global batch = {}, lmy last local batch={}, my last global batch={}".format(self._global_batch_id, self._local_batch_id, self.my_last_global_batch))
        self.total_batches_required = math.ceil(self._global_size/self.batch_size)
        self.total_batches_self = math.ceil(self._size/self.batch_size)
        print("PIPELINE : Total Batches required = {}, this dl batches={}".format(self.total_batches_required, self.total_batches_self))
        print("PIPELINE - Num gpu={}, size={}, global_size={}, batch_size={}, iter ID={}, total_dl:{}".format(self._num_gpus, self._size, self._global_size, self.batch_size, self._iter_id, self._total_dl))
        assert self._size != 0, "Size cannot be 0"
        assert self._size > 0 or (self._size < 0 and len(pipelines) == 1), "Negative size is supported only for a single pipeline"
        if self._size < 0:
            self._auto_reset = False
            self._fill_last_batch = False
            self._last_batch_padded = False
        self._pipes = pipelines
        # Build all pipelines
        for p in self._pipes:
            with p._check_api_type_scope(types.PipelineAPIType.ITERATOR):
                p.build()
        # Use double-buffering of data batches
        self._data_batches = [None for i in range(self._num_gpus)]
        self._counter = self._local_batch_id * self.batch_size
        self._global_counter = self._global_batch_id * self.batch_size
        assert len(set(output_map)) == len(output_map), "output_map names should be distinct"
        self._output_categories = set(output_map)
        self.output_map = output_map

        # We need data about the batches (like shape information),
        # so we need to run a single batch as part of setup to get that info
        for p in self._pipes:
            with p._check_api_type_scope(types.PipelineAPIType.ITERATOR):
                p.schedule_run()

        self._last_batch_returned = 0
        self._pin_batch_counter = 0
        self._total_batches_other_dl = self.total_batches_self*(self._total_dl-1)
        self._total_batches = self.total_batches_self*self._total_dl
        print("TOTAL BATCHES={}".format(self._total_batches))
            


    def shutdown(self):
        self.done_event.set()
        return
           

    def __next__(self):

        print("[{}] Requested batch {} from {}".format(self._iter_id, self._global_batch_id, self._dl_to_fetch_from))
        #Reset this dataloader only when global batch count is reached
        if self._global_batch_id >= self.total_batches_self*self._total_dl:
            print("[{}] Requested RESET".format(self._iter_id))
            if self._auto_reset:
                self.reset()
            raise StopIteration
            #else:
            #    print("[{}]REMAINING : {}".format(self._iter_id, self._global_batch_id)) 

        #Do a roundrobin to fetch from different DLs
        #print("DL to fetch from:{} for batch:{}".format(self._dl_to_fetch_from, self._global_batch_id))

        if self._dl_to_fetch_from != self._iter_id:
            self.consume_from_map(self._dl_to_fetch_from, self._global_batch_id) 
            
            self._dl_to_fetch_from = (self._dl_to_fetch_from + 1) % (self._total_dl)
            self._global_counter += self._num_gpus * self.batch_size
            self._global_batch_id += 1
            #print("FETCH FROM OTHER DL :{}, batch:{}".format(self._dl_to_fetch_from, self._global_batch_id))
            
            #dead = self._iter_id
            return -1, None

        # If you need to fetch from your own DL
        # then go ahead and collect the batch from workers
        else:
            #print("FETCH FROM LOCAL DL :{}, batch:{}".format(self._dl_to_fetch_from, self._global_batch_id))
            print("[{}] Getting batch {}, prev batch = {}".format(self._iter_id, self._global_batch_id, self.my_last_global_batch))
            s = time.time()
            # if my last bacth hasn't been consumed, wait
            if self.my_last_global_batch != -1:
                while self.my_last_global_batch in self.counter_maps[self._iter_id]:
                    continue
            print("Wait = {}".format(time.time() - s))
            # Gather outputs
            outputs = []
            for p in self._pipes:
                with p._check_api_type_scope(types.PipelineAPIType.ITERATOR):
                   outputs.append(p.share_outputs())
            for i in range(self._num_gpus):
                dev_id = self._pipes[i].device_id
                # initialize dict for all output categories
                category_outputs = dict()
                # segregate outputs into categories
                for j, out in enumerate(outputs[i]):
                    category_outputs[self.output_map[j]] = out
                

                # Change DALI TensorLists into Tensors
                category_tensors = dict()
                category_shapes = dict()
                for category, out in category_outputs.items():
                    category_tensors[category] = out.as_tensor()
                    category_shapes[category] = category_tensors[category].shape()

                # If we did not yet allocate memory for that batch, do it now
                if self._data_batches[i] is None:
                    category_torch_type = dict()
                    category_device = dict()
                    torch_gpu_device = torch.device('cuda', dev_id)
                    torch_cpu_device = torch.device('cpu')
                    # check category and device
                    for category in self._output_categories:
                        category_torch_type[category] = to_torch_type[np.dtype(category_tensors[category].dtype())]
                        from nvidia.dali.backend import TensorGPU
                        if type(category_tensors[category]) is TensorGPU:
                            category_device[category] = torch_gpu_device
                        else:
                            category_device[category] = torch_cpu_device

                    pyt_tensors = dict()
                    for category in self._output_categories:
                        pyt_tensors[category] = torch.zeros(category_shapes[category],
                                                        dtype=category_torch_type[category],
                                                        device=category_device[category])

                    self._data_batches[i] = pyt_tensors
                else:
                    pyt_tensors = self._data_batches[i]

                # Copy data from DALI Tensors to torch tensors
                for category, tensor in category_tensors.items():
                      if self._dynamic_shape and tensor.shape() != list(pyt_tensors[category].size()):
                          pyt_tensors[category] = torch.zeros(category_shapes[category],
                                                          dtype=pyt_tensors[category].dtype,
                                                          device=pyt_tensors[category].device)
                      feed_ndarray(tensor, pyt_tensors[category])

            for p in self._pipes:
                 with p._check_api_type_scope(types.PipelineAPIType.ITERATOR):
                    p.release_outputs()
                    p.schedule_run()


            if (not self._fill_last_batch) and (self._counter > self._size) and self._size > 0:
                # First calculate how much data is required to return exactly self._size entries.
                diff = self._num_gpus * self.batch_size - (self._counter - self._size)
                # Figure out how many GPUs to grab from.
                numGPUs_tograb = int(np.ceil(diff/self.batch_size))
                # Figure out how many results to grab from the last GPU (as a fractional GPU batch may be required to
                # bring us right up to self._size).
                mod_diff = diff % self.batch_size
                data_fromlastGPU = mod_diff if mod_diff else self.batch_size

                # Grab the relevant data.
                # 1) Grab everything from the relevant GPUs.
                # 2) Grab the right data from the last GPU.
                # 3) Append data together correctly and return.
                output = self._data_batches[0:numGPUs_tograb]
                output[-1] = output[-1].copy()
                for category in self._output_categories:
                    output[-1][category] = output[-1][category][0:data_fromlastGPU]
                return output

            self._dl_to_fetch_from = (self._dl_to_fetch_from + 1) % (self._total_dl)
            #print("Returning Local Batch {}, samples done={}".format(self._global_batch_id, self._counter))

            # Add it to the appropriate batch map
            self._add_to_map(self._data_batches, self._global_batch_id)
            self.my_last_global_batch = self._global_batch_id
            self._counter += self._num_gpus * self.batch_size
            self._global_counter += self._num_gpus * self.batch_size 
            self._local_batch_id += 1
            self._global_batch_id += 1
            return True


    #Each map has a disjoint set of batch_ids
    #This need not block the current DL becuase it
    # can be pipelined with next fetch
    def add_to_map(self, batch, global_batch_id):
        t = threading.Thread(target=self._add_to_map, args=(batch, global_batch_id))
        self._threads.append(t)
        t.start()

    def _add_to_map(self, batch, global_batch_id):
        # Do a sequential add - add batch to batch map
        # Then update the counter map with a lock held
        #print("In add to batch")
        
        self.batch_maps[self._iter_id][global_batch_id] = batch
        #Set counter to the num total jobs - 1 because this job
        #will get it directly from the loader
        with self.locks[self._iter_id]:
            self.counter_maps[self._iter_id][global_batch_id] = self._total_dl - 1
        #map_c = self.counter_maps[self._iter_id]
        #for i in map_c.keys():
            #print("[{} PUT] Batch ID:{}\t Counter:{}".format(self._iter_id, i, map_c[i]))
            
    def consume_from_map(self, dl_id, global_batch_id):
        while global_batch_id not in self.counter_maps[dl_id]:
            continue
        with self.locks[dl_id]:
            current_counter = self.counter_maps[dl_id][global_batch_id]
            if current_counter > 1:
                self.counter_maps[dl_id][global_batch_id] -= 1
            elif current_counter == 1:   
                self.counter_maps[dl_id][global_batch_id] -= 1  
                del self.batch_maps[dl_id][global_batch_id]
                del self.counter_maps[dl_id][global_batch_id] 

    def next(self):
        """
        Returns the next batch of data.
        """
        return self.__next__()

    def __iter__(self):
        return self

    def reset(self):
        """
        Resets the iterator after the full epoch.
        DALI iterators do not support resetting before the end of the epoch
        and will ignore such request.
        """
        if self._counter >= self._size or self._size < 0:
            if self._fill_last_batch and not self._last_batch_padded:
                self._counter = self._counter % self._size
            else:
                print("[{}] SUCCESSFUL RESET BECAUSE : counter={}, size={}, local_batch={}".format(self._iter_id, self._counter, self._size, self._local_batch_id))
                print("[{}] SUCCESSFUL RESET BECAUSE : global_counter={}, global_size={}, global_batch={}".format(self._iter_id, self._global_counter, self._global_size, self._global_batch_id))

                for t in self._threads:
                    t.join()
                self._counter = 0
                self._global_counter = 0
                self._global_batch_id = 0
                self._local_batch_id = 0
                self._dl_to_fetch_from = 0
                self.my_last_global_batch = -1
                map_c = self.counter_maps[self._iter_id]
                print("[{}] Counter map length={}".format(self._iter_id, len(map_c.keys())))
                #for i in map_c.keys():
                #    print("[{}] : Not deleted {}:{}".format(self._iter_id, i, map_c[i])) 
                #Wait for the maps to be consumed by other jobs
                while len(map_c.keys()) > 0:
                    continue
            for p in self._pipes:
                p.reset()
            
        else:
            print("[{}] COULDNT RESET BECAUSE : counter={}, size={}, local-batch={}".format(self._iter_id, self._counter, self._size, self._local_batch_id))
            print("[{}] COULDNT RESET BECAUSE : global_counter={}, global_size={}, global-batch:{}".format(self._iter_id, self._global_counter, self._global_size, self._global_batch_id))
            logging.warning("DALI iterator does not support resetting while epoch is not finished. Ignoring...")



class TorchPythonFunction(ops.PythonFunctionBase):
    ops.register_cpu_op('TorchPythonFunction')
    ops.register_gpu_op('TorchPythonFunction')

    def _torch_stream_wrapper(self, function, *ins):
        with torch.cuda.stream(self.stream):
            out = function(*ins)
        self.stream.synchronize()
        return out

    def torch_wrapper(self, batch_processing, function, device, *args):
        func = function if device == 'cpu' else \
               lambda *ins: self._torch_stream_wrapper(function, *ins)
        if batch_processing:
            return ops.PythonFunction.function_wrapper_batch(func,
                                                             torch.utils.dlpack.from_dlpack,
                                                             torch.utils.dlpack.to_dlpack,
                                                             *args)
        else:
            return ops.PythonFunction.function_wrapper_per_sample(func,
                                                                  torch_dlpack.from_dlpack,
                                                                  torch_dlpack.to_dlpack,
                                                                  *args)

    def __call__(self, *inputs, **kwargs):
        if self.stream is None:
            self.stream = torch.cuda.Stream(device=Pipeline.current().device_id)
        return super(TorchPythonFunction, self).__call__(*inputs, **kwargs)

    def __init__(self, function, num_outputs=1, device='cpu', batch_processing=False, **kwargs):
        self.stream = None
        super(TorchPythonFunction, self).__init__(impl_name="DLTensorPythonFunctionImpl",
                                                  function=lambda *ins:
                                                  self.torch_wrapper(batch_processing,
                                                                    function, device,
                                                                    *ins),
                                                  num_outputs=num_outputs, device=device,
                                                  batch_processing=batch_processing, **kwargs)
