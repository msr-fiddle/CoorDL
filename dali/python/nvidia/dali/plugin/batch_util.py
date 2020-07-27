r""""Contains definitions of the methods used by the _DataLoaderIter to put
fetched tensors into pinned memory.

These **needs** to be in global scope since Py2 doesn't support serializing
static methods.
"""

import torch
#import Queue
from torch._six import queue, container_abcs, string_classes
#from torch._utils import ExceptionWrapper
import time
import os
import threading
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
from nvidia.dali import types
import torch.utils.dlpack as torch_dlpack 
import ctypes 
from nvidia.dali.backend import TensorListGPU 
import functools 
import math 
import numpy as np 

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
   

def get_next_batch(shared_batch_maps, shared_counter_maps, shared_locks, out_queue, dl_id, batches_to_return, total_dl, done_event, pipes, num_gpus, output_map, data_batches, output_categories, dynamic_shape, fill_last_batch, _size, _global_size, batch_size, must_save, gpu_per_job):
    _outstanding_batches = 0
    _counter = 0
    _global_counter = 0
    _global_batch_id = 0
    _local_batch_id = 0
    dl_to_fetch_from = dl_id % gpu_per_job

    while not done_event.is_set():
        #print("[BATCH TH {}] Start prep batch {}".format(dl_id, _global_batch_id))
        # If there are more than 2 items in pipeline
        #while len(shared_counter_maps[dl_id]) > 2:
        #while len(shared_counter_maps[dl_id]) > 0:
        #while out_queue.qsize() > 2 or len(shared_counter_maps[dl_id]) > 2:
        #    continue

        #while out_queue.qsize() > 1:
        #    continue

        if _global_batch_id >= batches_to_return: 
            #Reset because epoch ended
            print("[{}] SUCCESSFUL RESET BECAUSE : global_counter={}, global_size={}, global_batch={}".format(dl_id, _global_counter, _global_size, _global_batch_id))
            _outstanding_batches = 0
            _counter = 0
            _global_counter = 0
            _global_batch_id = 0
            _local_batch_id = 0
            dl_to_fetch_from = dl_id % gpu_per_job
            map_c = shared_counter_maps[dl_id]
            while len(map_c.keys()) > 0:
                continue
            
            while out_queue.qsize() > 0:
                time.sleep(5)
                continue

            for p in pipes:
                p.reset()
                if p.empty():
                    with p._check_api_type_scope(types.PipelineAPIType.ITERATOR):
                        p.schedule_run() 
            print("[{}] ACTUALLY SUCCESSFUL".format(dl_id))
        
        if dl_to_fetch_from == dl_id:
           while len(shared_counter_maps[dl_id]) > 0:
                continue
           #print("[{}] : {}:{}".format(dl_id, len(shared_counter_maps[dl_id]), out_queue.qsize()))
           #print("[BATCH TH {}] prep batch {}".format(dl_id, _global_batch_id))
           outputs = []
           for p in pipes:
               with p._check_api_type_scope(types.PipelineAPIType.ITERATOR):
                   outputs.append(p.share_outputs())
           #print("[BATCH TH {}] batch {} : Share output".format(dl_id, _global_batch_id))
           for i in range(num_gpus):
               dev_id = pipes[i].device_id
               category_outputs = dict()
               for j, out in enumerate(outputs[i]):
                   category_outputs[output_map[j]] = out 

               # Change DALI TensorLists into Tensors
               category_tensors = dict()
               category_shapes = dict()
               for category, out in category_outputs.items(): 
                   category_tensors[category] = out.as_tensor() 
                   category_shapes[category] = category_tensors[category].shape()

               # If we did not yet allocate memory for that batch, do it now
               if data_batches[i] is None:
                   category_torch_type = dict() 
                   category_device = dict() 
                   torch_gpu_device = torch.device('cuda', dev_id) 
                   torch_cpu_device = torch.device('cpu') 
                   for category in output_categories: 
                       category_torch_type[category] = to_torch_type[np.dtype(category_tensors[category].dtype())] 
                       from nvidia.dali.backend import TensorGPU
                       if type(category_tensors[category]) is TensorGPU:
                           category_device[category] = torch_gpu_device
                       else:
                           category_device[category] = torch_cpu_device

                   pyt_tensors = dict()
                   for category in output_categories:
                        pyt_tensors[category] = torch.zeros(category_shapes[category],
                                               dtype=category_torch_type[category],
                                               device=category_device[category])
                   data_batches[i] = pyt_tensors
               else:
                   pyt_tensors = data_batches[i] 

               # Copy data from DALI Tensors to torch tensors    
               #print("[BATCH TH {}] batch {} : To Torch tensors".format(dl_id, _global_batch_id))
               for category, tensor in category_tensors.items():
                   if dynamic_shape and tensor.shape() != list(pyt_tensors[category].size()):  
                       pyt_tensors[category] = torch.zeros(category_shapes[category],
                                               dtype=pyt_tensors[category].dtype,
                                               device=pyt_tensors[category].device)
                   feed_ndarray(tensor, pyt_tensors[category]) 

           for p in pipes: 
               with p._check_api_type_scope(types.PipelineAPIType.ITERATOR):  
                   p.release_outputs()
                   p.schedule_run() 
           #print("[BATCH TH {}] batch {} : Outputs released".format(dl_id, _global_batch_id))

           if (not fill_last_batch) and (_counter > _size) and _size > 0:
               diff = num_gpus * batch_size - (_counter - _size)
               numGPUs_tograb = int(np.ceil(diff/batch_size))
               mod_diff = diff % batch_size
               data_fromlastGPU = mod_diff if mod_diff else batch_size 
               output = data_batches[0:numGPUs_tograb] 
               output[-1] = output[-1].copy() 
               for category in output_categories: 
                   output[-1][category] = output[-1][category][0:data_fromlastGPU]
               return output  
         
           dl_to_fetch_from = (dl_to_fetch_from + gpu_per_job) % (total_dl)
           #add_to_map(data_batches, _global_batch_id)
           #s = time.time() 
           shared_batch_maps[dl_id][_global_batch_id] = data_batches
           #print("Dur for batch {} = {}".format(_global_batch_id, time.time()-s))
           if must_save:
               print("Saving orig {}-{}".format(dl_id, _global_batch_id))
               fname='orig-'+str(dl_id) + '-' + str(_global_batch_id)+'.pt'
               torch.save(data_batches[0]["data"], fname)
           with shared_locks[dl_id]:
               #shared_counter_maps[dl_id][_global_batch_id] = total_dl 
               shared_counter_maps[dl_id][_global_batch_id] = total_dl - gpu_per_job 
           out_queue.put((_global_batch_id, dl_id, data_batches)) 
           #out_queue.put((_global_batch_id, data_batches), block=True) 
           #if dl_id == 0:
           #    print("[BATCH TH {}] put batch {} in shm map {}, val={}".format(dl_id, _global_batch_id, dl_id, total_dl - 1))
           _counter += num_gpus * batch_size
           _global_counter += num_gpus * batch_size
           _local_batch_id += gpu_per_job
           _global_batch_id += gpu_per_job

        else:
            while _global_batch_id not in shared_counter_maps[dl_to_fetch_from]:
                continue
            batch_data = shared_batch_maps[dl_to_fetch_from][_global_batch_id]
            out_queue.put((_global_batch_id,dl_to_fetch_from, batch_data))
            #with shared_locks[dl_to_fetch_from]:
            #    current_counter = shared_counter_maps[dl_to_fetch_from][_global_batch_id]
            #    if current_counter > 1:
            #        shared_counter_maps[dl_to_fetch_from][_global_batch_id] -= 1
                #elif current_counter == 1:
                    #del shared_counter_maps[dl_to_fetch_from][_global_batch_id]
                    #del shared_batch_maps[dl_to_fetch_from][_global_batch_id] 
            
                
            dl_to_fetch_from = (dl_to_fetch_from + gpu_per_job) % (total_dl) 
            _global_counter += num_gpus * batch_size
            _global_batch_id += gpu_per_job

           
    assert done_event.is_set()
    print("BATCH {} DONE EVENT".format(dl_id))
    return

