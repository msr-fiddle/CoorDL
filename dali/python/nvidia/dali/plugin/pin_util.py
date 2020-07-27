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


def pin_memory_loop(shared_batch_maps, shared_counter_maps, shared_locks, out_queue, device_id, batches_to_pin, pinned_counter, total_dl, done_event):
    torch.cuda.set_device(device_id)
    global_batch_to_pin = 0
    dl_id = device_id
    print("PIN THREAD : Batches to pin :{}".format(batches_to_pin))
    while not done_event.is_set():
        if pinned_counter == batches_to_pin:
            #Reset because epoch ended
            global_batch_to_pin = 0 
            pinned_counter = 0
        #Since we do a round robin, we can exactly know what batches are in other DL maps. Device id = DL ID
        dl_to_fetch_from = global_batch_to_pin % total_dl
        #if dl_to_fetch_from == dl_id:
        #   global_batch_to_pin += 1
        #   pinned_counter += 1
        #   continue

        #while out_queue.qsize() > 24*2:
        #    continue

        while global_batch_to_pin not in shared_counter_maps[dl_to_fetch_from]:
            continue

        #print("[PIN TH {}] : Trying to pin {} from {}".format(dl_id, global_batch_to_pin, dl_to_fetch_from))
        try:
            s = time.time()
            batch_data = shared_batch_maps[dl_to_fetch_from][global_batch_to_pin]
            #print(type(batch_data))
            #print("[{}] : Got deserialized batch {} from {}".format(dl_id, global_batch_to_pin, dl_to_fetch_from))
            #batch_data = pin_one_batch(shared_batch_maps[dl_to_fetch_from][global_batch_to_pin], batch_id=global_batch_to_pin, dl=dl_id)
            dur = time.time() - s
            #if dl_id == 0:
                #print("Duration to PIN {} = {}s".format(global_batch_to_pin, dur))
        except Exception:  
            out_queue.put((global_batch_to_pin, "ERROR" ))
                #out_queue.put((global_batch_to_pin, ExceptionWrapper(sys.exc_info())))
        else:
            try:
                out_queue.put((global_batch_to_pin, batch_data), block=True)
                #print("[PIN TH {}] : PUT PINNED BATCH {} from DL {}, q len={}, q empty={}".format(dl_id, global_batch_to_pin, dl_to_fetch_from, out_queue.qsize(), out_queue.empty()))
            except Exception:
                raise ("[PIN] ERROR PUTTING IN Q")
        with shared_locks[dl_to_fetch_from]:
            current_counter = shared_counter_maps[dl_to_fetch_from][global_batch_to_pin]
            if current_counter > 1: 
                shared_counter_maps[dl_to_fetch_from][global_batch_to_pin] -= 1 
            elif current_counter == 1: 
                del shared_counter_maps[dl_to_fetch_from][global_batch_to_pin]
                del shared_batch_maps[dl_to_fetch_from][global_batch_to_pin]
        global_batch_to_pin += 1 
        pinned_counter += 1
        dl_to_fetch_from = global_batch_to_pin % total_dl
        #print("[{}] : Trying to pin {} from {}".format(dl_id, global_batch_to_pin, dl_to_fetch_from))

    assert done_event.is_set()
    print("PIN DONE EVENT {}".format(dl_id))
    return



def pin_one_batch(batch, batch_id=0, dl=0):
    if isinstance(batch, torch.Tensor):
        #print("[{} PIN] : Tensor pin {}".format(dl, batch_id))
        return batch.pin_memory()
    elif isinstance(batch, string_classes):
        return batch
    elif isinstance(batch, container_abcs.Mapping):
        return {k: pin_one_batch(sample, batch_id=batch_id, dl=dl) for k, sample in batch.items()}
    elif isinstance(batch, tuple) and hasattr(batch, '_fields'):  # namedtuple
        return type(batch)(*(pin_one_batch(sample, batch_id=batch_id, dl=dl) for sample in batch))
    elif isinstance(batch, container_abcs.Sequence):
        return [pin_one_batch(sample, batch_id=batch_id, dl=dl) for sample in batch]
    elif hasattr(batch, "pin_memory"):
        return batch.pin_memory()
    else:
        print("No match for pinning - {}, [0]:{}".format(type(batch), type(batch[0])))
        return batch
