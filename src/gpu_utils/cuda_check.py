import os
import sys

CUDA_AVAILABLE = False
cp = None

def check_cuda():
    global CUDA_AVAILABLE, cp
    
    try:
        import cupy as _cp
        _cp.cuda.Device(0).compute_capability
        CUDA_AVAILABLE = True
        cp = _cp
        return True
    except Exception as e:
        CUDA_AVAILABLE = False
        cp = None
        return False

def get_array_module():
    if CUDA_AVAILABLE and cp is not None:
        return cp
    else:
        import numpy as np
        return np

def to_gpu(array):
    if CUDA_AVAILABLE and cp is not None:
        if isinstance(array, cp.ndarray):
            return array
        return cp.asarray(array)
    else:
        import numpy as np
        if isinstance(array, np.ndarray):
            return array
        return np.asarray(array)

def to_cpu(array):
    if CUDA_AVAILABLE and cp is not None:
        if isinstance(array, cp.ndarray):
            return cp.asnumpy(array)
        return array
    else:
        import numpy as np
        if isinstance(array, np.ndarray):
            return array
        return np.asarray(array)

def get_device_info():
    if CUDA_AVAILABLE and cp is not None:
        try:
            device = cp.cuda.Device(0)
            props = device.attributes
            return {
                'name': cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
                'compute_capability': device.compute_capability,
                'total_memory': device.mem_info[1] // (1024**3),
                'available': True
            }
        except:
            pass
    
    return {
        'name': 'CPU',
        'compute_capability': None,
        'total_memory': None,
        'available': False
    }

check_cuda()
