import torch
import numpy
import math
import psutil

def check_available_ram(device='cpu'):

    if isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, torch.device):
        device = device
    else:
        raise RuntimeError("`device` must be str or torch.device")

    if device.type == 'cpu':
        return psutil.virtual_memory().available
    else:
        total = torch.cuda.get_device_properties(device)
        used = torch.cuda.memory_allocated(device)
        return total - used

def will_it_fit(size, device="cpu", safe_mode=True):
    if safe_mode:
        try:
            torch.empty(size, device=device, dtype=torch.uint8)
        except:
            return False
        return True
    else:
        return check_available_ram(device) >= size

def find_optimal_splits(n, get_required_memory, device="cpu", safe_mode=True):

    splits = 1
    sub_n = n

    while True:
        if splits > n:
            splits = n
            break
        
        sub_n = math.ceil(n / splits)
        required_memory = get_required_memory(sub_n)

        if will_it_fit(required_memory, device, safe_mode):
            break
        else:
            splits *= 2
            continue

    return splits
