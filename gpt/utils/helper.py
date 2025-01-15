import torch
from contextlib import nullcontext

def get_autocast_context(device, use_autocast=False, autocast_dtype=torch.bfloat16):
    """
    Returns a context manager that either does autocast with the specified dtype
    or does nothing (nullcontext), depending on use_autocast.
    """
    if use_autocast:
        return torch.autocast(device_type=device, dtype=autocast_dtype)
    else:
        return nullcontext()
