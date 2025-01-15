import os 
import torch 
import torch.distributed as dist 


def init_distributed():
    '''
    Initializes distributed data parallel if environment 
    variable indicates a multi-process setup. 
    Returns a tuple: (ddp, ddp_rank, ddp_local_rank, 
                      ddp_world_size, master_process, device)
    '''
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        assert torch.cuda.is_available(), "CUDA is needed for DDP"
        dist.init_process_group(backend='nccl')
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = (ddp_rank == 0)

    else: 
        ddp_rank = 0 
        ddp_local_rank = 0 
        ddp_world_size = 1 
        master_process = True
        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda'
        print('Using device: ', device)
    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, master_process, device

def cleanup_distributed():
    """
    Properly destroys the process group to avoid hanging.
    """
    dist.destroy_process_group()
