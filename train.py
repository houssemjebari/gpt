from  model import GPT, GPTConfig
from data_loader import DataLoader
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch
import time
import math
import os

ddp = int(os.environ.get('RANK', -1)) != -1

if ddp:
    assert torch.cuda.is_available(), "CUDA is needed for DDP"
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process is the one responsible for logging and saving checkpoints

else: 
    ddp_rank = 0
    ddp_local_rank = 0 
    ddp_world_size = 1
    master_process = True
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'gpu'
    print('Using device: ', device)

warmup_steps = 10
max_steps = 50
max_lr = 6e-4
min_lr = max_lr * 0.1

def get_lr(it):
    if it < warmup_steps: 
        return max_lr * (it+1) / warmup_steps
    if it> max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)  

# Generate the random seed
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

total_batch_size = 2**19
batch_size = 4 
context_size = 1024
assert total_batch_size % ( batch_size * context_size * ddp_world_size) 
grad_accum_steps = total_batch_size // (batch_size * context_size * ddp_world_size)   

if master_process:
    print(f'total desired batch size: {total_batch_size}')
    print(f'=> Calculated gradient accumulation_steps: {grad_accum_steps}')


# Import the data loader 
train_loader = DataLoader(batch_size,context_size, process_rank=ddp_rank, num_processes=ddp_world_size)

# Instantiate the Model
model = GPT(GPTConfig)
model.to(device)
model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# Optimize 
train_steps = 19532 # 10 BTokens // total_batch_size
optimizer = torch.optim.AdamW(model.parameters(),lr =3e-4, betas=(0.9, 0.95), eps=1e-8)
for step in range(train_steps):
    t0 = time.time()
    # Zero the grads before the forward pass
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        # Get batches and send them to the Device
        x,y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        # Compute the forward pass 
        with torch.autocast(device_type=device, dtype=torch.bfloat16):  
            logits, loss = model(x,y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        # Compute the grads
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        loss.backward()
   # Get the average of the accumulated loss from all the devices
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    # Clip the gradients to avoid model shock
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # Get the iteration learning rate 
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    # Update the weights
    optimizer.step()
    # Performance calculation and logging
    torch.cuda.synchronize()
    step_time = time.time() - t0
    tokens_per_sec = (batch_size * context_size * grad_accum_steps) / step_time
    if master_process:
        print(f'step {step} | loss: {loss_accum:.6f} | lr: {lr} | norm: {norm:.4f} | dt: {step_time:.2f} seconds  | tok/sec: {tokens_per_sec}')
if ddp:
    destroy_process_group()

