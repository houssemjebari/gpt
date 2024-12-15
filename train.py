from  model import GPT, GPTConfig
from data_loader import DataLoader
import torch
import time
import math

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
    

# detect the device to use
device = "cpu"
torch.manual_seed(1337)
if torch.cuda.is_available():
    device = 'cuda'
    torch.cuda.manual_seed(1337)
print(f'using device: {device}')

# Import the data loader 
batch_size = 4
context_size = 1024
train_loader = DataLoader(batch_size,context_size)

# Instantiate the Model
model = GPT(GPTConfig)
model.to(device)
model = torch.compile(model)

# Optimize 
train_steps = 50
optimizer = torch.optim.AdamW(model.parameters(),lr =3e-4, betas=(0.9, 0.95), eps=1e-8)
for step in range(train_steps):
    t0 = time.time()
    # Get batches and send them to the Device
    x,y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    # Zero the grads before the forward pass
    optimizer.zero_grad()
    # Compute the forward pass 
    with torch.autocast(device_type=device, dtype=torch.bfloat16):  
        logits, loss = model(x,y)
    # Compute the grads
    loss.backward()
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
    tokens_per_sec = (batch_size * context_size) / step_time
    print(f'step {step} | loss: {loss.item()} | lr: {lr} | norm: {norm:.4f} | dt: {step_time:.2f} ms  | tok/sec: {tokens_per_sec}')


