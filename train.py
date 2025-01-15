from  gpt.model import GPT, GPTConfig
from gpt.data_loader import DataLoader
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch
import time
import math
import os
import tiktoken
import torch.nn.functional as F
from gpt.hellaswag import * 


def get_most_likely_row(tokens, mask, logits):
    # shift the logits and tokens for the autoregressive task
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    # flatten the logits and tokens for loss computation
    flat_logits = shift_logits.view(-1, shift_logits.size(-1)) # (batch_size*context_window, vocab_size)
    flat_tokens = shift_tokens.view(-1) # (batch_size*context_window, )
    shift_losses = F.cross_entropy(flat_logits, flat_tokens, reduction='none') # (batch_size*context_window, )
    shift_losses = shift_losses.view(tokens.size(0), -1) # (batch_size, context_window)
    # shift the mask and apply to the loss 
    shift_mask = (mask[..., 1:]).contiguous() # (batch_size, context_window)
    masked_losses = shift_losses * shift_mask
    sum_loss = masked_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    pred_norm = avg_loss.argmin().item()
    return pred_norm



log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
log_file =os.path.join(log_dir,"log.txt")
#open file to clear
with open(log_file, 'w') as f:  
    pass

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
        device = 'cuda'
    print('Using device: ', device)

warmup_steps = 715
max_steps = 19053
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
assert (total_batch_size % ( batch_size * context_size * ddp_world_size))== 0
grad_accum_steps = total_batch_size // (batch_size * context_size * ddp_world_size)   

if master_process:
    print(f'total desired batch size: {total_batch_size}')
    print(f'=> Calculated gradient accumulation_steps: {grad_accum_steps}')


# Import the data loader 
train_loader = DataLoader(batch_size,context_size, process_rank=ddp_rank, num_processes=ddp_world_size, split='train')
val_loader = DataLoader(batch_size,context_size, process_rank=ddp_rank, num_processes=ddp_world_size, split='val')

# Instantiate the Model
model = GPT(GPTConfig)
model.to(device)
model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# Optimize 
train_steps = 19053 # 10 BTokens // total_batch_size
optimizer = torch.optim.AdamW(model.parameters(),lr =3e-4, betas=(0.9, 0.95), eps=1e-8)
enc = tiktoken.get_encoding('gpt2')
for step in range(train_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # Once in a while evaluate the loss 
    if (step % 250 == 0 or last_step):
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
            if ddp: 
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            if master_process: 
                print(f'Validation loss: {val_loss_accum.item():4f}')
                with open(log_file, 'a') as f:
                    f.write(f'{step} val {val_loss_accum.item():4f}')
                # optionally save the model
                if (step % 5000 == 0 or last_step):
                    checkpoint_path = os.path.join(log_dir, f'model_{step:05d}.pt')
                    raw_model = model.module if ddp else model
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'config': raw_model.config,
                        'step': step,
                        'val_loss': val_loss_accum.item(),
                    }
                    torch.save(checkpoint, checkpoint_path)
    
    # Once in a while generate from the model
    if (step % 1000 == 0 or last_step):
        model.eval()
        num_return_sequences = 4 
        max_length = 32 
        tokens = enc.encode("Hello I am a language model")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:
            with torch.no_grad():
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(x,y)
                logits = logits[:,-1,:]
                probs = F.softmax(logits, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1) # (B, 50), (B, 50)
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                xcol = torch.gather(topk_indices, -1, ix) # (B,1)
                xgen = torch.cat((xgen, xcol), dim=1)
        for i in range(num_return_sequences):
            tokens = xgen[i,:max_length].tolist()
            decoded = enc.decode(tokens)
            print(f'rank {ddp_rank} sample {i} : {decoded}')

    # Once in a while do a hella swag evaluation
    if (step % 5000 == 0 or last_step):
        num_total = 0.
        num_correct_norm = 0.
        if master_process: 
            for example in iterate_examples('val'):
                    tokens, mask, label = render_example(example)
                    tokens = tokens.to(device)
                    mask = mask.to(device)
                    with torch.no_grad():
                        with torch.autocast(device_type=device, dtype=torch.bfloat16):
                            logits, _ = model(tokens)
                    pred_norm = get_most_likely_row(tokens, mask, logits)
                    num_total += 1 
                    num_correct_norm += int(pred_norm == label)
            print(f"Hellaswag accuracy: {num_correct_norm / num_total}")

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

