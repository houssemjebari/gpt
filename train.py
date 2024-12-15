from  model import GPT, GPTConfig
from data_loader import DataLoader
import torch
import time

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
optimizer = torch.optim.AdamW(model.parameters(),lr =3e-4)
for i in range(train_steps):
    t0 = time.time()
    x,y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    with torch.autocast(device_type=device, dtype=torch.bfloat16):  
        logits, loss = model(x,y)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    step_time = time.time() - t0
    tokens_per_sec = (batch_size * context_size) / step_time
    print(f'step {i}, loss: {loss.item()} dt: {step_time:.2f} ms tok/sec: {tokens_per_sec}')


