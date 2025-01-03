
import torch
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from training.subject import Subject
from utils.helper import get_autocast_context


class Trainer(Subject):
    def __init__(self, model, optimizer, scheduler, train_loader, ddp, config, device='cuda'):
        self.model = model 
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.config = config 
        self.device = device
        self.observers = [] 
        self.train_steps = config.train_steps
        self.ddp = ddp
        self.bfloat16 = config.bfloat16

    def attach(self, observer):
        self.observers.append(observer)

    def detach(self, observer):
        self.observers.remove(observer)
    
    def notify(self, event_type, data):
        for observer in self.observers:
            observer.update(event_type, data)
    
    def train(self):
        for step in range(self.train_steps):
            self.notify('on_step_start', {"step": step,
                                           "model": self.model})
            # Zero the grad before the forward pass 
            self.optimizer.zero_grad()  
            loss_accum = 0.
            for micro_step in range(self.grad_accum_steps):
                # Get batches and send to device
                x,y = self.train_loader.next_batch()
                x, y = x.to(self.device), y.to(self.device)
                # Compute the Forward pass 
                autocast_ctx = get_autocast_context(
                        device=self.device,
                        use_autocast=self.bfloat16, 
                        autocast_dtype=torch.bfloat16
                    )  
                with autocast_ctx:
                    logits, loss = self.model(x,y)
                loss = loss / self.grad_accum_steps
                loss_accum += loss.detach()
                # Compute the grads
                if self.ddp:
                    self.model.require_backward_grad_sync = (micro_step == self.grad_accum_steps - 1)
                loss.backward()
            # Aggregate losses from all the devices and clip the gradients
            if self.ddp:
                dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            lr = self.scheduler.get_lr(step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            # Update the weights
            self.optimizer.step()
            # Performance calculation and logging
            torch.cuda.synchronize()
            self.notify('on_step_end', {"step": step,
                                        "model": self.model,
                                        "loss": loss,
                                        "lr": lr})
        if self.ddp:
            destroy_process_group()
    


            
