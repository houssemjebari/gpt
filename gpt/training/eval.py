import torch
import torch.distributed as dist
from gpt.utils.helper import get_autocast_context
from gpt.training.observer import Observer

class Evaluator(Observer):
    def __init__(self, val_loader, config, master, ddp, device):
        self.val_loader = val_loader
        self.eval_interval = config.eval_interval
        self.use_bfloat16 = config.bfloat16
        self.device = device
        self.ddp = ddp
        self.master = master

    def update(self, event_type, data):
        if (event_type == 'on_step_end') and (data["step"] % self.eval_interval == 0):
            model = data['model'] 
            val_loss_accum = 0.
            val_loss_steps = 20
            
            model.eval()
            self.val_loader.reset()
            with torch.no_grad():
                for _ in range(val_loss_steps):
                    x, y = self.val_loader.next_batch()
                    x, y = x.to(self.device), y.to(self.device)
                    autocast_ctx = get_autocast_context(
                        device=self.device,
                        use_autocast=self.use_bfloat16, 
                        autocast_dtype=torch.bfloat16
                    )  
                    with autocast_ctx:
                        _ , loss = model.forward(x, y)
                    loss = loss / val_loss_steps
                    val_loss_accum += loss.detach()
                if self.ddp: 
                    dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
                if self.master: 
                    print(f'Validation loss: {val_loss_accum.item():4f}')

