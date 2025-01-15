import pytest 
from gpt.training.eval import Evaluator
from gpt.training.observer import Observer
from torch import nn
import torch



class MockModel(nn.Module):
    def __init__(self):
        super(MockModel, self).__init__()
        self.linear = nn.Linear(10,1)
    
    def forward(self, x, y=None):
        logits = self.linear(x)
        loss = None
        if y is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, y)
        return logits, loss

class MockValLoader:
    def __iter__(self):
        return self
    
    def next_batch(self):
        x = torch.randn(10)
        y = torch.tensor([1],dtype=torch.float32)
        return x,y
    
    def reset(self):
        pass
    
def test_evaluator_prints_validation_loss(capfd):
    '''
    Test Objective: 
    - Ensure that the Evaluator correctly computes 
    and prints validation loss at specified intervals.
    '''
    model = MockModel()
    model.eval() 

    config = type('Config', (object,), {
        'eval_interval': 1,
        'bfloat16': False
    })()

    val_loader = MockValLoader()
    evaluator = Evaluator(master=True, ddp=False, device='cpu', config=config, val_loader=val_loader)
    event_type = 'on_step_end'
    data = {
            "step": 0,
            "model": model,
            "loss": 0.5,
            "lr": 0.001
        }
    evaluator.update(event_type, data)
    # Capture the printed output
    captured = capfd.readouterr()
    assert "Validation loss: " in captured.out

