import pytest 
from training.trainer import Trainer
from training.observer import Observer
from torch.optim import AdamW
from utils.cosine_scheduler import WarmupCosineScheduler
import torch
import math
from torch import nn


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

class MockDataLoader:
    def __init__(self, num_batches=10):
        self.num_batches = num_batches
        self.current_batch = 0
    
    def next_batch(self):
        if self.current_batch >= self.num_batches:
            raise StopIteration
        x = torch.randn(10)
        y = torch.tensor([1], dtype=torch.float32)
        self.current_batch += 1
        return x, y

class TestObserver(Observer):
    def __init__(self):
        self.events = []
        self.event_type_update = 'on_step_end'

    def update(self, event_type, data):
        if event_type == self.event_type_update:
            self.events.append((event_type, data))
        

def test_trainer_train_loop(tmp_path):
    '''
    Test Objective:
    - Ensure that the Trainer class 
    correctly performs the training loop,
    interacts with observers, and updates
    learning rates
    '''
    # Initialize components
    model = MockModel()
    optimizer = AdamW(model.parameters(), lr=0.1)
    scheduler_config = type('Config', (object,), {
        'warmup_steps': 2,
        'max_steps': 4,
        'max_lr': 0.1,
        'min_lr': 0.01
    })()
    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        config=scheduler_config
    )
    train_loader =MockDataLoader(num_batches=4)
    config = type('Config', (object,), {
        'train_steps': 4,
        'bfloat16': False,
        'grad_accum_steps': 1,
        'warmup_steps': 2,
        'max_lr': 0.1,
        'min_lr': 0.01,
        'total_batch_size': 400,
        'context_size': 100,
        'batch_size': 4
    })()
    trainer = Trainer(
        model= model,
        optimizer= optimizer,
        scheduler= scheduler,
        train_loader= train_loader,
        ddp= False,
        config= config,
        device= 'cpu',
        world_size=1,
        master_process=True,
    )

    # Create Mock Observers
    logger = TestObserver()
    evaluator = TestObserver()

    # Attach Observers 
    trainer.attach(logger)
    trainer.attach(evaluator)

    # Run Training
    trainer.train()

    # Verify that observers received the correct number of notifications
    assert len(logger.events) == 4
    assert len(evaluator.events) == 4 

    # Verify that learning rates were updated correctly
    expected_lrs = [0.1, 0.1, 0.05500000000000001, 0.01]
    for i, (event_type, data) in enumerate(logger.events):
        assert event_type == 'on_step_end'
        assert data['step'] == i
        assert abs(data['lr'] - expected_lrs[i]) < 1e-6
        assert 'loss' in data
