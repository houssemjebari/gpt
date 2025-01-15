import os 
import torch 
from gpt.configs.train_config import TrainConfig
from gpt.configs.gpt_config import GPTConfig
from gpt.configs.lr_config import LearningRateConfig
from gpt.utils.distributed import init_distributed, cleanup_distributed
from gpt.utils.cosine_scheduler import WarmupCosineScheduler
from gpt.training.trainer import Trainer
from gpt.training.logger import Logger
from gpt.training.eval import Evaluator
from gpt.model import GPT
from gpt.data_loader import DataLoader


def main():
    # initialize the distributed training settings
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, master_process, device = init_distributed()
    # initialize the dependencies to inject
    model_config = GPTConfig()
    lr_config = LearningRateConfig()
    train_config = TrainConfig()
    # initialize the model 
    model = GPT(model_config)
    model.to(device)
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[ddp_local_rank], 
            output_device=ddp_local_rank
        )
    # initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    # initialize the scheduler 
    scheduler = WarmupCosineScheduler(optimizer,lr_config)
    # initialize the data loaders 
    train_loader = DataLoader(train_config.batch_size, train_config.context_size,ddp_rank,ddp_world_size,'train')
    val_loader = DataLoader(train_config.batch_size, train_config.context_size,ddp_rank,ddp_world_size,'val')
    # initialize the trainer 
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        ddp=ddp,
        world_size= ddp_world_size,
        master_process=master_process,
        config=train_config,
        device=device
    )
    # initialize the observers
    file_logger = Logger(log_file='logs/log.txt',master=master_process)
    evaluator = Evaluator(val_loader=val_loader, config=train_config, master=master_process, ddp=ddp, device=device)
    # attach the observers to the trainer 
    trainer.attach(file_logger)
    trainer.attach(evaluator)
    # run the training process
    trainer.train()
    if ddp:
        cleanup_distributed()

if __name__ == "__main__":
    main()
