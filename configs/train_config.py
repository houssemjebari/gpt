from dataclasses import dataclass

@dataclass
class TrainConfig:
    total_batch_size: int = 2**19
    batch_size: int = 4 
    context_size: int = 1024
    train_steps: int = 19053 # 10Billion Tokens training 
    bfloat16: bool = True
    ddp: bool = True
