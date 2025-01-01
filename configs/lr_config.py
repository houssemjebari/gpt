from dataclasses import dataclass

@dataclass
class LearningRateConfig:
    warmup_steps: int = 715
    max_steps: int = 19053
    max_lr: float = 6e-4
    min_lr: float = max_lr * 0.1
