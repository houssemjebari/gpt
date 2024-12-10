from dataclasses import dataclass
import torch 
import torch.nn as nn 
from torch.nn import functional as F 
import math
import tiktoken


@dataclass 
class GPTConfig:
    block_size: int = 1024 # max sequence length 
    vocab_size: int = 50257 # number of tokens: 50k BPE Merges
    n_layer:  int = 12 # number of attention block layers
    n_head: int = 12 # number of heads 
    n_embd: int = 768 # Embedding


class GPT(nn.Module):

    def __init__(self, config):
        super.__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(nn.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # Block still needs to be defined 
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
