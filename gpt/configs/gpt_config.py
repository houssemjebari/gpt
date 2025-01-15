from dataclasses import dataclass

@dataclass 
class GPTConfig:
    block_size: int = 1024 # max sequence length 
    vocab_size: int = 50304 # number of tokens: 50k BPE Merges
    n_layer:  int = 12 # number of attention block layers
    n_head: int = 12 # number of heads 
    n_embd: int = 768 # Embedding
    flash_attn: bool = True