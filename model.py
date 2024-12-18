from dataclasses import dataclass
import torch 
import torch.nn as nn 
from torch.nn import functional as F 
import math
import tiktoken


@dataclass 
class GPTConfig:
    block_size: int = 1024 # max sequence length 
    vocab_size: int = 50304 # number of tokens: 50k BPE Merges
    n_layer:  int = 12 # number of attention block layers
    n_head: int = 12 # number of heads 
    n_embd: int = 768 # Embedding
    flash_attn: bool = True

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert (config.n_embd % config.n_head) == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANO_GPT_SCALE_INIT = 1 # flag for output projection before the Layer Normalization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1,1, config.block_size, config.block_size))
        self.flash_attn = config.flash_attn
    
    def forward(self, x):
        B, T, C = x.size()
        qkv: torch.Tensor = self.c_attn(x) # (B,T, 3 * n_embd)
        q, k, v = qkv.split(self.n_embd, dim=2) 
        q = q.view(B,T,self.n_head, C // self.n_head).transpose(1,2) # (B, n_head, T, n_embd / n_head)
        k = k.view(B,T,self.n_head, C // self.n_head).transpose(1,2) # (B, n_head, T, n_embd / n_head)
        v = v.view(B,T,self.n_head, C // self.n_head).transpose(1,2) # (B, n_head, T, n_embd / n_head)
        if self.flash_attn: 
            attn = F.scaled_dot_product_attention(q,k,v, is_causal=True)
        else: 
            attn = q @ k.transpose(-2,-1) / math.sqrt(k.size(-1)) 
            attn = attn.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            attn = F.softmax(attn, dim=-1) 
            attn = attn @ v # (B, n_head, T, C // T) @ (B, n_head, T, C // n_head)
        x = attn.transpose(1,2).contiguous().view(B,T,C)
        y = self.c_proj(x)
        return y 

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config) 
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config) 

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), 
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        ################### Apply weight sharing: ##########################
        # The same set of weights is used in multiple parts  of a neural network
        # to reduce the number of parameters and potentially improve training 
        # efficiency. In the context of transformer models, this can be 
        # particularly useful for reducing memory usage
        ####################################################################.
        self.transformer.wte.weight = self.lm_head.weight
        
        # Apply weight initialization
        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (self.config.n_layer * 2) ** -0.5
            torch.nn.init.normal_(module.weight,mean=0.,std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight,mean=0.,std=0.02)
     

    def forward(self, x, targets=None):
        B, T = x.size() # (B,T)
        assert T <= self.config.block_size, "The input sequence shouldn't be longer than {self.config.block_size} words"
        positions = torch.arange(0, T, dtype=torch.long).to('cuda') # (T,)
        x = self.transformer.wte(x) + self.transformer.wpe(positions) # (B,T,n_embd) + (T,n_embd) -> (B,T,n_embd)
        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1,logits.size(-1)),targets.view(-1))
        return logits, loss