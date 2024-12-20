import os
import torch
import numpy as np

def load_tokens(shard_path):
    """
    This function loads tokens from a shard file. The tokens might be saved in a binary format,
    where each token is an integer representing a vocabulary index.

    Args:
        shard_path (str): The path to the shard file to load.

    Returns:
        torch.Tensor: A tensor of tokens loaded from the shard file.
    """
    # Example for reading a binary file (adjust according to actual data format)
    try:
        # If tokens are stored as 32-bit integers in binary format, load them
        tokens = np.fromfile(shard_path, dtype=np.uint16)
        return torch.tensor(tokens, dtype=torch.int64)
    
    except Exception as e:
        print(f"Error loading tokens from {shard_path}: {e}")
        raise

class DataLoader:

    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B 
        self.T = T 
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        data_root = 'edu_fineweb10B'
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root,s) for s in shards]
        self.shards = shards  
        assert len(shards) > 0, f'No shards found for split {split}'
        print(f'Found {len(self.shards)} shards for split {split}')
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank


    def next_batch(self):
        buf = self.tokens[self.current_position: self.current_position + self.B * self.T + 1]
        x = buf[ :-1].view(self.B, self.T)
        y = buf[1:  ].view(self.B, self.T)
        self.current_position += self.B * self.T * self.num_processes
        if (self.current_position + (self.B * self.T * self.num_processes + 1) >= len(self.tokens)):
            self.current_position = self.B * self.T * self.process_rank
        return x,y
