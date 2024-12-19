import torch
import tiktoken


class DataLoader:

    def __init__(self, B, T, process_rank, num_processes):
        self.B = B 
        self.T = T 
        self.process_rank = process_rank
        self.num_processes = num_processes

        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f'loaded {len(self.tokens)} tokens')
        self.current_position = self.B * self.T * self.process_rank


    def next_batch(self):
        buf = self.tokens[self.current_position: self.current_position + self.B * self.T + 1]
        x = buf[ :-1].view(self.B, self.T)
        y = buf[1:  ].view(self.B, self.T)
        self.current_position += self.B * self.T * self.num_processes
        if (self.current_position + (self.B * self.T * self.num_processes + 1) >= len(self.tokens)):
            self.current_position = self.B * self.T * self.process_rank
        return x,y
