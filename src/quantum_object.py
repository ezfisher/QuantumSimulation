import torch
from torch import nn

class BaseQuantumObject(nn.Module):
    def __init__(self, size, num_states, device='cpu'):
        self.size = size
        self.num_states = num_states
        self.device = device
    
    def __define__(self, inp_state):
        if not torch.is_complex(inp_state):
            inp_state = torch.complex(inp_state, torch.zeros_like(inp_state))
        return self.__norm__(inp_state)
        
    def __norm__(self, inp):
        if torch.is_complex(inp):
            norm_const = torch.sqrt((inp*inp.conj()).sum(dim=1).real.unsqueeze(1))
        else:
            norm_const = torch.sqrt((inp*inp).sum(dim=1).real.unsqueeze(1))
        return inp / norm_const