import torch
import numpy as np
from ..quantum_object import BaseOperator

inv_sqrt2 = 1 / np.sqrt(2)

class Gate(BaseOperator):
    '''
    Custom quantum gate/operator.
    '''
    def __init__(self, gate_inp, size: int = 1, device: str = 'cpu'):
        super().__init__(size=size, num_states=2, device=device)
        self.set_gate(gate_inp)

class H(BaseOperator):
    '''
    Hadamard gate.
    '''
    def __init__(self, device: str = 'cpu'):
        super().__init__(size=1, num_states=2, device=device)
        h_matrix = torch.tensor([
            [inv_sqrt2, inv_sqrt2],
            [inv_sqrt2, -inv_sqrt2]
        ], dtype=torch.complex64)
        self.set_gate(h_matrix)

class X(BaseOperator):
    '''
    Pauli-X gate.
    '''
    def __init__(self, device: str = 'cpu'):
        super().__init__(size=1, num_states=2, device=device)
        x_matrix = torch.tensor([
            [0, 1],
            [1, 0]
        ], dtype=torch.complex64)
        self.set_gate(x_matrix)

class Y(BaseOperator):
    '''
    Pauli-Y gate.
    '''
    def __init__(self, device: str = 'cpu'):
        super().__init__(size=1, num_states=2, device=device)
        y_matrix = torch.tensor([
            [0, -1j],
            [1j, 0]
        ], dtype=torch.complex64)
        self.set_gate(y_matrix)

class Z(BaseOperator):
    '''
    Pauli-Z gate.
    '''
    def __init__(self, device: str = 'cpu'):
        super().__init__(size=1, num_states=2, device=device)
        z_matrix = torch.tensor([
            [1, 0],
            [0, -1]
        ], dtype=torch.complex64)
        self.set_gate(z_matrix)


