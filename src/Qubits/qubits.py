import torch
from ..quantum_object import BaseQubit

class Qubit(BaseQubit):
    '''
    Arbitrary qubit state, automatically normalized.
    '''
    def __init__(self, inp_state, num_states: int = 2, adjoint: bool = False, device: str = 'cpu'):
        super().__init__(size=1, num_states=num_states, device=device)
        self.adjoint = adjoint
        self.set_state(inp_state)

class Zero(BaseQubit):
    '''
    Computational basis |0>.
    '''
    def __init__(self, device: str = 'cpu'):
        super().__init__(size=1, num_states=2, device=device)
        self.set_state([1, 0])

class One(BaseQubit):
    '''
    Computational basis |1>.
    '''
    def __init__(self, device: str = 'cpu'):
        super().__init__(size=1, num_states=2, device=device)
        self.set_state([0, 1])

class Plus(BaseQubit):
    '''
    Hadamard basis |+> = (|0> + |1>)/sqrt(2).
    '''
    def __init__(self, device: str = 'cpu'):
        super().__init__(size=1, num_states=2, device=device)
        self.set_state([1, 1])

class Minus(BaseQubit):
    '''
    Hadamard basis |-> = (|0> - |1>)/sqrt(2).
    '''
    def __init__(self, device: str = 'cpu'):
        super().__init__(size=1, num_states=2, device=device)
        self.set_state([1, -1])


