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


class Controlled(BaseOperator):
    """
    Controlled-U gate for qubits (control 0, target 1).
    U: single-qubit operator.
    Matrix: |0><0| ⊗ I + |1><1| ⊗ U
    """
    def __init__(self, U, device: str = 'cpu'):
        super().__init__(size=2, num_states=2, device=device)
        self.U = U
        i = torch.eye(2, dtype=torch.complex64, device=device)
        p0 = torch.tensor([[1,0],[0,0]], dtype=torch.complex64, device=device)
        p1 = torch.tensor([[0,0],[0,1]], dtype=torch.complex64, device=device)
        p0_i = torch.kron(p0.unsqueeze(0), i.unsqueeze(0))
        p1_u = torch.kron(p1.unsqueeze(0), self.U.gate)
        controlled_gate = p0_i + p1_u
        controlled_gate = controlled_gate.squeeze(0)
        self.set_gate(controlled_gate)


class CX(Controlled):
    """
    CNOT / CX gate (control 0, X on target 1).
    """
    def __init__(self, device: str = 'cpu'):
        super().__init__(X(device=device), device=device)


class CH(Controlled):
    """
    Controlled-Hadamard.
    """
    def __init__(self, device: str = 'cpu'):
        super().__init__(H(device=device), device=device)


class CY(Controlled):
    """
    Controlled-Y.
    """
    def __init__(self, device: str = 'cpu'):
        super().__init__(Y(device=device), device=device)


class CZ(Controlled):
    """
    Controlled-Z.
    """
    def __init__(self, device: str = 'cpu'):
        super().__init__(Z(device=device), device=device)


