import torch
from torch import nn

class BaseQuantumObject(nn.Module):
    '''
    Generic base class for quantum objects (states and operators).
    Handles device placement and complex tensor conversion.
    '''
    def __init__(self, size: int, num_states: int = 2, device: str = 'cpu'):
        super().__init__()
        self.size = size
        self.num_states = num_states
        self.device = device
        self._state = None  # To be set by subclasses

    def to_state(self, inp: torch.Tensor) -> torch.Tensor:
        '''
        Convert input to complex tensor.
        '''
        if not torch.is_complex(inp):
            inp = inp.to(torch.float32)
            inp = torch.complex(inp, torch.zeros_like(inp))
        return inp.to(self.device)

    def _validate_shape(self, tensor: torch.Tensor, expected_shape: tuple):
        '''
        Validate tensor shape.
        '''
        if tensor.shape != expected_shape:
            raise ValueError(f"Expected shape {expected_shape}, got {tensor.shape}")

class BaseQubit(BaseQuantumObject):
    '''
    Base class for qubit states. Handles state normalization.
    Shape: [batch, num_states, 1]
    '''
    def __init__(self, size: int = 1, num_states: int = 2, device: str = 'cpu'):
        super().__init__(size, num_states, device)
        self._state = None

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        '''
        L2 normalize state vector (for pure states, ||psi||=1).
        '''
        norm = torch.sqrt((state * state.conj()).sum(dim=1, keepdim=True).real)
        norm = torch.clamp(norm, min=1e-8)  # Avoid div by zero
        return state / norm

    def set_state(self, inp_state):
        '''
        Set and normalize state.
        '''
        if not isinstance(inp_state, torch.Tensor):
            has_complex = any(isinstance(x, complex) for x in inp_state)
            dtype = torch.complex64 if has_complex else torch.float32
            inp_state = torch.tensor(inp_state, dtype=dtype, device=self.device)
        else:
            inp_state = inp_state.to(self.device)
        state = self.to_state(inp_state)
        # Reshape to [1, num_states, 1]
        while len(state.shape) < 3:
            state = state.unsqueeze(0 if len(state.shape) == 1 else -1)
        if state.shape[1] != self.num_states:
            state = state.unsqueeze(0) if len(state.shape) == 2 else state
            state = state[:, :self.num_states, :].unsqueeze(0)
        self._validate_shape(state, (1, self.num_states, 1))
        self._state = self.normalize_state(state)

    @property
    def state(self):
        if self._state is None:
            raise ValueError("State not set. Call set_state() first.")
        return self._state

    def measure(self, basis=None, num_shots=1):
        '''
        Measure qubit in computational basis (default) or given basis.
        Returns (outcomes, probs, collapsed_state)
        '''
        probs = (self.state * self.state.conj()).real.squeeze()
        probs = probs / probs.sum()  # Ensure normalized (should be)
        if basis is None:
            # Computational basis |0>, |1>, ...
            outcomes = torch.multinomial(probs, num_shots)
            # Collapse to first outcome for simplicity
            outcome = outcomes[0].item()
            collapsed = torch.zeros_like(self.state)
            collapsed[0, outcome, 0] = 1.0
        else:
            # Custom basis: project onto basis states
            projections = torch.stack([torch.vdot(b.state.squeeze(), self.state.squeeze()).abs()**2 for b in basis])
            probs = projections / projections.sum()
            outcome = torch.multinomial(probs, 1).item()
            collapsed_state = basis[outcome].state.clone()
            collapsed = collapsed_state
        return outcome, probs, collapsed.to(self.device)

class BaseOperator(BaseQuantumObject):
    '''
    Base class for quantum operators/gates. No normalization (unitary preserved).
    Shape: [batch, dim, dim] where dim = num_states ** size
    '''
    def __init__(self, size: int = 1, num_states: int = 2, device: str = 'cpu'):
        super().__init__(size, num_states, device)
        self._gate = None
        self.dim = num_states ** size

    def set_gate(self, gate_inp):
        '''
        Set gate matrix as complex tensor. Ensures unitary (checks U @ U^\dagger ≈ I).
        No L2 normalization.
        '''
        if not isinstance(gate_inp, torch.Tensor):
            # Simple check for complex in nested list
            def is_complex_list(lst):
                for item in lst:
                    if isinstance(item, complex):
                        return True
                    if isinstance(item, (list, tuple)) and is_complex_list(item):
                        return True
                return False
            has_complex = is_complex_list(gate_inp)
            dtype = torch.complex64 if has_complex else torch.float32
            gate_inp = torch.tensor(gate_inp, dtype=dtype, device=self.device)
        else:
            gate_inp = gate_inp.to(self.device)
        gate = self.to_state(gate_inp)
        # Reshape to [1, dim, dim]
        if gate.shape[-1] == self.dim and gate.shape[-2] == self.dim:
            gate = gate.unsqueeze(0)
        else:
            # Assume flat or 2D matrix
            dim_sq = self.dim * self.dim
            if gate.numel() != dim_sq:
                raise ValueError(f"Gate must have {dim_sq} elements for dim={self.dim}")
            flat_size = self.dim
            gate = gate.view(1, flat_size, flat_size)
        self._validate_shape(gate, (1, self.dim, self.dim))
        # Verify unitary: gate @ gate.adjoint() ≈ I
        identity = torch.eye(self.dim, device=self.device, dtype=torch.complex64).unsqueeze(0)
        unitary_check = torch.allclose(torch.matmul(gate, gate.mH), identity, atol=1e-6)
        if not unitary_check:
            print("Warning: Gate is not unitary.")
        self._gate = gate

    @property
    def gate(self):
        if self._gate is None:
            raise ValueError("Gate not set. Call set_gate() first.")
        return self._gate

# Placeholder for QuantumCircuit
class BaseQuantumCircuit(nn.Module):
    '''
    Placeholder for quantum circuit.
    Future: list of qubits, gates, layers; forward() to apply sequence.
    '''
    def __init__(self):
        super().__init__()
        # qubits: list[BaseQubit]
        # gates: list[BaseOperator]
        # circuit_layers: list
        pass

    def forward(self, input_state):
        raise NotImplementedError("Circuit simulation to be implemented.")

    def add_qubit(self, qubit):
        pass

    def add_gate(self, gate, target):
        pass


