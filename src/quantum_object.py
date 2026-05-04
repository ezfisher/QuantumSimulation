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

def tensor_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Tensor (Kronecker) product for states or gates.
    a, b: [batch, dim, ...]
    Handles states [1, d, 1] x [1, d2, 1] -> [1, d1*d2, 1]
    Gates [1, d1, d1] x [1, d2, d2] -> [1, d1*d2, d1*d2]
    """
    batch_a = a.shape[0]
    batch_b = b.shape[0]
    if a.shape[-1] == 1 and b.shape[-1] == 1:  # States
        kron = torch.kron(a.squeeze(-1), b.squeeze(-1))
        return kron.view(batch_a * batch_b, -1, 1)
    else:  # Gates
        return torch.kron(a, b)

class BaseQuantumCircuit(nn.Module):
    """
    Quantum circuit: add qubits/gates, forward simulates on |00..>.
    Supports single-qubit gates embedded with identities.
    """
    def __init__(self, device: str = 'cpu'):
        super().__init__()
        self.device = device
        self.qubit_list = []
        self.gate_queue = []  # (gate, target)

    def add_qubit(self, qubit):
        if not isinstance(qubit, BaseQubit):
            raise ValueError('Add BaseQubit instance')
        self.qubit_list.append(qubit)
        qubit.device = self.device  # Sync device

    def add_gate(self, gate, target_qubits):
        if not isinstance(gate, BaseOperator):
            raise ValueError('Add BaseOperator instance')
        if isinstance(target_qubits, int):
            target_qubits = [target_qubits]
        max_target = max(target_qubits)
        if max_target >= len(self.qubit_list):
            raise ValueError(f'Target qubits {target_qubits} out of range {len(self.qubit_list)}')
        self.gate_queue.append((gate, target_qubits))

    def forward(self):
        if not self.qubit_list:
            return torch.empty(0)
        # Init |0>
        n_qubits = len(self.qubit_list)
        dim = 1 << n_qubits  # 2**n
        state = torch.zeros(1, dim, 1, dtype=torch.complex64, device=self.device)
        state[0, 0, 0] = 1.0
        # Apply gates
        for gate, target_qubits in self.gate_queue:
            gate_full = self._expand_gate(gate.gate, target_qubits)
            state = torch.matmul(gate_full, state)
        return state

    def _expand_gate(self, gate, target_qubits):
        """
        Embed multi-qubit gate on target_qubits with I elsewhere.
        gate.size == len(target_qubits)
        """
        n_qubits = len(self.qubit_list)
        full_gate = torch.eye(1, dtype=torch.complex64, device=self.device).unsqueeze(0)
        # Assume target_qubits sorted, consecutive for simplicity
        start = min(target_qubits)
        gate_qubits = len(target_qubits)
        # Pre qubits I
        for i in range(start):
            i2 = torch.eye(2, dtype=torch.complex64, device=self.device).unsqueeze(0)
            full_gate = tensor_product(full_gate, i2)
        # Gate
        full_gate = tensor_product(full_gate, gate)
        # Post qubits I
        for i in range(start + gate_qubits, n_qubits):
            i2 = torch.eye(2, dtype=torch.complex64, device=self.device).unsqueeze(0)
            full_gate = tensor_product(full_gate, i2)
        return full_gate

    def measure(self, num_shots=1):
        state = self.forward()
        probs = torch.real(state * torch.conj(state)).squeeze()
        probs /= probs.sum()
        outcome = torch.multinomial(probs, num_shots, replacement=True)[0].item()
        collapsed = torch.zeros_like(state)
        collapsed[0, outcome, 0] = 1.0
        return outcome, probs, collapsed


