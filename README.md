# QuantumSimulation

A quantum computation simulator built on PyTorch's neural network structure. This package provides a foundation for simulating quantum states and operations using tensors, making it easy to integrate with deep learning workflows.

## Structure

```
QuantumSimulation/
├── src/
│   ├── quantum_object.py       # Base quantum object with normalization
│   ├── Qubits/
│   │   └── qubits.py           # Qubit states (Zero, One, Plus, Minus, custom Qubit)
│   └── Operators/
│       └── operators.py        # Quantum gates (X, Y, Z, H, custom Gate)
├── dev/                         # Development notebooks and experiments
├── tests/                       # Unit tests
└── README.md
```

## Installation

The only dependency is PyTorch. Ensure you have it installed:

```bash
pip install torch
```

## Usage

```python
from QuantumSimulation.src.Qubits import Zero, One, Plus, Qubit
from QuantumSimulation.src.Operators import H, X, Y, Z

# Predefined qubits
q0 = Zero()
q1 = One()
q_plus = Plus()

# Custom qubit (automatically normalized)
q = Qubit([1, 1j])

# Gates
hadamard = H()
pauli_x = X()

# Apply gate to qubit via PyTorch matmul
result = torch.matmul(hadamard.gate, q0.state)
```

## Testing

Run all unit tests from the repo root:

```bash
python -m unittest discover QuantumSimulation/tests/ -v
```

Or run individual test files:

```bash
python QuantumSimulation/tests/test_qubits.py
python QuantumSimulation/tests/test_operators.py
python QuantumSimulation/tests/test_base.py
```

## Qubits

The following predefined qubits are available in `QuantumSimulation.src.Qubits`:

- `Zero` — |0⟩
- `One` — |1⟩
- `Plus` — |+⟩ = (|0⟩ + |1⟩)/√2
- `Minus` — |-⟩ = (|0⟩ − |1⟩)/√2
- `Qubit(inp_state)` — arbitrary state vector (automatically normalized)

All qubit states are stored as complex tensors with shape `(1, num_states, 1)`.

## Operators

The following quantum gates are available in `QuantumSimulation.src.Operators`:

- `X` — Pauli-X (bit flip)
- `Y` — Pauli-Y
- `Z` — Pauli-Z (phase flip)
- `H` — Hadamard gate
- `Gate(matrix, size)` — custom gate matrix (automatically normalized)

All gate matrices are stored as complex tensors with shape `(1, 2**size, 2**size)`.

## Package Imports

The repo is structured as a Python package. After adding the repo root to your `PYTHONPATH`:

```python
from QuantumSimulation.src.Qubits import Qubit, Zero, One, Plus, Minus
from QuantumSimulation.src.Operators import Gate, H, X, Y, Z
```

## Future Work

- Multi-qubit gates (e.g., CNOT)
- Tensor product operations for composite systems
- Measurement simulation
- GPU support
- Qudit support (higher-dimensional states)

