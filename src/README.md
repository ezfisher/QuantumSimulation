# Source Package

## `quantum_object.py`

Base class `BaseQuantumObject` extending `nn.Module`. Provides:

- `__define__(state)` — converts real tensors to complex and normalizes
- `__norm__(inp)` — L2 normalization along dimension 1

## `Qubits/`

Predefined and custom qubit states with shape `(1, num_states, 1)`.

| Class | Description |
|-------|-------------|
| `Qubit` | Arbitrary single-qubit state |
| `Zero`  | Computational basis state \|0⟩ |
| `One`   | Computational basis state \|1⟩ |
| `Plus`  | Superposition (\|0⟩ + \|1⟩)/√2 |
| `Minus` | Superposition (\|0⟩ − \|1⟩)/√2 |

## `Operators/`

Quantum gates with shape `(1, 2**size, 2**size)`.

| Class | Description |
|-------|-------------|
| `Gate` | Custom gate from a matrix |
| `H`    | Hadamard gate |
| `X`    | Pauli-X gate |
| `Y`    | Pauli-Y gate |
| `Z`    | Pauli-Z gate |

