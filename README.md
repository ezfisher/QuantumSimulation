# QuantumSimulation

A quantum computation simulator built on PyTorch's neural network structure with clear class hierarchy. Integrates quantum simulation with deep learning workflows.

## Hierarchy

- `BaseQuantumObject` (nn.Module): Generic base.
- `BaseQubit`: States with L2 normalization (pure states).
- `BaseOperator`: Gates with unitary check (U U† = I), no L2 norm.
- `BaseQuantumCircuit`: Circuits with add_qubit/add_gate, forward (multi-qubit sim), measure.

Predefined classes inherit accordingly.

## Structure

```
QuantumSimulation/
├── src/
│   ├── __init__.py
│   ├── quantum_object.py       # Bases: BaseQuantumObject, BaseQubit, BaseOperator, BaseQuantumCircuit
│   ├── Qubits/
│   │   ├── __init__.py
│   │   └── qubits.py           # Qubit, Zero, One, Plus, Minus
│   └── Operators/
│       ├── __init__.py
│       └── operators.py        # Gate, H, X, Y, Z
├── tests/                       # Unit tests (all pass)
├── dev/                         # Notebooks
└── README.md
```

## Installation

```bash
pip install torch
```

Add to PYTHONPATH or `pip install -e .` (add pyproject.toml if needed).

## Usage

```python
from QuantumSimulation.src import Qubit, Zero, H, X

q0 = Zero()
h = H()
result = torch.matmul(h.gate, q0.state)
print(result.shape)  # torch.Size([1, 2, 1])
```

**Examples working** (output demo):
```
1. Custom state:
State: tensor([0.6000, 0.8000])
Norm: 1.00
2. Apply H to |0>:
H|0> = tensor([0.7071+0.j, 0.7071+0.j])
...
```
Run `cd QuantumSimulation && python examples/basic_usage.py`




## Testing

All 38 tests pass (use `discover`):
```bash
cd QuantumSimulation && python -m unittest discover tests -v
```


## Future Work

- Controlled gates (CNOT/CX).
- GPU/multi-qubit opt.
- Qudits.
- Noise models.


