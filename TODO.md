# QuantumSimulation Refactor TODO

## Completed
- [ ]

## Pending Steps
1. **[Complete] Update src/quantum_object.py**: Added BaseQuantumObject, BaseQubit, BaseOperator, BaseQuantumCircuit.
2. **[Complete] Refactor src/Qubits/qubits.py**: Inherit BaseQubit, use set_state.
3. **[Complete] Refactor src/Operators/operators.py**: Inherit BaseOperator, fixed matrices.
4. **[Complete] BaseQuantumCircuit added to quantum_object.py.
5. **[Complete] Update __init__.py files**: Exports added.
6. **[Complete] Update tests/**: Rewrote test_base.py, fixed APIs, all tests pass.
7. **[Complete] Update README.md**: Document hierarchy.
8. **[Complete] Run tests: All 31 tests OK.
9. **[Complete] Add measurement to BaseQubit: `measure(basis=None)` computational default, custom basis, returns (outcome, probs, collapsed_state).
10. **[Complete] Tests: measure_zero, plus, hadamard_basis.
11. **[Complete] 34 tests OK.

**Complete!**




Progress tracked here after each step.

