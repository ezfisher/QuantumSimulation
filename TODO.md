# QuantumSimulation TODO

## Performance refactor (dense state, faster gate application)
- [x] Add fast helpers in `src/quantum_object.py` to apply single-qubit gates without building embedded Kronecker matrices.
- [x] Add fast helper to apply controlled 2-qubit gates (control/target convention) directly on the state vector.


- [x] Update `BaseQuantumCircuit.forward()` to dispatch to fast paths for single-qubit + supported controlled gates; keep fallback to existing `_expand_gate()`.

- [x] Add/adjust unit tests to ensure correctness for the fast paths (especially CX / Bell state).

- [x] Run full unittest suite.



