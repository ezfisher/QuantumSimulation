'''
Basic usage example: custom state, apply gate, measure.
'''

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import Qubit, Zero, One, H
import torch




print("1. Custom state:")
q_custom = Qubit([3, 4])  # Normalized to [0.6, 0.8]
print(f"State: {q_custom.state.squeeze().real}")
print(f"Norm: {torch.norm(q_custom.state).item():.2f}")

print("\n2. Apply H to |0>:")
z = Zero()
h = H()
h_z = torch.matmul(h.gate, z.state)
print(f"H|0> = {h_z.squeeze()}")
plus = Qubit([1,1])  # Should match
print(f"Matches |+> = {plus.state.squeeze()}")

print("\n3. Apply H to |1>:")
o = One()
h_o = torch.matmul(h.gate, o.state)
print(f"H|1> = {h_o.squeeze()}")
minus = Qubit([1,-1])  # Should match
print(f"Matches |-> = {minus.state.squeeze()}")

print("\n4. Measure |+>:")
plus = Qubit([1,1])
outcome, probs, collapsed = plus.measure()
print(f"Outcome: {outcome}, Probs: {probs}, Collapsed: {collapsed.squeeze()}")

print("\n5. Measure in Hadamard basis:")
z = Zero()
basis_h = [Qubit([1,1]), Qubit([1,-1])]  # |+>, |->
outcome_h, probs_h, collapsed_h = z.measure(basis=basis_h)
print(f"H-basis on |0>: outcome={outcome_h}, probs={probs_h}")

