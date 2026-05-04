import unittest
import torch
import sys
import os
root_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_dir)
from src import BaseQuantumCircuit, Zero, H, Gate

class TestQuantumCircuit(unittest.TestCase):
    def setUp(self):
        self.circuit = BaseQuantumCircuit()

    def test_init(self):
        self.assertIsInstance(self.circuit, torch.nn.Module)

    def test_add_qubit(self):
        q0 = Zero()
        self.circuit.add_qubit(q0)
        self.assertEqual(len(self.circuit.qubit_list), 1)

    def test_add_gate(self):
        self.circuit.add_qubit(Zero())
        gate = H()
        self.circuit.add_gate(gate, target=0)
        self.assertEqual(len(self.circuit.gate_queue), 1)

    def test_single_qubit_h(self):
        self.circuit.add_qubit(Zero())
        self.circuit.add_gate(H(), target=0)
        result = self.circuit.forward()
        probs = torch.real(result * torch.conj(result)).squeeze()
        self.assertTrue(torch.allclose(probs, torch.tensor([0.5, 0.5]), atol=1e-4))

if __name__ == '__main__':
    unittest.main()
