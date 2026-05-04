import unittest
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.quantum_object import BaseQuantumObject, BaseQubit, BaseOperator

class TestBaseClasses(unittest.TestCase):
    def test_base_init(self):
        obj = BaseQuantumObject(size=2, num_states=4, device='cpu')
        self.assertEqual(obj.size, 2)
        self.assertEqual(obj.num_states, 4)
        self.assertEqual(obj.device, 'cpu')

    def test_qubit_set_state_list(self):
        qb = BaseQubit(size=1, num_states=2)
        qb.set_state([3, 4])
        expected_norm = torch.tensor([0.6, 0.8], dtype=torch.float32)
        self.assertTrue(torch.allclose(qb.state.squeeze().real, expected_norm))

    def test_qubit_set_state_tensor(self):
        qb = BaseQubit(size=1, num_states=2)
        qb.set_state(torch.tensor([1j, 0]))
        self.assertTrue(torch.allclose(qb.state, torch.tensor([[[1j], [0]]])))

    def test_operator_set_gate(self):
        op = BaseOperator(size=1, num_states=2)
        op.set_gate([[0,1],[1,0]])
        expected = torch.tensor([[[0,1],[1,0]]], dtype=torch.complex64)
        self.assertTrue(torch.allclose(op.gate, expected))

    def test_shapes(self):
        qb = BaseQubit(1,2)
        qb.set_state([1,0])
        self.assertEqual(qb.state.shape, (1,2,1))
        op = BaseOperator(1,2)
        op.set_gate([[1,0],[0,1]])
        self.assertEqual(op.gate.shape, (1,2,2))

if __name__ == '__main__':
    unittest.main()
