import unittest
import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from QuantumSimulation.src.Operators import H, X, Y, Z, Gate
from QuantumSimulation.src.Qubits import Zero, One, Plus


class TestOperators(unittest.TestCase):
    def test_x_gate_values(self):
        g = X()
        expected = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]], dtype=torch.complex64)
        self.assertTrue(torch.allclose(g.gate, expected))

    def test_x_gate_shape(self):
        g = X()
        self.assertEqual(g.gate.shape, (1, 2, 2))

    def test_z_gate_values(self):
        g = Z()
        expected = torch.tensor([[[1.0, 0.0], [0.0, -1.0]]], dtype=torch.complex64)
        self.assertTrue(torch.allclose(g.gate, expected))

    def test_h_gate_values(self):
        g = H()
        inv_sqrt2 = 1.0 / torch.sqrt(torch.tensor(2.0))
        expected = torch.tensor(
            [[[inv_sqrt2, inv_sqrt2], [inv_sqrt2, -inv_sqrt2]]],
            dtype=torch.complex64,
        )
        self.assertTrue(torch.allclose(g.gate, expected))

    def test_y_gate_values(self):
        g = Y()
        expected = torch.tensor(
            [[[0.0, -1.0j], [1.0j, 0.0]]],
            dtype=torch.complex64,
        )
        self.assertTrue(torch.allclose(g.gate, expected))

    def test_y_gate_is_complex(self):
        g = Y()
        self.assertTrue(torch.is_complex(g.gate))

    def test_all_gates_are_complex(self):
        for GateClass in [H, X, Y, Z]:
            g = GateClass()
            self.assertTrue(torch.is_complex(g.gate))

    def test_custom_gate_identity(self):
        g = Gate([[1, 0], [0, 1]], size=1)
        expected = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.complex64)
        self.assertTrue(torch.allclose(g.gate, expected))

    def test_custom_gate_unnormalized(self):
        g = Gate([[2, 0], [0, 2]], size=1)
        expected = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]], dtype=torch.complex64)
        self.assertTrue(torch.allclose(g.gate, expected))


    def test_custom_gate_shape(self):
        g = Gate([[1, 0], [0, 1]], size=1)
        self.assertEqual(g.gate.shape, (1, 2, 2))

    def test_x_on_zero(self):
        x = X()
        z = Zero()
        result = torch.matmul(x.gate, z.state)
        expected = One().state
        self.assertTrue(torch.allclose(result, expected))

    def test_h_on_zero(self):
        h = H()
        z = Zero()
        result = torch.matmul(h.gate, z.state)
        expected = Plus().state
        self.assertTrue(torch.allclose(result, expected))

    def test_gate_device(self):
        g = X(device='cpu')
        self.assertEqual(g.device, 'cpu')


if __name__ == '__main__':
    unittest.main()

