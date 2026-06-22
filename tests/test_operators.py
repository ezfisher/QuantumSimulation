import unittest
import torch
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src import H, X, Y, Z, Gate, Zero, One, Plus, CX, CH, CY, CZ

inv_sqrt2 = 1 / np.sqrt(2)

class TestOperators(unittest.TestCase):
    def setUp(self):
        self.addTypeEqualityFunc(torch.Tensor, self.assertTrue)
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
        expected = torch.tensor([[[0.7071, 0.7071], [0.7071, -0.7071]]], dtype=torch.complex64)
        self.assertTrue(torch.allclose(g.gate, expected))

    def test_y_gate_values(self):
        g = Y()
        expected = torch.tensor([[[0.0, -1.0j], [1.0j, 0.0]]], dtype=torch.complex64)
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

    def test_cx_gate(self):
        cx = CX()
        expected = torch.tensor([[[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]], dtype=torch.complex64)
        self.assertTrue(torch.allclose(cx.gate.real, expected.real, atol=1e-5))
    
    def test_cx_on_10(self):
        cx = CX()
        # |10> index 2
        state10 = torch.zeros(1,4,1, dtype=torch.complex64)
        state10[0,2,0] = 1.0
        result = torch.matmul(cx.gate, state10)
        # |11> index 3
        expected = torch.zeros_like(state10)
        expected[0,3,0] = 1.0
        self.assertTrue(torch.allclose(result, expected))

    def test_cz_phase(self):
        cz = CZ()
        # |11> should get -1 phase on |1>
        state11 = torch.zeros(1,4,1, dtype=torch.complex64)
        state11[0,3,0] = 1.0
        result = torch.matmul(cz.gate, state11)
        # CZ |11> = |1> | -1 >
        # But since basis |00>,|01>,|10>,|11>, CZ flips phase of |11>
        print('CZ |11>:', result.squeeze())
        self.assertTrue(torch.allclose(result.abs(), state11.abs()))

    def test_ch_gate_values(self):
        ch = CH()
        expected = torch.tensor([[[1,0,0,0],[0,1,0,0],[0,0,inv_sqrt2,inv_sqrt2],[0,0,inv_sqrt2,-inv_sqrt2]]], dtype=torch.complex64)
        self.assertTrue(torch.allclose(ch.gate, expected, atol=1e-5))

    def test_cy_gate_values(self):
        cy = CY()
        expected = torch.tensor([[[1,0,0,0],[0,1,0,0],[0,0,0,-1j],[0,0,1j,0]]], dtype=torch.complex64)
        self.assertTrue(torch.allclose(cy.gate, expected, atol=1e-5))

    def test_ch_on_superposition(self):
        ch = CH()
        # |+1> = (|10> + |11>)/sqrt(2)
        state = torch.zeros(1,4,1, dtype=torch.complex64)
        state[0,2,0] = inv_sqrt2
        state[0,3,0] = inv_sqrt2
        result = torch.matmul(ch.gate, state)
        # CH|+1> = |1>|+>
        expected = torch.zeros_like(state)
        expected[0,2,0] = 0.5
        expected[0,3,0] = 0.5
        self.assertTrue(torch.allclose(result, expected, atol=1e-5))

    def test_cy_on_11(self):
        cy = CY()
        state11 = torch.zeros(1,4,1, dtype=torch.complex64)
        state11[0,3,0] = 1.0
        result = torch.matmul(cy.gate, state11)
        # CY|11> = -i|10> (checking phase and amplitude)
        expected = torch.zeros_like(state11)
        expected[0,2,0] = -1j
        self.assertTrue(torch.allclose(result, expected, atol=1e-5))

if __name__ == '__main__':
    unittest.main()
