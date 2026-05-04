import unittest
import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from QuantumSimulation.src.Qubits import Qubit, Zero, One, Plus, Minus


class TestQubits(unittest.TestCase):
    def test_zero_qubit_values(self):
        q = Zero()
        expected = torch.tensor([[[1.0], [0.0]]], dtype=torch.complex64)
        self.assertTrue(torch.allclose(q.state, expected))

    def test_zero_qubit_shape(self):
        q = Zero()
        self.assertEqual(q.state.shape, (1, 2, 1))

    def test_one_qubit_values(self):
        q = One()
        expected = torch.tensor([[[0.0], [1.0]]], dtype=torch.complex64)
        self.assertTrue(torch.allclose(q.state, expected))

    def test_one_qubit_shape(self):
        q = One()
        self.assertEqual(q.state.shape, (1, 2, 1))

    def test_plus_qubit_values(self):
        q = Plus()
        inv_sqrt2 = 1.0 / torch.sqrt(torch.tensor(2.0))
        expected = torch.tensor([[[inv_sqrt2], [inv_sqrt2]]], dtype=torch.complex64)
        self.assertTrue(torch.allclose(q.state, expected))

    def test_minus_qubit_values(self):
        q = Minus()
        inv_sqrt2 = 1.0 / torch.sqrt(torch.tensor(2.0))
        expected = torch.tensor([[[inv_sqrt2], [-inv_sqrt2]]], dtype=torch.complex64)
        self.assertTrue(torch.allclose(q.state, expected))

    def test_custom_qubit_real_normalized(self):
        q = Qubit([3, 4])
        expected = torch.tensor([[[0.6], [0.8]]], dtype=torch.complex64)
        self.assertTrue(torch.allclose(q.state, expected))

    def test_custom_qubit_complex(self):
        q = Qubit([1j, 0])
        expected = torch.tensor([[[1.0j], [0.0]]], dtype=torch.complex64)
        self.assertTrue(torch.allclose(q.state, expected))

    def test_custom_qubit_from_tensor(self):
        inp = torch.tensor([1.0, 1.0])
        q = Qubit(inp)
        inv_sqrt2 = 1.0 / torch.sqrt(torch.tensor(2.0))
        expected = torch.tensor([[[inv_sqrt2], [inv_sqrt2]]], dtype=torch.complex64)
        self.assertTrue(torch.allclose(q.state, expected))

    def test_all_qubits_are_complex(self):
        for QubitClass in [Zero, One, Plus, Minus]:
            q = QubitClass()
            self.assertTrue(torch.is_complex(q.state))

    def test_qubit_norm(self):
        for QubitClass in [Zero, One, Plus, Minus]:
            q = QubitClass()
            norm = torch.sqrt((q.state * q.state.conj()).sum().real)
            self.assertTrue(torch.allclose(norm, torch.tensor(1.0)))

    def test_zero_one_orthonormal(self):
        z = Zero()
        o = One()
        inner = (z.state.conj() * o.state).sum()
        self.assertTrue(torch.allclose(inner.real, torch.tensor(0.0)))
        self.assertTrue(torch.allclose(inner.imag, torch.tensor(0.0)))

    def test_qubit_device(self):
        q = Zero(device='cpu')
        self.assertEqual(q.device, 'cpu')


    def test_measure_zero(self):
        z = Zero()
        outcome, probs, collapsed = z.measure()
        self.assertEqual(outcome, 0)
        self.assertTrue(torch.allclose(probs, torch.tensor([1.0, 0.0])))

    def test_measure_plus(self):
        p = Plus()
        outcome, probs, collapsed = p.measure()
        self.assertTrue(torch.allclose(probs, torch.tensor([0.5, 0.5])))
        self.assertIn(collapsed.squeeze().abs().max(), [1.0, 1.0])  # |0> or |1>

    def test_measure_hadamard_basis(self):
        p = Plus()
        basis = [Zero(), One()]
        outcome, probs, collapsed = p.measure(basis=basis)
        self.assertTrue(torch.allclose(probs, torch.tensor([0.5, 0.5])))

if __name__ == '__main__':
    unittest.main()


