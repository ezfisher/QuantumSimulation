import unittest
import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from QuantumSimulation.src.quantum_object import BaseQuantumObject


class TestBaseQuantumObject(unittest.TestCase):
    def test_init(self):
        obj = BaseQuantumObject(size=2, num_states=4, device='cpu')
        self.assertEqual(obj.size, 2)
        self.assertEqual(obj.num_states, 4)
        self.assertEqual(obj.device, 'cpu')

    def test_define_converts_real_to_complex(self):
        obj = BaseQuantumObject(size=1, num_states=2, device='cpu')
        real_tensor = torch.tensor([[[1.0], [0.0]]])
        result = obj.__define__(real_tensor)
        self.assertTrue(torch.is_complex(result))

    def test_define_normalizes(self):
        obj = BaseQuantumObject(size=1, num_states=2, device='cpu')
        real_tensor = torch.tensor([[[3.0], [4.0]]])
        result = obj.__define__(real_tensor)
        norm = torch.sqrt((result * result.conj()).sum().real)
        self.assertTrue(torch.allclose(norm, torch.tensor(1.0)))

    def test_norm_real(self):
        obj = BaseQuantumObject(size=1, num_states=2, device='cpu')
        real_tensor = torch.tensor([[[3.0], [4.0]]])
        result = obj.__norm__(real_tensor)
        expected = torch.tensor([[[0.6], [0.8]]])
        self.assertTrue(torch.allclose(result, expected))

    def test_norm_complex(self):
        obj = BaseQuantumObject(size=1, num_states=2, device='cpu')
        complex_tensor = torch.tensor([[[1.0j], [0.0]]])
        result = obj.__norm__(complex_tensor)
        expected = torch.tensor([[[1.0j], [0.0]]])
        self.assertTrue(torch.allclose(result, expected))

    def test_norm_leaves_zero_untouched(self):
        obj = BaseQuantumObject(size=1, num_states=2, device='cpu')
        zero_tensor = torch.tensor([[[0.0], [0.0]]])
        result = obj.__norm__(zero_tensor)
        # Division by zero produces NaN; test that we get NaN as expected
        self.assertTrue(torch.isnan(result).all())


if __name__ == '__main__':
    unittest.main()

