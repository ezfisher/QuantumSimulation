import torch
import src.quantum_object as qo

class Gate(qo.BaseQuantumObject):
    def __init__(self, gate, size, device='cpu'):
        super().__init__(size=1, num_states=2, device=device)
        self.num_states = 2
        self.size = size
        self.device = device
        self.gate = self.define(gate)
    
    def define(self, gate):
        if not isinstance(gate, torch.Tensor):
            gate = torch.Tensor((gate)).reshape((1, self.num_states**self.size, -1))
        return super().__define__(gate)

class H(qo.BaseQuantumObject):
    def __init__(self, device='cpu'):
        super().__init__(size=1, num_states=2, device=device)
        self.num_states = 2
        self.size = 1
        self.device = device
        self.gate = self.define()
    
    def define(self):
        gate = torch.Tensor(((1, 1), (1, -1))).reshape((1, 2, 2))
        return super().__define__(gate)

class Y(qo.BaseQuantumObject):
    def __init__(self, device='cpu'):
        super().__init__(size=1, num_states=2, device=device)
        self.num_states = 2
        self.size = 1
        self.device = device
        self.gate = self.define()
    
    def define(self):
        gate_re = torch.Tensor(((0, 0), (0, 0))).reshape((1, 2, 2))
        gate_im = torch.Tensor(((0, -1), (1, 0))).reshape((1, 2, 2))
        gate = torch.complex(gate_re, gate_im)
        return super().__define__(gate)

class X(qo.BaseQuantumObject):
    def __init__(self, device='cpu'):
        super().__init__(size=1, num_states=2, device=device)
        self.num_states = 2
        self.size = 1
        self.device = device
        self.gate = self.define()
    
    def define(self):
        gate = torch.Tensor(((0, 1), (1, 0))).reshape((1, 2, 2))
        return super().__define__(gate)

class Z(qo.BaseQuantumObject):
    def __init__(self, device='cpu'):
        super().__init__(size=1, num_states=2, device=device)
        self.num_states = 2
        self.size = 1
        self.device = device
        self.gate = self.define()
    
    def define(self):
        gate = torch.Tensor(((1, 0), (0, -1))).reshape((1, 2, 2))
        return super().__define__(gate)