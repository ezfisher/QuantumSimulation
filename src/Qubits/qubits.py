import torch
import src.quantum_object as qo

class Qubit(qo.BaseQuantumObject):
    def __init__(self, inp_state, num_states=2, adjoint=False, device = 'cpu'):
        super().__init__(size=1, num_states=num_states, device=device)
        self.num_states = num_states
        self.adjoint = adjoint
        self.device = device
        self.state = self.define(inp_state)
    
    def define(self, inp_state):
        if not isinstance(inp_state, torch.Tensor):
            inp_state = torch.Tensor((inp_state)).reshape((1, self.num_states, 1))
        return super().__define__(inp_state)
    
class Zero(qo.BaseQuantumObject):
    def __init__(self, device='cpu'):
        super().__init__(size=1, num_states=1, device=device)
        self.num_states = 2
        self.size = 1
        self.device = device
        self.state = self.define()
    
    def define(self):
        inp_state = torch.Tensor((1, 0)).reshape((1, self.num_states, 1))
        return super().__define__(inp_state)

class One(qo.BaseQuantumObject):
    def __init__(self, device='cpu'):
        super().__init__(size=1, num_states=1, device=device)
        self.num_states = 2
        self.size = 1
        self.device = device
        self.state = self.define()
    
    def define(self):
        inp_state = torch.Tensor((0, 1)).reshape((1, self.num_states, 1))
        return super().__define__(inp_state)

class Plus(qo.BaseQuantumObject):
    def __init__(self, device='cpu'):
        super().__init__(size=1, num_states=2, device=device)
        self.num_states = 2
        self.size = 1
        self.device = device
        self.state = self.define()
    
    def define(self):
        inp_state = torch.Tensor((1, 1)).reshape((1, self.num_states, 1))
        return super().__define__(inp_state)
    
class Minus(qo.BaseQuantumObject):
    def __init__(self, device='cpu'):
        super().__init__(size=1, num_states=2, device=device)
        self.num_states = 2
        self.size = 1
        self.device = device
        self.state = self.define()
    
    def define(self):
        inp_state = torch.Tensor((1, -1)).reshape((1, self.num_states, 1))
        return super().__define__(inp_state)