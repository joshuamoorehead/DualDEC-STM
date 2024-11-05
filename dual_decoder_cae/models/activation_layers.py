import torch
import torch.nn as nn
import torch.nn.functional as F

class Swish(nn.Module):
    """
    Swish activation function: x * sigmoid(x)
    """
    def forward(self, x):
        return x * torch.sigmoid(x)

class Mish(nn.Module):
    """
    Mish activation function: x * tanh(softplus(x))
    """
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class GELU(nn.Module):
    """
    Gaussian Error Linear Unit
    """
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0/torch.pi)) * 
                                       (x + 0.044715 * torch.pow(x, 3))))

class AdaptiveActivation(nn.Module):
    """
    Learnable activation function that adapts during training
    """
    def __init__(self, num_params=3):
        super().__init__()
        self.params = nn.Parameter(torch.ones(num_params))
        
    def forward(self, x):
        # Combine different activation functions with learnable weights
        return (self.params[0] * F.relu(x) +
                self.params[1] * torch.tanh(x) +
                self.params[2] * torch.sigmoid(x))

class GLU(nn.Module):
    """
    Gated Linear Unit
    """
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim
        
    def forward(self, x):
        x, gates = torch.chunk(x, 2, dim=self.dim)
        return x * torch.sigmoid(gates)

class Snake(nn.Module):
    """
    Snake activation function: x + (1/a) * sin^2(ax)
    """
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, x):
        return x + (1.0/self.alpha) * torch.pow(torch.sin(self.alpha * x), 2)

def get_activation(activation_type):
    """
    Factory function to get activation layers
    """
    activations = {
        'relu': nn.ReLU(),
        'leaky_relu': nn.LeakyReLU(0.2),
        'swish': Swish(),
        'mish': Mish(),
        'gelu': GELU(),
        'adaptive': AdaptiveActivation(),
        'glu': GLU(),
        'snake': Snake(),
        'tanh': nn.Tanh(),
        'sigmoid': nn.Sigmoid()
    }
    
    return activations.get(activation_type.lower(), nn.ReLU())

class ActivationScheduler:
    """
    Scheduler to dynamically switch between activation functions during training
    """
    def __init__(self, model, schedule):
        self.model = model
        self.schedule = schedule  # Dict of epoch: activation_type
        
    def step(self, epoch):
        if epoch in self.schedule:
            new_activation = self.schedule[epoch]
            self._update_activations(self.model, new_activation)
    
    def _update_activations(self, module, activation_type):
        """Recursively update activation functions in the model"""
        for name, child in module.named_children():
            if isinstance(child, (nn.ReLU, nn.LeakyReLU, Swish, Mish, GELU,
                                AdaptiveActivation, GLU, Snake)):
                setattr(module, name, get_activation(activation_type))
            else:
                self._update_activations(child, activation_type)
