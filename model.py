import torch
import torch.nn as nn
import torch.nn.functional as F

from kernels import get_kernels
from config import DEVICE, STD_DTYPE

class NCA(nn.Module):

    def __init__(self, input_shape, model_size:tuple[int] = (128,), perception_kernels_names:tuple[str] = ("sobel","laplacian"), device=DEVICE, periodic=False):
        """
        Args:
            input_shape: (B, C, H, W) 
            perception_kernels_names: Names of the kernels to use for perception
            device: Device to use (CPU or GPU)
            periodic: Whether to use periodic boundary conditions
        """
        super().__init__()
        self.device = device
        self.periodic = periodic

        if self.periodic:
            self.padding = 0
        else:
            self.padding = "same"

        self.B, self.C, self.H, self.W = input_shape

        diffs = get_kernels(perception_kernels_names)
        self.num_kernels = diffs.shape[0]

        perception_kernel = diffs.repeat((self.C,1,1,1))
        self.register_buffer("perception_kernel", perception_kernel)


        self.layers = []
        #Base layer that takes the perception kernel output
        self.layers.append(nn.Conv2d(self.C * self.num_kernels, model_size[0], kernel_size=1))
        self.layers.append(nn.ReLU())
        for i in range(1, len(model_size)):
            self.layers.append(nn.Conv2d(model_size[i-1], model_size[i], kernel_size=1))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Conv2d(model_size[-1], self.C, kernel_size=1))
        self.linear = nn.Sequential(*self.layers)
        
        self.reset_parameters()
        self.to(dtype=STD_DTYPE, device=DEVICE)

        self.config = {
            "input_shape": input_shape, 
            "model_size": model_size,
            "perception_kernels_names": perception_kernels_names,
            "num_kernels": self.num_kernels,
            "device": device,
            "periodic": periodic,
            "padding": self.padding,
            "num_layers": len(model_size),
        }

    def reset_parameters(self):
        """Recursively initialize the weights of the submodule layers"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _padding(self, X: torch.Tensor) -> torch.Tensor:
        """Pad the input tensor"""
        if self.periodic:
            # Check if any kernel is 5x5 (spatial size > 3)
            k_size = self.perception_kernel.shape[-1]
            pad_val = k_size // 2
            return F.pad(X, (pad_val, pad_val, pad_val, pad_val), mode="circular")
        else:
            return X
        
    def forward(self, X: torch.Tensor, dt:float = 0.001):
        out = self._padding(X)
        out = F.conv2d(out, self.perception_kernel, padding = self.padding, groups = self.C)
        out = self.linear(out)
        return dt * out

    @staticmethod
    def from_file(path:str):
        """Load from a file, weights_only = False necessary for perception kernel loading"""
        return torch.load(path, weights_only = False).to(device = DEVICE, dtype = STD_DTYPE)
    
    def save(self, path:str):
        """Save the model to a file"""
        torch.save(self, path)

def evolve(model: NCA, X0, iters, dt):
    """Run the NCA forward for `iters` steps.

    Args:
        model: The NCA model.
        X0: Initial state tensor (B, C, H, W) on device.
        iters: Number of simulation steps.
        dt: Time step size.

    Returns:
        Y: Trajectory tensor (iters, C, H, W) on CPU.
    """
    Y = torch.zeros((iters,) + X0.shape, dtype=STD_DTYPE)

    X = X0.clone()
    for i in range(iters):
        X = X + model(X, dt)
        Y[i] = X.clone()

    return Y