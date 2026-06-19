import torch
import torch.nn as nn
import torch.nn.functional as F

from kernels import get_kernels
from config import DEVICE, STD_DTYPE

class NCA(nn.Module):

    def __init__(self, input_shape, perception_kernels_names:tuple[str] = ("sobel","laplacian"), obs_channels:int = 1, device=DEVICE, periodic=False):
        """
        Args:
            input_shape: (B, C, H, W) 
            perception_depth: The number of times we apply the perception kernels to the input tensor, the model recieves the first and second pass independently
            perception_kernels_names: Names of the kernels to use for perception
            obs_channels: Number of observation channels
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

        self.obs_channels = obs_channels
        self.B, self.C, self.H, self.W = input_shape

        diffs = get_kernels(perception_kernels_names)
        self.num_kernels = diffs.shape[0]

        perception_kernel = diffs.repeat((self.C,1,1,1))
        self.register_buffer("perception_kernel", perception_kernel)

        self.linear = nn.Sequential(
            nn.Conv2d(self.C * self.num_kernels, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, self.C, kernel_size=1)
        )
        
        self.reset_parameters()
        self.to(dtype=STD_DTYPE, device=DEVICE)

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
            return F.pad(X, (1,1,1,1), mode="circular")
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

def evolve(model: NCA, X0, iters, dt, store_every=1):
    """Run the NCA forward for `iters` steps.

    Args:
        model: The NCA model.
        X0: Initial state tensor (B, C, H, W) on device.
        iters: Number of simulation steps.
        dt: Time step size.
        store_every: Store a frame every N steps (1 = every step).

    Returns:
        Y: Trajectory tensor (num_stored, C, H, W) on CPU.
    """
    num_stored = iters // store_every
    # Store trajectory on CPU to keep GPU memory free.
    Y = torch.zeros((num_stored,) + X0.shape[1:], dtype=STD_DTYPE)

    X = X0.clone()
    with torch.no_grad():
        store_idx = 0
        for i in range(iters):
            X = X + model(X, dt)
            if (i + 1) % store_every == 0:
                Y[store_idx] = X.cpu()
                store_idx += 1

    return Y

if __name__ == "__main__":
    model = NCA((1,1,256,256), periodic = True, perception_kernels_names=("sobel","laplacian","biharmonic"))
    model.save("model.pt")
    model_2 = NCA.from_file("model.pt")
    print(model_2)
    x = torch.randn(1,1,256,256, device=DEVICE, dtype=STD_DTYPE)
    print(model_2(x))
    print(model_2.perception_kernel)
    