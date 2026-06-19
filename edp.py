import torch
import torch.nn.functional as F

from tqdm import tqdm

from config import DEVICE, STD_DTYPE
from kernels import get_kernels

DIFFERENTIAL = get_kernels(("sobel", "laplacian"))
LAPLACIAN = get_kernels(("laplacian",))

def compute_derivatives(X: torch.Tensor, periodic = False):
    """
    X: (B,C,H,W)

    Returns:
        dx  : Sobel x derivative
        dy  : Sobel y derivative
        lap : Laplacian
    """

    B, C, H, W = X.shape

    if periodic:
        X = F.pad(X, (1,1,1,1), mode="circular")
        padding = 0
    else:
        padding = "same"

    kernel = DIFFERENTIAL.repeat((C,1,1,1))
    out = F.conv2d(X, kernel, padding=padding, groups = C)

    return out

def laplacian(X: torch.Tensor, periodic = False) -> torch.Tensor:
    """
    X: (B,C,H,W)

    Returns:
        lap : Laplacian
    """
    B, C, H, W = X.shape
    if periodic:
        X = F.pad(X, (1,1,1,1), mode="circular")
        padding = 0
    else:
        padding = "same"

    kernel = LAPLACIAN.repeat((C,1,1,1))
    out = F.conv2d(X, kernel, padding=padding, groups = C)
    return out

def simulate_edp(X0, f, t:int, dt: float = 0.001, **kwargs)-> torch.Tensor:
    """
    X: (1,C,H,W)
    
    Returns:

    Y: (iter, C, H, W)
    """
    
    shape = X0.shape
    B, C, W, H = shape
    Y = torch.zeros((t,)+shape, dtype= STD_DTYPE, device=DEVICE)
    X = X0.clone().to(DEVICE)

    for i in tqdm(range(t), ncols = 100, desc = f"Simulating {f.__name__}"):
        diffs = compute_derivatives(X)
        diffs = diffs.view(B, 3, C, W, H)

        dx  = diffs[:, 0]
        dy  = diffs[:, 1]
        lap = diffs[:, 2]
        
        X = f(X, dx, dy, lap, dt = dt, **kwargs)
        Y[i] = X.clone()
    
    return Y