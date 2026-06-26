import torch
import torch.nn.functional as F

from tqdm import tqdm

from config import DEVICE, STD_DTYPE
from kernels import get_kernels

DIFFERENTIAL = get_kernels(("sobel", "laplacian"))
LAPLACIAN = get_kernels(("laplacian",))

def compute_derivatives(X: torch.Tensor, periodic = True):
    """
    X: (B,C,H,W)

    Returns:
        dx  : Sobel x derivative
        dy  : Sobel y derivative
        lap : Laplacian
    """

    B, C, H, W = X.shape
    DIFF_pad = DIFFERENTIAL.shape[-1]//2

    if periodic:
        X = F.pad(X, (DIFF_pad,DIFF_pad,DIFF_pad,DIFF_pad), mode="circular")
        padding = 0
    else:
        padding = "same"

    kernel = DIFFERENTIAL.repeat((C,1,1,1))
    out = F.conv2d(X, kernel, padding=padding, groups = C)

    return out

def laplacian(X: torch.Tensor, periodic = True) -> torch.Tensor:
    """
    X: (B,C,H,W)

    Returns:
        lap : Laplacian value across space
    """
    B, C, H, W = X.shape
    LAP_pad = LAPLACIAN.shape[-1]//2
    if periodic:
        X = F.pad(X, (LAP_pad,LAP_pad,LAP_pad,LAP_pad), mode="circular")
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
    filters = DIFFERENTIAL.shape[0]
    Y = torch.zeros((t,)+shape, dtype= STD_DTYPE, device=DEVICE)
    X = X0.clone().to(DEVICE)

    for i in tqdm(range(t), ncols = 100, desc = f"Simulating {f.__name__}"):
        diffs = compute_derivatives(X)
        diffs = diffs.view(B, filters, C, W, H)

        dx  = diffs[:, 0]
        dy  = diffs[:, 1]
        lap = diffs[:, 2]
        
        X = f(X, dx, dy, lap, dt = dt, **kwargs)
        Y[i] = X.clone()
    
    return Y