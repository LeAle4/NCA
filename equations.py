import torch
from edp import laplacian
import numpy as np
from scipy.fft import fft2, ifft2, fftfreq

def F_swift_hohen(X, Xdx, Xdy, Xdd, r=-0.1, dt = 0.001):
    Xdd2 = laplacian(Xdd)
    return X + dt * (
    (r - 1.0) * X
    - 2.0 * Xdd
    - Xdd2
    - X * X * X)

def F_swift_periodic(X: torch.Tensor, Xdx, Xdy, Xdd, dt:float = 0.001, r:float= -0.1)-> torch.Tensor:
    #Swift Hohenberg EDP implementation courtesy of Nan Li, Ph.D. candidate in Mathematics at the University of Minnesota.
    #Public access in:
    #https://github.com/eigenan/Dynamical-Bestiary-Swift-Hohenberg-Equation/blob/main/swift_hohenberg_2d_continuation.ipynb
    #Is assumes a periodic domain, so no special boundary conditions are needed.
    """
    Semi-implicit Euler step:
    (u_new - u)/dt = (r - (1 + del^2)^2) u_new - u^3
    => [1 - dt(r - (1 + del^2)^2)] u_new = u - dt * u^3
    """
    u = X
    Nx = X.shape[2]
    Ny = X.shape[3]
    Lx = X.shape[1]
    Ly = X.shape[2]

    kx = 2 * np.pi * fftfreq(Nx, d=Lx / Nx)
    ky = 2 * np.pi * fftfreq(Ny, d=Ly / Ny)

    KY, KX = np.meshgrid(ky, kx, indexing="ij")
    K2 = KX**2 + KY**2

    # Dealiasing mask (2/3 rule)
    kx_cut = (2.0 / 3.0) * np.max(np.abs(kx))
    ky_cut = (2.0 / 3.0) * np.max(np.abs(ky))
    mask = (np.abs(KX) <= kx_cut) & (np.abs(KY) <= ky_cut)
    
    device = X.device
    dtype = X.dtype
    
    mask_pt = torch.from_numpy(mask).to(device=device, dtype=dtype)
    denom = torch.from_numpy(1.0 - dt * (r - (1 - K2)**2)).to(device=device, dtype=dtype)
    
    u_hat = torch.fft.fft2(u, dim=(-2, -1))
    
    # Nonlinear term: u^3
    # Apply dealiasing to the nonlinear product
    nonlinear = u**3
    nonlinear_hat = torch.fft.fft2(nonlinear, dim=(-2, -1))
    nonlinear_hat = nonlinear_hat * mask_pt
    
    # Update
    # u_hat_new = (u_hat - dt * nonlinear_hat) / denom
    # Note: The equation is u_t = L u - u^3. 
    # Implicit linear: u_new - dt L u_new = u - dt u^3
    numer = u_hat - dt * nonlinear_hat
    u_hat_new = numer / denom
    
    return torch.real(torch.fft.ifft2(u_hat_new, dim=(-2, -1)))