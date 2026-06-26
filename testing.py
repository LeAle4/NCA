"""
testing.py — Evaluation script to load a trained NCA model and its JSON configuration
and test it against the Swift–Hohenberg PDE using each starting condition in patterns.py.
Saves comparison plots for all initial conditions in a dedicated model directory.
"""

import time
import numpy as np
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

from config import DEVICE, STD_DTYPE
from equations import F_swift_hohen
from model import NCA, evolve
from edp import simulate_edp

# Import the pattern generator functions
import patterns

# ═══════════════════════════════════════════════════════════════════════════
# Configurations
# ═══════════════════════════════════════════════════════════════════════════
SEED = 42
EXTRA_T_FACTOR = 2
MODEL_NAME = "ide_bih-full-True-8-0.5-8-0.1"
MODELS_DIR = "models"
OUTPUT_DIR = os.path.join("evaluation", MODEL_NAME)

# Load configuration and weights
config_path = os.path.join(MODELS_DIR, f"{MODEL_NAME}_config.json")
model_path = os.path.join(MODELS_DIR, f"{MODEL_NAME}.pt")

if not os.path.exists(config_path):
    raise FileNotFoundError(f"Configuration file not found: {config_path}")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

with open(config_path, "r") as f:
    config = json.load(f)

print(f"[Testing] Loaded configuration for {MODEL_NAME}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# Load Model
# ═══════════════════════════════════════════════════════════════════════════
# Use loading method from NCA or load manually
try:
    nca = NCA.from_file(model_path)
except Exception:
    # Fallback to manual initialization and loading weights
    nca = NCA(
        input_shape=(1, config["nca_channels"], config["size"], config["size"]),
        model_size=tuple(config["model_size"]),
        perception_kernels_names=tuple(config["kernels"]),
        periodic=config["periodic"]
    )
    nca.load_state_dict(torch.load(model_path, map_location=DEVICE).state_dict())

nca.eval()
print(f"[Testing] Model loaded successfully on {DEVICE}")

# ═══════════════════════════════════════════════════════════════════════════
# Generate initial conditions to test
# ═══════════════════════════════════════════════════════════════════════════
shape = (1, config["pde_channels"], config["size"], config["size"])

torch.manual_seed(SEED)
np.random.seed(SEED)
test_patterns = {
    "random_noise": patterns.make_random_noise(shape, scale=0.1),
    "gaussian_bump": patterns.make_gaussian_bump(shape, sigma=5.0, amplitude=1.0),
    "multi_gaussian_bumps": patterns.make_multiple_gaussian_bumps(shape, num_bumps=4, sigma_range=(3.0, 6.0)),
    "cosine_waves": patterns.make_cos_sine_waves(shape, frequency=2.0, amplitude=1.0, mode="both"),
    "step_function": patterns.make_step_function(shape, radius=0.25, value_inside=1.0, value_outside=0.0),
    "striped_pattern": patterns.make_striped_pattern(shape, stripe_width=6, orientation="horizontal"),
    "complete": patterns.complete_pattern(shape, weights=(0.05, 0.05, 0.8, 0.2))
}

# ═══════════════════════════════════════════════════════════════════════════
# Testing loop over each pattern
# ═══════════════════════════════════════════════════════════════════════════
T = config["T"]
DT = config["dt"]
STEP_MUL = config["step_mul"]
SH_R = config.get("sh_r", -0.1)
ITER_N = config["iter_n"]

results_summary = {}

# ═══════════════════════════════════════════════════════════════════════════
# Generalization testing
# ═══════════════════════════════════════════════════════════════════════════

T_test = int(T*EXTRA_T_FACTOR)
total_pde_steps = T_test * STEP_MUL
total_nca_steps = T_test - 1

for name, x0 in test_patterns.items():
    print(f"\n--- Evaluating pattern: {name} ---")
    
    # 1. Run PDE simulation (ground truth)
    pde_trajectory = simulate_edp(x0, F_swift_hohen, t=total_pde_steps, dt=DT, r=SH_R)
    pde_trajectory = pde_trajectory[::STEP_MUL]  # Subsample
    
    # Normalize PDE simulation as the trainer did
    pde_min = pde_trajectory.min()
    pde_max = pde_trajectory.max()
    pde_range = pde_max - pde_min
    if pde_range > 0:
        pde_normalized = (pde_trajectory - pde_min) / pde_range
    else:
        pde_normalized = torch.zeros_like(pde_trajectory)
    
    pde_normalized = pde_normalized.to(DEVICE, dtype=STD_DTYPE)
    
    # 2. Run NCA evolution
    # Prep the initial condition (the first normalized frame from PDE)
    ic = pde_normalized[0].clone()  # (1, C, H, W)
    
    C_pde = ic.shape[1]
    C_nca = nca.C
    if C_nca > C_pde:
        hidden = torch.zeros(
            ic.shape[0], C_nca - C_pde, ic.shape[2], ic.shape[3],
            device=DEVICE, dtype=STD_DTYPE
        )
        ic = torch.cat([ic, hidden], dim=1)
        
    n_eval_steps = total_nca_steps
    with torch.no_grad():
        #Time the evolution
        start = time.time()
        nca_trajectory = evolve(nca, ic, iters=n_eval_steps, dt=DT)
        end = time.time()
        print(f"NCA evolution time: {end - start} | {T_test/(end-start)} it/s")
        
    # Squeeze / extract channels to match comparison
    nca_traj = nca_trajectory[:, :, :C_pde, :, :].squeeze(1).cpu()  # (T-1, C, H, W)
    pde_traj = pde_normalized[1:].squeeze(1).cpu()                 # (T-1, C, H, W)
    
    n_frames = min(nca_traj.shape[0], pde_traj.shape[0])
    nca_traj = nca_traj[:n_frames]
    pde_traj = pde_traj[:n_frames]
    
    # 3. Calculate metrics
    per_frame_mse = ((nca_traj - pde_traj) ** 2).mean(dim=(1, 2, 3)).numpy()
    overall_mse = per_frame_mse.mean()
    overall_mae = (nca_traj - pde_traj).abs().mean().item()
    
    results_summary[name] = {
        "overall_mse": float(overall_mse),
        "overall_mae": float(overall_mae),
        "peak_mse": float(per_frame_mse.max())
    }
    
    print(f"  MSE: {overall_mse:.6f} | MAE: {overall_mae:.6f}")
    
    # 4. Generate Plot
    n_snapshots = min(5, n_frames)
    snap_idx = np.linspace(0, n_frames - 1, n_snapshots, dtype=int)
    
    fig, axes = plt.subplots(3, n_snapshots, figsize=(3.5 * n_snapshots, 9))
    vmin = min(pde_traj.min().item(), nca_traj.min().item())
    vmax = max(pde_traj.max().item(), nca_traj.max().item())
    
    for j, t in enumerate(snap_idx):
        # PDE Row
        im_pde = axes[0, j].imshow(pde_traj[t, 0].numpy(), cmap="viridis", vmin=vmin, vmax=vmax)
        axes[0, j].set_title(f"PDE t={t}")
        axes[0, j].axis("off")
        
        # NCA Row
        axes[1, j].imshow(nca_traj[t, 0].numpy(), cmap="viridis", vmin=vmin, vmax=vmax)
        axes[1, j].set_title(f"NCA t={t}")
        axes[1, j].axis("off")
        
        # Error Row
        err = (pde_traj[t, 0] - nca_traj[t, 0]).abs().numpy()
        im_err = axes[2, j].imshow(err, cmap="hot")
        axes[2, j].set_title(f"|Err| t={t}")
        axes[2, j].axis("off")
        
    axes[0, 0].set_ylabel("PDE", fontsize=12)
    axes[1, 0].set_ylabel("NCA", fontsize=12)
    axes[2, 0].set_ylabel("|Error|", fontsize=12)
    
    fig.suptitle(f"Swift-Hohenberg Comparison: Pattern '{name}'", fontsize=14, fontweight="bold")
    fig.tight_layout()
    
    # Save the figure
    plot_path = os.path.join(OUTPUT_DIR, f"pattern_{name}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Saved figure to {plot_path}")

# ═══════════════════════════════════════════════════════════════════════════
# Save summary metrics
# ═══════════════════════════════════════════════════════════════════════════
summary_path = os.path.join(OUTPUT_DIR, "patterns_evaluation_summary.json")
with open(summary_path, "w") as f:
    json.dump(results_summary, f, indent=4)
print(f"\n[Testing] Finished! Saved summary performance to {summary_path}")
