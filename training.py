"""
validation.py — Validation script for NCA training on the explicit-Euler
Swift–Hohenberg equation (F_swift_hohen).

Runs a training loop, saves hyperparameters to a JSON file, and compares
the NCA-predicted trajectory against the PDE ground truth.
"""

from patterns import make_multiple_gaussian_bumps, make_random_noise
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

from config import DEVICE, STD_DTYPE
from equations import F_swift_hohen
from model import NCA, evolve
from train import NCA_PDE_Trainer
from viz import plot_snapshots
from patterns import *

def kernels_to_name(kernels:list[str]):
    return "_".join([k[:3] for k in kernels])

# ═══════════════════════════════════════════════════════════════════════════
# Hyperparameters Configuration
# ═══════════════════════════════════════════════════════════════════════════
SEED            = 42
SIZE            = 64                      # Spatial resolution (H = W)
NCA_CHANNELS    = 8
PDE_CHANNELS    = 1                       # PDE channels (scalar field u)
T               = 1024                      # Number of kept PDE frames
STEP_MUL        = 1                       # PDE sub-steps per kept frame
DT              = 0.001                   # PDE integration time-step
ITER_N          = 8                       # NCA forward steps per training step

TRAIN_ITERS     = 32                   # Training iterations
LEARN_RATE      = 2e-3
NUM_BATCHES     = 1
BATCH_SIZE      = 16
OPTIMIZER       = "Nadam"
SCHEDULER_GAMMA = 0.9999
NOISE_FRAC      = 0.0
REG_COEFF       = 0.0
UPDATE_RATE     = 1.0
TRAIN_MODE      = "differential"
NORM_GRADS      = True

# NCA Model architecture configuration
MODEL_SIZE      = (128,)
#KERNELS         = ("identity", "sobel", "laplacian", "biharmonic")
KERNELS         = ("identity", "biharmonic")
PERIODIC        = True

# Swift–Hohenberg control parameter
SH_R            = 0.5

# Directories
TRY = 0
MODEL_NAME      = f"{kernels_to_name(KERNELS)}-{TRAIN_MODE}-{PERIODIC}-{ITER_N}-{SH_R}-{NCA_CHANNELS}-{NOISE_FRAC}-{TRY}"
DIRECTORY       = "models/"
LOG_DIR         = f"runs/{MODEL_NAME}"

# ═══════════════════════════════════════════════════════════════════════════
# Reproducibility
# ═══════════════════════════════════════════════════════════════════════════
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ═══════════════════════════════════════════════════════════════════════════
# Initial condition
# ═══════════════════════════════════════════════════════════════════════════
#x0 = 0.1 * torch.randn(1, C, SIZE, SIZE, device=DEVICE, dtype=STD_DTYPE)
x0 = 0.1 * make_random_noise(
    shape=(1, PDE_CHANNELS, SIZE, SIZE), device=DEVICE, dtype=STD_DTYPE
    ) + 0.6*make_multiple_gaussian_bumps(
    shape=(1, PDE_CHANNELS, SIZE, SIZE), 
    num_bumps=10,
    device=DEVICE, dtype=STD_DTYPE)

# ═══════════════════════════════════════════════════════════════════════════
# NCA model
# ═══════════════════════════════════════════════════════════════════════════
nca = NCA(
    input_shape=(1, NCA_CHANNELS, SIZE, SIZE),
    model_size=MODEL_SIZE,
    perception_kernels_names=KERNELS,
    periodic=PERIODIC,
)
n_params = sum(p.numel() for p in nca.parameters())
print(f"[Validation] NCA on {DEVICE}  |  parameters: {n_params:,}")

# ═══════════════════════════════════════════════════════════════════════════
# Save configuration to JSON file
# ═══════════════════════════════════════════════════════════════════════════
config_data = {
    "model_name": MODEL_NAME,
    "seed": SEED,
    "size": SIZE,
    "nca_channels": NCA_CHANNELS,
    "pde_channels": PDE_CHANNELS,
    "T": T,
    "step_mul": STEP_MUL,
    "dt": DT,
    "iter_n": ITER_N,
    "train_iters": TRAIN_ITERS,
    "learn_rate": LEARN_RATE,
    "batch_size": BATCH_SIZE,
    "optimizer": OPTIMIZER,
    "scheduler_gamma": SCHEDULER_GAMMA,
    "noise_frac": NOISE_FRAC,
    "reg_coeff": REG_COEFF,
    "update_rate": UPDATE_RATE,
    "train_mode": TRAIN_MODE,
    "norm_grads": NORM_GRADS,
    "model_size": MODEL_SIZE,
    "kernels": KERNELS,
    "periodic": PERIODIC,
    "sh_r": SH_R,
    "device": str(DEVICE),
    "n_parameters": n_params,
    "num_batches": NUM_BATCHES
}

os.makedirs(DIRECTORY, exist_ok=True)
config_path = os.path.join(DIRECTORY, f"{MODEL_NAME}_config.json")
with open(config_path, "w") as f:
    json.dump(config_data, f, indent=4)
print(f"[Validation] Saved hyperparameter configurations to {config_path}")

# ═══════════════════════════════════════════════════════════════════════════
# Build trainer (runs PDE solver internally)
# ═══════════════════════════════════════════════════════════════════════════
trainer = NCA_PDE_Trainer(
    nca_model=nca,
    x0=x0,
    F_pde=F_swift_hohen,
    T=T,
    N_BATCHES=NUM_BATCHES,
    step_mul=STEP_MUL,
    dt=DT,
    model_filename=MODEL_NAME,
    directory=DIRECTORY,
    log_dir=LOG_DIR,
    noise_frac=NOISE_FRAC,
    r=SH_R
)

# ═══════════════════════════════════════════════════════════════════════════
# Train
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  Training NCA on explicit-Euler Swift–Hohenberg")
print("=" * 60 + "\n")

trainer.train_sequence(
    TRAIN_ITERS=TRAIN_ITERS,
    iter_n=ITER_N,
    UPDATE_RATE=UPDATE_RATE,
    REG_COEFF=REG_COEFF,
    LEARN_RATE=LEARN_RATE,
    OPTIMIZER=OPTIMIZER,
    BATCH_SIZE=BATCH_SIZE,
    TRAIN_MODE=TRAIN_MODE,
    NORM_GRADS=NORM_GRADS,
    SCHEDULER_GAMMA=SCHEDULER_GAMMA,
)

# ═══════════════════════════════════════════════════════════════════════════
# Evaluate — roll out the NCA and compare to PDE ground truth
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  Evaluation")
print("=" * 60 + "\n")

nca.eval()

# Use the same IC the trainer used (first sample, normalised)
ic = trainer.data[0].clone().to(DEVICE)  # (1, C, H, W)

# Pad with hidden channels if NCA has more channels than the PDE
C_pde = ic.shape[1]
C_nca = nca.C
if C_nca > C_pde:
    hidden = torch.zeros(
        ic.shape[0], C_nca - C_pde, ic.shape[2], ic.shape[3],
        device=DEVICE, dtype=STD_DTYPE,
    )
    ic = torch.cat([ic, hidden], dim=1)

n_eval_steps = T - 1  # same horizon as the training data

with torch.no_grad():
    nca_frames = []
    X = ic.clone()
    for _ in range(n_eval_steps):
        for _ in range(ITER_N):
            X = X + nca(X, DT)
        nca_frames.append(X[:, :C_pde].clone().cpu())

# Stack into trajectory: (T-1, C, H, W)
nca_traj = torch.stack(nca_frames, dim=0).squeeze(1)   # (T-1, C, H, W)
pde_traj = trainer.data[1:].cpu().squeeze(1)            # (T-1, C, H, W)

# ── Per-frame MSE ─────────────────────────────────────────────────────────
n_frames = min(nca_traj.shape[0], pde_traj.shape[0])
nca_traj = nca_traj[:n_frames]
pde_traj = pde_traj[:n_frames]

per_frame_mse = ((nca_traj - pde_traj) ** 2).mean(dim=(1, 2, 3)).numpy()
overall_mse = per_frame_mse.mean()
overall_mae = (nca_traj - pde_traj).abs().mean().item()

print(f"  Overall MSE : {overall_mse:.6f}")
print(f"  Overall MAE : {overall_mae:.6f}")
print(f"  Peak MSE    : {per_frame_mse.max():.6f}  (frame {per_frame_mse.argmax()})")
print(f"  Min  MSE    : {per_frame_mse.min():.6f}  (frame {per_frame_mse.argmin()})")

# Save results to the JSON configuration file as well
try:
    with open(config_path, "r") as f:
        data = json.load(f)
    data["results"] = {
        "overall_mse": float(overall_mse),
        "overall_mae": float(overall_mae),
        "peak_mse": float(per_frame_mse.max()),
        "min_mse": float(per_frame_mse.min())
    }
    with open(config_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"[Validation] Appended validation results to {config_path}")
except Exception as e:
    print(f"[Validation] Warning: Could not append results to JSON: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# Plots
# ═══════════════════════════════════════════════════════════════════════════

# 1) Per-frame MSE curve
fig_mse, ax_mse = plt.subplots(figsize=(8, 4))
ax_mse.plot(per_frame_mse, linewidth=1.5, color="#2196F3")
ax_mse.set_xlabel("Frame")
ax_mse.set_ylabel("MSE")
ax_mse.set_title("Per-frame MSE  —  NCA vs PDE (Swift–Hohenberg)")
ax_mse.grid(True, alpha=0.3)
fig_mse.tight_layout()

plt.savefig(f"runs/{MODEL_NAME}/mse.png")

# 2) Side-by-side snapshot comparison
n_snapshots = min(5, n_frames)
snap_idx = np.linspace(0, n_frames - 1, n_snapshots, dtype=int)

fig_cmp, axes = plt.subplots(
    3, n_snapshots,
    figsize=(4 * n_snapshots, 10),
)

vmin = min(pde_traj.min().item(), nca_traj.min().item())
vmax = max(pde_traj.max().item(), nca_traj.max().item())

for j, t in enumerate(snap_idx):
    # PDE ground truth
    im_pde = axes[0, j].imshow(
        pde_traj[t, 0].numpy(), cmap="viridis", vmin=vmin, vmax=vmax,
    )
    axes[0, j].set_title(f"PDE  t={t}")
    axes[0, j].axis("off")

    # NCA prediction
    axes[1, j].imshow(
        nca_traj[t, 0].numpy(), cmap="viridis", vmin=vmin, vmax=vmax,
    )
    axes[1, j].set_title(f"NCA  t={t}")
    axes[1, j].axis("off")

    # Absolute error
    err = (pde_traj[t, 0] - nca_traj[t, 0]).abs().numpy()
    axes[2, j].imshow(err, cmap="hot")
    axes[2, j].set_title(f"|Error|  t={t}")
    axes[2, j].axis("off")

axes[0, 0].set_ylabel("PDE", fontsize=12)
axes[1, 0].set_ylabel("NCA", fontsize=12)
axes[2, 0].set_ylabel("|Error|", fontsize=12)

fig_cmp.suptitle(
    "Swift–Hohenberg: PDE Ground Truth vs NCA Prediction",
    fontsize=14, fontweight="bold",
)
fig_cmp.tight_layout()

plt.savefig(f"runs/{MODEL_NAME}/comparison.png")
plt.show()

print("\n[Validation] Done.")
