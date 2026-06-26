import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
from time import time
from tqdm import tqdm

from config import DEVICE, STD_DTYPE
from edp import simulate_edp
from model import NCA, evolve

class NCA_PDE_Trainer:
    """
    Train a PyTorch NCA to reproduce the output of a PDE simulation.

    The trainer:
      1. Runs a PDE solver (`simulate_edp`) to generate ground-truth data.
      2. Slices the trajectory into (x0, target) pairs.
      3. Trains the NCA so that `x0 + iter_n * model(x)` ≈ target.

    Data layout is NCHW = (B, C, H, W) throughout.
    """

    def __init__(
        self,
        nca_model: NCA,
        x0: torch.Tensor,
        F_pde,
        T: int,
        N_BATCHES: int = 1,
        step_mul: int = 1,
        dt: float = 0.001,
        model_filename: str | None = None,
        directory: str = "models/",
        log_dir: str = "runs/",
        noise_frac: float = 0.0,
        **pde_kwargs,
    ):
        """
        Args:
            nca_model:    An NCA instance (already on DEVICE).
            x0:           Initial condition, shape (1, C, H, W).
            F_pde:        PDE right-hand side function for `simulate_edp`.
            T:            Number of time-steps to keep after subsampling.
            N_BATCHES:    How many copies of the trajectory to use per
                          training iteration (all share the same IC here).
            step_mul:     PDE steps per kept frame (subsampling factor).
            dt:           PDE integration time-step.
            model_filename: Base name for saved checkpoints (no extension).
            directory:    Folder for saved checkpoints.
            log_dir:      TensorBoard log directory.
            noise_frac:   Fraction of uniform noise mixed into inputs/targets
                          as regularisation (0 = no noise).
            **pde_kwargs: Extra keyword arguments forwarded to F_pde.
        """
        self.nca_model = nca_model
        self.T = T
        self.N_BATCHES = N_BATCHES
        self.noise_frac = noise_frac
        self.model_filename = model_filename
        self.directory = directory
        self.log_dir = log_dir
        self.dt = dt

        # --- Run PDE solver to generate ground truth --------------------------
        total_pde_steps = T * step_mul
        # simulate_edp returns (total_pde_steps, B, C, H, W)
        data = simulate_edp(x0, F_pde, t=total_pde_steps, dt=dt, **pde_kwargs)
        # Subsample to keep T frames
        data = data[::step_mul]  # (T, B, C, H, W)

        # --- Normalise data to [0, 1] ----------------------------------------
        self.data_min = data.min().item()
        self.data_max = data.max().item()
        data_range = self.data_max - self.data_min
        if data_range > 0:
            data = (data - self.data_min) / data_range
        else:
            data = torch.zeros_like(data)

        self.data = data.to(device=DEVICE, dtype=STD_DTYPE)  # (T, B, C, H, W)

        # --- Build (x0, target) pairs -----------------------------------------
        # x0[i]     = PDE state at time i     → NCA input
        # target[i] = PDE state at time i+1   → NCA should produce this
        # Flatten the time and batch dimensions: ((T-1)*N_BATCHES, C, H, W)
        x0_seq = self.data[:-1]  # (T-1, B, C, H, W)
        target_seq = self.data[1:]   # (T-1, B, C, H, W)

        if N_BATCHES > 1:
            # Tile each frame N_BATCHES times along the batch axis
            x0_seq = x0_seq.repeat(1, N_BATCHES, 1, 1, 1)
            target_seq = target_seq.repeat(1, N_BATCHES, 1, 1, 1)

        T_pairs = x0_seq.shape[0]
        B_total = x0_seq.shape[1]

        # Merge time and batch → single batch dimension
        self.x0 = x0_seq.reshape(T_pairs * B_total, *x0_seq.shape[2:])
        self.target = target_seq.reshape(T_pairs * B_total, *target_seq.shape[2:])
        self.x0_true = self.x0.clone()  # pristine copy for resets

        # Add hidden channels to x0 (NCA may have more channels than the PDE)
        C_pde = self.x0.shape[1]
        C_nca = nca_model.C
        if C_nca > C_pde:
            hidden = torch.zeros(
                self.x0.shape[0], C_nca - C_pde,
                self.x0.shape[2], self.x0.shape[3],
                device=DEVICE, dtype=STD_DTYPE,
            )
            self.x0 = torch.cat([self.x0, hidden], dim=1)
            self.x0_true = self.x0.clone()

        self.OBS_CHANNELS = C_pde  # observable (PDE) channels

        # Ensure save directory exists
        os.makedirs(self.directory, exist_ok=True)

        print(f"Ground truth data shape : {self.data.shape}")
        print(f"x0 shape                : {self.x0.shape}")
        print(f"target shape            : {self.target.shape}")
        print(f"Data range              : [{self.data_min:.4f}, {self.data_max:.4f}]")

        self.setup_tb_log()

    # ------------------------------------------------------------------
    # TensorBoard helpers
    # ------------------------------------------------------------------

    def setup_tb_log(self):
        """Initialise TensorBoard writer and log the ground-truth trajectory."""
        self.writer = SummaryWriter(log_dir=self.log_dir)

        for i in range(min(self.T, self.data.shape[0])):
            # data[i] has shape (B, C, H, W); take the first sample
            frame = self.data[i, 0]  # (C, H, W)

            if frame.shape[0] == 1:
                # Single-channel: normalise to [0,1] for TensorBoard grayscale
                frame_vis = frame.clone()
                fmin, fmax = frame_vis.min(), frame_vis.max()
                if fmax - fmin > 0:
                    frame_vis = (frame_vis - fmin) / (fmax - fmin)
                self.writer.add_image("PDE_ground_truth", frame_vis, global_step=i)
            else:
                # Multi-channel: log first 3 channels as pseudo-RGB
                rgb = frame[:3]
                self.writer.add_image("PDE_ground_truth", rgb, global_step=i)

    def tb_training_loop_log(self, mean_loss, losses, x, step):
        """Log training metrics to TensorBoard."""
        self.writer.add_scalar("Loss/mean", mean_loss.item(), global_step=step)
        self.writer.add_histogram("Loss/distribution", losses, global_step=step)

        if step % 10 == 0:
            for name, param in self.nca_model.named_parameters():
                if param.numel() > 0:
                    self.writer.add_histogram(f"params/{name}", param, global_step=step)
                    if param.grad is not None:
                        self.writer.add_histogram(
                            f"grads/{name}", param.grad, global_step=step
                        )

    # ------------------------------------------------------------------
    # Training logic
    # ------------------------------------------------------------------

    def train_step(
        self,
        x: torch.Tensor,
        iter_n: int,
        REG_COEFF: float = 0.0,
        update_gradients: bool = True,
        LOSS_FUNC=None,
        BATCH_SIZE: int = 64,
        TRAIN_MODE: str = "full",
        NORM_GRADS: bool = True,
    ):
        """
        Execute one training step over all (x0, target) pairs.

        Args:
            x:           Current NCA states, shape (N, C_nca, H, W).
            iter_n:      Number of NCA forward steps per training step.
            REG_COEFF:   Coefficient for the out-of-[0,1] regularisation.
            update_gradients: Whether to actually update weights (for
                              stochastic update rate).
            LOSS_FUNC:   Custom loss(pred, true) or None → MSE.
            BATCH_SIZE:  Mini-batch size.
            TRAIN_MODE:  "full" (predict state) or "differential" (predict Δ).
            NORM_GRADS:  Whether to normalise gradients.

        Returns:
            x_updated:  Updated NCA states after forward steps.
            mean_loss:  Scalar mean loss.
            all_losses: Tensor of per-batch losses.
        """
        if LOSS_FUNC is None:
            loss_func = F.mse_loss
        else:
            # Wrap so it only compares observable channels
            loss_func = lambda pred, true_val: LOSS_FUNC(
                pred[:, :self.OBS_CHANNELS], true_val
            )

        # --- Apply noise regularisation ---
        noise_x = torch.empty_like(x).uniform_(0.0, 1.0)
        noise_t = torch.empty_like(self.target).uniform_(0.0, 1.0)

        x_noisy = (1 - self.noise_frac) * x + self.noise_frac * noise_x
        t_noisy = (1 - self.noise_frac) * self.target + self.noise_frac * noise_t

        dataset_size = x.shape[0]
        batch_indices = torch.randperm(dataset_size, device=DEVICE)
        all_losses = []

        n_batches = max(1, dataset_size // BATCH_SIZE)

        for b in range(n_batches):
            idx = batch_indices[b * BATCH_SIZE : (b + 1) * BATCH_SIZE]
            if len(idx) == 0:
                continue

            X_b = x_noisy[idx].clone().detach().requires_grad_(False)
            T_b = t_noisy[idx]

            X_original = X_b.clone()
            reg_loss = torch.tensor(0.0, device=DEVICE)

            if update_gradients:
                self.optimizer.zero_grad()

            # --- NCA forward steps ---
            for _ in range(iter_n):
                X_b = X_b + self.nca_model(X_b, self.dt)
                reg_loss = reg_loss + torch.sum(
                    F.relu(-X_b[:, :self.OBS_CHANNELS])
                    + F.relu(X_b[:, :self.OBS_CHANNELS] - 1.0)
                )

            # --- Compute loss ---
            if TRAIN_MODE == "differential":
                pred = X_b[:, :self.OBS_CHANNELS] - X_original[:, :self.OBS_CHANNELS]
                loss = loss_func(pred, T_b)
            else:  # "full"
                loss = loss_func(X_b[:, :self.OBS_CHANNELS], T_b)

            mean_loss = loss.mean() + REG_COEFF * (reg_loss / max(iter_n, 1))
            all_losses.append(loss.detach())

            if update_gradients:
                mean_loss.backward()

                if NORM_GRADS:
                    with torch.no_grad():
                        for param in self.nca_model.parameters():
                            if param.grad is not None:
                                grad_norm = torch.norm(param.grad) + 1e-8
                                param.grad.div_(grad_norm)

                self.optimizer.step()

            # --- Propagate updated states (no grad) ---
            with torch.no_grad():
                x_noisy[idx] = X_b.detach()

        all_losses_tensor = torch.stack(all_losses) if all_losses else torch.tensor([0.0], device=DEVICE)
        return x_noisy, all_losses_tensor.mean(), all_losses_tensor

    def train_sequence(
        self,
        TRAIN_ITERS: int,
        iter_n: int,
        UPDATE_RATE: float = 1.0,
        REG_COEFF: float = 0.0,
        LOSS_FUNC=None,
        LEARN_RATE: float = 1e-3,
        OPTIMIZER: str = "Nadam",
        BATCH_SIZE: int = 64,
        TRAIN_MODE: str = "full",
        NORM_GRADS: bool = True,
        SCHEDULER_GAMMA: float = 0.9999,
    ):
        """
        Full training loop.

        Args:
            TRAIN_ITERS:     Total number of training iterations.
            iter_n:          NCA forward steps per training step.
            UPDATE_RATE:     Probability of actually updating gradients
                             on each iteration (stochastic update).
            REG_COEFF:       Out-of-[0,1] regularisation coefficient.
            LOSS_FUNC:       Custom loss or None → MSE.
            LEARN_RATE:      Learning rate.
            OPTIMIZER:       One of {"Adagrad","Adam","Adadelta","Nadam","RMSprop"}.
            BATCH_SIZE:      Mini-batch size.
            TRAIN_MODE:      "full" or "differential".
            NORM_GRADS:      Normalise gradients.
            SCHEDULER_GAMMA: Exponential LR decay factor per step.
        """
        # --- Setup optimiser ---
        optimizers = {
            "Adagrad":  optim.Adagrad,
            "Adam":     optim.Adam,
            "Adadelta": optim.Adadelta,
            "Nadam":    optim.NAdam,
            "RMSprop":  optim.RMSprop,
        }
        if OPTIMIZER not in optimizers:
            raise ValueError(
                f"Unsupported optimizer '{OPTIMIZER}'. "
                f"Choose from: {set(optimizers.keys())}"
            )
        self.optimizer = optimizers[OPTIMIZER](
            self.nca_model.parameters(), lr=LEARN_RATE
        )

        scheduler = ExponentialLR(self.optimizer, gamma=SCHEDULER_GAMMA)

        # --- Handle differential mode ---
        if TRAIN_MODE == "differential":
            # Target becomes the difference between consecutive frames
            # x0 and target are already consecutive pairs; compute Δ
            self.target = self.target - self.x0_true[:, :self.OBS_CHANNELS]

        best_mean_loss = float("inf")
        previous_mean_loss = float("inf")
        self.time_of_best_model = 0
        self.BEST_TRAJECTORY = None

        N_BATCHES = self.N_BATCHES
        start_time = time()

        for i in tqdm(range(TRAIN_ITERS), desc="Training NCA", ncols=100):
            update_grad = np.random.uniform() <= UPDATE_RATE

            x, mean_loss, losses = self.train_step(
                self.x0,
                iter_n,
                REG_COEFF=REG_COEFF,
                update_gradients=update_grad,
                LOSS_FUNC=LOSS_FUNC,
                BATCH_SIZE=BATCH_SIZE,
                TRAIN_MODE=TRAIN_MODE,
                NORM_GRADS=NORM_GRADS,
            )

            assert not torch.isnan(x).any(), "|-|-|-|-|-|-  X reached NaN  -|-|-|-|-|-|"

            # --- Maintain sequence state across steps ---
            with torch.no_grad():
                # Shift: the output of step i becomes the input for step i+1
                self.x0[N_BATCHES:] = x[:-N_BATCHES].clone()
                # Reset the first frame of each batch to the true IC
                if N_BATCHES > 1:
                    self.x0[::N_BATCHES][1:] = self.x0_true[::N_BATCHES][1:].clone()

            scheduler.step()

            # --- Save best model checkpoint ---
            if (
                mean_loss < best_mean_loss
                and mean_loss < previous_mean_loss
                and i > TRAIN_ITERS // 20
            ):
                if self.model_filename is not None:
                    save_path = os.path.join(
                        self.directory, f"{self.model_filename}.pt"
                    )
                    self.nca_model.save(save_path)
                    tqdm.write(f"--- Model saved at epoch {i}  (loss={mean_loss:.6f}) ---")

                with torch.no_grad():
                    ic = self.x0[:1].clone()
                    self.BEST_TRAJECTORY = evolve(
                        self.nca_model, ic, iters=iter_n * self.T * 2, dt=self.dt
                    )

                self.time_of_best_model = i
                best_mean_loss = mean_loss.item()

            previous_mean_loss = mean_loss.item()
            self.tb_training_loop_log(mean_loss, losses, x, i)

        elapsed = time() - start_time
        print("-------- Training complete ---------")
        print(f"Time taken: {elapsed:.1f}s  |  Best loss: {best_mean_loss:.6f} at epoch {self.time_of_best_model}")

        self.writer.close()
