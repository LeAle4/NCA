# Neural Cellular Automata (NCA) for Swift–Hohenberg PDE Simulation

This project implements a PyTorch-based Neural Cellular Automata (NCA) trained to reproduce the dynamics of the 2D Swift–Hohenberg Partial Differential Equation (PDE) simulated with explicit Euler methods.

> [!NOTE]  
> The training framework (`validation.py` / `training.py`) is based on [AlexDR1998/NCA](https://github.com/AlexDR1998/NCA).

---

## Project Structure

- **[config.py](config.py)**: Configures global variables (device, precision dtype).
- **[edp.py](edp.py)**: Numerical PDE simulation framework containing derivative calculations (Sobel, Laplacian) and numerical integrators.
- **[equations.py](equations.py)**: Physics equations defining the right-hand side, specifically the Swift–Hohenberg equation (`F_swift_hohen`).
- **[kernels.py](kernels.py)**: Sobel, Laplacian, Identity, Average, and Biharmonic convolutional filters.
- **[model.py](model.py)**: NCA model architecture definition using circular/periodic padding and 1x1 convolutions for updating state.
- **[patterns.py](patterns.py)**: Generators for starting condition tensors used for testing (random noise, Gaussian bumps, sine waves, striped grids, and step functions).
- **[validation.py](validation.py)**: Runs training on the Swift–Hohenberg equation, logs hyperparameter configurations to a JSON file, and computes overall metrics (MSE/MAE).
- **[testing.py](testing.py)**: Evaluates trained NCA models on all standard patterns in `patterns.py` and saves comparative snapshot figures into `evaluation/<MODEL_NAME>/`.
- **[viz.py](viz.py)**: Utilities for animating simulation rollouts and plotting snapshots.

---

## Setup & Dependencies

Install dependencies from the `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Training & Validation
Run `validation.py` to train an NCA on the Swift–Hohenberg equation. You can adjust all hyperparameters up top:
```bash
python validation.py
```
This saves:
- The model weights checkpoint to `models/<MODEL_NAME>.pt`.
- The configuration and validation performance to `models/<MODEL_NAME>_config.json`.
- Tensorboard runs logging the training loss to `runs/`.

### 2. Multi-Pattern Testing
To evaluate your trained model against the explicit Euler PDE across various starting conditions (e.g., Gaussian bumps, step functions):
```bash
python testing.py
```
This runs the simulation and model rollouts for all patterns in `patterns.py` and saves:
- Performance summary JSON: `evaluation/<MODEL_NAME>/patterns_evaluation_summary.json`
- Side-by-side comparative snapshots: `evaluation/<MODEL_NAME>/pattern_<NAME>.png`
