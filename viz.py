import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animate_pde(
    x: torch.Tensor,
    interval: int = 50,
    cmap: str = "viridis",
    save_path: str | None = None,
):
    """
    Animate a PDE simulation.

    Parameters
    ----------
    x : Tensor
        Shape (T,1,W,H)
    interval : int
        Milliseconds between frames
    cmap : str
        Matplotlib colormap
    save_path : str or None
        If provided, saves animation to file
    """
    data = x.detach().cpu().numpy()[:, 0]

    fig, ax = plt.subplots()

    im = ax.imshow(
        data[0],
        cmap=cmap,
        animated=True,
        vmin=data.min(),
        vmax=data.max(),
    )

    plt.colorbar(im, ax=ax)

    def update(frame):
        im.set_array(data[frame])
        ax.set_title(f"t = {frame}")
        return [im]

    ani = FuncAnimation(
        fig,
        update,
        frames=len(data),
        interval=interval,
        blit=True,
    )

    if save_path is not None:
        ani.save(save_path)

    plt.show()

    return ani


def plot_snapshots(
    x: torch.Tensor,
    n_snapshots: int = 6,
    cmap: str = "viridis",
):
    """
    Display evenly-spaced snapshots.
    """
    T = x.shape[0]

    times = np.linspace(
        0,
        T - 1,
        n_snapshots,
        dtype=int
    )

    rows = int(np.ceil(np.sqrt(n_snapshots)))
    cols = int(np.ceil(n_snapshots / rows))

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(4 * cols, 4 * rows)
    )

    axes = np.atleast_1d(axes).flatten()

    for ax, t in zip(axes, times):
        im = ax.imshow(
            x[t, 0].cpu(),
            cmap=cmap
        )

        ax.set_title(f"t = {t}")
        ax.axis("off")

    for ax in axes[len(times):]:
        ax.remove()

    fig.colorbar(im, ax=axes.tolist())
    plt.tight_layout()
    plt.show()


def plot_spacetime(
    x: torch.Tensor,
    axis: str = "horizontal",
    cmap: str = "viridis",
):
    """
    Create a space-time diagram.

    axis='horizontal'
        Takes middle row.

    axis='vertical'
        Takes middle column.
    """
    _, _, W, H = x.shape

    if axis == "horizontal":
        slice_data = x[:, 0, W // 2, :]
    elif axis == "vertical":
        slice_data = x[:, 0, :, H // 2]
    else:
        raise ValueError(
            "axis must be 'horizontal' or 'vertical'"
        )

    slice_data = slice_data.cpu().numpy()

    plt.figure(figsize=(8, 6))

    plt.imshow(
        slice_data,
        aspect="auto",
        origin="lower",
        cmap=cmap,
    )

    plt.xlabel("Space")
    plt.ylabel("Time")
    plt.title("Space-Time Diagram")
    plt.colorbar()
    plt.show()


def plot_statistics(x: torch.Tensor):
    """
    Plot useful global statistics.
    """

    mean_u = x.mean(dim=(1, 2, 3)).cpu().numpy()

    mean_u2 = (
        (x ** 2)
        .mean(dim=(1, 2, 3))
        .cpu()
        .numpy()
    )

    std_u = (
        x.std(dim=(1, 2, 3))
        .cpu()
        .numpy()
    )

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(8, 8),
        sharex=True,
    )

    axes[0].plot(mean_u)
    axes[0].set_ylabel("<u>")

    axes[1].plot(mean_u2)
    axes[1].set_ylabel("<u²>")

    axes[2].plot(std_u)
    axes[2].set_ylabel("std(u)")
    axes[2].set_xlabel("Time")

    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":

    T = 200
    W = 128
    H = 128

    x = torch.randn(T, 1, W, H)

    plot_snapshots(x)
    plot_spacetime(x)
    plot_statistics(x)

    animate_pde(
        x,
        interval=30,
        save_path=None
    )