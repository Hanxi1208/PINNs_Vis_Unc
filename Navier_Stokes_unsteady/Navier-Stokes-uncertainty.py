from pathlib import Path
import argparse
import json
import random
import time
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
import torch.nn as nn


DTYPE = torch.float32
PDE_LAMBDA1 = 1.0
PDE_LAMBDA2 = 0.01

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
MAT_PATH = DATA_DIR / "cylinder_nektar_wake.mat"
OUTPUT_ROOT = PROJECT_DIR / "outputs" / "navier_stokes_uncertainty"
CHECKPOINT_ROOT = PROJECT_DIR / "checkpoints" / "navier_stokes_uncertainty"
OBSERVATION_CACHE_ROOT = PROJECT_DIR / "data" / "cached_observation_data" / "navier_stokes_uncertainty"


# CLI and runtime utilities
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Navier-Stokes uncertainty experiment with optional Gaussian noise "
        "and cosine observation bias on the training observations."
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--n-train", type=int, default=5000, help="Number of training points sampled from the cached observation dataset.")
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[3, 20, 20, 20, 20, 20, 20, 20, 20, 2],
        help="MLP layer sizes written as a sequence, for example: --layers 3 20 20 20 2",
    )
    parser.add_argument("--adam-steps", type=int, default=10000)
    parser.add_argument("--adam-lr", type=float, default=1e-3, help="Learning rate for Adam.")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--loss-weight", type=float, default=0.5, help="Weight on the data loss term; the PDE loss uses 1 - loss_weight.")
    parser.add_argument(
        "--noise-type",
        type=str,
        choices=["none", "gaussian"],
        default="none",
        help="Type of stochastic corruption added to the training observations.",
    )
    parser.add_argument("--noise-level", type=float, default=0.0, help="Strength of the Gaussian observation noise.")
    parser.add_argument(
        "--bias-type",
        type=str,
        choices=["none", "cosine"],
        default="none",
        help="Type of deterministic observation bias added to the training observations.",
    )
    parser.add_argument("--bias-level", type=float, default=0.0, help="Amplitude of the deterministic observation bias.")
    parser.add_argument("--print-every", type=int, default=100, help="Print training logs every N Adam steps.")
    parser.add_argument("--plot-t-idx", type=int, default=100, help="Time index used for the final u/v/p slice plots.")
    parser.add_argument("--eval-subset-size", type=int, default=100000, help="Number of flattened reference points used for the final relative L2 evaluation.")
    parser.add_argument("--refresh-observation-cache", action="store_true", help="Regenerate the full corrupted observation dataset instead of reusing the cached .npz file.")
    parser.add_argument("--device", type=str, default="auto", help="Torch device, for example auto, cpu, cuda, or cuda:0.")
    parser.add_argument("--tag", type=str, default="", help="Optional suffix added to the output directory name.")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Data preparation and caching
def load_reference_solution(mat_path: Path) -> dict[str, np.ndarray]:
    raw = scipy.io.loadmat(mat_path)
    return {
        "x": raw["X_star"][:, 0:1].astype("float32"),
        "y": raw["X_star"][:, 1:2].astype("float32"),
        "t": raw["t"].astype("float32"),
        "u": raw["U_star"][:, 0, :].astype("float32"),
        "v": raw["U_star"][:, 1, :].astype("float32"),
        "p": raw["p_star"].astype("float32"),
    }


def flatten_reference(reference: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    n_space = reference["x"].shape[0]
    n_time = reference["t"].shape[0]

    x = np.tile(reference["x"], (1, n_time)).reshape(-1, 1)
    y = np.tile(reference["y"], (1, n_time)).reshape(-1, 1)
    t = np.tile(reference["t"].T, (n_space, 1)).reshape(-1, 1)

    u = reference["u"].reshape(-1, 1)
    v = reference["v"].reshape(-1, 1)
    p = reference["p"].reshape(-1, 1)

    return {"x": x, "y": y, "t": t, "u": u, "v": v, "p": p}


def build_observation_dataset_np(
    flat_reference: dict[str, np.ndarray],
    seed: int,
    noise_type: str = "none",
    noise_level: float = 0.0,
    bias_type: str = "none",
    bias_level: float = 0.0,
) -> dict[str, np.ndarray]:
    x = flat_reference["x"].astype("float32").copy()
    y = flat_reference["y"].astype("float32").copy()
    t = flat_reference["t"].astype("float32").copy()
    u = flat_reference["u"].astype("float32").copy()
    v = flat_reference["v"].astype("float32").copy()
    p = flat_reference["p"].astype("float32").copy()

    rng = np.random.default_rng(seed)

    if bias_type == "cosine" and bias_level > 0.0:
        bias_field = np.cos(x + 2.0 * y).astype("float32")
        u = u + bias_level * bias_field
        v = v + bias_level * bias_field

    if noise_type == "gaussian" and noise_level > 0.0:
        u = u + noise_level * u.std() * rng.standard_normal(u.shape).astype("float32")
        v = v + noise_level * v.std() * rng.standard_normal(v.shape).astype("float32")

    return {"x": x, "y": y, "t": t, "u": u, "v": v, "p": p}


def sample_training_points_np(
    flat_observation: dict[str, np.ndarray],
    n_train: int,
    seed: int,
) -> dict[str, np.ndarray]:
    total = flat_observation["x"].shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.choice(total, n_train, replace=False)

    return {
        "idx": idx.astype("int64"),
        "x": flat_observation["x"][idx].astype("float32").copy(),
        "y": flat_observation["y"][idx].astype("float32").copy(),
        "t": flat_observation["t"][idx].astype("float32").copy(),
        "u": flat_observation["u"][idx].astype("float32").copy(),
        "v": flat_observation["v"][idx].astype("float32").copy(),
    }


def training_points_to_torch(
    points_np: dict[str, np.ndarray],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    x = torch.tensor(points_np["x"], dtype=torch.get_default_dtype(), device=device)
    y = torch.tensor(points_np["y"], dtype=torch.get_default_dtype(), device=device)
    t = torch.tensor(points_np["t"], dtype=torch.get_default_dtype(), device=device)
    x.requires_grad_(True)
    y.requires_grad_(True)
    t.requires_grad_(True)

    u = torch.tensor(points_np["u"], dtype=torch.get_default_dtype(), device=device)
    v = torch.tensor(points_np["v"], dtype=torch.get_default_dtype(), device=device)
    return {"x": x, "y": y, "t": t, "u": u, "v": v}


def format_level_for_name(value: float) -> str:
    return f"{value:.3f}".replace(".", "p")


def make_observation_cache_name(
    *,
    seed: int,
    noise_type: str,
    noise_level: float,
    bias_type: str,
    bias_level: float,
) -> str:
    parts = [
        f"seed_{seed}",
        f"noise_{noise_type}_{format_level_for_name(noise_level)}",
        f"bias_{bias_type}_{format_level_for_name(bias_level)}",
    ]
    return "__".join(parts) + ".npz"


def get_observation_dataset(
    flat_reference: dict[str, np.ndarray],
    seed: int,
    noise_type: str,
    noise_level: float,
    bias_type: str,
    bias_level: float,
    refresh_cache: bool,
) -> tuple[dict[str, np.ndarray], Path, bool]:
    OBSERVATION_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    cache_path = OBSERVATION_CACHE_ROOT / make_observation_cache_name(
        seed=seed,
        noise_type=noise_type,
        noise_level=noise_level,
        bias_type=bias_type,
        bias_level=bias_level,
    )

    if cache_path.exists() and not refresh_cache:
        cached = np.load(cache_path)
        flat_observation = {key: cached[key] for key in ["x", "y", "t", "u", "v", "p"]}
        cache_hit = True
    else:
        flat_observation = build_observation_dataset_np(
            flat_reference,
            seed,
            noise_type=noise_type,
            noise_level=noise_level,
            bias_type=bias_type,
            bias_level=bias_level,
        )
        np.savez_compressed(cache_path, **flat_observation)
        cache_hit = False

    return flat_observation, cache_path, cache_hit


# Model and PDE
class NavierStokesPINN(nn.Module):
    def __init__(
        self,
        layers: tuple[int, ...],
        lb: tuple[float, float, float],
        ub: tuple[float, float, float],
    ):
        super().__init__()
        blocks = []
        for in_features, out_features in zip(layers[:-2], layers[1:-1]):
            blocks.append(nn.Linear(in_features, out_features))
            blocks.append(nn.Tanh())
        blocks.append(nn.Linear(layers[-2], layers[-1]))
        self.net = nn.Sequential(*blocks)

        self.register_buffer("lb", torch.tensor(lb, dtype=torch.get_default_dtype()))
        self.register_buffer("ub", torch.tensor(ub, dtype=torch.get_default_dtype()))
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = torch.cat([x, y, t], dim=1)
        inputs = 2.0 * (inputs - self.lb) / (self.ub - self.lb) - 1.0
        outputs = self.net(inputs)
        psi = outputs[:, :1]
        p = outputs[:, 1:2]
        return psi, p


def gradient(
    dy: torch.Tensor,
    dx: Union[List[torch.Tensor], torch.Tensor],
    grad_outputs: Optional[List[Optional[torch.Tensor]]] = None,
    create_graph: bool = True,
) -> List[torch.Tensor]:
    if grad_outputs is None:
        grad_outputs = [torch.ones_like(dy)]
    if isinstance(dx, torch.Tensor):
        dx = [dx]
    grads = torch.autograd.grad(
        [dy],
        dx,
        grad_outputs=grad_outputs,
        create_graph=create_graph,
        retain_graph=True,
        allow_unused=False,
    )
    return [grad if grad is not None else torch.zeros_like(dx[i]) for i, grad in enumerate(grads)]


def navier_stokes_outputs(
    model: NavierStokesPINN,
    x: torch.Tensor,
    y: torch.Tensor,
    t: torch.Tensor,
) -> dict[str, torch.Tensor]:
    psi, p = model(x, y, t)

    u = gradient(psi, y)[0]
    v = -gradient(psi, x)[0]

    u_x, u_y, u_t = gradient(u, [x, y, t])
    u_xx = gradient(u_x, x)[0]
    u_yy = gradient(u_y, y)[0]

    v_x, v_y, v_t = gradient(v, [x, y, t])
    v_xx = gradient(v_x, x)[0]
    v_yy = gradient(v_y, y)[0]

    p_x, p_y = gradient(p, [x, y])

    f_u = u_t + PDE_LAMBDA1 * (u * u_x + v * u_y) + p_x - PDE_LAMBDA2 * (u_xx + u_yy)
    f_v = v_t + PDE_LAMBDA1 * (u * v_x + v * v_y) + p_y - PDE_LAMBDA2 * (v_xx + v_yy)

    return {"psi": psi, "p": p, "u": u, "v": v, "f_u": f_u, "f_v": f_v}


# Loss, optimization, and evaluation
def sse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sum((pred - target).square())


def pinn_loss_tensors(
    model: NavierStokesPINN,
    points: dict[str, torch.Tensor],
    loss_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    outputs = navier_stokes_outputs(model, points["x"], points["y"], points["t"])
    zero_u = torch.zeros_like(outputs["f_u"])
    zero_v = torch.zeros_like(outputs["f_v"])

    loss_data = sse(outputs["u"], points["u"]) + sse(outputs["v"], points["v"])
    loss_pde = sse(outputs["f_u"], zero_u) + sse(outputs["f_v"], zero_v)
    loss = loss_weight * loss_data + (1.0 - loss_weight) * loss_pde
    return loss, loss_data, loss_pde


def pinn_loss(
    model: NavierStokesPINN,
    points: dict[str, torch.Tensor],
    loss_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    loss, loss_data, loss_pde = pinn_loss_tensors(model, points, loss_weight)
    stats = {
        "loss": float(loss.detach().cpu()),
        "loss_data": float(loss_data.detach().cpu()),
        "loss_pde": float(loss_pde.detach().cpu()),
        "loss_weight": float(loss_weight),
        "pde_weight": float(1.0 - loss_weight),
    }
    return loss, stats


def train_adam(
    model: NavierStokesPINN,
    points: dict[str, torch.Tensor],
    *,
    adam_steps: int,
    adam_lr: float,
    weight_decay: float,
    print_every: int,
    loss_weight: float,
) -> list[dict[str, float]]:
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=adam_lr,
        weight_decay=weight_decay,
    )

    history = []
    start = time.time()

    for step in range(1, adam_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, stats = pinn_loss(model, points, loss_weight)
        loss.backward()
        optimizer.step()

        stats["step"] = step
        stats["lr"] = optimizer.param_groups[0]["lr"]
        history.append(stats)

        if step % print_every == 0 or step == 1:
            elapsed = time.time() - start
            print(
                f"Adam step {step:6d} | "
                f"loss={stats['loss']:.3e} | "
                f"data={stats['loss_data']:.3e} | "
                f"pde={stats['loss_pde']:.3e} | "
                f"elapsed={elapsed:.1f}s"
            )

    return history


def predict_chunk(
    model: NavierStokesPINN,
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_tensor = torch.tensor(x, dtype=torch.get_default_dtype(), device=device)
    y_tensor = torch.tensor(y, dtype=torch.get_default_dtype(), device=device)
    t_tensor = torch.tensor(t, dtype=torch.get_default_dtype(), device=device)
    x_tensor.requires_grad_(True)
    y_tensor.requires_grad_(True)
    t_tensor.requires_grad_(True)

    model.eval()
    outputs = navier_stokes_outputs(model, x_tensor, y_tensor, t_tensor)
    return (
        outputs["u"].detach().cpu().numpy(),
        outputs["v"].detach().cpu().numpy(),
        outputs["p"].detach().cpu().numpy(),
    )


def relative_l2(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.linalg.norm(pred - target) / np.linalg.norm(target))


# Visualization and output helpers
def make_run_name(
    loss_weight: float,
    tag: str,
    noise_type: str,
    noise_level: float,
    bias_type: str,
    bias_level: float,
) -> str:
    parts = [f"lw_{loss_weight:.3f}".replace(".", "p")]
    if noise_type == "gaussian" and noise_level > 0.0:
        parts.append(f"{noise_type}_{noise_level:.3f}".replace(".", "p"))
    if bias_type == "cosine" and bias_level > 0.0:
        parts.append(f"{bias_type}_{bias_level:.3f}".replace(".", "p"))
    if tag:
        parts.append(tag)
    return "_".join(parts)


def save_history_plot(history: list[dict[str, float]], out_path: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4), constrained_layout=True)

    ax.plot([row["step"] for row in history], [row["loss"] for row in history], label="total loss")
    ax.plot([row["step"] for row in history], [row["loss_data"] for row in history], label="data loss")
    ax.plot([row["step"] for row in history], [row["loss_pde"] for row in history], label="PDE loss")
    ax.set_yscale("log")
    ax.set_xlabel("Adam step")
    ax.set_ylabel("SSE loss")
    ax.set_title("Training history")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_u_plot(
    reference: dict[str, np.ndarray],
    flat_observation: dict[str, np.ndarray],
    model: NavierStokesPINN,
    device: torch.device,
    plot_t_idx: int,
    out_path: Path,
) -> None:
    x_slice = reference["x"]
    y_slice = reference["y"]
    t_slice = np.full_like(x_slice, float(reference["t"][plot_t_idx, 0]))

    u_slice_pred, _, _ = predict_chunk(model, x_slice, y_slice, t_slice, device)
    u_slice_true = reference["u"][:, plot_t_idx : plot_t_idx + 1]
    n_space = reference["x"].shape[0]
    n_time = reference["t"].shape[0]
    u_slice_obs = flat_observation["u"].reshape(n_space, n_time)[:, plot_t_idx : plot_t_idx + 1]

    nx = np.unique(reference["x"][:, 0]).size
    ny = np.unique(reference["y"][:, 0]).size
    x_grid = reference["x"].reshape(ny, nx)
    y_grid = reference["y"].reshape(ny, nx)
    u_true_grid = u_slice_true.reshape(ny, nx)
    u_obs_grid = u_slice_obs.reshape(ny, nx)
    u_pred_grid = u_slice_pred.reshape(ny, nx)
    u_vmin = float(u_obs_grid.min())
    u_vmax = float(u_obs_grid.max())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)

    mesh0 = axes[0].pcolormesh(
        x_grid, y_grid, u_true_grid, shading="auto", cmap="coolwarm", vmin=u_vmin, vmax=u_vmax
    )
    axes[0].set_title("Original PDE / reference u(x, y)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    plt.colorbar(mesh0, ax=axes[0], shrink=0.9)

    mesh1 = axes[1].pcolormesh(
        x_grid, y_grid, u_obs_grid, shading="auto", cmap="coolwarm", vmin=u_vmin, vmax=u_vmax
    )
    axes[1].set_title("Training observation u(x, y)")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    plt.colorbar(mesh1, ax=axes[1], shrink=0.9)

    mesh2 = axes[2].pcolormesh(
        x_grid, y_grid, u_pred_grid, shading="auto", cmap="coolwarm", vmin=u_vmin, vmax=u_vmax
    )
    axes[2].set_title("PINN prediction u(x, y)")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    plt.colorbar(mesh2, ax=axes[2], shrink=0.9)

    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_v_plot(
    reference: dict[str, np.ndarray],
    flat_observation: dict[str, np.ndarray],
    model: NavierStokesPINN,
    device: torch.device,
    plot_t_idx: int,
    out_path: Path,
) -> None:
    x_slice = reference["x"]
    y_slice = reference["y"]
    t_slice = np.full_like(x_slice, float(reference["t"][plot_t_idx, 0]))

    _, v_slice_pred, _ = predict_chunk(model, x_slice, y_slice, t_slice, device)
    v_slice_true = reference["v"][:, plot_t_idx : plot_t_idx + 1]
    n_space = reference["x"].shape[0]
    n_time = reference["t"].shape[0]
    v_slice_obs = flat_observation["v"].reshape(n_space, n_time)[:, plot_t_idx : plot_t_idx + 1]

    nx = np.unique(reference["x"][:, 0]).size
    ny = np.unique(reference["y"][:, 0]).size
    x_grid = reference["x"].reshape(ny, nx)
    y_grid = reference["y"].reshape(ny, nx)
    v_true_grid = v_slice_true.reshape(ny, nx)
    v_obs_grid = v_slice_obs.reshape(ny, nx)
    v_pred_grid = v_slice_pred.reshape(ny, nx)
    v_vmin = float(v_obs_grid.min())
    v_vmax = float(v_obs_grid.max())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)

    mesh0 = axes[0].pcolormesh(
        x_grid, y_grid, v_true_grid, shading="auto", cmap="coolwarm", vmin=v_vmin, vmax=v_vmax
    )
    axes[0].set_title("Original PDE / reference v(x, y)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    plt.colorbar(mesh0, ax=axes[0], shrink=0.9)

    mesh1 = axes[1].pcolormesh(
        x_grid, y_grid, v_obs_grid, shading="auto", cmap="coolwarm", vmin=v_vmin, vmax=v_vmax
    )
    axes[1].set_title("Training observation v(x, y)")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    plt.colorbar(mesh1, ax=axes[1], shrink=0.9)

    mesh2 = axes[2].pcolormesh(
        x_grid, y_grid, v_pred_grid, shading="auto", cmap="coolwarm", vmin=v_vmin, vmax=v_vmax
    )
    axes[2].set_title("PINN prediction v(x, y)")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    plt.colorbar(mesh2, ax=axes[2], shrink=0.9)

    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_p_plot(
    reference: dict[str, np.ndarray],
    model: NavierStokesPINN,
    device: torch.device,
    plot_t_idx: int,
    out_path: Path,
) -> None:
    x_slice = reference["x"]
    y_slice = reference["y"]
    t_slice = np.full_like(x_slice, float(reference["t"][plot_t_idx, 0]))

    _, _, p_slice_pred = predict_chunk(model, x_slice, y_slice, t_slice, device)
    p_slice_true = reference["p"][:, plot_t_idx : plot_t_idx + 1]
    p_slice_pred_aligned = p_slice_pred - np.mean(p_slice_pred - p_slice_true)

    nx = np.unique(reference["x"][:, 0]).size
    ny = np.unique(reference["y"][:, 0]).size
    x_grid = reference["x"].reshape(ny, nx)
    y_grid = reference["y"].reshape(ny, nx)
    p_true_grid = p_slice_true.reshape(ny, nx)
    p_pred_grid = p_slice_pred_aligned.reshape(ny, nx)
    p_err_grid = np.abs(p_pred_grid - p_true_grid)
    p_vmin = float(p_true_grid.min())
    p_vmax = float(p_true_grid.max())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)

    mesh0 = axes[0].pcolormesh(
        x_grid, y_grid, p_true_grid, shading="auto", cmap="coolwarm", vmin=p_vmin, vmax=p_vmax
    )
    axes[0].set_title("Reference p(x, y)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    plt.colorbar(mesh0, ax=axes[0], shrink=0.9)

    mesh1 = axes[1].pcolormesh(
        x_grid, y_grid, p_pred_grid, shading="auto", cmap="coolwarm", vmin=p_vmin, vmax=p_vmax
    )
    axes[1].set_title("PINN prediction p(x, y)")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    plt.colorbar(mesh1, ax=axes[1], shrink=0.9)

    mesh2 = axes[2].pcolormesh(x_grid, y_grid, p_err_grid, shading="auto", cmap="magma")
    axes[2].set_title("Absolute error")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    plt.colorbar(mesh2, ax=axes[2], shrink=0.9)

    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# Main entry point
def main() -> None:
    args = parse_args()
    if not 0.0 <= args.loss_weight <= 1.0:
        raise ValueError("--loss-weight must be between 0 and 1.")
    if args.noise_level < 0.0:
        raise ValueError("--noise-level must be non-negative.")
    if args.bias_level < 0.0:
        raise ValueError("--bias-level must be non-negative.")

    torch.set_default_dtype(DTYPE)
    seed_everything(args.seed)
    device = resolve_device(args.device)

    reference = load_reference_solution(MAT_PATH)
    flat_reference = flatten_reference(reference)

    lb = (
        float(reference["x"].min()),
        float(reference["y"].min()),
        float(reference["t"].min()),
    )
    ub = (
        float(reference["x"].max()),
        float(reference["y"].max()),
        float(reference["t"].max()),
    )

    model = NavierStokesPINN(
        layers=tuple(args.layers),
        lb=lb,
        ub=ub,
    ).to(device)

    flat_observation, observation_cache_path, observation_cache_hit = get_observation_dataset(
        flat_reference,
        args.seed,
        noise_type=args.noise_type,
        noise_level=args.noise_level,
        bias_type=args.bias_type,
        bias_level=args.bias_level,
        refresh_cache=args.refresh_observation_cache,
    )
    train_points_np = sample_training_points_np(flat_observation, args.n_train, args.seed)
    train_points = training_points_to_torch(train_points_np, device)

    print(
        f"device={device}, n_train={args.n_train}, adam_steps={args.adam_steps}, "
        f"loss_weight={args.loss_weight}, pde_weight={1.0 - args.loss_weight}, "
        f"noise_type={args.noise_type}, noise_level={args.noise_level}, "
        f"bias_type={args.bias_type}, bias_level={args.bias_level}, "
        f"observation_cache_hit={observation_cache_hit}"
    )
    print(f"observation_cache_path={observation_cache_path}")

    history = train_adam(
        model,
        train_points,
        adam_steps=args.adam_steps,
        adam_lr=args.adam_lr,
        weight_decay=args.weight_decay,
        print_every=args.print_every,
        loss_weight=args.loss_weight,
    )

    rng = np.random.default_rng(args.seed)
    eval_idx = rng.choice(flat_reference["x"].shape[0], size=args.eval_subset_size, replace=False)
    u_eval, v_eval, p_eval = predict_chunk(
        model,
        flat_reference["x"][eval_idx],
        flat_reference["y"][eval_idx],
        flat_reference["t"][eval_idx],
        device,
    )
    p_eval_aligned = p_eval - np.mean(p_eval - flat_reference["p"][eval_idx])

    metrics = {
        "rel_l2_u": relative_l2(u_eval, flat_reference["u"][eval_idx]),
        "rel_l2_v": relative_l2(v_eval, flat_reference["v"][eval_idx]),
        "rel_l2_p": relative_l2(p_eval_aligned, flat_reference["p"][eval_idx]),
        "loss_weight": float(args.loss_weight),
        "pde_weight": float(1.0 - args.loss_weight),
        "pde_lambda1": PDE_LAMBDA1,
        "pde_lambda2": PDE_LAMBDA2,
        "adam_steps": int(args.adam_steps),
        "adam_lr": float(args.adam_lr),
        "weight_decay": float(args.weight_decay),
        "n_train": int(args.n_train),
        "noise_type": args.noise_type,
        "noise_level": float(args.noise_level),
        "bias_type": args.bias_type,
        "bias_level": float(args.bias_level),
        "seed": int(args.seed),
        "device": str(device),
        "observation_cache_path": str(observation_cache_path),
        "observation_cache_hit": bool(observation_cache_hit),
    }

    run_name = make_run_name(
        args.loss_weight,
        args.tag,
        args.noise_type,
        args.noise_level,
        args.bias_type,
        args.bias_level,
    )
    out_dir = OUTPUT_ROOT / run_name
    ckpt_dir = CHECKPOINT_ROOT / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    save_history_plot(history, out_dir / "history.png")
    save_u_plot(
        reference,
        flat_observation,
        model,
        device,
        args.plot_t_idx,
        out_dir / "u_slice.png",
    )
    save_v_plot(
        reference,
        flat_observation,
        model,
        device,
        args.plot_t_idx,
        out_dir / "v_slice.png",
    )
    save_p_plot(reference, model, device, args.plot_t_idx, out_dir / "p_slice.png")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
            "history": history,
            "loss_weight": args.loss_weight,
        },
        ckpt_dir / "model.pt",
    )

    print(json.dumps(metrics, indent=2))
    print(f"saved_outputs={out_dir}")
    print(f"saved_checkpoint={ckpt_dir / 'model.pt'}")


if __name__ == "__main__":
    main()
