from __future__ import annotations

from pathlib import Path
import argparse
import copy
import json
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
import torch.nn as nn


DTYPE = torch.float32
PDE_LAMBDA1 = 1.0
PDE_LAMBDA2 = 0.01

SCRIPT_DIR = Path(__file__).resolve().parent
MAT_PATH = SCRIPT_DIR.parent.parent / "data" / "cylinder_nektar_wake.mat"
OUTPUT_ROOT = SCRIPT_DIR / "output"
CHECKPOINT_ROOT = SCRIPT_DIR / "checkpoints"
OBSERVATION_CACHE_ROOT = SCRIPT_DIR / "obs_cache"


# CLI
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Navier-Stokes PINN, uvp full-observation + full-collocation:"
                    " u, v, p are all observed; data and PDE both use all 1M points, with no random sampling."
    )
    p.add_argument("--seed", type=int, default=1234,
                   help="Only affects model init / observation noise / eval index; unrelated to training-point sampling (there is none).")
    p.add_argument("--layers", type=int, nargs="+",
                   default=[3, 20, 20, 20, 20, 20, 20, 20, 20, 2])
    p.add_argument("--adam-steps", type=int, default=10000)
    p.add_argument("--adam-lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--loss-weight", type=float, default=0.5)
    p.add_argument("--early-stop-patience", type=int, default=0,
                   help="Stop the Adam phase early if total loss does not decrease for N consecutive steps, "
                        "rolling back to the best state. 0 disables it.")
    p.add_argument("--lr-schedule", type=str, default="cosine", choices=["none", "cosine"],
                   help="Adam learning-rate schedule. cosine = CosineAnnealingLR(T_max=adam_steps).")
    p.add_argument("--lr-eta-min", type=float, default=0.0,
                   help="Lower-bound lr at the end of the cosine schedule (default 0 = decay smoothly to 0).")

    # Observation corruption (uvp version: added to all three of u, v, p)
    p.add_argument("--noise-level", type=float, default=0.0,
                   help="Gaussian noise amplitude coefficient (multiplied by each field's std). 0 means clean.")
    p.add_argument("--bias-level", type=float, default=0.0,
                   help="cos/sin bias amplitude b. 0 means no bias.")
    p.add_argument("--bias-type", type=str, default="cos", choices=["cos", "sin"])

    # L-BFGS
    p.add_argument("--lbfgs-max-iter", type=int, default=0)
    p.add_argument("--lbfgs-history-size", type=int, default=50)
    p.add_argument("--lbfgs-tol-grad", type=float, default=1e-9)
    p.add_argument("--lbfgs-tol-change", type=float, default=1e-12)
    p.add_argument("--lbfgs-print-every", type=int, default=500)

    p.add_argument("--print-every", type=int, default=500)
    p.add_argument("--plot-t-idx", type=int, default=100)
    p.add_argument("--eval-time-bins", type=int, default=20)
    # p sparse-observation subgrid: at each time step, p is observed on p_nx x p_ny uniformly distributed spatial points
    p.add_argument("--p-nx", type=int, default=5, help="Number of x-direction points (uniform) for the sparse p observations.")
    p.add_argument("--p-ny", type=int, default=10, help="Number of y-direction points (uniform) for the sparse p observations.")
    p.add_argument("--eval-subset-size", type=int, default=100000,
                   help="Number of random spacetime points used for the overall rel_l2 evaluation.")
    p.add_argument("--refresh-observation-cache", action="store_true")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--resume-from", type=str, default="",
                   help="Load an existing checkpoint (model.pt) and continue training. "
                        "If set, Adam is skipped and training goes straight to L-BFGS; typically used to extend L-BFGS. "
                        "The scenario (noise/bias/lw) must match the original run, otherwise the result is meaningless.")
    return p.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# Data
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


def sparse_p_flat_indices(nx, ny, n_time, p_nx, p_ny):
    """Flat indices of the sparse p observation points within the 1M array from flatten_reference.

    - Space: p_nx x p_ny uniformly distributed points (np.linspace, endpoints included).
    - Time: observed at every time step; all n_time steps share the same set of spatial points.
    - flatten order: flat_idx = i_space * n_time + i_time, i_space = i_y * nx + i_x.

    Returns (sorted 1D array of flat_idx, sorted 1D array of selected i_space).
    """
    ix = np.unique(np.linspace(0, nx - 1, p_nx).round().astype(np.int64))
    iy = np.unique(np.linspace(0, ny - 1, p_ny).round().astype(np.int64))
    i_space = np.array([yy * nx + xx for yy in iy for xx in ix], dtype=np.int64)
    flat = (i_space[:, None] * n_time + np.arange(n_time, dtype=np.int64)[None, :]).ravel()
    return np.sort(flat), np.sort(i_space)


def build_observation_uvp_np(flat_reference, seed, noise_level, bias_level, bias_type):
    """uvp observation corruption: noise / bias are added to all three of u, v, p (p is a supervised observation too).

    - bias: all three get the same b * f(x+2y) bias field (f = cos or sin).
    - noise: each gets noise_level * std(field) * N(0,1); the three fields' noise is sampled independently.
    """
    x = flat_reference["x"].astype("float32").copy()
    y = flat_reference["y"].astype("float32").copy()
    t = flat_reference["t"].astype("float32").copy()
    u = flat_reference["u"].astype("float32").copy()
    v = flat_reference["v"].astype("float32").copy()
    p = flat_reference["p"].astype("float32").copy()

    if bias_level > 0.0:
        if bias_type == "cos":
            field = np.cos(x + 2.0 * y).astype("float32")
        elif bias_type == "sin":
            field = np.sin(x + 2.0 * y).astype("float32")
        else:
            raise ValueError(f"unknown bias_type: {bias_type}")
        u = u + bias_level * field
        v = v + bias_level * field
        p = p + bias_level * field

    if noise_level > 0.0:
        rng = np.random.default_rng(seed)
        u = u + noise_level * u.std() * rng.standard_normal(u.shape).astype("float32")
        v = v + noise_level * v.std() * rng.standard_normal(v.shape).astype("float32")
        p = p + noise_level * p.std() * rng.standard_normal(p.shape).astype("float32")

    return {"x": x, "y": y, "t": t, "u": u, "v": v, "p": p}


def format_level_for_name(v: float) -> str:
    return f"{v:.3f}".replace(".", "p")


def make_observation_cache_name(*, seed, noise_level, bias_level, bias_type):
    # _uvp suffix: distinguishes from the uv-fulltime cache (the uvp version also corrupts p, so the contents differ)
    parts = [f"seed_{seed}", f"noise_gaussian_uvp_{format_level_for_name(noise_level)}"]
    if bias_level > 0:
        parts.append(f"bias_{bias_type}_uvp_{format_level_for_name(bias_level)}")
    return "__".join(parts) + ".npz"


def get_observation_dataset(flat_reference, seed, noise_level, bias_level, bias_type, refresh_cache):
    OBSERVATION_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    cp = OBSERVATION_CACHE_ROOT / make_observation_cache_name(
        seed=seed, noise_level=noise_level, bias_level=bias_level, bias_type=bias_type)
    if cp.exists() and not refresh_cache:
        z = np.load(cp)
        flat = {k: z[k] for k in ["x", "y", "t", "u", "v", "p"]}
        hit = True
    else:
        flat = build_observation_uvp_np(flat_reference, seed, noise_level, bias_level, bias_type)
        np.savez_compressed(cp, **flat)
        hit = False
    return flat, cp, hit


def build_pools(flat_obs, flat_ref, device):
    dtype = torch.get_default_dtype()
    # uvp version: data_pool includes p (p is a supervised observation too)
    data_pool = {k: torch.tensor(flat_obs[k], dtype=dtype, device=device)
                 for k in ("x", "y", "t", "u", "v", "p")}
    pde_pool = {k: torch.tensor(flat_ref[k], dtype=dtype, device=device)
                for k in ("x", "y", "t")}
    return data_pool, pde_pool


def build_full_batch(pool):
    """x/y/t are made grad leaves (autograd differentiates w.r.t. them); u/v/p are the supervision targets (if present in the pool)."""
    batch = {k: pool[k].detach().requires_grad_(True) for k in ("x", "y", "t")}
    for k in ("u", "v", "p"):
        if k in pool:
            batch[k] = pool[k]
    return batch


# Model + PDE
class NavierStokesPINN(nn.Module):
    def __init__(self, layers, lb, ub):
        super().__init__()
        blocks = []
        for i, o in zip(layers[:-2], layers[1:-1]):
            blocks += [nn.Linear(i, o), nn.Tanh()]
        blocks.append(nn.Linear(layers[-2], layers[-1]))
        self.net = nn.Sequential(*blocks)
        self.register_buffer("lb", torch.tensor(lb, dtype=torch.get_default_dtype()))
        self.register_buffer("ub", torch.tensor(ub, dtype=torch.get_default_dtype()))
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x, y, t):
        z = torch.cat([x, y, t], dim=1)
        z = 2.0 * (z - self.lb) / (self.ub - self.lb) - 1.0
        out = self.net(z)
        return out[:, :1], out[:, 1:2]  # psi, p


def grad(dy, dx, create_graph=True):
    if isinstance(dx, torch.Tensor): dx = [dx]
    g = torch.autograd.grad([dy], dx, grad_outputs=[torch.ones_like(dy)],
                            create_graph=create_graph, retain_graph=True, allow_unused=False)
    return [gi if gi is not None else torch.zeros_like(dx[i]) for i, gi in enumerate(g)]


def model_uvp_output(model, x, y, t):
    """uvp version: returns u, v, p. u, v come from psi gradients; p is the network's second direct output."""
    psi, p = model(x, y, t)
    return grad(psi, y)[0], -grad(psi, x)[0], p


def pde_residual(model, x, y, t):
    psi, p = model(x, y, t)
    u = grad(psi, y)[0]
    v = -grad(psi, x)[0]
    u_x, u_y, u_t = grad(u, [x, y, t])
    u_xx = grad(u_x, x)[0]; u_yy = grad(u_y, y)[0]
    v_x, v_y, v_t = grad(v, [x, y, t])
    v_xx = grad(v_x, x)[0]; v_yy = grad(v_y, y)[0]
    p_x, p_y = grad(p, [x, y])
    f_u = u_t + PDE_LAMBDA1 * (u * u_x + v * u_y) + p_x - PDE_LAMBDA2 * (u_xx + u_yy)
    f_v = v_t + PDE_LAMBDA1 * (u * v_x + v * v_y) + p_y - PDE_LAMBDA2 * (v_xx + v_yy)
    return f_u, f_v


def sse(a, b): return torch.sum((a - b).square())


def total_loss(model, data_pts, pde_pts, lw):
    # uvp-sparseP version: full-field data loss for u, v; p only on the sparse subgrid data_pts["p_sparse_idx"]
    u_p, v_p, p_p = model_uvp_output(model, data_pts["x"], data_pts["y"], data_pts["t"])
    idx = data_pts["p_sparse_idx"]
    L_data = (sse(u_p, data_pts["u"])
              + sse(v_p, data_pts["v"])
              + sse(p_p[idx], data_pts["p"][idx]))
    f_u, f_v = pde_residual(model, pde_pts["x"], pde_pts["y"], pde_pts["t"])
    L_pde = sse(f_u, torch.zeros_like(f_u)) + sse(f_v, torch.zeros_like(f_v))
    L = lw * L_data + (1.0 - lw) * L_pde
    return L, L_data, L_pde


# Training
def _zero_leaf_grads(*batches):
    for b in batches:
        for k in ("x", "y", "t"):
            b[k].grad = None


def train_adam(model, data_pts, pde_pts, *, adam_steps, adam_lr, weight_decay, lw,
               print_every, early_stop_patience=0,
               lr_schedule="none", lr_eta_min=0.0):
    """Adam training. If early_stop_patience > 0, enable best-loss + patience early stopping,
    and at the end roll the model back to the state with the lowest total loss seen.

    lr_schedule:
      - 'none':   fixed lr = adam_lr
      - 'cosine': CosineAnnealingLR, T_max = adam_steps, eta_min = lr_eta_min
    """
    opt = torch.optim.Adam(model.parameters(), lr=adam_lr, weight_decay=weight_decay)
    if lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=adam_steps, eta_min=lr_eta_min)
    else:
        scheduler = None
    hist = []; start = time.time()

    # early stopping bookkeeping
    best_loss = float("inf")
    best_step = 0
    best_state = None
    steps_since_improve = 0
    stopped_early = False

    for step in range(1, adam_steps + 1):
        opt.zero_grad(set_to_none=True)
        _zero_leaf_grads(data_pts, pde_pts)
        L, Ld, Lp = total_loss(model, data_pts, pde_pts, lw)
        L.backward(); opt.step()
        cur_lr = float(opt.param_groups[0]["lr"])
        if scheduler is not None:
            scheduler.step()
        cur_loss = float(L.detach())
        hist.append({"step": step, "phase": "adam",
                     "loss": cur_loss, "loss_data": float(Ld.detach()),
                     "loss_pde": float(Lp.detach()), "lr": cur_lr})
        if step % print_every == 0 or step == 1:
            print(f"Adam {step:6d} | L={cur_loss:.3e} | data={hist[-1]['loss_data']:.3e} | "
                  f"pde={hist[-1]['loss_pde']:.3e} | lr={cur_lr:.2e} | "
                  f"{time.time()-start:.1f}s")

        # Early-stopping logic (runs only when enabled)
        if early_stop_patience > 0:
            if cur_loss < best_loss:
                best_loss = cur_loss
                best_step = step
                best_state = copy.deepcopy(model.state_dict())
                steps_since_improve = 0
            else:
                steps_since_improve += 1
                if steps_since_improve >= early_stop_patience:
                    stopped_early = True
                    print(f"[early-stop] step {step}: total loss did not decrease for {early_stop_patience} "
                          f"consecutive steps (best={best_loss:.3e} at step {best_step}); stopping early and rolling back to best state.")
                    break

    if early_stop_patience > 0 and best_state is not None:
        model.load_state_dict(best_state)
        print(f"[early-stop] model rolled back to the state at step {best_step} (best_loss={best_loss:.3e}). "
              f"Ran {len(hist)} steps, planned {adam_steps}, "
              f"stopped_early={stopped_early}.")
    return hist, {
        "early_stop_patience": int(early_stop_patience),
        "early_stop_triggered": bool(stopped_early),
        "early_stop_step": int(hist[-1]["step"]) if hist else 0,
        "best_loss": float(best_loss) if best_state is not None else float(hist[-1]["loss"]) if hist else float("nan"),
        "best_step": int(best_step) if best_state is not None else int(hist[-1]["step"]) if hist else 0,
    }


def train_lbfgs(model, data_pts, pde_pts, *, max_iter, history_size, tol_grad, tol_change,
                lw, print_every):
    opt = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=max_iter,
                             max_eval=int(1.25 * max_iter), tolerance_grad=tol_grad,
                             tolerance_change=tol_change, history_size=history_size,
                             line_search_fn="strong_wolfe")
    hist = []; counter = {"n": 0}; start = time.time()

    def closure():
        opt.zero_grad(set_to_none=True)
        _zero_leaf_grads(data_pts, pde_pts)
        L, Ld, Lp = total_loss(model, data_pts, pde_pts, lw)
        L.backward(); counter["n"] += 1
        hist.append({"eval": counter["n"], "phase": "lbfgs",
                     "loss": float(L.detach()), "loss_data": float(Ld.detach()),
                     "loss_pde": float(Lp.detach())})
        if counter["n"] % print_every == 0 or counter["n"] == 1:
            print(f"L-BFGS {counter['n']:6d} | L={hist[-1]['loss']:.3e} | "
                  f"data={hist[-1]['loss_data']:.3e} | pde={hist[-1]['loss_pde']:.3e} | "
                  f"{time.time()-start:.1f}s")
        return L

    opt.step(closure)
    return hist


# Evaluation
def predict_full_spacetime(model, ref, device, chunk=200_000):
    flat = flatten_reference(ref)
    n = flat["x"].shape[0]
    dtype = torch.get_default_dtype()
    u_all = np.empty((n, 1), dtype="float32")
    v_all = np.empty((n, 1), dtype="float32")
    p_all = np.empty((n, 1), dtype="float32")
    model.eval()
    for i in range(0, n, chunk):
        j = min(i + chunk, n)
        x = torch.tensor(flat["x"][i:j], dtype=dtype, device=device).requires_grad_(True)
        y = torch.tensor(flat["y"][i:j], dtype=dtype, device=device).requires_grad_(True)
        t = torch.tensor(flat["t"][i:j], dtype=dtype, device=device).requires_grad_(True)
        psi, p = model(x, y, t)
        u_all[i:j] = grad(psi, y)[0].detach().cpu().numpy()
        v_all[i:j] = (-grad(psi, x)[0]).detach().cpu().numpy()
        p_all[i:j] = p.detach().cpu().numpy()
    return u_all, v_all, p_all


def rel_l2_per_time_bin(pred, ref_flat_field, n_space, n_time, n_bins):
    pred_g = pred.reshape(n_space, n_time)
    ref_g = ref_flat_field.reshape(n_space, n_time)
    bins = np.array_split(np.arange(n_time), n_bins)
    out = []
    for k, idxs in enumerate(bins):
        pr = pred_g[:, idxs].ravel(); rf = ref_g[:, idxs].ravel()
        rl = float(np.linalg.norm(pr - rf) / max(np.linalg.norm(rf), 1e-30))
        out.append({"bin_idx": k, "t_start_idx": int(idxs[0]), "t_end_idx": int(idxs[-1]), "rel_l2": rl})
    return out


# Visualization
def save_history_plot(history, out_path):
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5), constrained_layout=True)
    x = list(range(1, len(history) + 1))
    ax.plot(x, [r["loss"] for r in history], label="total")
    ax.plot(x, [r["loss_data"] for r in history], label="data (u,v,p)")
    ax.plot(x, [r["loss_pde"] for r in history], label="PDE (collocation)")
    phases = [r.get("phase", "adam") for r in history]
    if "lbfgs" in phases:
        ax.axvline(phases.index("lbfgs") + 1, color="gray", linestyle="--", alpha=0.6, label="Adam | L-BFGS")
    ax.set_yscale("log"); ax.set_xlabel("cumulative iter")
    ax.set_ylabel("SSE loss"); ax.grid(alpha=0.3); ax.legend()
    fig.savefig(out_path, dpi=160, bbox_inches="tight"); plt.close(fig)


def save_slice_plot(ref, flat_obs, pred_uvp, plot_t_idx, out_path):
    n_space = ref["x"].shape[0]; n_time = ref["t"].shape[0]
    x_grid = ref["x"].reshape(50, 100); y_grid = ref["y"].reshape(50, 100)
    u_ref = ref["u"][:, plot_t_idx:plot_t_idx + 1].reshape(50, 100)
    v_ref = ref["v"][:, plot_t_idx:plot_t_idx + 1].reshape(50, 100)
    p_ref = ref["p"][:, plot_t_idx:plot_t_idx + 1].reshape(50, 100)
    u_obs = flat_obs["u"].reshape(n_space, n_time)[:, plot_t_idx:plot_t_idx + 1].reshape(50, 100)
    v_obs = flat_obs["v"].reshape(n_space, n_time)[:, plot_t_idx:plot_t_idx + 1].reshape(50, 100)
    p_obs = flat_obs["p"].reshape(n_space, n_time)[:, plot_t_idx:plot_t_idx + 1].reshape(50, 100)
    u_pred = pred_uvp[0].reshape(n_space, n_time)[:, plot_t_idx:plot_t_idx + 1].reshape(50, 100)
    v_pred = pred_uvp[1].reshape(n_space, n_time)[:, plot_t_idx:plot_t_idx + 1].reshape(50, 100)
    p_pred = pred_uvp[2].reshape(n_space, n_time)[:, plot_t_idx:plot_t_idx + 1].reshape(50, 100)
    p_pred = p_pred - np.mean(p_pred - p_ref)

    fig, axes = plt.subplots(3, 4, figsize=(18, 12), constrained_layout=True)
    fig.suptitle(f"uvp-fulltime (full-obs, full-collocation) @ t_idx={plot_t_idx}")
    # uvp version: p has observations too, so all three rows have an obs panel
    fields = [("u", u_ref, u_obs, u_pred), ("v", v_ref, v_obs, v_pred), ("p", p_ref, p_obs, p_pred)]
    for row, (key, gt, obs, pr) in enumerate(fields):
        err = np.abs(pr - gt)
        vmin = float(gt.min()); vmax = float(gt.max())
        panels = [(f"{key}: GT", gt, "coolwarm", vmin, vmax)]
        if obs is not None:
            panels.append((f"{key}: obs", obs, "coolwarm", vmin, vmax))
        else:
            panels.append((f"{key}: (no obs, PDE only)", np.zeros_like(gt), "Greys", 0, 1))
        panels += [(f"{key}: pred", pr, "coolwarm", vmin, vmax),
                   (f"{key}: |err|", err, "magma", None, None)]
        for col, (title, field, cmap, vlo, vhi) in enumerate(panels):
            ax = axes[row, col]
            m = ax.pcolormesh(x_grid, y_grid, field, shading="auto", cmap=cmap, vmin=vlo, vmax=vhi)
            ax.set_title(title); ax.set_xlabel("x"); ax.set_ylabel("y")
            plt.colorbar(m, ax=ax, shrink=0.9)
    fig.savefig(out_path, dpi=160, bbox_inches="tight"); plt.close(fig)


def save_rel_l2_vs_t_plot(rel_u_bins, rel_v_bins, rel_p_bins, ref_t, out_path):
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.5), constrained_layout=True)
    for bins, key, color in [(rel_u_bins, "u", "C0"), (rel_v_bins, "v", "C1"), (rel_p_bins, "p", "C2")]:
        ts = [0.5 * (float(ref_t[b["t_start_idx"], 0]) + float(ref_t[b["t_end_idx"], 0])) for b in bins]
        ys = [b["rel_l2"] for b in bins]
        ax.plot(ts, ys, marker="o", label=f"rel_l2_{key}", color=color)
    ax.set_yscale("log"); ax.set_xlabel("t (bin center)"); ax.set_ylabel("rel_l2 within bin")
    ax.set_title("uvp-fulltime (full-obs, full-collocation): rel_l2 along time horizon")
    ax.grid(alpha=0.3); ax.legend()
    fig.savefig(out_path, dpi=160, bbox_inches="tight"); plt.close(fig)


def scenario_id(noise_level, bias_level, bias_type):
    parts = []
    if noise_level > 0: parts.append(f"noise_uvp_{format_level_for_name(noise_level)}")
    if bias_level > 0:  parts.append(f"bias_{bias_type}_uvp_{format_level_for_name(bias_level)}")
    return "__".join(parts) if parts else "clean"


def make_run_name(lw, sid, tag):
    parts = [f"lw_{lw:.3f}".replace(".", "p"), sid]
    if tag: parts.append(tag)
    return "_".join(parts)


def main():
    # Flush print immediately so that with nohup the log file can be followed live, without waiting for the buffer to fill.
    sys.stdout.reconfigure(line_buffering=True)

    args = parse_args()
    if not 0.0 <= args.loss_weight <= 1.0:
        raise ValueError("--loss-weight must be in [0, 1].")

    torch.set_default_dtype(DTYPE)
    seed_everything(args.seed)
    device = resolve_device(args.device)

    ref = load_reference_solution(MAT_PATH)
    flat_ref = flatten_reference(ref)
    lb = (float(ref["x"].min()), float(ref["y"].min()), float(ref["t"].min()))
    ub = (float(ref["x"].max()), float(ref["y"].max()), float(ref["t"].max()))

    model = NavierStokesPINN(layers=tuple(args.layers), lb=lb, ub=ub).to(device)

    flat_obs, cache_path, cache_hit = get_observation_dataset(
        flat_ref, seed=args.seed,
        noise_level=args.noise_level, bias_level=args.bias_level, bias_type=args.bias_type,
        refresh_cache=args.refresh_observation_cache,
    )

    data_pool, pde_pool = build_pools(flat_obs, flat_ref, device)
    data_pool_size = int(data_pool["x"].shape[0])
    pde_pool_size = int(pde_pool["x"].shape[0])
    data_pts = build_full_batch(data_pool)
    pde_pts = build_full_batch(pde_pool)

    # Sparse p observations: compute the flat indices of the sparse subgrid and put them into data_pts for total_loss
    n_time_full = ref["t"].shape[0]
    NY, NX = 50, 100  # fixed grid of cylinder_nektar_wake.mat
    p_flat_idx, p_space_idx = sparse_p_flat_indices(NX, NY, n_time_full, args.p_nx, args.p_ny)
    data_pts["p_sparse_idx"] = torch.tensor(p_flat_idx, dtype=torch.long, device=device)
    n_p_obs = int(p_flat_idx.size)

    sid = scenario_id(args.noise_level, args.bias_level, args.bias_type)
    print(f"device={device}, data: u/v full {data_pool_size}, "
          f"p sparse {len(p_space_idx)} pts/step x {n_time_full} steps = {n_p_obs}, "
          f"pde=full {pde_pool_size}, "
          f"adam_steps={args.adam_steps}, lw={args.loss_weight}, scenario={sid}, "
          f"cache_hit={cache_hit}, resume_from={args.resume_from or 'none'}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    early_stop_info = {"early_stop_patience": int(args.early_stop_patience),
                       "early_stop_triggered": False,
                       "early_stop_step": 0, "best_loss": float("nan"), "best_step": 0}

    if args.resume_from:
        ckpt_in = torch.load(args.resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt_in["model_state_dict"])
        history = list(ckpt_in.get("history", []))
        print(f"resumed from {args.resume_from}, {len(history)} prior history entries; "
              f"Adam skipped.")
    else:
        history, early_stop_info = train_adam(
            model, data_pts, pde_pts,
            adam_steps=args.adam_steps, adam_lr=args.adam_lr,
            weight_decay=args.weight_decay, lw=args.loss_weight,
            print_every=args.print_every,
            early_stop_patience=args.early_stop_patience,
            lr_schedule=args.lr_schedule, lr_eta_min=args.lr_eta_min,
        )

    if args.lbfgs_max_iter > 0:
        print(f"--- L-BFGS, max_iter={args.lbfgs_max_iter} ---")
        history += train_lbfgs(model, data_pts, pde_pts,
                                max_iter=args.lbfgs_max_iter,
                                history_size=args.lbfgs_history_size,
                                tol_grad=args.lbfgs_tol_grad,
                                tol_change=args.lbfgs_tol_change,
                                lw=args.loss_weight,
                                print_every=args.lbfgs_print_every)

    peak_gb = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else None
    if peak_gb is not None:
        print(f"peak GPU memory: {peak_gb:.2f} GB")

    print("Evaluation: full-spacetime inference...")
    u_pred, v_pred, p_pred = predict_full_spacetime(model, ref, device)
    n_space = ref["x"].shape[0]; n_time = ref["t"].shape[0]
    p_aligned = p_pred - np.mean(p_pred - flat_ref["p"])

    rng = np.random.default_rng(args.seed)
    eval_idx = rng.choice(flat_ref["x"].shape[0], size=args.eval_subset_size, replace=False)
    rel_u_total = float(np.linalg.norm(u_pred[eval_idx] - flat_ref["u"][eval_idx]) /
                       np.linalg.norm(flat_ref["u"][eval_idx]))
    rel_v_total = float(np.linalg.norm(v_pred[eval_idx] - flat_ref["v"][eval_idx]) /
                       np.linalg.norm(flat_ref["v"][eval_idx]))
    rel_p_total = float(np.linalg.norm(p_aligned[eval_idx] - flat_ref["p"][eval_idx]) /
                       np.linalg.norm(flat_ref["p"][eval_idx]))

    rel_u_bins = rel_l2_per_time_bin(u_pred, flat_ref["u"], n_space, n_time, args.eval_time_bins)
    rel_v_bins = rel_l2_per_time_bin(v_pred, flat_ref["v"], n_space, n_time, args.eval_time_bins)
    rel_p_bins = rel_l2_per_time_bin(p_aligned, flat_ref["p"], n_space, n_time, args.eval_time_bins)

    run_name = make_run_name(args.loss_weight, sid, args.tag)
    out_dir = OUTPUT_ROOT / run_name; out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = CHECKPOINT_ROOT / run_name; ckpt_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "setup": "uvp_sparseP_fullobs_full_collocation",
        "data_targets": "uvp_sparseP",
        "p_obs_nx": int(args.p_nx),
        "p_obs_ny": int(args.p_ny),
        "p_obs_per_step": int(len(p_space_idx)),
        "p_obs_total": int(n_p_obs),
        "loss_weight": float(args.loss_weight),
        "pde_weight": float(1.0 - args.loss_weight),
        "data_pool_size": data_pool_size,
        "pde_pool_size": pde_pool_size,
        "peak_gpu_gb": peak_gb,
        "noise_level": float(args.noise_level),
        "bias_level": float(args.bias_level),
        "bias_type": args.bias_type if args.bias_level > 0 else "none",
        "noise_targets": ["u", "v", "p"],
        "bias_targets": ["u", "v", "p"] if args.bias_level > 0 else [],
        "adam_steps": int(args.adam_steps), "adam_lr": float(args.adam_lr),
        "lr_schedule": args.lr_schedule, "lr_eta_min": float(args.lr_eta_min),
        "lbfgs_max_iter": int(args.lbfgs_max_iter),
        "lbfgs_n_evals": int(sum(1 for h in history if h.get("phase") == "lbfgs")),
        "optimizer": "adam+lbfgs" if args.lbfgs_max_iter > 0 else "adam",
        # Early-stopping diagnostic fields (meaningful when args.early_stop_patience > 0)
        "early_stop_patience": early_stop_info["early_stop_patience"],
        "early_stop_triggered": early_stop_info["early_stop_triggered"],
        "early_stop_step": early_stop_info["early_stop_step"],
        "best_loss_in_adam": early_stop_info["best_loss"],
        "best_step_in_adam": early_stop_info["best_step"],
        "pde_lambda1": PDE_LAMBDA1, "pde_lambda2": PDE_LAMBDA2,
        "seed": int(args.seed), "device": str(device),
        "rel_l2_u": rel_u_total, "rel_l2_v": rel_v_total, "rel_l2_p": rel_p_total,
        "rel_l2_u_bins": rel_u_bins, "rel_l2_v_bins": rel_v_bins, "rel_l2_p_bins": rel_p_bins,
        "observation_cache_path": str(cache_path), "observation_cache_hit": bool(cache_hit),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    save_history_plot(history, out_dir / "history.png")
    save_rel_l2_vs_t_plot(rel_u_bins, rel_v_bins, rel_p_bins, ref["t"], out_dir / "rel_l2_vs_t.png")
    save_slice_plot(ref, flat_obs, (u_pred, v_pred, p_pred), args.plot_t_idx,
                    out_dir / f"slice_t{args.plot_t_idx:03d}.png")

    # Save the full-field prediction as (n_time, ny, nx) so the viz notebook can use it directly.
    # ref / grid / t can all be rebuilt from the original .mat, so they are not stored redundantly.
    ny, nx = 50, 100  # fixed grid of cylinder_nektar_wake.mat
    def _to_tnyx(flat_field):
        # flat: (n_space * n_time, 1) -> (n_space, n_time) -> (n_time, n_space) -> (n_time, ny, nx)
        return flat_field.reshape(n_space, n_time).T.reshape(n_time, ny, nx).astype("float32")
    np.savez_compressed(
        out_dir / "pred.npz",
        u_pred=_to_tnyx(u_pred), v_pred=_to_tnyx(v_pred), p_pred=_to_tnyx(p_aligned),
    )

    torch.save({"model_state_dict": model.state_dict(), "metrics": metrics,
                "history": history, "loss_weight": args.loss_weight},
               ckpt_dir / "model.pt")

    print()
    print(f"rel_l2 (100K random eval points): u={rel_u_total:.4f}  v={rel_v_total:.4f}  p={rel_p_total:.4f}")
    print(f"saved_outputs={out_dir}")
    print(f"saved_checkpoint={ckpt_dir / 'model.pt'}")


if __name__ == "__main__":
    main()
