## Unsteady Navier-Stokes (`/Navier_Stokes_unsteady`)

#### Governing PDE

2D unsteady incompressible Navier-Stokes; we treat `λ₁`, `λ₂` as **known** constants (not inverse problem):

$$
u_t + \lambda_1 (u u_x + v u_y) = -p_x + \lambda_2 (u_{xx} + u_{yy}),
$$

$$
v_t + \lambda_1 (u v_x + v v_y) = -p_y + \lambda_2 (v_{xx} + v_{yy}),
$$

$$
u_x + v_y = 0,
$$

with `λ₁ = 1.0`, `λ₂ = 0.01`.

> **Difference from the original paper.** Raissi et al. treated `λ₁` and `λ₂` as **unknowns** to be learned from observations (an inverse problem). Here we instead fix them to their ground-truth values and only ask the network to reconstruct the full fields `(u, v, p)`.


### Experimental setup

#### Data dimensions

From `cylinder_nektar_wake.mat`:

- **Space:** `50 × 100` grid points; `x ∈ [1, 8]`, `y ∈ [-2, 2]`
- **Time:** `200` snapshots; `t ∈ [0, 19.9]`, `Δt = 0.1`
- **Fields:** `u`, `v`, `p` each shaped `(5000, 200)`, equivalently `(200, 50, 100)`

Training draws `5000` points sampled **uniformly at random (without replacement) over the full `1M`-point spacetime grid** — i.e. scattered across all `200` time snapshots, not restricted to the initial condition `t = 0`.


#### Network

MLP `[3, 20, 20, 20, 20, 20, 20, 20, 20, 2]`, Tanh activations, Xavier init, `(x, y, t)` inputs normalized to `[-1, 1]`.

- **Outputs:** stream function `ψ` and pressure `p`.
- **Velocities:** `u = ∂ψ/∂y`, `v = -∂ψ/∂x` → incompressibility `u_x + v_y = 0` is satisfied **exactly**, not only in the loss.

#### Loss

$$ \mathcal{L} = w \cdot \mathcal{L}_{\text{data}} + (1-w)\cdot \mathcal{L}_{\text{PDE}} $$

where `w = loss_weight ∈ [0, 1]` is a scalar we sweep. Both terms are **SSE** (sum of squared errors), not MSE. Data loss depends on the observation regime:

- **uv-only setting:** `L_data = SSE(u_pred, u_obs) + SSE(v_pred, v_obs)`; `p` reconstructed entirely via PDE.
- **uvp setting:** `L_data = SSE(u_pred, u_obs) + SSE(v_pred, v_obs) + SSE(p_pred, p_obs)`; `p` is also a supervised observation.

#### Training

| Component       | Value                                                    |
|-----------------|----------------------------------------------------------|
| Training points | `n_train = 5000` random `(x, y, t)` (same set for data loss and PDE collocation) |
| Seed            | `1234`                                                   |
| Adam            | `10000` steps, `lr = 1e-3`, no weight decay              |
| L-BFGS (optional, suffix `_lbfgs`) | `max_iter = 10000`, `history_size = 50`, strong-Wolfe line search |
| L-BFGS extended (suffix `_r2`)     | resumed from `_lbfgs` checkpoint, `max_iter = 50000`, same other settings |

Run directories use suffixes to disambiguate:
- *no suffix* — Adam-only
- `_lbfgs` — Adam + L-BFGS 10K
- `_r2` — Adam + L-BFGS 10K + L-BFGS 50K extension

#### Observation corruption

Three corruption types, applied on top of the clean reference fields:

| Type           | Form                                                            | Levels swept            |
|----------------|-----------------------------------------------------------------|-------------------------|
| Gaussian noise | `field_obs = field_true + noise_level × std(field) × N(0, 1)`   | `0.05` (≈ 5% per-field) |
| Cosine bias    | `field_obs = field_true + b × cos(x + 2y)`                      | `b ∈ {0.3, 0.5, 0.8}`   |
| Sin bias       | `field_obs = field_true + b × sin(x + 2y)`                      | `b ∈ {0.3, 0.5, 0.8}`   |

Combined (e.g. `noise + cosine bias`) scenarios are also covered. In the **uv-only** setting, corruption is applied to `u` and `v` only; in the **uvp** setting, it is applied to all three of `u, v, p` (and for noise, sampled independently per field).

#### Loss-weight sweep

For each scenario, we sweep `loss_weight ∈ {0.01, 0.05, 0.2, 0.5, 0.8, 1.0}`:
- `w = 1.0` ⇒ pure data fit (PDE term off — degenerates to ordinary supervised regression)
- `w → 0` ⇒ PDE-dominated (data only as a soft hint)
- `w = 0` was previously included but **omitted from releases** because it produces a degenerate solution `ψ ≡ 0`, `p ≡ const` regardless of scenario (no data anchor; `rel_l2_u ≈ 1.0` always).

> ⚠️ **Caveat on `loss_weight` magnitude.** The nominal `w = 0.5` does **not** correspond to a 50:50 split between data and PDE losses, because the two terms have very different absolute magnitudes (data SSE is typically much larger than PDE residual SSE at convergence). Empirically a nominal `w = 0.5` behaves more like a 87:13 data-vs-PDE split. To genuinely probe the "PDE-dominated" regime one needs `w ∈ {0.01, 0.05, 0.1}` — which is why those small values are included in the sweep.

### Demo

For the demo, take a look at `Navier-Stokes-uncertainty-visualization.ipynb`. This notebook visualizes the results under different loss weights and different uncertainty settings, and contains the detailed experimental setup.

### Data

All experimental data is under `/Navier_Stokes_unsteady`:

- [`DATA_observations/`](./Navier_Stokes_unsteady/DATA_observations) — reference solution, experiment summary, and observation data:
  - `reference_t100.npz` — shared reference solution (`u_ref`, `v_ref`, `p_ref`) + grid, at `t = 10.0`
  - `summary.csv` — config for all 40 runs (`run_name`, `scenario`, `loss_weight`, noise/bias settings)
  - `clean.npz` — clean observation
  - `noise_gaussian_0p050.npz` — Gaussian noise only
  - `bias_cosine_0p{300,500,800}.npz` — cosine bias only, at different magnitudes
  - `noise_gaussian_0p050__bias_cosine_0p{300,500,800}.npz` — combined Gaussian noise + cosine bias
- [`DATA_per_model_t100/`](./Navier_Stokes_unsteady/DATA_per_model_t100) — predicted fields (`psi`, `u`, `v`, `pressure`) for each of the 40 runs

See [`load_one_export_example.ipynb`](./Navier_Stokes_unsteady/load_one_export_example.ipynb) for a full guide to loading and visualizing all of the above (reference, summary, observations, predictions), including an explanation of what `loss_weight` means.

### Training

The model was trained using `Navier-Stokes-uncertainty.py`. All the training commands are in [`./Navier_Stokes_unsteady/run_command.txt`](./Navier_Stokes_unsteady/run_command.txt).


### Notes

`Navier-Stokes.ipynb` (the reproduction of the original paper above) was used for yesterday's presentation, so it is not directly related to the demo.

---

## Lid-Driven Cavity (`/Navier_Stokes_liddriven`)

2D **steady** incompressible Navier-Stokes on $[0,1]^2$, driven by a top lid with parabolic profile $u = a\,x(1-x),\; v = 0$ (other walls no-slip). No training code yet — this sub-project is currently just an exploration of the reference data.

### Data

Reference solutions come from the [PINNacle](https://github.com/i207M/PINNacle) benchmark (COMSOL-generated `.dat` files under `ref/lid_driven_a{2,4,6,8,10}.dat`, corresponding to $a \in \{2,4,6,8,10\}$, i.e. Re = 25a). The notebooks read them directly from `/data/pinns/PINNacle/ref/` — adjust the path if you move the data.

### Notebooks

- [`ns_liddriven.ipynb`](./Navier_Stokes_liddriven/ns_liddriven.ipynb) — **PDE statement + raw data visualization.** Writes out the 2D steady incompressible NS equations, boundary conditions, and loss structure used by PINNacle, then visualizes the COMSOL reference fields: $u$, $v$, $p$, $|V|$ heatmaps for $a=4$, a standalone streamline plot, and a side-by-side streamline comparison across all five $a$ values.
- [`explore_stream_function.ipynb`](./Navier_Stokes_liddriven/explore_stream_function.ipynb) — **Stream function of the raw data.** Derives $\psi$ from the reference $(u,v)$ at $a=4$: theory (definition, existence, uniqueness), three recovery methods (line integral / Poisson solve / least squares), and a worked implementation of the Poisson method including vorticity, $\psi$ heatmap, and streamline contours, with sanity checks.

