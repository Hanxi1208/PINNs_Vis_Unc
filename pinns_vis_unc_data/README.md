# PINNs Uncertainty Visualization
## 1. Physical problem
2-D unsteady incompressible Navier–Stokes, cylinder wake (the dataset used by Raissi et al. 2019, [*Physics-informed neural networks*](https://www.sciencedirect.com/science/article/pii/S0021999118307125)):

$$
u_t +  u u_x + v u_y = -p_x + 0.01 (u_{xx} + u_{yy}),
$$
$$
v_t + u v_x + v v_y = -p_y + 0.01 (v_{xx} + v_{yy}),
$$
$$
u_x + v_y = 0.
$$

**Problem setup.** We study a forward problem for the purpose of uncertainty visualization:

1. **Observations.** The observations are constructed from the **reference solution** (the GT field from `cylinder_nektar_wake.mat`) by adding noise or sin/cos bias on top of it — i.e. `observation = reference + noise`. They are full field of `(u, v)` or `(u, v, p)`. The corruption is Gaussian noise or a deterministic bias; the `clean` scenario uses the reference directly.
   > An earlier version of this experiment randomly sampled `5,000` spacetime points each iteration and evaluated both the data loss and the PDE residual on that subset — a quick first pass to try out PINNs. This release instead uses the full grid for both, removing the subsampling.
2. **Hypothesized PDE.** We assume the observations are governed by the Navier–Stokes equations above — a cylinder-wake flow at Reynolds number `Re = 100`. The PDE acts as a hypothesized physical prior on the data.
3. **Goal.** Train a neural network to reconstruct the full field `(u, v, p)`, trading off fidelity to the observations against the hypothesized PDE. The loss weight `w` controls that trade-off.

### Data dimensions

Reference solution from `cylinder_nektar_wake.mat`:

| Quantity        | Value                                                        |
|-----------------|--------------------------------------------------------------|
| Spatial grid    | `50 × 100 = 5000` points; `x ∈ [1, 8]`, `y ∈ [-2, 2]`        |
| Temporal grid   | `200` snapshots; `t ∈ [0, 19.9]`, `Δt = 0.1`                 |
| Fields          | `u`, `v` (velocity), `p` (pressure)                          |
| Spacetime total | `5000 × 200 = 1,000,000` points                              |

The full field's shape is **`(n_time, ny, nx) = (200, 50, 100)`**.

---

## 2. Experiment design
We consider three different observation regimes:
- Full field `(u, v)` is observed
- Full field `(u, v, p)` is observed
- Full field `(u, v)` is observed, and sparse uniform `p` is observed 
> Velocity `(u, v)` can be measured over the full domain by Particle Image Velocimetry (PIV), whereas the pressure `p` is usually only accessible at a few sparse sensor locations.

The observed data may carry various kinds of uncertainty:
- Gaussian noise
- Deterministic sin/cos bias 


### 2.1 Observation scenario

A scenario defines how the observations are corrupted: `f_obs = f + corruption`. 

| Scenario        | Definition                                                       |
|-----------------|------------------------------------------------------------------|
| `clean`         | no corruption (the observation is the reference itself)          |
| `noise0.05`     | Gaussian noise: `f_obs = f + 0.05 · std(f) · N(0, 1)` (sampled independently per field) |
| `bias_cos0.3`   | deterministic bias: `f_obs = f + 0.3 · cos(x + 2y)`              |
| `bias_cos0.5`   | deterministic bias: `f_obs = f + 0.5 · cos(x + 2y)`              |
| `bias_sin0.3`   | deterministic bias: `f_obs = f + 0.3 · sin(x + 2y)`              |
| `bias_sin0.5`   | deterministic bias: `f_obs = f + 0.5 · sin(x + 2y)`              |

### 2.2 Loss weight sweep

The training objective mixes a data term and a PDE-residual term:

$$ \mathcal{L} = w \cdot \mathcal{L}_{\text{data}} + (1-w) \cdot \mathcal{L}_{\text{PDE}} $$

with **`w = loss_weight ∈ {0.01, 0.05, 0.2, 0.5, 0.8, 1.0}`**:

- `w = 1.0` — pure observation data fit, PDE term off.
- `w = 0.01` — the smallest weight in the data, strongly PDE-dominated.

> The sweep stops at `w = 0.01`, not `0`. With no data term, the PDE residual is the only constraint; but our PDE carries no initial/boundary condition, so it does not pin down a unique solution — the network collapses to a trivial solution. A small data term must therefore remain as a supervised anchor.

Both `L_data` and `L_PDE` are SSE (sum of squared errors).

### 2.3 Run count
- `clean` (only `w = 0.5`) 
-  5 corrupted scenarios × 6 loss weights = 30 runs. 
  
Three regimes → 93 runs total.

---

## 3. Training method
Identical for every run.

### Network
- Inputs `(x, y, t)` normalized to `[-1, 1]`.
- Outputs the stream function `ψ` and pressure `p`. Velocity is recovered as `u = ∂ψ/∂y`, `v = -∂ψ/∂x`, so the incompressibility naturally constraint `u_x + v_y = 0`.
- MLP, layer sizes `[3, 20, 20, 20, 20, 20, 20, 20, 20, 2]` — 8 hidden layers, width 20; `tanh` activations, Xavier-normal init.

### Optimizer — Adam then L-BFGS
1. Adam: `10,000` steps, learning rate `1e-3`, no weight decay.
2. L-BFGS: started from the Adam result, `max_iter = 10,000`, history size `50`.

### Full-observation / full-collocation
Both the data loss and the PDE residual are evaluated on full field every iteration.


### Code

The training scripts are in `training_code/`.

---



