# PINNs_uncertainty

This project focuses on different Navier-Stokes Equations.

## Sub-projects

- [Unsteady Navier-Stokes](./Navier_Stokes_unsteady) — the first Navier-Stokes Equation in this project.

---

## Unsteady Navier-Stokes (`/Navier_Stokes_unsteady`)

The dataset comes from the paper *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations* (Raissi et al.), where it was originally used to solve an inverse problem. `Navier-Stokes.ipynb` is an implementation of that original paper. However, our current experimental setup is not entirely consistent with the original paper — details are described below.

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
