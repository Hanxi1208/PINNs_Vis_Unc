# PINNs_uncertainty

This project focuses on Navier-Stokes Equations.

## Sub-projects

- [Unsteady Navier-Stokes](./Navier_Stokes_unsteady) — The dataset comes from the paper [*Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations*](https://www.sciencedirect.com/science/article/pii/S0021999118307125), where it was used to solve an inverse problem. `Navier-Stokes.ipynb` is an implementation of that original paper. However, our current experimental setup is not entirely consistent with the original paper — details are described below. 
- Lid-Driven Cavity (steady) **Not yet completed.**

---

## `pinns_vis_unc_data/`-- Data release  

The full dataset for the unsteady Navier-Stokes uncertainty experiments: the reference solution, the (corrupted) observations, predictions from all 93 trained models over the full time domain, and the training scripts.

**Download (~1.1 GB zip):** https://drive.google.com/file/d/1zJyVYh0e65n8yhxoUBKqtnldQtCbuiIa/view?usp=drive_link

See [`pinns_vis_unc_data/README.md`](./pinns_vis_unc_data/README.md) for the experiment design, data layout, and a loading example.

---

### `Navier_Stokes_unsteady/`

This is the first, simplified version of the unsteady Navier-Stokes experiment: randomly samples only `5,000` spacetime points to compute both the data and PDE loss. **Now deprecated** — superseded by the `pinns_vis_unc_data/` data release above.

---

