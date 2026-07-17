# Data generation guide

This document describes how to run Atlas experiments from YAML configs, which
plots/CSVs each configuration produces, and how to open the interactive
lattice dashboard.

## Running an experiment

From the repository root:

```bash
python -m atlas.main --config configs/<config-file>.yaml
```

Each run writes artifacts under a timestamped directory:

```
atlas/data/<experiment.name>_<YYYYMMDD_HHMMSS>/
```

The base path is controlled by `output.directory` in the config (default: `atlas/data`).

## Interactive lattice dashboard

For Hamiltonian-simulation time series with per-site observables, open a
Matplotlib dashboard (time slider + play/pause) without writing a full batch
output tree:

```bash
python scripts/run_lattice_dashboard.py
python scripts/run_lattice_dashboard.py --config configs/tfim_hamiltonian_sim_time_sweep.yaml --method strang
python scripts/run_lattice_dashboard.py --n 6 --times 0,0.25,0.5,1,2 --observable z --save frame.png --no-show
```

Requires `analysis.site_observables` (for example `local_z`) so site arrays
exist on the result. The dashboard never recomputes physics.

## Global requirements

| Setting | Effect |
|---------|--------|
| `output.plots: true` | Generate PNG plots (including lattice dashboard snapshots when site data exist) |
| `output.csv: true` | Write CSV summaries |
| `analysis.fidelity: true` | Compute fidelity/infidelity (required for infidelity plots) |
| `analysis.observables: tfim_default` | Enable scalar observable plots (`zz`, `x`, …) |
| `analysis.site_observables: local_z` | Per-site arrays (`z`) for the lattice dashboard (`local_x` / `local_y` also supported) |

State-overlap plots use **infidelity** (`1 − fidelity`) on a log y-axis. Fidelity is still stored in results and CSVs.

## Evolution method comparison

Hamiltonian simulation configs use `evolution_methods: [lie, strang]` so Lie and Strang are compared on the same axes (legend per method). VQE/VQD configs are variational and do not use product formulas.

---

## Config catalog

| Config | Algorithm | Type | Sweep / notes |
|--------|-----------|------|----------------|
| `tfim_vqe_single.yaml` | VQE | `single_point` | — |
| `tfim_vqe.yaml` | VQE | `sweep` | `h` |
| `tfim_vqd.yaml` | VQD | `single_point` | — |
| `tfim_vqd_sweep.yaml` | VQD | `sweep` | `h` |
| `tfim_hamiltonian_sim_single.yaml` | hamiltonian_sim | `single_point` | Lie + Strang |
| `tfim_hamiltonian_sim_time_sweep.yaml` | hamiltonian_sim | `sweep` | `evolution_time`, Lie + Strang; includes `site_observables: local_z` |
| `tfim_hamiltonian_sim_h_sweep.yaml` | hamiltonian_sim | `sweep` | `h`, Lie + Strang |
| `tfim_hamiltonian_sim_j_sweep.yaml` | hamiltonian_sim | `sweep` | `J`, Lie + Strang |
| `tfim_trotter_validation.yaml` | hamiltonian_sim | `trotter_validation` | Trotter steps, Lie + Strang |

---

## Plots by algorithm

### VQE

| Plot | When generated |
|------|----------------|
| `vqe_single_point_energy.png` | `single_point` |
| `energy_vs_hJ.png` | `sweep` over `h` |
| `magnetization_comparison.png` | same |
| `infidelity_vs_hJ.png` | same |
| `sim_vs_hardware_error.png` | same |
| `sim_vs_hardware_parity.png` | same + `hardware.enabled: true` |
| `relative_error.png` | same + hardware |

**CSV:** `tfim_benchmark.csv` (sweep only)

---

### VQD

| Plot | When generated |
|------|----------------|
| `vqd_single_point_summary.png` | `single_point` |
| `vqd_energy_vs_hJ.png` | `sweep` over `h` |
| `vqd_infidelity_vs_hJ.png` | same |
| `vqd_energy_error_vs_hJ.png` | same |

**CSV:** `tfim_vqd_benchmark.csv` (sweep only)

---

### Hamiltonian simulation — single point

| Plot | When generated |
|------|----------------|
| `sim_single_point_observables.png` | Grouped exact vs sim bars per method |
| `lattice_dashboard_{obs}_{sim\|exact}_{method}.png` | When `site_observables` is set |

---

### Hamiltonian simulation — sweeps (`evolution_time`, `h`, `J`)

| Plot | `system.sweep.parameter` |
|------|--------------------------|
| `sim_infidelity_vs_evolution_time.png` | `evolution_time` |
| `sim_{obs}_evolution_vs_time.png` | `evolution_time` |
| `sim_infidelity_vs_sweep.png` | `h` or `J` |
| `sim_{obs}_vs_sweep.png` | `h` or `J` |
| `lattice_dashboard_{obs}_{sim\|exact}_{method}.png` | When `site_observables` is set (snapshot of last frame) |

Multi-method runs include one line per method (Lie, Strang) plus exact reference on observable plots.

**CSV:** `sim_validation.csv` (multi-method) or `sim_benchmark.csv` (single-method)

---

### Trotter validation

Dedicated step-count study at fixed `evolution_time`. Requires `num_trotter_steps` as a **list** in algorithm parameters.

| Plot | Notes |
|------|-------|
| `validation_infidelity_vs_trotter_steps.png` | log–log |
| `validation_operator_error_vs_trotter_steps.png` | log–log; Lie ≈ −1 / Strang ≈ −2 guides |
| `validation_circuit_depth_vs_infidelity.png` | log y |
| `sim_infidelity_vs_sweep.png` | Same data, sweep-style layout |
| `sim_{obs}_vs_sweep.png` | Per observable |

**CSV:** `sim_validation.csv`

---

## Quick lookup

| I want… | Config / command |
|---------|------------------|
| Lie vs Strang, Trotter steps | `tfim_trotter_validation.yaml` |
| Infidelity vs evolution time (Lie + Strang) | `tfim_hamiltonian_sim_time_sweep.yaml` |
| Interactive spin-chain dashboard | `scripts/run_lattice_dashboard.py` (+ time-sweep config) |
| Observables vs `h` or `J` (Lie + Strang) | `tfim_hamiltonian_sim_h_sweep.yaml` / `_j_sweep.yaml` |
| Single-point method comparison | `tfim_hamiltonian_sim_single.yaml` |
| VQE / VQD benchmarks | `tfim_vqe.yaml` / `tfim_vqd_sweep.yaml` |

## Examples

```bash
python -m atlas.main --config configs/tfim_trotter_validation.yaml
python -m atlas.main --config configs/tfim_hamiltonian_sim_time_sweep.yaml
python -m atlas.main --config configs/tfim_hamiltonian_sim_single.yaml
python -m atlas.main --config configs/tfim_vqe.yaml
python scripts/run_lattice_dashboard.py --config configs/tfim_hamiltonian_sim_time_sweep.yaml --method strang
```
