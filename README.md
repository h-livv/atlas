# Atlas

**Atlas is a physics-first quantum simulation framework for studying quantum systems.**

The framework combines modular Hamiltonian definitions, quantum algorithms, exact reference solvers, and benchmarking tools into a unified workflow for studying equilibrium properties and real-time dynamics.

---

## Philosophy

Most quantum software organizes around algorithms or hardware backends. Atlas organizes around **physical models**.

1. **Define the system** — Hamiltonian $H$, parameters, and physically meaningful observables.
2. **Choose a quantum method** — variational ground-state search, excited-state deflation, or Hamiltonian simulation.
3. **Compare against physics** — on small systems, Atlas computes exact reference solutions (diagonalization, exact time evolution) so algorithm output is judged against the true quantum-mechanical answer, not just internal consistency.
4. **Run on hardware** — run algorithms on IBM Quantum devices and compare with simulation results to analyze effects of noise.
5. **Explore parameter space** — sweep coupling ratios, evolution times, or discretization steps and record how physical quantities and errors respond.

This makes Atlas a **benchmarking and discovery environment** for NISQ-era methods applied to condensed-matter-style problems.

---

## Current Capabilities

### Physical Models

* Transverse-field Ising model (TFIM)
* Exact diagonalization (small systems)
* Exact time evolution
* Observable evaluation (global scalars and optional per-site arrays)

### Quantum Algorithms

* **VQE** — Ground-state search
* **VQD** — Excited-state computation
* **Hamiltonian Simulation** — Lie-Trotter and Strang product formulas

### Validation & Analysis

* Exact reference benchmarking
* State fidelity / infidelity
* Energy and observable errors
* Operator error
* Circuit depth
* Publication-quality static plots
* Interactive lattice dashboard (Matplotlib time slider + play/pause)

### Experiment Pipeline

* YAML-based experiment configuration
* Experiment orchestration (`single_point`, `sweep`, `trotter_validation`)
* Simulator and IBM Quantum execution (variational path)
* Timestamped output directories (CSV + PNG artifacts)

---

## Quick start

From the repository root (with dependencies from `requirements.txt` installed):

```bash
# Batch experiment (CSV / static plots)
python -m atlas.main --config configs/tfim_vqe_single.yaml
python -m atlas.main --config configs/tfim_hamiltonian_sim_time_sweep.yaml

# Interactive lattice dashboard (dynamics + per-site observables)
python scripts/run_lattice_dashboard.py
python scripts/run_lattice_dashboard.py --config configs/tfim_hamiltonian_sim_time_sweep.yaml --method strang
```

Each `atlas.main` run writes under `atlas/data/<experiment_name>_<timestamp>/`.

The lattice dashboard consumes precomputed site series (for example `analysis.site_observables: local_z`). It does not recompute physics. See `docs/data/data_generation_guide.md`.

---

## Architecture

Atlas is built from modular components that can be extended independently:

* **Physics** — Hamiltonians, observables, exact solvers
* **Algorithms** — VQE, VQD, Hamiltonian simulation
* **Circuits** — Ansatzes and evolution circuits
* **Execution** — Simulator and hardware backends
* **Analysis** — Validation metrics
* **Visualization** — Static plots and a physics-agnostic lattice dashboard
* **Experiments** — Workflow orchestration

Dynamics visualization under `atlas/visualization/hamiltonian_sim/` is a **renderer only**: it draws supplied geometry and site values (1D chains today; graphs/2D layouts via the same renderer). It never inspects Hamiltonians or evolution methods.

---

## Documentation

| Document | Contents |
|----------|----------|
| [`docs/data/data_generation_guide.md`](docs/data/data_generation_guide.md) | How to run configs; plot/CSV catalog; lattice dashboard |
| [`docs/architecture/atlas_component_extension_guide.md`](docs/architecture/atlas_component_extension_guide.md) | How to extend systems, algorithms, observables, plots |
| [`docs/architecture/architecture_analysis.md`](docs/architecture/architecture_analysis.md) | Current codebase architecture |
| [`docs/architecture/hamiltonian_sim_performance_optimization_analysis.md`](docs/architecture/hamiltonian_sim_performance_optimization_analysis.md) | Dynamics performance bottleneck & optimization design (no impl yet) |
| [`docs/validation/vqe_tfim.md`](docs/validation/vqe_tfim.md) | VQE / TFIM physics and validation notes |
| [`docs/architecture/architecture_review_for_ham_sim.md`](docs/architecture/architecture_review_for_ham_sim.md) | Historical ham-sim design review (superseded) |

---

## Roadmap

Future development focuses on expanding both supported physical systems and quantum methods.

**Physical systems**

- Heisenberg model
- Hubbard model
- Molecular Hamiltonians

**Quantum methods**

- Higher-order Suzuki formulas
- QDrift and randomized product formulas
- Quantum Signal Processing (QSP)
- Qubitization
- Phase Estimation
- Time-dependent Hamiltonian simulation

**Visualization**

- Edge / bond observable overlays on the lattice dashboard
- Additional lattice views (2D grids, arbitrary graphs) using the existing graph renderer

---
