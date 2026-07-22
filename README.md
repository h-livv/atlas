# Atlas

**A physics-first framework for studying quantum systems through simulation and quantum algorithms.**

Atlas is designed to serve as an experimental environment for implementing quantum algorithms from first principles, reproducing results from literature, validating them against exact solutions, and investigating their behavior on both simulators and quantum hardware.

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

### Implemented Quantum Methods
* **VQE** — Ground-state search
* **VQD** — Excited-state computation
* **Hamiltonian Simulation** — Lie-Trotter and Strang product formulas

### Validation

- Exact reference solutions
- Fidelity and observable errors
- Circuit resource analysis
- Static and interactive visualization

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

---

## Documentation

- [`Usage Guide`](docs/data/data_generation_guide.md)
- [`Extension Guide`](docs/architecture/atlas_component_extension_guide.md)
- [`Architecture`](docs/architecture/architecture_analysis.md)
- [`Simulation Performace Analysis`](docs/architecture/hamiltonian_sim_performance_optimization_analysis.md)
- [`VQE Notes`](docs/validation/vqe_tfim.md)

---

## Research Directions

### Quantum Algorithms

- Variational algorithms
- Hamiltonian simulation
- Fault-tolerant simulation algorithms
- Quantum Phase Estimation
- Quantum Signal Processing
- Qubitization

### Physical Systems

- Strongly correlated spin systems
- Lattice fermion models
- Quantum chemistry
- Open quantum systems

### Validation Studies

- Product formula error analysis
- Noise characterization
- Hardware benchmarking
- Resource estimation

---

Atlas is an evolving laboratory for computational quantum physics. Its goal is to understand quantum algorithms by implementing them from first principles, validating them against known physics, and exploring where they succeed, fail, and scale.

---
