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
* Observable evaluation

### Quantum Algorithms

* **VQE** — Ground-state search
* **VQD** — Excited-state computation
* **Hamiltonian Simulation** — Lie-Trotter and Strang product formulas

### Validation & Analysis

* Exact reference benchmarking
* State fidelity
* Energy and observable errors
* Operator error
* Circuit depth and gate counts
* Publication-quality plots

### Experiment Pipeline

* YAML-based experiment configuration
* Experiment orchestration
* Simulator and IBM Quantum execution
* Timestamped output directories

## Architecture

Atlas is built from modular components that can be extended independently:

* **Physics** — Hamiltonians, observables, exact solvers
* **Algorithms** — VQE, VQD, Hamiltonian simulation
* **Circuits** — Ansatzes and evolution circuits
* **Execution** — Simulator and hardware backends
* **Analysis** — Validation metrics
* **Visualization** — Plotting utilities
* **Experiments** — Workflow orchestration

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


For implementation details and extension guidelines, see the documentation in `docs/`.

---
