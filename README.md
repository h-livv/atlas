# Atlas

### A modular quantum simulation framework for simulating and benchmarking physically relevant quantum systems. 

VQE is the first implemented algorithm; the package is organized around framework-level concepts: Hamiltonians, ansätze, algorithms, execution backends, analysis, and visualization. New models and solvers can be added without restructuring the core.

## Architecture

```
Hamiltonian → Algorithm → Backend → Experiment → Analysis → Visualization
```

The `atlas/` package separates physical models from algorithm logic, execution, and post-processing:

```text
atlas/
  main.py                 # CLI orchestration
  config.py               # Experiment and execution configuration
  physics/                # Hamiltonians and observables
  circuits/ansatzes/      # Parameterized circuit families
  algorithms/vqe.py       # VQE (first implemented algorithm)
  optimization/           # Classical optimizers
  execution/              # Simulator and IBM Runtime backends
  experiments/            # Experiment orchestration and result containers
  analysis/               # Post-processing metrics
  visualization/          # Plotting and circuit drawing
  io/                     # CSV loading and writing
```

Legacy scripts and benchmark data from the original monolithic implementation live under `vqe_legacy/`.

## Current Capabilities

- **Hamiltonians** — n-qubit Transverse Field Ising Model (TFIM) with exact diagonalization for small-system benchmarks
- **Ansätze** — hardware-efficient ansatz with configurable depth
- **Algorithms** — multi-start Variational Quantum Eigensolver (VQE)
- **Optimization** — COBYLA via SciPy, with multi-start restarts
- **Execution** — local statevector simulator and IBM Quantum Runtime (`EstimatorV2`)
- **Experiments** — single-point simulator runs, hardware evaluation, and transverse-field benchmark sweeps
- **Analysis** — ground-state energy, state fidelity, magnetization observables, absolute and relative error
- **Visualization** — energy curves, magnetization phase diagrams, parity plots, and circuit drawing
- **I/O** — CSV export and import for benchmark and hardware results

## Quick Start

Atlas currently supports simulator execution, IBM Quantum Runtime execution, and automated TFIM benchmark sweeps.

Install dependencies:

```bash
pip install -r requirements.txt
```

Run a single-point simulator benchmark (default: 2-qubit TFIM, J = 1.0, h = 0.5):

```bash
python -m atlas.main --mode sim --J 1.0 --h 0.5
```

Run a transverse-field sweep with plots and CSV output:

```bash
python -m atlas.main --mode benchmark --J 1.0 --h-values 0.1,0.5,1.0,1.5,2.0 --output-dir atlas/data
```

Evaluate pre-optimized parameters on IBM hardware (requires a configured IBM Quantum account):

```bash
python -m atlas.main --mode hardware --J 1.0 --h 0.5
```

See [docs/atlas_component_extension_guide.md](docs/atlas_component_extension_guide.md) for how to add new Hamiltonians, ansätze, algorithms, backends, and experiments.

## Variational Quantum Eigensolver

- The first implemented algorithm in Atlas is the Variational Quantum Eigensolver (VQE), demonstrated on the Transverse Field Ising Model (TFIM)
- The algorithm was run for several values of magnetization ratios
- The results were benchmarked against analytical diagonalization and actual hardware execution

<img width="800" height="500" alt="energy_vs_hJ" src="https://github.com/user-attachments/assets/e16cde82-4e27-4bb2-be78-d92840b6ca82" />

## Roadmap

The VQE module is the first completed algorithmic component. Planned development expands Atlas into a broader quantum simulation platform:

```
Current
✓ Variational Quantum Eigensolver (VQE)
✓ Modular n-qubit TFIM Hamiltonian and hardware-efficient ansatz
✓ IBM Quantum Runtime integration
✓ TFIM benchmarking pipeline

↓

Variational Quantum Deflation (VQD)

↓

Expanded System Sizes and Ansätze
(4, 8, and 16 qubits; UCCSD and Hamiltonian Variational Ansatz)

↓

Quantum Dynamics
(Trotter-Suzuki decomposition, product formula methods)

↓

Hardware-Aware Execution
(Zero-noise extrapolation, probabilistic error cancellation via Mitiq)

↓

Additional Hamiltonian systems
(Heisenberg, Hubbard, molecular Hamiltonians)
```

| Area | Status |
|------|--------|
| VQE | Done |
| IBM Quantum Runtime | Done |
| TFIM benchmarking | Done |
| n-qubit Hamiltonian and ansatz infrastructure | Done |
| Variational Quantum Deflation (VQD) | Planned |
| Qubit count expansion (4, 8, 16) | Planned |
| Alternative ansätze (UCCSD, HVA) | Planned |
| Quantum dynamics (Trotterization) | Planned |
| Advanced error mitigation (ZNE, PEC) | Planned |
| Additional Hamiltonians | Planned |

## Documentation

- [VQE for TFIM — theory, implementation, and validation](docs/vqe_tfim.md)
- [Component extension guide](docs/atlas_component_extension_guide.md)
