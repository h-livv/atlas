# Atlas

A modular quantum simulation framework for simulating and benchmarking physically relevant quantum systems. Experiments are driven by YAML configuration files; VQE and VQD are the first implemented variational algorithms. The package is organized around Hamiltonians, ansätze, algorithms, execution backends, analysis, and visualization so new models and solvers can be added without restructuring the core.

## Architecture

```
YAML config → load_config → factory/builders → experiment → outputs (CSV, plots)
```

The `atlas/` package separates physical models from algorithm logic, execution, and post-processing:

```text
atlas/
  main.py                      # Thin CLI: --config only
  config.py                    # AtlasConfig and legacy dataclasses
  io/
    yaml_config.py             # YAML loading and validation
    csv_io.py                  # TFIM benchmark CSV I/O
  experiments/
    builders.py                # Component builders (hamiltonian, ansatz, algorithm, …)
    factory.py                 # Experiment assembly and run dispatch
    tfim.py                    # TFIM sweep and single-point workflows
    outputs.py                 # Per-run output directory, CSV, plots, summaries
    results.py                 # VQE, VQD, and TFIM result containers
  physics/                     # Hamiltonians and observables
  circuits/ansatzes/           # Parameterized circuit families
  algorithms/
    vqe.py                     # Multi-start VQE
    vqd.py                     # VQD built on VQE (overlap penalties)
    hamiltonian_sim.py         # Product-formula Hamiltonian simulation
  optimization/                # Classical optimizers (SciPy COBYLA)
  execution/                   # Statevector estimator, evolver, and IBM Runtime
  analysis/                    # Post-processing metrics
  visualization/
    tfim_plots.py              # VQE benchmark plots
    tfim_vqd_plots.py          # VQD benchmark and single-point plots
    circuit_drawer.py          # Circuit drawing utilities

configs/                       # Example experiment YAML files
```

Legacy scripts and benchmark data from the original monolithic implementation live under `vqe_legacy/`.

## Current Capabilities

- **Configuration** — YAML-driven experiments via `AtlasConfig`; no TFIM/VQE flags on the CLI
- **Hamiltonians** — n-qubit Transverse Field Ising Model (TFIM) with exact diagonalization and spectrum for small-system benchmarks
- **Ansätze** — hardware-efficient ansatz with configurable depth (`reps`)
- **Algorithms**
  - **VQE** — multi-start Variational Quantum Eigensolver
  - **VQD** — Variational Quantum Deflation reusing VQE's ground-state stage and shared optimization loop
  - **Hamiltonian Simulation** — Trotter product formulas (Lie, Strang) with exact reference evolution
- **Optimization** — COBYLA via SciPy, with configurable multi-start restarts
- **Execution** — local statevector simulator and IBM Quantum Runtime (`EstimatorV2`)
- **Experiments** — single-point runs and transverse-field (`h`) sweeps for VQE and VQD
- **Analysis** — ground-state and excited-state energies, per-state fidelity, absolute error, magnetization observables
- **Visualization** — VQE benchmark plots (energy, fidelity, magnetization, errors); VQD per-state summary and sweep plots
- **I/O** — CSV export for VQE and VQD benchmarks; each run writes to a unique timestamped folder under `atlas/data`

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run a single-point VQE simulation (edit `configs/tfim_vqe_single.yaml` to change parameters):

```bash
python -m atlas.main --config configs/tfim_vqe_single.yaml
```

Run a transverse-field VQE benchmark sweep with plots and CSV:

```bash
python -m atlas.main --config configs/tfim_vqe.yaml
```

Run a single-point VQD simulation (finds multiple eigenstates via overlap penalties):

```bash
python -m atlas.main --config configs/tfim_vqd.yaml
```

Each run creates a unique output directory:

```text
atlas/data/{experiment_name}_{YYYYMMDD_HHMMSS}/
```

Enable or disable CSV and plots in the YAML `output` section:

```yaml
output:
  directory: atlas/data
  csv: true
  plots: true
```

To evaluate on IBM hardware, set `hardware.enabled: true` in the config (requires a configured IBM Quantum account). When enabled, the factory resolves a suitable backend automatically or uses `hardware.backend_name` if set.

## Example Configurations

| File | Type | Algorithm | Description |
|------|------|-----------|-------------|
| `configs/tfim_vqe_single.yaml` | single_point | VQE | One `(J, h)` point |
| `configs/tfim_vqe.yaml` | sweep | VQE | `h` sweep with benchmark CSV and plots |
| `configs/tfim_vqd.yaml` | single_point | VQD | Multi-state VQD with per-state error and fidelity |
| `configs/tfim_hamiltonian_sim_single.yaml` | single_point | hamiltonian_sim | Single-point Trotter evolution |
| `configs/tfim_hamiltonian_sim_time_sweep.yaml` | sweep | hamiltonian_sim | Evolution-time sweep with CSV and plots |
| `configs/tfim_trotter_validation.yaml` | trotter_validation | hamiltonian_sim | Lie vs Strang Trotter validation plots |

See [docs/atlas_component_extension_guide.md](docs/atlas_component_extension_guide.md) for how to register new Hamiltonians, ansätze, algorithms, backends, and experiments.

## Variational Algorithms

### VQE

The Variational Quantum Eigensolver minimizes Hamiltonian expectation value over a parameterized ansatz using a classical optimizer. Atlas runs multi-start COBYLA optimization and benchmarks against exact diagonalization.

### VQD

Variational Quantum Deflation finds successive eigenstates by adding overlap penalties against previously found states. The ground state uses `VQE.run()`; excited states reuse `VQE._optimize_cost()` with a deflated cost function. Configure via:

```yaml
algorithm:
  name: vqd
  parameters:
    num_states: 2
    beta: 1.0
```
  
See [docs/vqe_tfim.md](docs/vqe_tfim.md) for VQE theory, TFIM validation methodology, and benchmark discussion.

## Roadmap

```
Current
✓ YAML-driven experiment configuration
✓ Variational Quantum Eigensolver (VQE)
✓ Variational Quantum Deflation (VQD)
✓ Modular n-qubit TFIM Hamiltonian and hardware-efficient ansatz
✓ IBM Quantum Runtime integration
✓ TFIM VQE/VQD benchmarking pipeline with per-run output folders

↓

Expanded System Sizes and Ansätze
(8 and 16 qubits; UCCSD and Hamiltonian Variational Ansatz)

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
| YAML-driven config and factory | Done |
| VQE | Done |
| VQD | Done |
| IBM Quantum Runtime | Done |
| TFIM benchmarking (VQE and VQD) | Done |
| n-qubit Hamiltonian and ansatz infrastructure | Done |
| Qubit count expansion (8, 16) | Planned |
| Alternative ansätze (UCCSD, HVA) | Planned |
| Quantum dynamics (Trotterization) | Planned |
| Advanced error mitigation (ZNE, PEC) | Planned |
| Additional Hamiltonians | Planned |

## Documentation

- [VQE for TFIM — theory, implementation, and validation](docs/vqe_tfim.md)
- [Component extension guide](docs/atlas_component_extension_guide.md)
