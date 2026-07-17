# Atlas Component Extension Guide

This document describes how to add new components to the Atlas quantum simulation framework. Atlas is YAML-driven: users describe experiments in config files, and the factory/builder layer assembles concrete Python components. This is a design guide, not an implementation file.

The core rule is that new components should plug into existing boundaries without forcing changes to unrelated layers. A new Hamiltonian should not require changes to the VQE optimizer loop. A new optimizer should not require changes to a physics model. A new plot should not run a quantum job.

## Current Package Layout

```text
atlas/
  main.py                      # CLI: parses --config, loads YAML, runs experiment, saves outputs
  config.py                    # AtlasConfig and legacy dataclasses
  io/
    yaml_config.py             # load_config(path) -> AtlasConfig
    csv_io.py                  # TFIM / sim benchmark CSV I/O
  experiments/
    builders.py                # build_hamiltonian, build_ansatz, build_algorithm, …
    factory.py                 # build_experiment, run_experiment, hardware backend resolution
    tfim.py                    # TFIMExperiment (VQE/VQD workflows)
    hamiltonian_sim_experiment.py  # HamiltonianSimExperiment (dynamics workflows)
    outputs.py                 # save_outputs, unique run directories, plot/CSV dispatch
    results.py                 # VQEResult, SimPointResult, SimBenchmarkResult, …
  physics/                     # Hamiltonians and observables
  circuits/
    ansatzes/                  # Parameterized circuit families
    initial_states.py          # Initial states for dynamics
    evolution/                 # Product-formula circuit builders
  algorithms/
    vqe.py                     # VQE with shared _optimize_cost helper
    vqd.py                     # VQD composed on VQE
    hamiltonian_sim.py         # Product-formula time evolution
  optimization/                # Classical optimizers
  execution/
    estimator.py               # Variational cost evaluation
    evolver.py                 # Dynamics circuit execution
    simulator.py               # Backward-compatible re-exports
    ibm_runtime.py             # IBM Runtime hardware evaluation
  analysis/                    # Pure post-processing metrics
  visualization/
    tfim_plots.py              # VQE benchmark plots
    tfim_vqd_plots.py          # VQD plots
    sim_plots.py               # Hamiltonian simulation scalar plots
    sim_validation/            # Trotter validation plot registry
    hamiltonian_sim/           # Physics-agnostic lattice dashboard
      lattice.py              # LatticeGeometry / Frame / Trajectory
      layouts.py               # chain_1d, grid_2d (display geometry only)
      colors.py                # Diverging colormaps
      renderer.py              # LatticeGraphRenderer (geometry-agnostic)
      adapters.py              # SimBenchmarkResult → LatticeTrajectory
      dashboard.py             # LatticeDashboard (slider + play/pause)
      views/                   # Chain1DView, GraphLatticeView, …
  configs/                     # Example YAML experiment files (repository root)
scripts/
  run_lattice_dashboard.py     # Interactive lattice dashboard entry point
```

Run an experiment from the repository root:

```bash
python -m atlas.main --config configs/tfim_vqe_single.yaml
```

Open an interactive lattice dashboard (dynamics + site observables):

```bash
python scripts/run_lattice_dashboard.py
python scripts/run_lattice_dashboard.py --config configs/tfim_hamiltonian_sim_time_sweep.yaml --method strang
```

Each batch run writes to a unique folder under the configured output base:

```text
atlas/data/{experiment_name}_{YYYYMMDD_HHMMSS}/
```

## Runtime Flow

```text
YAML file
  → atlas.io.yaml_config.load_config()
  → atlas.experiments.factory.run_experiment()
      → atlas.experiments.builders (…)
      → TFIMExperiment  or  HamiltonianSimExperiment
  → atlas.experiments.outputs.save_outputs()
      → CSV, static plots, lattice dashboard PNG snapshots, console summary
```

`main.py` must remain a thin orchestrator. Do not add Hamiltonian, algorithm, or plotting logic there.

## YAML Configuration

Experiments are described by YAML files with these sections:

| Section | Purpose |
|---------|---------|
| `experiment` | `type` (`single_point`, `sweep`, or `trotter_validation`), `name` |
| `system` | Hamiltonian name, parameters, optional sweep |
| `algorithm` | `vqe`, `vqd`, or `hamiltonian_sim`, plus algorithm parameters |
| `ansatz` | Circuit family and parameters (**required for variational algorithms**) |
| `optimizer` | Optimizer name and parameters (**required for variational algorithms**) |
| `initial_state` | Initial state for `hamiltonian_sim` (optional; defaults to `\|0...0>`) |
| `backend` | `statevector` (VQE/VQD) or `statevector_evolver` (dynamics) |
| `hardware` | IBM Runtime settings (optional) |
| `analysis` | Observables preset, optional `site_observables`, fidelity toggle |
| `output` | Base directory, CSV/plot toggles |

`analysis.site_observables` may be `local_z`, `local_x`, or `local_y`. When set, dynamics experiments store per-site arrays on `SimPointResult.site_observables` / `exact_site_observables` (keys `z`, `x`, or `y`) for the lattice dashboard. Scalar observables (`tfim_default`: `zz`, `x`) remain unchanged.

Example (VQE single point):

```yaml
experiment:
  type: single_point
  name: tfim_vqe_single

system:
  name: tfim
  parameters:
    num_qubits: 2
    J: 1.0
    h: 0.5

algorithm:
  name: vqe

ansatz:
  name: hardware_efficient
  parameters:
    reps: 1

optimizer:
  name: scipy
  parameters:
    method: COBYLA
    maxiter: 200
    num_starts: 10

backend:
  name: statevector

output:
  directory: atlas/data
  csv: false
  plots: true
```

To add a new selectable component name (system, algorithm, ansatz, etc.):

1. Implement the component (see sections below).
2. Register it in `atlas/experiments/builders.py`.
3. Add the name to the supported set in `atlas/io/yaml_config.py`.
4. Add an example config under `configs/`.

## Extension Principles

- Keep components narrow: each component should own one concept.
- Prefer explicit data flow over hidden imports.
- Preserve `num_qubits` as a first-class value across Hamiltonians, ansatzes, observables, and results.
- Keep algorithm code model-agnostic: `VQE` depends on Hamiltonian interface, ansatz spec, estimator, and optimizer — not TFIM classes.
- Keep visualization passive: plotting functions consume result objects only.
- Register new YAML-selectable components in `builders.py` and `yaml_config.py`.
- Add abstractions only when at least two real implementations need the same contract.

## Adding A New Hamiltonian

Place the implementation in `atlas/physics/hamiltonians.py` or a focused file under `atlas/physics/`.

Required responsibilities:

- Store model parameters and `num_qubits`.
- Implement `operator() -> SparsePauliOp`.
- Use deterministic Pauli-string ordering.
- Implement `exact_ground_state()` and, when needed for multi-state benchmarks, `exact_spectrum(num_states)`.

Register in `build_hamiltonian()` in `atlas/experiments/builders.py`:

```python
if config.system.name == "new_model":
    return NewHamiltonian(**params)
```

Also add `"new_model"` to `_SUPPORTED["system"]` in `atlas/io/yaml_config.py`.

Review checklist:

- Does `operator()` return a valid `SparsePauliOp` for supported qubit counts?
- Are coefficients and endian conventions consistent?
- Is exact diagonalization clearly limited to small systems?

## Adding A New Ansatz

Place the implementation under `atlas/circuits/ansatzes/`.

Required responsibilities:

- Build a parameterized `QuantumCircuit`.
- Own parameter ordering.
- Report `num_qubits` and `num_parameters`.
- Provide `bind(values)` returning a bound circuit.

The existing `AnsatzSpec` dataclass in `atlas/circuits/ansatzes/hardware_efficient.py` is the reference pattern.

Register in `build_ansatz()` in `atlas/experiments/builders.py` and in `yaml_config.py`.

Review checklist:

- Does `ansatz.num_parameters` match `bind()`?
- Does `VQE` derive initial-point size from the ansatz?
- Does hardware execution use the same ansatz object as simulation?

## Adding A New Algorithm

Place the implementation under `atlas/algorithms/`.

Required responsibilities:

- Accept Hamiltonian, ansatz, estimator, optimizer (or wrap an existing algorithm).
- Return a structured result object from `atlas/experiments/results.py`.
- Avoid importing TFIM, plotting, or `main.py` code.

For algorithms that share VQE's multi-start loop, call `VQE._optimize_cost()` rather than duplicating the optimization loop. VQD is the reference pattern: ground state via `VQE.run()`, excited states via deflated cost functions and `_optimize_cost()`.

Register in `build_algorithm()` in `atlas/experiments/builders.py` and wire through `TFIMExperiment` or a new experiment module if the workflow differs.

Review checklist:

- Does the algorithm avoid importing concrete experiment classes?
- Can an experiment runner compose it without duplicating construction?
- Are result fields stable for I/O and visualization?

## Adding A New Optimizer

Place the implementation under `atlas/optimization/`.

Expected interface:

```python
optimizer.minimize(cost_fn, initial_point) -> OptimizerRunResult
```

Register in `build_optimizer()` in `atlas/experiments/builders.py`. Only pass optimizer-specific kwargs to the constructor (for example, filter out `num_starts`, which belongs to VQE config).

## Adding A New Estimator Or Backend

Place the implementation under `atlas/execution/`.

Current modules:

- `atlas/execution/simulator.py` — local statevector estimator
- `atlas/execution/ibm_runtime.py` — IBM Runtime evaluation after optimization

Register in `build_estimator()` in `atlas/experiments/builders.py`.

For hardware execution during an experiment, backend resolution lives in `atlas/experiments/factory.py` (`resolve_hardware_backend`).

## Adding New Observables

Place the implementation in `atlas/physics/observables.py` or an experiment-specific module.

Register in `build_observables()` in `atlas/experiments/builders.py`. The existing `tfim_observables()` function is the reference pattern.

## Adding A New Experiment

Place the implementation under `atlas/experiments/`. The current TFIM workflow is `atlas/experiments/tfim.py`.

Required responsibilities:

- Accept `AtlasConfig` and a pre-built algorithm (constructed by the factory).
- Build per-point Hamiltonian, ansatz, and observables via builders.
- Return structured result objects from `results.py`.
- Delegate CSV, plots, and console output to `outputs.py`.

To add a new experiment type:

1. Implement the experiment class.
2. Dispatch from `build_experiment()` / `ConfiguredExperimentRunner` in `factory.py`.
3. Add output handling in `save_outputs()` in `outputs.py`.
4. Add a YAML `experiment.type` value and validation in `yaml_config.py` if needed.

Review checklist:

- Does the experiment own only workflow composition?
- Can results be saved and plotted without rerunning computation?
- Does `main.py` remain free of internals?

## Adding Visualization

Place passive plotting functions under `atlas/visualization/`.

- VQE benchmark plots: `atlas/visualization/tfim_plots.py`
- VQD plots: `atlas/visualization/tfim_vqd_plots.py`
- Dynamics scalar plots: `atlas/visualization/sim_plots.py`
- Trotter validation registry: `atlas/visualization/sim_validation/`
- Lattice dashboard (physics-agnostic): `atlas/visualization/hamiltonian_sim/`

### Lattice dashboard

The lattice package is a **renderer only**. It consumes `LatticeTrajectory`
(geometry + per-site values over time) and never computes observables or
inspects Hamiltonians / evolution methods.

| Piece | Role |
|-------|------|
| `LatticeGraphRenderer` | Geometry-agnostic nodes + independent edges |
| `GraphLatticeView` | Dashboard panel for arbitrary graph layouts |
| `Chain1DView` | Chain-tuned defaults; layout still comes from upstream geometry |
| `LatticeDashboard` | Metadata panel, time slider, play/pause, window title |
| Adapters | `SimBenchmarkResult` / `SimPointResult` → `LatticeTrajectory` |

Site arrays are produced upstream via `analysis.site_observables`
(`local_z` / `local_x` / `local_y`) and stored on
`SimPointResult.site_observables` / `exact_site_observables`.

When `output.plots: true` and site arrays are present, `outputs.py` writes
PNG snapshots named `lattice_dashboard_{obs}_{sim|exact}_{method}.png`.
Interactive use is via `scripts/run_lattice_dashboard.py` (Matplotlib GUI
backend; not GIF/MP4).

Wire new plots through `atlas/experiments/outputs.py` when they belong in the
default output set for a result type.

Review checklist:

- No side effects other than writing plot files / showing a GUI window?
- No reconstruction of Hamiltonians or re-running of algorithms?
- Lattice renderers remain free of `atlas.physics` imports?
- New lattice geometries added as layouts/views, not by hard-coding physics?

## Adding CSV Or Persistence Support

Place writers under `atlas/io/` or add dispatch logic in `outputs.py`.

- VQE benchmark: `write_tfim_benchmark()` in `atlas/io/csv_io.py`
- VQD benchmark: `_write_vqd_benchmark_csv()` in `atlas/experiments/outputs.py`

Each run's CSV and plots go into that run's unique output directory.

## Adding Tests For New Components

Minimum useful tests:

- Hamiltonians: Pauli terms, coefficients, exact energies for small systems.
- Ansatzes: qubit count, parameter count, bind behavior.
- Optimizers: interface compatibility with a simple cost function.
- VQE/VQD: smoke test on a tiny Hamiltonian.
- YAML loading: valid configs parse; unknown component names fail clearly.
- I/O: CSV columns match result fields.

## Common Anti-Patterns

- Importing TFIM classes inside `VQE` or `VQD`.
- Hardcoding component choices in `main.py`.
- Rebuilding the ansatz separately in simulator, hardware, and analysis code.
- Letting plotting functions compute experiment results.
- Adding a plugin registry before multiple implementations need it.
- Duplicating VQE's multi-start loop instead of calling `_optimize_cost()`.

## Recommended Addition Workflow

1. Identify which component type is being added.
2. Implement the smallest interface-compatible module under `atlas/`.
3. Register the component in `builders.py` and `yaml_config.py`.
4. Wire through an experiment module and `outputs.py` if new result types or plots are needed.
5. Add an example config under `configs/`.
6. Add focused tests for the component boundary.
7. Update documentation with purpose, interface, and limitations.
