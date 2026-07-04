# Atlas Component Extension Guide

This document describes the process for adding new components to the Atlas quantum simulation framework. VQE is the first implemented algorithm; the package is organized around framework-level concepts rather than VQE itself. This is a design guide, not an implementation file.

The core rule is that new components should plug into existing boundaries without forcing changes to unrelated layers. A new Hamiltonian should not require changes to the VQE optimizer loop. A new optimizer should not require changes to a physics model. A new plot should not run a quantum job.

## Current Package Layout

```text
atlas/
  main.py                 # CLI orchestration entry point
  config.py               # Experiment and execution configuration
  physics/                # Hamiltonians and observables
  circuits/ansatzes/      # Parameterized circuit families (variational ansatzes)
  algorithms/vqe.py       # VQE (first implemented algorithm)
  optimization/           # Classical optimizers used by variational algorithms
  execution/              # Simulator and hardware backends
  experiments/            # Experiment orchestration and result containers
  analysis/               # Pure post-processing metrics
  visualization/          # Passive plotting and circuit drawing
  io/                     # CSV loading and writing
```

Run the current TFIM workflow from the repository root:

```bash
python -m atlas.main --mode sim --J 1.0 --h 0.5
```

Default benchmark output directory: `atlas/data`.

## Extension Principles

- Keep components narrow: each component should own one concept, such as a Hamiltonian, circuit/ansatz, optimizer, estimator, observable set, experiment runner, or plot family.
- Prefer explicit data flow over hidden imports: pass Hamiltonians, circuits, observables, and configs into the components that need them.
- Preserve `num_qubits` as a first-class value: Hamiltonians, circuits, observables, result records, and hardware validation should agree on qubit count.
- Keep algorithm code model-agnostic: `VQE` should depend on a Hamiltonian interface, ansatz spec, estimator, and optimizer, not on TFIM-specific classes.
- Keep visualization passive: plotting functions consume result objects and never run optimization, simulation, or hardware execution.
- Add abstractions only when at least two real implementations need the same contract.

## Adding A New Hamiltonian

Use this when adding a new physical model, such as Heisenberg, XXZ, Hubbard after mapping, or another spin Hamiltonian.

Place the implementation in `atlas/physics/hamiltonians.py` or a focused file under `atlas/physics/` if the module becomes too large.

Required responsibilities:

- Store model parameters and `num_qubits`.
- Implement `operator() -> SparsePauliOp`.
- Use a deterministic Pauli-string ordering so tests and CSV output are reproducible.
- Support `exact_ground_state()` only when dense diagonalization is appropriate for the configured size.

Expected interface:

```python
class NewHamiltonian(Hamiltonian):
    def __init__(self, num_qubits: int, ...):
        ...

    def operator(self) -> SparsePauliOp:
        ...
```

Process:

1. Define the physical parameters and qubit-count rules.
2. Build the Pauli terms in `operator()`.
3. Add small-system tests for known Pauli terms and known energies when available.
4. Add an experiment config or experiment runner only if the Hamiltonian needs its own sweep, observables, or plots.
5. Do not modify `VQE` unless the new Hamiltonian exposes a real missing algorithm contract.

Review checklist:

- Does `operator()` return a valid `SparsePauliOp` for `num_qubits=1`, `2`, and a representative larger size when meaningful?
- Are coefficients and Pauli-string endian conventions consistent with the rest of the framework?
- Is exact diagonalization clearly treated as a small-system benchmark?

## Adding A New Ansatz

Use this when adding a new parameterized circuit family for a variational algorithm.

Place the implementation under `atlas/circuits/ansatzes/`. The `circuits/` namespace holds circuit construction; variational ansatzes live in the `ansatzes/` subpackage. Non-variational circuits (for example evolution circuits) would belong under `circuits/` when they are added.

Required responsibilities:

- Build a parameterized `QuantumCircuit`.
- Own parameter ordering.
- Report `num_qubits` and `num_parameters`.
- Provide a binding method that returns a bound circuit for a parameter vector.

Expected interface:

```python
spec = build_new_ansatz(num_qubits=..., ...)
spec.circuit
spec.parameters
spec.num_qubits
spec.num_parameters
spec.bind(values)
```

The existing `AnsatzSpec` dataclass in `atlas/circuits/ansatzes/hardware_efficient.py` is the reference pattern.

Process:

1. Decide whether the ansatz is generic or tied to a specific experiment.
2. Add a builder function or small class returning `AnsatzSpec`.
3. Make `num_qubits` explicit and validate unsupported sizes early.
4. Keep parameter naming and ordering deterministic.
5. Add tests that verify parameter count and the circuit structure for small sizes.
6. Update the relevant experiment config to select the new ansatz.

Review checklist:

- Does `ansatz.num_parameters` match the number of values required by `bind()`?
- Does `VQE` derive initial-point size from the ansatz rather than hardcoding it?
- Does hardware execution use the same ansatz object as simulator execution?

## Adding A New Algorithm

Use this when adding a new quantum algorithm (for example VQD, QAOA, or a dynamics method) once its workflow is understood.

Place the implementation under `atlas/algorithms/`. VQE currently lives at `atlas/algorithms/vqe.py` as the reference pattern: the algorithm module should not import TFIM-specific or plotting code.

Required responsibilities:

- Accept the inputs its workflow needs (Hamiltonian, circuit, estimator, optimizer, or other algorithm-specific dependencies).
- Return a structured result object defined in or alongside `atlas/experiments/results.py`.
- Keep physics, execution, and visualization outside the algorithm module.

Process:

1. Implement the algorithm with the smallest interface that matches the workflow.
2. Add a result dataclass if existing result types are not appropriate.
3. Wire the algorithm through an experiment module, not through `main.py` or plotting.
4. Do not modify unrelated algorithms unless a shared contract is genuinely missing.

Review checklist:

- Does the new algorithm avoid importing concrete experiment or TFIM classes?
- Can an experiment runner compose the algorithm without duplicating circuit or Hamiltonian construction?
- Are result fields stable enough for I/O and visualization to consume?

## Adding A New Optimizer

Use this when adding another classical optimizer, such as SPSA, Nelder-Mead, gradient-based methods, or a Qiskit optimizer wrapper.

Place the implementation under `atlas/optimization/`.

Required responsibilities:

- Accept a scalar cost function and an initial point.
- Return objective value, optimal parameters, function evaluations if available, and useful metadata.
- Avoid importing Hamiltonian, ansatz, estimator, or experiment modules.

Expected interface:

```python
optimizer.minimize(cost_fn, initial_point) -> OptimizerRunResult
```

The existing `ScipyOptimizer` in `atlas/optimization/scipy_optimizer.py` is the reference pattern.

Process:

1. Add the optimizer wrapper with the same result shape as existing optimizer wrappers.
2. Map optimizer-specific output fields into framework result fields.
3. Add config fields only for options that are actually needed.
4. Update experiment wiring to choose the optimizer from config.
5. Do not change Hamiltonian, ansatz, hardware, or plotting modules.

Review checklist:

- Can `VQE` call the optimizer through the same `minimize()` shape?
- Are optimizer-specific options isolated in config?
- Are missing metadata fields handled explicitly rather than guessed?

## Adding A New Estimator Or Backend

Use this when adding a new execution target, such as another simulator primitive, a shot-based simulator, or another hardware provider.

Place the implementation under `atlas/execution/`. Current modules:

- `atlas/execution/simulator.py` — local `StatevectorEstimator` wrapper
- `atlas/execution/ibm_runtime.py` — IBM Runtime `EstimatorV2` wrapper

Required responsibilities:

- Evaluate expectation values for variational cost evaluation, or evaluate final observables after optimization.
- Keep provider-specific setup, transpilation, layouts, options, and job metadata inside the execution module.
- Validate qubit count and backend capability before execution.

Expected simulator-style interface:

```python
estimator.expectation(circuit, observable, parameter_values) -> float
```

Expected hardware-style interface:

```python
executor.evaluate(ansatz, hamiltonian, parameter_values, observables) -> HardwareEvaluationResult
```

Process:

1. Decide whether the backend participates in iterative optimization or only final evaluation.
2. Implement the appropriate execution interface.
3. Keep backend authentication and provider-specific options in execution config.
4. Preserve observable names when returning measured values.
5. Add tests or dry-run checks around layout application and qubit-count validation when possible.

Review checklist:

- Does the backend code avoid rebuilding Hamiltonians, ansatzes, or observables internally?
- Are provider-specific fields stored as metadata rather than leaking into generic result objects?
- Does the execution path fail clearly when `num_qubits` exceeds backend capacity?

## Adding New Observables

Use this when adding quantities measured after a simulation or optimization run, such as correlations, magnetization, particle number, or custom diagnostics.

Place the implementation in `atlas/physics/observables.py` or an experiment-specific observable module.

Required responsibilities:

- Return named `SparsePauliOp` values via `ObservableSpec`.
- Make `num_qubits` explicit.
- Keep names stable because CSV and plots may use them.

Expected interface:

```python
observables = build_observables(num_qubits)  # list[ObservableSpec]
```

The existing `tfim_observables()` function is the reference pattern for experiment-specific observable sets.

Process:

1. Define the physical meaning and naming convention.
2. Generate Pauli strings from `num_qubits`.
3. Add the observables to the relevant experiment runner.
4. Update metrics, CSV, and plotting only if those outputs should expose the new observable.

Review checklist:

- Are observable names stable and descriptive?
- Are Pauli strings valid for all supported qubit counts?
- Are hardware and simulator paths consuming the same observable definitions?

## Adding A New Experiment

Use this when combining existing components into a new workflow, such as a new Hamiltonian sweep or a new benchmark.

Place the implementation under `atlas/experiments/`. The current TFIM workflow is at `atlas/experiments/tfim.py`; shared result containers live in `atlas/experiments/results.py`.

Required responsibilities:

- Own experiment-specific composition.
- Build the Hamiltonian, ansatz, observables, optimizer, and execution components from config.
- Assemble structured result objects.
- Delegate metrics, I/O, and visualization to their own modules.

Expected interface:

```python
experiment.run_single_point(...) -> PointResult
experiment.run_sweep(...) -> BenchmarkResult
```

Process:

1. Define the experiment config and sweep parameters (extend `atlas/config.py` if needed).
2. Reuse existing Hamiltonian, ansatz, optimizer, estimator, and metric components.
3. Create result containers if existing ones are not appropriate.
4. Keep `atlas/main.py` as a thin orchestration layer that selects and invokes the experiment.
5. Add plotting and CSV support only after the result shape is stable.

Review checklist:

- Does the experiment own only workflow composition?
- Can the same result be saved, plotted, or printed without rerunning computation?
- Does `main.py` remain free of circuit, optimizer, and backend internals?

## Adding Visualization

Use this when adding plots, circuit drawings, convergence views, or benchmark summaries.

Place the implementation under `atlas/visualization/`. TFIM benchmark plots live in `atlas/visualization/tfim_plots.py`; circuit drawing is in `atlas/visualization/circuit_drawer.py`.

Required responsibilities:

- Consume structured results.
- Write visual artifacts to an output path.
- Avoid running algorithms, exact diagonalization, hardware jobs, or CSV loading.

Expected interface:

```python
plot_new_view(result, output_dir)
```

Process:

1. Confirm the required values already exist in result objects.
2. If values are missing, add them to analysis or experiment output first.
3. Implement the plot as a passive function.
4. Add the plot to a `plot_all_*` helper only if it belongs to the default output set.

Review checklist:

- Does the plot function have no side effects other than writing the plot?
- Does it avoid reconstructing Hamiltonians, ansatzes, or statevectors?
- Are plot labels tied to the experiment rather than a generic algorithm?

## Adding CSV Or Persistence Support

Use this when saving or loading benchmark data, hardware results, or experiment summaries.

Place the implementation under `atlas/io/`. The current TFIM benchmark writer is `write_tfim_benchmark()` in `atlas/io/csv_io.py`.

Required responsibilities:

- Convert result objects to rows.
- Load existing result formats into result objects or compatible records.
- Keep schema changes deliberate and documented.

Expected interface:

```python
write_result(result, path)
load_result(path)
```

Process:

1. Define the schema and stable column names.
2. Keep units and naming consistent with result fields.
3. Preserve legacy columns when reproducing legacy benchmark output.
4. Add migration helpers only when old data must remain readable.

Review checklist:

- Is CSV I/O separate from metrics and plotting?
- Are optional hardware fields handled cleanly when absent?
- Can loaded hardware results be merged into benchmark results without rerunning hardware jobs?

## Adding Tests For New Components

Minimum useful tests:

- Hamiltonians: Pauli term construction, coefficient signs, exact energy for small systems.
- Ansatzes: qubit count, parameter count, deterministic parameter order, bind behavior.
- Optimizers: interface compatibility with a simple quadratic cost function.
- VQE: smoke test on a tiny Hamiltonian with deterministic seed or fixed initial point.
- Observables: Pauli-string construction and names.
- Metrics: fidelity, expectation value, absolute error, relative error.
- I/O: CSV columns and round-trip behavior.

## Common Anti-Patterns

- Importing a concrete TFIM class inside `VQE` or another algorithm module.
- Rebuilding the ansatz separately in simulator, hardware, and analysis code.
- Creating observables inside hardware execution.
- Letting plotting functions compute experiment results.
- Adding a broad plugin system before multiple real implementations need it.
- Treating exact diagonalization as scalable for large `num_qubits`.
- Hardcoding parameter count or qubit count outside legacy-compatibility defaults.
- Placing framework-level code under algorithm-specific paths (for example putting shared physics under `algorithms/`).

## Recommended Addition Workflow

1. Identify which component type is being added.
2. Add the smallest interface-compatible implementation in the correct module under `atlas/`.
3. Add or update config in `atlas/config.py` only for values users must choose.
4. Wire the component through an experiment module, not through an algorithm or plotting code.
5. Add focused tests for the component boundary.
6. Update documentation with the component's purpose, interface, dependencies, and limitations.
