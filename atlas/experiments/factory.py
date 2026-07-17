"""Experiment assembly and execution from ``AtlasConfig``.

This module is the top-level orchestration layer: given a validated
``AtlasConfig``, it builds the appropriate experiment object (TFIM variational
or Hamiltonian simulation), runs the requested experiment type
(``single_point``, ``sweep``, or ``trotter_validation``), and returns a
structured ``ExperimentRunResult``. Optional IBM hardware backends are
resolved here when ``config.hardware.enabled`` is set.

Architectural role:
    ``run_experiment`` is the usual entry point from ``atlas.main`` and CLI
    scripts. It depends on builders and experiment workflows; it does not
    implement physics or plotting itself (outputs are handled separately).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from atlas.config import AtlasConfig, is_dynamics_algorithm, is_variational_algorithm
from atlas.experiments.builders import (
    build_algorithm,
    build_estimator,
    build_evolver,
    build_optimizer,
    resolve_evolution_methods,
    resolve_evolution_time,
    resolve_trotter_step_values,
)
from atlas.experiments.hamiltonian_sim_experiment import HamiltonianSimExperiment
from atlas.experiments.results import (
    SimBenchmarkResult,
    SimPointResult,
    SimValidationResult,
    TFIMBenchmarkResult,
    TFIMPointResult,
    TFIMVQDBenchmarkResult,
    TFIMVQDPointResult,
)
from atlas.experiments.tfim import TFIMExperiment


def resolve_hardware_backend(config: AtlasConfig, num_qubits: int):
    """Resolve an IBM Quantum backend when hardware execution is enabled.

    Purpose:
        Obtain a live Qiskit Runtime backend for hardware evaluation, or
        fail soft so the experiment can continue in simulation.

    Inputs:
        config: Uses ``config.hardware.backend_name`` when set.
        num_qubits: Minimum qubit requirement for ``least_busy`` selection.

    Process:
        Try to construct ``QiskitRuntimeService``. On success, either fetch
        the named backend or the least-busy operational device with enough
        qubits. On any failure, print guidance and return ``None``.

    Outputs:
        A backend object, or ``None`` if credentials/backends are unavailable.

    Side effects:
        May print error/help messages to stdout. May contact IBM Quantum
        services when credentials are present.
    """

    try:
        from qiskit_ibm_runtime import QiskitRuntimeService

        service = QiskitRuntimeService()
    except Exception as exc:
        print(
            "Could not reach the IBM Quantum Runtime service: no account/credentials "
            "are configured in this environment.\n"
            "To fix this, save an account first, e.g.:\n"
            '    QiskitRuntimeService.save_account(channel="ibm_quantum", token="<your token>")\n'
            f"(Underlying error: {exc})"
        )
        return None

    try:
        if config.hardware.backend_name:
            return service.backend(config.hardware.backend_name)
        return service.least_busy(
            operational=True, simulator=False, min_num_qubits=num_qubits
        )
    except Exception as exc:
        print(
            f"Could not find an operational IBM Quantum backend with at least "
            f"{num_qubits} qubits.\n(Underlying error: {exc})"
        )
        return None


def _build_variational_experiment(config: AtlasConfig) -> TFIMExperiment:
    """Assemble estimator, optimizer, algorithm, and ``TFIMExperiment``.

    Purpose:
        Private helper for variational (VQE/VQD) wiring from config.

    Inputs:
        config: Full Atlas configuration for a variational algorithm.

    Process:
        Build estimator and optimizer, construct the algorithm with both,
        then wrap in ``TFIMExperiment``.

    Outputs:
        A ready-to-run ``TFIMExperiment``.

    Side effects:
        None (construction only).
    """

    estimator = build_estimator(config)
    optimizer = build_optimizer(config)
    algorithm = build_algorithm(config, estimator=estimator, optimizer=optimizer)
    return TFIMExperiment(config=config, algorithm=algorithm)


def _build_dynamics_experiment(config: AtlasConfig) -> HamiltonianSimExperiment:
    """Assemble evolver, dynamics algorithm, and ``HamiltonianSimExperiment``.

    Purpose:
        Private helper for Hamiltonian-simulation wiring from config.

    Inputs:
        config: Full Atlas configuration for a dynamics algorithm.

    Process:
        Build evolver, construct ``HamiltonianSimulation`` via
        ``build_algorithm``, then wrap in ``HamiltonianSimExperiment``.

    Outputs:
        A ready-to-run ``HamiltonianSimExperiment``.

    Side effects:
        None (construction only).
    """

    evolver = build_evolver(config)
    algorithm = build_algorithm(config, evolver=evolver)
    return HamiltonianSimExperiment(config=config, algorithm=algorithm)


def build_experiment(
    config: AtlasConfig,
) -> Union[TFIMExperiment, HamiltonianSimExperiment]:
    """Wire configured components into the appropriate experiment workflow.

    Purpose:
        Choose TFIM (variational) vs Hamiltonian-sim (dynamics) based on
        algorithm family.

    Inputs:
        config: Validated ``AtlasConfig``.

    Process:
        Dispatch on ``is_variational_algorithm`` / ``is_dynamics_algorithm``.

    Outputs:
        ``TFIMExperiment`` or ``HamiltonianSimExperiment``.

    Side effects:
        None. Raises ``ValueError`` for unknown algorithm names.
    """

    if is_variational_algorithm(config.algorithm.name):
        return _build_variational_experiment(config)
    if is_dynamics_algorithm(config.algorithm.name):
        return _build_dynamics_experiment(config)
    raise ValueError(f"Unknown algorithm '{config.algorithm.name}'.")


@dataclass
class ExperimentRunResult:
    """Container for a completed experiment run.

    Responsibility:
        Bundle the config that produced a run with the typed result object
        and lightweight metadata for output routing and logging.

    State:
        config: The ``AtlasConfig`` used for this run.
        result: One of the TFIM or simulation point/benchmark/validation
            result types.
        experiment_type: e.g. ``single_point``, ``sweep``,
            ``trotter_validation``.
        algorithm_name: e.g. ``vqe``, ``vqd``, ``hamiltonian_sim``.

    Usage:
        Returned by ``ConfiguredExperimentRunner.run`` / ``run_experiment``;
        consumed by ``atlas.experiments.outputs.save_outputs``.
    """

    config: AtlasConfig
    result: Union[
        TFIMPointResult,
        TFIMBenchmarkResult,
        TFIMVQDPointResult,
        TFIMVQDBenchmarkResult,
        SimPointResult,
        SimBenchmarkResult,
        SimValidationResult,
    ]
    experiment_type: str
    algorithm_name: str


class ConfiguredExperimentRunner:
    """Run an experiment described entirely by ``AtlasConfig``.

    Responsibility:
        Interpret ``config.experiment.type`` and call the matching methods on
        the injected experiment object, including hardware mode resolution
        for variational runs.

    State:
        config: Source of truth for parameters, sweep, and hardware flags.
        experiment: Pre-built ``TFIMExperiment`` or
            ``HamiltonianSimExperiment``.

    Usage:
        Prefer ``run_experiment(config)``, which builds the experiment and
        runner for you. Call ``run()`` to execute and obtain
        ``ExperimentRunResult``.
    """

    def __init__(
        self,
        config: AtlasConfig,
        experiment: Union[TFIMExperiment, HamiltonianSimExperiment],
    ):
        """Store config and the already-wired experiment workflow.

        Inputs:
            config: Experiment configuration.
            experiment: Built experiment object matching the algorithm family.
        """

        self.config = config
        self.experiment = experiment

    def run(self) -> ExperimentRunResult:
        """Execute the configured experiment type and wrap the result.

        Purpose:
            Dispatch to single-point, sweep, or Trotter validation runners.

        Inputs:
            None (uses ``self.config`` / ``self.experiment``).

        Process:
            Branch on ``config.experiment.type``, run the corresponding
            private method, then package config + result + metadata.

        Outputs:
            ``ExperimentRunResult``.

        Side effects:
            Runs algorithms (and optionally hardware jobs); may print via
            hardware resolution helpers. No filesystem output here.
        """

        if self.config.experiment.type == "single_point":
            result = self._run_single_point()
        elif self.config.experiment.type == "sweep":
            result = self._run_sweep()
        elif self.config.experiment.type == "trotter_validation":
            result = self._run_trotter_validation()
        else:
            raise ValueError(f"Unknown experiment type '{self.config.experiment.type}'.")

        return ExperimentRunResult(
            config=self.config,
            result=result,
            experiment_type=self.config.experiment.type,
            algorithm_name=self.config.algorithm.name,
        )

    def _execution_mode_and_backend(self):
        """Decide simulation vs hardware mode for variational experiments.

        Purpose:
            Centralize hardware enablement and soft-fail fallback to sim.

        Inputs:
            None (reads ``self.config.hardware`` and system qubit count).

        Process:
            If hardware is disabled, return ``("sim", None)``. Otherwise
            resolve a backend; if resolution fails, fall back to simulation.

        Outputs:
            Tuple ``(mode, backend)`` where mode is ``"sim"`` or
            ``"hardware"`` and backend may be ``None``.

        Side effects:
            May contact IBM Quantum and print messages via
            ``resolve_hardware_backend``.
        """

        if not self.config.hardware.enabled:
            return "sim", None

        num_qubits = int(self.config.system.parameters["num_qubits"])
        backend = resolve_hardware_backend(self.config, num_qubits)
        if backend is None:
            # Soft-fail: keep the run useful even without credentials/backends.
            return "sim", None
        return "hardware", backend

    def _run_single_point(self):
        """Run one parameter point for the configured algorithm family.

        Purpose:
            Map config system parameters onto the experiment's single-point
            API (VQE, VQD, or dynamics).

        Inputs:
            None (reads ``J``, ``h``, and algorithm name from config).

        Process:
            For dynamics, call ``run_single_point(J, h)`` (evolution time
            resolved inside the experiment). For variational, resolve
            sim/hardware mode and call VQE or VQD single-point methods.

        Outputs:
            A point-result dataclass (TFIM or Sim).

        Side effects:
            Executes algorithms; may submit hardware jobs.
        """

        params = self.config.system.parameters
        J = float(params["J"])

        if is_dynamics_algorithm(self.config.algorithm.name):
            h = float(params["h"])
            return self.experiment.run_single_point(J=J, h=h)

        h = float(params["h"])
        mode, backend = self._execution_mode_and_backend()

        if self.config.algorithm.name == "vqd":
            return self.experiment.run_single_point_vqd(J, h, mode=mode, backend=backend)
        return self.experiment.run_single_point(J, h, mode=mode, backend=backend)

    def _run_sweep(self):
        """Run a parameter sweep for the configured algorithm family.

        Purpose:
            Drive ``system.sweep`` values through the experiment's sweep APIs.

        Inputs:
            None (requires ``config.system.sweep`` with ``parameter``/``values``).

        Process:
            For dynamics, pass fixed companions (``h`` / ``evolution_time``)
            depending on which parameter is swept. For variational, only
            ``h`` sweeps are supported; resolve hardware inclusion and call
            VQE or VQD sweep methods.

        Outputs:
            A benchmark/result collection for the sweep.

        Side effects:
            Executes many algorithm runs; may use hardware or CSV merge paths.
            Raises ``ValueError`` if sweep config is missing or unsupported.
        """

        sweep = self.config.system.sweep
        if sweep is None:
            raise ValueError("Experiment type 'sweep' requires system.sweep in config.")

        parameter = sweep.get("parameter", "h")
        sweep_values = sweep["values"]
        J = float(self.config.system.parameters["J"])

        if is_dynamics_algorithm(self.config.algorithm.name):
            fixed_h = float(self.config.system.parameters["h"])
            fixed_evolution_time = resolve_evolution_time(self.config)
            # Only pass fixed companions that are *not* the swept axis.
            return self.experiment.run_sweep(
                J=J,
                sweep_values=sweep_values,
                sweep_parameter=parameter,
                fixed_h=fixed_h if parameter in ("evolution_time", "num_trotter_steps") else None,
                fixed_evolution_time=(
                    fixed_evolution_time
                    if parameter in ("h", "num_trotter_steps")
                    else None
                ),
            )

        if parameter != "h":
            raise ValueError(
                f"Sweep parameter '{parameter}' is not supported for variational experiments."
            )

        mode, backend = self._execution_mode_and_backend()
        hardware_source = self.config.hardware.hardware_csv_path
        include_hardware = self.config.hardware.enabled and backend is not None

        if self.config.algorithm.name == "vqd":
            return self.experiment.run_sweep_vqd(
                J,
                sweep_values,
                include_hardware=include_hardware,
                hardware_source=hardware_source,
                backend=backend,
            )

        return self.experiment.run_sweep(
            J,
            sweep_values,
            include_hardware=include_hardware,
            hardware_source=hardware_source,
            backend=backend,
        )

    def _run_trotter_validation(self):
        """Compare Trotter methods/steps at fixed physics parameters.

        Purpose:
            Run dynamics-only validation that sweeps step counts (and
            optionally methods) at fixed ``J``, ``h``, and evolution time.

        Inputs:
            None (reads system parameters and algorithm evolution settings).

        Process:
            Require a dynamics algorithm; resolve evolution time, method
            names, and step values; call ``run_trotter_validation``.

        Outputs:
            ``SimValidationResult``.

        Side effects:
            Executes many simulation runs. Raises ``ValueError`` if the
            algorithm is not dynamics.
        """

        if not is_dynamics_algorithm(self.config.algorithm.name):
            raise ValueError(
                "Experiment type 'trotter_validation' requires a dynamics algorithm."
            )

        params = self.config.system.parameters
        J = float(params["J"])
        h = float(params["h"])
        evolution_time = resolve_evolution_time(self.config)
        method_names = resolve_evolution_methods(self.config)
        step_values = resolve_trotter_step_values(self.config)

        return self.experiment.run_trotter_validation(
            J=J,
            h=h,
            evolution_time=evolution_time,
            method_names=method_names,
            step_values=step_values,
        )


def run_experiment(config: AtlasConfig) -> ExperimentRunResult:
    """Build and run an experiment from configuration.

    Purpose:
        One-call entry point: assemble components, execute, return results.

    Inputs:
        config: Validated ``AtlasConfig``.

    Process:
        ``build_experiment`` → ``ConfiguredExperimentRunner`` → ``run()``.

    Outputs:
        ``ExperimentRunResult`` ready for ``save_outputs``.

    Side effects:
        Same as ``ConfiguredExperimentRunner.run`` (compute / optional hardware).
    """

    experiment = build_experiment(config)
    runner = ConfiguredExperimentRunner(config, experiment)
    return runner.run()
