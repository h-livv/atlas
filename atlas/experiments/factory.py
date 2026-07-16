"""Experiment assembly and execution from ``AtlasConfig``."""

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
    """Resolve an IBM Quantum backend when hardware execution is enabled."""

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
    estimator = build_estimator(config)
    optimizer = build_optimizer(config)
    algorithm = build_algorithm(config, estimator=estimator, optimizer=optimizer)
    return TFIMExperiment(config=config, algorithm=algorithm)


def _build_dynamics_experiment(config: AtlasConfig) -> HamiltonianSimExperiment:
    evolver = build_evolver(config)
    algorithm = build_algorithm(config, evolver=evolver)
    return HamiltonianSimExperiment(config=config, algorithm=algorithm)


def build_experiment(
    config: AtlasConfig,
) -> Union[TFIMExperiment, HamiltonianSimExperiment]:
    """Wire configured components into the appropriate experiment workflow."""

    if is_variational_algorithm(config.algorithm.name):
        return _build_variational_experiment(config)
    if is_dynamics_algorithm(config.algorithm.name):
        return _build_dynamics_experiment(config)
    raise ValueError(f"Unknown algorithm '{config.algorithm.name}'.")


@dataclass
class ExperimentRunResult:
    """Container for a completed experiment run."""

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
    """Run an experiment described entirely by ``AtlasConfig``."""

    def __init__(
        self,
        config: AtlasConfig,
        experiment: Union[TFIMExperiment, HamiltonianSimExperiment],
    ):
        self.config = config
        self.experiment = experiment

    def run(self) -> ExperimentRunResult:
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
        if not self.config.hardware.enabled:
            return "sim", None

        num_qubits = int(self.config.system.parameters["num_qubits"])
        backend = resolve_hardware_backend(self.config, num_qubits)
        if backend is None:
            return "sim", None
        return "hardware", backend

    def _run_single_point(self):
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
        sweep = self.config.system.sweep
        if sweep is None:
            raise ValueError("Experiment type 'sweep' requires system.sweep in config.")

        parameter = sweep.get("parameter", "h")
        sweep_values = sweep["values"]
        J = float(self.config.system.parameters["J"])

        if is_dynamics_algorithm(self.config.algorithm.name):
            fixed_h = float(self.config.system.parameters["h"])
            fixed_evolution_time = resolve_evolution_time(self.config)
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
    """Build and run an experiment from configuration."""

    experiment = build_experiment(config)
    runner = ConfiguredExperimentRunner(config, experiment)
    return runner.run()
