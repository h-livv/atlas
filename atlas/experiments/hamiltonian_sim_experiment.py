"""Hamiltonian simulation experiment orchestration.

Composes Hamiltonians, initial states, and observables from configuration,
runs the configured dynamics algorithm, and computes fidelity and
observable metrics against exact time evolution.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from qiskit.quantum_info import Statevector

from atlas.algorithms.hamiltonian_sim import HamiltonianSimulation
from atlas.analysis.metrics import (
    absolute_error,
    expectation_values,
    site_expectation_arrays,
    state_fidelity_to_exact,
)
from atlas.config import AtlasConfig
from atlas.experiments.builders import (
    build_evolution_method_named,
    build_hamiltonian,
    build_initial_state,
    build_observables,
    build_site_observables,
    resolve_evolution_time,
)
from atlas.experiments.results import (
    SimBenchmarkResult,
    SimPointResult,
    SimValidationResult,
)
from atlas import profiling as profile

# Parameters that control the dynamics run but are not Hamiltonian constructor args.
_NON_HAMILTONIAN_SYSTEM_KEYS = frozenset({"evolution_time"})


class HamiltonianSimExperiment:
    """Dynamics experiment workflow driven entirely by ``AtlasConfig``."""

    def __init__(self, config: AtlasConfig, algorithm: HamiltonianSimulation):
        self.config = config
        self.algorithm = algorithm
        self.num_qubits = int(config.system.parameters["num_qubits"])
        self.compute_fidelity = config.analysis.fidelity

    def _resolved_system_parameters(
        self, parameter_overrides: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        params = dict(self.config.system.parameters)
        if parameter_overrides:
            params.update(parameter_overrides)
        return params

    def _resolve_num_trotter_steps(self, override: Optional[int] = None) -> int:
        if override is not None:
            return int(override)
        raw = self.config.algorithm.parameters.get("num_trotter_steps", 10)
        if isinstance(raw, list):
            return int(raw[0])
        return int(raw)

    def _build_algorithm(
        self,
        method_name: str,
        num_trotter_steps: Optional[int] = None,
    ) -> HamiltonianSimulation:
        steps = self._resolve_num_trotter_steps(num_trotter_steps)
        with profile.span("build.evolution_method", method=method_name, steps=steps):
            evolution_method = build_evolution_method_named(method_name, steps)
        return HamiltonianSimulation(
            evolution_method=evolution_method,
            evolver=self.algorithm.evolver,
        )

    def _resolve_evolution_time(
        self,
        system_parameters: dict[str, Any],
        evolution_time: Optional[float] = None,
    ) -> float:
        if evolution_time is not None:
            return float(evolution_time)
        if "evolution_time" in system_parameters:
            return float(system_parameters["evolution_time"])
        return resolve_evolution_time(self.config)

    def _build_point_context(self, system_parameters: dict[str, Any]):
        hamiltonian_overrides = {
            key: value
            for key, value in system_parameters.items()
            if key not in _NON_HAMILTONIAN_SYSTEM_KEYS
        }
        with profile.span("build.hamiltonian"):
            hamiltonian = build_hamiltonian(
                self.config, parameter_overrides=hamiltonian_overrides
            )
        with profile.span("build.initial_state"):
            initial_state = build_initial_state(self.config, self.num_qubits)
        with profile.span("build.observables"):
            observables = build_observables(self.config, self.num_qubits)
        with profile.span("build.site_observables"):
            site_observables = build_site_observables(self.config, self.num_qubits)
        return hamiltonian, initial_state, observables, site_observables

    def _evaluate_point(
        self,
        system_parameters: dict[str, Any],
        evolution_time: float,
        algorithm: HamiltonianSimulation,
    ) -> SimPointResult:
        meta = {"evolution_time": float(evolution_time)}
        with profile.span("point.context_build", **meta):
            hamiltonian, initial_state, observables, site_obs_groups = (
                self._build_point_context(system_parameters)
            )
        with profile.span("point.initial_statevector", **meta):
            exact_initial = initial_state.statevector()
        with profile.span("point.exact_evolution", **meta):
            exact_result = hamiltonian.exact_time_evolution(exact_initial, evolution_time)
        with profile.span("point.simulated_evolution", **meta):
            sim_result = algorithm.run(hamiltonian, initial_state, evolution_time)

        with profile.span("point.state_wrap", **meta):
            sim_state = Statevector(sim_result.statevector)
            exact_state = Statevector(exact_result.statevector)
        with profile.span("point.fidelity", **meta):
            fidelity = (
                state_fidelity_to_exact(sim_state, exact_state)
                if self.compute_fidelity
                else float("nan")
            )
        with profile.span("obs.scalar_expectations", **meta):
            sim_observables = expectation_values(sim_state, observables)
            exact_observables = expectation_values(exact_state, observables)
            observable_errors = {
                name: absolute_error(exact_observables[name], sim_observables[name])
                for name in sim_observables
            }
        with profile.span("obs.site_expectations", **meta):
            sim_site = site_expectation_arrays(sim_state, site_obs_groups)
            exact_site = site_expectation_arrays(exact_state, site_obs_groups)

        with profile.span("point.result_pack", **meta):
            return SimPointResult(
                system_parameters=dict(system_parameters),
                num_qubits=self.num_qubits,
                evolution_time=evolution_time,
                exact_state=exact_result.statevector,
                sim_result=sim_result,
                fidelity=fidelity,
                observables=sim_observables,
                exact_observables=exact_observables,
                observable_errors=observable_errors,
                site_observables=sim_site,
                exact_site_observables=exact_site,
            )

    def _sweep_point_kwargs(
        self,
        value: float,
        sweep_parameter: str,
        system_parameters: dict[str, Any],
        fixed_parameters: Optional[dict[str, Any]],
        fixed_evolution_time: Optional[float],
    ) -> tuple[dict[str, Any], float, Optional[int]]:
        """Return per-point system parameters, evolution time, and optional steps."""

        point_parameters = dict(system_parameters)
        if fixed_parameters:
            point_parameters.update(fixed_parameters)

        if sweep_parameter == "evolution_time":
            return point_parameters, float(value), None
        if sweep_parameter == "num_trotter_steps":
            evolution_time = (
                fixed_evolution_time
                if fixed_evolution_time is not None
                else self._resolve_evolution_time(point_parameters)
            )
            return point_parameters, evolution_time, int(value)

        point_parameters[sweep_parameter] = value
        evolution_time = (
            fixed_evolution_time
            if fixed_evolution_time is not None
            else self._resolve_evolution_time(point_parameters)
        )
        return point_parameters, evolution_time, None

    def run_single_point(
        self,
        parameter_overrides: Optional[dict[str, Any]] = None,
        evolution_time: Optional[float] = None,
        num_trotter_steps: Optional[int] = None,
        method_name: Optional[str] = None,
    ) -> SimPointResult:
        system_parameters = self._resolved_system_parameters(parameter_overrides)
        evolution_time = self._resolve_evolution_time(system_parameters, evolution_time)

        if method_name is not None:
            algorithm = self._build_algorithm(method_name, num_trotter_steps)
        elif num_trotter_steps is not None:
            algorithm = self._build_algorithm(
                self.algorithm.evolution_method.name, num_trotter_steps
            )
        else:
            algorithm = self.algorithm

        return self._evaluate_point(system_parameters, evolution_time, algorithm)

    def run_single_point_comparison(
        self,
        method_names: Sequence[str],
        parameter_overrides: Optional[dict[str, Any]] = None,
        evolution_time: Optional[float] = None,
        num_trotter_steps: Optional[int] = None,
    ) -> SimValidationResult:
        """Evaluate one parameter point for each evolution method."""

        system_parameters = self._resolved_system_parameters(parameter_overrides)
        evolution_time = self._resolve_evolution_time(system_parameters, evolution_time)
        series = []
        for method_name in method_names:
            point = self.run_single_point(
                parameter_overrides=parameter_overrides,
                evolution_time=evolution_time,
                num_trotter_steps=num_trotter_steps,
                method_name=method_name,
            )
            series.append(
                SimBenchmarkResult(sweep_parameter="single_point", points=[point])
            )

        return SimValidationResult(
            system_parameters=dict(system_parameters),
            evolution_time=evolution_time,
            series=series,
        )

    def run_sweep(
        self,
        sweep_values: Sequence[float],
        sweep_parameter: str = "evolution_time",
        fixed_parameters: Optional[dict[str, Any]] = None,
        fixed_evolution_time: Optional[float] = None,
        method_name: Optional[str] = None,
    ) -> SimBenchmarkResult:
        system_parameters = self._resolved_system_parameters(fixed_parameters)
        method = method_name or self.algorithm.evolution_method.name
        points = []
        with profile.span("sweep.total", sweep_parameter=sweep_parameter, method=method):
            for value in sweep_values:
                point_parameters, ev_time, steps = self._sweep_point_kwargs(
                    value,
                    sweep_parameter,
                    system_parameters,
                    fixed_parameters,
                    fixed_evolution_time,
                )
                with profile.span(
                    "sweep.iteration",
                    evolution_time=float(ev_time),
                    sweep_value=float(value),
                    method=method,
                ):
                    with profile.span(
                        "sweep.rebuild_algorithm",
                        evolution_time=float(ev_time),
                    ):
                        algorithm = self._build_algorithm(method, steps)
                    points.append(
                        self._evaluate_point(point_parameters, ev_time, algorithm)
                    )

        return SimBenchmarkResult(sweep_parameter=sweep_parameter, points=points)

    def run_sweep_comparison(
        self,
        method_names: Sequence[str],
        sweep_values: Sequence[float],
        sweep_parameter: str = "evolution_time",
        fixed_parameters: Optional[dict[str, Any]] = None,
        fixed_evolution_time: Optional[float] = None,
    ) -> SimValidationResult:
        """Sweep a parameter independently for each evolution method."""

        system_parameters = self._resolved_system_parameters(fixed_parameters)
        series = [
            self.run_sweep(
                sweep_values=sweep_values,
                sweep_parameter=sweep_parameter,
                fixed_parameters=fixed_parameters,
                fixed_evolution_time=fixed_evolution_time,
                method_name=method_name,
            )
            for method_name in method_names
        ]
        evolution_time = (
            fixed_evolution_time
            if fixed_evolution_time is not None
            else self._resolve_evolution_time(system_parameters)
        )
        return SimValidationResult(
            system_parameters=dict(system_parameters),
            evolution_time=evolution_time,
            series=series,
        )

    def run_trotter_validation(
        self,
        evolution_time: float,
        method_names: Sequence[str],
        step_values: Sequence[int],
        parameter_overrides: Optional[dict[str, Any]] = None,
    ) -> SimValidationResult:
        system_parameters = self._resolved_system_parameters(parameter_overrides)
        series = []
        for method_name in method_names:
            points = []
            for steps in step_values:
                algorithm = self._build_algorithm(method_name, int(steps))
                points.append(
                    self._evaluate_point(system_parameters, evolution_time, algorithm)
                )
            series.append(
                SimBenchmarkResult(sweep_parameter="num_trotter_steps", points=points)
            )

        return SimValidationResult(
            system_parameters=dict(system_parameters),
            evolution_time=evolution_time,
            series=series,
        )
