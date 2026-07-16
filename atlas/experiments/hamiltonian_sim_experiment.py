"""Hamiltonian simulation experiment orchestration.

Composes Hamiltonians, initial states, and observables for a given parameter
point, runs the configured dynamics algorithm, and computes fidelity and
observable metrics against exact time evolution.
"""

from __future__ import annotations

from typing import Optional, Sequence

from qiskit.quantum_info import Statevector

from atlas.algorithms.hamiltonian_sim import HamiltonianSimulation
from atlas.analysis.metrics import absolute_error, expectation_values, state_fidelity_to_exact
from atlas.config import AtlasConfig
from atlas.experiments.builders import (
    build_evolution_method_named,
    build_hamiltonian,
    build_initial_state,
    build_observables,
    resolve_evolution_time,
)
from atlas.experiments.results import (
    SimBenchmarkResult,
    SimPointResult,
    SimValidationResult,
)


class HamiltonianSimExperiment:
    """Dynamics experiment workflow using YAML-configured components."""

    def __init__(self, config: AtlasConfig, algorithm: HamiltonianSimulation):
        self.config = config
        self.algorithm = algorithm
        self.num_qubits = int(config.system.parameters["num_qubits"])
        self.compute_fidelity = config.analysis.fidelity

    def _build_point_context(self, J: float, h: float):
        hamiltonian = build_hamiltonian(
            self.config, parameter_overrides={"J": J, "h": h}
        )
        initial_state = build_initial_state(self.config, self.num_qubits)
        observables = build_observables(self.config, self.num_qubits)
        return hamiltonian, initial_state, observables

    def _evaluate_point(
        self,
        J: float,
        h: float,
        evolution_time: float,
        algorithm: HamiltonianSimulation,
    ) -> SimPointResult:
        hamiltonian, initial_state, observables = self._build_point_context(J, h)
        exact_initial = initial_state.statevector()
        exact_result = hamiltonian.exact_time_evolution(exact_initial, evolution_time)
        sim_result = algorithm.run(hamiltonian, initial_state, evolution_time)

        sim_state = Statevector(sim_result.statevector)
        exact_state = Statevector(exact_result.statevector)
        fidelity = (
            state_fidelity_to_exact(sim_state, exact_state)
            if self.compute_fidelity
            else float("nan")
        )
        sim_observables = expectation_values(sim_state, observables)
        exact_observables = expectation_values(exact_state, observables)
        observable_errors = {
            name: absolute_error(exact_observables[name], sim_observables[name])
            for name in sim_observables
        }

        return SimPointResult(
            h=h,
            J=J,
            num_qubits=self.num_qubits,
            evolution_time=evolution_time,
            exact_state=exact_result.statevector,
            sim_result=sim_result,
            fidelity=fidelity,
            observables=sim_observables,
            exact_observables=exact_observables,
            observable_errors=observable_errors,
        )

    def run_single_point(
        self,
        J: float,
        h: float,
        evolution_time: Optional[float] = None,
        num_trotter_steps: Optional[int] = None,
    ) -> SimPointResult:
        """Run exact and simulated time evolution for one parameter point."""

        if evolution_time is None:
            evolution_time = resolve_evolution_time(self.config)

        algorithm = self.algorithm
        if num_trotter_steps is not None:
            method_name = self.algorithm.evolution_method.name
            evolution_method = build_evolution_method_named(
                method_name, num_trotter_steps
            )
            algorithm = HamiltonianSimulation(
                evolution_method=evolution_method,
                evolver=self.algorithm.evolver,
            )

        return self._evaluate_point(J, h, evolution_time, algorithm)

    def run_sweep(
        self,
        J: float,
        sweep_values: Sequence[float],
        sweep_parameter: str = "evolution_time",
        fixed_h: Optional[float] = None,
        fixed_evolution_time: Optional[float] = None,
    ) -> SimBenchmarkResult:
        """Run dynamics for each value in ``sweep_values``."""

        points = []
        for value in sweep_values:
            if sweep_parameter == "evolution_time":
                h = fixed_h if fixed_h is not None else float(self.config.system.parameters["h"])
                point = self.run_single_point(J=J, h=h, evolution_time=float(value))
            elif sweep_parameter == "h":
                evolution_time = (
                    fixed_evolution_time
                    if fixed_evolution_time is not None
                    else resolve_evolution_time(self.config)
                )
                point = self.run_single_point(J=J, h=float(value), evolution_time=evolution_time)
            elif sweep_parameter == "num_trotter_steps":
                h = fixed_h if fixed_h is not None else float(self.config.system.parameters["h"])
                evolution_time = (
                    fixed_evolution_time
                    if fixed_evolution_time is not None
                    else resolve_evolution_time(self.config)
                )
                point = self.run_single_point(
                    J=J,
                    h=h,
                    evolution_time=evolution_time,
                    num_trotter_steps=int(value),
                )
            else:
                raise ValueError(f"Unsupported sweep parameter '{sweep_parameter}'.")
            points.append(point)

        return SimBenchmarkResult(sweep_parameter=sweep_parameter, points=points)

    def run_trotter_validation(
        self,
        J: float,
        h: float,
        evolution_time: float,
        method_names: Sequence[str],
        step_values: Sequence[int],
    ) -> SimValidationResult:
        """Sweep Trotter steps for one or more evolution methods."""

        series = []
        for method_name in method_names:
            points = []
            for steps in step_values:
                evolution_method = build_evolution_method_named(method_name, int(steps))
                algorithm = HamiltonianSimulation(
                    evolution_method=evolution_method,
                    evolver=self.algorithm.evolver,
                )
                points.append(self._evaluate_point(J, h, evolution_time, algorithm))
            series.append(
                SimBenchmarkResult(sweep_parameter="num_trotter_steps", points=points)
            )

        return SimValidationResult(
            h=h,
            J=J,
            evolution_time=evolution_time,
            series=series,
        )
