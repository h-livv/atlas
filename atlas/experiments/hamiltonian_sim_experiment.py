"""Hamiltonian simulation experiment orchestration.

Composes Hamiltonians, initial states, and observables for a given parameter
point, runs the configured dynamics algorithm, and computes fidelity and
observable metrics against exact time evolution.

Architectural role:
    Parallel to ``TFIMExperiment`` but for product-formula dynamics. Called by
    ``ConfiguredExperimentRunner`` for ``hamiltonian_sim`` (and related)
    algorithms. Plotting/CSV remain in ``atlas.experiments.outputs``.
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
    """Dynamics experiment workflow using YAML-configured components.

    Responsibility:
        For each TFIM parameter point and evolution time, compare exact
        Schrödinger evolution to a Trotterized circuit simulation, recording
        fidelity, observables, and per-observable absolute errors.

    State:
        config: ``AtlasConfig`` for builders and defaults.
        algorithm: Default ``HamiltonianSimulation`` (method + evolver).
        num_qubits: From system parameters.
        compute_fidelity: Whether fidelity is computed (else NaN).

    Usage:
        Built by the factory for dynamics algorithms. Call
        ``run_single_point``, ``run_sweep``, or ``run_trotter_validation``.
        Temporary algorithm instances are created when overriding Trotter
        steps or validating multiple methods, always reusing the same evolver.
    """

    def __init__(self, config: AtlasConfig, algorithm: HamiltonianSimulation):
        """Store config, default algorithm, and analysis flags.

        Inputs:
            config: Full experiment configuration.
            algorithm: Configured ``HamiltonianSimulation`` instance.
        """

        self.config = config
        self.algorithm = algorithm
        self.num_qubits = int(config.system.parameters["num_qubits"])
        self.compute_fidelity = config.analysis.fidelity

    def _build_point_context(self, J: float, h: float):
        """Build Hamiltonian, initial state, and observables for one point.

        Purpose:
            Share construction across single-point evaluation and validation.

        Inputs:
            J: Coupling override.
            h: Transverse-field override.

        Process:
            Build Hamiltonian with overrides; initial state and observables
            from config and qubit count.

        Outputs:
            Tuple ``(hamiltonian, initial_state, observables)``.

        Side effects:
            None.
        """

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
        """Compare exact and simulated evolution for one point and algorithm.

        Purpose:
            Core evaluation used by single-point, sweep, and validation paths
            so metrics stay consistent.

        Inputs:
            J: Coupling.
            h: Transverse field.
            evolution_time: Total evolution time ``t``.
            algorithm: ``HamiltonianSimulation`` to run (may differ from
                ``self.algorithm`` when steps/methods are overridden).

        Process:
            Exact evolve initial statevector under the Hamiltonian; run the
            Trotter algorithm; compute fidelity (optional), sim/exact
            observables, and absolute errors per observable.

        Outputs:
            ``SimPointResult``.

        Side effects:
            Dense linear algebra for exact evolution; circuit evolution via
            the evolver. No filesystem I/O.
        """

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
        """Run exact and simulated time evolution for one parameter point.

        Purpose:
            Entry point for single-shot dynamics experiments and one step of
            a sweep.

        Inputs:
            J: Coupling.
            h: Transverse field.
            evolution_time: Optional; defaults via ``resolve_evolution_time``.
            num_trotter_steps: Optional override; when set, rebuilds the
                evolution method with the same method name as
                ``self.algorithm`` but a new step count.

        Process:
            Resolve time; optionally clone ``HamiltonianSimulation`` with a
            new method; call ``_evaluate_point``.

        Outputs:
            ``SimPointResult``.

        Side effects:
            Same as ``_evaluate_point``.
        """

        if evolution_time is None:
            evolution_time = resolve_evolution_time(self.config)

        algorithm = self.algorithm
        if num_trotter_steps is not None:
            # Keep method family (lie/strang) but change discretization for sweeps.
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
        """Run dynamics for each value in ``sweep_values``.

        Purpose:
            Build a curve of fidelity/depth vs ``evolution_time``, ``h``, or
            ``num_trotter_steps``.

        Inputs:
            J: Fixed coupling.
            sweep_values: Grid along the chosen axis.
            sweep_parameter: ``"evolution_time"``, ``"h"``, or
                ``"num_trotter_steps"``.
            fixed_h: Companion ``h`` when not sweeping ``h``.
            fixed_evolution_time: Companion time when not sweeping time.

        Process:
            For each value, resolve companions from arguments or config and
            call ``run_single_point`` with the appropriate kwargs.

        Outputs:
            ``SimBenchmarkResult`` tagged with ``sweep_parameter``.

        Side effects:
            Many evaluations. Raises ``ValueError`` for unsupported sweep
            parameters.
        """

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
        """Sweep Trotter steps for one or more evolution methods.

        Purpose:
            Compare product formulas (e.g. Lie vs Strang) as step count
            increases, at fixed physics and total time.

        Inputs:
            J, h: Fixed TFIM parameters.
            evolution_time: Fixed total evolution time.
            method_names: Evolution method names (``lie``, ``strang``, …).
            step_values: Trotter step grid.

        Process:
            For each method, build a fresh ``HamiltonianSimulation`` per step
            count (reusing ``self.algorithm.evolver``), evaluate, and collect
            one ``SimBenchmarkResult`` series per method.

        Outputs:
            ``SimValidationResult`` with one series per method.

        Side effects:
            Many simulation evaluations; no filesystem I/O.
        """

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
