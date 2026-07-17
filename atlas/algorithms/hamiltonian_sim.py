"""Hamiltonian time-evolution simulation orchestration.

Independent of any specific physical model or evolution method implementation.
Works with duck-typed objects exposing ``.operator()`` and ``.num_qubits`` for
the Hamiltonian, ``.preparation_circuit()`` for the initial state, and
``.build_circuit()`` / ``.evolve()`` on the injected evolution and execution
components.

Architectural role:
    This class does not implement Trotter math or statevector simulation; it
    sequences those collaborators and packages a ``SimulationResult``.
"""

from __future__ import annotations

from atlas.execution.evolver import StatevectorEvolver
from atlas.experiments.results import SimulationResult
from atlas import profiling as profile


class HamiltonianSimulation:
    """Builds an evolution circuit and returns the simulated final state.

    Responsibility:
        Orchestrate one dynamics shot: ask the evolution method for a circuit,
        ask the evolver for the final statevector, attach Trotter metadata.

    State:
        evolution_method: Object with ``.build_circuit(H, ψ₀, t)`` returning a
            spec that includes circuit + method/step metadata.
        evolver: Execution backend with ``.evolve(circuit)`` returning an
            object that exposes ``.statevector``.

    Usage:
        Built by ``build_hamiltonian_simulation``. Experiments may construct
        additional instances that reuse the same evolver with different
        Trotter step counts or method names.
    """

    def __init__(self, evolution_method, evolver: StatevectorEvolver):
        """Inject circuit-construction and execution strategies.

        Inputs:
            evolution_method: Product formula (or other) circuit builder.
            evolver: Backend that simulates a fully bound circuit.
        """

        self.evolution_method = evolution_method
        self.evolver = evolver

    def run(self, hamiltonian, initial_state, evolution_time: float) -> SimulationResult:
        """Simulate time evolution under ``hamiltonian`` from ``initial_state``.

        Purpose:
            Approximate ``e^{-i H t} |ψ₀⟩`` via the configured product formula
            and return the simulated state plus circuit metadata.

        Inputs:
            hamiltonian: Model with ``.operator()`` / ``.num_qubits``.
            initial_state: Pure-state prep object consumed by the evolution
                method (typically ``.preparation_circuit()``).
            evolution_time: Total evolution time ``t``.

        Process:
            1. ``build_circuit`` → ``EvolutionCircuitSpec`` (prep + Trotter gates).
            2. ``evolver.evolve(spec.circuit)`` → final amplitudes.
            3. Copy amplitudes and Trotter metadata into ``SimulationResult``.

        Outputs:
            ``SimulationResult`` suitable for fidelity / observable comparison.

        Side effects:
            Statevector simulation of the evolution circuit. No filesystem I/O.
        """

        with profile.span("sim.circuit_build"):
            spec = self.evolution_method.build_circuit(
                hamiltonian,
                initial_state,
                evolution_time,
            )
        with profile.span("sim.evolver_evolve"):
            evolution = self.evolver.evolve(spec.circuit)
        return SimulationResult(
            statevector=evolution.statevector,
            num_qubits=hamiltonian.num_qubits,
            evolution_time=evolution_time,
            method_name=spec.method_name,
            num_trotter_steps=spec.num_trotter_steps,
            circuit_depth=spec.circuit_depth,
        )
