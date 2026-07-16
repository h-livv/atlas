"""Hamiltonian time-evolution simulation orchestration.

Independent of any specific physical model or evolution method implementation.
Works with duck-typed objects exposing ``.operator()`` and ``.num_qubits`` for
the Hamiltonian, ``.preparation_circuit()`` for the initial state, and
``.build_circuit()`` / ``.evolve()`` on the injected evolution and execution
components.
"""

from __future__ import annotations

from atlas.execution.evolver import StatevectorEvolver
from atlas.experiments.results import SimulationResult


class HamiltonianSimulation:
    """Builds an evolution circuit and returns the simulated final state."""

    def __init__(self, evolution_method, evolver: StatevectorEvolver):
        self.evolution_method = evolution_method
        self.evolver = evolver

    def run(self, hamiltonian, initial_state, evolution_time: float) -> SimulationResult:
        """Simulate time evolution under ``hamiltonian`` from ``initial_state``."""

        spec = self.evolution_method.build_circuit(
            hamiltonian,
            initial_state,
            evolution_time,
        )
        evolution = self.evolver.evolve(spec.circuit)
        return SimulationResult(
            statevector=evolution.statevector,
            num_qubits=hamiltonian.num_qubits,
            evolution_time=evolution_time,
            method_name=spec.method_name,
            num_trotter_steps=spec.num_trotter_steps,
            circuit_depth=spec.circuit_depth,
        )
