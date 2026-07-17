"""Exact local simulation of parameter-free evolution circuits.

Distinct from estimators: an evolver runs a *complete* circuit once and
returns the final statevector. That matches Hamiltonian simulation, where
the product-formula circuit is already fully bound (no variational θ).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector


@dataclass
class EvolutionResult:
    """Final amplitudes after simulating an evolution circuit.

    Responsibility:
        Carry the dense state produced by ``StatevectorEvolver`` so algorithms
        can package a ``SimulationResult`` without depending on Qiskit types.

    State:
        statevector: Complex amplitude array of length ``2 ** num_qubits``.
        num_qubits: Register size taken from the simulated circuit.

    Usage:
        Returned by ``StatevectorEvolver.evolve``; consumed immediately by
        ``HamiltonianSimulation.run``.
    """

    statevector: np.ndarray
    num_qubits: int


class StatevectorEvolver:
    """Simulate a fully bound circuit and return its final statevector.

    Responsibility:
        Provide the dynamics execution primitive (ideal, noiseless).

    State:
        Stateless — no instance fields; safe to reuse across many runs.

    Usage:
        Built when ``backend.name == \"statevector_evolver\"``. Injected into
        ``HamiltonianSimulation``.
    """

    def evolve(self, circuit: QuantumCircuit) -> EvolutionResult:
        """Classically simulate ``circuit`` and return final amplitudes.

        Purpose:
            Evaluate a Trotterized evolution circuit exactly (as a statevector).

        Inputs:
            circuit: Parameter-free preparation + evolution circuit.

        Process:
            Construct Qiskit ``Statevector(circuit)`` and convert to ``numpy``.

        Outputs:
            ``EvolutionResult`` with amplitudes and qubit count.

        Side effects:
            Ideal simulation; memory scales as ``2 ** n``. No I/O.
        """

        statevector = np.asarray(Statevector(circuit))
        return EvolutionResult(
            statevector=statevector,
            num_qubits=circuit.num_qubits,
        )
