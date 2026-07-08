"""Variational Quantum Deflation built on the existing VQE optimization pipeline.

Ground state: ``VQE.run()``. Excited states: deflated cost functions that
reuse ``VQE._optimize_cost()`` with overlap penalties against previously
found states.
"""

from __future__ import annotations

import numpy as np
from qiskit.quantum_info import Statevector, state_fidelity

from atlas.algorithms.vqe import VQE
from atlas.experiments.results import VQDResult, VQEResult


class VQD:
    """Sequential VQD using overlap penalties and the shared VQE optimizer."""

    def __init__(self, vqe: VQE, num_states: int = 2, beta: float = 1.0):
        if num_states < 1:
            raise ValueError("num_states must be at least 1.")
        self.vqe = vqe
        self.num_states = num_states
        self.beta = beta

    def run(self, hamiltonian, ansatz) -> VQDResult:
        """Find ``num_states`` eigenstates via sequential deflated VQE."""

        operator = hamiltonian.operator()
        states: list[VQEResult] = []
        previous_statevectors: list[Statevector] = []

        ground = self.vqe.run(hamiltonian, ansatz)
        states.append(ground)
        previous_statevectors.append(
            Statevector(ansatz.bind(ground.optimal_parameters))
        )

        for _ in range(1, self.num_states):
            def cost_fn(param_values, prev=previous_statevectors):
                energy = self.vqe.estimator.expectation(
                    ansatz.circuit, operator, param_values
                )
                candidate = Statevector(ansatz.bind(param_values))
                overlap_sum = sum(
                    abs(state_fidelity(candidate, prev_state)) ** 2
                    for prev_state in prev
                )
                return energy + self.beta * overlap_sum

            excited = self.vqe._optimize_cost(
                cost_fn, ansatz, hamiltonian.num_qubits
            )
            states.append(excited)
            previous_statevectors.append(
                Statevector(ansatz.bind(excited.optimal_parameters))
            )

        return VQDResult(states=states, num_qubits=hamiltonian.num_qubits)
