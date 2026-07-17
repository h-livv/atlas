"""Variational Quantum Deflation built on the existing VQE optimization pipeline.

Ground state: ``VQE.run()``. Excited states: deflated cost functions that
reuse ``VQE._optimize_cost()`` with overlap penalties against previously
found states.

Overlap penalties are computed from exact statevectors (not shot estimates),
so this path assumes a statevector-capable backend.
"""

from __future__ import annotations

import numpy as np
from qiskit.quantum_info import Statevector, state_fidelity

from atlas.algorithms.vqe import VQE
from atlas.experiments.results import VQDResult, VQEResult


class VQD:
    """Sequential VQD using overlap penalties and the shared VQE optimizer.

    Responsibility:
        Find ``num_states`` low-lying eigenstates one after another. Each new
        state minimizes energy plus a penalty for overlap with previously
        discovered states.

    State:
        vqe: Shared ``VQE`` instance (estimator, optimizer, multi-start, RNG).
        num_states: How many eigenstates to compute (≥ 1).
        beta: Weight of the overlap penalty term.

    Usage:
        Built when ``algorithm.name == \"vqd\"``. Call ``run(hamiltonian, ansatz)``;
        the ansatz must support ``.bind(params)`` for statevector construction.
    """

    def __init__(self, vqe: VQE, num_states: int = 2, beta: float = 1.0):
        """Attach to an existing VQE and configure deflation settings.

        Inputs:
            vqe: Pre-built VQE providing multi-start optimization.
            num_states: Number of eigenstates to find sequentially.
            beta: Multiplier for Σ |⟨ψ|ψₖ⟩|² in the deflated cost.
        """

        if num_states < 1:
            raise ValueError("num_states must be at least 1.")
        self.vqe = vqe
        self.num_states = num_states
        self.beta = beta

    def run(self, hamiltonian, ansatz) -> VQDResult:
        """Find ``num_states`` eigenstates via sequential deflated VQE.

        Purpose:
            Approximate low-lying spectrum without changing the ansatz family.

        Inputs:
            hamiltonian: Object with ``.operator()`` and ``.num_qubits``.
            ansatz: Parameterized circuit supporting ``.circuit``,
                ``.num_parameters``, and ``.bind``.

        Process:
            1. Obtain the ground state with plain ``vqe.run`` (energy only).
            2. Bind optimal parameters and store the statevector.
            3. For each subsequent state, minimize
               ``E(θ) + β Σ_k |⟨ψ(θ)|ψₖ⟩|²`` via ``vqe._optimize_cost``.
            4. Append each new statevector so later penalties include it.

        Outputs:
            ``VQDResult`` whose ``states`` list holds one ``VQEResult`` per
            eigenstate (index 0 = ground).

        Side effects:
            Many estimator evaluations and statevector constructions. No I/O.
        """

        operator = hamiltonian.operator()
        states: list[VQEResult] = []
        previous_statevectors: list[Statevector] = []

        # Ground state uses the unmodified energy cost.
        ground = self.vqe.run(hamiltonian, ansatz)
        states.append(ground)
        previous_statevectors.append(
            Statevector(ansatz.bind(ground.optimal_parameters))
        )

        for _ in range(1, self.num_states):
            # Default-arg capture keeps ``prev`` bound to the growing list so
            # each new cost sees all previously found states.
            def cost_fn(param_values, prev=previous_statevectors):
                energy = self.vqe.estimator.expectation(
                    ansatz.circuit, operator, param_values
                )
                candidate = Statevector(ansatz.bind(param_values))
                # Penalize overlap so the optimizer is pushed into an orthogonal subspace.
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
