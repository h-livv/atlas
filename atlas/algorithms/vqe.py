"""Generic multi-start VQE optimization loop.

This module is independent of any specific Hamiltonian, ansatz family, or
execution backend. It works with any object exposing `.operator()` and
`.num_qubits` (see `atlas.physics.hamiltonians.Hamiltonian`) and any
`AnsatzSpec`-shaped object exposing `.circuit` and `.num_parameters`. It must
never import TFIM-specific or hardware-specific code.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from atlas.experiments.results import VQEResult


class VQE:
    """Runs multi-start classical optimization of an estimator-based cost function."""

    def __init__(self, estimator, optimizer, num_starts: int = 10, seed: Optional[int] = None):
        self.estimator = estimator
        self.optimizer = optimizer
        self.num_starts = num_starts
        self.seed = seed
        self._rng = np.random.default_rng(seed) if seed is not None else None

    def _draw_initial_point(self, size: int) -> np.ndarray:
        if self._rng is not None:
            return self._rng.uniform(0, 2 * np.pi, size=size)
        return np.random.uniform(0, 2 * np.pi, size=size)

    def run(self, hamiltonian, ansatz) -> VQEResult:
        """Optimize the ansatz parameters to minimize `hamiltonian`'s energy."""

        operator = hamiltonian.operator()

        def cost_fn(param_values):
            return self.estimator.expectation(ansatz.circuit, operator, param_values)

        global_best_energy = float("inf")
        global_best_parameters = None
        global_best_nfev = 0

        for _ in range(self.num_starts):
            initial_point = self._draw_initial_point(ansatz.num_parameters)
            run_result = self.optimizer.minimize(cost_fn, initial_point)
            if run_result.energy < global_best_energy:
                global_best_energy = run_result.energy
                global_best_parameters = run_result.parameters
                global_best_nfev = run_result.nfev

        return VQEResult(
            energy=global_best_energy,
            optimal_parameters=global_best_parameters,
            nfev=global_best_nfev,
            num_starts=self.num_starts,
            num_qubits=hamiltonian.num_qubits,
            optimizer_method=self.optimizer.method,
        )
