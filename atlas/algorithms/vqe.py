"""Generic multi-start VQE optimization loop.

This module is independent of any specific Hamiltonian, ansatz family, or
execution backend. It works with any object exposing ``.operator()`` and
``.num_qubits`` and any ``AnsatzSpec``-shaped object exposing ``.circuit`` and
``.num_parameters``. It must never import TFIM-specific or hardware-specific
code.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from atlas.experiments.results import VQEResult


class VQE:
    """Runs multi-start classical optimization of an estimator-based cost function.

    Responsibility:
        Own the outer variational loop: draw random initial parameters, call
        the classical optimizer, and keep the best energy found across starts.
        Does not build Hamiltonians or ansätze.

    State:
        estimator: Object with ``.expectation(circuit, operator, params)``.
        optimizer: Object with ``.minimize(cost_fn, initial_point)`` and
            ``.method``.
        num_starts: How many independent optimizations to attempt.
        seed: Optional RNG seed for reproducible initial points.
        _rng: ``numpy`` Generator when ``seed`` is set; otherwise ``None``
            (falls back to global ``np.random``).

    Usage:
        Construct via ``build_vqe`` / inject into ``VQD``. Call ``run(H, ansatz)``
        for ground-state search, or ``_optimize_cost`` with a custom scalar
        cost (used by VQD deflation).
    """

    def __init__(self, estimator, optimizer, num_starts: int = 10, seed: Optional[int] = None):
        """Store execution/optimizer dependencies for later ``run`` calls.

        Inputs:
            estimator: Backend used to evaluate ⟨ψ(θ)|H|ψ(θ)⟩.
            optimizer: Classical minimizer wrapper.
            num_starts: Multi-start count (local minima mitigation).
            seed: If set, fix the RNG used for initial parameter draws.
        """

        self.estimator = estimator
        self.optimizer = optimizer
        self.num_starts = num_starts
        self.seed = seed
        self._rng = np.random.default_rng(seed) if seed is not None else None

    def _draw_initial_point(self, size: int) -> np.ndarray:
        """Sample a random parameter vector uniformly from ``[0, 2π)``.

        Purpose:
            Provide diverse starting points so multi-start can escape poor
            local minima of the variational landscape.

        Inputs:
            size: Number of ansatz parameters.

        Process:
            Draw from the seeded generator when available; otherwise use the
            process-global ``np.random``.

        Outputs:
            1-D float array of length ``size``.

        Side effects:
            Advances RNG state.
        """

        if self._rng is not None:
            return self._rng.uniform(0, 2 * np.pi, size=size)
        return np.random.uniform(0, 2 * np.pi, size=size)

    def _optimize_cost(
        self,
        cost_fn: Callable,
        ansatz,
        num_qubits: int,
    ) -> VQEResult:
        """Run multi-start optimization for an arbitrary scalar ``cost_fn``.

        Purpose:
            Share one multi-start implementation between plain VQE (energy)
            and VQD (energy + overlap penalties) without duplicating the loop.

        Inputs:
            cost_fn: Callable ``params -> float`` to minimize.
            ansatz: Object exposing ``.num_parameters`` (for initial draws).
            num_qubits: Copied into the result for downstream bookkeeping.

        Process:
            For each start, draw an initial point, call ``optimizer.minimize``,
            and keep the run with the lowest reported energy. Record that run's
            ``nfev`` (not the sum across starts).

        Outputs:
            ``VQEResult`` for the best start found.

        Side effects:
            Many cost evaluations via the optimizer/estimator. No filesystem I/O.
        """

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
            num_qubits=num_qubits,
            optimizer_method=self.optimizer.method,
        )

    def run(self, hamiltonian, ansatz) -> VQEResult:
        """Optimize the ansatz parameters to minimize ``hamiltonian``'s energy.

        Purpose:
            Standard VQE ground-state search using the injected estimator.

        Inputs:
            hamiltonian: Object with ``.operator()`` and ``.num_qubits``.
            ansatz: Object with ``.circuit`` and ``.num_parameters``.

        Process:
            Build a cost closure that asks the estimator for
            ``⟨ψ(θ)|H|ψ(θ)⟩``, then delegate to ``_optimize_cost``.

        Outputs:
            ``VQEResult`` with best energy and parameters across starts.

        Side effects:
            Simulator/optimizer work only; no I/O.
        """

        operator = hamiltonian.operator()

        def cost_fn(param_values):
            return self.estimator.expectation(ansatz.circuit, operator, param_values)

        return self._optimize_cost(cost_fn, ansatz, hamiltonian.num_qubits)
