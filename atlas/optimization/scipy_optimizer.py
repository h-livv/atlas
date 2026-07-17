"""Thin wrapper around SciPy's classical optimizers.

This module is reusable across any Hamiltonian and ansatz: it owns only
optimizer settings and knows nothing about Qiskit, quantum circuits, or
physics. `VQE` calls `minimize()` once per multi-start iteration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass
class OptimizerRunResult:
    """Outcome of a single `scipy.optimize.minimize` call.

    Responsibility:
        Normalize SciPy's result object into the small interface VQE expects
        (``.energy``, ``.parameters``, ``.nfev``).

    State:
        energy: Final cost value (``result.fun``).
        parameters: Optimal parameter vector (``result.x``).
        nfev: Number of cost-function evaluations for this start.

    Usage:
        Returned by ``ScipyOptimizer.minimize``; compared across multi-starts
        inside ``VQE._optimize_cost``.
    """

    energy: float
    parameters: np.ndarray
    nfev: int


class ScipyOptimizer:
    """Configures and runs a SciPy classical optimizer for a cost function.

    Responsibility:
        Hold method/maxiter settings and invoke ``scipy.optimize.minimize``
        without knowing what the cost function represents physically.

    State:
        method: SciPy method name (default ``COBYLA``).
        maxiter: Forwarded as ``options[\"maxiter\"]``.

    Usage:
        Constructed by ``build_optimizer``; injected into ``VQE``. Call
        ``minimize`` once per random initial point.
    """

    def __init__(self, method: str = "COBYLA", maxiter: int = 200):
        """Store classical optimizer settings.

        Inputs:
            method: SciPy optimizer identifier.
            maxiter: Iteration budget passed to SciPy options.
        """

        self.method = method
        self.maxiter = maxiter

    def minimize(self, cost_fn, initial_point) -> OptimizerRunResult:
        """Minimize `cost_fn` starting from `initial_point`.

        Purpose:
            Perform one classical optimization run for VQE multi-start.

        Inputs:
            cost_fn: Callable mapping a parameter vector to a scalar cost
                (typically the estimator energy).
            initial_point: 1-D array of starting parameters.

        Process:
            Call ``scipy.optimize.minimize`` with the configured method and
            ``maxiter``; package ``fun``, ``x``, and ``nfev``.

        Outputs:
            ``OptimizerRunResult`` for this start. Does not inspect
            ``result.success`` — callers treat ``fun`` as the reported energy.

        Side effects:
            Evaluates ``cost_fn`` many times (CPU / simulator load). No I/O.
        """

        result = minimize(
            cost_fn,
            initial_point,
            method=self.method,
            options={"maxiter": self.maxiter},
        )
        return OptimizerRunResult(
            energy=result.fun,
            parameters=result.x,
            nfev=result.nfev,
        )
