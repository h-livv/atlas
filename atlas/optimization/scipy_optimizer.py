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
    """Outcome of a single `scipy.optimize.minimize` call."""

    energy: float
    parameters: np.ndarray
    nfev: int


class ScipyOptimizer:
    """Configures and runs a SciPy classical optimizer for a cost function."""

    def __init__(self, method: str = "COBYLA", maxiter: int = 200):
        self.method = method
        self.maxiter = maxiter

    def minimize(self, cost_fn, initial_point) -> OptimizerRunResult:
        """Minimize `cost_fn` starting from `initial_point`."""

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
