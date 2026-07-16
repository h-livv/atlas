"""Backward-compatible re-exports.

Prefer :mod:`atlas.execution.estimator` for variational cost evaluation and
:mod:`atlas.execution.evolver` for Hamiltonian simulation.
"""

from __future__ import annotations

from atlas.execution.estimator import SimulatorEstimator

__all__ = ["SimulatorEstimator"]
