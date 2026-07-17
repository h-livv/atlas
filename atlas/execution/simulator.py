"""Backward-compatible re-exports.

Prefer :mod:`atlas.execution.estimator` for variational cost evaluation and
:mod:`atlas.execution.evolver` for Hamiltonian simulation.

This module exists so older imports of ``SimulatorEstimator`` from
``atlas.execution.simulator`` keep working after the estimator/evolver split.
"""

from __future__ import annotations

from atlas.execution.estimator import SimulatorEstimator

__all__ = ["SimulatorEstimator"]
