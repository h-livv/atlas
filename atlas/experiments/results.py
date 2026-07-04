"""Structured result containers shared by experiment, analysis, I/O, and
plotting code.

These replace the parallel arrays used throughout `vqe_legacy/tfim_vis.py`
with explicit, self-describing records. `physics/` must never import from
this module -- dependencies point from `experiments/` toward `physics/`,
never the other way around.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class VQEResult:
    """Outcome of a (multi-start) VQE optimization run."""

    energy: float
    optimal_parameters: np.ndarray
    nfev: int
    num_starts: int
    num_qubits: int
    optimizer_method: str


@dataclass
class HardwareEvaluationResult:
    """Outcome of evaluating a bound ansatz on IBM hardware."""

    backend: str
    energy: float
    observables: dict
    job_id: str
    energy_stderr: Optional[float] = None
    observable_stderr: Optional[dict] = None


@dataclass
class TFIMPointResult:
    """All results (exact, simulated, optionally hardware) for one `(J, h)` point."""

    h: float
    J: float
    num_qubits: int
    exact_energy: float
    exact_state: np.ndarray
    vqe_result: VQEResult
    fidelity: float
    observables: dict
    hardware_result: Optional[HardwareEvaluationResult] = None


@dataclass
class TFIMBenchmarkResult:
    """A collection of `TFIMPointResult` for a full `h` sweep."""

    points: list = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.points)

    def __iter__(self):
        return iter(self.points)

    @property
    def h_values(self) -> np.ndarray:
        return np.array([p.h for p in self.points])

    @property
    def J_values(self) -> np.ndarray:
        return np.array([p.J for p in self.points])

    @property
    def exact_energies(self) -> np.ndarray:
        return np.array([p.exact_energy for p in self.points])

    @property
    def vqe_energies(self) -> np.ndarray:
        return np.array([p.vqe_result.energy for p in self.points])

    @property
    def fidelities(self) -> np.ndarray:
        return np.array([p.fidelity for p in self.points])

    @property
    def iterations(self) -> np.ndarray:
        return np.array([p.vqe_result.nfev for p in self.points])

    def observable_values(self, name: str) -> np.ndarray:
        return np.array([p.observables.get(name) for p in self.points])

    @property
    def hardware_energies(self) -> np.ndarray:
        return np.array(
            [p.hardware_result.energy if p.hardware_result else None for p in self.points]
        )

    def hardware_observable_values(self, name: str) -> np.ndarray:
        return np.array(
            [
                p.hardware_result.observables.get(name) if p.hardware_result else None
                for p in self.points
            ]
        )
