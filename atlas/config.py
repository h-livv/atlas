"""Small typed configuration objects for Atlas experiments and execution.

These data classes hold values that were previously hardcoded in the legacy
scripts (``J``, ``h``, ``num_qubits``, ``num_starts``, ``maxiter``, optimizer
method, output directory, hardware backend/CSV settings). They own user
intent and runtime settings only -- never computed results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence


@dataclass
class TFIMConfig:
    """Physical model parameters for the transverse field Ising model."""

    num_qubits: int = 2
    J: float = 1.0
    h: float = 0.5
    h_values: Optional[Sequence[float]] = None


@dataclass
class OptimizerConfig:
    """Classical optimizer settings for the VQE loop."""

    method: str = "COBYLA"
    maxiter: int = 200
    num_starts: int = 10


@dataclass
class SimulatorConfig:
    """Local statevector simulator settings."""

    seed: Optional[int] = None


@dataclass
class HardwareConfig:
    """IBM Runtime backend selection and execution settings."""

    backend_name: Optional[str] = None
    resilience_level: int = 1
    optimization_level: int = 3
    hardware_csv_path: Optional[str] = None


@dataclass
class OutputConfig:
    """Output locations for plots and benchmark CSVs."""

    output_dir: str = "atlas/data"


@dataclass
class ExperimentConfig:
    """Top-level configuration combining all sub-configs for an experiment."""

    tfim: TFIMConfig = field(default_factory=TFIMConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    simulator: SimulatorConfig = field(default_factory=SimulatorConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def default_tfim_config() -> ExperimentConfig:
    """Return an `ExperimentConfig` matching legacy default behavior."""

    return ExperimentConfig()
