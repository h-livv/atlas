"""Configuration objects for Atlas experiments and execution.

These dataclasses hold user intent and runtime settings only -- never
computed results. YAML loading via ``atlas.io.yaml_config`` produces
an ``AtlasConfig``; legacy ``ExperimentConfig`` remains for compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence


# ---------------------------------------------------------------------------
# Legacy dataclasses (retained for backward compatibility)
# ---------------------------------------------------------------------------


@dataclass
class TFIMConfig:
    """Physical model parameters for the transverse field Ising model."""

    num_qubits: int = 2
    J: float = 1.0
    h: float = 0.5
    h_values: Optional[Sequence[float]] = None


@dataclass
class OptimizerConfig:
    """Classical optimizer settings for variational algorithms."""

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

    provider: str = "ibm_runtime"
    enabled: bool = False
    backend_name: Optional[str] = None
    resilience_level: int = 1
    optimization_level: int = 3
    hardware_csv_path: Optional[str] = None


@dataclass
class OutputConfig:
    """Output locations and toggles for plots and benchmark CSVs."""

    output_dir: str = "atlas/data"
    csv: bool = True
    plots: bool = True


@dataclass
class ExperimentConfig:
    """Legacy top-level configuration combining all sub-configs."""

    tfim: TFIMConfig = field(default_factory=TFIMConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    simulator: SimulatorConfig = field(default_factory=SimulatorConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def default_tfim_config() -> ExperimentConfig:
    """Return an ``ExperimentConfig`` matching legacy default behavior."""

    return ExperimentConfig()


# ---------------------------------------------------------------------------
# YAML-driven generic configuration
# ---------------------------------------------------------------------------


@dataclass
class ExperimentMetaConfig:
    """Experiment type and display name."""

    type: str = "single_point"
    name: str = ""


@dataclass
class SystemConfig:
    """Physical system selection and parameters."""

    name: str = "tfim"
    parameters: dict[str, Any] = field(default_factory=dict)
    sweep: Optional[dict[str, Any]] = None


@dataclass
class AlgorithmConfig:
    """Algorithm selection (vqe, vqd, ...)."""

    name: str = "vqe"
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnsatzConfig:
    """Parameterized circuit family selection."""

    name: str = "hardware_efficient"
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizerSettingsConfig:
    """Named optimizer backend and its parameters."""

    name: str = "scipy"
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class BackendConfig:
    """Simulator or execution backend for algorithm cost evaluation."""

    name: str = "statevector"
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisConfig:
    """Post-run analysis options."""

    observables: str = "tfim_default"
    fidelity: bool = True


@dataclass
class AtlasConfig:
    """Top-level YAML-driven experiment configuration."""

    experiment: ExperimentMetaConfig = field(default_factory=ExperimentMetaConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    ansatz: AnsatzConfig = field(default_factory=AnsatzConfig)
    optimizer: OptimizerSettingsConfig = field(default_factory=OptimizerSettingsConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
