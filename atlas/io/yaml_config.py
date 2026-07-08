"""Load Atlas experiment configuration from YAML files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from atlas.config import (
    AlgorithmConfig,
    AnalysisConfig,
    AnsatzConfig,
    AtlasConfig,
    BackendConfig,
    ExperimentMetaConfig,
    HardwareConfig,
    OptimizerSettingsConfig,
    OutputConfig,
    SystemConfig,
)

_REQUIRED_SECTIONS = (
    "experiment",
    "system",
    "algorithm",
    "ansatz",
    "optimizer",
    "backend",
    "hardware",
    "analysis",
    "output",
)

_SUPPORTED = {
    "experiment_type": {"single_point", "sweep"},
    "system": {"tfim"},
    "algorithm": {"vqe", "vqd"},
    "ansatz": {"hardware_efficient"},
    "optimizer": {"scipy"},
    "backend": {"statevector"},
    "observables": {"tfim_default"},
}


def _section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    if key not in raw:
        raise ValueError(f"Missing required config section '{key}'.")
    section = raw[key]
    if not isinstance(section, dict):
        raise ValueError(f"Config section '{key}' must be a mapping.")
    return section


def _named_section(raw: dict[str, Any], key: str, supported: set[str]) -> tuple[str, dict]:
    section = _section(raw, key)
    name = section.get("name")
    if not name:
        raise ValueError(f"Section '{key}' requires a 'name' field.")
    if name not in supported:
        raise ValueError(
            f"Unknown {key} '{name}'. Supported: {', '.join(sorted(supported))}."
        )
    parameters = section.get("parameters") or {}
    if not isinstance(parameters, dict):
        raise ValueError(f"Section '{key}.parameters' must be a mapping.")
    return name, parameters


def load_config(path: str | Path) -> AtlasConfig:
    """Load and validate an ``AtlasConfig`` from a YAML file."""

    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, dict):
        raise ValueError(f"Config file {config_path} must contain a YAML mapping.")

    for section in _REQUIRED_SECTIONS:
        if section not in raw:
            raise ValueError(f"Missing required config section '{section}' in {config_path}.")

    experiment_raw = _section(raw, "experiment")
    exp_type = experiment_raw.get("type", "single_point")
    if exp_type not in _SUPPORTED["experiment_type"]:
        raise ValueError(
            f"Unknown experiment type '{exp_type}'. "
            f"Supported: {', '.join(sorted(_SUPPORTED['experiment_type']))}."
        )

    system_raw = _section(raw, "system")
    system_name = system_raw.get("name")
    if not system_name or system_name not in _SUPPORTED["system"]:
        raise ValueError(
            f"Unknown system '{system_name}'. Supported: {', '.join(sorted(_SUPPORTED['system']))}."
        )
    system_params = system_raw.get("parameters") or {}
    if not isinstance(system_params, dict):
        raise ValueError("system.parameters must be a mapping.")
    sweep = system_raw.get("sweep")
    if sweep is not None and not isinstance(sweep, dict):
        raise ValueError("system.sweep must be a mapping when present.")

    algorithm_name, algorithm_params = _named_section(
        raw, "algorithm", _SUPPORTED["algorithm"]
    )
    ansatz_name, ansatz_params = _named_section(raw, "ansatz", _SUPPORTED["ansatz"])
    optimizer_name, optimizer_params = _named_section(
        raw, "optimizer", _SUPPORTED["optimizer"]
    )
    backend_name, backend_params = _named_section(raw, "backend", _SUPPORTED["backend"])

    hardware_raw = _section(raw, "hardware")
    analysis_raw = _section(raw, "analysis")
    output_raw = _section(raw, "output")

    observables = analysis_raw.get("observables", "tfim_default")
    if observables not in _SUPPORTED["observables"]:
        raise ValueError(
            f"Unknown observables '{observables}'. "
            f"Supported: {', '.join(sorted(_SUPPORTED['observables']))}."
        )

    return AtlasConfig(
        experiment=ExperimentMetaConfig(
            type=exp_type,
            name=experiment_raw.get("name", ""),
        ),
        system=SystemConfig(
            name=system_name,
            parameters=dict(system_params),
            sweep=dict(sweep) if sweep else None,
        ),
        algorithm=AlgorithmConfig(name=algorithm_name, parameters=dict(algorithm_params)),
        ansatz=AnsatzConfig(name=ansatz_name, parameters=dict(ansatz_params)),
        optimizer=OptimizerSettingsConfig(
            name=optimizer_name, parameters=dict(optimizer_params)
        ),
        backend=BackendConfig(name=backend_name, parameters=dict(backend_params)),
        hardware=HardwareConfig(
            provider=hardware_raw.get("provider", "ibm_runtime"),
            enabled=bool(hardware_raw.get("enabled", False)),
            backend_name=hardware_raw.get("backend_name"),
            resilience_level=int(hardware_raw.get("resilience_level", 1)),
            optimization_level=int(hardware_raw.get("optimization_level", 3)),
            hardware_csv_path=hardware_raw.get("hardware_csv_path"),
        ),
        analysis=AnalysisConfig(
            observables=observables,
            fidelity=bool(analysis_raw.get("fidelity", True)),
        ),
        output=OutputConfig(
            output_dir=output_raw.get("directory", "atlas/data"),
            csv=bool(output_raw.get("csv", True)),
            plots=bool(output_raw.get("plots", True)),
        ),
    )
