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
    DYNAMICS_ALGORITHMS,
    ExperimentMetaConfig,
    HardwareConfig,
    InitialStateConfig,
    OptimizerSettingsConfig,
    OutputConfig,
    SystemConfig,
    VARIATIONAL_ALGORITHMS,
    is_dynamics_algorithm,
    is_variational_algorithm,
)

_REQUIRED_SECTIONS = (
    "experiment",
    "system",
    "algorithm",
    "backend",
    "hardware",
    "analysis",
    "output",
)

_SUPPORTED = {
    "experiment_type": {"single_point", "sweep", "trotter_validation"},
    "system": {"tfim"},
    "algorithm": VARIATIONAL_ALGORITHMS | DYNAMICS_ALGORITHMS,
    "ansatz": {"hardware_efficient"},
    "optimizer": {"scipy"},
    "backend": {"statevector", "statevector_evolver"},
    "observables": {"tfim_default"},
    "initial_state": {"computational"},
    "evolution_method": {"lie", "strang"},
    "sweep_parameter": {"h", "evolution_time", "num_trotter_steps"},
}


def _section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    if key not in raw:
        raise ValueError(f"Missing required config section '{key}'.")
    section = raw[key]
    if not isinstance(section, dict):
        raise ValueError(f"Config section '{key}' must be a mapping.")
    return section


def _optional_section(raw: dict[str, Any], key: str) -> dict[str, Any] | None:
    if key not in raw:
        return None
    section = raw[key]
    if section is None:
        return None
    if not isinstance(section, dict):
        raise ValueError(f"Config section '{key}' must be a mapping when present.")
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


def _validate_variational_sections(raw: dict[str, Any]) -> tuple[AnsatzConfig, OptimizerSettingsConfig]:
    ansatz_name, ansatz_params = _named_section(raw, "ansatz", _SUPPORTED["ansatz"])
    optimizer_name, optimizer_params = _named_section(
        raw, "optimizer", _SUPPORTED["optimizer"]
    )
    return (
        AnsatzConfig(name=ansatz_name, parameters=dict(ansatz_params)),
        OptimizerSettingsConfig(
            name=optimizer_name, parameters=dict(optimizer_params)
        ),
    )


def _validate_dynamics_sections(raw: dict[str, Any]) -> InitialStateConfig:
    initial_state_raw = _optional_section(raw, "initial_state")
    if initial_state_raw is None:
        return InitialStateConfig()

    name = initial_state_raw.get("name", "computational")
    if name not in _SUPPORTED["initial_state"]:
        raise ValueError(
            f"Unknown initial_state '{name}'. "
            f"Supported: {', '.join(sorted(_SUPPORTED['initial_state']))}."
        )
    parameters = initial_state_raw.get("parameters") or {}
    if not isinstance(parameters, dict):
        raise ValueError("initial_state.parameters must be a mapping.")
    return InitialStateConfig(name=name, parameters=dict(parameters))


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
    if sweep is not None:
        if not isinstance(sweep, dict):
            raise ValueError("system.sweep must be a mapping when present.")
        sweep_parameter = sweep.get("parameter", "h")
        if sweep_parameter not in _SUPPORTED["sweep_parameter"]:
            raise ValueError(
                f"Unknown sweep parameter '{sweep_parameter}'. "
                f"Supported: {', '.join(sorted(_SUPPORTED['sweep_parameter']))}."
            )

    algorithm_name, algorithm_params = _named_section(
        raw, "algorithm", _SUPPORTED["algorithm"]
    )

    if is_variational_algorithm(algorithm_name):
        if "ansatz" not in raw:
            raise ValueError(
                f"Algorithm '{algorithm_name}' requires an 'ansatz' config section."
            )
        if "optimizer" not in raw:
            raise ValueError(
                f"Algorithm '{algorithm_name}' requires an 'optimizer' config section."
            )
        ansatz_config, optimizer_config = _validate_variational_sections(raw)
    else:
        ansatz_config = AnsatzConfig()
        optimizer_config = OptimizerSettingsConfig()

    if is_dynamics_algorithm(algorithm_name):
        evolution_methods = algorithm_params.get("evolution_methods")
        if evolution_methods is not None:
            if not isinstance(evolution_methods, list) or not evolution_methods:
                raise ValueError(
                    "algorithm.parameters.evolution_methods must be a non-empty list."
                )
            for method in evolution_methods:
                if method not in _SUPPORTED["evolution_method"]:
                    raise ValueError(
                        f"Unknown evolution_method '{method}' in evolution_methods. "
                        f"Supported: {', '.join(sorted(_SUPPORTED['evolution_method']))}."
                    )
        else:
            evolution_method = algorithm_params.get("evolution_method", "strang")
            if evolution_method not in _SUPPORTED["evolution_method"]:
                raise ValueError(
                    f"Unknown evolution_method '{evolution_method}'. "
                    f"Supported: {', '.join(sorted(_SUPPORTED['evolution_method']))}."
                )
        if exp_type == "trotter_validation":
            step_values = algorithm_params.get("num_trotter_steps")
            if not isinstance(step_values, list) or not step_values:
                raise ValueError(
                    "Experiment type 'trotter_validation' requires "
                    "algorithm.parameters.num_trotter_steps as a non-empty list."
                )
        initial_state_config = _validate_dynamics_sections(raw)
    else:
        initial_state_config = None

    backend_name, backend_params = _named_section(raw, "backend", _SUPPORTED["backend"])
    if is_variational_algorithm(algorithm_name) and backend_name != "statevector":
        raise ValueError(
            f"Variational algorithm '{algorithm_name}' requires backend 'statevector'."
        )
    if is_dynamics_algorithm(algorithm_name) and backend_name != "statevector_evolver":
        raise ValueError(
            f"Dynamics algorithm '{algorithm_name}' requires backend 'statevector_evolver'."
        )

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
        ansatz=ansatz_config,
        optimizer=optimizer_config,
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
        initial_state=initial_state_config,
    )
