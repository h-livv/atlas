"""Load Atlas experiment configuration from YAML files.

This module is the only place that interprets raw YAML into typed
``AtlasConfig`` objects. It validates required sections, allow-listed
component names, and algorithm-specific requirements (ansatz for VQE,
evolver backend for dynamics, etc.) before any experiment code runs.
"""

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

# Every experiment YAML must declare these top-level keys so builders never
# see a partially specified config.
_REQUIRED_SECTIONS = (
    "experiment",
    "system",
    "algorithm",
    "backend",
    "hardware",
    "analysis",
    "output",
)

# Allow-lists keep unsupported names failing fast with a clear message.
# Builders re-check the same names when constructing objects.
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
    """Return a required mapping section from the raw YAML root.

    Purpose:
        Fail early when a mandatory section is missing or mistyped.

    Inputs:
        raw: Top-level YAML mapping.
        key: Section name to extract.

    Process:
        Look up ``key``; verify the value is a ``dict``.

    Outputs:
        The section mapping.

    Side effects:
        None (raises ``ValueError`` on invalid input).
    """

    if key not in raw:
        raise ValueError(f"Missing required config section '{key}'.")
    section = raw[key]
    if not isinstance(section, dict):
        raise ValueError(f"Config section '{key}' must be a mapping.")
    return section


def _optional_section(raw: dict[str, Any], key: str) -> dict[str, Any] | None:
    """Return an optional mapping section, or ``None`` if absent.

    Purpose:
        Support sections that only some algorithm families need (e.g.
        ``initial_state`` for dynamics).

    Inputs:
        raw: Top-level YAML mapping.
        key: Section name that may be omitted.

    Process:
        If missing or explicitly ``null``, return ``None``; otherwise require
        a mapping.

    Outputs:
        The section dict, or ``None``.

    Side effects:
        None (raises ``ValueError`` if present but not a mapping).
    """

    if key not in raw:
        return None
    section = raw[key]
    if section is None:
        return None
    if not isinstance(section, dict):
        raise ValueError(f"Config section '{key}' must be a mapping when present.")
    return section


def _named_section(raw: dict[str, Any], key: str, supported: set[str]) -> tuple[str, dict]:
    """Extract ``name`` and ``parameters`` from a named component section.

    Purpose:
        Most Atlas components are selected by a string name plus a free-form
        parameters dict; this helper standardizes that pattern.

    Inputs:
        raw: Top-level YAML mapping.
        key: Section name (e.g. ``\"algorithm\"``).
        supported: Allow-list of valid ``name`` values.

    Process:
        Require the section, require ``name`` ∈ ``supported``, coerce missing
        ``parameters`` to ``{}``, and verify ``parameters`` is a mapping.

    Outputs:
        ``(name, parameters_dict)``.

    Side effects:
        None (raises ``ValueError`` on invalid input).
    """

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
    """Validate and build ansatz/optimizer configs for VQE/VQD.

    Purpose:
        Variational algorithms need a parameterized circuit and a classical
        optimizer; this packages those YAML sections into typed configs.

    Inputs:
        raw: Top-level YAML mapping that must contain ``ansatz`` and
            ``optimizer`` sections.

    Process:
        Parse each named section against the allow-lists and construct
        dataclass instances (copying parameters so later mutation is safe).

    Outputs:
        ``(AnsatzConfig, OptimizerSettingsConfig)``.

    Side effects:
        None.
    """

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
    """Validate and build the initial-state config for dynamics experiments.

    Purpose:
        Hamiltonian simulation needs a pure starting state; default to the
        computational all-zero state when the YAML omits the section.

    Inputs:
        raw: Top-level YAML mapping.

    Process:
        If ``initial_state`` is absent, return defaults. Otherwise validate
        ``name`` against the allow-list and copy ``parameters``.

    Outputs:
        An ``InitialStateConfig``.

    Side effects:
        None.
    """

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
    """Load and validate an ``AtlasConfig`` from a YAML file.

    Purpose:
        Turn a human-edited experiment description into the typed object the
        rest of Atlas consumes, rejecting unsupported combinations early.

    Inputs:
        path: Filesystem path to a YAML experiment file.

    Process:
        1. Parse YAML into a mapping.
        2. Ensure all required top-level sections exist.
        3. Validate experiment type, system, and algorithm names.
        4. Branch: variational requires ansatz/optimizer; dynamics validates
           evolution method(s), Trotter step lists for validation runs, and
           initial state.
        5. Enforce backend pairing (VQE/VQD ↔ ``statevector``, dynamics ↔
           ``statevector_evolver``).
        6. Validate hardware/analysis/output fields and assemble ``AtlasConfig``.

    Outputs:
        A fully populated ``AtlasConfig`` (intent only; no computed results).

    Side effects:
        Reads the YAML file from disk. Does not create output directories.
    """

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

    # Variational paths need circuit + classical optimizer sections.
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
        # Dynamics configs omit these; keep dataclass defaults for a complete AtlasConfig.
        ansatz_config = AnsatzConfig()
        optimizer_config = OptimizerSettingsConfig()

    if is_dynamics_algorithm(algorithm_name):
        # Prefer an explicit method list (validation) over a single method name.
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
        # Trotter validation compares accuracy across a step grid; require that grid here.
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
    # Estimators evaluate parameterized costs; evolvers return full statevectors.
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
            # YAML uses "directory"; the dataclass field is output_dir.
            output_dir=output_raw.get("directory", "atlas/data"),
            csv=bool(output_raw.get("csv", True)),
            plots=bool(output_raw.get("plots", True)),
        ),
        initial_state=initial_state_config,
    )
