"""Configuration objects for Atlas experiments and execution.

These dataclasses hold user intent and runtime settings only -- never
computed results. YAML loading via ``atlas.io.yaml_config`` produces
an ``AtlasConfig``; legacy ``ExperimentConfig`` remains for compatibility.

Architectural role:
    Config objects are the single source of truth for *what* to run. Builders
    and factory code translate these fields into concrete Hamiltonians,
    algorithms, and backends. Result containers live in ``experiments.results``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence


# ---------------------------------------------------------------------------
# Legacy dataclasses (retained for backward compatibility)
# ---------------------------------------------------------------------------


@dataclass
class TFIMConfig:
    """Physical model parameters for the transverse field Ising model.

    Responsibility:
        Hold TFIM sizes and couplings for the older nested config layout.

    State:
        num_qubits: Chain length.
        J: Ising coupling strength.
        h: Transverse field strength (single-point default).
        h_values: Optional list of field values for sweeps.

    Usage:
        Nested under ``ExperimentConfig.tfim``. The YAML path uses
        ``SystemConfig`` instead; this class is kept for compatibility.
    """

    num_qubits: int = 2
    J: float = 1.0
    h: float = 0.5
    h_values: Optional[Sequence[float]] = None


@dataclass
class OptimizerConfig:
    """Classical optimizer settings for variational algorithms.

    Responsibility:
        Capture SciPy-style method limits for the legacy config tree.

    State:
        method: SciPy optimizer name (e.g. ``COBYLA``).
        maxiter: Maximum optimizer iterations per start.
        num_starts: How many random initial points VQE should try.

    Usage:
        Nested under ``ExperimentConfig.optimizer``. YAML uses
        ``OptimizerSettingsConfig`` with a ``parameters`` dict.
    """

    method: str = "COBYLA"
    maxiter: int = 200
    num_starts: int = 10


@dataclass
class SimulatorConfig:
    """Local statevector simulator settings.

    Responsibility:
        Hold reproducibility knobs for the legacy simulator section.

    State:
        seed: Optional RNG seed; ``None`` means unreproducible draws.

    Usage:
        Nested under ``ExperimentConfig.simulator``. YAML places the seed under
        ``backend.parameters.seed``.
    """

    seed: Optional[int] = None


@dataclass
class HardwareConfig:
    """IBM Runtime backend selection and execution settings.

    Responsibility:
        Describe whether and how to evaluate optimized circuits on hardware
        (or merge previously recorded hardware CSV results).

    State:
        provider: Backend provider identifier (currently ``ibm_runtime``).
        enabled: When ``True``, experiments attempt live hardware evaluation.
        backend_name: Optional fixed backend; otherwise least-busy is used.
        resilience_level: EstimatorV2 resilience option.
        optimization_level: Transpiler optimization level (stored for config
            completeness; the IBM evaluator currently hard-codes level 3).
        hardware_csv_path: Optional path to merge offline hardware results.

    Usage:
        Shared by both legacy ``ExperimentConfig`` and YAML ``AtlasConfig``.
    """

    provider: str = "ibm_runtime"
    enabled: bool = False
    backend_name: Optional[str] = None
    resilience_level: int = 1
    optimization_level: int = 3
    hardware_csv_path: Optional[str] = None


@dataclass
class OutputConfig:
    """Output locations and toggles for plots and benchmark CSVs.

    Responsibility:
        Control where artifacts are written and which artifact kinds are enabled.

    State:
        output_dir: Base directory; each run creates a timestamped subdirectory.
        csv: Whether to write benchmark CSV files.
        plots: Whether to write PNG plots.

    Usage:
        Shared by legacy and YAML configs. YAML key ``directory`` maps here
        as ``output_dir`` during load.
    """

    output_dir: str = "atlas/data"
    csv: bool = True
    plots: bool = True


@dataclass
class ExperimentConfig:
    """Legacy top-level configuration combining all sub-configs.

    Responsibility:
        Aggregate the older nested config objects into one root.

    State:
        Nested ``TFIMConfig``, ``OptimizerConfig``, ``SimulatorConfig``,
        ``HardwareConfig``, and ``OutputConfig`` instances.

    Usage:
        Constructed by ``default_tfim_config()``. Not produced by the current
        YAML loader, which returns ``AtlasConfig`` instead.
    """

    tfim: TFIMConfig = field(default_factory=TFIMConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    simulator: SimulatorConfig = field(default_factory=SimulatorConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def default_tfim_config() -> ExperimentConfig:
    """Return an ``ExperimentConfig`` matching legacy default behavior.

    Purpose:
        Provide a convenient default for older call sites.

    Inputs:
        None.

    Process:
        Instantiate ``ExperimentConfig`` with all nested dataclass defaults.

    Outputs:
        A fully populated legacy ``ExperimentConfig``.

    Side effects:
        None.
    """

    return ExperimentConfig()


# ---------------------------------------------------------------------------
# YAML-driven generic configuration
# ---------------------------------------------------------------------------


@dataclass
class ExperimentMetaConfig:
    """Experiment type and display name.

    Responsibility:
        Tell the factory *how* to run (single point, sweep, validation) and
        what label to use for output directories.

    State:
        type: ``single_point``, ``sweep``, or ``trotter_validation``.
        name: Human-readable run label (used in output folder names).

    Usage:
        Populated from the YAML ``experiment`` section.
    """

    type: str = "single_point"
    name: str = ""


@dataclass
class SystemConfig:
    """Physical system selection and parameters.

    Responsibility:
        Identify which Hamiltonian model to build and supply its parameters
        (and optional sweep definition).

    State:
        name: System identifier (currently only ``tfim``).
        parameters: Dict of model parameters such as ``num_qubits``, ``J``, ``h``.
        sweep: Optional ``{parameter, values}`` mapping for sweep experiments.

    Usage:
        Consumed by builders when constructing Hamiltonians and by the factory
        when iterating sweep values.
    """

    name: str = "tfim"
    parameters: dict[str, Any] = field(default_factory=dict)
    sweep: Optional[dict[str, Any]] = None


@dataclass
class AlgorithmConfig:
    """Algorithm selection (vqe, vqd, hamiltonian_sim, ...).

    Responsibility:
        Choose the quantum method and carry algorithm-specific knobs
        (e.g. VQD ``num_states``, Trotter ``evolution_method``).

    State:
        name: Algorithm identifier.
        parameters: Free-form dict interpreted by builders/experiments.

    Usage:
        Factory uses ``name`` for variational vs dynamics dispatch; builders
        read ``parameters`` when constructing concrete algorithm objects.
    """

    name: str = "vqe"
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnsatzConfig:
    """Parameterized circuit family selection.

    Responsibility:
        Describe which variational ansatz to build and with which options.

    State:
        name: Ansatz family (currently ``hardware_efficient``).
        parameters: Options such as ``reps``.

    Usage:
        Required for variational algorithms; builders ignore it for dynamics.
    """

    name: str = "hardware_efficient"
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizerSettingsConfig:
    """Named optimizer backend and its parameters.

    Responsibility:
        Select the classical optimizer implementation used by VQE/VQD.

    State:
        name: Optimizer backend (currently ``scipy``).
        parameters: Options such as ``method``, ``maxiter``, ``num_starts``.

    Usage:
        Required for variational algorithms; unused on the dynamics path.
    """

    name: str = "scipy"
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class BackendConfig:
    """Simulator or execution backend for algorithm cost evaluation.

    Responsibility:
        Select the local execution primitive (estimator vs evolver).

    State:
        name: ``statevector`` (variational) or ``statevector_evolver`` (dynamics).
        parameters: Backend options such as RNG ``seed``.

    Usage:
        Builders construct ``SimulatorEstimator`` or ``StatevectorEvolver``.
        YAML validation enforces the algorithm/backend pairing.
    """

    name: str = "statevector"
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class InitialStateConfig:
    """Initial state selection for Hamiltonian simulation.

    Responsibility:
        Describe the pure state that time evolution starts from.

    State:
        name: Preparation family (currently ``computational``).
        parameters: Options such as ``bitstring``.

    Usage:
        Optional on ``AtlasConfig``; dynamics configs populate it, variational
        configs leave it ``None``.
    """

    name: str = "computational"
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisConfig:
    """Post-run analysis options.

    Responsibility:
        Control which observables and fidelity checks experiments compute
        after the quantum method finishes.

    State:
        observables: Named observable set (currently ``tfim_default``).
        fidelity: When ``True``, compare against exact reference states.

    Usage:
        Experiment classes read these flags while packaging point results.
    """

    observables: str = "tfim_default"
    fidelity: bool = True


@dataclass
class AtlasConfig:
    """Top-level YAML-driven experiment configuration.

    Responsibility:
        Aggregate every section the factory needs to build and run an
        experiment. This is the object returned by ``load_config``.

    State:
        Nested section configs for experiment metadata, system, algorithm,
        ansatz, optimizer, backend, hardware, analysis, and output. Dynamics
        runs may also set ``initial_state``.

    Usage:
        Pass to ``run_experiment`` / ``save_outputs``. Treat as immutable
        intent: do not store computed energies here.
    """

    experiment: ExperimentMetaConfig = field(default_factory=ExperimentMetaConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    ansatz: AnsatzConfig = field(default_factory=AnsatzConfig)
    optimizer: OptimizerSettingsConfig = field(default_factory=OptimizerSettingsConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    initial_state: Optional[InitialStateConfig] = None


# Algorithm families drive validation (YAML) and wiring (factory/builders).
VARIATIONAL_ALGORITHMS = frozenset({"vqe", "vqd"})
DYNAMICS_ALGORITHMS = frozenset({"hamiltonian_sim"})


def is_variational_algorithm(name: str) -> bool:
    """Return whether ``name`` is a variational algorithm (VQE/VQD).

    Purpose:
        Centralize family membership so validation and wiring stay consistent.

    Inputs:
        name: Algorithm identifier from config (e.g. ``\"vqe\"``).

    Process:
        Membership test against ``VARIATIONAL_ALGORITHMS``.

    Outputs:
        ``True`` if the algorithm needs an ansatz, optimizer, and estimator.

    Side effects:
        None.
    """

    return name in VARIATIONAL_ALGORITHMS


def is_dynamics_algorithm(name: str) -> bool:
    """Return whether ``name`` is a dynamics / Hamiltonian-simulation algorithm.

    Purpose:
        Centralize family membership for evolution-method wiring.

    Inputs:
        name: Algorithm identifier from config (e.g. ``\"hamiltonian_sim\"``).

    Process:
        Membership test against ``DYNAMICS_ALGORITHMS``.

    Outputs:
        ``True`` if the algorithm needs an initial state and evolver.

    Side effects:
        None.
    """

    return name in DYNAMICS_ALGORITHMS
