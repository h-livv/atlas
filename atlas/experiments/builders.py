"""Component builders from ``AtlasConfig``.

This module is the config-to-object layer for Atlas experiments. Each
``build_*`` function reads a slice of ``AtlasConfig`` and constructs the
corresponding runtime object (Hamiltonian, ansatz, optimizer, estimator,
evolver, algorithm, observables, or initial state). Resolve helpers extract
lists or scalars used by sweeps and Trotter validation without constructing
full algorithm instances.

Architectural role:
    Factory code and experiment workflows call these builders; builders may
    import physics, circuits, algorithms, and execution modules, but they do
    not run optimizations or write outputs themselves.
"""

from __future__ import annotations

from typing import Optional

from atlas.algorithms.hamiltonian_sim import HamiltonianSimulation
from atlas.algorithms.vqe import VQE
from atlas.algorithms.vqd import VQD
from atlas.circuits.ansatzes.hardware_efficient import build_hardware_efficient_ansatz
from atlas.circuits.evolution.product_formulas import LieTrotter, StrangTrotter
from atlas.circuits.initial_states import ComputationalBasisState
from atlas.config import AtlasConfig, is_dynamics_algorithm, is_variational_algorithm
from atlas.execution.estimator import SimulatorEstimator
from atlas.execution.evolver import StatevectorEvolver
from atlas.optimization.scipy_optimizer import ScipyOptimizer
from atlas.physics.hamiltonians import TFIMHamiltonian
from atlas.physics.observables import local_pauli_observables, tfim_observables


# Only these keys are forwarded into TFIMHamiltonian; other system parameters
# (e.g. evolution_time) are handled by dynamics helpers, not the Hamiltonian ctor.
_HAMILTONIAN_PARAM_KEYS = frozenset({"num_qubits", "J", "h"})


def build_hamiltonian(config: AtlasConfig, parameter_overrides: Optional[dict] = None):
    """Construct a Hamiltonian from ``config.system``.

    Purpose:
        Map the configured physical system name and parameters onto a concrete
        ``Hamiltonian`` subclass instance.

    Inputs:
        config: Full Atlas configuration; uses ``config.system``.
        parameter_overrides: Optional dict merged on top of
            ``config.system.parameters`` (e.g. a sweep's current ``J``/``h``).

    Process:
        Copy system parameters, apply overrides, then dispatch on
        ``config.system.name``. For TFIM, only Hamiltonian constructor keys
        are forwarded.

    Outputs:
        A Hamiltonian instance (currently ``TFIMHamiltonian``).

    Side effects:
        None. Raises ``ValueError`` for unsupported system names.
    """

    params = dict(config.system.parameters)
    if parameter_overrides:
        params.update(parameter_overrides)

    if config.system.name == "tfim":
        hamiltonian_params = {
            key: params[key] for key in _HAMILTONIAN_PARAM_KEYS if key in params
        }
        return TFIMHamiltonian(**hamiltonian_params)

    raise ValueError(
        f"Unknown system '{config.system.name}'. Supported: tfim."
    )


def build_ansatz(config: AtlasConfig, num_qubits: int):
    """Construct an ansatz for the configured qubit count.

    Purpose:
        Produce the variational circuit template used by VQE/VQD.

    Inputs:
        config: Uses ``config.ansatz.name`` and ``config.ansatz.parameters``.
        num_qubits: Width of the ansatz circuit.

    Process:
        Dispatch on ansatz name; for ``hardware_efficient``, read ``reps``
        (default 1) and call the circuit builder.

    Outputs:
        An ansatz object with ``.circuit``, ``.bind``, and ``.num_parameters``.

    Side effects:
        None. Raises ``ValueError`` for unknown ansatz names.
    """

    if config.ansatz.name == "hardware_efficient":
        reps = config.ansatz.parameters.get("reps", 1)
        return build_hardware_efficient_ansatz(num_qubits=num_qubits, reps=reps)

    raise ValueError(
        f"Unknown ansatz '{config.ansatz.name}'. Supported: hardware_efficient."
    )


def build_optimizer(config: AtlasConfig):
    """Construct the configured classical optimizer.

    Purpose:
        Wrap SciPy (or future) optimizers with Atlas's minimize interface.

    Inputs:
        config: Uses ``config.optimizer.name`` and ``.parameters``.

    Process:
        For ``scipy``, read ``method`` (default COBYLA) and ``maxiter``
        (default 200).

    Outputs:
        An optimizer instance (currently ``ScipyOptimizer``).

    Side effects:
        None. Raises ``ValueError`` for unknown optimizer names.
    """

    if config.optimizer.name == "scipy":
        params = config.optimizer.parameters
        return ScipyOptimizer(
            method=params.get("method", "COBYLA"),
            maxiter=int(params.get("maxiter", 200)),
        )

    raise ValueError(
        f"Unknown optimizer '{config.optimizer.name}'. Supported: scipy."
    )


def build_estimator(config: AtlasConfig):
    """Construct the configured estimator for variational cost evaluation.

    Purpose:
        Provide the backend that evaluates ⟨ψ(θ)|H|ψ(θ)⟩ during VQE/VQD.

    Inputs:
        config: Uses ``config.backend.name``.

    Process:
        Dispatch on backend name; ``statevector`` yields a local simulator
        estimator.

    Outputs:
        An estimator with ``.expectation(...)``.

    Side effects:
        None. Raises ``ValueError`` for unsupported backends.
    """

    if config.backend.name == "statevector":
        return SimulatorEstimator()

    raise ValueError(
        f"Unknown estimator backend '{config.backend.name}'. Supported: statevector."
    )


def build_evolver(config: AtlasConfig):
    """Construct the configured evolver for Hamiltonian simulation.

    Purpose:
        Provide the backend that applies a parameter-free evolution circuit
        to an initial statevector.

    Inputs:
        config: Uses ``config.backend.name``.

    Process:
        Dispatch on backend name; ``statevector_evolver`` yields local
        statevector evolution.

    Outputs:
        An evolver instance (currently ``StatevectorEvolver``).

    Side effects:
        None. Raises ``ValueError`` for unsupported backends.
    """

    if config.backend.name == "statevector_evolver":
        return StatevectorEvolver()

    raise ValueError(
        f"Unknown evolver backend '{config.backend.name}'. "
        "Supported: statevector_evolver."
    )


def build_evolution_method_named(method_name: str, num_trotter_steps: int):
    """Construct a product-formula evolution method by explicit name and step count.

    Purpose:
        Build Lie or Strang Trotter objects when the caller already knows the
        method name and step count (e.g. validation sweeps).

    Inputs:
        method_name: ``"lie"`` or ``"strang"``.
        num_trotter_steps: Number of Trotter repetitions.

    Process:
        Dispatch on ``method_name`` and pass ``num_trotter_steps`` through.

    Outputs:
        A product-formula evolution method instance.

    Side effects:
        None. Raises ``ValueError`` for unknown method names.
    """

    if method_name == "lie":
        return LieTrotter(num_trotter_steps=num_trotter_steps)
    if method_name == "strang":
        return StrangTrotter(num_trotter_steps=num_trotter_steps)
    raise ValueError(
        f"Unknown evolution_method '{method_name}'. Supported: lie, strang."
    )


def build_evolution_method(config: AtlasConfig, num_trotter_steps: Optional[int] = None):
    """Construct the configured product-formula evolution method.

    Purpose:
        Read evolution settings from algorithm parameters and build one
        Trotter method for a single dynamics run.

    Inputs:
        config: Uses ``config.algorithm.parameters``
            (``evolution_method``, ``num_trotter_steps``).
        num_trotter_steps: Optional override; when set, ignores the config
            step count (including list form).

    Process:
        Resolve method name (default ``strang``). Resolve steps from the
        override, or from config: if ``num_trotter_steps`` is a list (as in
        validation configs), use the first entry for a single-point build.

    Outputs:
        A product-formula evolution method from ``build_evolution_method_named``.

    Side effects:
        None.
    """

    params = config.algorithm.parameters
    method_name = params.get("evolution_method", "strang")
    if num_trotter_steps is not None:
        steps = num_trotter_steps
    else:
        raw_steps = params.get("num_trotter_steps", 10)
        # Validation YAMLs store a list of steps; a single-point build needs one int.
        if isinstance(raw_steps, list):
            steps = int(raw_steps[0])
        else:
            steps = int(raw_steps)
    return build_evolution_method_named(method_name, steps)


def resolve_evolution_methods(config: AtlasConfig) -> list[str]:
    """Return the evolution methods to compare during validation.

    Purpose:
        Normalize config into an explicit list of method names for
        ``trotter_validation`` experiments.

    Inputs:
        config: Uses ``algorithm.parameters.evolution_methods`` or
            ``evolution_method``.

    Process:
        Prefer ``evolution_methods`` when present (must be a non-empty list);
        otherwise wrap the single ``evolution_method`` (default ``strang``).

    Outputs:
        List of method name strings.

    Side effects:
        None. Raises ``ValueError`` if ``evolution_methods`` is present but
        not a non-empty list.
    """

    params = config.algorithm.parameters
    methods = params.get("evolution_methods")
    if methods:
        if not isinstance(methods, list) or not methods:
            raise ValueError("algorithm.parameters.evolution_methods must be a non-empty list.")
        return [str(name) for name in methods]
    return [str(params.get("evolution_method", "strang"))]


def resolve_trotter_step_values(config: AtlasConfig) -> list[int]:
    """Return Trotter step counts for validation or sweeps.

    Purpose:
        Collect the discrete step grid used by Trotter validation or a
        ``num_trotter_steps`` sweep.

    Inputs:
        config: Inspects ``algorithm.parameters.num_trotter_steps`` and,
            as a fallback, ``system.sweep`` when it targets that parameter.

    Process:
        If ``num_trotter_steps`` is a list, return it as ints. If it is a
        scalar, wrap it. Else, if a system sweep is defined on
        ``num_trotter_steps``, return those values. Else default to ``[10]``.

    Outputs:
        Non-empty list of integer step counts.

    Side effects:
        None.
    """

    params = config.algorithm.parameters
    step_values = params.get("num_trotter_steps")
    if isinstance(step_values, list):
        return [int(value) for value in step_values]
    if step_values is not None:
        return [int(step_values)]

    sweep = config.system.sweep
    if sweep is not None and sweep.get("parameter") == "num_trotter_steps":
        return [int(value) for value in sweep["values"]]

    return [int(params.get("num_trotter_steps", 10))]


def build_initial_state(config: AtlasConfig, num_qubits: int):
    """Construct the configured initial state for dynamics experiments.

    Purpose:
        Provide the starting statevector / circuit for Hamiltonian simulation.

    Inputs:
        config: Uses ``config.initial_state`` (optional; defaults to
            computational |0…0⟩).
        num_qubits: System width.

    Process:
        For missing or ``computational`` initial state, read ``bitstring``
        (default all zeros) and build ``ComputationalBasisState``.

    Outputs:
        An initial-state object exposing ``.statevector()``.

    Side effects:
        None. Raises ``ValueError`` for unknown initial-state names.
    """

    if config.initial_state is None or config.initial_state.name == "computational":
        params = {} if config.initial_state is None else config.initial_state.parameters
        bitstring = params.get("bitstring", "0" * num_qubits)
        return ComputationalBasisState(num_qubits=num_qubits, bitstring=bitstring)

    raise ValueError(
        f"Unknown initial_state '{config.initial_state.name}'. Supported: computational."
    )


def build_vqe(config: AtlasConfig, estimator, optimizer) -> VQE:
    """Build a ``VQE`` instance from config.

    Purpose:
        Wire estimator/optimizer with multi-start and optional RNG seed.

    Inputs:
        config: Reads ``backend.parameters.seed`` and
            ``optimizer.parameters.num_starts`` (default 10).
        estimator: Cost-evaluation backend.
        optimizer: Classical minimizer.

    Process:
        Extract seed and num_starts, then construct ``VQE``.

    Outputs:
        A configured ``VQE`` instance.

    Side effects:
        None.
    """

    seed = config.backend.parameters.get("seed")
    num_starts = config.optimizer.parameters.get("num_starts", 10)
    return VQE(
        estimator=estimator,
        optimizer=optimizer,
        num_starts=num_starts,
        seed=seed,
    )


def build_variational_algorithm(config: AtlasConfig, estimator, optimizer):
    """Build ``VQE`` or ``VQD`` (which wraps ``VQE`` internally).

    Purpose:
        Select the variational algorithm named in config and inject shared
        estimator/optimizer dependencies.

    Inputs:
        config: Uses ``algorithm.name`` and VQD parameters when applicable.
        estimator: Shared cost evaluator.
        optimizer: Shared classical optimizer.

    Process:
        Always build an inner ``VQE``. Return it for ``vqe``; wrap it in
        ``VQD`` for ``vqd`` using ``num_states`` and ``beta``.

    Outputs:
        A ``VQE`` or ``VQD`` instance.

    Side effects:
        None. Raises ``ValueError`` for unknown variational algorithm names.
    """

    vqe = build_vqe(config, estimator, optimizer)

    if config.algorithm.name == "vqe":
        return vqe

    if config.algorithm.name == "vqd":
        params = config.algorithm.parameters
        return VQD(
            vqe=vqe,
            num_states=int(params.get("num_states", 2)),
            beta=float(params.get("beta", 1.0)),
        )

    raise ValueError(
        f"Unknown variational algorithm '{config.algorithm.name}'. Supported: vqe, vqd."
    )


def build_hamiltonian_simulation(
    config: AtlasConfig,
    evolver: StatevectorEvolver,
    evolution_method,
) -> HamiltonianSimulation:
    """Build a ``HamiltonianSimulation`` instance from config.

    Purpose:
        Pair a product-formula method with an evolver into the dynamics
        algorithm object. ``config`` is accepted for API symmetry with other
        builders; construction currently uses only the explicit arguments.

    Inputs:
        config: Present for consistent builder signatures (unused today).
        evolver: Backend that evolves statevectors under a circuit.
        evolution_method: Lie/Strang (or compatible) product formula.

    Process:
        Construct ``HamiltonianSimulation`` with the given method and evolver.

    Outputs:
        A ``HamiltonianSimulation`` instance.

    Side effects:
        None.
    """

    return HamiltonianSimulation(
        evolution_method=evolution_method,
        evolver=evolver,
    )


def build_algorithm(config: AtlasConfig, estimator=None, optimizer=None, evolver=None):
    """Dispatch algorithm construction based on the configured algorithm name.

    Purpose:
        Single entry point that chooses variational vs dynamics wiring.

    Inputs:
        config: Uses ``algorithm.name`` and helpers
            ``is_variational_algorithm`` / ``is_dynamics_algorithm``.
        estimator: Required for variational algorithms.
        optimizer: Required for variational algorithms.
        evolver: Required for dynamics algorithms.

    Process:
        For variational algorithms, require estimator+optimizer and call
        ``build_variational_algorithm``. For dynamics, require evolver, build
        the evolution method from config, then ``build_hamiltonian_simulation``.

    Outputs:
        A runnable algorithm instance (``VQE``, ``VQD``, or
        ``HamiltonianSimulation``).

    Side effects:
        None. Raises ``ValueError`` if required dependencies are missing or
        the algorithm name is unknown.
    """

    if is_variational_algorithm(config.algorithm.name):
        if estimator is None or optimizer is None:
            raise ValueError(
                "Variational algorithms require both estimator and optimizer."
            )
        return build_variational_algorithm(config, estimator, optimizer)

    if is_dynamics_algorithm(config.algorithm.name):
        if evolver is None:
            raise ValueError("Dynamics algorithms require an evolver.")
        evolution_method = build_evolution_method(config)
        return build_hamiltonian_simulation(config, evolver, evolution_method)

    raise ValueError(f"Unknown algorithm '{config.algorithm.name}'.")


def build_observables(config: AtlasConfig, num_qubits: int):
    """Return observable specs for the configured analysis.

    Purpose:
        Attach named Pauli observables used for post-run metrics and plots.

    Inputs:
        config: Uses ``config.analysis.observables``.
        num_qubits: System width for observable construction.

    Process:
        Dispatch on the observables preset name (currently ``tfim_default``).

    Outputs:
        A list of ``ObservableSpec`` (or equivalent) objects.

    Side effects:
        None. Raises ``ValueError`` for unknown presets.
    """

    if config.analysis.observables == "tfim_default":
        return tfim_observables(num_qubits)

    raise ValueError(
        f"Unknown observables '{config.analysis.observables}'. "
        "Supported: tfim_default."
    )


def build_site_observables(config: AtlasConfig, num_qubits: int):
    """Return grouped per-site observable specs for lattice visualization.

    Purpose:
        Produce named collections of local Pauli operators whose expectations
        become arrays on ``SimPointResult.site_observables``.

    Inputs:
        config: Uses ``config.analysis.site_observables`` (optional preset).
        num_qubits: System width.

    Process:
        Dispatch on the site-observables preset. ``local_z`` / ``local_x`` /
        ``local_y`` build one Pauli per site under series name ``z`` / ``x`` /
        ``y``.

    Outputs:
        Dict mapping series name → list of ``ObservableSpec`` ordered by site,
        or an empty dict when site observables are disabled.

    Side effects:
        None. Raises ``ValueError`` for unknown presets.
    """

    preset = getattr(config.analysis, "site_observables", None)
    if not preset:
        return {}

    if preset == "local_z":
        return {"z": local_pauli_observables(num_qubits, "Z", prefix="z")}
    if preset == "local_x":
        return {"x": local_pauli_observables(num_qubits, "X", prefix="x")}
    if preset == "local_y":
        return {"y": local_pauli_observables(num_qubits, "Y", prefix="y")}

    raise ValueError(
        f"Unknown site_observables '{preset}'. "
        "Supported: local_z, local_x, local_y."
    )


def resolve_evolution_time(config: AtlasConfig) -> float:
    """Return the evolution time for a single-point dynamics experiment.

    Purpose:
        Centralize the precedence rule for ``evolution_time`` so factory and
        experiment code stay consistent.

    Inputs:
        config: Checks ``system.parameters.evolution_time`` first, then
            ``algorithm.parameters.evolution_time`` (default 1.0).

    Process:
        Prefer the system-level value when present; otherwise fall back to
        the algorithm parameter (or 1.0).

    Outputs:
        Evolution time as ``float``.

    Side effects:
        None.
    """

    system_time = config.system.parameters.get("evolution_time")
    if system_time is not None:
        return float(system_time)
    return float(config.algorithm.parameters.get("evolution_time", 1.0))
