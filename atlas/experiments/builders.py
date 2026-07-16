"""Component builders from ``AtlasConfig``."""

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
from atlas.physics.observables import tfim_observables


_HAMILTONIAN_PARAM_KEYS = frozenset({"num_qubits", "J", "h"})


def build_hamiltonian(config: AtlasConfig, parameter_overrides: Optional[dict] = None):
    """Construct a Hamiltonian from ``config.system``."""

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
    """Construct an ansatz for the configured qubit count."""

    if config.ansatz.name == "hardware_efficient":
        reps = config.ansatz.parameters.get("reps", 1)
        return build_hardware_efficient_ansatz(num_qubits=num_qubits, reps=reps)

    raise ValueError(
        f"Unknown ansatz '{config.ansatz.name}'. Supported: hardware_efficient."
    )


def build_optimizer(config: AtlasConfig):
    """Construct the configured classical optimizer."""

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
    """Construct the configured estimator for variational cost evaluation."""

    if config.backend.name == "statevector":
        return SimulatorEstimator()

    raise ValueError(
        f"Unknown estimator backend '{config.backend.name}'. Supported: statevector."
    )


def build_evolver(config: AtlasConfig):
    """Construct the configured evolver for Hamiltonian simulation."""

    if config.backend.name == "statevector_evolver":
        return StatevectorEvolver()

    raise ValueError(
        f"Unknown evolver backend '{config.backend.name}'. "
        "Supported: statevector_evolver."
    )


def build_evolution_method_named(method_name: str, num_trotter_steps: int):
    """Construct a product-formula evolution method by explicit name and step count."""

    if method_name == "lie":
        return LieTrotter(num_trotter_steps=num_trotter_steps)
    if method_name == "strang":
        return StrangTrotter(num_trotter_steps=num_trotter_steps)
    raise ValueError(
        f"Unknown evolution_method '{method_name}'. Supported: lie, strang."
    )


def build_evolution_method(config: AtlasConfig, num_trotter_steps: Optional[int] = None):
    """Construct the configured product-formula evolution method."""

    params = config.algorithm.parameters
    method_name = params.get("evolution_method", "strang")
    if num_trotter_steps is not None:
        steps = num_trotter_steps
    else:
        raw_steps = params.get("num_trotter_steps", 10)
        if isinstance(raw_steps, list):
            steps = int(raw_steps[0])
        else:
            steps = int(raw_steps)
    return build_evolution_method_named(method_name, steps)


def resolve_evolution_methods(config: AtlasConfig) -> list[str]:
    """Return the evolution methods to compare during validation."""

    params = config.algorithm.parameters
    methods = params.get("evolution_methods")
    if methods:
        if not isinstance(methods, list) or not methods:
            raise ValueError("algorithm.parameters.evolution_methods must be a non-empty list.")
        return [str(name) for name in methods]
    return [str(params.get("evolution_method", "strang"))]


def resolve_trotter_step_values(config: AtlasConfig) -> list[int]:
    """Return Trotter step counts for validation or sweeps."""

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
    """Construct the configured initial state for dynamics experiments."""

    if config.initial_state is None or config.initial_state.name == "computational":
        params = {} if config.initial_state is None else config.initial_state.parameters
        bitstring = params.get("bitstring", "0" * num_qubits)
        return ComputationalBasisState(num_qubits=num_qubits, bitstring=bitstring)

    raise ValueError(
        f"Unknown initial_state '{config.initial_state.name}'. Supported: computational."
    )


def build_vqe(config: AtlasConfig, estimator, optimizer) -> VQE:
    """Build a ``VQE`` instance from config."""

    seed = config.backend.parameters.get("seed")
    num_starts = config.optimizer.parameters.get("num_starts", 10)
    return VQE(
        estimator=estimator,
        optimizer=optimizer,
        num_starts=num_starts,
        seed=seed,
    )


def build_variational_algorithm(config: AtlasConfig, estimator, optimizer):
    """Build ``VQE`` or ``VQD`` (which wraps ``VQE`` internally)."""

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
    """Build a ``HamiltonianSimulation`` instance from config."""

    return HamiltonianSimulation(
        evolution_method=evolution_method,
        evolver=evolver,
    )


def build_algorithm(config: AtlasConfig, estimator=None, optimizer=None, evolver=None):
    """Dispatch algorithm construction based on the configured algorithm name."""

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
    """Return observable specs for the configured analysis."""

    if config.analysis.observables == "tfim_default":
        return tfim_observables(num_qubits)

    raise ValueError(
        f"Unknown observables '{config.analysis.observables}'. "
        "Supported: tfim_default."
    )


def resolve_evolution_time(config: AtlasConfig) -> float:
    """Return the evolution time for a single-point dynamics experiment."""

    system_time = config.system.parameters.get("evolution_time")
    if system_time is not None:
        return float(system_time)
    return float(config.algorithm.parameters.get("evolution_time", 1.0))
