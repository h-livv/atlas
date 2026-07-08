"""Component builders from ``AtlasConfig``."""

from __future__ import annotations

from typing import Optional

from atlas.algorithms.vqe import VQE
from atlas.algorithms.vqd import VQD
from atlas.circuits.ansatzes.hardware_efficient import build_hardware_efficient_ansatz
from atlas.config import AtlasConfig
from atlas.execution.simulator import SimulatorEstimator
from atlas.optimization.scipy_optimizer import ScipyOptimizer
from atlas.physics.hamiltonians import TFIMHamiltonian
from atlas.physics.observables import tfim_observables


def build_hamiltonian(config: AtlasConfig, parameter_overrides: Optional[dict] = None):
    """Construct a Hamiltonian from ``config.system``."""

    params = dict(config.system.parameters)
    if parameter_overrides:
        params.update(parameter_overrides)

    if config.system.name == "tfim":
        return TFIMHamiltonian(**params)

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
    """Construct the configured execution backend for cost evaluation."""

    if config.backend.name == "statevector":
        return SimulatorEstimator()

    raise ValueError(
        f"Unknown backend '{config.backend.name}'. Supported: statevector."
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


def build_algorithm(config: AtlasConfig, estimator, optimizer):
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
        f"Unknown algorithm '{config.algorithm.name}'. Supported: vqe, vqd."
    )


def build_observables(config: AtlasConfig, num_qubits: int):
    """Return observable specs for the configured analysis."""

    if config.analysis.observables == "tfim_default":
        return tfim_observables(num_qubits)

    raise ValueError(
        f"Unknown observables '{config.analysis.observables}'. "
        "Supported: tfim_default."
    )
