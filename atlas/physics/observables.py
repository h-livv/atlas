"""Named observables associated with a TFIM experiment.

Observables are experiment metrics, not part of the Hamiltonian. They are
generated from ``num_qubits`` using the same open-chain position convention as
``atlas.physics.hamiltonians.TFIMHamiltonian``.

Architectural role:
    Analysis/hardware code measures these operators on optimized or evolved
    states. Coefficients are ``+1.0`` (expectation values), unlike Hamiltonian
    weights which carry ``-J`` / ``-h``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from qiskit.quantum_info import SparsePauliOp


@dataclass
class ObservableSpec:
    """A named qubit operator to measure after VQE optimization or simulation.

    Responsibility:
        Pair a stable string key (used in CSV/plot columns) with a SparsePauliOp.

    State:
        name: Short identifier such as ``\"zz\"`` or ``\"x\"``.
        operator: Qiskit operator evaluated via statevector expectation values
            or hardware Estimator pubs.

    Usage:
        Produced by ``tfim_observables`` / ``build_observables``; consumed by
        ``analysis.metrics.expectation_values`` and IBM evaluation.
    """

    name: str
    operator: SparsePauliOp


def tfim_observables(num_qubits: int = 2) -> list[ObservableSpec]:
    """Return TFIM ``zz`` (nearest-neighbor bonds) and ``x`` (total transverse).

    Purpose:
        Provide the default analysis operators for TFIM experiments at any
        configured qubit count.

    Inputs:
        num_qubits: System size; must match the Hamiltonian under study.

    Process:
        Always build total magnetization-like ``x = Σ Xᵢ``. When ``n ≥ 2``,
        also build ``zz = Σ_{i} Zᵢ Zᵢ₊₁`` over all open-boundary bonds
        (matching the Hamiltonian interaction graph).

    Outputs:
        List of ``ObservableSpec``. When ZZ is present it is placed first so
        column order is ``zz`` then ``x``.

    Notes:
        For ``num_qubits=2`` this reproduces the legacy operators:
        ``zz_op = SparsePauliOp.from_list([(\"ZZ\", 1.0)])`` and
        ``x_op = SparsePauliOp.from_list([(\"XI\", 1.0), (\"IX\", 1.0)])``.
    """

    if num_qubits < 1:
        raise ValueError("num_qubits must be at least 1.")

    x_terms = []
    for i in range(num_qubits):
        pauli = ["I"] * num_qubits
        pauli[i] = "X"
        x_terms.append(("".join(pauli), 1.0))

    specs = [ObservableSpec(name="x", operator=SparsePauliOp.from_list(x_terms))]

    if num_qubits >= 2:
        zz_terms = []
        for i in range(num_qubits - 1):
            pauli = ["I"] * num_qubits
            pauli[i] = "Z"
            pauli[i + 1] = "Z"
            zz_terms.append(("".join(pauli), 1.0))
        specs.insert(0, ObservableSpec(name="zz", operator=SparsePauliOp.from_list(zz_terms)))

    return specs


def local_pauli_observables(
    num_qubits: int,
    pauli: str = "Z",
    *,
    prefix: Optional[str] = None,
) -> list[ObservableSpec]:
    """Return one single-qubit Pauli observable per site.

    Purpose:
        Provide physics-agnostic per-site operators (e.g. ``Z_i``) for lattice
        visualization and local magnetization analysis. Not tied to TFIM.

    Inputs:
        num_qubits: System size; must be >= 1.
        pauli: One of ``X``, ``Y``, ``Z`` (case-insensitive).
        prefix: Name prefix for specs (default: lowercase ``pauli``). Site
            ``i`` is named ``{prefix}_{i}``.

    Outputs:
        List of ``ObservableSpec`` ordered by site index.
    """

    if num_qubits < 1:
        raise ValueError("num_qubits must be at least 1.")

    letter = pauli.strip().upper()
    if letter not in {"X", "Y", "Z"}:
        raise ValueError(f"Unsupported Pauli '{pauli}'. Supported: X, Y, Z.")

    name_prefix = prefix if prefix is not None else letter.lower()
    specs = []
    for i in range(num_qubits):
        label = ["I"] * num_qubits
        label[i] = letter
        specs.append(
            ObservableSpec(
                name=f"{name_prefix}_{i}",
                operator=SparsePauliOp.from_list([("".join(label), 1.0)]),
            )
        )
    return specs
