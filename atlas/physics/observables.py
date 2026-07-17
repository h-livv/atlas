"""Named observables associated with a TFIM experiment.

Observables are experiment metrics, not part of the Hamiltonian. They are
generated from `num_qubits` using the same position convention as
`atlas.physics.hamiltonians.TFIMHamiltonian` so the 2-qubit case matches the
legacy `zz_op`/`x_op` operators exactly.

Architectural role:
    Analysis/hardware code measures these operators on optimized or evolved
    states. Coefficients are ``+1.0`` (expectation values), unlike Hamiltonian
    weights which carry ``-J`` / ``-h``.
"""

from __future__ import annotations

from dataclasses import dataclass

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
    """Return the TFIM `zz` (nearest-neighbor) and `x` (total transverse) observables.

    Purpose:
        Provide the default analysis operators for TFIM experiments.

    Inputs:
        num_qubits: System size; must match the Hamiltonian under study.

    Process:
        Always build total magnetization-like ``x = Σ Xᵢ``. When ``n ≥ 2``,
        also insert a ZZ observable on qubits 0 and 1 only (legacy convention —
        not a sum over all bonds).

    Outputs:
        List of ``ObservableSpec``. When ZZ is present it is placed first so
        column order matches historical ``zz`` then ``x`` layouts.

    Side effects:
        None.

    Notes:
        For ``num_qubits=2`` this reproduces the legacy operators exactly:
        ``zz_op = SparsePauliOp.from_list([(\"ZZ\", 1.0)])`` and
        ``x_op = SparsePauliOp.from_list([(\"XI\", 1.0), (\"IX\", 1.0)])``.
    """

    x_terms = []
    for i in range(num_qubits):
        pauli = ["I"] * num_qubits
        pauli[i] = "X"
        x_terms.append(("".join(pauli), 1.0))

    specs = [ObservableSpec(name="x", operator=SparsePauliOp.from_list(x_terms))]

    if num_qubits >= 2:
        # First-bond ZZ only: preserves the 2-qubit legacy metric definition.
        zz_pauli = ["I"] * num_qubits
        zz_pauli[0] = "Z"
        zz_pauli[1] = "Z"
        zz_terms = [("".join(zz_pauli), 1.0)]
        specs.insert(0, ObservableSpec(name="zz", operator=SparsePauliOp.from_list(zz_terms)))

    return specs
