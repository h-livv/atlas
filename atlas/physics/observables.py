"""Named observables associated with a TFIM experiment.

Observables are experiment metrics, not part of the Hamiltonian. They are
generated from `num_qubits` using the same position convention as
`atlas.physics.hamiltonians.TFIMHamiltonian` so the 2-qubit case matches the
legacy `zz_op`/`x_op` operators exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

from qiskit.quantum_info import SparsePauliOp


@dataclass
class ObservableSpec:
    """A named qubit operator to measure after VQE optimization."""

    name: str
    operator: SparsePauliOp


def tfim_observables(num_qubits: int = 2) -> list[ObservableSpec]:
    """Return the TFIM `zz` (nearest-neighbor) and `x` (total transverse) observables.

    For `num_qubits=2` this reproduces the legacy operators exactly:
    `zz_op = SparsePauliOp.from_list([("ZZ", 1.0)])` and
    `x_op = SparsePauliOp.from_list([("XI", 1.0), ("IX", 1.0)])`.
    """

    x_terms = []
    for i in range(num_qubits):
        pauli = ["I"] * num_qubits
        pauli[i] = "X"
        x_terms.append(("".join(pauli), 1.0))

    specs = [ObservableSpec(name="x", operator=SparsePauliOp.from_list(x_terms))]

    if num_qubits >= 2:
        zz_pauli = ["I"] * num_qubits
        zz_pauli[0] = "Z"
        zz_pauli[1] = "Z"
        zz_terms = [("".join(zz_pauli), 1.0)]
        specs.insert(0, ObservableSpec(name="zz", operator=SparsePauliOp.from_list(zz_terms)))

    return specs
