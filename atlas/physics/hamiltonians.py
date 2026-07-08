"""Physical models that produce qubit operators.

`Hamiltonian` defines the minimal interface used by algorithms: `operator()`
and `num_qubits`. `TFIMHamiltonian` generalizes the legacy 2-qubit TFIM model
to an arbitrary number of qubits while reproducing the legacy behavior
bit-for-bit at `num_qubits=2`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.linalg import eigh
from qiskit.quantum_info import SparsePauliOp


@dataclass
class ExactResult:
    """Result of a dense diagonalization of a Hamiltonian."""

    energy: float
    statevector: np.ndarray


class Hamiltonian:
    """Minimal interface for a physical model backing a `SparsePauliOp`."""

    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits

    def operator(self) -> SparsePauliOp:
        """Return the qubit operator for this Hamiltonian."""

        raise NotImplementedError

    def exact_ground_state(self) -> ExactResult:
        """Diagonalize ``operator()`` and return the ground energy and state.

        Matrix size scales as ``2 ** num_qubits``; this is a small-system
        benchmark only, not a scalable solver.
        """

        eigenvalues, eigenvectors = eigh(self.operator().to_matrix())
        lowest_index = int(np.argmin(eigenvalues))
        return ExactResult(
            energy=float(eigenvalues[lowest_index]),
            statevector=eigenvectors[:, lowest_index],
        )

    def exact_spectrum(self, num_states: int) -> list[ExactResult]:
        """Return the lowest ``num_states`` exact eigenpairs by dense diagonalization."""

        if num_states < 1:
            raise ValueError("num_states must be at least 1.")
        eigenvalues, eigenvectors = eigh(self.operator().to_matrix())
        indices = np.argsort(eigenvalues)[:num_states]
        return [
            ExactResult(
                energy=float(eigenvalues[i]),
                statevector=eigenvectors[:, i],
            )
            for i in indices
        ]


class TFIMHamiltonian(Hamiltonian):
    """Nearest-neighbor transverse field Ising model Hamiltonian.

    For `num_qubits=2` this reproduces the legacy Hamiltonian
    `[("ZZ", -J), ("XI", -h), ("IX", -h)]` exactly.
    """

    def __init__(self, num_qubits: int, J: float = 1.0, h: float = 1.0):
        super().__init__(num_qubits)
        self.J = J
        self.h = h

    def operator(self) -> SparsePauliOp:
        terms = []
        for i in range(self.num_qubits - 1):
            pauli = ["I"] * self.num_qubits
            pauli[i] = "Z"
            pauli[i + 1] = "Z"
            terms.append(("".join(pauli), -self.J))
        for i in range(self.num_qubits):
            pauli = ["I"] * self.num_qubits
            pauli[i] = "X"
            terms.append(("".join(pauli), -self.h))
        return SparsePauliOp.from_list(terms)
