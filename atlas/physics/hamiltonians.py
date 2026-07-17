"""Physical models that produce qubit operators.

`Hamiltonian` defines the minimal interface used by algorithms: `operator()`
and `num_qubits`. `TFIMHamiltonian` generalizes the legacy 2-qubit TFIM model
to an arbitrary number of qubits while reproducing the legacy behavior
bit-for-bit at `num_qubits=2`.

Architectural role:
    This module sits at the bottom of the dependency graph. Algorithms and
    experiments may depend on it; it must never import experiment or I/O code.
    Exact diagonalization / time evolution helpers exist for small-system
    benchmarking against quantum methods.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.linalg import eigh
from qiskit.quantum_info import SparsePauliOp
from scipy.linalg import expm


@dataclass
class ExactResult:
    """Result of a dense diagonalization or exact time evolution.

    Responsibility:
        Carry a classical reference energy and statevector so experiments can
        compute fidelity and energy error without re-running linear algebra.

    State:
        energy: Scalar energy (eigenvalue or expectation after evolution).
        statevector: Complex amplitude vector of length ``2 ** num_qubits``.

    Usage:
        Returned by ``Hamiltonian.exact_*`` methods; fields are copied into
        experiment point results.
    """

    energy: float
    statevector: np.ndarray


class Hamiltonian:
    """Minimal interface for a physical model backing a `SparsePauliOp`.

    Responsibility:
        Expose the qubit operator algorithms need, plus optional dense exact
        solvers for NISQ-scale validation.

    State:
        num_qubits: System size (Hilbert space dimension ``2 ** num_qubits``).

    Usage:
        Subclass and implement ``operator()``. Algorithms typically only call
        ``operator()`` and read ``num_qubits``; experiment classes call the
        ``exact_*`` helpers for classical baselines.
    """

    def __init__(self, num_qubits: int):
        """Store the qubit count for this model instance.

        Inputs:
            num_qubits: Number of qubits in the physical system.
        """

        self.num_qubits = num_qubits

    def operator(self) -> SparsePauliOp:
        """Return the qubit operator for this Hamiltonian.

        Purpose:
            Provide the SparsePauliOp used by VQE cost evaluation and Trotter
            circuit construction.

        Outputs:
            A Qiskit ``SparsePauliOp`` (subclasses must implement).

        Side effects:
            None for well-behaved subclasses (no I/O).
        """

        raise NotImplementedError

    def exact_ground_state(self) -> ExactResult:
        """Diagonalize ``operator()`` and return the ground energy and state.

        Purpose:
            Give VQE experiments a classical ground-truth reference.

        Inputs:
            None (uses ``self.operator()``).

        Process:
            Convert the Pauli operator to a dense matrix, Hermitian-diagonalize
            with ``eigh``, and select the lowest eigenvalue/eigenvector.

        Outputs:
            ``ExactResult`` with ground energy and statevector.

        Side effects:
            None. Matrix size scales as ``2 ** num_qubits`` — small-system only.
        """

        eigenvalues, eigenvectors = eigh(self.operator().to_matrix())
        lowest_index = int(np.argmin(eigenvalues))
        return ExactResult(
            energy=float(eigenvalues[lowest_index]),
            statevector=eigenvectors[:, lowest_index],
        )

    def exact_spectrum(self, num_states: int) -> list[ExactResult]:
        """Return the lowest ``num_states`` exact eigenpairs by dense diagonalization.

        Purpose:
            Provide classical references for VQD excited-state comparisons.

        Inputs:
            num_states: How many lowest eigenpairs to keep (must be ≥ 1).

        Process:
            Dense ``eigh``, sort eigenvalues ascending, take the first
            ``num_states`` indices.

        Outputs:
            List of ``ExactResult`` ordered from lowest energy upward.

        Side effects:
            None. Same ``2 ** n`` dense-matrix caveat as ``exact_ground_state``.
        """

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

    def exact_time_evolution(
        self,
        initial_state: np.ndarray,
        time: float,
    ) -> ExactResult:
        """Return the exact evolved state ``exp(-i H t) |psi_0>`` for small systems.

        Purpose:
            Classical baseline for Hamiltonian-simulation fidelity and
            observable-error metrics.

        Inputs:
            initial_state: Complex amplitude vector (length ``2 ** n``).
            time: Evolution time ``t`` in the same units as ``H``.

        Process:
            Form the dense matrix of ``H``, apply ``expm(-i H t)`` to ``|ψ₀⟩``,
            renormalize, and compute the energy expectation of the evolved state.

        Outputs:
            ``ExactResult`` whose ``statevector`` is the evolved state and
            ``energy`` is ``⟨ψ(t)|H|ψ(t)⟩``.

        Side effects:
            None. Uses dense matrix exponentiation (``2 ** n`` scaling).
        """

        from atlas import profiling as profile

        with profile.span("exact.to_matrix"):
            matrix = self.operator().to_matrix()
        state = np.asarray(initial_state, dtype=complex).reshape(-1)
        # Schrödinger evolution under a time-independent Hermitian H.
        with profile.span("exact.expm_apply"):
            evolved = expm(-1j * matrix * time) @ state
            norm = np.linalg.norm(evolved)
            if norm == 0:
                raise ValueError("Time evolution produced a zero-norm statevector.")
            # Numerical expm can drift slightly off the unit sphere; renormalize.
            evolved = evolved / norm
            energy = float(np.real(np.vdot(evolved, matrix @ evolved)))
        return ExactResult(energy=energy, statevector=evolved)

class TFIMHamiltonian(Hamiltonian):
    """Nearest-neighbor transverse field Ising model Hamiltonian.

    Responsibility:
        Encode open-boundary TFIM couplings as a SparsePauliOp:
        ``H = -J Σ ZᵢZᵢ₊₁ - h Σ Xᵢ``.

    State:
        num_qubits (inherited), J (Ising coupling), h (transverse field).

    Usage:
        Constructed by ``build_hamiltonian`` when ``system.name == \"tfim\"``.
        For ``num_qubits=2`` this reproduces the legacy Hamiltonian
        ``[(\"ZZ\", -J), (\"XI\", -h), (\"IX\", -h)]`` exactly.
    """

    def __init__(self, num_qubits: int, J: float = 1.0, h: float = 1.0):
        """Create a TFIM instance with fixed couplings.

        Inputs:
            num_qubits: Chain length.
            J: Nearest-neighbor ZZ coupling (positive favors ferromagnetic
                alignment given the overall minus sign in ``operator``).
            h: Transverse field strength on each site.
        """

        super().__init__(num_qubits)
        self.J = J
        self.h = h

    def operator(self) -> SparsePauliOp:
        """Build the TFIM SparsePauliOp for the current ``(J, h, n)``.

        Purpose:
            Materialize the qubit Hamiltonian algorithms and Trotter methods use.

        Process:
            For each bond ``i``, append ``-J ZᵢZᵢ₊₁``; for each site, append
            ``-h Xᵢ``. Open boundaries: no wrap-around ZZ term.

        Outputs:
            ``SparsePauliOp`` assembled via ``from_list``.

        Side effects:
            None (rebuilds the term list on every call; no caching).
        """

        terms = []
        # Interaction terms: nearest-neighbor ZZ on an open chain.
        for i in range(self.num_qubits - 1):
            pauli = ["I"] * self.num_qubits
            pauli[i] = "Z"
            pauli[i + 1] = "Z"
            terms.append(("".join(pauli), -self.J))
        # Transverse field terms: X on every site.
        for i in range(self.num_qubits):
            pauli = ["I"] * self.num_qubits
            pauli[i] = "X"
            terms.append(("".join(pauli), -self.h))
        return SparsePauliOp.from_list(terms)
