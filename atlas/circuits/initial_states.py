"""Initial states for Hamiltonian simulation.

Dynamics experiments need a *pure* preparation circuit (no variational
parameters), parallel in role to ``AnsatzSpec`` but simpler. The computational-
basis implementation covers the bitstrings used in current TFIM demos.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector


@dataclass
class InitialStateSpec:
    """Abstract pure state used as the starting point for time evolution.

    Responsibility:
        Define how to prepare ``|ψ₀⟩`` as a circuit and as a dense amplitude
        vector for classical exact evolution.

    State:
        num_qubits: Register size the preparation must match.

    Usage:
        Subclass and implement ``preparation_circuit``. Evolution methods
        compose that circuit before Trotter gates; experiments call
        ``statevector()`` for exact ``expm`` baselines.
    """

    num_qubits: int

    def preparation_circuit(self) -> QuantumCircuit:
        """Return a circuit that maps ``|0...0⟩`` to this state.

        Purpose:
            Provide the unitary (or gate sequence) prepended to evolution.

        Outputs:
            A ``QuantumCircuit`` on ``num_qubits`` qubits.

        Side effects:
            None for implementations (construct a new circuit).
        """

        raise NotImplementedError

    def statevector(self) -> np.ndarray:
        """Return dense normalized amplitudes for this pure state.

        Purpose:
            Feed classical exact time evolution without re-deriving the bitstring.

        Process:
            Simulate ``preparation_circuit()`` with Qiskit's ``Statevector``.

        Outputs:
            Complex ``numpy`` array of length ``2 ** num_qubits``.

        Side effects:
            Builds and classically simulates the preparation circuit.
        """

        return np.asarray(Statevector(self.preparation_circuit()))


@dataclass
class ComputationalBasisState(InitialStateSpec):
    """Product state ``|bitstring⟩`` in the computational basis.

    Responsibility:
        Prepare a single computational-basis vector using X gates on ``1`` bits.

    State:
        num_qubits: Register size.
        bitstring: String of ``0``/``1`` characters with length ``num_qubits``.
            Index ``0`` corresponds to qubit ``0`` (same convention as
            ``enumerate`` over the string).

    Usage:
        Default dynamics initial state via ``build_initial_state``.
    """

    bitstring: str

    def __post_init__(self):
        """Validate bitstring length and alphabet after dataclass construction."""

        if len(self.bitstring) != self.num_qubits:
            raise ValueError(
                f"bitstring length {len(self.bitstring)} does not match "
                f"num_qubits={self.num_qubits}."
            )
        if any(bit not in "01" for bit in self.bitstring):
            raise ValueError(f"bitstring must contain only 0 and 1, got '{self.bitstring}'.")

    def preparation_circuit(self) -> QuantumCircuit:
        """Build X gates on every qubit whose bit is ``1``.

        Purpose:
            Map the all-zero register to ``|bitstring⟩``.

        Process:
            Start from an empty circuit; apply ``X(i)`` wherever
            ``bitstring[i] == \"1\"``.

        Outputs:
            Parameter-free ``QuantumCircuit``.

        Side effects:
            None.
        """

        circuit = QuantumCircuit(self.num_qubits)
        for index, bit in enumerate(self.bitstring):
            if bit == "1":
                circuit.x(index)
        return circuit


def default_initial_state(num_qubits: int) -> ComputationalBasisState:
    """Return the all-zero computational basis state ``|0...0⟩``.

    Purpose:
        Convenience constructor for the most common dynamics starting state.

    Inputs:
        num_qubits: Register size.

    Process:
        Build a bitstring of zeros and wrap it in ``ComputationalBasisState``.

    Outputs:
        A validated ``ComputationalBasisState``.

    Side effects:
        None.
    """

    return ComputationalBasisState(num_qubits=num_qubits, bitstring="0" * num_qubits)
