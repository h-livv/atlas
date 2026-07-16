"""Initial state specifications for Hamiltonian simulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector


@dataclass
class InitialStateSpec:
    """Describes a pure state used as the starting point for time evolution."""

    num_qubits: int

    def preparation_circuit(self) -> QuantumCircuit:
        """Return a circuit that prepares this state from ``|0...0>``."""

        raise NotImplementedError

    def statevector(self) -> np.ndarray:
        """Return the normalized statevector of this initial state."""

        return np.asarray(Statevector(self.preparation_circuit()))


@dataclass
class ComputationalBasisState(InitialStateSpec):
    """A computational-basis product state ``|bitstring>``."""

    bitstring: str

    def __post_init__(self):
        if len(self.bitstring) != self.num_qubits:
            raise ValueError(
                f"bitstring length {len(self.bitstring)} does not match "
                f"num_qubits={self.num_qubits}."
            )
        if any(bit not in "01" for bit in self.bitstring):
            raise ValueError(f"bitstring must contain only 0 and 1, got '{self.bitstring}'.")

    def preparation_circuit(self) -> QuantumCircuit:
        circuit = QuantumCircuit(self.num_qubits)
        for index, bit in enumerate(self.bitstring):
            if bit == "1":
                circuit.x(index)
        return circuit


def default_initial_state(num_qubits: int) -> ComputationalBasisState:
    """Return ``|0...0>`` for the given qubit count."""

    return ComputationalBasisState(num_qubits=num_qubits, bitstring="0" * num_qubits)
