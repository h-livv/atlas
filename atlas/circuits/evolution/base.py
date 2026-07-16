"""Base types for Hamiltonian time-evolution circuit construction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from qiskit.circuit import QuantumCircuit

from atlas.circuits.initial_states import InitialStateSpec
from atlas.physics.hamiltonians import Hamiltonian


@dataclass
class EvolutionCircuitSpec:
    """A deterministic evolution circuit plus simulation metadata."""

    circuit: QuantumCircuit
    num_qubits: int
    num_trotter_steps: int
    evolution_time: float
    method_name: str

    @property
    def circuit_depth(self) -> int:
        return self.circuit.depth()


class EvolutionMethod(ABC):
    """Builds a time-evolution circuit from a Hamiltonian and initial state."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this evolution method."""

    @abstractmethod
    def build_circuit(
        self,
        hamiltonian: Hamiltonian,
        initial_state: InitialStateSpec,
        evolution_time: float,
    ) -> EvolutionCircuitSpec:
        """Return the full evolution circuit and metadata."""
