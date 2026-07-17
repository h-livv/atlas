"""Shared types for product-formula / evolution circuit construction.

Parallel to ``AnsatzSpec``, but for *deterministic* (parameter-free) circuits
that approximate ``e^{-iHt}``. Algorithms consume ``EvolutionCircuitSpec``;
concrete methods subclass ``EvolutionMethod``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from qiskit.circuit import QuantumCircuit

from atlas.circuits.initial_states import InitialStateSpec
from atlas.physics.hamiltonians import Hamiltonian


@dataclass
class EvolutionCircuitSpec:
    """Fully built evolution circuit plus Trotter metadata.

    Responsibility:
        Carry the executable circuit and the bookkeeping fields needed when
        packaging a ``SimulationResult`` (method name, step count, time).

    State:
        circuit: Prep + product-formula gates (no free parameters).
        num_qubits: Register size.
        num_trotter_steps: Number of Trotter repetitions ``r``.
        evolution_time: Total time ``t`` the circuit approximates.
        method_name: Short label such as ``\"lie\"`` or ``\"strang\"``.

    Usage:
        Returned by ``EvolutionMethod.build_circuit``; passed to an evolver;
        metadata copied into algorithm results (the circuit itself is usually
        discarded afterward).
    """

    circuit: QuantumCircuit
    num_qubits: int
    num_trotter_steps: int
    evolution_time: float
    method_name: str

    @property
    def circuit_depth(self) -> int:
        """Qiskit circuit depth — used when reporting resource cost vs accuracy."""

        return self.circuit.depth()


class EvolutionMethod(ABC):
    """Strategy interface: Hamiltonian + initial state + time → evolution circuit.

    Responsibility:
        Abstract how ``e^{-iHt}`` is approximated as gates so algorithms can
        swap Lie, Strang, or future methods without changing orchestration.

    State:
        Subclasses typically store discretization knobs (e.g. Trotter steps).

    Usage:
        Implement ``name`` and ``build_circuit``. Inject into
        ``HamiltonianSimulation``.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in results and plots (e.g. ``\"lie\"``)."""

    @abstractmethod
    def build_circuit(
        self,
        hamiltonian: Hamiltonian,
        initial_state: InitialStateSpec,
        evolution_time: float,
    ) -> EvolutionCircuitSpec:
        """Construct the preparation + approximate evolution circuit.

        Purpose:
            Translate a physical evolution problem into an executable circuit.

        Inputs:
            hamiltonian: Model providing ``.operator()`` / ``.num_qubits``.
            initial_state: Pure state with ``.preparation_circuit()``.
            evolution_time: Total time ``t`` to simulate.

        Outputs:
            ``EvolutionCircuitSpec`` ready for the evolver.

        Side effects:
            None required (build a new circuit each call).
        """
