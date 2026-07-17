"""First- and second-order Trotter product formulas.

These classes approximate continuous-time evolution ``e^{-iHt}`` when
``H = Σⱼ Hⱼ`` is a sum of Pauli terms. Lie–Trotter is first-order;
Strang (symmetric) splitting is second-order. Both prepend the initial-state
preparation circuit before applying ``PauliEvolutionGate`` layers.
"""

from __future__ import annotations

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Pauli, SparsePauliOp

from atlas.circuits.evolution.base import EvolutionCircuitSpec, EvolutionMethod
from atlas.circuits.initial_states import InitialStateSpec
from atlas.physics.hamiltonians import Hamiltonian


def _pauli_terms(operator: SparsePauliOp) -> list[tuple[Pauli, complex]]:
    """Return ``(Pauli, coefficient)`` pairs with deterministic ordering.

    Purpose:
        Give Lie and Strang a stable term sequence so Strang's reverse pass
        is a true symmetric reflection of the forward pass.

    Inputs:
        operator: Hamiltonian as a ``SparsePauliOp``.

    Process:
        Zip Qiskit's ``paulis`` and ``coeffs`` into an explicit list.

    Outputs:
        Ordered list of ``(Pauli, complex coeff)`` pairs.

    Side effects:
        None.
    """

    return list(zip(operator.paulis, operator.coeffs))


def _apply_pauli_evolution(
    circuit: QuantumCircuit,
    pauli: Pauli,
    time: float,
) -> None:
    """Append a single-Pauli time-evolution gate ``exp(-i * pauli * time)``.

    Purpose:
        Encapsulate Qiskit's ``PauliEvolutionGate`` so step builders stay short.

    Inputs:
        circuit: Circuit to mutate in place.
        pauli: Pauli string to evolve under.
        time: Evolution angle/time for this term (includes coefficient scaling).

    Process:
        Append ``PauliEvolutionGate(pauli, time=time)`` on all qubits.

    Outputs:
        None.

    Side effects:
        Mutates ``circuit`` by appending a gate.
    """

    circuit.append(PauliEvolutionGate(pauli, time=time), circuit.qubits)


class _ProductFormulaBase(EvolutionMethod):
    """Shared helpers for Trotter-style product formulas.

    Responsibility:
        Own the common compose loop (prep → repeat step builder ``r`` times)
        so Lie and Strang only differ in how one step is expanded.

    State:
        num_trotter_steps: Number of repetitions ``r`` (≥ 1).

    Usage:
        Not constructed directly by builders; subclassed by ``LieTrotter`` and
        ``StrangTrotter``.
    """

    def __init__(self, num_trotter_steps: int = 1):
        """Store the Trotter repetition count.

        Inputs:
            num_trotter_steps: Positive integer ``r``; larger ``r`` usually means
                smaller local error and deeper circuits.
        """

        if num_trotter_steps < 1:
            raise ValueError("num_trotter_steps must be at least 1.")
        self.num_trotter_steps = num_trotter_steps

    def _compose_circuit(
        self,
        hamiltonian: Hamiltonian,
        initial_state: InitialStateSpec,
        evolution_time: float,
        method_name: str,
        step_builder,
    ) -> EvolutionCircuitSpec:
        """Assemble prep + repeated Trotter steps into an ``EvolutionCircuitSpec``.

        Purpose:
            Factor out the identical scaffolding around Lie vs Strang step bodies.

        Inputs:
            hamiltonian: Provides qubit count and Pauli operator.
            initial_state: Provides the preparation circuit.
            evolution_time: Total time ``t``.
            method_name: Label stored on the returned spec.
            step_builder: Callable ``(circuit, terms, step_time) -> None`` that
                appends one Trotter step's gates.

        Process:
            1. Create an empty circuit and compose state preparation.
            2. Split total time into equal steps ``Δt = t / r``.
            3. Extract Pauli terms once (order reused every step).
            4. Invoke ``step_builder`` once per Trotter repetition.

        Outputs:
            ``EvolutionCircuitSpec`` with circuit and metadata.

        Side effects:
            Builds a new circuit; ``step_builder`` mutates that circuit.
        """

        circuit = QuantumCircuit(hamiltonian.num_qubits)
        circuit.compose(initial_state.preparation_circuit(), inplace=True)

        # Split the total evolution time into equal Trotter steps so that
        # each iteration approximates evolution over Δt = t / r.
        step_time = evolution_time / self.num_trotter_steps
        terms = _pauli_terms(hamiltonian.operator())

        for _ in range(self.num_trotter_steps):
            step_builder(circuit, terms, step_time)

        return EvolutionCircuitSpec(
            circuit=circuit,
            num_qubits=hamiltonian.num_qubits,
            num_trotter_steps=self.num_trotter_steps,
            evolution_time=evolution_time,
            method_name=method_name,
        )


class LieTrotter(_ProductFormulaBase):
    """First-order Lie–Trotter product formula.

    Responsibility:
        Approximate each step as the forward product ``Πⱼ exp(-i Hⱼ Δt)``.

    State:
        Inherited ``num_trotter_steps``.

    Usage:
        Selected when config ``evolution_method`` / method name is ``\"lie\"``.
    """

    @property
    def name(self) -> str:
        """Stable label written into simulation results and plots."""

        return "lie"

    def build_circuit(
        self,
        hamiltonian: Hamiltonian,
        initial_state: InitialStateSpec,
        evolution_time: float,
    ) -> EvolutionCircuitSpec:
        """Build a first-order Trotter circuit for time ``evolution_time``.

        Purpose:
            Implement Lie–Trotter as a concrete ``EvolutionMethod``.

        Inputs:
            hamiltonian: Pauli-sum Hamiltonian.
            initial_state: State preparation.
            evolution_time: Total time ``t``.

        Process:
            Each step walks Pauli terms in order and evolves each with
            ``Δt * Re(coeff)`` (imaginary parts ignored; TFIM coeffs are real).

        Outputs:
            ``EvolutionCircuitSpec`` labeled ``\"lie\"``.

        Side effects:
            None beyond constructing a new circuit.
        """

        def step_builder(circuit, terms, step_time):
            for pauli, coeff in terms:
                # Scale the step by the Pauli coefficient so Σ cⱼ Pⱼ is respected.
                _apply_pauli_evolution(circuit, pauli, step_time * coeff.real)

        return self._compose_circuit(
            hamiltonian,
            initial_state,
            evolution_time,
            self.name,
            step_builder,
        )


class StrangTrotter(_ProductFormulaBase):
    """Second-order Strang splitting (symmetric Trotter).

    Responsibility:
        Approximate each step with a symmetric half-step forward / reverse
        product, cancelling first-order error terms.

    State:
        Inherited ``num_trotter_steps``.

    Usage:
        Default dynamics method in many configs (``evolution_method: strang``).
    """

    @property
    def name(self) -> str:
        """Stable label written into simulation results and plots."""

        return "strang"

    def build_circuit(
        self,
        hamiltonian: Hamiltonian,
        initial_state: InitialStateSpec,
        evolution_time: float,
    ) -> EvolutionCircuitSpec:
        """Build a second-order Strang–Trotter circuit for time ``evolution_time``.

        Purpose:
            Implement symmetric Trotter as a concrete ``EvolutionMethod``.

        Inputs:
            hamiltonian: Pauli-sum Hamiltonian.
            initial_state: State preparation.
            evolution_time: Total time ``t``.

        Process:
            Each step applies half-time evolutions in forward term order, then
            half-time evolutions in reverse term order.

        Outputs:
            ``EvolutionCircuitSpec`` labeled ``\"strang\"``.

        Side effects:
            None beyond constructing a new circuit.
        """

        def step_builder(circuit, terms, step_time):
            half_time = step_time / 2
            # Forward half-steps: Πⱼ exp(-i Hⱼ Δt/2).
            for pauli, coeff in terms:
                _apply_pauli_evolution(circuit, pauli, half_time * coeff.real)
            # Reverse half-steps close the symmetric formula.
            for pauli, coeff in reversed(terms):
                _apply_pauli_evolution(circuit, pauli, half_time * coeff.real)

        return self._compose_circuit(
            hamiltonian,
            initial_state,
            evolution_time,
            self.name,
            step_builder,
        )
