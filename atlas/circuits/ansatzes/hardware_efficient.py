"""Hardware-efficient ansatz used by Atlas variational algorithms.

Single source of truth for the parameterized circuit previously duplicated
across legacy optimizer, hardware, and visualization scripts. Algorithms
consume an ``AnsatzSpec``; they never rebuild parameter indices themselves.
"""

from __future__ import annotations

from dataclasses import dataclass

from qiskit.circuit import ParameterVector, QuantumCircuit


@dataclass
class AnsatzSpec:
    """Parameterized ansatz circuit plus its parameter metadata.

    Responsibility:
        Bundle the Qiskit circuit, its ``ParameterVector``, and qubit count so
        VQE/VQD/hardware code can bind and evaluate without reconstructing
        the ansatz.

    State:
        circuit: Parameterized ``QuantumCircuit``.
        parameters: The ``ParameterVector`` embedded in ``circuit``.
        num_qubits: Register size (must match the Hamiltonian).

    Usage:
        Produced by ``build_hardware_efficient_ansatz``. Call ``bind`` to obtain
        a concrete circuit for statevector or hardware evaluation.
    """

    circuit: QuantumCircuit
    parameters: ParameterVector
    num_qubits: int

    @property
    def num_parameters(self) -> int:
        """Number of free variational angles in this ansatz."""

        return len(self.parameters)

    def bind(self, values) -> QuantumCircuit:
        """Return a new circuit with parameters assigned to ``values``.

        Purpose:
            Produce a fully bound circuit for statevector construction or
            hardware submission without mutating the parameterized template.

        Inputs:
            values: Sequence of floats with length ``num_parameters``.

        Process:
            Delegate to Qiskit ``assign_parameters``.

        Outputs:
            A bound ``QuantumCircuit`` (original ``self.circuit`` unchanged).

        Side effects:
            None on ``self``; Qiskit returns a new circuit object.
        """

        return self.circuit.assign_parameters(values)


def build_hardware_efficient_ansatz(num_qubits: int = 2, reps: int = 1) -> AnsatzSpec:
    """Construct a linear-connectivity hardware-efficient ansatz (HEA).

    Purpose:
        Provide the default variational form for TFIM VQE/VQD experiments.

    Inputs:
        num_qubits: Register size.
        reps: Number of entangling blocks after the initial RY layer.
            Total parameters = ``num_qubits * (reps + 1)``.

    Process:
        1. Allocate a ``ParameterVector('θ', ...)``.
        2. Apply an RY rotation on every qubit (initial layer).
        3. For each repetition: CX chain ``0→1→…→n-2``, then another RY layer.

    Outputs:
        ``AnsatzSpec`` wrapping the parameterized circuit.

    Side effects:
        None.

    Notes:
        Connectivity is a 1D line (device-native mapping happens later during
        transpilation). Defaults reproduce the legacy 2-qubit 4-parameter circuit
        when ``num_qubits=2`` and ``reps=1``.
    """

    num_parameters = num_qubits * (reps + 1)
    parameters = ParameterVector("θ", num_parameters)
    circuit = QuantumCircuit(num_qubits)
    param_index = 0

    def ry_layer():
        """Append one RY on each qubit, consuming the next ``num_qubits`` parameters."""

        nonlocal param_index
        for qubit in range(num_qubits):
            circuit.ry(parameters[param_index], qubit)
            param_index += 1

    # Initial single-qubit layer before any entanglement.
    ry_layer()
    for _ in range(reps):
        # Linear CX ladder creates entanglement along the chain.
        for qubit in range(num_qubits - 1):
            circuit.cx(qubit, qubit + 1)
        ry_layer()

    return AnsatzSpec(circuit=circuit, parameters=parameters, num_qubits=num_qubits)
