"""Hardware-efficient ansatz factory.

Provides the single source of truth for the parameterized circuit currently
duplicated across the legacy optimizer, hardware execution, and
visualization scripts.
"""

from __future__ import annotations

from dataclasses import dataclass

from qiskit.circuit import ParameterVector, QuantumCircuit


@dataclass
class AnsatzSpec:
    """A parameterized circuit together with its parameter metadata."""

    circuit: QuantumCircuit
    parameters: ParameterVector
    num_qubits: int

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def bind(self, values) -> QuantumCircuit:
        """Return a bound circuit for the given parameter values."""

        return self.circuit.assign_parameters(values)


def build_hardware_efficient_ansatz(num_qubits: int = 2, reps: int = 1) -> AnsatzSpec:
    """Build a hardware-efficient ansatz: RY layer, then `reps` x (CX-layer, RY-layer).

    For `num_qubits=2, reps=1` this produces exactly the legacy circuit:
    `RY(θ0,0); RY(θ1,1); CX(0,1); RY(θ2,0); RY(θ3,1)` using
    `ParameterVector('θ', 4)`.
    """

    num_parameters = num_qubits * (reps + 1)
    parameters = ParameterVector("θ", num_parameters)
    circuit = QuantumCircuit(num_qubits)

    param_index = 0

    def ry_layer():
        nonlocal param_index
        for qubit in range(num_qubits):
            circuit.ry(parameters[param_index], qubit)
            param_index += 1

    ry_layer()
    for _ in range(reps):
        for qubit in range(num_qubits - 1):
            circuit.cx(qubit, qubit + 1)
        ry_layer()

    return AnsatzSpec(circuit=circuit, parameters=parameters, num_qubits=num_qubits)
