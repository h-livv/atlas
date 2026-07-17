"""Local statevector estimator for variational cost evaluation.

Owns the Qiskit ``StatevectorEstimator`` so VQE/VQD never construct estimator
pubs directly. This is the noiseless simulation backend for expectation values.
"""

from __future__ import annotations

from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import Statevector


class SimulatorEstimator:
    """Evaluate expectation values via an ideal statevector simulator.

    Responsibility:
        Translate ``(circuit, observable, parameters)`` into a scalar cost for
        classical optimizers during VQE/VQD.

    State:
        _estimator: Private Qiskit ``StatevectorEstimator`` instance created
            once at construction and reused for every call.

    Usage:
        Built when ``backend.name == \"statevector\"``. Injected into ``VQE``.
    """

    def __init__(self):
        """Create the underlying Qiskit statevector estimator primitive."""

        self._estimator = StatevectorEstimator()

    def expectation(self, circuit, observable, parameter_values) -> float:
        """Return ``⟨ψ(θ)|O|ψ(θ)⟩`` for a parameterized circuit.

        Purpose:
            Provide the scalar cost VQE minimizes (typically ``O = H``).

        Inputs:
            circuit: Parameterized quantum circuit (ansatz template).
            observable: Qiskit-compatible operator (usually ``SparsePauliOp``).
            parameter_values: Numeric binding for the circuit parameters.

        Process:
            Pack a single Estimator pub, run the primitive, and cast the
            expectation value array entry to ``float``.

        Outputs:
            Real expectation value as a Python float.

        Side effects:
            Ideal statevector simulation (CPU). No filesystem I/O.
        """

        pub = (circuit, observable, parameter_values)
        return float(self._estimator.run([pub]).result()[0].data.evs)

    def statevector(self, bound_circuit) -> Statevector:
        """Return the exact statevector of an already-bound circuit.

        Purpose:
            Convenience helper for callers that need amplitudes rather than
            an expectation value. Dynamics normally uses ``StatevectorEvolver``
            instead.

        Inputs:
            bound_circuit: Fully assigned ``QuantumCircuit``.

        Process:
            Construct a Qiskit ``Statevector`` from the circuit.

        Outputs:
            ``Statevector`` instance.

        Side effects:
            Ideal simulation of the bound circuit.
        """

        return Statevector(bound_circuit)
