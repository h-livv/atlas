"""Local statevector-simulator estimator execution.

Owns the `StatevectorEstimator` primitive so the VQE algorithm never
constructs Qiskit primitives or estimator pubs directly.
"""

from __future__ import annotations

from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import Statevector


class SimulatorEstimator:
    """Evaluates expectation values and statevectors via local simulation."""

    def __init__(self):
        self._estimator = StatevectorEstimator()

    def expectation(self, circuit, observable, parameter_values) -> float:
        """Return `<circuit(parameter_values)| observable |circuit(parameter_values)>`."""

        pub = (circuit, observable, parameter_values)
        return float(self._estimator.run([pub]).result()[0].data.evs)

    def statevector(self, bound_circuit) -> Statevector:
        """Return the statevector of an already-bound circuit."""

        return Statevector(bound_circuit)
