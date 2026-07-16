"""Local statevector estimator for variational algorithm cost evaluation.

Owns the ``StatevectorEstimator`` primitive so VQE/VQD never construct
Qiskit estimator pubs directly.
"""

from __future__ import annotations

from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import Statevector


class SimulatorEstimator:
    """Evaluates expectation values via local statevector simulation."""

    def __init__(self):
        self._estimator = StatevectorEstimator()

    def expectation(self, circuit, observable, parameter_values) -> float:
        """Return ``<circuit(params)| observable |circuit(params)>``."""

        pub = (circuit, observable, parameter_values)
        return float(self._estimator.run([pub]).result()[0].data.evs)

    def statevector(self, bound_circuit) -> Statevector:
        """Return the statevector of an already-bound circuit."""

        return Statevector(bound_circuit)
