"""Pure numerical post-processing for VQE results.

These functions compute fidelity, expectation values, and error metrics
without running optimization, simulation, or hardware jobs, and without
plotting anything. They mirror the inline computations in the legacy
`vqe_legacy/tfim_vis.py` script (`state_fidelity`, `Statevector.expectation_value`,
and the absolute/relative hardware-error calculations) as standalone,
testable functions.
"""

from __future__ import annotations

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector, state_fidelity

from atlas.physics.observables import ObservableSpec


def state_fidelity_to_exact(state, exact_state) -> float:
    """Return the state fidelity between `state` and `exact_state`."""

    return state_fidelity(state, exact_state)


def expectation_values(state, observables: list[ObservableSpec]) -> dict[str, float]:
    """Return `{spec.name: <state|spec.operator|state>.real}` for each spec.

    `state` may be a `Statevector` or a bound `QuantumCircuit`; circuits are
    converted to a `Statevector` first.
    """

    if isinstance(state, QuantumCircuit):
        state = Statevector(state)

    return {spec.name: state.expectation_value(spec.operator).real for spec in observables}


def absolute_error(reference: float, value: float) -> float:
    """Return `|reference - value|`."""

    return abs(reference - value)


def relative_error_percent(reference: float, value: float) -> float:
    """Return `|reference - value| / |reference| * 100`."""

    return abs(reference - value) / abs(reference) * 100
