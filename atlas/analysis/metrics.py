"""Pure numerical post-processing for VQE results.

These functions compute fidelity, expectation values, and error metrics
without running optimization, simulation, or hardware jobs, and without
plotting anything. They mirror the inline computations in the legacy
`vqe_legacy/tfim_vis.py` script (`state_fidelity`, `Statevector.expectation_value`,
and the absolute/relative hardware-error calculations) as standalone,
testable functions.

Architectural role:
    Stateless helpers used by experiment workflows after a state is known.
    Safe to import from tests and notebooks; must stay free of I/O and of
    TFIM-specific wiring.
"""

from __future__ import annotations

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector, state_fidelity

from atlas.physics.observables import ObservableSpec


def state_fidelity_to_exact(state, exact_state) -> float:
    """Return the state fidelity between `state` and `exact_state`.

    Purpose:
        Quantify overlap between a variational/simulated state and a
        classical reference statevector.

    Inputs:
        state: Approximate state (``Statevector`` or compatible).
        exact_state: Reference state (same Hilbert space).

    Process:
        Delegate to Qiskit's ``state_fidelity``.

    Outputs:
        Fidelity as a float in ``[0, 1]`` (up to numerical noise).

    Side effects:
        None.
    """

    return state_fidelity(state, exact_state)


def expectation_values(state, observables: list[ObservableSpec]) -> dict[str, float]:
    """Return `{spec.name: <state|spec.operator|state>.real}` for each spec.

    `state` may be a `Statevector` or a bound `QuantumCircuit`; circuits are
    converted to a `Statevector` first.

    Purpose:
        Evaluate a batch of named observables for reporting and error
        analysis without coupling to a specific Hamiltonian family.

    Inputs:
        state: ``Statevector`` or bound ``QuantumCircuit``.
        observables: List of ``ObservableSpec`` with ``.name`` and
            ``.operator``.

    Process:
        Promote circuits to statevectors; for each spec, take the real part
        of ``state.expectation_value(spec.operator)``.

    Outputs:
        Dict mapping observable name → real expectation value.

    Side effects:
        None (local conversion of circuits to statevectors only).
    """

    if isinstance(state, QuantumCircuit):
        state = Statevector(state)

    return {spec.name: state.expectation_value(spec.operator).real for spec in observables}


def site_expectation_arrays(
    state, grouped_observables: dict[str, list[ObservableSpec]]
) -> dict[str, np.ndarray]:
    """Return ``{series_name: array(n_sites,)}`` for grouped local observables.

    Purpose:
        Package per-site expectations for lattice visualization without
        mixing them into scalar observable dicts.

    Inputs:
        state: ``Statevector`` or bound ``QuantumCircuit``.
        grouped_observables: Mapping of series name → ordered site specs
            (e.g. ``{\"z\": [z_0, z_1, ...]}``).

    Outputs:
        Dict of series name → 1-D numpy array of real expectations.
    """

    if not grouped_observables:
        return {}

    if isinstance(state, QuantumCircuit):
        state = Statevector(state)

    arrays = {}
    for series_name, specs in grouped_observables.items():
        arrays[series_name] = np.asarray(
            [state.expectation_value(spec.operator).real for spec in specs],
            dtype=float,
        )
    return arrays


def absolute_error(reference: float, value: float) -> float:
    """Return `|reference - value|`.

    Purpose:
        Shared absolute-error definition for energy and observable metrics.

    Inputs:
        reference: Exact or baseline scalar.
        value: Approximate scalar.

    Process:
        Compute the absolute difference.

    Outputs:
        Non-negative float.

    Side effects:
        None.
    """

    return abs(reference - value)


def relative_error_percent(reference: float, value: float) -> float:
    """Return `|reference - value| / |reference| * 100`.

    Purpose:
        Express error as a percentage of the reference magnitude (e.g.
        hardware energy error summaries).

    Inputs:
        reference: Exact or baseline scalar (must be non-zero).
        value: Approximate scalar.

    Process:
        Divide absolute error by ``|reference|`` and scale to percent.

    Outputs:
        Relative error in percent.

    Side effects:
        None. Division by zero if ``reference == 0``.
    """

    return abs(reference - value) / abs(reference) * 100
