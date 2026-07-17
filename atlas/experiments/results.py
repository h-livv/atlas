"""Structured result containers shared by experiment, analysis, I/O, and
plotting code.

These replace the parallel arrays used throughout `vqe_legacy/tfim_vis.py`
with explicit, self-describing records. `physics/` must never import from
this module -- dependencies point from `experiments/` toward `physics/`,
never the other way around.

Architectural role:
    Pure data containers and light array accessors. No optimization, no I/O,
    and no plotting. Algorithms and experiments populate these types;
    visualization and CSV writers consume them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class VQEResult:
    """Outcome of a (multi-start) VQE optimization run.

    Responsibility:
        Carry the best energy and parameters found by multi-start VQE, plus
        bookkeeping fields for reporting and CSV export.

    State:
        energy: Best variational energy found.
        optimal_parameters: Parameter vector for that best start.
        nfev: Function evaluations for the winning start (not summed).
        num_starts: How many starts were attempted.
        num_qubits: System size for downstream labeling.
        optimizer_method: Classical optimizer name (e.g. COBYLA).

    Usage:
        Returned by ``VQE.run`` / ``VQE._optimize_cost``; embedded in
        ``TFIMPointResult`` and listed inside ``VQDResult.states``.
    """

    energy: float
    optimal_parameters: np.ndarray
    nfev: int
    num_starts: int
    num_qubits: int
    optimizer_method: str


@dataclass
class VQDResult:
    """Outcome of a sequential VQD run (one ``VQEResult`` per eigenstate).

    Responsibility:
        Hold the ordered list of deflated eigenstate optimizations so
        experiments can compare energies and fidelities per state index.

    State:
        states: List of ``VQEResult`` from ground state upward.
        num_qubits: System size.

    Usage:
        Returned by ``VQD.run``. Use ``energies``, ``ground_state_result``,
        or index into ``states`` for excited levels.
    """

    states: list
    num_qubits: int

    @property
    def energies(self) -> np.ndarray:
        """Return variational energies for all recovered states.

        Purpose:
            Compact array for plotting or error calculations.

        Outputs:
            1-D ``numpy`` array of energies in state order.

        Side effects:
            None.
        """

        return np.array([state.energy for state in self.states])

    @property
    def ground_state_result(self) -> VQEResult:
        """Return the ground-state ``VQEResult`` (first deflation step).

        Purpose:
            Convenience for hardware evaluation and ground-state observables,
            which always target state index 0.

        Outputs:
            ``self.states[0]``.

        Side effects:
            None.
        """

        return self.states[0]


@dataclass
class HardwareEvaluationResult:
    """Outcome of evaluating a bound ansatz on IBM hardware.

    Responsibility:
        Store hardware energy and named observable estimates (and optional
        stderr) from a Runtime job or an offline CSV merge.

    State:
        backend: Backend name, or ``"csv"`` for offline merges.
        energy: Estimated Hamiltonian expectation.
        observables: Mapping of observable name → estimate.
        job_id: Runtime job id (empty string for CSV).
        energy_stderr: Optional stderr on energy.
        observable_stderr: Optional mapping of name → stderr.

    Usage:
        Attached to ``TFIMPointResult.hardware_result`` / VQD point results;
        plotted and summarized by outputs/visualization code.
    """

    backend: str
    energy: float
    observables: dict
    job_id: str
    energy_stderr: Optional[float] = None
    observable_stderr: Optional[dict] = None


@dataclass
class TFIMPointResult:
    """All results (exact, simulated, optionally hardware) for one ``(J, h)`` point.

    Responsibility:
        Bundle classical ground truth with VQE (and optional hardware) so a
        single point can drive console summaries and single-point plots.

    State:
        h, J, num_qubits: Physics/system labels.
        exact_energy, exact_state: Exact diagonalization reference.
        vqe_result: Variational optimization outcome.
        fidelity: State fidelity (or NaN if disabled).
        observables: Sim expectation values by name.
        hardware_result: Optional hardware/CSV evaluation.

    Usage:
        Produced by ``TFIMExperiment.run_single_point``; collected into
        ``TFIMBenchmarkResult`` for sweeps.
    """

    h: float
    J: float
    num_qubits: int
    exact_energy: float
    exact_state: np.ndarray
    vqe_result: VQEResult
    fidelity: float
    observables: dict
    hardware_result: Optional[HardwareEvaluationResult] = None


@dataclass
class TFIMVQDPointResult:
    """VQD results for one ``(J, h)`` point with per-state fidelities and errors.

    Responsibility:
        Carry exact vs VQD spectra at one field value, plus ground-state
        observables and optional hardware on the ground state.

    State:
        h, J, num_qubits: Labels.
        exact_energies, exact_states: Classical spectrum references.
        vqd_result: Full sequential VQD outcome.
        fidelities, absolute_errors: Per-state metrics aligned with states.
        observables: Ground-state sim expectations.
        hardware_result: Optional ground-state hardware eval.

    Usage:
        Produced by ``TFIMExperiment.run_single_point_vqd``; aggregated in
        ``TFIMVQDBenchmarkResult``.
    """

    h: float
    J: float
    num_qubits: int
    exact_energies: list
    exact_states: list
    vqd_result: VQDResult
    fidelities: list
    absolute_errors: list
    observables: dict
    hardware_result: Optional[HardwareEvaluationResult] = None


@dataclass
class TFIMVQDBenchmarkResult:
    """A collection of ``TFIMVQDPointResult`` for a full ``h`` sweep.

    Responsibility:
        Provide sequence semantics and per-state column extractors for VQD
        benchmark plots and CSV export.

    State:
        points: List of ``TFIMVQDPointResult`` along the sweep.

    Usage:
        Iterate like a list; use ``state_*`` helpers with a state index to
        build arrays across ``h``.
    """

    points: list = field(default_factory=list)

    def __len__(self) -> int:
        """Return the number of sweep points.

        Outputs:
            ``len(self.points)``.
        """

        return len(self.points)

    def __iter__(self):
        """Iterate over point results in sweep order.

        Outputs:
            Iterator over ``TFIMVQDPointResult``.
        """

        return iter(self.points)

    @property
    def h_values(self) -> np.ndarray:
        """Return transverse-field values for each sweep point.

        Outputs:
            1-D array of ``h``.
        """

        return np.array([p.h for p in self.points])

    @property
    def num_states(self) -> int:
        """Return how many VQD states were recovered (0 if empty).

        Purpose:
            Drive plot loops without assuming a non-empty sweep.

        Outputs:
            Integer state count from the first point, or 0.
        """

        if not self.points:
            return 0
        return len(self.points[0].vqd_result.states)

    def state_exact_energies(self, state_index: int) -> np.ndarray:
        """Return exact energies for one state index across the sweep.

        Inputs:
            state_index: Eigenstate index (0 = ground).

        Outputs:
            1-D array aligned with ``h_values``.
        """

        return np.array([p.exact_energies[state_index] for p in self.points])

    def state_vqd_energies(self, state_index: int) -> np.ndarray:
        """Return VQD energies for one state index across the sweep.

        Inputs:
            state_index: Eigenstate index (0 = ground).

        Outputs:
            1-D array aligned with ``h_values``.
        """

        return np.array([p.vqd_result.states[state_index].energy for p in self.points])

    def state_fidelities(self, state_index: int) -> np.ndarray:
        """Return fidelities for one state index across the sweep.

        Inputs:
            state_index: Eigenstate index (0 = ground).

        Outputs:
            1-D array aligned with ``h_values``.
        """

        return np.array([p.fidelities[state_index] for p in self.points])

    def state_absolute_errors(self, state_index: int) -> np.ndarray:
        """Return absolute energy errors for one state index across the sweep.

        Inputs:
            state_index: Eigenstate index (0 = ground).

        Outputs:
            1-D array aligned with ``h_values``.
        """

        return np.array([p.absolute_errors[state_index] for p in self.points])


@dataclass
class TFIMBenchmarkResult:
    """A collection of `TFIMPointResult` for a full `h` sweep.

    Responsibility:
        Expose column-like arrays (energies, fidelities, hardware) for VQE
        benchmark plotting and CSV writing.

    State:
        points: List of ``TFIMPointResult`` along the sweep.

    Usage:
        Iterate like a list; use properties/helpers instead of manual
        comprehensions in visualization code.
    """

    points: list = field(default_factory=list)

    def __len__(self) -> int:
        """Return the number of sweep points.

        Outputs:
            ``len(self.points)``.
        """

        return len(self.points)

    def __iter__(self):
        """Iterate over point results in sweep order.

        Outputs:
            Iterator over ``TFIMPointResult``.
        """

        return iter(self.points)

    @property
    def h_values(self) -> np.ndarray:
        """Return transverse-field values for each sweep point."""

        return np.array([p.h for p in self.points])

    @property
    def J_values(self) -> np.ndarray:
        """Return coupling values for each sweep point (often constant)."""

        return np.array([p.J for p in self.points])

    @property
    def exact_energies(self) -> np.ndarray:
        """Return exact ground energies across the sweep."""

        return np.array([p.exact_energy for p in self.points])

    @property
    def vqe_energies(self) -> np.ndarray:
        """Return VQE energies across the sweep."""

        return np.array([p.vqe_result.energy for p in self.points])

    @property
    def fidelities(self) -> np.ndarray:
        """Return state fidelities across the sweep."""

        return np.array([p.fidelity for p in self.points])

    @property
    def iterations(self) -> np.ndarray:
        """Return optimizer ``nfev`` for each point's winning VQE start."""

        return np.array([p.vqe_result.nfev for p in self.points])

    def observable_values(self, name: str) -> np.ndarray:
        """Return one named sim observable across the sweep.

        Inputs:
            name: Observable key (e.g. ``"zz"``, ``"x"``).

        Outputs:
            1-D array (entries may be ``None`` if missing).
        """

        return np.array([p.observables.get(name) for p in self.points])

    @property
    def hardware_energies(self) -> np.ndarray:
        """Return hardware energies, or ``None`` where hardware is absent.

        Purpose:
            Keep array length aligned with the sweep even when only some
            points have hardware data.
        """

        return np.array(
            [p.hardware_result.energy if p.hardware_result else None for p in self.points]
        )

    def hardware_observable_values(self, name: str) -> np.ndarray:
        """Return one named hardware observable across the sweep.

        Inputs:
            name: Observable key.

        Outputs:
            1-D array with ``None`` where hardware or the key is missing.
        """

        return np.array(
            [
                p.hardware_result.observables.get(name) if p.hardware_result else None
                for p in self.points
            ]
        )


@dataclass
class SimulationResult:
    """Outcome of a single Hamiltonian simulation run.

    Responsibility:
        Record the evolved statevector and circuit metadata from one
        product-formula + evolver execution.

    State:
        statevector: Complex amplitudes after evolution.
        num_qubits: System size.
        evolution_time: Total evolution time used.
        method_name: Product-formula name (e.g. ``lie``, ``strang``).
        num_trotter_steps: Discretization count.
        circuit_depth: Depth of the compiled evolution circuit.

    Usage:
        Returned by ``HamiltonianSimulation.run``; embedded in
        ``SimPointResult.sim_result``.
    """

    statevector: np.ndarray
    num_qubits: int
    evolution_time: float
    method_name: str
    num_trotter_steps: int
    circuit_depth: int


@dataclass
class SimPointResult:
    """Exact and simulated dynamics results for one parameter point.

    Responsibility:
        Compare one Trotterized evolution against exact Schrödinger
        evolution at fixed ``(J, h, t)``.

    State:
        h, J, num_qubits, evolution_time: Labels.
        exact_state: Exact evolved statevector.
        sim_result: Simulated ``SimulationResult``.
        fidelity: Overlap metric (or NaN if disabled).
        observables / exact_observables: Named expectations.
        observable_errors: Absolute errors per observable name.

    Usage:
        Produced by ``HamiltonianSimExperiment``; collected into
        ``SimBenchmarkResult`` or validation series.
    """

    h: float
    J: float
    num_qubits: int
    evolution_time: float
    exact_state: np.ndarray
    sim_result: SimulationResult
    fidelity: float
    observables: dict
    exact_observables: dict
    observable_errors: dict

    @property
    def num_trotter_steps(self) -> int:
        """Forward ``sim_result.num_trotter_steps`` for sweep accessors."""

        return self.sim_result.num_trotter_steps

    @property
    def circuit_depth(self) -> int:
        """Forward ``sim_result.circuit_depth`` for plots and tables."""

        return self.sim_result.circuit_depth

    @property
    def method_name(self) -> str:
        """Forward ``sim_result.method_name`` for labeling series."""

        return self.sim_result.method_name

    @property
    def max_operator_error(self) -> float:
        """Return the largest absolute observable error at this point.

        Purpose:
            Single scalar summary for validation tables/plots.

        Outputs:
            ``max(observable_errors.values())``, or NaN if empty.
        """

        if not self.observable_errors:
            return float("nan")
        return float(max(self.observable_errors.values()))


@dataclass
class SimBenchmarkResult:
    """A collection of ``SimPointResult`` for a parameter sweep.

    Responsibility:
        Hold dynamics points along one swept axis and expose metric columns
        for plotting and CSV export.

    State:
        sweep_parameter: Name of the swept field (``evolution_time``, ``h``,
            or ``num_trotter_steps``).
        points: Ordered ``SimPointResult`` list.

    Usage:
        Produced by ``run_sweep`` / validation series. Use ``sweep_values``
        and ``metric_values`` rather than hand-rolled loops when possible.
    """

    sweep_parameter: str = "evolution_time"
    points: list = field(default_factory=list)

    def __len__(self) -> int:
        """Return the number of sweep points."""

        return len(self.points)

    def __iter__(self):
        """Iterate over ``SimPointResult`` in sweep order."""

        return iter(self.points)

    @property
    def sweep_values(self) -> np.ndarray:
        """Return the swept coordinate for each point.

        Purpose:
            Unify access whether the axis is a top-level field or nested
            under ``num_trotter_steps``.

        Process:
            If sweeping steps, read ``p.num_trotter_steps``; otherwise
            ``getattr(p, sweep_parameter)``.
        """

        if self.sweep_parameter == "num_trotter_steps":
            return np.array([p.num_trotter_steps for p in self.points])
        return np.array([getattr(p, self.sweep_parameter) for p in self.points])

    @property
    def method_name(self) -> str:
        """Return the method name from the first point (or empty if none).

        Purpose:
            Label a validation series; assumes one method per series.
        """

        if not self.points:
            return ""
        return self.points[0].method_name

    def metric_values(self, metric: str) -> np.ndarray:
        """Return a named metric column across the sweep.

        Purpose:
            Single dispatcher for validation/comparison plot registries.

        Inputs:
            metric: One of ``fidelity``, ``max_operator_error``,
                ``circuit_depth``, ``num_trotter_steps``, or
                ``operator_error:<name>``.

        Outputs:
            1-D array of metric values.

        Side effects:
            None. Raises ``ValueError`` for unknown metric names.
        """

        if metric == "fidelity":
            return self.fidelities
        if metric == "max_operator_error":
            return np.array([p.max_operator_error for p in self.points])
        if metric == "circuit_depth":
            return np.array([p.circuit_depth for p in self.points])
        if metric == "num_trotter_steps":
            return np.array([p.num_trotter_steps for p in self.points])
        if metric.startswith("operator_error:"):
            name = metric.split(":", 1)[1]
            return np.array([p.observable_errors.get(name, float("nan")) for p in self.points])
        raise ValueError(f"Unknown benchmark metric '{metric}'.")

    @property
    def fidelities(self) -> np.ndarray:
        """Return fidelities for each sweep point."""

        return np.array([p.fidelity for p in self.points])

    def observable_values(self, name: str) -> np.ndarray:
        """Return simulated expectations for one observable across the sweep.

        Inputs:
            name: Observable key.
        """

        return np.array([p.observables.get(name) for p in self.points])

    def exact_observable_values(self, name: str) -> np.ndarray:
        """Return exact expectations for one observable across the sweep.

        Inputs:
            name: Observable key.
        """

        return np.array([p.exact_observables.get(name) for p in self.points])


@dataclass
class SimValidationResult:
    """Trotter-step validation across one or more evolution methods.

    Responsibility:
        Group multiple ``SimBenchmarkResult`` series (one per method) at
        fixed physics parameters for comparison plots and CSV export.

    State:
        h, J, evolution_time: Fixed physics/time labels.
        series: List of ``SimBenchmarkResult``, each typically sweeping
            ``num_trotter_steps`` for one method.

    Usage:
        Produced by ``HamiltonianSimExperiment.run_trotter_validation``;
        consumed by validation plotters and ``write_sim_validation``.
    """

    h: float
    J: float
    evolution_time: float
    series: list = field(default_factory=list)

    def __len__(self) -> int:
        """Return the number of method series."""

        return len(self.series)

    def __iter__(self):
        """Iterate over per-method ``SimBenchmarkResult`` series."""

        return iter(self.series)
