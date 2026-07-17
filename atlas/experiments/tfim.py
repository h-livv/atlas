"""TFIM-specific experiment orchestration.

Owns the TFIM-specific workflow: composing a Hamiltonian, ansatz, and
observables for a given ``(J, h)`` point, running the configured algorithm,
computing fidelity/observable metrics, and optionally running or merging IBM
hardware evaluation. Plotting and CSV writing are handled by
``atlas.experiments.outputs``.

Architectural role:
    Called by ``ConfiguredExperimentRunner`` after builders wire the algorithm.
    Depends on physics builders and analysis metrics; does not own classical
    optimization internals (those live in ``atlas.algorithms``).
"""

from __future__ import annotations

from typing import Optional, Sequence

from qiskit.quantum_info import Statevector

from atlas.analysis.metrics import expectation_values, state_fidelity_to_exact
from atlas.config import AtlasConfig
from atlas.execution.ibm_runtime import IBMRuntimeEstimator
from atlas.experiments.builders import (
    build_ansatz,
    build_hamiltonian,
    build_observables,
)
from atlas.experiments.results import (
    HardwareEvaluationResult,
    TFIMBenchmarkResult,
    TFIMPointResult,
    TFIMVQDBenchmarkResult,
    TFIMVQDPointResult,
)
from atlas.io.csv_io import load_hardware_results


def _optional_float(value) -> Optional[float]:
    """Convert a pandas cell to ``float``, mapping missing/NaN values to ``None``.

    Purpose:
        Hardware CSV columns for stderr may be empty or NaN; callers want
        ``Optional[float]`` rather than crashing on missing data.

    Inputs:
        value: Cell value from a DataFrame (may be ``None``, NaN, or numeric).

    Process:
        Return ``None`` for ``None`` or NaN (detected via ``value != value``).
        Otherwise cast to ``float``. TypeErrors during NaN checks are ignored
        so non-numeric types still attempt ``float(value)``.

    Outputs:
        ``float`` or ``None``.

    Side effects:
        None.
    """

    if value is None:
        return None
    try:
        # NaN is the unique float with value != value.
        if value != value:
            return None
    except TypeError:
        pass
    return float(value)


class TFIMExperiment:
    """TFIM experiment workflow using YAML-configured components.

    Responsibility:
        For each transverse-field Ising point ``(J, h)``, build the model and
        ansatz, run VQE or VQD against exact diagonalization references,
        compute fidelity/observables, and optionally evaluate on hardware or
        merge offline hardware CSV rows into sweep results.

    State:
        config: ``AtlasConfig`` driving builders and analysis flags.
        algorithm: Injected ``VQE`` or ``VQD`` instance.
        num_qubits: From ``config.system.parameters``.
        resilience_level: IBM Runtime resilience setting for hardware eval.
        compute_fidelity: Whether to compute state fidelity (else NaN).

    Usage:
        Construct via ``build_experiment`` / factory helpers. Call
        ``run_single_point`` / ``run_single_point_vqd`` or the sweep variants.
        Do not call plotting/CSV writers from here.
    """

    def __init__(self, config: AtlasConfig, algorithm):
        """Store config, algorithm, and cached analysis/hardware settings.

        Inputs:
            config: Full experiment configuration.
            algorithm: Configured ``VQE`` or ``VQD`` instance.
        """

        self.config = config
        self.algorithm = algorithm
        self.num_qubits = int(config.system.parameters["num_qubits"])
        self.resilience_level = config.hardware.resilience_level
        self.compute_fidelity = config.analysis.fidelity

    def _build_point_context(self, J: float, h: float):
        """Build Hamiltonian, ansatz, and observables for one ``(J, h)`` point.

        Purpose:
            Share identical construction between VQE and VQD entry points.

        Inputs:
            J: Coupling strength override for this point.
            h: Transverse field override for this point.

        Process:
            Call builders with parameter overrides for ``J``/``h`` and the
            experiment's qubit count.

        Outputs:
            Tuple ``(hamiltonian, ansatz, observables)``.

        Side effects:
            None.
        """

        hamiltonian = build_hamiltonian(
            self.config, parameter_overrides={"J": J, "h": h}
        )
        ansatz = build_ansatz(self.config, self.num_qubits)
        observables = build_observables(self.config, self.num_qubits)
        return hamiltonian, ansatz, observables

    def run_single_point(
        self,
        J: float,
        h: float,
        mode: str = "sim",
        backend=None,
    ) -> TFIMPointResult:
        """Run exact diagonalization + VQE (and optionally hardware) for one point.

        Purpose:
            Produce a complete ``TFIMPointResult`` comparing classical ground
            truth to the variational estimate (and optional hardware energy).

        Inputs:
            J: Ising coupling.
            h: Transverse field.
            mode: ``"sim"`` (default) or ``"hardware"``.
            backend: Required when ``mode="hardware"``.

        Process:
            Build point context; exact ground state; run VQE; bind optimal
            parameters and form a statevector; optionally compute fidelity
            and always compute sim observables; if hardware mode, evaluate
            energy/observables via ``IBMRuntimeEstimator``.

        Outputs:
            ``TFIMPointResult``.

        Side effects:
            Runs classical diagonalization and VQE optimization; may submit
            IBM Runtime jobs in hardware mode. Raises ``ValueError`` for
            invalid mode/backend combinations.
        """

        if mode not in ("sim", "hardware"):
            raise ValueError(f"Unknown mode '{mode}'; expected 'sim' or 'hardware'.")
        if mode == "hardware" and backend is None:
            raise ValueError("mode='hardware' requires a `backend` to be supplied.")

        hamiltonian, ansatz, observables = self._build_point_context(J, h)
        exact_result = hamiltonian.exact_ground_state()
        vqe_result = self.algorithm.run(hamiltonian, ansatz)

        bound_circuit = ansatz.bind(vqe_result.optimal_parameters)
        vqe_state = Statevector(bound_circuit)

        fidelity = (
            state_fidelity_to_exact(vqe_state, exact_result.statevector)
            if self.compute_fidelity
            else float("nan")
        )
        sim_observables = expectation_values(vqe_state, observables)

        hardware_result: Optional[HardwareEvaluationResult] = None
        if mode == "hardware":
            hardware_estimator = IBMRuntimeEstimator(
                backend, resilience_level=self.resilience_level
            )
            hardware_result = hardware_estimator.evaluate(
                ansatz, hamiltonian, vqe_result.optimal_parameters, observables
            )

        return TFIMPointResult(
            h=h,
            J=J,
            num_qubits=self.num_qubits,
            exact_energy=exact_result.energy,
            exact_state=exact_result.statevector,
            vqe_result=vqe_result,
            fidelity=fidelity,
            observables=sim_observables,
            hardware_result=hardware_result,
        )

    def run_single_point_vqd(
        self,
        J: float,
        h: float,
        mode: str = "sim",
        backend=None,
    ) -> TFIMVQDPointResult:
        """Run exact spectrum + VQD (and optionally hardware on ground state).

        Purpose:
            Compare several low-lying exact eigenpairs to sequential VQD
            states, plus ground-state observables (and optional hardware).

        Inputs:
            J: Ising coupling.
            h: Transverse field.
            mode: ``"sim"`` or ``"hardware"``.
            backend: Required for hardware mode.

        Process:
            Exact spectrum for ``algorithm.num_states``; run VQD; per state,
            compute absolute energy error and optional fidelity; evaluate
            observables on the VQD ground state; hardware evaluation (if any)
            uses only ground-state optimal parameters.

        Outputs:
            ``TFIMVQDPointResult``.

        Side effects:
            Multiple VQE-like optimizations inside VQD; optional hardware job.
            Raises ``ValueError`` for invalid mode/backend.
        """

        if mode not in ("sim", "hardware"):
            raise ValueError(f"Unknown mode '{mode}'; expected 'sim' or 'hardware'.")
        if mode == "hardware" and backend is None:
            raise ValueError("mode='hardware' requires a `backend` to be supplied.")

        hamiltonian, ansatz, observables = self._build_point_context(J, h)
        num_states = self.algorithm.num_states
        exact_pairs = hamiltonian.exact_spectrum(num_states)
        vqd_result = self.algorithm.run(hamiltonian, ansatz)

        fidelities = []
        absolute_errors = []
        for index, state_result in enumerate(vqd_result.states):
            bound = ansatz.bind(state_result.optimal_parameters)
            v_state = Statevector(bound)
            exact_energy = exact_pairs[index].energy
            absolute_errors.append(abs(exact_energy - state_result.energy))
            if self.compute_fidelity:
                fidelities.append(
                    state_fidelity_to_exact(v_state, exact_pairs[index].statevector)
                )
            else:
                fidelities.append(float("nan"))

        ground_state = Statevector(
            ansatz.bind(vqd_result.ground_state_result.optimal_parameters)
        )
        sim_observables = expectation_values(ground_state, observables)

        hardware_result: Optional[HardwareEvaluationResult] = None
        if mode == "hardware":
            hardware_estimator = IBMRuntimeEstimator(
                backend, resilience_level=self.resilience_level
            )
            hardware_result = hardware_estimator.evaluate(
                ansatz,
                hamiltonian,
                vqd_result.ground_state_result.optimal_parameters,
                observables,
            )

        return TFIMVQDPointResult(
            h=h,
            J=J,
            num_qubits=self.num_qubits,
            exact_energies=[pair.energy for pair in exact_pairs],
            exact_states=[pair.statevector for pair in exact_pairs],
            vqd_result=vqd_result,
            fidelities=fidelities,
            absolute_errors=absolute_errors,
            observables=sim_observables,
            hardware_result=hardware_result,
        )

    def run_sweep(
        self,
        J: float,
        h_values: Sequence[float],
        include_hardware: bool = False,
        hardware_source: Optional[str] = None,
        backend=None,
    ) -> TFIMBenchmarkResult:
        """Run VQE for every ``h`` in ``h_values``.

        Purpose:
            Produce a benchmark curve of energies/fidelities vs transverse
            field at fixed ``J``.

        Inputs:
            J: Fixed coupling.
            h_values: Sequence of transverse-field values.
            include_hardware: If True and ``backend`` is set, each point runs
                in hardware mode.
            hardware_source: Optional path to offline hardware CSV to merge
                after the sim (or live) sweep.
            backend: IBM backend for live hardware points.

        Process:
            Loop ``run_single_point`` with mode hardware or sim; then merge
            CSV hardware rows when ``hardware_source`` is provided.

        Outputs:
            ``TFIMBenchmarkResult`` collecting all points.

        Side effects:
            Many VQE runs; optional live hardware jobs; may read a CSV via
            ``_merge_hardware_csv``.
        """

        live_hardware = include_hardware and backend is not None

        points = [
            self.run_single_point(
                J,
                h,
                mode="hardware" if live_hardware else "sim",
                backend=backend if live_hardware else None,
            )
            for h in h_values
        ]

        self._merge_hardware_csv(points, hardware_source)
        return TFIMBenchmarkResult(points=points)

    def run_sweep_vqd(
        self,
        J: float,
        h_values: Sequence[float],
        include_hardware: bool = False,
        hardware_source: Optional[str] = None,
        backend=None,
    ) -> TFIMVQDBenchmarkResult:
        """Run VQD for every ``h`` in ``h_values``.

        Purpose:
            Benchmark low-lying spectrum recovery across a transverse-field
            sweep.

        Inputs:
            J: Fixed coupling.
            h_values: Transverse-field grid.
            include_hardware: Live hardware when True and ``backend`` set.
            hardware_source: Accepted for API symmetry with VQE sweep; not
                merged in the current implementation.
            backend: IBM backend for live evaluation.

        Process:
            Loop ``run_single_point_vqd`` with hardware or sim mode.

        Outputs:
            ``TFIMVQDBenchmarkResult``.

        Side effects:
            Many VQD runs; optional hardware jobs. Does not merge CSV today.
        """

        live_hardware = include_hardware and backend is not None

        points = [
            self.run_single_point_vqd(
                J,
                h,
                mode="hardware" if live_hardware else "sim",
                backend=backend if live_hardware else None,
            )
            for h in h_values
        ]

        return TFIMVQDBenchmarkResult(points=points)

    def _merge_hardware_csv(self, points, hardware_source: Optional[str]) -> None:
        """Attach offline hardware CSV rows onto matching sweep points in place.

        Purpose:
            Allow plotting/hardware-error summaries without re-running jobs,
            by joining previously saved IBM results on the ``h`` column.

        Inputs:
            points: Mutable list of ``TFIMPointResult`` (modified in place).
            hardware_source: CSV path, or ``None`` to no-op.

        Process:
            Load the CSV; for each point with a matching ``h`` row, overwrite
            ``point.hardware_result`` with energy/observables/stderr fields
            and ``backend="csv"``.

        Outputs:
            None.

        Side effects:
            Mutates ``points``; reads from the filesystem when
            ``hardware_source`` is set.
        """

        if hardware_source is None:
            return

        hardware_df = load_hardware_results(hardware_source)
        for point in points:
            matching_rows = hardware_df[hardware_df["h"] == point.h]
            if matching_rows.empty:
                continue
            row = matching_rows.iloc[0]
            point.hardware_result = HardwareEvaluationResult(
                backend="csv",
                energy=float(row["energy"]),
                observables={"zz": float(row["zz"]), "x": float(row["x"])},
                job_id="",
                energy_stderr=_optional_float(row.get("energy_stderr")),
                observable_stderr={
                    "zz": _optional_float(row.get("zz_stderr")),
                    "x": _optional_float(row.get("x_stderr")),
                },
            )
