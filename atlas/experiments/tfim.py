"""TFIM-specific experiment orchestration.

Owns the TFIM-specific workflow: composing a `TFIMHamiltonian`, ansatz, and
observables for a given `(J, h)` point, running exact diagonalization and
simulator VQE, computing fidelity/observable metrics, and optionally running
or merging IBM hardware evaluation. This module never plots and never writes
CSVs -- it only reads pre-collected hardware results via
`atlas.io.csv_io.load_hardware_results` so a sweep can be benchmarked against
data collected outside the current process. Plotting and CSV writing stay in
`atlas/visualization/` and `atlas/io/csv_io.py::write_tfim_benchmark`
respectively; those remain `main.py`'s responsibility.
"""

from __future__ import annotations

from typing import Optional, Sequence

from qiskit.quantum_info import Statevector

from atlas.algorithms.vqe import VQE
from atlas.analysis.metrics import expectation_values, state_fidelity_to_exact
from atlas.circuits.ansatzes.hardware_efficient import build_hardware_efficient_ansatz
from atlas.config import ExperimentConfig
from atlas.execution.ibm_runtime import IBMRuntimeEstimator
from atlas.execution.simulator import SimulatorEstimator
from atlas.experiments.results import (
    HardwareEvaluationResult,
    TFIMBenchmarkResult,
    TFIMPointResult,
)
from atlas.io.csv_io import load_hardware_results
from atlas.optimization.scipy_optimizer import ScipyOptimizer
from atlas.physics.hamiltonians import TFIMHamiltonian
from atlas.physics.observables import tfim_observables


def _optional_float(value) -> Optional[float]:
    """Convert a pandas cell to `float`, mapping missing/NaN values to `None`."""

    if value is None:
        return None
    try:
        if value != value:  # NaN check without importing numpy/pandas here.
            return None
    except TypeError:
        pass
    return float(value)


class TFIMExperiment:
    """Owns the TFIM-specific VQE experiment workflow.

    Holds everything needed to build a `TFIMHamiltonian`, ansatz, and
    optimizer/estimator stack for any `(J, h)` point: `num_qubits`, ansatz
    `reps`, and the optimizer/simulator settings from `config`. `J` and `h`
    themselves are per-call parameters (a single experiment spans a sweep of
    `h` values), not stored state.
    """

    def __init__(self, config: ExperimentConfig, ansatz_reps: int = 1):
        self.config = config
        self.num_qubits = config.tfim.num_qubits
        self.ansatz_reps = ansatz_reps
        self.optimizer_method = config.optimizer.method
        self.optimizer_maxiter = config.optimizer.maxiter
        self.num_starts = config.optimizer.num_starts
        self.seed = config.simulator.seed
        self.resilience_level = config.hardware.resilience_level

    def run_single_point(
        self,
        J: float,
        h: float,
        mode: str = "sim",
        backend=None,
    ) -> TFIMPointResult:
        """Run exact diagonalization + simulator VQE (and optionally hardware) for one point.

        `mode="hardware"` requires `backend` to be supplied; it evaluates the
        optimized simulator parameters on `backend` via `IBMRuntimeEstimator`
        and attaches the resulting `HardwareEvaluationResult`.
        """

        if mode not in ("sim", "hardware"):
            raise ValueError(f"Unknown mode '{mode}'; expected 'sim' or 'hardware'.")
        if mode == "hardware" and backend is None:
            raise ValueError("mode='hardware' requires a `backend` to be supplied.")

        hamiltonian = TFIMHamiltonian(num_qubits=self.num_qubits, J=J, h=h)
        ansatz = build_hardware_efficient_ansatz(
            num_qubits=self.num_qubits, reps=self.ansatz_reps
        )
        observables = tfim_observables(self.num_qubits)

        exact_result = hamiltonian.exact_ground_state()

        vqe = VQE(
            estimator=SimulatorEstimator(),
            optimizer=ScipyOptimizer(
                method=self.optimizer_method, maxiter=self.optimizer_maxiter
            ),
            num_starts=self.num_starts,
            seed=self.seed,
        )
        vqe_result = vqe.run(hamiltonian, ansatz)

        bound_circuit = ansatz.bind(vqe_result.optimal_parameters)
        vqe_state = Statevector(bound_circuit)

        fidelity = state_fidelity_to_exact(vqe_state, exact_result.statevector)
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

    def run_sweep(
        self,
        J: float,
        h_values: Sequence[float],
        include_hardware: bool = False,
        hardware_source: Optional[str] = None,
        backend=None,
    ) -> TFIMBenchmarkResult:
        """Run exact + simulator VQE for every `h` in `h_values`.

        Live hardware execution only happens per-point when the caller both
        opts in (`include_hardware=True`) and supplies a `backend`; otherwise
        every point stays simulator-only, matching the legacy script's
        default of loading pre-collected hardware CSV results instead of
        running hardware live. If `hardware_source` is given, it is loaded
        via `load_hardware_results()` and merged into each point's
        `hardware_result` by exact `h` match; points with no matching CSV row
        are left with `hardware_result=None`.
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

        if hardware_source is not None:
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

        return TFIMBenchmarkResult(points=points)
