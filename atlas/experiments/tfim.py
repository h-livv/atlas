"""TFIM-specific experiment orchestration.

Owns the TFIM-specific workflow: composing a Hamiltonian, ansatz, and
observables for a given ``(J, h)`` point, running the configured algorithm,
computing fidelity/observable metrics, and optionally running or merging IBM
hardware evaluation. Plotting and CSV writing are handled by
``atlas.experiments.outputs``.
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
    """Convert a pandas cell to ``float``, mapping missing/NaN values to ``None``."""

    if value is None:
        return None
    try:
        if value != value:
            return None
    except TypeError:
        pass
    return float(value)


class TFIMExperiment:
    """TFIM experiment workflow using YAML-configured components."""

    def __init__(self, config: AtlasConfig, algorithm):
        self.config = config
        self.algorithm = algorithm
        self.num_qubits = int(config.system.parameters["num_qubits"])
        self.resilience_level = config.hardware.resilience_level
        self.compute_fidelity = config.analysis.fidelity

    def _build_point_context(self, J: float, h: float):
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
        """Run exact diagonalization + VQE (and optionally hardware) for one point."""

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
        """Run exact spectrum + VQD (and optionally hardware on ground state)."""

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
        """Run VQE for every ``h`` in ``h_values``."""

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
        """Run VQD for every ``h`` in ``h_values``."""

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
