"""Command-line orchestration entry point for the Atlas quantum simulation framework.

This module is intentionally thin: it parses CLI arguments into an
`ExperimentConfig`, builds a `TFIMExperiment`, delegates all physics,
optimization, hardware execution, analysis, plotting, and CSV work to the
dedicated modules, and prints concise summaries mirroring the legacy
`vqe_legacy/tfim_main.py` (simulator single-point) and `vqe_legacy/tfim_vis.py`
(benchmark sweep) scripts. It never constructs circuits, Hamiltonians,
estimator pubs, or plots directly.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from atlas.config import (
    ExperimentConfig,
    HardwareConfig,
    OptimizerConfig,
    OutputConfig,
    SimulatorConfig,
    TFIMConfig,
)
from atlas.experiments.tfim import TFIMExperiment
from atlas.io.csv_io import write_tfim_benchmark
from atlas.visualization.tfim_plots import plot_all_tfim


def parse_args(argv=None) -> argparse.Namespace:
    """Parse CLI arguments for `main()`."""

    parser = argparse.ArgumentParser(
        description="Run the Atlas quantum simulation framework in simulator, hardware, or benchmark mode."
    )
    parser.add_argument(
        "--mode",
        choices=["sim", "hardware", "benchmark"],
        default="sim",
        help="Execution mode: single-point simulator, single-point hardware, or an h-sweep benchmark.",
    )
    parser.add_argument("--J", type=float, default=1.0, help="Coupling constant (used in all modes).")
    parser.add_argument(
        "--h", type=float, default=0.5, help="Transverse field strength (sim/hardware modes)."
    )
    parser.add_argument(
        "--h-values",
        type=str,
        default="0.1,0.5,1.0,1.5,2.0",
        help="Comma-separated transverse field values to sweep (benchmark mode).",
    )
    parser.add_argument("--num-qubits", type=int, default=2, help="Number of qubits in the TFIM chain.")
    parser.add_argument(
        "--num-starts", type=int, default=10, help="Number of multi-start VQE optimization restarts."
    )
    parser.add_argument(
        "--maxiter", type=int, default=200, help="Maximum optimizer iterations per start."
    )
    parser.add_argument(
        "--hardware-csv",
        type=str,
        default=None,
        help="Path to a pre-collected hardware results CSV to merge into a benchmark sweep.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OutputConfig().output_dir,
        help="Directory to write benchmark CSVs and plots into.",
    )
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    """Translate parsed CLI arguments into an `ExperimentConfig`."""

    return ExperimentConfig(
        tfim=TFIMConfig(num_qubits=args.num_qubits, J=args.J, h=args.h),
        optimizer=OptimizerConfig(num_starts=args.num_starts, maxiter=args.maxiter),
        simulator=SimulatorConfig(),
        hardware=HardwareConfig(hardware_csv_path=args.hardware_csv),
        output=OutputConfig(output_dir=args.output_dir),
    )


def _parse_h_values(raw: str) -> list[float]:
    """Parse a comma-separated string of floats, ignoring blank entries."""

    return [float(token) for token in raw.split(",") if token.strip()]


def _resolve_hardware_backend(num_qubits: int):
    """Return a least-busy operational non-simulator backend with enough qubits.

    Mirrors the commented-out backend selection in `vqe_legacy/tfim_vis.py`
    (`service.least_busy(operational=True, simulator=False, min_num_qubits=...)`).
    Returns `None` and prints an actionable message on any failure -- most
    commonly a missing/unconfigured IBM Quantum account -- instead of raising.
    """

    try:
        from qiskit_ibm_runtime import QiskitRuntimeService

        service = QiskitRuntimeService()
    except Exception as exc:
        print(
            "Could not reach the IBM Quantum Runtime service: no account/credentials "
            "are configured in this environment.\n"
            "To fix this, save an account first, e.g.:\n"
            '    QiskitRuntimeService.save_account(channel="ibm_quantum", token="<your token>")\n'
            f"(Underlying error: {exc})"
        )
        return None

    try:
        return service.least_busy(operational=True, simulator=False, min_num_qubits=num_qubits)
    except Exception as exc:
        print(
            f"Could not find an operational IBM Quantum backend with at least "
            f"{num_qubits} qubits.\n(Underlying error: {exc})"
        )
        return None


def run_sim(experiment: TFIMExperiment, args: argparse.Namespace) -> None:
    """Single-point simulator mode: mirrors `vqe_legacy/tfim_main.py` output."""

    print("=" * 50)
    print(f" {args.num_qubits}-Qubit TFIM Simulator (Single-Shot)")
    print(f" Parameters: J = {args.J}, h = {args.h}")
    print("=" * 50)

    print("Running Exact Classical Diagonalization...")
    print(f"Running Quantum VQE (Multi-Start: {args.num_starts})...")

    point = experiment.run_single_point(args.J, args.h, mode="sim")
    vqe_result = point.vqe_result
    error = abs(point.exact_energy - vqe_result.energy)

    print("\n--- Final Results ---")
    print(f"Analytical Ground State: {point.exact_energy:.6f}")
    print(f"VQE Computed Energy:     {vqe_result.energy:.6f}")
    print(f"Absolute Error:          {error:.6e}")
    print(f"Optimizer Iterations:    {vqe_result.nfev}")
    print(f"Optimal Thetas (rads):   {np.round(vqe_result.optimal_parameters, 4)}")


def run_hardware(experiment: TFIMExperiment, args: argparse.Namespace) -> None:
    """Single-point hardware mode: optimizes on simulator, then evaluates on IBM hardware."""

    print("=" * 50)
    print(f" {args.num_qubits}-Qubit TFIM Hardware Run")
    print(f" Parameters: J = {args.J}, h = {args.h}")
    print("=" * 50)

    backend = _resolve_hardware_backend(args.num_qubits)
    if backend is None:
        return

    print(f"Selected backend: {backend.name}")
    print(f"Running simulator VQE (Multi-Start: {args.num_starts}) for optimized parameters...")

    point = experiment.run_single_point(args.J, args.h, mode="hardware", backend=backend)
    hardware_result = point.hardware_result

    print("\n--- Hardware Results ---")
    print(f"Backend:         {hardware_result.backend}")
    print(f"Job ID:          {hardware_result.job_id}")
    print(f"Hardware Energy: {hardware_result.energy:.6f}")
    for name, value in hardware_result.observables.items():
        print(f"Hardware <{name}>:   {value:.6f}")


def run_benchmark(experiment: TFIMExperiment, args: argparse.Namespace) -> None:
    """Sweep mode: mirrors `vqe_legacy/tfim_vis.py`'s sweep + plots + CSV + summary."""

    h_values = _parse_h_values(args.h_values)

    print("=" * 50)
    print(f" {args.num_qubits}-Qubit TFIM Benchmark Sweep")
    print(f" Parameters: J = {args.J}, h values = {h_values}")
    print("=" * 50)

    result = experiment.run_sweep(
        args.J,
        h_values,
        include_hardware=bool(args.hardware_csv),
        hardware_source=args.hardware_csv,
    )

    print(f"\n{'h/J Ratio':<10} | {'Delta E':<12} | {'Fidelity':<10} | {'Iters'}")
    print("-" * 50)
    for point in result:
        delta_e = abs(point.exact_energy - point.vqe_result.energy)
        print(
            f"{point.h / args.J:<10.3f} | {delta_e:<12.2e} | "
            f"{point.fidelity:<10.4f} | {point.vqe_result.nfev}"
        )

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "tfim_benchmark.csv")
    write_tfim_benchmark(result, csv_path)
    plot_all_tfim(result, args.output_dir)

    fidelities = result.fidelities
    sim_abs_error = np.abs(result.vqe_energies - result.exact_energies)

    print("\nBenchmark Summary")
    print("-----------------")
    print(f"Mean fidelity:           {np.mean(fidelities):.5f}")
    print(f"Max fidelity:            {np.max(fidelities):.5f}")
    print(f"Mean sim abs error:      {np.mean(sim_abs_error):.5e}")
    print(f"Max sim abs error:       {np.max(sim_abs_error):.5e}")

    hardware_points = [p for p in result if p.hardware_result is not None]
    if hardware_points:
        hw_abs_error = np.array(
            [abs(p.hardware_result.energy - p.exact_energy) for p in hardware_points]
        )
        hw_rel_error = np.array(
            [
                abs(p.hardware_result.energy - p.exact_energy) / abs(p.exact_energy) * 100
                for p in hardware_points
            ]
        )
        print(f"Mean hardware abs error: {np.mean(hw_abs_error):.5f}")
        print(f"Max hardware abs error:  {np.max(hw_abs_error):.5f}")
        print(f"Mean relative error:     {np.mean(hw_rel_error):.2f}%")
        print(f"Max relative error:      {np.max(hw_rel_error):.2f}%")

    print(f"\nBenchmark CSV saved to: {csv_path}")
    print(f"Plots saved to:         {args.output_dir}")


def main(argv=None) -> None:
    args = parse_args(argv)
    config = build_config(args)
    experiment = TFIMExperiment(config)

    if args.mode == "sim":
        run_sim(experiment, args)
    elif args.mode == "hardware":
        run_hardware(experiment, args)
    elif args.mode == "benchmark":
        run_benchmark(experiment, args)


if __name__ == "__main__":
    main()
