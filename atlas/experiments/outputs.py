"""Save experiment outputs (CSV, plots) based on configuration."""

from __future__ import annotations

import os
from datetime import datetime

import numpy as np

from atlas.config import AtlasConfig
from atlas.experiments.factory import ExperimentRunResult
from atlas.experiments.results import (
    SimBenchmarkResult,
    SimPointResult,
    SimValidationResult,
    TFIMBenchmarkResult,
    TFIMPointResult,
    TFIMVQDBenchmarkResult,
    TFIMVQDPointResult,
)
from atlas.io.csv_io import write_sim_benchmark, write_sim_validation, write_tfim_benchmark
from atlas.visualization.sim_plots import plot_all_sim, plot_sim_single_point
from atlas.visualization.sim_validation import plot_validation_comparison, plot_validation_result
from atlas.visualization.tfim_plots import plot_all_tfim, plot_vqe_single_point
from atlas.visualization.tfim_vqd_plots import plot_all_tfim_vqd, plot_vqd_single_point


def resolve_run_output_dir(config: AtlasConfig) -> str:
    """Create a unique run directory under the configured output base path."""

    base_dir = config.output.output_dir
    name = config.experiment.name or f"{config.system.name}_{config.algorithm.name}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"{name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_outputs(run: ExperimentRunResult) -> None:
    """Write configured outputs for a completed experiment run."""

    config: AtlasConfig = run.config
    result = run.result
    output_dir = resolve_run_output_dir(config)

    print(f"\nRun output directory: {output_dir}")

    if isinstance(result, TFIMBenchmarkResult):
        _save_vqe_benchmark(config, result, output_dir)
    elif isinstance(result, TFIMPointResult):
        _save_vqe_single_point(config, result, output_dir)
    elif isinstance(result, TFIMVQDPointResult):
        _save_vqd_single_point(config, result, output_dir)
    elif isinstance(result, TFIMVQDBenchmarkResult):
        _save_vqd_benchmark(config, result, output_dir)
    elif isinstance(result, SimBenchmarkResult):
        _save_sim_benchmark(config, result, output_dir)
    elif isinstance(result, SimPointResult):
        _save_sim_single_point(config, result, output_dir)
    elif isinstance(result, SimValidationResult):
        _save_sim_validation(config, result, output_dir)


def _save_vqe_benchmark(config: AtlasConfig, result: TFIMBenchmarkResult, output_dir: str) -> None:
    if config.output.csv:
        csv_path = os.path.join(output_dir, "tfim_benchmark.csv")
        write_tfim_benchmark(result, csv_path)
        print(f"Benchmark CSV saved to: {csv_path}")

    if config.output.plots:
        plot_all_tfim(result, output_dir)
        print(f"Plots saved to: {output_dir}")

    _print_vqe_benchmark_summary(result, config)


def _save_vqe_single_point(
    config: AtlasConfig, point: TFIMPointResult, output_dir: str
) -> None:
    _print_vqe_single_point(point, config)

    if config.output.plots:
        plot_vqe_single_point(point, output_dir)
        print(f"Plots saved to: {output_dir}")


def _save_vqd_single_point(
    config: AtlasConfig, point: TFIMVQDPointResult, output_dir: str
) -> None:
    _print_vqd_single_point(point, config)

    if config.output.plots:
        plot_vqd_single_point(point, output_dir)
        print(f"Plots saved to: {output_dir}")


def _save_vqd_benchmark(
    config: AtlasConfig, result: TFIMVQDBenchmarkResult, output_dir: str
) -> None:
    _print_vqd_benchmark(result, config)

    if config.output.csv:
        _write_vqd_benchmark_csv(result, output_dir)

    if config.output.plots:
        plot_all_tfim_vqd(result, output_dir)
        print(f"Plots saved to: {output_dir}")


def _print_vqe_single_point(point: TFIMPointResult, config: AtlasConfig) -> None:
    params = config.system.parameters
    vqe_result = point.vqe_result
    error = abs(point.exact_energy - vqe_result.energy)

    print("=" * 50)
    print(f" {point.num_qubits}-Qubit TFIM Simulator (Single-Shot)")
    print(f" Parameters: J = {params['J']}, h = {params['h']}")
    print("=" * 50)
    print("\n--- Final Results ---")
    print(f"Analytical Ground State: {point.exact_energy:.6f}")
    print(f"VQE Computed Energy:     {vqe_result.energy:.6f}")
    print(f"Absolute Error:          {error:.6e}")
    print(f"Optimizer Iterations:    {vqe_result.nfev}")
    print(f"Optimal Thetas (rads):   {np.round(vqe_result.optimal_parameters, 4)}")


def _print_vqe_benchmark_summary(result: TFIMBenchmarkResult, config: AtlasConfig) -> None:
    J = float(config.system.parameters["J"])
    h_values = result.h_values

    print("=" * 50)
    print(f" {config.system.parameters['num_qubits']}-Qubit TFIM Benchmark Sweep")
    print(f" Parameters: J = {J}, h values = {list(h_values)}")
    print("=" * 50)

    print(f"\n{'h/J Ratio':<10} | {'Delta E':<12} | {'Fidelity':<10} | {'Iters'}")
    print("-" * 50)
    for point in result:
        delta_e = abs(point.exact_energy - point.vqe_result.energy)
        print(
            f"{point.h / J:<10.3f} | {delta_e:<12.2e} | "
            f"{point.fidelity:<10.4f} | {point.vqe_result.nfev}"
        )

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


def _print_vqd_single_point(point: TFIMVQDPointResult, config: AtlasConfig) -> None:
    params = config.system.parameters
    num_states = len(point.vqd_result.states)

    print("=" * 50)
    print(f" {point.num_qubits}-Qubit TFIM VQD (Single-Shot)")
    print(f" Parameters: J = {params['J']}, h = {params['h']}")
    print(f" States: {num_states}")
    print("=" * 50)

    print(
        f"\n{'State':<8} | {'Exact E':<12} | {'VQD E':<12} | "
        f"{'Abs Error':<12} | {'Fidelity':<10}"
    )
    print("-" * 62)
    for index, (exact_e, state_result, abs_error, fidelity) in enumerate(
        zip(
            point.exact_energies,
            point.vqd_result.states,
            point.absolute_errors,
            point.fidelities,
        )
    ):
        print(
            f"{index:<8} | {exact_e:<12.6f} | {state_result.energy:<12.6f} | "
            f"{abs_error:<12.2e} | {fidelity:<10.4f}"
        )


def _print_vqd_benchmark(result: TFIMVQDBenchmarkResult, config: AtlasConfig) -> None:
    J = float(config.system.parameters["J"])
    print("=" * 50)
    print(f" {config.system.parameters['num_qubits']}-Qubit TFIM VQD Benchmark Sweep")
    print(f" Parameters: J = {J}")
    print("=" * 50)

    for point in result.points:
        print(f"\nh = {point.h}")
        for index, (exact_e, state_result, abs_error, fidelity) in enumerate(
            zip(
                point.exact_energies,
                point.vqd_result.states,
                point.absolute_errors,
                point.fidelities,
            )
        ):
            print(
                f"  state {index}: exact={exact_e:.6f}, vqd={state_result.energy:.6f}, "
                f"abs_error={abs_error:.2e}, fid={fidelity:.4f}"
            )


def _write_vqd_benchmark_csv(result: TFIMVQDBenchmarkResult, output_dir: str) -> None:
    import pandas as pd

    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for point in result.points:
        for index, (exact_e, state_result, abs_error, fidelity) in enumerate(
            zip(
                point.exact_energies,
                point.vqd_result.states,
                point.absolute_errors,
                point.fidelities,
            )
        ):
            rows.append(
                {
                    "h": point.h,
                    "J": point.J,
                    "state_index": index,
                    "exact_energy": exact_e,
                    "vqd_energy": state_result.energy,
                    "absolute_error": abs_error,
                    "fidelity": fidelity,
                }
            )

    csv_path = os.path.join(output_dir, "tfim_vqd_benchmark.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"VQD benchmark CSV saved to: {csv_path}")


def _save_sim_single_point(
    config: AtlasConfig, point: SimPointResult, output_dir: str
) -> None:
    _print_sim_single_point(point, config)

    if config.output.plots:
        plot_sim_single_point(point, output_dir)
        print(f"Plots saved to: {output_dir}")


def _save_sim_benchmark(
    config: AtlasConfig, result: SimBenchmarkResult, output_dir: str
) -> None:
    _print_sim_benchmark(result, config)

    if config.output.csv:
        csv_path = os.path.join(output_dir, "sim_benchmark.csv")
        write_sim_benchmark(result, csv_path)
        print(f"Simulation benchmark CSV saved to: {csv_path}")

    if config.output.plots:
        plot_all_sim(result, output_dir)
        if result.sweep_parameter == "num_trotter_steps":
            plot_validation_comparison([result], output_dir)
        print(f"Plots saved to: {output_dir}")


def _print_sim_single_point(point: SimPointResult, config: AtlasConfig) -> None:
    params = config.system.parameters
    sim = point.sim_result

    print("=" * 50)
    print(f" {point.num_qubits}-Qubit TFIM Hamiltonian Simulation (Single-Shot)")
    print(f" Parameters: J = {params['J']}, h = {params['h']}")
    print(f" Evolution time: {point.evolution_time}")
    print(f" Method: {sim.method_name}, steps: {sim.num_trotter_steps}")
    print("=" * 50)
    print("\n--- Final Results ---")
    print(f"State fidelity:          {point.fidelity:.6f}")
    print(f"Circuit depth:           {sim.circuit_depth}")
    for name, value in point.observables.items():
        exact = point.exact_observables[name]
        error = point.observable_errors[name]
        print(
            f"<{name}>: sim={value:.6f}, exact={exact:.6f}, "
            f"abs_error={error:.6e}"
        )


def _print_sim_benchmark(result: SimBenchmarkResult, config: AtlasConfig) -> None:
    J = float(config.system.parameters["J"])
    print("=" * 50)
    print(
        f" {config.system.parameters['num_qubits']}-Qubit TFIM Simulation Benchmark"
    )
    print(f" Sweep parameter: {result.sweep_parameter}")
    print(f" Parameters: J = {J}")
    print("=" * 50)

    print(
        f"\n{result.sweep_parameter:<16} | {'Fidelity':<10} | "
        f"{'Depth':<8} | {'Method'}"
    )
    print("-" * 50)
    for point in result.points:
        sweep_value = getattr(point, result.sweep_parameter)
        print(
            f"{sweep_value:<16.4f} | {point.fidelity:<10.4f} | "
            f"{point.sim_result.circuit_depth:<8} | {point.sim_result.method_name}"
        )

    print("\nBenchmark Summary")
    print("-----------------")
    print(f"Mean fidelity:           {np.mean(result.fidelities):.5f}")
    print(f"Max fidelity:            {np.max(result.fidelities):.5f}")


def _save_sim_validation(
    config: AtlasConfig, result: SimValidationResult, output_dir: str
) -> None:
    _print_sim_validation(result, config)

    if config.output.csv:
        csv_path = os.path.join(output_dir, "sim_validation.csv")
        write_sim_validation(result, csv_path)
        print(f"Validation CSV saved to: {csv_path}")

    if config.output.plots:
        plot_validation_result(result, output_dir)
        print(f"Validation plots saved to: {output_dir}")


def _print_sim_validation(result: SimValidationResult, config: AtlasConfig) -> None:
    print("=" * 50)
    print(
        f" {config.system.parameters['num_qubits']}-Qubit TFIM Trotter Validation"
    )
    print(f" Parameters: J = {result.J}, h = {result.h}")
    print(f" Evolution time: {result.evolution_time}")
    print(f" Methods: {[series.method_name for series in result.series]}")
    print("=" * 50)

    for series in result.series:
        print(f"\nMethod: {series.method_name}")
        print(f"{'Steps':<8} | {'Fidelity':<10} | {'Max Op Err':<12} | {'Depth'}")
        print("-" * 50)
        for point in series.points:
            print(
                f"{point.num_trotter_steps:<8} | {point.fidelity:<10.4f} | "
                f"{point.max_operator_error:<12.2e} | {point.circuit_depth}"
            )
