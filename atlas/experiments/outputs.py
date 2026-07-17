"""Save experiment outputs (CSV, plots) based on configuration.

After ``run_experiment`` returns an ``ExperimentRunResult``, this module
creates a timestamped run directory, optionally writes CSVs, generates
plots, and prints human-readable summaries. It does not run algorithms;
it only persists and reports what experiments already computed.

Architectural role:
    Terminal side of the pipeline: depends on result containers, I/O helpers,
    and visualization modules. Experiment workflows should not import this
    module (keep computation separate from presentation).
"""

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
from atlas.visualization.sim_plots import (
    plot_all_sim,
    plot_all_sim_comparison,
    plot_sim_single_point,
)
from atlas.visualization.sim_validation import plot_validation_comparison
from atlas.visualization.hamiltonian_sim import (
    LatticeDashboard,
    available_site_observables,
    trajectory_from_sim_benchmark,
    trajectory_from_sim_point,
)
from atlas.visualization.tfim_plots import plot_all_tfim, plot_vqe_single_point
from atlas.visualization.tfim_vqd_plots import plot_all_tfim_vqd, plot_vqd_single_point


def resolve_run_output_dir(config: AtlasConfig) -> str:
    """Create a unique run directory under the configured output base path.

    Purpose:
        Avoid overwriting previous runs by appending a timestamp to the
        experiment name.

    Inputs:
        config: Uses ``output.output_dir`` and ``experiment.name`` (falls
            back to ``{system}_{algorithm}`` when name is empty).

    Process:
        Build ``{base}/{name}_{YYYYMMDD_HHMMSS}`` and ``makedirs``.

    Outputs:
        Absolute or relative path string of the created directory.

    Side effects:
        Creates directories on disk.
    """

    base_dir = config.output.output_dir
    name = config.experiment.name or f"{config.system.name}_{config.algorithm.name}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"{name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_outputs(run: ExperimentRunResult) -> None:
    """Write configured outputs for a completed experiment run.

    Purpose:
        Dispatch to type-specific save helpers based on the result class.

    Inputs:
        run: ``ExperimentRunResult`` from ``run_experiment``.

    Process:
        Resolve a fresh run directory, print its path, then branch on
        ``isinstance`` for VQE/VQD/sim point, benchmark, and validation.

    Outputs:
        None.

    Side effects:
        May write CSVs, generate plots, and print summaries to stdout.
    """

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
    """Persist VQE sweep CSV/plots and print a benchmark summary.

    Purpose:
        Honor ``config.output.csv`` / ``.plots`` for TFIM VQE benchmarks.

    Inputs:
        config: Output flags and system labels for the summary.
        result: Sweep results.
        output_dir: Destination directory for this run.

    Process:
        Optionally write ``tfim_benchmark.csv`` and call ``plot_all_tfim``;
        always print the tabular summary.

    Outputs:
        None.

    Side effects:
        Filesystem writes and stdout.
    """

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
    """Print a VQE single-point summary and optionally plot it.

    Purpose:
        Present one-shot VQE results without requiring a CSV (single points
        are summarized on the console).

    Inputs:
        config: System parameters for labeling.
        point: Single-point result.
        output_dir: Plot destination when enabled.

    Process:
        Print metrics; if plots are enabled, call ``plot_vqe_single_point``.

    Outputs:
        None.

    Side effects:
        Stdout; optional plot files.
    """

    _print_vqe_single_point(point, config)

    if config.output.plots:
        plot_vqe_single_point(point, output_dir)
        print(f"Plots saved to: {output_dir}")


def _save_vqd_single_point(
    config: AtlasConfig, point: TFIMVQDPointResult, output_dir: str
) -> None:
    """Print a VQD single-point summary and optionally plot it.

    Purpose:
        Present per-state exact vs VQD comparison for one field value.

    Inputs:
        config: System parameters for labeling.
        point: VQD point result.
        output_dir: Plot destination when enabled.

    Process:
        Print the state table; optionally call ``plot_vqd_single_point``.

    Outputs:
        None.

    Side effects:
        Stdout; optional plot files.
    """

    _print_vqd_single_point(point, config)

    if config.output.plots:
        plot_vqd_single_point(point, output_dir)
        print(f"Plots saved to: {output_dir}")


def _save_vqd_benchmark(
    config: AtlasConfig, result: TFIMVQDBenchmarkResult, output_dir: str
) -> None:
    """Persist VQD sweep summary, optional CSV, and optional plots.

    Purpose:
        Handle VQD benchmark outputs (CSV written locally here rather than
        via ``csv_io`` helpers used for VQE).

    Inputs:
        config: Output flags.
        result: VQD sweep results.
        output_dir: Destination directory.

    Process:
        Print summary; optionally write ``tfim_vqd_benchmark.csv`` and call
        ``plot_all_tfim_vqd``.

    Outputs:
        None.

    Side effects:
        Stdout; optional CSV and plots.
    """

    _print_vqd_benchmark(result, config)

    if config.output.csv:
        _write_vqd_benchmark_csv(result, output_dir)

    if config.output.plots:
        plot_all_tfim_vqd(result, output_dir)
        print(f"Plots saved to: {output_dir}")


def _print_vqe_single_point(point: TFIMPointResult, config: AtlasConfig) -> None:
    """Print a formatted VQE single-shot results block to stdout.

    Purpose:
        Give operators an immediate exact-vs-VQE comparison without opening
        plot files.

    Inputs:
        point: Completed VQE point.
        config: System parameters for the header.

    Process:
        Compute absolute energy error and print energy, iterations, and
        rounded optimal parameters.

    Outputs:
        None.

    Side effects:
        Writes to stdout.
    """

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
    """Print per-point and aggregate VQE sweep statistics.

    Purpose:
        Summarize fidelity and energy error across ``h``, plus optional
        hardware error stats when hardware rows are present.

    Inputs:
        result: VQE benchmark collection.
        config: System parameters (``J``, qubit count) for headers.

    Process:
        Print a table of ``h/J``, ΔE, fidelity, and iterations; then mean/max
        fidelity and sim absolute error; if any points have hardware, print
        absolute and relative hardware error aggregates.

    Outputs:
        None.

    Side effects:
        Writes to stdout.
    """

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
    """Print a per-state VQD comparison table for one point.

    Purpose:
        Show exact vs VQD energy, absolute error, and fidelity for each
        recovered eigenstate.

    Inputs:
        point: VQD point result.
        config: System parameters for the header.

    Process:
        Zip exact energies, VQD state results, errors, and fidelities into
        a formatted table.

    Outputs:
        None.

    Side effects:
        Writes to stdout.
    """

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
    """Print nested VQD sweep results (per ``h``, per state).

    Purpose:
        Console overview of spectrum recovery across a transverse-field sweep.

    Inputs:
        result: VQD benchmark collection.
        config: System parameters for the header.

    Process:
        For each point, print ``h`` then one line per state with exact/VQD
        energy, absolute error, and fidelity.

    Outputs:
        None.

    Side effects:
        Writes to stdout.
    """

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
    """Write a long-form CSV with one row per (h, state_index).

    Purpose:
        Export VQD sweep metrics for external analysis without a dedicated
        ``csv_io`` writer yet.

    Inputs:
        result: VQD benchmark collection.
        output_dir: Directory for ``tfim_vqd_benchmark.csv``.

    Process:
        Flatten points × states into row dicts; write via pandas.

    Outputs:
        None.

    Side effects:
        Creates ``output_dir`` if needed; writes a CSV; prints its path.
    """

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


def _format_system_parameters(parameters: dict) -> str:
    """Format system parameters for console summaries without assuming model keys."""

    parts = []
    for key, value in parameters.items():
        if key == "num_qubits":
            continue
        parts.append(f"{key} = {value}")
    return ", ".join(parts) if parts else "(none)"


def _save_lattice_dashboard_snapshots(
    series_list: list[SimBenchmarkResult],
    output_dir: str,
) -> None:
    """Write lattice-dashboard PNG snapshots for series with site observables.

    Purpose:
        Persist a representative frame from the interactive dashboard without
        requiring a GUI. Uses the last frame of each trajectory. Separate from
        scalar sim plots; only runs when site arrays are present on results.
    """

    for series in series_list:
        if not series.points:
            continue
        names = available_site_observables(series)
        if not names:
            continue
        method = series.method_name or "sim"
        for name in names:
            for source in ("sim", "exact"):
                try:
                    trajectory = trajectory_from_sim_benchmark(
                        series, site_observable=name, source=source
                    )
                except KeyError:
                    continue
                dashboard = LatticeDashboard(
                    trajectory,
                    show_slider=False,
                    show_playback=False,
                    initial_frame=len(trajectory.frames) - 1,
                    window_title="Atlas — Spin Chain Visualization",
                )
                snap = os.path.join(
                    output_dir, f"lattice_dashboard_{name}_{source}_{method}.png"
                )
                dashboard.save_snapshot(snap)
                dashboard.close()
                print(f"Lattice dashboard snapshot saved to: {snap}")


def _save_sim_point_dashboard_snapshots(
    point: SimPointResult,
    output_dir: str,
) -> None:
    """Write single-frame lattice dashboard PNGs for a dynamics point."""

    names = available_site_observables(point)
    if not names:
        return
    method = point.method_name or "sim"
    for name in names:
        for source in ("sim", "exact"):
            try:
                trajectory = trajectory_from_sim_point(
                    point, site_observable=name, source=source
                )
            except KeyError:
                continue
            dashboard = LatticeDashboard(
                trajectory,
                show_slider=False,
                show_playback=False,
                window_title="Atlas — Spin Chain Visualization",
            )
            snap = os.path.join(
                output_dir, f"lattice_dashboard_{name}_{source}_{method}.png"
            )
            dashboard.save_snapshot(snap)
            dashboard.close()
            print(f"Lattice dashboard snapshot saved to: {snap}")


def _save_sim_single_point(
    config: AtlasConfig, point: SimPointResult, output_dir: str
) -> None:
    """Print a dynamics single-point summary and optionally plot it.

    Purpose:
        Present fidelity, depth, and per-observable errors for one evolution.

    Inputs:
        config: System parameters for labeling.
        point: Simulation point result.
        output_dir: Plot destination when enabled.

    Process:
        Print results; optionally call ``plot_sim_single_point``.

    Outputs:
        None.

    Side effects:
        Stdout; optional plots.
    """

    _print_sim_single_point(point, config)

    if config.output.plots:
        plot_sim_single_point(point, output_dir)
        _save_sim_point_dashboard_snapshots(point, output_dir)
        print(f"Plots saved to: {output_dir}")


def _save_sim_benchmark(
    config: AtlasConfig, result: SimBenchmarkResult, output_dir: str
) -> None:
    """Persist dynamics sweep CSV/plots and print a summary.

    Purpose:
        Honor output flags for Hamiltonian-simulation benchmarks.

    Inputs:
        config: Output flags and system labels.
        result: Simulation benchmark collection.
        output_dir: Destination directory.

    Process:
        Print summary; optionally write ``sim_benchmark.csv`` and call
        ``plot_all_sim``. When the sweep axis is ``num_trotter_steps``, also
        emit a validation-style comparison plot for the single series.

    Outputs:
        None.

    Side effects:
        Stdout; optional CSV and plots.
    """

    _print_sim_benchmark(result, config)

    if config.output.csv:
        csv_path = os.path.join(output_dir, "sim_benchmark.csv")
        write_sim_benchmark(result, csv_path)
        print(f"Simulation benchmark CSV saved to: {csv_path}")

    if config.output.plots:
        plot_all_sim_comparison([result], output_dir)
        if result.sweep_parameter == "num_trotter_steps":
            plot_validation_comparison([result], output_dir)
        _save_lattice_dashboard_snapshots([result], output_dir)
        print(f"Plots saved to: {output_dir}")


def _print_sim_single_point(point: SimPointResult, config: AtlasConfig) -> None:
    """Print fidelity, depth, and observable errors for one dynamics point.

    Purpose:
        Console summary of exact vs simulated expectations.

    Inputs:
        point: Simulation point result.
        config: Kept for API symmetry with other printers; parameter labels
            come from ``point.system_parameters``.
    """

    _ = config
    sim = point.sim_result

    print("=" * 50)
    print(f" {point.num_qubits}-Qubit Hamiltonian Simulation (Single-Shot)")
    print(f" Parameters: {_format_system_parameters(point.system_parameters)}")
    print(f" Evolution time: {point.evolution_time}")
    print(f" Method: {sim.method_name}, steps: {sim.num_trotter_steps}")
    print("=" * 50)
    print("\n--- Final Results ---")
    print(f"State fidelity:          {point.fidelity:.6f}")
    print(f"State infidelity:        {point.infidelity:.6e}")
    print(f"Circuit depth:           {sim.circuit_depth}")
    for name, value in point.observables.items():
        exact = point.exact_observables[name]
        error = point.observable_errors[name]
        print(
            f"<{name}>: sim={value:.6f}, exact={exact:.6f}, "
            f"abs_error={error:.6e}"
        )


def _print_sim_benchmark(result: SimBenchmarkResult, config: AtlasConfig) -> None:
    """Print a dynamics sweep table and mean/max fidelity summary.

    Purpose:
        Quick overview of how fidelity and depth track the swept parameter.
    """

    num_qubits = config.system.parameters.get("num_qubits", "?")
    print("=" * 50)
    print(f" {num_qubits}-Qubit Simulation Benchmark")
    print(f" Sweep parameter: {result.sweep_parameter}")
    print(f" Parameters: {_format_system_parameters(dict(config.system.parameters))}")
    print("=" * 50)

    print(
        f"\n{result.sweep_parameter:<16} | {'Fidelity':<10} | "
        f"{'Depth':<8} | {'Method'}"
    )
    print("-" * 50)
    for point in result.points:
        sweep_value = result.sweep_value(point)
        print(
            f"{float(sweep_value):<16.4f} | {point.fidelity:<10.4f} | "
            f"{point.sim_result.circuit_depth:<8} | {point.sim_result.method_name}"
        )

    print("\nBenchmark Summary")
    print("-----------------")
    print(f"Mean fidelity:           {np.mean(result.fidelities):.5f}")
    print(f"Max fidelity:            {np.max(result.fidelities):.5f}")


def _save_sim_validation(
    config: AtlasConfig, result: SimValidationResult, output_dir: str
) -> None:
    """Persist Trotter validation CSV/plots and print per-method tables.

    Purpose:
        Save multi-method step sweeps used to compare product formulas.
    """

    _print_sim_validation(result, config)

    if config.output.csv:
        csv_path = os.path.join(output_dir, "sim_validation.csv")
        write_sim_validation(result, csv_path)
        print(f"Validation CSV saved to: {csv_path}")

    if config.output.plots:
        plot_all_sim_comparison(result.series, output_dir)
        if result.sweep_parameter == "num_trotter_steps":
            plot_validation_comparison(result.series, output_dir)
        _save_lattice_dashboard_snapshots(list(result.series), output_dir)
        print(f"Validation plots saved to: {output_dir}")


def _print_sim_validation(result: SimValidationResult, config: AtlasConfig) -> None:
    """Print per-method tables for a multi-method dynamics comparison."""

    num_qubits = result.system_parameters.get(
        "num_qubits", config.system.parameters.get("num_qubits", "?")
    )
    print("=" * 50)
    print(f" {num_qubits}-Qubit Evolution Method Comparison")
    print(f" Parameters: {_format_system_parameters(result.system_parameters)}")
    if result.sweep_parameter != "single_point":
        print(f" Sweep parameter: {result.sweep_parameter}")
    print(f" Evolution time: {result.evolution_time}")
    print(f" Methods: {result.method_names}")
    print("=" * 50)

    for series in result.series:
        print(f"\nMethod: {series.method_name}")
        if result.sweep_parameter == "single_point":
            point = series.points[0]
            print(f"State infidelity:        {point.infidelity:.6e}")
            print(f"Circuit depth:           {point.circuit_depth}")
            for name, value in point.observables.items():
                exact = point.exact_observables[name]
                error = point.observable_errors[name]
                print(
                    f"<{name}>: sim={value:.6f}, exact={exact:.6f}, "
                    f"abs_error={error:.6e}"
                )
            continue

        print(
            f"{result.sweep_parameter:<16} | {'Infidelity':<12} | "
            f"{'Max Op Err':<12} | {'Depth'}"
        )
        print("-" * 62)
        for point in series.points:
            sweep_value = series.sweep_value(point)
            print(
                f"{float(sweep_value):<16.4f} | {point.infidelity:<12.2e} | "
                f"{point.max_operator_error:<12.2e} | {point.circuit_depth}"
            )
