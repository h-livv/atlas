"""TFIM benchmark plots.

Passive plotting functions that consume a ``TFIMBenchmarkResult`` and write PNG
artifacts. They never run VQE, exact diagonalization, hardware jobs, or CSV
loading -- all values are already computed on the result object. Filenames,
labels, and semantics match the legacy ``vqe_legacy/tfim_vis.py`` script.

Every function degrades gracefully when no hardware data is present on the
result (i.e. every ``TFIMPointResult.hardware_result is None``), since
``include_hardware`` defaults to ``False`` upstream in the experiment layer:
hardware-only series are simply omitted from the comparison plots, and the
hardware-only plots (``plot_parity``, ``plot_relative_error``) no-op with an
early return instead of raising or producing a meaningless plot.

``matplotlib.use("Agg")`` is set before importing ``pyplot`` so these functions
work in headless CI / server environments without a display.
"""

from __future__ import annotations

import os

import matplotlib

# Agg is a non-interactive backend: required so plot scripts work without a GUI.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from atlas.experiments.results import TFIMBenchmarkResult

# Tiny floor added before semilogy so exact-zero errors remain visible on a log axis.
_EPS = 1e-12


def _has_hardware_data(result: TFIMBenchmarkResult) -> bool:
    """Return whether any sweep point carries a hardware result.

    Purpose:
        Gate hardware series and hardware-only plots so simulator-only
        benchmarks skip those overlays cleanly.

    Inputs:
        result: A ``TFIMBenchmarkResult`` whose ``points`` may or may not have
            ``hardware_result`` populated.

    Process:
        Scan ``result.points`` and return True if any point's
        ``hardware_result`` is not ``None``.

    Outputs:
        ``True`` if at least one hardware result exists; otherwise ``False``.

    Side Effects:
        None.
    """

    return any(point.hardware_result is not None for point in result.points)


def _hardware_energy_stderr(result: TFIMBenchmarkResult) -> np.ndarray:
    """Collect per-point hardware energy standard errors for error bars.

    Purpose:
        Build the ``yerr`` array for energy errorbars without requiring every
        point to have hardware data (missing points become ``None`` entries).

    Inputs:
        result: A ``TFIMBenchmarkResult`` aligned with ``result.h_values``.

    Process:
        For each point, read ``hardware_result.energy_stderr`` when present,
        otherwise insert ``None``, then wrap as a NumPy array.

    Outputs:
        A 1-D ``np.ndarray`` of stderr values (``None`` where no hardware data).

    Side Effects:
        None.
    """

    return np.array(
        [
            point.hardware_result.energy_stderr if point.hardware_result else None
            for point in result.points
        ]
    )


def _hardware_observable_stderr(result: TFIMBenchmarkResult, name: str) -> np.ndarray:
    """Collect per-point hardware observable standard errors for error bars.

    Purpose:
        Same role as ``_hardware_energy_stderr``, but for a named observable's
        stderr dictionary on each hardware result.

    Inputs:
        result: A ``TFIMBenchmarkResult`` aligned with ``result.h_values``.
        name: Observable key (e.g. ``"zz"`` or ``"x"``) looked up in
            ``hardware_result.observable_stderr``.

    Process:
        For each point, if a hardware result and stderr dict exist, take
        ``observable_stderr.get(name)``; otherwise ``None``. Wrap as an array.

    Outputs:
        A 1-D ``np.ndarray`` of stderr values (``None`` where unavailable).

    Side Effects:
        None.
    """

    return np.array(
        [
            point.hardware_result.observable_stderr.get(name)
            if point.hardware_result and point.hardware_result.observable_stderr
            else None
            for point in result.points
        ]
    )


def plot_energy(result: TFIMBenchmarkResult, output_dir: str) -> None:
    """Plot exact, VQE, and (if present) hardware energy versus ``h/J``.

    Purpose:
        Show ground-state energy across the transverse-field sweep, comparing
        the analytical/exact reference, the VQE approximation, and optionally
        IBM hardware estimates with stderr error bars.

    Inputs:
        result: A ``TFIMBenchmarkResult`` with ``h_values``, ``exact_energies``,
            ``vqe_energies``, and optionally ``hardware_energies`` / stderrs.
        output_dir: Directory for ``energy_vs_hJ.png`` (must already exist;
            ``plot_all_tfim`` creates it when used as the entry point).

    Process:
        1. Plot exact (dashed) and VQE (markers) series vs ``h_values``.
        2. If any hardware data exists, overlay hardware energies with errorbars.
        3. Style, save as ``energy_vs_hJ.png``, and close the figure.

    Outputs:
        None. Writes ``energy_vs_hJ.png`` under ``output_dir``.

    Side Effects:
        Creates/overwrites the PNG; allocates and closes a matplotlib figure.
        Does not create ``output_dir``.
    """

    h_values = result.h_values

    plt.figure(figsize=(8, 5))
    plt.plot(h_values, result.exact_energies, "k--", label="Exact Analytical", linewidth=2)
    plt.plot(h_values, result.vqe_energies, "ro", label="VQE Approximation", alpha=0.7)
    if _has_hardware_data(result):
        # Hardware is optional upstream; skip the series when absent.
        plt.errorbar(
            h_values,
            result.hardware_energies,
            yerr=_hardware_energy_stderr(result),
            fmt="kx",
            markersize=10,
            capsize=4,
            label="IBM Hardware",
        )
    plt.title("Energy vs Transverse Field (h/J)")
    plt.xlabel("h/J Ratio (Transverse Field Strength)")
    plt.ylabel("Energy")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.savefig(os.path.join(output_dir, "energy_vs_hJ.png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_magnetization(result: TFIMBenchmarkResult, output_dir: str) -> None:
    """Plot simulator ``<ZZ>``/``<X0+X1>`` and (if present) hardware magnetization.

    Purpose:
        Visualize the longitudinal vs transverse magnetization competition that
        characterizes the TFIM phase structure as ``h/J`` varies.

    Inputs:
        result: A ``TFIMBenchmarkResult`` exposing ``observable_values`` for
            ``"zz"`` and ``"x"``, and optionally matching hardware series.
        output_dir: Directory for ``magnetization_comparison.png``.

    Process:
        1. Plot simulator ``<ZZ>`` and ``<X0+X1>`` vs ``h_values``.
        2. If hardware data exists, overlay both observables with errorbars.
        3. Add a zero reference line, style, save, and close.

    Outputs:
        None. Writes ``magnetization_comparison.png`` under ``output_dir``.

    Side Effects:
        Creates/overwrites the PNG; allocates and closes a matplotlib figure.
    """

    h_values = result.h_values

    plt.figure(figsize=(8, 5))
    plt.plot(
        h_values,
        result.observable_values("zz"),
        "b-s",
        label=r"$\langle ZZ \rangle$ (Longitudinal)",
        linewidth=2,
    )
    plt.plot(
        h_values,
        result.observable_values("x"),
        "c-o",
        label=r"$\langle X_0+X_1 \rangle$ (Transverse)",
        linewidth=2,
    )
    if _has_hardware_data(result):
        # Hardware is optional upstream; skip the series when absent.
        plt.errorbar(
            h_values,
            result.hardware_observable_values("zz"),
            yerr=_hardware_observable_stderr(result, "zz"),
            fmt="ks",
            markersize=8,
            capsize=4,
            label="Hardware <ZZ>",
        )
        plt.errorbar(
            h_values,
            result.hardware_observable_values("x"),
            yerr=_hardware_observable_stderr(result, "x"),
            fmt="k^",
            markersize=8,
            capsize=4,
            label="Hardware <X0+X1>",
        )
    plt.title("Magnetization Phase Competition vs h/J")
    plt.xlabel("h/J Ratio (Transverse Field Strength)")
    plt.ylabel("Expectation Value")
    plt.axhline(0, color="gray", linestyle="--", alpha=0.5)
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.savefig(
        os.path.join(output_dir, "magnetization_comparison.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()


def plot_energy_errors(result: TFIMBenchmarkResult, output_dir: str) -> None:
    """Semilogy plot of simulator absolute energy error and (if present) hardware error.

    Purpose:
        Compare absolute energy error (vs exact) for the VQE simulator path and,
        when available, the hardware path, on a log-y scale so small errors remain
        readable.

    Inputs:
        result: A ``TFIMBenchmarkResult`` with ``vqe_energies``,
            ``exact_energies``, and optionally ``hardware_energies``.
        output_dir: Directory for ``sim_vs_hardware_error.png``.

    Process:
        1. Compute absolute simulator error; add ``_EPS`` before ``semilogy``.
        2. If hardware exists, compute and plot its absolute error similarly.
        3. Style, save, and close.

    Outputs:
        None. Writes ``sim_vs_hardware_error.png`` under ``output_dir``.

    Side Effects:
        Creates/overwrites the PNG; allocates and closes a matplotlib figure.
    """

    h_values = result.h_values
    sim_error = np.abs(result.vqe_energies - result.exact_energies)

    plt.figure(figsize=(8, 5))
    # _EPS avoids log(0) / invisible points when the error is exactly zero.
    plt.semilogy(h_values, sim_error + _EPS, "bo-", label="Simulator")
    if _has_hardware_data(result):
        hardware_error = np.abs(result.hardware_energies - result.exact_energies)
        plt.semilogy(h_values, hardware_error + _EPS, "ro-", label="Hardware")
    plt.legend()
    plt.xlabel("h/J")
    plt.ylabel("Absolute Energy Error")
    plt.title("Energy Error Comparison")
    plt.grid(True)
    plt.savefig(
        os.path.join(output_dir, "sim_vs_hardware_error.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()


def plot_parity(result: TFIMBenchmarkResult, output_dir: str) -> None:
    """Scatter plot of simulator vs hardware energy. No-ops if no hardware data exists.

    Purpose:
        Parity plot: each point is one ``h/J`` value, plotted as (simulator
        energy, hardware energy). Agreement with the diagonal means hardware
        matches the simulator. Colored by ``h/J``.

    Inputs:
        result: A ``TFIMBenchmarkResult``. If no point has hardware data, the
            function returns immediately without writing a file.
        output_dir: Directory for ``sim_vs_hardware_parity.png``.

    Process:
        1. Early-return when hardware is absent (avoids an empty/meaningless plot).
        2. Scatter VQE vs hardware energies, color by ``h_values``.
        3. Draw the y=x reference line over the shared data range.
        4. Force equal axes, save, and close.

    Outputs:
        None. Writes ``sim_vs_hardware_parity.png`` when hardware data exists;
        otherwise writes nothing.

    Side Effects:
        May create/overwrite the PNG and allocate/close a matplotlib figure;
        otherwise none.
    """

    if not _has_hardware_data(result):
        # Hardware-only plot: skip rather than emit a blank or misleading figure.
        return

    h_values = result.h_values
    vqe_energies = result.vqe_energies
    hardware_energies = result.hardware_energies

    plt.figure(figsize=(6, 6))
    plt.scatter(vqe_energies, hardware_energies, c=h_values, cmap="viridis", s=80)
    plt.colorbar(label="h/J")

    # Shared limits so the diagonal reference line spans the full data range.
    all_energy = np.concatenate([vqe_energies, hardware_energies])
    lims = [np.min(all_energy), np.max(all_energy)]
    plt.plot(lims, lims, "k--")

    plt.axis("equal")
    plt.xlim(lims)
    plt.ylim(lims)
    plt.xlabel("Simulator Energy")
    plt.ylabel("Hardware Energy")
    plt.title("Simulator vs Hardware")
    plt.savefig(
        os.path.join(output_dir, "sim_vs_hardware_parity.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()


def plot_relative_error(result: TFIMBenchmarkResult, output_dir: str) -> None:
    """Plot hardware relative energy error vs exact. No-ops if no hardware data exists.

    Purpose:
        Report hardware energy error as a percentage of the exact energy
        magnitude across the ``h/J`` sweep.

    Inputs:
        result: A ``TFIMBenchmarkResult``. Early-returns if no hardware data.
        output_dir: Directory for ``relative_error.png``.

    Process:
        1. Early-return when hardware is absent.
        2. Compute
           ``|E_hw - E_exact| / |E_exact| * 100`` per sweep point.
        3. Plot vs ``h_values``, save, and close.

    Outputs:
        None. Writes ``relative_error.png`` when hardware data exists;
        otherwise writes nothing.

    Side Effects:
        May create/overwrite the PNG and allocate/close a matplotlib figure;
        otherwise none.
    """

    if not _has_hardware_data(result):
        # Hardware-only plot: skip rather than emit a blank or misleading figure.
        return

    h_values = result.h_values
    relative_error = (
        np.abs(result.hardware_energies - result.exact_energies)
        / np.abs(result.exact_energies)
        * 100
    )

    plt.figure(figsize=(8, 5))
    plt.plot(h_values, relative_error, "ro-")
    plt.xlabel("h/J")
    plt.ylabel("Relative Error (%)")
    plt.title("Relative Energy Error (Hardware vs Exact)")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "relative_error.png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_fidelity(result: TFIMBenchmarkResult, output_dir: str) -> None:
    """Plot VQE state fidelity to the exact ground state versus ``h/J``.

    Purpose:
        Show how close the VQE optimized state is to the exact ground state
        across the transverse-field sweep (simulator-side fidelity only).

    Inputs:
        result: A ``TFIMBenchmarkResult`` with ``h_values`` and ``fidelities``.
        output_dir: Directory for ``fidelity_vs_hJ.png``.

    Process:
        Plot fidelity vs ``h_values`` with a y-axis focused near 1, save, close.

    Outputs:
        None. Writes ``fidelity_vs_hJ.png`` under ``output_dir``.

    Side Effects:
        Creates/overwrites the PNG; allocates and closes a matplotlib figure.
    """

    plt.figure(figsize=(8, 5))
    plt.plot(result.h_values, result.fidelities, "g-", marker="^")
    plt.title("VQE State Fidelity vs Exact Ground State")
    plt.xlabel("h/J Ratio (Transverse Field Strength)")
    plt.ylabel("Fidelity (0 to 1)")
    # Legacy y-limits zoom on the high-fidelity regime typical for this benchmark.
    plt.ylim([0.8, 1.05])
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.savefig(os.path.join(output_dir, "fidelity_vs_hJ.png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_vqe_single_point(point, output_dir: str) -> None:
    """Bar chart comparing exact and VQE ground-state energy for one point.

    Purpose:
        Summarize a single ``(h, J)`` VQE run: exact vs VQE energy bars, with
        absolute error and fidelity annotated on the figure.

    Inputs:
        point: A TFIM point result with ``exact_energy``, ``vqe_result.energy``,
            ``fidelity``, ``h``, and ``J`` (typically a ``TFIMPointResult``).
        output_dir: Directory for ``vqe_single_point_energy.png``. Created if
            missing.

    Process:
        1. Ensure ``output_dir`` exists.
        2. Draw Exact/VQE energy bars and annotate absolute error + fidelity.
        3. Label each bar with its numeric energy, save, and close.

    Outputs:
        None. Writes ``vqe_single_point_energy.png`` under ``output_dir``.

    Side Effects:
        Creates ``output_dir`` if missing; creates/overwrites the PNG; allocates
        and closes a matplotlib figure.
    """

    os.makedirs(output_dir, exist_ok=True)
    labels = ["Exact", "VQE"]
    energies = [point.exact_energy, point.vqe_result.energy]
    colors = ["black", "crimson"]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, energies, color=colors, alpha=0.75, width=0.5)
    ax.set_ylabel("Energy")
    ax.set_title(f"Ground-State Energy (h = {point.h}, J = {point.J})")
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)

    error = abs(point.exact_energy - point.vqe_result.energy)
    ax.text(
        0.5,
        0.02,
        f"Absolute error: {error:.3e}\nFidelity: {point.fidelity:.4f}",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    for bar, energy in zip(bars, energies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{energy:.6f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.savefig(os.path.join(output_dir, "vqe_single_point_energy.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_all_tfim(result: TFIMBenchmarkResult, output_dir: str) -> None:
    """Generate the full default plot set, creating ``output_dir`` if missing.

    Purpose:
        Single entry point for experiment runners to dump the legacy TFIM
        visualization suite after a sweep completes.

    Inputs:
        result: A ``TFIMBenchmarkResult`` (simulator-only or with hardware).
        output_dir: Destination directory for all PNG artifacts.

    Process:
        1. Create ``output_dir`` if needed.
        2. Call energy, magnetization, energy-error, parity, relative-error,
           and fidelity plotters in that order.
        Hardware-only plots (``plot_parity``, ``plot_relative_error``) and
        hardware-only series within the other plots are skipped cleanly when the
        result has no hardware data, so this works end-to-end for an
        all-simulator benchmark result.

    Outputs:
        None. Multiple PNG files under ``output_dir``.

    Side Effects:
        Creates ``output_dir`` if missing; delegates file writes and figure
        lifecycle to the individual plot functions.
    """

    os.makedirs(output_dir, exist_ok=True)

    plot_energy(result, output_dir)
    plot_magnetization(result, output_dir)
    plot_energy_errors(result, output_dir)
    plot_parity(result, output_dir)
    plot_relative_error(result, output_dir)
    plot_fidelity(result, output_dir)
