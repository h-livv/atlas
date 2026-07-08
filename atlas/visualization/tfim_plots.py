"""TFIM benchmark plots.

Passive plotting functions that consume a `TFIMBenchmarkResult` and write PNG
artifacts. They never run VQE, exact diagonalization, hardware jobs, or CSV
loading -- all values are already computed on the result object. Filenames,
labels, and semantics match the legacy `vqe_legacy/tfim_vis.py` script.

Every function degrades gracefully when no hardware data is present on the
result (i.e. every `TFIMPointResult.hardware_result is None`), since
`include_hardware` defaults to `False` upstream in the experiment layer:
hardware-only series are simply omitted from the comparison plots, and the
hardware-only plots (`plot_parity`, `plot_relative_error`) no-op with an
early return instead of raising or producing a meaningless plot.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from atlas.experiments.results import TFIMBenchmarkResult

_EPS = 1e-12


def _has_hardware_data(result: TFIMBenchmarkResult) -> bool:
    return any(point.hardware_result is not None for point in result.points)


def _hardware_energy_stderr(result: TFIMBenchmarkResult) -> np.ndarray:
    return np.array(
        [
            point.hardware_result.energy_stderr if point.hardware_result else None
            for point in result.points
        ]
    )


def _hardware_observable_stderr(result: TFIMBenchmarkResult, name: str) -> np.ndarray:
    return np.array(
        [
            point.hardware_result.observable_stderr.get(name)
            if point.hardware_result and point.hardware_result.observable_stderr
            else None
            for point in result.points
        ]
    )


def plot_energy(result: TFIMBenchmarkResult, output_dir: str) -> None:
    """Plot exact, VQE, and (if present) hardware energy versus `h/J`."""

    h_values = result.h_values

    plt.figure(figsize=(8, 5))
    plt.plot(h_values, result.exact_energies, "k--", label="Exact Analytical", linewidth=2)
    plt.plot(h_values, result.vqe_energies, "ro", label="VQE Approximation", alpha=0.7)
    if _has_hardware_data(result):
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
    """Plot simulator `<ZZ>`/`<X0+X1>` and (if present) hardware magnetization."""

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
    """Semilogy plot of simulator absolute energy error and (if present) hardware error."""

    h_values = result.h_values
    sim_error = np.abs(result.vqe_energies - result.exact_energies)

    plt.figure(figsize=(8, 5))
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
    """Scatter plot of simulator vs hardware energy. No-ops if no hardware data exists."""

    if not _has_hardware_data(result):
        return

    h_values = result.h_values
    vqe_energies = result.vqe_energies
    hardware_energies = result.hardware_energies

    plt.figure(figsize=(6, 6))
    plt.scatter(vqe_energies, hardware_energies, c=h_values, cmap="viridis", s=80)
    plt.colorbar(label="h/J")

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
    """Plot hardware relative energy error vs exact. No-ops if no hardware data exists."""

    if not _has_hardware_data(result):
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
    """Plot VQE state fidelity to the exact ground state versus `h/J`."""

    plt.figure(figsize=(8, 5))
    plt.plot(result.h_values, result.fidelities, "g-", marker="^")
    plt.title("VQE State Fidelity vs Exact Ground State")
    plt.xlabel("h/J Ratio (Transverse Field Strength)")
    plt.ylabel("Fidelity (0 to 1)")
    plt.ylim([0.8, 1.05])
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.savefig(os.path.join(output_dir, "fidelity_vs_hJ.png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_vqe_single_point(point, output_dir: str) -> None:
    """Bar chart comparing exact and VQE ground-state energy for one point."""

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
    """Generate the full default plot set, creating `output_dir` if missing.

    Hardware-only plots (`plot_parity`, `plot_relative_error`) and
    hardware-only series within the other plots are skipped cleanly when the
    result has no hardware data, so this works end-to-end for an
    all-simulator benchmark result.
    """

    os.makedirs(output_dir, exist_ok=True)

    plot_energy(result, output_dir)
    plot_magnetization(result, output_dir)
    plot_energy_errors(result, output_dir)
    plot_parity(result, output_dir)
    plot_relative_error(result, output_dir)
    plot_fidelity(result, output_dir)
