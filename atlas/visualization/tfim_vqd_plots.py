"""VQD benchmark and single-point plots."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from atlas.experiments.results import TFIMVQDBenchmarkResult, TFIMVQDPointResult

_EPS = 1e-12
_STATE_COLORS = ["crimson", "darkorange", "forestgreen", "royalblue", "purple"]


def plot_vqd_single_point(point: TFIMVQDPointResult, output_dir: str) -> None:
    """Per-state energy, absolute error, and fidelity for a single VQD run."""

    os.makedirs(output_dir, exist_ok=True)
    num_states = len(point.vqd_result.states)
    state_labels = [f"State {index}" for index in range(num_states)]
    x = np.arange(num_states)
    width = 0.35

    exact = np.array(point.exact_energies)
    vqd = np.array([state.energy for state in point.vqd_result.states])
    errors = np.array(point.absolute_errors)
    fidelities = np.array(point.fidelities)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].bar(x - width / 2, exact, width, label="Exact", color="black", alpha=0.7)
    axes[0].bar(x + width / 2, vqd, width, label="VQD", color="crimson", alpha=0.7)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(state_labels)
    axes[0].set_ylabel("Energy")
    axes[0].set_title("Energy Comparison")
    axes[0].legend()
    axes[0].grid(True, axis="y", linestyle=":", alpha=0.6)

    axes[1].bar(state_labels, errors, color="steelblue", alpha=0.8)
    axes[1].set_ylabel("Absolute Error")
    axes[1].set_title("Per-State Absolute Error")
    axes[1].grid(True, axis="y", linestyle=":", alpha=0.6)

    axes[2].bar(state_labels, fidelities, color="forestgreen", alpha=0.8)
    axes[2].set_ylabel("Fidelity")
    axes[2].set_ylim(0, 1.05)
    axes[2].set_title("Per-State Fidelity")
    axes[2].grid(True, axis="y", linestyle=":", alpha=0.6)

    fig.suptitle(f"VQD Results (h = {point.h}, J = {point.J})", fontsize=12)
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "vqd_single_point_summary.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_vqd_energy(result: TFIMVQDBenchmarkResult, output_dir: str) -> None:
    """Plot exact and VQD energies versus ``h/J`` for each state."""

    h_values = result.h_values
    J = result.points[0].J if result.points else 1.0
    h_over_J = h_values / J

    plt.figure(figsize=(8, 5))
    for state_index in range(result.num_states):
        color = _STATE_COLORS[state_index % len(_STATE_COLORS)]
        plt.plot(
            h_over_J,
            result.state_exact_energies(state_index),
            "--",
            color=color,
            linewidth=2,
            label=f"Exact state {state_index}",
        )
        plt.plot(
            h_over_J,
            result.state_vqd_energies(state_index),
            "o",
            color=color,
            alpha=0.7,
            label=f"VQD state {state_index}",
        )
    plt.title("VQD Energy vs Transverse Field (h/J)")
    plt.xlabel("h/J Ratio (Transverse Field Strength)")
    plt.ylabel("Energy")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.savefig(os.path.join(output_dir, "vqd_energy_vs_hJ.png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_vqd_fidelity(result: TFIMVQDBenchmarkResult, output_dir: str) -> None:
    """Plot per-state fidelity versus ``h/J``."""

    h_values = result.h_values
    J = result.points[0].J if result.points else 1.0
    h_over_J = h_values / J

    plt.figure(figsize=(8, 5))
    for state_index in range(result.num_states):
        color = _STATE_COLORS[state_index % len(_STATE_COLORS)]
        plt.plot(
            h_over_J,
            result.state_fidelities(state_index),
            "-^",
            color=color,
            label=f"State {state_index}",
        )
    plt.title("VQD State Fidelity vs Exact")
    plt.xlabel("h/J Ratio (Transverse Field Strength)")
    plt.ylabel("Fidelity (0 to 1)")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.savefig(
        os.path.join(output_dir, "vqd_fidelity_vs_hJ.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()


def plot_vqd_energy_errors(result: TFIMVQDBenchmarkResult, output_dir: str) -> None:
    """Semilogy plot of per-state absolute energy error versus ``h/J``."""

    h_values = result.h_values
    J = result.points[0].J if result.points else 1.0
    h_over_J = h_values / J

    plt.figure(figsize=(8, 5))
    for state_index in range(result.num_states):
        color = _STATE_COLORS[state_index % len(_STATE_COLORS)]
        plt.semilogy(
            h_over_J,
            result.state_absolute_errors(state_index) + _EPS,
            "-o",
            color=color,
            label=f"State {state_index}",
        )
    plt.title("VQD Absolute Energy Error vs h/J")
    plt.xlabel("h/J Ratio (Transverse Field Strength)")
    plt.ylabel("Absolute Energy Error")
    plt.legend()
    plt.grid(True)
    plt.savefig(
        os.path.join(output_dir, "vqd_energy_error_vs_hJ.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()


def plot_all_tfim_vqd(result: TFIMVQDBenchmarkResult, output_dir: str) -> None:
    """Generate the default VQD sweep plot set."""

    os.makedirs(output_dir, exist_ok=True)
    plot_vqd_energy(result, output_dir)
    plot_vqd_fidelity(result, output_dir)
    plot_vqd_energy_errors(result, output_dir)
