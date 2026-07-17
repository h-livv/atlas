"""VQD benchmark and single-point plots.

Passive plotting helpers for Variational Quantum Deflation (VQD) results on the
TFIM. They consume already-computed ``TFIMVQDPointResult`` /
``TFIMVQDBenchmarkResult`` objects and write PNG artifacts — they never run
optimization, exact diagonalization, or hardware jobs.

VQD recovers multiple low-lying eigenstates; these plots therefore loop over
``result.num_states`` (or the states on a single point) and use a shared color
palette so each state index stays visually consistent across energy, fidelity,
and error figures.

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

from atlas.experiments.results import TFIMVQDBenchmarkResult, TFIMVQDPointResult

# Tiny floor added before semilogy so exact-zero errors remain visible on a log axis.
_EPS = 1e-12
# Stable per-state colors so state k looks the same across energy/fidelity/error plots.
_STATE_COLORS = ["crimson", "darkorange", "forestgreen", "royalblue", "purple"]


def plot_vqd_single_point(point: TFIMVQDPointResult, output_dir: str) -> None:
    """Per-state energy, absolute error, and fidelity for a single VQD run.

    Purpose:
        Summarize one ``(h, J)`` VQD optimization as a three-panel figure:
        exact vs VQD energies, absolute energy errors, and fidelities — one bar
        group / bar per recovered state.

    Inputs:
        point: A ``TFIMVQDPointResult`` with ``exact_energies``,
            ``vqd_result.states``, ``absolute_errors``, ``fidelities``, ``h``,
            and ``J``.
        output_dir: Directory for ``vqd_single_point_summary.png``. Created if
            missing.

    Process:
        1. Ensure ``output_dir`` exists.
        2. Collect per-state exact/VQD energies, errors, and fidelities.
        3. Draw three subplots side by side and save a single summary PNG.

    Outputs:
        None. Writes ``vqd_single_point_summary.png`` under ``output_dir``.

    Side Effects:
        Creates ``output_dir`` if missing; creates/overwrites the PNG; allocates
        and closes a matplotlib figure.
    """

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
    """Plot exact and VQD energies versus ``h/J`` for each state.

    Purpose:
        Show how each recovered eigenstate's energy tracks the exact spectrum
        across the transverse-field sweep.

    Inputs:
        result: A ``TFIMVQDBenchmarkResult`` with ``h_values``, ``num_states``,
            and helpers ``state_exact_energies`` / ``state_vqd_energies``.
            ``J`` is taken from the first point (default ``1.0`` if empty) so
            the x-axis is plotted as the dimensionless ratio ``h/J``.
        output_dir: Directory for ``vqd_energy_vs_hJ.png``.

    Process:
        1. Convert ``h_values`` to ``h/J``.
        2. For each state index, plot exact (dashed) and VQD (markers) series
           in a shared color from ``_STATE_COLORS``.
        3. Style, save, and close.

    Outputs:
        None. Writes ``vqd_energy_vs_hJ.png`` under ``output_dir``.

    Side Effects:
        Creates/overwrites the PNG; allocates and closes a matplotlib figure.
        Does not create ``output_dir`` (``plot_all_tfim_vqd`` does).
    """

    h_values = result.h_values
    # Use the sweep's J so the x-axis is the physical ratio h/J, not raw h.
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
    """Plot per-state fidelity versus ``h/J``.

    Purpose:
        Show how faithfully each VQD state matches the corresponding exact
        eigenstate across the transverse-field sweep.

    Inputs:
        result: A ``TFIMVQDBenchmarkResult`` with ``state_fidelities`` and the
            same ``h``/``J`` layout as ``plot_vqd_energy``.
        output_dir: Directory for ``vqd_fidelity_vs_hJ.png``.

    Process:
        Convert to ``h/J``, plot one fidelity curve per state, save, close.

    Outputs:
        None. Writes ``vqd_fidelity_vs_hJ.png`` under ``output_dir``.

    Side Effects:
        Creates/overwrites the PNG; allocates and closes a matplotlib figure.
    """

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
    """Semilogy plot of per-state absolute energy error versus ``h/J``.

    Purpose:
        Compare absolute energy error (VQD vs exact) for each recovered state
        on a log-y scale so small errors remain visible.

    Inputs:
        result: A ``TFIMVQDBenchmarkResult`` with ``state_absolute_errors``.
        output_dir: Directory for ``vqd_energy_error_vs_hJ.png``.

    Process:
        Convert to ``h/J``, ``semilogy`` each state's absolute errors plus
        ``_EPS``, style, save, close.

    Outputs:
        None. Writes ``vqd_energy_error_vs_hJ.png`` under ``output_dir``.

    Side Effects:
        Creates/overwrites the PNG; allocates and closes a matplotlib figure.
    """

    h_values = result.h_values
    J = result.points[0].J if result.points else 1.0
    h_over_J = h_values / J

    plt.figure(figsize=(8, 5))
    for state_index in range(result.num_states):
        color = _STATE_COLORS[state_index % len(_STATE_COLORS)]
        # _EPS avoids log(0) / invisible points when the error is exactly zero.
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
    """Generate the default VQD sweep plot set.

    Purpose:
        Convenience entry point for experiment runners to dump energy,
        fidelity, and absolute-error sweep plots after a VQD benchmark.

    Inputs:
        result: A ``TFIMVQDBenchmarkResult`` covering the full ``h`` sweep.
        output_dir: Destination directory for the PNG artifacts.

    Process:
        1. Create ``output_dir`` if needed.
        2. Call ``plot_vqd_energy``, ``plot_vqd_fidelity``, and
           ``plot_vqd_energy_errors`` in that order.

    Outputs:
        None. Three PNG files under ``output_dir``.

    Side Effects:
        Creates ``output_dir`` if missing; delegates file writes and figure
        lifecycle to the individual plot functions.
    """

    os.makedirs(output_dir, exist_ok=True)
    plot_vqd_energy(result, output_dir)
    plot_vqd_fidelity(result, output_dir)
    plot_vqd_energy_errors(result, output_dir)
