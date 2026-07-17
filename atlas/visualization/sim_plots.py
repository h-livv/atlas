"""Plots for Hamiltonian simulation benchmark results.

Passive plotting helpers that consume already-computed benchmark series and
write PNG artifacts. All sweep and single-point plots accept one or more
``SimBenchmarkResult`` series so evolution methods can be compared on the same
axes.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from atlas.experiments.results import SimBenchmarkResult, SimPointResult
from atlas.visualization.sim_validation.series import series_label


def plot_sim_single_point(point: SimPointResult, output_dir: str) -> None:
    """Bar chart comparing exact and simulated observable expectations."""

    os.makedirs(output_dir, exist_ok=True)
    names = list(point.observables.keys())
    exact_values = [point.exact_observables[name] for name in names]
    sim_values = [point.observables[name] for name in names]

    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, exact_values, width, label="Exact")
    ax.bar(x + width / 2, sim_values, width, label="Simulated")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Expectation value")
    ax.set_title(
        f"Dynamics observables (t={point.evolution_time}, "
        f"method={point.sim_result.method_name})"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "sim_single_point_observables.png"))
    plt.close(fig)


def plot_sim_single_point_comparison(
    series_list: Sequence[SimBenchmarkResult],
    output_dir: str,
) -> None:
    """Compare single-point observable expectations across evolution methods."""

    if not series_list:
        return
    if len(series_list) == 1 and len(series_list[0].points) == 1:
        plot_sim_single_point(series_list[0].points[0], output_dir)
        return

    reference = series_list[0].points[0]
    names = list(reference.observables.keys())
    exact_values = [reference.exact_observables[name] for name in names]
    num_methods = len(series_list)
    cluster_width = 0.8
    bar_width = cluster_width / (num_methods + 1)

    os.makedirs(output_dir, exist_ok=True)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.5), 5))
    ax.bar(
        x - cluster_width / 2 + bar_width / 2,
        exact_values,
        bar_width,
        label="Exact",
        color="black",
        alpha=0.75,
    )
    for index, series in enumerate(series_list):
        point = series.points[0]
        sim_values = [point.observables[name] for name in names]
        offset = -cluster_width / 2 + (index + 1.5) * bar_width
        ax.bar(
            x + offset,
            sim_values,
            bar_width,
            label=series_label(series),
            alpha=0.8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Expectation value")
    ax.set_title(
        f"Dynamics observables (t={reference.evolution_time}, method comparison)"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "sim_single_point_observables.png"))
    plt.close(fig)


def _sweep_axis_label(sweep_parameter: str) -> str:
    if sweep_parameter == "evolution_time":
        return "Evolution time"
    return sweep_parameter


def _infidelity_plot_filename(sweep_parameter: str) -> str:
    if sweep_parameter == "evolution_time":
        return "sim_infidelity_vs_evolution_time.png"
    return "sim_infidelity_vs_sweep.png"


def _infidelity_plot_title(sweep_parameter: str) -> str:
    if sweep_parameter == "evolution_time":
        return "Infidelity vs evolution time"
    return f"Simulation infidelity vs {sweep_parameter}"


def plot_infidelity_comparison(
    series_list: Sequence[SimBenchmarkResult],
    output_dir: str,
) -> None:
    """Plot state infidelity for one or more evolution-method series."""

    if not series_list:
        return

    sweep_parameter = series_list[0].sweep_parameter
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for series in series_list:
        if not series.points:
            continue
        label = series_label(series) if len(series_list) > 1 else None
        ax.plot(
            series.sweep_values,
            series.infidelities,
            marker="o",
            label=label,
        )

    ax.set_xlabel(_sweep_axis_label(sweep_parameter))
    ax.set_ylabel("State infidelity")
    ax.set_title(_infidelity_plot_title(sweep_parameter))
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    if len(series_list) > 1:
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, _infidelity_plot_filename(sweep_parameter)))
    plt.close(fig)


def _observable_plot_filename(sweep_parameter: str, name: str) -> str:
    if sweep_parameter == "evolution_time":
        return f"sim_{name}_evolution_vs_time.png"
    return f"sim_{name}_vs_sweep.png"


def _observable_plot_title(sweep_parameter: str, name: str) -> str:
    if sweep_parameter == "evolution_time":
        return f"Observable evolution vs time ({name})"
    return f"Observable {name} vs {sweep_parameter}"


def plot_observable_comparison(
    series_list: Sequence[SimBenchmarkResult],
    output_dir: str,
    name: str,
) -> None:
    """Plot exact and simulated observable values for one or more methods."""

    if not series_list or not series_list[0].points:
        return

    sweep_parameter = series_list[0].sweep_parameter
    reference = series_list[0]
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        reference.sweep_values,
        reference.exact_observable_values(name),
        marker="o",
        label=f"Exact <{name}>",
    )
    for series in series_list:
        if not series.points:
            continue
        sim_label = (
            f"Sim <{name}> ({series_label(series)})"
            if len(series_list) > 1
            else f"Sim <{name}>"
        )
        ax.plot(
            series.sweep_values,
            series.observable_values(name),
            marker="s",
            label=sim_label,
        )

    ax.set_xlabel(_sweep_axis_label(sweep_parameter))
    ax.set_ylabel("Expectation value")
    ax.set_title(_observable_plot_title(sweep_parameter, name))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, _observable_plot_filename(sweep_parameter, name)))
    plt.close(fig)


def plot_all_sim_comparison(
    series_list: Sequence[SimBenchmarkResult],
    output_dir: str,
) -> None:
    """Generate the default comparison plot set for one or more method series."""

    if not series_list:
        return

    sweep_parameter = series_list[0].sweep_parameter
    if sweep_parameter == "single_point":
        plot_sim_single_point_comparison(series_list, output_dir)
        return

    plot_infidelity_comparison(series_list, output_dir)
    if series_list[0].points:
        for name in series_list[0].points[0].observables:
            plot_observable_comparison(series_list, output_dir, name)


def plot_all_sim(result: SimBenchmarkResult, output_dir: str) -> None:
    """Backward-compatible wrapper for a single-method benchmark."""

    plot_all_sim_comparison([result], output_dir)


# Backward-compatible aliases used by older call sites.
def plot_infidelity_vs_sweep(result: SimBenchmarkResult, output_dir: str) -> None:
    plot_infidelity_comparison([result], output_dir)


def plot_infidelity_vs_evolution_time(result: SimBenchmarkResult, output_dir: str) -> None:
    plot_infidelity_comparison([result], output_dir)


def plot_observable_vs_sweep(result: SimBenchmarkResult, output_dir: str, name: str) -> None:
    plot_observable_comparison([result], output_dir, name)


def plot_observable_evolution_vs_time(
    result: SimBenchmarkResult, output_dir: str, name: str
) -> None:
    plot_observable_comparison([result], output_dir, name)
