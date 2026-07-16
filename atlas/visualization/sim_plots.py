"""Plots for Hamiltonian simulation benchmark results."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from atlas.experiments.results import SimBenchmarkResult, SimPointResult


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
        f"TFIM dynamics (t={point.evolution_time}, method={point.sim_result.method_name})"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "sim_single_point_observables.png"))
    plt.close(fig)


def plot_fidelity_vs_sweep(result: SimBenchmarkResult, output_dir: str) -> None:
    """Plot state fidelity against the swept parameter."""

    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(result.sweep_values, result.fidelities, marker="o")
    ax.set_xlabel(result.sweep_parameter)
    ax.set_ylabel("Fidelity")
    ax.set_title("Simulation fidelity vs sweep parameter")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "sim_fidelity_vs_sweep.png"))
    plt.close(fig)


def plot_observable_vs_sweep(result: SimBenchmarkResult, output_dir: str, name: str) -> None:
    """Plot one observable's exact and simulated values across a sweep."""

    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        result.sweep_values,
        result.exact_observable_values(name),
        marker="o",
        label=f"Exact <{name}>",
    )
    ax.plot(
        result.sweep_values,
        result.observable_values(name),
        marker="s",
        label=f"Sim <{name}>",
    )
    ax.set_xlabel(result.sweep_parameter)
    ax.set_ylabel("Expectation value")
    ax.set_title(f"Observable {name} vs {result.sweep_parameter}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"sim_{name}_vs_sweep.png"))
    plt.close(fig)


def plot_all_sim(result: SimBenchmarkResult, output_dir: str) -> None:
    """Generate the default plot set for a simulation benchmark."""

    plot_fidelity_vs_sweep(result, output_dir)
    if result.points:
        for name in result.points[0].observables:
            plot_observable_vs_sweep(result, output_dir, name)
