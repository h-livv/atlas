"""Plots for Hamiltonian simulation benchmark results.

Passive plotting helpers that consume already-computed
``SimBenchmarkResult`` / ``SimPointResult`` objects and write PNG artifacts.
They never run simulation, exact evolution, or hardware jobs — all numeric
values come from the result objects produced upstream in the experiment layer.

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

from atlas.experiments.results import SimBenchmarkResult, SimPointResult


def plot_sim_single_point(point: SimPointResult, output_dir: str) -> None:
    """Bar chart comparing exact and simulated observable expectations.

    Purpose:
        Visualize, for a single evolution-time / method point, how each named
        observable's simulated expectation compares to the exact reference.

    Inputs:
        point: A ``SimPointResult`` with populated ``observables`` (simulated)
            and ``exact_observables`` (reference) dictionaries keyed by the same
            observable names, plus metadata used in the plot title
            (``evolution_time``, ``sim_result.method_name``).
        output_dir: Directory where ``sim_single_point_observables.png`` is
            written. Created if it does not already exist.

    Process:
        1. Ensure ``output_dir`` exists.
        2. Collect observable names and paired exact/simulated values.
        3. Draw grouped bars (exact vs simulated) and label axes/title.
        4. Save the figure and close it to free memory.

    Outputs:
        None. Writes ``sim_single_point_observables.png`` under ``output_dir``.

    Side Effects:
        Creates ``output_dir`` if missing; creates/overwrites the PNG; allocates
        and closes a matplotlib figure.
    """

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
    """Plot state fidelity against the swept parameter.

    Purpose:
        Show how state fidelity (simulated vs exact) varies across the sweep
        axis stored on ``result`` (e.g. evolution time or Trotter steps).

    Inputs:
        result: A ``SimBenchmarkResult`` providing ``sweep_values``,
            ``fidelities``, and ``sweep_parameter`` (used as the x-axis label).
        output_dir: Directory for ``sim_fidelity_vs_sweep.png``. Created if
            missing.

    Process:
        1. Ensure ``output_dir`` exists.
        2. Plot fidelity vs sweep values with markers.
        3. Label axes from the result metadata, add a light grid, and save.

    Outputs:
        None. Writes ``sim_fidelity_vs_sweep.png`` under ``output_dir``.

    Side Effects:
        Creates ``output_dir`` if missing; creates/overwrites the PNG; allocates
        and closes a matplotlib figure.
    """

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
    """Plot one observable's exact and simulated values across a sweep.

    Purpose:
        Overlay exact and simulated expectation values for a single named
        observable so systematic bias or divergence along the sweep is visible.

    Inputs:
        result: A ``SimBenchmarkResult`` with helpers
            ``exact_observable_values(name)`` and ``observable_values(name)``,
            plus ``sweep_values`` / ``sweep_parameter``.
        output_dir: Directory for ``sim_{name}_vs_sweep.png``. Created if
            missing.
        name: Observable key (must exist on each point's observable dicts).

    Process:
        1. Ensure ``output_dir`` exists.
        2. Plot exact and simulated series vs the sweep axis.
        3. Label, legend, grid, save, and close the figure.

    Outputs:
        None. Writes ``sim_{name}_vs_sweep.png`` under ``output_dir``.

    Side Effects:
        Creates ``output_dir`` if missing; creates/overwrites the PNG; allocates
        and closes a matplotlib figure.
    """

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
    """Generate the default plot set for a simulation benchmark.

    Purpose:
        Convenience entry point used by experiment runners to dump the standard
        fidelity-vs-sweep plot plus one observable-vs-sweep plot per observable
        found on the first sweep point.

    Inputs:
        result: A ``SimBenchmarkResult``. If ``result.points`` is empty, only
            the fidelity plot is attempted (observable plots are skipped).
        output_dir: Destination directory passed through to the individual
            plotters.

    Process:
        1. Always call ``plot_fidelity_vs_sweep``.
        2. If there is at least one point, iterate observable names from
           ``result.points[0].observables`` and call ``plot_observable_vs_sweep``
           for each.

    Outputs:
        None. Multiple PNG files under ``output_dir`` (via the helpers above).

    Side Effects:
        Delegates all filesystem and matplotlib side effects to the called
        plot functions.
    """

    plot_fidelity_vs_sweep(result, output_dir)
    if result.points:
        # Observable names are taken from the first point; all points share keys.
        for name in result.points[0].observables:
            plot_observable_vs_sweep(result, output_dir, name)
