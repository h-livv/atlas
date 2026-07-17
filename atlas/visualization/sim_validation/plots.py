"""Render validation comparison plots for Hamiltonian simulation.

This module is the drawing side of validation plotting. It registers the
default ``ValidationPlotSpec`` entries at import time and exposes
``plot_validation_comparison`` to walk selected registry names and call the
matching renderer for each.

``matplotlib.use("Agg")`` is set before importing ``pyplot`` so these functions
work in headless CI / server environments without a display.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import matplotlib

# Agg is a non-interactive backend: required so plot scripts work without a GUI.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from atlas.visualization.sim_validation.registry import (
    DEFAULT_VALIDATION_PLOTS,
    VALIDATION_PLOT_REGISTRY,
    VALIDATION_PLOT_RENDERERS,
    register_validation_plot,
)
from atlas.visualization.sim_validation.series import (
    ValidationPlotSpec,
    series_label,
    series_metric_values,
)


def render_method_comparison(
    series_list: Sequence,
    output_dir: str,
    spec: ValidationPlotSpec,
) -> None:
    """Plot one metric against another with one line per evolution method.

    Purpose:
        Default renderer for registered validation specs: overlay each
        ``SimBenchmarkResult`` series as a labeled line using the metrics named
        in ``spec``.

    Inputs:
        series_list: Sequence of ``SimBenchmarkResult`` (or compatible) objects,
            typically one per evolution method under comparison.
        output_dir: Directory where ``spec.filename`` is written. Created if
            missing.
        spec: Plot declaration (metrics, labels, filename, optional log axes).

    Process:
        1. Ensure ``output_dir`` exists.
        2. For each non-empty series, extract x/y via ``series_metric_values``
           and plot with ``series_label`` as the legend entry.
        3. Apply labels, optional log scales, grid; add a legend only when
           more than one series was supplied.
        4. Save ``spec.filename`` and close the figure.

    Outputs:
        None. Writes ``spec.filename`` under ``output_dir``.

    Side Effects:
        Creates ``output_dir`` if missing; creates/overwrites the PNG; allocates
        and closes a matplotlib figure.
    """

    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    for benchmark in series_list:
        if not benchmark.points:
            # Empty series would produce empty arrays; skip rather than clutter the legend.
            continue
        x_values = series_metric_values(benchmark, spec.x_metric)
        y_values = series_metric_values(benchmark, spec.y_metric)
        ax.plot(
            x_values,
            y_values,
            marker="o",
            label=series_label(benchmark),
        )

    ax.set_xlabel(spec.x_label)
    ax.set_ylabel(spec.y_label)
    ax.set_title(spec.title)
    if spec.log_x:
        ax.set_xscale("log")
    if spec.log_y:
        ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    if len(series_list) > 1:
        # Single-series plots don't need a legend (only one method).
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, spec.filename))
    plt.close(fig)


def _register_default_validation_plots() -> None:
    """Register the built-in fidelity / operator-error / depth plots.

    Purpose:
        Populate the validation registry at import time so
        ``DEFAULT_VALIDATION_PLOTS`` names resolve without an explicit setup
        call from callers.

    Inputs:
        None.

    Process:
        Call ``register_validation_plot`` three times with the standard specs
        and ``render_method_comparison`` as the shared renderer.

    Outputs:
        None.

    Side Effects:
        Mutates ``VALIDATION_PLOT_REGISTRY`` and ``VALIDATION_PLOT_RENDERERS``
        via ``register_validation_plot``.
    """

    register_validation_plot(
        ValidationPlotSpec(
            name="fidelity_vs_trotter_steps",
            filename="validation_fidelity_vs_trotter_steps.png",
            x_metric="num_trotter_steps",
            y_metric="fidelity",
            x_label="Trotter steps",
            y_label="State fidelity",
            title="Fidelity vs Trotter steps",
            log_x=True,
        ),
        render_method_comparison,
    )
    register_validation_plot(
        ValidationPlotSpec(
            name="operator_error_vs_trotter_steps",
            filename="validation_operator_error_vs_trotter_steps.png",
            x_metric="num_trotter_steps",
            y_metric="max_operator_error",
            x_label="Trotter steps",
            y_label="Max operator error",
            title="Operator error vs Trotter steps",
            log_x=True,
            log_y=True,
        ),
        render_method_comparison,
    )
    register_validation_plot(
        ValidationPlotSpec(
            name="circuit_depth_vs_accuracy",
            filename="validation_circuit_depth_vs_accuracy.png",
            x_metric="circuit_depth",
            y_metric="fidelity",
            x_label="Circuit depth",
            y_label="State fidelity (accuracy)",
            title="Circuit depth vs accuracy",
        ),
        render_method_comparison,
    )


# Import-time registration so DEFAULT_VALIDATION_PLOTS is usable immediately.
_register_default_validation_plots()


def plot_validation_comparison(
    series_list: Sequence,
    output_dir: str,
    plot_names: Sequence[str] | None = None,
) -> None:
    """Generate registered validation plots for one or more method series.

    Purpose:
        Drive the registry: for each requested plot name, look up the spec and
        renderer and invoke the renderer with the provided method series.

    Inputs:
        series_list: Sequence of benchmark series (one per method / config).
        output_dir: Destination directory passed to each renderer.
        plot_names: Optional subset of registered names. Defaults to
            ``DEFAULT_VALIDATION_PLOTS``.

    Process:
        1. Choose ``plot_names`` or the default tuple.
        2. For each name, fetch the ``ValidationPlotSpec`` from
           ``VALIDATION_PLOT_REGISTRY`` (KeyError if unknown).
        3. Fetch a custom renderer from ``VALIDATION_PLOT_RENDERERS`` if
           present; otherwise fall back to ``render_method_comparison``.
        4. Call ``renderer(series_list, output_dir, spec)``.

    Outputs:
        None. One PNG per selected plot (via the renderers).

    Side Effects:
        Delegates filesystem and matplotlib side effects to the chosen
        renderers. Raises ``KeyError`` if a requested name is not registered.
    """

    selected = plot_names or DEFAULT_VALIDATION_PLOTS
    for plot_name in selected:
        spec = VALIDATION_PLOT_REGISTRY[plot_name]
        # Specs without a custom renderer share the default method-comparison drawer.
        renderer = VALIDATION_PLOT_RENDERERS.get(plot_name, render_method_comparison)
        renderer(series_list, output_dir, spec)
