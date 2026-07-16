"""Render validation comparison plots for Hamiltonian simulation."""

from __future__ import annotations

import os
from collections.abc import Sequence

import matplotlib

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
    """Plot one metric against another with one line per evolution method."""

    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    for benchmark in series_list:
        if not benchmark.points:
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
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, spec.filename))
    plt.close(fig)


def _register_default_validation_plots() -> None:
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


_register_default_validation_plots()


def plot_validation_comparison(
    series_list: Sequence,
    output_dir: str,
    plot_names: Sequence[str] | None = None,
) -> None:
    """Generate registered validation plots for one or more method series."""

    selected = plot_names or DEFAULT_VALIDATION_PLOTS
    for plot_name in selected:
        spec = VALIDATION_PLOT_REGISTRY[plot_name]
        renderer = VALIDATION_PLOT_RENDERERS.get(plot_name, render_method_comparison)
        renderer(series_list, output_dir, spec)
