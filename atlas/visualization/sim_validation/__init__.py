"""Validation plots comparing Hamiltonian simulation evolution methods."""

from atlas.visualization.sim_validation.plots import plot_validation_comparison
from atlas.visualization.sim_validation.registry import (
    DEFAULT_VALIDATION_PLOTS,
    register_validation_plot,
)
from atlas.visualization.sim_validation.series import ValidationPlotSpec

__all__ = [
    "DEFAULT_VALIDATION_PLOTS",
    "ValidationPlotSpec",
    "plot_validation_comparison",
    "plot_validation_result",
    "register_validation_plot",
]


def plot_validation_result(result, output_dir: str, plot_names=None) -> None:
    """Plot all series contained in a ``SimValidationResult``."""

    plot_validation_comparison(result.series, output_dir, plot_names=plot_names)
