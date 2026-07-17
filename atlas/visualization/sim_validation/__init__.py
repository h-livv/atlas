"""Public API for Hamiltonian simulation validation plots.

This package compares multiple evolution-method series (e.g. different Trotter
formulas or step counts) by extracting metrics from ``SimBenchmarkResult``
objects and rendering registered comparison plots.

Import from here rather than the submodules when you only need the common
entry points: ``plot_validation_result``, ``plot_validation_comparison``,
``register_validation_plot``, ``ValidationPlotSpec``, and
``DEFAULT_VALIDATION_PLOTS``.
"""

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
    """Plot all series contained in a ``SimValidationResult``.

    Purpose:
        Thin adapter from Atlas's validation result object to the lower-level
        ``plot_validation_comparison`` API, which expects a sequence of
        benchmark series rather than the wrapper result type.

    Inputs:
        result: A ``SimValidationResult`` (or compatible object) exposing a
            ``.series`` attribute — a sequence of ``SimBenchmarkResult``
            instances, one per evolution method / configuration under test.
        output_dir: Directory where registered validation PNGs are written.
        plot_names: Optional sequence of registered plot names to render.
            When ``None``, ``DEFAULT_VALIDATION_PLOTS`` is used (via
            ``plot_validation_comparison``).

    Process:
        Forward ``result.series``, ``output_dir``, and ``plot_names`` to
        ``plot_validation_comparison``.

    Outputs:
        None. PNG files are written by the comparison renderer.

    Side Effects:
        Delegates filesystem and matplotlib side effects to
        ``plot_validation_comparison`` and its registered renderers.
    """

    plot_validation_comparison(result.series, output_dir, plot_names=plot_names)
