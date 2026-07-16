"""Registry of Hamiltonian simulation validation comparison plots."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from atlas.visualization.sim_validation.series import ValidationPlotSpec

ValidationPlotFn = Callable[[Sequence, str, ValidationPlotSpec], None]

VALIDATION_PLOT_REGISTRY: dict[str, ValidationPlotSpec] = {}
VALIDATION_PLOT_RENDERERS: dict[str, ValidationPlotFn] = {}


def register_validation_plot(
    spec: ValidationPlotSpec,
    renderer: ValidationPlotFn | None = None,
) -> None:
    """Register a validation plot for use in comparison workflows."""

    VALIDATION_PLOT_REGISTRY[spec.name] = spec
    if renderer is not None:
        VALIDATION_PLOT_RENDERERS[spec.name] = renderer


DEFAULT_VALIDATION_PLOTS: tuple[str, ...] = (
    "fidelity_vs_trotter_steps",
    "operator_error_vs_trotter_steps",
    "circuit_depth_vs_accuracy",
)
