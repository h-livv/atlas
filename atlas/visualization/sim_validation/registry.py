"""Registry of Hamiltonian simulation validation comparison plots.

Validation plots are declared as ``ValidationPlotSpec`` objects (what to plot)
and optional renderer callables (how to draw them). A registry pattern lets
callers select plots by string name and lets custom workflows register extra
plots without editing the core comparison loop.

Module-level state:
    VALIDATION_PLOT_REGISTRY: Maps plot name -> ``ValidationPlotSpec``.
    VALIDATION_PLOT_RENDERERS: Maps plot name -> renderer callable. If a name
        has a spec but no renderer, ``plots.render_method_comparison`` is used
        as the default at draw time.
    DEFAULT_VALIDATION_PLOTS: Ordered names rendered when the caller does not
        pass an explicit ``plot_names`` list.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from atlas.visualization.sim_validation.series import ValidationPlotSpec

ValidationPlotFn = Callable[[Sequence, str, ValidationPlotSpec], None]

# Populated by register_validation_plot / default registration in plots.py.
VALIDATION_PLOT_REGISTRY: dict[str, ValidationPlotSpec] = {}
VALIDATION_PLOT_RENDERERS: dict[str, ValidationPlotFn] = {}


def register_validation_plot(
    spec: ValidationPlotSpec,
    renderer: ValidationPlotFn | None = None,
) -> None:
    """Register a validation plot for use in comparison workflows.

    Purpose:
        Add (or replace) a named plot specification and optionally bind a
        custom renderer so ``plot_validation_comparison`` can discover it by
        ``spec.name``.

    Inputs:
        spec: A ``ValidationPlotSpec`` describing metrics, labels, filename,
            and optional log axes. ``spec.name`` is the registry key.
        renderer: Optional callable
            ``(series_list, output_dir, spec) -> None``. When ``None``, only
            the spec is stored; drawing falls back to the default method-
            comparison renderer.

    Process:
        1. Store ``spec`` under ``VALIDATION_PLOT_REGISTRY[spec.name]``.
        2. If ``renderer`` is provided, also store it under
           ``VALIDATION_PLOT_RENDERERS[spec.name]``.

    Outputs:
        None.

    Side Effects:
        Mutates the module-level ``VALIDATION_PLOT_REGISTRY`` and, when a
        renderer is given, ``VALIDATION_PLOT_RENDERERS`` dictionaries. Later
        registrations with the same name overwrite earlier ones.
    """

    VALIDATION_PLOT_REGISTRY[spec.name] = spec
    if renderer is not None:
        VALIDATION_PLOT_RENDERERS[spec.name] = renderer


# Names used when plot_names is omitted; must match keys registered in plots.py.
DEFAULT_VALIDATION_PLOTS: tuple[str, ...] = (
    "infidelity_vs_trotter_steps",
    "operator_error_vs_trotter_steps",
    "circuit_depth_vs_infidelity",
)
