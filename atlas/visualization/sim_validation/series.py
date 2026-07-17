"""Extract and compare validation metrics from simulation benchmark results.

This module is the data side of validation plotting: it turns a
``SimBenchmarkResult`` (one evolution-method series) into numeric arrays that
plotters can draw. Metrics are addressed by string name via accessors so plot
specs stay declarative and new metrics can be registered without changing the
renderer.

Built-in accessors cover fidelity, infidelity, circuit depth, Trotter step
count, and max observable error. Per-observable errors use the
``operator_error:<name>`` prefix resolved at lookup time.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from atlas.experiments.results import SimBenchmarkResult, SimPointResult


PointAccessor = Callable[["SimPointResult"], float]


def _max_operator_error(point: SimPointResult) -> float:
    """Return the largest absolute observable error on a single point.

    Purpose:
        Provide a scalar summary of operator accuracy when a plot needs one
        y-value per Trotter/settings point rather than per observable.

    Inputs:
        point: A ``SimPointResult`` whose ``observable_errors`` maps observable
            names to absolute errors (may be empty).

    Process:
        If ``observable_errors`` is empty, return ``nan`` (so plots can skip or
        show a gap rather than inventing zero error). Otherwise return the max
        of the stored values as a float.

    Outputs:
        A float (possibly ``nan``).

    Side Effects:
        None.
    """

    if not point.observable_errors:
        # Empty errors => undefined max; nan keeps the series length aligned.
        return float("nan")
    return float(max(point.observable_errors.values()))


def _operator_error(name: str) -> PointAccessor:
    """Build an accessor that reads one named observable's error.

    Purpose:
        Factory for ``operator_error:<name>`` metrics so each observable does
        not need a hand-written entry in ``POINT_ACCESSORS``.

    Inputs:
        name: Observable key present in ``point.observable_errors``.

    Process:
        Return a closure that looks up ``point.observable_errors[name]`` and
        casts to float.

    Outputs:
        A ``PointAccessor`` callable.

    Side Effects:
        None at construction time. The returned accessor raises ``KeyError`` if
        ``name`` is missing on a given point.
    """

    def accessor(point: SimPointResult) -> float:
        return float(point.observable_errors[name])

    return accessor


POINT_ACCESSORS: dict[str, PointAccessor] = {
    "fidelity": lambda point: float(point.fidelity),
    "infidelity": lambda point: float(point.infidelity),
    "circuit_depth": lambda point: float(point.sim_result.circuit_depth),
    "num_trotter_steps": lambda point: float(point.sim_result.num_trotter_steps),
    "max_operator_error": _max_operator_error,
}


def resolve_point_accessor(name: str) -> PointAccessor:
    """Return a metric accessor by name.

    Purpose:
        Map the string metric names used in ``ValidationPlotSpec`` to callables
        that extract floats from ``SimPointResult`` instances.

    Inputs:
        name: Built-in names ``fidelity``, ``infidelity``, ``circuit_depth``,
            ``num_trotter_steps``, ``max_operator_error``, or a dynamic
            ``operator_error:<observable>`` key.

    Process:
        1. If ``name`` is in ``POINT_ACCESSORS``, return that accessor.
        2. Else if ``name`` starts with ``operator_error:``, split off the
           observable name and return ``_operator_error(...)``.
        3. Otherwise raise ``ValueError`` listing supported names.

    Outputs:
        A ``PointAccessor``.

    Side Effects:
        None (unless the caller later invokes the accessor).

    Raises:
        ValueError: When ``name`` is not a known built-in or
            ``operator_error:...`` form.
    """

    if name in POINT_ACCESSORS:
        return POINT_ACCESSORS[name]
    if name.startswith("operator_error:"):
        # Dynamic per-observable metric; not pre-registered in POINT_ACCESSORS.
        observable_name = name.split(":", 1)[1]
        return _operator_error(observable_name)
    raise ValueError(
        f"Unknown validation metric '{name}'. "
        f"Supported: {', '.join(sorted(POINT_ACCESSORS))} and operator_error:<name>."
    )


def register_point_accessor(name: str, accessor: PointAccessor) -> None:
    """Register a custom metric accessor for validation plots.

    Purpose:
        Allow experiment-specific metrics (e.g. wall-clock time, gate counts)
        to plug into the same ``ValidationPlotSpec`` / series extraction path.

    Inputs:
        name: Registry key used later in ``x_metric`` / ``y_metric`` fields.
        accessor: Callable ``(SimPointResult) -> float``.

    Process:
        Assign ``POINT_ACCESSORS[name] = accessor`` (overwrites on collision).

    Outputs:
        None.

    Side Effects:
        Mutates the module-level ``POINT_ACCESSORS`` dictionary.
    """

    POINT_ACCESSORS[name] = accessor


def series_metric_values(
    benchmark: SimBenchmarkResult,
    accessor_name: str,
) -> np.ndarray:
    """Extract a metric across all points in one benchmark series.

    Purpose:
        Produce the x or y array for one evolution-method line on a validation
        comparison plot.

    Inputs:
        benchmark: A ``SimBenchmarkResult`` whose ``points`` are ordered along
            the sweep (e.g. increasing Trotter steps).
        accessor_name: Metric name accepted by ``resolve_point_accessor``.

    Process:
        Resolve the accessor, evaluate it on every point, and stack into a
        NumPy array.

    Outputs:
        A 1-D ``np.ndarray`` of floats, one entry per point (may include
        ``nan`` for some metrics).

    Side Effects:
        None.
    """

    accessor = resolve_point_accessor(accessor_name)
    return np.array([accessor(point) for point in benchmark.points])


def series_label(benchmark: SimBenchmarkResult) -> str:
    """Return a human-readable label for a benchmark series.

    Purpose:
        Provide the legend entry for one method line. Uses the evolution
        ``method_name`` from the first point (all points in a series share a
        method by construction).

    Inputs:
        benchmark: A ``SimBenchmarkResult`` that may be empty.

    Process:
        If there are no points, return ``"empty"``. Otherwise return
        ``benchmark.points[0].sim_result.method_name``.

    Outputs:
        A string label.

    Side Effects:
        None.
    """

    if not benchmark.points:
        return "empty"
    return benchmark.points[0].sim_result.method_name


@dataclass(frozen=True)
class ValidationPlotSpec:
    """Declarative description of a method-comparison validation plot.

    Purpose / responsibility:
        Capture everything needed to render one registered validation figure
        without hard-coding metric names in the renderer: which metrics map to
        x/y, axis labels, title, output filename, and optional log scales.

    State stored:
        name: Registry key (also used in ``DEFAULT_VALIDATION_PLOTS``).
        filename: Basename of the PNG written under the output directory.
        x_metric / y_metric: Accessor names resolved via
            ``resolve_point_accessor``.
        x_label / y_label / title: Matplotlib axis/title strings.
        log_x / log_y: When True, the corresponding axis uses a log scale
            (useful for Trotter-step sweeps and error magnitudes).

    How to use:
        Construct a frozen instance and pass it to
        ``register_validation_plot(spec, renderer=...)``. The comparison
        renderer reads fields but does not mutate the spec.
    """

    name: str
    filename: str
    x_metric: str
    y_metric: str
    x_label: str
    y_label: str
    title: str
    log_x: bool = False
    log_y: bool = False
