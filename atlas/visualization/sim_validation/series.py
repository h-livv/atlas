"""Extract and compare validation metrics from simulation benchmark results."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from atlas.experiments.results import SimBenchmarkResult, SimPointResult


PointAccessor = Callable[["SimPointResult"], float]


def _max_operator_error(point: SimPointResult) -> float:
    if not point.observable_errors:
        return float("nan")
    return float(max(point.observable_errors.values()))


def _operator_error(name: str) -> PointAccessor:
    def accessor(point: SimPointResult) -> float:
        return float(point.observable_errors[name])

    return accessor


POINT_ACCESSORS: dict[str, PointAccessor] = {
    "fidelity": lambda point: float(point.fidelity),
    "circuit_depth": lambda point: float(point.sim_result.circuit_depth),
    "num_trotter_steps": lambda point: float(point.sim_result.num_trotter_steps),
    "max_operator_error": _max_operator_error,
}


def resolve_point_accessor(name: str) -> PointAccessor:
    """Return a metric accessor by name.

    Built-in names: ``fidelity``, ``circuit_depth``, ``num_trotter_steps``,
    ``max_operator_error``. Per-observable errors use ``operator_error:<name>``.
    """

    if name in POINT_ACCESSORS:
        return POINT_ACCESSORS[name]
    if name.startswith("operator_error:"):
        observable_name = name.split(":", 1)[1]
        return _operator_error(observable_name)
    raise ValueError(
        f"Unknown validation metric '{name}'. "
        f"Supported: {', '.join(sorted(POINT_ACCESSORS))} and operator_error:<name>."
    )


def register_point_accessor(name: str, accessor: PointAccessor) -> None:
    """Register a custom metric accessor for validation plots."""

    POINT_ACCESSORS[name] = accessor


def series_metric_values(
    benchmark: SimBenchmarkResult,
    accessor_name: str,
) -> np.ndarray:
    """Extract a metric across all points in one benchmark series."""

    accessor = resolve_point_accessor(accessor_name)
    return np.array([accessor(point) for point in benchmark.points])


def series_label(benchmark: SimBenchmarkResult) -> str:
    """Return a human-readable label for a benchmark series."""

    if not benchmark.points:
        return "empty"
    return benchmark.points[0].sim_result.method_name


@dataclass(frozen=True)
class ValidationPlotSpec:
    """Declarative description of a method-comparison validation plot."""

    name: str
    filename: str
    x_metric: str
    y_metric: str
    x_label: str
    y_label: str
    title: str
    log_x: bool = False
    log_y: bool = False
