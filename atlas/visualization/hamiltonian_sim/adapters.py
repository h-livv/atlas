"""Adapters that map experiment results onto lattice visualization data.

Passive only: read already-computed site series from result containers.
Never import Hamiltonians, evolution methods, or recompute observables.
"""

from __future__ import annotations

from typing import Literal, Optional, Sequence

import numpy as np

from atlas.experiments.results import SimBenchmarkResult, SimPointResult
from atlas.visualization.hamiltonian_sim.lattice import (
    LatticeFrame,
    LatticeGeometry,
    LatticeTrajectory,
)
from atlas.visualization.hamiltonian_sim.layouts import chain_1d


SourceKind = Literal["sim", "exact"]


def _default_geometry(num_sites: int) -> LatticeGeometry:
    return chain_1d(num_sites)


def trajectory_from_site_series(
    times: Sequence[float],
    values: np.ndarray,
    *,
    value_name: str = "z",
    geometry: Optional[LatticeGeometry] = None,
    metadata: Optional[dict] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> LatticeTrajectory:
    """Build a ``LatticeTrajectory`` from times and a ``(T, n_sites)`` array.

    Purpose:
        Lowest-level adapter for any precomputed site series, independent of
        Atlas experiment types.
    """

    values_arr = np.asarray(values, dtype=float)
    if values_arr.ndim != 2:
        raise ValueError("values must have shape (n_frames, n_sites).")
    if len(times) != values_arr.shape[0]:
        raise ValueError("times length must match the number of frames.")

    geo = geometry or _default_geometry(values_arr.shape[1])
    if geo.num_sites != values_arr.shape[1]:
        raise ValueError(
            f"geometry has {geo.num_sites} sites but values have "
            f"{values_arr.shape[1]} columns."
        )

    meta = dict(metadata or {})
    frames = [
        LatticeFrame(values=values_arr[i], time=float(times[i]), metadata=dict(meta))
        for i in range(values_arr.shape[0])
    ]
    return LatticeTrajectory(
        geometry=geo,
        frames=frames,
        value_name=value_name,
        vmin=vmin,
        vmax=vmax,
    )


def trajectory_from_sim_point(
    point: SimPointResult,
    *,
    site_observable: str = "z",
    source: SourceKind = "sim",
    geometry: Optional[LatticeGeometry] = None,
) -> LatticeTrajectory:
    """Build a single-frame trajectory from one ``SimPointResult``."""

    series = (
        point.exact_site_observables
        if source == "exact"
        else point.site_observables
    )
    if site_observable not in series:
        raise KeyError(
            f"Site observable '{site_observable}' not found on point. "
            f"Available: {sorted(series)}"
        )
    values = np.asarray(series[site_observable], dtype=float).reshape(1, -1)
    return trajectory_from_site_series(
        times=[point.evolution_time],
        values=values,
        value_name=site_observable,
        geometry=geometry,
        metadata={"method_name": point.method_name, "source": source},
        vmin=-1.0,
        vmax=1.0,
    )


def trajectory_from_sim_benchmark(
    result: SimBenchmarkResult,
    *,
    site_observable: str = "z",
    source: SourceKind = "sim",
    geometry: Optional[LatticeGeometry] = None,
) -> LatticeTrajectory:
    """Build a multi-frame trajectory from a dynamics benchmark series.

    Purpose:
        Prefer ``evolution_time`` sweeps for the lattice dashboard; any sweep
        works as long as each point carries the named site observable array.
    """

    if not result.points:
        raise ValueError("SimBenchmarkResult has no points.")

    values = result.site_observable_series(site_observable, source=source)
    times = result.sweep_values
    method = result.method_name
    return trajectory_from_site_series(
        times=times,
        values=values,
        value_name=site_observable,
        geometry=geometry,
        metadata={"method_name": method, "source": source},
        vmin=-1.0,
        vmax=1.0,
    )


def available_site_observables(result: SimBenchmarkResult | SimPointResult) -> list[str]:
    """Return site-observable names present on the result (sim side)."""

    if isinstance(result, SimBenchmarkResult):
        if not result.points:
            return []
        return sorted(result.points[0].site_observables)
    return sorted(result.site_observables)
