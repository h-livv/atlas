"""Physics-agnostic lattice data containers for dynamics visualization.

These types describe *what to draw*, not how the values were computed.
Positions, connectivity, and per-site values are supplied by callers
(adapters or notebooks); renderers never reconstruct physics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class LatticeGeometry:
    """Display geometry for a lattice of sites.

    Responsibility:
        Hold node coordinates and optional undirected edges used only for
        rendering. Coordinates are in an arbitrary display space.

    State:
        positions: Array of shape ``(n_sites, dim)`` with ``dim >= 1``.
        edges: Optional array of shape ``(n_edges, 2)`` of integer site indices.
        labels: Optional per-site labels for annotations.

    Usage:
        Built by layout helpers (e.g. ``chain_1d``) or supplied externally for
        arbitrary graphs. Renderers consume this without interpreting physics.
    """

    positions: np.ndarray
    edges: Optional[np.ndarray] = None
    labels: Optional[tuple[str, ...]] = None

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions, dtype=float)
        if positions.ndim != 2 or positions.shape[0] < 1:
            raise ValueError("positions must have shape (n_sites, dim) with n_sites >= 1.")
        if positions.shape[1] < 1:
            raise ValueError("positions must have at least one spatial dimension.")
        object.__setattr__(self, "positions", positions)

        if self.edges is not None:
            edges = np.asarray(self.edges, dtype=int)
            if edges.ndim != 2 or edges.shape[1] != 2:
                raise ValueError("edges must have shape (n_edges, 2).")
            if edges.size and (edges.min() < 0 or edges.max() >= self.num_sites):
                raise ValueError("edge indices must be in [0, n_sites).")
            object.__setattr__(self, "edges", edges)

        if self.labels is not None and len(self.labels) != self.num_sites:
            raise ValueError("labels length must equal n_sites.")

    @property
    def num_sites(self) -> int:
        """Return the number of lattice sites."""

        return int(self.positions.shape[0])

    @property
    def dim(self) -> int:
        """Return the spatial dimension of the stored coordinates."""

        return int(self.positions.shape[1])


@dataclass(frozen=True)
class LatticeFrame:
    """One snapshot of per-site values on a lattice.

    Responsibility:
        Carry the node values and optional time/metadata for a single frame
        of a dashboard frame or static plot.

    State:
        values: Array of shape ``(n_sites,)`` used to color nodes.
        time: Optional scalar time label for the frame.
        metadata: Free-form display metadata (method name, etc.).
    """

    values: np.ndarray
    time: Optional[float] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float).reshape(-1)
        object.__setattr__(self, "values", values)


@dataclass
class LatticeTrajectory:
    """Time-ordered sequence of lattice frames sharing one geometry.

    Responsibility:
        Bundle geometry with frame data for dashboard views. The value name
        is a display label only (e.g. ``\"z\"`` for ``<Z_i>``).

    State:
        geometry: Shared lattice layout.
        frames: Ordered list of ``LatticeFrame``.
        value_name: Label for colorbars/titles.
        vmin / vmax: Optional fixed color limits; when unset, views may
            derive limits from all frame values.
    """

    geometry: LatticeGeometry
    frames: list[LatticeFrame]
    value_name: str = "value"
    vmin: Optional[float] = None
    vmax: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.frames:
            raise ValueError("LatticeTrajectory requires at least one frame.")
        n = self.geometry.num_sites
        for index, frame in enumerate(self.frames):
            if frame.values.shape[0] != n:
                raise ValueError(
                    f"Frame {index} has {frame.values.shape[0]} values; "
                    f"expected {n} to match geometry."
                )

    @property
    def times(self) -> np.ndarray:
        """Return frame times, using NaN where a frame has no time."""

        return np.array(
            [np.nan if frame.time is None else float(frame.time) for frame in self.frames],
            dtype=float,
        )

    @property
    def values(self) -> np.ndarray:
        """Return stacked per-site values with shape ``(n_frames, n_sites)``."""

        return np.stack([frame.values for frame in self.frames], axis=0)

    def color_limits(self) -> tuple[float, float]:
        """Return ``(vmin, vmax)``, deriving from data when unset.

        Diverging maps work best with a symmetric range about zero when the
        data straddles zero; otherwise the raw min/max are used.
        """

        if self.vmin is not None and self.vmax is not None:
            return float(self.vmin), float(self.vmax)

        data = self.values
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            return -1.0, 1.0

        data_min = float(np.min(finite))
        data_max = float(np.max(finite))
        if self.vmin is not None:
            data_min = float(self.vmin)
        if self.vmax is not None:
            data_max = float(self.vmax)

        if data_min < 0.0 < data_max:
            limit = max(abs(data_min), abs(data_max))
            return -limit, limit
        if data_min == data_max:
            pad = 1.0 if data_min == 0.0 else abs(data_min) * 0.1
            return data_min - pad, data_max + pad
        return data_min, data_max
