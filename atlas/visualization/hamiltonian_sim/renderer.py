"""Geometry-agnostic lattice/graph renderer.

Draws nodes and edges from supplied positions, connectivity, and scalar
values only. Never computes observables, layouts from physics, or inspects
Hamiltonians / evolution methods.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.figure import Figure

from atlas.visualization.hamiltonian_sim.colors import (
    DEFAULT_DIVERGING_CMAP,
    diverging_norm,
    resolve_colormap,
)
from atlas.visualization.hamiltonian_sim.lattice import (
    LatticeFrame,
    LatticeGeometry,
    LatticeTrajectory,
)

ColorLike = Union[str, Sequence[str], np.ndarray]


class LatticeGraphRenderer:
    """Render an arbitrary lattice/graph in the plane.

    Responsibility:
        Map geometry (positions, optional edges, optional labels) and
        per-node values onto a matplotlib axes. The renderer does not know
        whether the graph is a 1D chain, 2D grid, or arbitrary interaction
        graph.

    Extensibility:
        Edges are drawn independently from nodes so future edge-value
        colouring (e.g. bond observables) can be added without changing the
        node path or physics layer.
    """

    def __init__(
        self,
        *,
        cmap: str = DEFAULT_DIVERGING_CMAP,
        node_size: float = 700.0,
        node_edgecolors: str = "black",
        node_linewidths: float = 1.0,
        edge_color: str = "#1f77b4",
        edge_width: float = 3.5,
        edge_cmap: str = DEFAULT_DIVERGING_CMAP,
        show_labels: bool = True,
        label_fontsize: float = 11.0,
        label_offset: tuple[float, float] = (0.0, 18.0),
        axis_pad_fraction: float = 0.18,
        axis_pad_min: float = 0.35,
    ) -> None:
        self.cmap = cmap
        self.node_size = node_size
        self.node_edgecolors = node_edgecolors
        self.node_linewidths = node_linewidths
        self.edge_color = edge_color
        self.edge_width = edge_width
        self.edge_cmap = edge_cmap
        self.show_labels = show_labels
        self.label_fontsize = label_fontsize
        self.label_offset = label_offset
        self.axis_pad_fraction = axis_pad_fraction
        self.axis_pad_min = axis_pad_min

    @staticmethod
    def project_xy(geometry: LatticeGeometry) -> tuple[np.ndarray, np.ndarray]:
        """Project stored coordinates onto a 2D display plane.

        Uses the first two components when available; a pure 1-D position
        array is placed on ``y = 0``.
        """

        positions = geometry.positions
        if geometry.dim == 1:
            return positions[:, 0], np.zeros(geometry.num_sites, dtype=float)
        return positions[:, 0], positions[:, 1]

    def draw_edges(
        self,
        ax: plt.Axes,
        geometry: LatticeGeometry,
        *,
        edge_values: Optional[np.ndarray] = None,
        edge_colors: Optional[ColorLike] = None,
        edge_vmin: Optional[float] = None,
        edge_vmax: Optional[float] = None,
        zorder: float = 1,
    ) -> Optional[LineCollection]:
        """Draw graph edges independently of nodes.

        Purpose:
            Keep edge rendering separable so bond observables can later drive
            colour without touching the node path.

        Inputs:
            ax / geometry: Display target and connectivity.
            edge_values: Optional length-``n_edges`` scalars for future
                colouring. Ignored when ``edge_colors`` is provided.
            edge_colors: Explicit per-edge or single color override.
            edge_vmin / edge_vmax: Limits when mapping ``edge_values``.

        Outputs:
            The ``LineCollection``, or ``None`` when there are no edges.
        """

        if geometry.edges is None or len(geometry.edges) == 0:
            return None

        x, y = self.project_xy(geometry)
        segments = [[(x[i], y[i]), (x[j], y[j])] for i, j in geometry.edges]

        kwargs: dict = {
            "linewidths": self.edge_width,
            "zorder": zorder,
        }

        if edge_colors is not None:
            kwargs["colors"] = edge_colors
        elif edge_values is not None:
            values = np.asarray(edge_values, dtype=float).reshape(-1)
            if values.shape[0] != len(segments):
                raise ValueError(
                    f"edge_values length ({values.shape[0]}) must match "
                    f"number of edges ({len(segments)})."
                )
            vmin = float(np.min(values) if edge_vmin is None else edge_vmin)
            vmax = float(np.max(values) if edge_vmax is None else edge_vmax)
            if vmin == vmax:
                vmin, vmax = vmin - 1.0, vmax + 1.0
            norm = diverging_norm(vmin, vmax)
            cmap = resolve_colormap(self.edge_cmap)
            kwargs["colors"] = cmap(norm(values))
        else:
            kwargs["colors"] = self.edge_color

        collection = LineCollection(segments, **kwargs)
        ax.add_collection(collection)
        return collection

    def draw_nodes(
        self,
        ax: plt.Axes,
        geometry: LatticeGeometry,
        values: np.ndarray,
        *,
        vmin: float,
        vmax: float,
        zorder: float = 2,
    ) -> PathCollection:
        """Draw colored nodes from per-site scalar values."""

        x, y = self.project_xy(geometry)
        norm = diverging_norm(vmin, vmax)
        return ax.scatter(
            x,
            y,
            c=np.asarray(values, dtype=float),
            s=self.node_size,
            cmap=resolve_colormap(self.cmap),
            norm=norm,
            edgecolors=self.node_edgecolors,
            linewidths=self.node_linewidths,
            zorder=zorder,
        )

    def draw_labels(
        self,
        ax: plt.Axes,
        geometry: LatticeGeometry,
        *,
        zorder: float = 3,
    ) -> None:
        """Annotate nodes with optional geometry labels."""

        if not self.show_labels or geometry.labels is None:
            return
        x, y = self.project_xy(geometry)
        dx, dy = self.label_offset
        for index, label in enumerate(geometry.labels):
            ax.annotate(
                label,
                (x[index], y[index]),
                textcoords="offset points",
                xytext=(dx, dy),
                ha="center",
                fontsize=self.label_fontsize,
                zorder=zorder,
            )

    def fit_axes(self, ax: plt.Axes, geometry: LatticeGeometry) -> None:
        """Fit axes limits tightly around the graph with modest padding."""

        x, y = self.project_xy(geometry)
        x_min, x_max = float(np.min(x)), float(np.max(x))
        y_min, y_max = float(np.min(y)), float(np.max(y))
        x_span = max(x_max - x_min, 1.0)
        y_span = y_max - y_min

        x_pad = max(self.axis_pad_min, self.axis_pad_fraction * x_span)
        if y_span < 1e-9:
            y_pad = max(self.axis_pad_min, 0.45 * x_pad)
        else:
            y_pad = max(self.axis_pad_min, self.axis_pad_fraction * y_span)

        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

    def draw_frame(
        self,
        ax: plt.Axes,
        geometry: LatticeGeometry,
        frame: LatticeFrame,
        *,
        vmin: float,
        vmax: float,
        edge_values: Optional[np.ndarray] = None,
        clear: bool = True,
    ) -> tuple[PathCollection, Optional[LineCollection]]:
        """Compose edges then nodes for one frame.

        Outputs:
            ``(node_collection, edge_collection)``.
        """

        if clear:
            ax.clear()
        edges = self.draw_edges(ax, geometry, edge_values=edge_values)
        nodes = self.draw_nodes(ax, geometry, frame.values, vmin=vmin, vmax=vmax)
        self.draw_labels(ax, geometry)
        self.fit_axes(ax, geometry)
        return nodes, edges

    def render_static(
        self,
        trajectory: LatticeTrajectory,
        *,
        frame_index: int = -1,
        figsize: tuple[float, float] = (10.0, 4.0),
        show_colorbar: bool = True,
    ) -> Figure:
        """Return a figure for a single trajectory frame (default: last)."""

        frame = trajectory.frames[frame_index]
        vmin, vmax = trajectory.color_limits()
        fig, ax = plt.subplots(figsize=figsize)
        nodes, _edges = self.draw_frame(
            ax,
            trajectory.geometry,
            frame,
            vmin=vmin,
            vmax=vmax,
        )
        if show_colorbar:
            cbar = fig.colorbar(nodes, ax=ax, fraction=0.035, pad=0.02)
            cbar.set_label(f"<{trajectory.value_name}>")
        fig.tight_layout()
        return fig


# Backward-compatible alias used by older call sites / docs.
LatticeNodeRenderer = LatticeGraphRenderer
