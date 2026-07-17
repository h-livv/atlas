"""Generic graph/lattice view for the dynamics dashboard.

Uses ``LatticeGraphRenderer`` so 1D chains, 2D grids, and arbitrary graphs
share one drawing path. Layout comes from the bound trajectory geometry.
"""

from __future__ import annotations

from typing import Optional

from matplotlib.collections import LineCollection, PathCollection
from matplotlib.colorbar import Colorbar

from atlas.visualization.hamiltonian_sim.colors import DEFAULT_DIVERGING_CMAP
from atlas.visualization.hamiltonian_sim.renderer import LatticeGraphRenderer
from atlas.visualization.hamiltonian_sim.views.base import LatticeView
from atlas import profiling as profile


class GraphLatticeView(LatticeView):
    """Dashboard panel that renders whatever geometry the trajectory carries.

    Responsibility:
        Bind a ``LatticeGraphRenderer`` to an axes and refresh node colours
        (and optional future edge values) when the frame index changes.

    Notes:
        Does not invent layouts or compute physics. Upstream adapters supply
        ``LatticeGeometry`` (e.g. ``chain_1d`` for today's TFIM runs).
    """

    def __init__(
        self,
        *,
        title: Optional[str] = None,
        renderer: Optional[LatticeGraphRenderer] = None,
        cmap: str = DEFAULT_DIVERGING_CMAP,
        node_size: float = 700.0,
        edge_color: str = "#1f77b4",
        edge_width: float = 3.5,
        show_labels: bool = True,
        show_colorbar: bool = True,
    ) -> None:
        super().__init__(title=title)
        self.renderer = renderer or LatticeGraphRenderer(
            cmap=cmap,
            node_size=node_size,
            edge_color=edge_color,
            edge_width=edge_width,
            show_labels=show_labels,
        )
        self.show_colorbar = show_colorbar
        self._scatter: Optional[PathCollection] = None
        self._edges: Optional[LineCollection] = None
        self._colorbar: Optional[Colorbar] = None

    def _initialize_artists(self) -> None:
        ax = self.ax
        # Remove prior colorbar axes if we are re-initializing.
        if self._colorbar is not None:
            try:
                self._colorbar.remove()
            except Exception:
                pass
            self._colorbar = None

        trajectory = self.trajectory
        vmin, vmax = trajectory.color_limits()
        frame0 = trajectory.frames[0]
        with profile.span("viz.draw_frame_artists"):
            self._scatter, self._edges = self.renderer.draw_frame(
                ax,
                trajectory.geometry,
                frame0,
                vmin=vmin,
                vmax=vmax,
                clear=True,
            )

        if self.show_colorbar and self._scatter is not None:
            with profile.span("viz.colorbar"):
                self._colorbar = ax.figure.colorbar(
                    self._scatter, ax=ax, fraction=0.03, pad=0.015
                )
                self._colorbar.set_label(f"<{trajectory.value_name}>")

        self.update(0)

    def update(self, frame_index: int) -> None:
        """Update node colours for the selected frame.

        Edge artists are retained as a separate collection so future bond
        colouring can update ``self._edges`` without redrawing nodes.
        """

        if self._scatter is None:
            self._initialize_artists()
            return

        trajectory = self.trajectory
        n_frames = len(trajectory.frames)
        if frame_index < 0 or frame_index >= n_frames:
            raise IndexError(
                f"frame_index {frame_index} out of range for {n_frames} frames."
            )

        frame = trajectory.frames[frame_index]
        self._scatter.set_array(frame.values)
        # Intentionally no title here — dashboard owns the metadata panel.
