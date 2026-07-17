"""1D spin-chain view: linear layout upstream, generic graph renderer here."""

from __future__ import annotations

from typing import Optional

from atlas.visualization.hamiltonian_sim.colors import DEFAULT_DIVERGING_CMAP
from atlas.visualization.hamiltonian_sim.renderer import LatticeGraphRenderer
from atlas.visualization.hamiltonian_sim.views.graph import GraphLatticeView


class Chain1DView(GraphLatticeView):
    """Dashboard view tuned for a 1D spin chain.

    Responsibility:
        Provide chain-friendly visual defaults (large nodes, vivid edges).
        Geometry still comes from the trajectory (typically ``chain_1d``);
        this class does not invent lattice structure from physics.

    Notes:
        Rendering is delegated entirely to ``LatticeGraphRenderer`` via
        ``GraphLatticeView``. 2D / arbitrary graphs should use
        ``GraphLatticeView`` (or another specialized view) instead.
    """

    def __init__(
        self,
        *,
        title: Optional[str] = None,
        renderer: Optional[LatticeGraphRenderer] = None,
        cmap: str = DEFAULT_DIVERGING_CMAP,
        node_size: float = 1100.0,
        edge_color: str = "#1f77b4",
        edge_width: float = 4.0,
        show_labels: bool = True,
        show_colorbar: bool = True,
    ) -> None:
        super().__init__(
            title=title,
            renderer=renderer
            or LatticeGraphRenderer(
                cmap=cmap,
                node_size=node_size,
                edge_color=edge_color,
                edge_width=edge_width,
                show_labels=show_labels,
                label_fontsize=12.0,
                label_offset=(0.0, 22.0),
                axis_pad_fraction=0.12,
                axis_pad_min=0.45,
            ),
            cmap=cmap,
            node_size=node_size,
            edge_color=edge_color,
            edge_width=edge_width,
            show_labels=show_labels,
            show_colorbar=show_colorbar,
        )
