"""Display-only lattice layout helpers.

These functions produce ``LatticeGeometry`` for rendering. They encode no
Hamiltonian, boundary conditions, or physics — only node placement and
optional edges for drawing.
"""

from __future__ import annotations

import numpy as np

from atlas.visualization.hamiltonian_sim.lattice import LatticeGeometry


def chain_1d(
    num_sites: int,
    spacing: float = 1.8,
    *,
    with_edges: bool = True,
) -> LatticeGeometry:
    """Return a horizontal 1D chain layout.

    Purpose:
        Default geometry for spin-chain visualizations when no external
        layout is supplied.

    Inputs:
        num_sites: Number of nodes (must be >= 1).
        spacing: Distance between neighboring sites along x.
        with_edges: When True, connect consecutive sites for drawing.

    Outputs:
        ``LatticeGeometry`` with shape ``(num_sites, 2)`` positions in the
        plane (y = 0) so 1D and 2D renderers share the same path.
    """

    if num_sites < 1:
        raise ValueError("num_sites must be at least 1.")

    x = spacing * np.arange(num_sites, dtype=float)
    positions = np.column_stack([x, np.zeros(num_sites, dtype=float)])
    edges = None
    if with_edges and num_sites >= 2:
        indices = np.arange(num_sites - 1)
        edges = np.column_stack([indices, indices + 1])
    labels = tuple(str(i) for i in range(num_sites))
    return LatticeGeometry(positions=positions, edges=edges, labels=labels)


def grid_2d(
    shape: tuple[int, int],
    spacing: float = 1.0,
    *,
    with_edges: bool = True,
) -> LatticeGeometry:
    """Return a rectangular 2D grid layout (row-major site indexing).

    Purpose:
        Forward-compatible layout for future 2D lattice dashboard views.

    Inputs:
        shape: ``(n_rows, n_cols)``.
        spacing: Lattice constant for both axes.
        with_edges: When True, connect nearest neighbors (open boundaries).

    Outputs:
        ``LatticeGeometry`` with ``n_rows * n_cols`` sites.
    """

    n_rows, n_cols = int(shape[0]), int(shape[1])
    if n_rows < 1 or n_cols < 1:
        raise ValueError("shape entries must be at least 1.")

    positions = []
    for row in range(n_rows):
        for col in range(n_cols):
            positions.append((spacing * col, -spacing * row))
    positions_arr = np.asarray(positions, dtype=float)

    edges = None
    if with_edges:
        edge_list = []
        for row in range(n_rows):
            for col in range(n_cols):
                index = row * n_cols + col
                if col + 1 < n_cols:
                    edge_list.append((index, index + 1))
                if row + 1 < n_rows:
                    edge_list.append((index, index + n_cols))
        if edge_list:
            edges = np.asarray(edge_list, dtype=int)

    labels = tuple(str(i) for i in range(positions_arr.shape[0]))
    return LatticeGeometry(positions=positions_arr, edges=edges, labels=labels)
