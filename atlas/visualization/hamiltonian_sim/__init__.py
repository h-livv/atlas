"""Physics-agnostic lattice visualization for Hamiltonian simulation.

The visualization layer is a renderer: it consumes precomputed site values
and geometry, and never evaluates observables or inspects Hamiltonians.

Primary interface: ``LatticeDashboard`` with pluggable ``LatticeView`` panels
(time slider + playback when interactive; PNG snapshots under Agg/headless).
"""

from atlas.visualization.hamiltonian_sim.adapters import (
    available_site_observables,
    trajectory_from_sim_benchmark,
    trajectory_from_sim_point,
    trajectory_from_site_series,
)
from atlas.visualization.hamiltonian_sim.dashboard import (
    DEFAULT_WINDOW_TITLE,
    LatticeDashboard,
    build_default_dashboard,
)
from atlas.visualization.hamiltonian_sim.lattice import (
    LatticeFrame,
    LatticeGeometry,
    LatticeTrajectory,
)
from atlas.visualization.hamiltonian_sim.layouts import chain_1d, grid_2d
from atlas.visualization.hamiltonian_sim.renderer import (
    LatticeGraphRenderer,
    LatticeNodeRenderer,
)
from atlas.visualization.hamiltonian_sim.views import (
    Chain1DView,
    GraphLatticeView,
    LatticeView,
)

__all__ = [
    "Chain1DView",
    "DEFAULT_WINDOW_TITLE",
    "GraphLatticeView",
    "LatticeDashboard",
    "LatticeFrame",
    "LatticeGeometry",
    "LatticeGraphRenderer",
    "LatticeNodeRenderer",
    "LatticeTrajectory",
    "LatticeView",
    "available_site_observables",
    "build_default_dashboard",
    "chain_1d",
    "grid_2d",
    "trajectory_from_sim_benchmark",
    "trajectory_from_sim_point",
    "trajectory_from_site_series",
]
