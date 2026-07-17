"""Abstract lattice view interface for the dynamics dashboard.

Views are pure presentation adapters: they draw precomputed lattice frames
onto a matplotlib axes and never compute physics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from matplotlib.axes import Axes

from atlas.visualization.hamiltonian_sim.lattice import LatticeTrajectory


class LatticeView(ABC):
    """Pluggable panel that renders one ``LatticeTrajectory``.

    Responsibility:
        Own drawing state for a single axes. The dashboard calls ``attach``,
        ``set_trajectory``, and ``update``; views must not reach into other
        views or into physics/experiment code.

    Extensibility:
        Add ``Grid2DView``, ``GraphView``, etc. by subclassing without changing
        ``LatticeDashboard``.
    """

    def __init__(self, *, title: Optional[str] = None) -> None:
        self._title_override = title
        self._ax: Optional[Axes] = None
        self._trajectory: Optional[LatticeTrajectory] = None

    @property
    def title(self) -> str:
        """Return the panel title used by the dashboard layout."""

        if self._title_override is not None:
            return self._title_override
        if self._trajectory is not None:
            return f"<{self._trajectory.value_name}>"
        return self.__class__.__name__

    @property
    def ax(self) -> Axes:
        """Return the attached axes (raises if ``attach`` was not called)."""

        if self._ax is None:
            raise RuntimeError("View has not been attached to an axes.")
        return self._ax

    @property
    def trajectory(self) -> LatticeTrajectory:
        """Return the bound trajectory (raises if unset)."""

        if self._trajectory is None:
            raise RuntimeError("View has no trajectory; call set_trajectory first.")
        return self._trajectory

    def attach(self, ax: Axes) -> None:
        """Bind this view to a matplotlib axes."""

        self._ax = ax

    def set_trajectory(self, trajectory: LatticeTrajectory) -> None:
        """Bind trajectory data and (re)initialize artists on the axes."""

        self._trajectory = trajectory
        if self._ax is not None:
            self._initialize_artists()

    @abstractmethod
    def _initialize_artists(self) -> None:
        """Create/reset artists for the current trajectory on ``self.ax``."""

    @abstractmethod
    def update(self, frame_index: int) -> None:
        """Synchronize artists to ``trajectory.frames[frame_index]``."""
