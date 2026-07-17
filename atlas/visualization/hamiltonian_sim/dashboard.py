"""Matplotlib dashboard for synchronized lattice trajectory views.

Presentation only: consumes ``LatticeTrajectory`` data, hosts pluggable
``LatticeView`` panels, metadata, playback controls, and a shared time
slider. Never evaluates observables or inspects Hamiltonians / evolution
methods.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.widgets import Button, Slider

from atlas.visualization.hamiltonian_sim.lattice import LatticeTrajectory
from atlas.visualization.hamiltonian_sim.views.base import LatticeView
from atlas.visualization.hamiltonian_sim.views.chain_1d import Chain1DView
from atlas import profiling as profile

DEFAULT_WINDOW_TITLE = "Atlas — Hamiltonian Simulation Dashboard"


def _format_method(name: Optional[str]) -> str:
    if not name:
        return "—"
    known = {"lie": "Lie", "strang": "Strang"}
    key = str(name).lower()
    if key in known:
        return known[key]
    return str(name).replace("_", " ").title()


def _format_source(source: Optional[str]) -> str:
    if not source:
        return "—"
    known = {"sim": "Simulated", "exact": "Exact"}
    return known.get(str(source).lower(), str(source).title())


class LatticeDashboard:
    """Reusable dashboard that synchronizes one or more lattice views in time.

    Responsibility:
        Own the figure, metadata panel, playback controls, optional time
        slider, and frame index. Delegate lattice drawing to registered
        ``LatticeView`` instances.

    Usage:
        Build a trajectory via adapters, construct the dashboard with one or
        more views, then ``show()`` interactively or ``save_snapshot()`` under
        a headless Agg backend.
    """

    def __init__(
        self,
        trajectory: LatticeTrajectory,
        views: Optional[Sequence[LatticeView]] = None,
        *,
        figsize: Optional[tuple[float, float]] = None,
        show_slider: bool = True,
        show_playback: bool = True,
        show_metadata: bool = True,
        initial_frame: int = 0,
        window_title: str = DEFAULT_WINDOW_TITLE,
        playback_interval_ms: int = 500,
        loop_playback: bool = True,
    ) -> None:
        self.trajectory = trajectory
        self.views: list[LatticeView] = list(views) if views else [Chain1DView()]
        if not self.views:
            raise ValueError("LatticeDashboard requires at least one view.")

        self.window_title = window_title
        self.playback_interval_ms = int(playback_interval_ms)
        self.loop_playback = bool(loop_playback)

        n_sites = trajectory.geometry.num_sites
        n_views = len(self.views)
        if figsize is None:
            width = max(11.0, 1.6 * n_sites + 3.5)
            height = max(5.2, 3.6 * n_views + 1.6)
            figsize = (width, height)

        self._show_slider = show_slider and len(trajectory.frames) > 1
        self._show_playback = show_playback and self._show_slider
        self._show_metadata = show_metadata

        with profile.span("viz.dashboard_init"):
            with profile.span("viz.figure_create"):
                self.fig: Figure = plt.figure(figsize=figsize)
                self._set_window_title(window_title)

            self._slider: Optional[Slider] = None
            self._play_button: Optional[Button] = None
            self._pause_button: Optional[Button] = None
            self._meta_ax = None
            self._meta_text = None
            self._timer = None
            self._playing = False
            self._frame_index = 0

            with profile.span("viz.layout_axes"):
                self._layout_axes(n_views)

            with profile.span("viz.artists_init"):
                for index, view in enumerate(self.views):
                    view.attach(self._view_axes[index])
                    view.set_trajectory(trajectory)

            if self._show_metadata:
                with profile.span("viz.metadata_panel"):
                    self._build_metadata_panel()
            if self._show_slider:
                with profile.span("viz.slider"):
                    self._build_slider()
            if self._show_playback:
                with profile.span("viz.playback"):
                    self._build_playback_controls()

            with profile.span("viz.set_frame"):
                self.set_frame(initial_frame)

    def _set_window_title(self, title: str) -> None:
        manager = getattr(self.fig.canvas, "manager", None)
        if manager is not None and hasattr(manager, "set_window_title"):
            manager.set_window_title(title)

    def _layout_axes(self, n_views: int) -> None:
        """Allocate most of the figure to lattice views; reserve control strips."""

        bottom = 0.18 if self._show_slider else 0.06
        right = 0.78 if self._show_metadata else 0.96
        self._view_axes = []
        height = (0.92 - bottom) / n_views
        for index in range(n_views):
            y0 = 0.92 - (index + 1) * height + 0.02
            ax = self.fig.add_axes([0.04, y0, right - 0.06, height - 0.04])
            self._view_axes.append(ax)

        if self._show_metadata:
            self._meta_ax = self.fig.add_axes([0.80, bottom + 0.02, 0.18, 0.90 - bottom])
            self._meta_ax.axis("off")

    @property
    def frame_index(self) -> int:
        """Return the currently displayed frame index."""

        return self._frame_index

    @property
    def num_frames(self) -> int:
        """Return the number of frames in the bound trajectory."""

        return len(self.trajectory.frames)

    @property
    def is_playing(self) -> bool:
        """Return whether automatic playback is active."""

        return self._playing

    def _build_metadata_panel(self) -> None:
        assert self._meta_ax is not None
        self._meta_text = self._meta_ax.text(
            0.0,
            1.0,
            "",
            transform=self._meta_ax.transAxes,
            va="top",
            ha="left",
            family="monospace",
            fontsize=11,
            linespacing=1.6,
        )

    def _metadata_block(self, frame_index: int) -> str:
        frame = self.trajectory.frames[frame_index]
        observable = self.trajectory.value_name
        method = _format_method(frame.metadata.get("method_name"))
        state = _format_source(frame.metadata.get("source"))
        if frame.time is not None and np.isfinite(frame.time):
            time_str = f"{float(frame.time):.2f}"
        else:
            time_str = f"frame {frame_index}"

        return (
            f"Observable: <{observable}>\n"
            f"Method:     {method}\n"
            f"State:      {state}\n"
            f"Time:       {time_str}"
        )

    def _refresh_metadata(self, frame_index: int) -> None:
        if self._meta_text is None:
            return
        self._meta_text.set_text(self._metadata_block(frame_index))

    def _build_slider(self) -> None:
        # Discrete frame index slider avoids continuous-value wrap glitches
        # (e.g. looping from t=1 back to t=0 briefly showing a bogus value).
        # Physical time is shown in the slider text and metadata panel.
        ax_slider = self.fig.add_axes([0.22, 0.07, 0.52, 0.045])
        self._slider = Slider(
            ax=ax_slider,
            label="time",
            valmin=0,
            valmax=max(self.num_frames - 1, 0),
            valinit=0,
            valstep=1,
        )
        self._set_slider_time_text(0)

        def _on_change(value: float) -> None:
            index = int(round(value))
            index = min(max(index, 0), self.num_frames - 1)
            # Manual scrubbing pauses playback so Play/slider do not fight.
            if self._playing:
                self.pause()
            self._apply_frame(index, sync_slider=False)
            self._set_slider_time_text(index)

        self._slider.on_changed(_on_change)

    def _set_slider_time_text(self, frame_index: int) -> None:
        if self._slider is None:
            return
        times = self.trajectory.times
        if frame_index < len(times) and np.isfinite(times[frame_index]):
            self._slider.valtext.set_text(f"{float(times[frame_index]):.2f}")
        else:
            self._slider.valtext.set_text(str(frame_index))

    def _build_playback_controls(self) -> None:
        ax_play = self.fig.add_axes([0.06, 0.065, 0.06, 0.05])
        ax_pause = self.fig.add_axes([0.13, 0.065, 0.06, 0.05])
        self._play_button = Button(ax_play, "Play")
        self._pause_button = Button(ax_pause, "Pause")
        self._play_button.on_clicked(lambda _event: self.play())
        self._pause_button.on_clicked(lambda _event: self.pause())

        if hasattr(self.fig.canvas, "new_timer"):
            self._timer = self.fig.canvas.new_timer(interval=self.playback_interval_ms)
            self._timer.add_callback(self._on_timer_tick)

    def _on_timer_tick(self) -> None:
        if not self._playing:
            return
        nxt = self._frame_index + 1
        if nxt >= self.num_frames:
            if self.loop_playback:
                nxt = 0
            else:
                self.pause()
                return
        self.set_frame(nxt)

    def play(self) -> None:
        """Start automatic playback (advances the shared time slider)."""

        if self.num_frames <= 1:
            return
        self._playing = True
        if self._timer is not None:
            self._timer.start()

    def pause(self) -> None:
        """Pause automatic playback; slider remains interactive."""

        self._playing = False
        if self._timer is not None:
            self._timer.stop()

    def _sync_slider(self, frame_index: int) -> None:
        if self._slider is None:
            return
        self._slider.eventson = False
        self._slider.set_val(frame_index)
        self._slider.eventson = True
        self._set_slider_time_text(frame_index)

    def _apply_frame(self, frame_index: int, *, sync_slider: bool) -> None:
        if frame_index < 0 or frame_index >= self.num_frames:
            raise IndexError(
                f"frame_index {frame_index} out of range for {self.num_frames} frames."
            )
        self._frame_index = frame_index
        for view in self.views:
            view.update(frame_index)
        self._refresh_metadata(frame_index)
        if sync_slider:
            self._sync_slider(frame_index)
        with profile.span("viz.canvas_draw"):
            self.fig.canvas.draw_idle()

    def set_frame(self, frame_index: int) -> None:
        """Programmatically select a frame and refresh all views.

        Works under interactive and headless (Agg) backends alike.
        Negative indices count from the end (``-1`` is the last frame).
        """

        if frame_index < 0:
            frame_index = self.num_frames + frame_index
        self._apply_frame(frame_index, sync_slider=True)

    def save_snapshot(
        self,
        path: str,
        *,
        frame_index: Optional[int] = None,
        dpi: int = 150,
    ) -> str:
        """Write a PNG of the current (or specified) frame and return ``path``."""

        was_playing = self._playing
        self.pause()
        if frame_index is not None:
            self.set_frame(frame_index)
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self.fig.savefig(path, dpi=dpi)
        if was_playing:
            self.play()
        return path

    def show(self) -> None:
        """Display the dashboard via ``plt.show()`` (interactive backends)."""

        plt.show()
        self.pause()

    def close(self) -> None:
        """Close the underlying matplotlib figure and stop playback."""

        self.pause()
        plt.close(self.fig)


def build_default_dashboard(
    trajectory: LatticeTrajectory,
    **kwargs,
) -> LatticeDashboard:
    """Return a dashboard with a single ``Chain1DView`` panel."""

    kwargs.setdefault("window_title", "Atlas — Spin Chain Visualization")
    return LatticeDashboard(trajectory, views=[Chain1DView()], **kwargs)
