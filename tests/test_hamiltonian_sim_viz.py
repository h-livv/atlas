"""Focused tests for physics-agnostic lattice visualization dashboard."""

from __future__ import annotations

import os
import tempfile
import unittest

import matplotlib

matplotlib.use("Agg")
import numpy as np

from atlas.experiments.results import (
    SimBenchmarkResult,
    SimPointResult,
    SimulationResult,
)
from atlas.physics.observables import local_pauli_observables
from atlas.visualization.hamiltonian_sim import (
    Chain1DView,
    GraphLatticeView,
    LatticeDashboard,
    LatticeGraphRenderer,
    chain_1d,
    grid_2d,
    trajectory_from_sim_benchmark,
    trajectory_from_site_series,
)
from atlas.visualization.hamiltonian_sim.views.base import LatticeView


def _dummy_point(t: float, site_z: np.ndarray) -> SimPointResult:
    n = len(site_z)
    return SimPointResult(
        system_parameters={"num_qubits": n},
        num_qubits=n,
        evolution_time=t,
        exact_state=np.zeros(2**n, dtype=complex),
        sim_result=SimulationResult(
            statevector=np.zeros(2**n, dtype=complex),
            num_qubits=n,
            evolution_time=t,
            method_name="lie",
            num_trotter_steps=10,
            circuit_depth=1,
        ),
        fidelity=1.0,
        observables={},
        exact_observables={},
        observable_errors={},
        site_observables={"z": np.asarray(site_z, dtype=float)},
        exact_site_observables={"z": np.asarray(site_z, dtype=float)},
    )


class LatticeLayoutTests(unittest.TestCase):
    def test_chain_1d_geometry(self):
        geo = chain_1d(4)
        self.assertEqual(geo.num_sites, 4)
        self.assertEqual(geo.dim, 2)
        self.assertEqual(len(geo.edges), 3)
        # Wider default spacing keeps nodes visually separated.
        self.assertAlmostEqual(float(geo.positions[1, 0] - geo.positions[0, 0]), 1.8)

    def test_grid_2d_geometry(self):
        geo = grid_2d((2, 3))
        self.assertEqual(geo.num_sites, 6)
        self.assertIsNotNone(geo.edges)


class LocalPauliTests(unittest.TestCase):
    def test_local_z_names(self):
        specs = local_pauli_observables(3, "Z")
        self.assertEqual([s.name for s in specs], ["z_0", "z_1", "z_2"])


class AdapterTests(unittest.TestCase):
    def test_trajectory_from_series(self):
        traj = trajectory_from_site_series(
            times=[0.0, 1.0],
            values=np.array([[1.0, -1.0], [0.5, -0.5]]),
            value_name="z",
        )
        self.assertEqual(len(traj.frames), 2)
        self.assertEqual(traj.geometry.num_sites, 2)
        vmin, vmax = traj.color_limits()
        self.assertLess(vmin, 0)
        self.assertGreater(vmax, 0)

    def test_trajectory_from_benchmark(self):
        result = SimBenchmarkResult(
            sweep_parameter="evolution_time",
            points=[
                _dummy_point(0.0, np.array([1.0, -1.0, 0.0])),
                _dummy_point(0.5, np.array([0.5, -0.5, 0.2])),
            ],
        )
        traj = trajectory_from_sim_benchmark(result, site_observable="z")
        self.assertEqual(traj.values.shape, (2, 3))
        self.assertEqual(list(traj.times), [0.0, 0.5])


class RendererTests(unittest.TestCase):
    def test_edges_and_nodes_are_independent(self):
        import matplotlib.pyplot as plt

        geo = chain_1d(3)
        renderer = LatticeGraphRenderer()
        fig, ax = plt.subplots()
        edges = renderer.draw_edges(ax, geo)
        nodes = renderer.draw_nodes(
            ax, geo, np.array([1.0, 0.0, -1.0]), vmin=-1.0, vmax=1.0
        )
        self.assertIsNotNone(edges)
        self.assertEqual(len(edges.get_segments()), 2)
        self.assertEqual(nodes.get_offsets().shape[0], 3)
        # Future bond colouring path accepts edge_values without touching nodes.
        ax.clear()
        colored = renderer.draw_edges(
            ax, geo, edge_values=np.array([-0.5, 0.5]), edge_vmin=-1.0, edge_vmax=1.0
        )
        self.assertIsNotNone(colored)
        plt.close(fig)


class DashboardSyncTests(unittest.TestCase):
    def test_set_frame_updates_all_views(self):
        traj = trajectory_from_site_series(
            times=[0.0, 0.5, 1.0],
            values=np.array(
                [
                    [1.0, -1.0],
                    [0.5, -0.5],
                    [0.0, 0.0],
                ]
            ),
            value_name="z",
        )
        updated: list[int] = []

        class RecordingView(LatticeView):
            def _initialize_artists(self) -> None:
                self.ax.clear()
                self.ax.text(0.5, 0.5, "init", ha="center")

            def update(self, frame_index: int) -> None:
                updated.append(frame_index)
                self.ax.set_title(f"frame={frame_index}")

        dashboard = LatticeDashboard(
            traj,
            views=[RecordingView(title="A"), RecordingView(title="B")],
            show_slider=False,
            show_playback=False,
            show_metadata=False,
            initial_frame=0,
        )
        # Construction calls set_frame(0) once per view.
        self.assertEqual(updated.count(0), 2)
        dashboard.set_frame(2)
        self.assertEqual(updated[-2:], [2, 2])
        self.assertEqual(dashboard.frame_index, 2)
        dashboard.close()

    def test_metadata_and_playback_controls(self):
        traj = trajectory_from_site_series(
            times=[0.0, 0.5, 1.0],
            values=np.array([[1.0, -1.0], [0.0, 0.0], [-1.0, 1.0]]),
            value_name="z",
            metadata={"method_name": "strang", "source": "sim"},
        )
        dashboard = LatticeDashboard(
            traj,
            views=[Chain1DView()],
            window_title="Atlas — Spin Chain Visualization",
        )
        block = dashboard._metadata_block(0)
        self.assertIn("Observable: <z>", block)
        self.assertIn("Method:     Strang", block)
        self.assertIn("State:      Simulated", block)
        self.assertIn("Time:       0.00", block)
        self.assertIsNotNone(dashboard._slider)
        self.assertIsNotNone(dashboard._play_button)
        self.assertIsNotNone(dashboard._pause_button)

        dashboard.play()
        self.assertTrue(dashboard.is_playing)
        dashboard._on_timer_tick()
        self.assertEqual(dashboard.frame_index, 1)
        dashboard.pause()
        self.assertFalse(dashboard.is_playing)
        dashboard.close()

    def test_graph_view_accepts_grid_geometry(self):
        geo = grid_2d((2, 2))
        traj = trajectory_from_site_series(
            times=[0.0, 1.0],
            values=np.array([[0.2, -0.3, 0.1, -0.1], [-0.1, 0.4, -0.2, 0.3]]),
            value_name="z",
            geometry=geo,
        )
        dashboard = LatticeDashboard(
            traj,
            views=[GraphLatticeView()],
            show_playback=False,
        )
        dashboard.set_frame(1)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "grid.png")
            dashboard.save_snapshot(path)
            self.assertTrue(os.path.isfile(path))
        dashboard.close()

    def test_chain_view_snapshot(self):
        traj = trajectory_from_site_series(
            times=[0.0, 1.0],
            values=np.array([[0.2, -0.3, 0.1], [-0.1, 0.4, -0.2]]),
            value_name="z",
        )
        dashboard = LatticeDashboard(
            traj,
            views=[Chain1DView()],
            show_slider=True,
            initial_frame=0,
        )
        dashboard.set_frame(1)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "dashboard.png")
            dashboard.save_snapshot(path)
            self.assertTrue(os.path.isfile(path))
            self.assertGreater(os.path.getsize(path), 0)
        dashboard.close()


if __name__ == "__main__":
    unittest.main()
