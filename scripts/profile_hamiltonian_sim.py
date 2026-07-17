#!/usr/bin/env python3
"""Profile a Hamiltonian-simulation + dashboard pipeline.

Enable ``ATLAS_PROFILE=1`` before importing Atlas. Defaults to a *minimal*
sweep for fast analysis; pass ``--full`` for the 10-qubit / 13-time workload.

Does not change simulation behavior; instrumentation is gated by the env var.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Must be set before Atlas imports so profiling spans are active.
os.environ["ATLAS_PROFILE"] = "1"

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")

from atlas import profiling as profile
from atlas.config import (
    AlgorithmConfig,
    AnalysisConfig,
    AtlasConfig,
    BackendConfig,
    ExperimentMetaConfig,
    InitialStateConfig,
    OutputConfig,
    SystemConfig,
)
from atlas.experiments.factory import run_experiment
from atlas.experiments.results import SimBenchmarkResult, SimValidationResult
from atlas.visualization.hamiltonian_sim import (
    LatticeDashboard,
    trajectory_from_sim_benchmark,
)


FULL_TIMES = [0.1, 0.25, 0.5, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
MINIMAL_TIMES = [0.1, 0.5, 1.0]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="10 qubits, 13 times, 20 Strang steps (slow).",
    )
    parser.add_argument("--n", type=int, default=None, help="Number of qubits")
    parser.add_argument(
        "--times",
        default=None,
        help="Comma-separated evolution times (overrides --full/--minimal defaults)",
    )
    parser.add_argument("--steps", type=int, default=None, help="Trotter steps")
    parser.add_argument(
        "--method",
        default="strang",
        choices=("lie", "strang"),
    )
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Skip visualization setup (simulation profile only)",
    )
    return parser.parse_args(argv)


def _parse_times(raw: str) -> list[float]:
    return [float(p.strip()) for p in raw.split(",") if p.strip()]


def build_profile_config(
    *,
    num_qubits: int,
    times: list[float],
    steps: int,
    method: str,
) -> AtlasConfig:
    return AtlasConfig(
        experiment=ExperimentMetaConfig(type="sweep", name="profile_ham_sim"),
        system=SystemConfig(
            name="tfim",
            parameters={"num_qubits": num_qubits, "J": 1.0, "h": 0.5},
            sweep={"parameter": "evolution_time", "values": list(times)},
        ),
        algorithm=AlgorithmConfig(
            name="hamiltonian_sim",
            parameters={
                "evolution_method": method,
                "num_trotter_steps": steps,
            },
        ),
        backend=BackendConfig(name="statevector_evolver"),
        initial_state=InitialStateConfig(
            name="computational",
            parameters={"bitstring": "0" * num_qubits},
        ),
        analysis=AnalysisConfig(
            observables="tfim_default",
            site_observables="local_z",
            fidelity=True,
        ),
        output=OutputConfig(csv=False, plots=False),
    )


def pick_series(result, method: str) -> SimBenchmarkResult:
    if isinstance(result, SimBenchmarkResult):
        return result
    if isinstance(result, SimValidationResult):
        for series in result.series:
            if series.method_name == method:
                return series
        return result.series[0]
    raise TypeError(type(result))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.full:
        num_qubits = args.n if args.n is not None else 10
        times = _parse_times(args.times) if args.times else list(FULL_TIMES)
        steps = args.steps if args.steps is not None else 20
    else:
        num_qubits = args.n if args.n is not None else 4
        times = _parse_times(args.times) if args.times else list(MINIMAL_TIMES)
        steps = args.steps if args.steps is not None else 10

    profile.PROFILER.reset()
    profile.PROFILER.mark_start()

    with profile.span("config.construct"):
        config = build_profile_config(
            num_qubits=num_qubits,
            times=times,
            steps=steps,
            method=args.method,
        )

    print(
        f"Profiling: n={num_qubits}, {args.method}, steps={steps}, "
        f"times={times}, site_observables=local_z",
        flush=True,
    )

    with profile.span("experiment.run_total"):
        run = run_experiment(config)

    series = pick_series(run.result, args.method)
    out = Path("/tmp/atlas_profile_dashboard.png")

    if not args.no_dashboard:
        with profile.span("viz.trajectory_adapt"):
            trajectory = trajectory_from_sim_benchmark(
                series, site_observable="z", source="sim"
            )

        with profile.span("viz.dashboard_total"):
            dashboard = LatticeDashboard(
                trajectory,
                window_title="Atlas — Profile Run",
                show_playback=True,
            )
            with profile.span("viz.save_snapshot"):
                dashboard.save_snapshot(str(out), frame_index=-1)
            dashboard.close()

    profile.PROFILER.mark_end()
    print(profile.PROFILER.report())

    counts = profile.PROFILER.counts
    n_iter = counts.get("sweep.iteration", 0)
    print("---- Repeated work detected ----")
    checks = [
        ("Hamiltonian object", "build.hamiltonian"),
        ("Scalar observables", "build.observables"),
        ("Site observables", "build.site_observables"),
        ("Initial state object", "build.initial_state"),
        ("Evolution method object", "build.evolution_method"),
        ("Dense H matrix (exact path)", "exact.to_matrix"),
        ("Matrix expm (exact path)", "exact.expm_apply"),
        ("Trotter circuit build", "sim.circuit_build"),
        ("Statevector evolver", "sim.evolver_evolve"),
        ("Dashboard figure", "viz.figure_create"),
        ("Dashboard artists", "viz.artists_init"),
        ("Colorbar", "viz.colorbar"),
        ("Slider", "viz.slider"),
    ]
    for label, key in checks:
        c = counts.get(key, 0)
        rebuilt = n_iter > 1 and c >= n_iter
        once = c == 1
        status = (
            f"REBUILT every iteration ({c}×)"
            if rebuilt
            else (f"once ({c}×)" if once else f"{c}×")
        )
        print(f"  {label}: {status}")
    print()
    print("---- Potential optimization opportunities (observation only) ----")
    print("  See aggregated spans above; do not optimize until reviewing this report.")
    if not args.no_dashboard:
        print(f"Snapshot written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
