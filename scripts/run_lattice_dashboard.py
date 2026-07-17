#!/usr/bin/env python3
"""Run an interactive lattice dashboard for Hamiltonian simulation.

Thin entry point: simulate a short time sweep, then open a Matplotlib
dashboard with a time slider. No physics is computed inside the viz layer.

Examples:
    python scripts/run_lattice_dashboard.py
    python scripts/run_lattice_dashboard.py --n 6 --h 1.0 --times 0,0.25,0.5,1,2
    python scripts/run_lattice_dashboard.py --observable x --source exact
    python scripts/run_lattice_dashboard.py --config configs/tfim_hamiltonian_sim_time_sweep.yaml
    python scripts/run_lattice_dashboard.py --save frame.png --no-show
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_OBS_TO_PRESET = {"z": "local_z", "x": "local_x", "y": "local_y"}
_GUI_BACKENDS = ("TkAgg", "QtAgg", "Qt5Agg", "MacOSX", "GTK4Agg", "GTK3Agg")
_NON_INTERACTIVE = frozenset({"agg", "pdf", "svg", "ps", "template"})


def _parse_times(raw: str) -> list[float]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("Provide at least one time value.")
    try:
        return [float(p) for p in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid --times value: {raw}") from exc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate a lattice trajectory and open the Atlas dashboard.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        help="Optional YAML experiment config. When set, physics knobs below are ignored.",
    )

    physics = parser.add_argument_group("simulation (ignored if --config is set)")
    physics.add_argument("--n", "--num-qubits", dest="num_qubits", type=int, default=4)
    physics.add_argument("--J", dest="J", type=float, default=1.0)
    physics.add_argument("--h", dest="h", type=float, default=0.5)
    physics.add_argument(
        "--times",
        type=_parse_times,
        default=_parse_times("0,0.25,0.5,1.0"),
        help="Comma-separated evolution times",
    )
    physics.add_argument("--steps", type=int, default=20, help="Trotter steps")
    physics.add_argument(
        "--method",
        default="lie",
        choices=("lie", "strang"),
        help="Evolution method (also selects series when config has several)",
    )
    physics.add_argument(
        "--bitstring",
        default=None,
        help="Initial computational bitstring (default: zeros of length n)",
    )

    viz = parser.add_argument_group("dashboard")
    viz.add_argument(
        "--observable",
        default="z",
        choices=sorted(_OBS_TO_PRESET),
        help="Per-site observable to color nodes",
    )
    viz.add_argument(
        "--source",
        default="sim",
        choices=("sim", "exact"),
        help="Use simulated or exact site values",
    )
    viz.add_argument(
        "--save",
        metavar="PATH",
        help="Write a PNG snapshot (last frame by default)",
    )
    viz.add_argument(
        "--frame",
        type=int,
        default=-1,
        help="Frame index for --save (negative counts from the end)",
    )
    viz.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open the interactive window",
    )
    return parser.parse_args(argv)


def _configure_interactive_backend() -> str:
    """Select a GUI matplotlib backend before pyplot is imported."""

    import matplotlib

    current = matplotlib.get_backend()
    if current.lower() not in _NON_INTERACTIVE:
        return current

    errors: list[str] = []
    for candidate in _GUI_BACKENDS:
        try:
            matplotlib.use(candidate, force=True)
            return matplotlib.get_backend()
        except Exception as exc:  # pragma: no cover - environment dependent
            errors.append(f"{candidate}: {exc}")

    detail = "; ".join(errors) if errors else "no candidates tried"
    raise RuntimeError(
        "No interactive Matplotlib backend available (stuck on Agg). "
        "Install a GUI toolkit (e.g. `pip install PyQt6` or ensure Tk is available). "
        f"Tried: {detail}"
    )


def _default_bitstring(num_qubits: int, bitstring: str | None) -> str:
    if bitstring is not None:
        if len(bitstring) != num_qubits:
            raise ValueError(
                f"--bitstring length ({len(bitstring)}) must match --n ({num_qubits})."
            )
        return bitstring
    return "0" * num_qubits


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Backend must be chosen before any Atlas viz import pulls in pyplot.
    if not args.no_show:
        backend = _configure_interactive_backend()
        print(f"Matplotlib backend: {backend}")

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
    from atlas.io.yaml_config import load_config
    from atlas.visualization.hamiltonian_sim import (
        LatticeDashboard,
        available_site_observables,
        trajectory_from_sim_benchmark,
    )

    def build_config_from_args() -> AtlasConfig:
        bitstring = _default_bitstring(args.num_qubits, args.bitstring)
        return AtlasConfig(
            experiment=ExperimentMetaConfig(type="sweep", name="lattice_dashboard"),
            system=SystemConfig(
                name="tfim",
                parameters={
                    "num_qubits": args.num_qubits,
                    "J": args.J,
                    "h": args.h,
                },
                sweep={"parameter": "evolution_time", "values": list(args.times)},
            ),
            algorithm=AlgorithmConfig(
                name="hamiltonian_sim",
                parameters={
                    "evolution_method": args.method,
                    "num_trotter_steps": args.steps,
                },
            ),
            backend=BackendConfig(name="statevector_evolver"),
            initial_state=InitialStateConfig(
                name="computational",
                parameters={"bitstring": bitstring},
            ),
            analysis=AnalysisConfig(
                observables="tfim_default",
                site_observables=_OBS_TO_PRESET[args.observable],
                fidelity=True,
            ),
            output=OutputConfig(csv=False, plots=False),
        )

    def ensure_site_observables(config: AtlasConfig) -> AtlasConfig:
        preset = _OBS_TO_PRESET[args.observable]
        if config.analysis.site_observables == preset:
            return config
        return AtlasConfig(
            experiment=config.experiment,
            system=config.system,
            algorithm=config.algorithm,
            ansatz=config.ansatz,
            optimizer=config.optimizer,
            backend=config.backend,
            hardware=config.hardware,
            analysis=AnalysisConfig(
                observables=config.analysis.observables,
                site_observables=preset,
                fidelity=config.analysis.fidelity,
            ),
            output=config.output,
            initial_state=config.initial_state,
        )

    def pick_series(result) -> SimBenchmarkResult:
        if isinstance(result, SimBenchmarkResult):
            return result
        if isinstance(result, SimValidationResult):
            for series in result.series:
                if series.method_name == args.method:
                    return series
            if result.series:
                print(
                    f"Method '{args.method}' not found; "
                    f"using '{result.series[0].method_name}'.",
                    file=sys.stderr,
                )
                return result.series[0]
            raise ValueError("Validation result has no method series.")
        raise TypeError(
            f"Expected a dynamics sweep result, got {type(result).__name__}. "
            "Use a hamiltonian_sim time-sweep config."
        )

    if args.config:
        config = ensure_site_observables(load_config(args.config))
    else:
        config = build_config_from_args()

    print("Running simulation…")
    run = run_experiment(config)
    series = pick_series(run.result)

    available = available_site_observables(series)
    if args.observable not in available:
        raise SystemExit(
            f"Site observable '{args.observable}' missing on results. "
            f"Available: {available or '(none — set analysis.site_observables)'}"
        )

    trajectory = trajectory_from_sim_benchmark(
        series,
        site_observable=args.observable,
        source=args.source,
    )
    dashboard = LatticeDashboard(
        trajectory,
        window_title="Atlas — Spin Chain Visualization",
    )

    if args.save:
        path = dashboard.save_snapshot(args.save, frame_index=args.frame)
        print(f"Saved snapshot: {path}")

    if args.no_show:
        dashboard.close()
        return 0

    import matplotlib

    if matplotlib.get_backend().lower() in _NON_INTERACTIVE:
        raise RuntimeError(
            f"Backend became non-interactive ({matplotlib.get_backend()}) after imports. "
            "Something forced Agg; interactive show() cannot open a window."
        )

    print(
        f"Dashboard: <{args.observable}_i> ({args.source}), "
        f"method={series.method_name}, frames={dashboard.num_frames}"
    )
    print("Close the window to exit.")
    dashboard.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
