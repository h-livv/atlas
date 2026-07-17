"""Optional high-resolution pipeline profiler.

Enable with environment variable ``ATLAS_PROFILE=1`` before importing Atlas
modules. When disabled, ``span`` is a no-op and incurs negligible overhead.

Remove by deleting this module and the ``profile.span(...)`` call sites.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from typing import Any, Iterator, Optional


def _env_enabled() -> bool:
    return os.environ.get("ATLAS_PROFILE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


class PipelineProfiler:
    """Accumulate named wall-clock spans via ``time.perf_counter()``."""

    def __init__(self) -> None:
        self.enabled = _env_enabled()
        self.totals: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)
        self.events: list[tuple[str, float, dict[str, Any]]] = []
        self._stack: list[str] = []
        self.t_start: Optional[float] = None
        self.t_end: Optional[float] = None

    def reset(self) -> None:
        self.totals.clear()
        self.counts.clear()
        self.events.clear()
        self._stack.clear()
        self.t_start = None
        self.t_end = None
        self.enabled = _env_enabled()

    def mark_start(self) -> None:
        if self.enabled:
            self.t_start = time.perf_counter()

    def mark_end(self) -> None:
        if self.enabled:
            self.t_end = time.perf_counter()

    @contextmanager
    def span(self, name: str, **meta: Any) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        self._stack.append(name)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self.totals[name] += dt
            self.counts[name] += 1
            self.events.append((name, dt, dict(meta)))
            self._stack.pop()

    def total_runtime(self) -> float:
        if self.t_start is None or self.t_end is None:
            return float(sum(self.totals.values()))
        return self.t_end - self.t_start

    def report(self) -> str:
        lines: list[str] = []
        lines.append("=" * 72)
        lines.append("Hamiltonian Simulation Profile")
        lines.append("=" * 72)
        lines.append("")
        lines.append("---- Aggregated spans (wall clock) ----")
        lines.append(f"{'span':<48} {'count':>6} {'total_s':>10} {'mean_s':>10}")
        lines.append("-" * 72)
        for name in sorted(self.totals, key=lambda k: -self.totals[k]):
            total = self.totals[name]
            count = self.counts[name]
            mean = total / count if count else 0.0
            lines.append(f"{name:<48} {count:>6d} {total:>10.4f} {mean:>10.4f}")
        lines.append("-" * 72)
        lines.append(f"{'TOTAL RUNTIME':<48} {'':>6} {self.total_runtime():>10.4f}")
        lines.append("")

        # Evolver-focused subsection
        evolver_keys = [k for k in self.totals if k.startswith("evolver.")]
        if evolver_keys:
            lines.append("---- sim.evolver_evolve internal breakdown ----")
            authentic = self.totals.get("evolver.statevector_from_circuit", 0.0)
            lines.append(
                f"Authentic Statevector(circuit) wall time: {authentic:.4f}s "
                f"({self.counts.get('evolver.statevector_from_circuit', 0)} call(s))"
            )
            diag_total = self.totals.get("evolver.diagnostic_decomposition", 0.0)
            if diag_total > 0:
                lines.append(
                    f"Diagnostic walk wall time (extra, profiling only): {diag_total:.4f}s"
                )
            mat = self.totals.get("evolver.diag.instruction_to_matrix", 0.0)
            wrap = self.totals.get("evolver.diag.operator_wrap", 0.0)
            evo = self.totals.get("evolver.diag.evolve_operator", 0.0)
            to_inst = self.totals.get("evolver.diag.to_instruction", 0.0)
            init0 = self.totals.get("evolver.diag.init_zero_statevector", 0.0)
            via_def = self.totals.get("evolver.diag.evolve_via_definition", 0.0)
            core = mat + wrap + evo + to_inst + init0 + via_def
            if core > 0:
                lines.append("")
                lines.append("Diagnostic path share of (mat+wrap+evo+to_inst+init+def):")
                for label, val in [
                    ("instruction_to_matrix (PauliEvolutionGate.to_matrix)", mat),
                    ("Operator(mat) wrap", wrap),
                    ("Statevector._evolve_operator", evo),
                    ("circuit.to_instruction", to_inst),
                    ("init |0…0⟩ Statevector", init0),
                    ("evolve via definition", via_def),
                ]:
                    pct = 100.0 * val / core
                    lines.append(f"  {label:<55} {val:8.4f}s  ({pct:5.1f}%)")
            # Structural summary from last summary event
            for name, dt, meta in reversed(self.events):
                if name == "evolver.diag.summary":
                    lines.append("")
                    lines.append(
                        f"Circuit: n={meta.get('num_qubits')}, "
                        f"size={meta.get('circuit_size')}, "
                        f"depth={meta.get('circuit_depth')}"
                    )
                    lines.append(f"Op histogram: {meta.get('op_counts')}")
                    lines.append(
                        f"Matrix-path ops: {meta.get('matrix_path_ops')}, "
                        f"definition-path ops: {meta.get('definition_path_ops')}"
                    )
                    break
            lines.append("")

        # Per-iteration blocks from sweep events.
        if any(name == "sweep.iteration" for name, _, _ in self.events):
            lines.append("---- Per evolution-time iteration ----")
            for name, dt, meta in self.events:
                if name != "sweep.iteration":
                    continue
                t = meta.get("evolution_time", "?")
                lines.append("")
                lines.append(f"Evolution time = {t}")
                children = [
                    (n, d, m)
                    for n, d, m in self.events
                    if m.get("evolution_time") == t and n != "sweep.iteration"
                ]
                for n, d, m in children:
                    lines.append(f"  {n:<44} {d:8.4f}s")
                lines.append(f"  {'iteration total':<44} {dt:8.4f}s")
            lines.append("")

        lines.append("---- Call counts (rebuild / reuse signal) ----")
        interesting = [
            "build.hamiltonian",
            "build.observables",
            "build.site_observables",
            "build.initial_state",
            "build.evolution_method",
            "exact.to_matrix",
            "exact.expm_apply",
            "sim.circuit_build",
            "sim.evolver_evolve",
            "evolver.statevector_from_circuit",
            "evolver.diag.instruction_to_matrix",
            "evolver.diag.evolve_operator",
            "obs.scalar_expectations",
            "obs.site_expectations",
            "viz.dashboard_init",
            "viz.figure_create",
            "viz.artists_init",
            "viz.colorbar",
            "viz.slider",
            "viz.playback",
            "viz.set_frame",
            "viz.canvas_draw",
        ]
        for name in interesting:
            if name in self.counts:
                lines.append(
                    f"  {name}: {self.counts[name]} call(s), {self.totals[name]:.4f}s total"
                )
        lines.append("")
        return "\n".join(lines)


PROFILER = PipelineProfiler()


def span(name: str, **meta: Any):
    """Public context manager; no-op when profiling is disabled."""

    return PROFILER.span(name, **meta)


def maybe_span(name: str, **meta: Any):
    """Alias kept for call-site clarity."""

    if not PROFILER.enabled:
        return nullcontext()
    return PROFILER.span(name, **meta)
