"""CSV loading and writing for TFIM benchmark data.

Isolates serialization from computation and plotting. `write_tfim_benchmark`
reproduces the exact column set written by the legacy
`vqe_legacy/tfim_vis.py` script's final `df`, sourced from
`TFIMBenchmarkResult` accessors, with `NaN` in hardware columns when no
hardware data exists for a point. `load_hardware_results` is a thin
`pandas.read_csv` wrapper -- merging a loaded hardware CSV into
`TFIMPointResult`/`HardwareEvaluationResult` objects is left to the
experiment layer (a later phase).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from atlas.analysis.metrics import absolute_error, relative_error_percent
from atlas.experiments.results import SimBenchmarkResult, SimValidationResult, TFIMBenchmarkResult


def _hardware_field(result: TFIMBenchmarkResult, getter) -> list:
    """Apply `getter(hardware_result)` per point, or `NaN` when hardware is absent.

    Purpose:
        Keep hardware CSV columns aligned with simulator rows even when only
        some (or no) points have hardware data.

    Inputs:
        result: Sweep container with per-point ``hardware_result`` fields.
        getter: Callable that extracts one scalar/field from a
            ``HardwareEvaluationResult``.

    Process:
        Walk points; if hardware is missing, emit ``np.nan`` so pandas keeps
        a stable column schema.

    Outputs:
        A list parallel to ``result.points``.

    Side effects:
        None.
    """

    values = []
    for point in result.points:
        hardware_result = point.hardware_result
        values.append(getter(hardware_result) if hardware_result is not None else np.nan)
    return values


def write_tfim_benchmark(result: TFIMBenchmarkResult, path: str) -> None:
    """Write `result` to a CSV matching the legacy `tfim_vis.py` benchmark schema.

    Purpose:
        Persist a VQE ``h``-sweep in a column layout expected by older analysis
        scripts and plotting tools.

    Inputs:
        result: Aggregated TFIM VQE sweep.
        path: Destination CSV filesystem path.

    Process:
        Collect exact/VQE energies and observables from accessors; fill hardware
        columns via ``_hardware_field``; compute abs/relative errors only where
        hardware energies exist.

    Outputs:
        None.

    Side effects:
        Writes a CSV file at ``path`` (overwrites if present).
    """

    exact_energies = result.exact_energies
    hardware_energies = _hardware_field(result, lambda hw: hw.energy)

    # Errors are only meaningful when a hardware energy was recorded for that h.
    abs_errors = [
        absolute_error(exact, hw_energy) if not np.isnan(hw_energy) else np.nan
        for exact, hw_energy in zip(exact_energies, hardware_energies)
    ]
    relative_errors = [
        relative_error_percent(exact, hw_energy) if not np.isnan(hw_energy) else np.nan
        for exact, hw_energy in zip(exact_energies, hardware_energies)
    ]

    df = pd.DataFrame(
        {
            "h": result.h_values,
            "exact_energy": exact_energies,
            "vqe_energy": result.vqe_energies,
            "hardware_energy": hardware_energies,
            "sim_zz": result.observable_values("zz"),
            "hardware_zz": _hardware_field(result, lambda hw: hw.observables.get("zz", np.nan)),
            "sim_x": result.observable_values("x"),
            "hardware_x": _hardware_field(result, lambda hw: hw.observables.get("x", np.nan)),
            "fidelity": result.fidelities,
            "abs_error": abs_errors,
            "relative_error_percent": relative_errors,
            "energy_stderr": _hardware_field(
                result, lambda hw: hw.energy_stderr if hw.energy_stderr is not None else np.nan
            ),
            "zz_stderr": _hardware_field(
                result, lambda hw: (hw.observable_stderr or {}).get("zz", np.nan)
            ),
            "x_stderr": _hardware_field(
                result, lambda hw: (hw.observable_stderr or {}).get("x", np.nan)
            ),
        }
    )
    df.to_csv(path, index=False)


def load_hardware_results(path: str) -> pd.DataFrame:
    """Load a hardware results CSV (`h`, `energy`, `zz`, `x`, `*_stderr` columns).

    Purpose:
        Read offline hardware measurements so experiments can attach them to
        simulator points without re-running jobs.

    Inputs:
        path: CSV path produced by a prior hardware run or external tool.

    Process:
        Delegate to ``pandas.read_csv``.

    Outputs:
        The raw ``DataFrame`` as-is; deciding how to merge it into
        ``TFIMPointResult``/``HardwareEvaluationResult`` objects is the
        experiment layer's responsibility.

    Side effects:
        Reads from disk.
    """

    return pd.read_csv(path)


def write_sim_benchmark(result: SimBenchmarkResult, path: str) -> None:
    """Write a Hamiltonian simulation sweep to CSV.

    Purpose:
        Persist fidelity and observable comparisons across a dynamics sweep
        (time, field, or Trotter steps).

    Inputs:
        result: Sweep of ``SimPointResult`` values.
        path: Destination CSV path.

    Process:
        Emit one row per point with fixed metadata columns, then expand
        observable names into ``sim_*`` / ``exact_*`` / ``error_*`` columns.

    Outputs:
        None.

    Side effects:
        Writes a CSV file at ``path``.
    """

    rows = []
    for point in result.points:
        row = {
            "h": point.h,
            "J": point.J,
            "evolution_time": point.evolution_time,
            "fidelity": point.fidelity,
            "method_name": point.sim_result.method_name,
            "num_trotter_steps": point.sim_result.num_trotter_steps,
            "circuit_depth": point.sim_result.circuit_depth,
        }
        # Dynamic columns keep the schema aligned with whichever observables
        # analysis requested (e.g. zz, x).
        for name, value in point.observables.items():
            row[f"sim_{name}"] = value
            row[f"exact_{name}"] = point.exact_observables[name]
            row[f"error_{name}"] = point.observable_errors[name]
        rows.append(row)

    pd.DataFrame(rows).to_csv(path, index=False)


def write_sim_validation(result: SimValidationResult, path: str) -> None:
    """Write a multi-method Trotter validation result to CSV.

    Purpose:
        Flatten the method × step grid into one CSV for offline comparison
        of Lie vs Strang (or other) product formulas.

    Inputs:
        result: Container whose ``series`` list holds one ``SimBenchmarkResult``
            per evolution method.
        path: Destination CSV path.

    Process:
        Nested loop over series and points; include ``max_operator_error`` and
        per-observable error columns.

    Outputs:
        None.

    Side effects:
        Writes a CSV file at ``path``.
    """

    rows = []
    for series in result.series:
        for point in series.points:
            row = {
                "h": point.h,
                "J": point.J,
                "evolution_time": point.evolution_time,
                "method_name": point.method_name,
                "num_trotter_steps": point.num_trotter_steps,
                "fidelity": point.fidelity,
                "max_operator_error": point.max_operator_error,
                "circuit_depth": point.circuit_depth,
            }
            for name, value in point.observables.items():
                row[f"sim_{name}"] = value
                row[f"exact_{name}"] = point.exact_observables[name]
                row[f"error_{name}"] = point.observable_errors[name]
            rows.append(row)

    pd.DataFrame(rows).to_csv(path, index=False)
