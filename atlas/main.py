"""Command-line entry point for the Atlas quantum simulation framework.

Parses ``--config``, loads YAML, builds and runs the configured experiment,
and saves outputs. All component construction lives in ``experiments.factory``.

This module intentionally stays thin: it wires three library calls together
and maps failures to a non-zero exit code for shell scripts and CI.
"""

from __future__ import annotations

import argparse
import sys

from atlas.experiments.factory import run_experiment
from atlas.experiments.outputs import save_outputs
from atlas.io.yaml_config import load_config


def parse_args(argv=None) -> argparse.Namespace:
    """Parse CLI arguments for ``main()``.

    Purpose:
        Define the single required flag Atlas needs to locate an experiment.

    Inputs:
        argv: Optional argument list. When ``None``, ``argparse`` reads
            ``sys.argv`` (normal CLI usage). Tests can pass an explicit list.

    Process:
        Build an ``ArgumentParser`` that requires ``--config``, then parse.

    Outputs:
        An ``argparse.Namespace`` with a ``.config`` string path.

    Side effects:
        None (pure parse; does not open the config file).
    """

    parser = argparse.ArgumentParser(
        description="Run an Atlas experiment from a YAML configuration file."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML experiment configuration file.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    """Run one Atlas experiment end-to-end from a YAML config path.

    Purpose:
        Provide the process entry point: load intent, execute, persist artifacts.

    Inputs:
        argv: Optional CLI argument list forwarded to ``parse_args``.

    Process:
        1. Parse ``--config``.
        2. Validate and load YAML into an ``AtlasConfig``.
        3. Build and run the configured experiment.
        4. Write CSV/plots/console summaries for the result type.

    Outputs:
        None. Success is indicated by a normal return; failures raise
        ``SystemExit(1)``.

    Side effects:
        May print an error to stderr, write files under the configured output
        directory, and print progress/summaries to stdout via ``save_outputs``.
    """

    args = parse_args(argv)

    try:
        # Intent only: typed config with no computed energies or states.
        config = load_config(args.config)
        # Construction + execution live in the experiments layer.
        run = run_experiment(config)
        # Persistence is separate so algorithms never touch the filesystem.
        save_outputs(run)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
