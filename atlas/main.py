"""Command-line entry point for the Atlas quantum simulation framework.

Parses ``--config``, loads YAML, builds and runs the configured experiment,
and saves outputs. All component construction lives in ``experiments.factory``.
"""

from __future__ import annotations

import argparse
import sys

from atlas.experiments.factory import run_experiment
from atlas.experiments.outputs import save_outputs
from atlas.io.yaml_config import load_config


def parse_args(argv=None) -> argparse.Namespace:
    """Parse CLI arguments for ``main()``."""

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
    args = parse_args(argv)

    try:
        config = load_config(args.config)
        run = run_experiment(config)
        save_outputs(run)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
