"""Command-line entry point for training, evaluation and environment checks."""

import argparse
import logging
import os

# Configure CPU libraries before importing the training stack.
for name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[name] = "1"

import torch

from .checks import check
from .evaluation import evaluate
from .training import train


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("Must be greater than zero.")
    return result


def probability(value: str) -> float:
    result = float(value)
    if not 0 <= result <= 1:
        raise argparse.ArgumentTypeError("Must be between zero and one.")
    return result


def main() -> None:
    """Parse one command and configure the CPU-only runtime."""
    logging.basicConfig(level=os.environ.get("DINO_LOG_LEVEL", "INFO"), format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "train", "evaluate", "play"):
        sub = commands.add_parser(name)
        if name == "evaluate":
            sub.add_argument(
                "--cpu-cores",
                type=float,
                default=1.0,
                help="Soft CPU budget in controlled-clock diagnostics only.",
            )
        if name != "play":
            sub.add_argument("--show-browser", action="store_true")
        if name == "check":
            sub.add_argument("--output", default="runs/check.json")
        else:
            sub.add_argument("--max-seconds", type=float, default=300)
        if name == "train":
            sub.add_argument(
                "--workers",
                type=int,
                choices=(1, 2, 4, 8),
                default=8,
                help="Parallel real Chrome games; gradient work scales with transitions.",
            )
            sub.add_argument("--run", default="runs/train")
            sub.add_argument(
                "--steps",
                type=positive_int,
                default=200000,
                help="Total training step target, including resumed steps.",
            )
            sub.add_argument("--seed", type=int, default=7)
            initialization = sub.add_mutually_exclusive_group()
            initialization.add_argument("--resume", help="Resume this native run's latest.json.")
            initialization.add_argument(
                "--initialize-from",
                help="Initialize policy weights from another run; replay and optimizer start fresh.",
            )
            sub.add_argument("--checkpoint-every", type=positive_int, default=25000)
            sub.add_argument("--eval-episodes", type=positive_int, default=8)
            sub.add_argument(
                "--exploration-final",
                type=probability,
                help="Final random-action probability; resume preserves the saved value unless overridden.",
            )
        elif name in ("evaluate", "play"):
            sub.add_argument("--checkpoint", default="models/best.json")
            sub.add_argument(
                "--run",
                default="runs/evaluation",
                help="Directory for standalone reports and TensorBoard data.",
            )
            sub.add_argument("--episodes", type=positive_int, default=5)
            if name == "evaluate":
                sub.add_argument(
                    "--controlled-clock",
                    action="store_true",
                    help="Legacy diagnostic only; normal evaluation uses native time.",
                )
            sub.add_argument(
                "--output", help="Defaults to a timestamped JSON report inside this run."
            )
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    {"check": check, "train": train, "evaluate": evaluate, "play": evaluate}[args.command](args)


if __name__ == "__main__":
    main()
