"""Checkpoint files, provenance and game-contract validation."""

import json
import logging
import pickle
import random
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecEnv

from .env import ACTIONS, SCHEMA, STEP_MS, DinoEnv
from .resources import CpuBudget

logger = logging.getLogger(__name__)


def write_json(path: str | Path, data: Any) -> None:
    """Atomically replace a JSON artifact after its complete contents are written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def append_json(path: str | Path, data: dict) -> None:
    """Append one complete record to a run's event log."""
    with Path(path).open("a") as handle:
        handle.write(json.dumps(data, allow_nan=False) + "\n")


def contract(env: DinoEnv | VecEnv) -> dict:
    """Describe the game interface needed to interpret a checkpoint."""
    versions = env.get_attr("browser_version") if isinstance(env, VecEnv) else [env.browser_version]
    if len(set(versions)) != 1:
        raise ValueError("All training workers must use the same Chrome version.")
    return {
        "schema": SCHEMA,
        "step_ms": STEP_MS,
        "actions": list(ACTIONS),
        "survival_reward_per_second": 1.0,
        "collision_penalty": -10.0,
        "chrome_version": versions[0],
    }


def resolve_checkpoint(pointer: str | Path) -> tuple[Path, dict]:
    """Resolve a trusted local pointer and read its checkpoint metadata."""
    pointer = Path(pointer).resolve()
    data = json.loads(pointer.read_text())
    checkpoint = pointer.parent / data["checkpoint"]
    return checkpoint, json.loads((checkpoint / "metadata.json").read_text())


def validate_contract(
    saved: dict, env: DinoEnv | VecEnv, *, accept_chrome_version: str | None = None
) -> None:
    """Require the same contract, with an explicit exact-version patch exception."""
    expected = contract(env)
    if accept_chrome_version is not None:
        if (
            accept_chrome_version != expected["chrome_version"]
            or saved["chrome_version"].rsplit(".", 1)[0] != accept_chrome_version.rsplit(".", 1)[0]
        ):
            raise ValueError("Only the exact installed Chrome patch version can be accepted.")
        saved = {**saved, "chrome_version": accept_chrome_version}
    if saved != expected:
        raise ValueError(f"Checkpoint/game contract differs. Saved: {saved}; current: {expected}")


def save_checkpoint(
    model: DQN,
    env: DinoEnv | VecEnv,
    run: Path,
    config: dict,
    evaluation: dict | None,
    best_quality: tuple[float, float],
    budget: CpuBudget,
) -> tuple[float, float]:
    """Save learning state before publishing latest/best checkpoint pointers."""
    quality = (
        (evaluation["median_survival_seconds"], evaluation["mean_survival_seconds"])
        if evaluation is not None
        else (-1.0, -1.0)
    )
    checkpoint = run / "checkpoints" / f"{model.num_timesteps:09d}-{time.time_ns()}"
    checkpoint.mkdir(parents=True)
    model.save(checkpoint / "model.zip")
    model.save_replay_buffer(checkpoint / "replay.pkl")
    with (checkpoint / "rng.pkl").open("wb") as handle:
        pickle.dump(
            {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
            },
            handle,
        )
    metadata = {
        "format_version": 1,
        "algorithm": "DQN",
        "clock_mode": "native",
        "steps": model.num_timesteps,
        "contract": contract(env),
        "lifetime_training_steps": model.num_timesteps + config.get("source_training_steps", 0),
        "config": config,
        "evaluation": evaluation,
        "best_quality": list(max(best_quality, quality)),
        "gradient_updates": model._n_updates,
        "replay_size": model.replay_buffer.size() * model.replay_buffer.n_envs,
        "workers": model.n_envs,
        "packages": {
            name: version(name)
            for name in ("torch", "stable-baselines3", "gymnasium", "playwright")
        },
        "resources": budget.summary(),
        "saved_at": datetime.now(UTC).isoformat(),
    }
    write_json(checkpoint / "metadata.json", metadata)
    pointer = {"checkpoint": str(checkpoint.relative_to(run)), "steps": model.num_timesteps}
    write_json(run / "latest.json", pointer)
    if evaluation is not None and quality >= best_quality:
        write_json(run / "best.json", pointer)
    logger.info("Saved %s", checkpoint / "model.zip")
    return max(best_quality, quality)
