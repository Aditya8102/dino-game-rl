"""Fixed-policy native-game evaluation and bird metrics."""

import json
import statistics
from argparse import Namespace
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv

from .checkpoints import contract, resolve_checkpoint, validate_contract, write_json
from .env import DinoEnv, empty_bird_metrics
from .resources import ResourceMonitor


def bird_scalars(metrics: dict[str, dict[str, int]]) -> dict[str, float]:
    """Flatten per-height outcomes; omit rates with no resolved encounters."""
    values = {}
    for height, counts in metrics.items():
        values.update({f"{name}_y{height}": count for name, count in counts.items()})
        resolved = counts["passes"] + counts["collisions"]
        if resolved:
            values[f"pass_rate_y{height}"] = counts["passes"] / resolved
    return values


def rollout(
    env: DinoEnv | VecEnv,
    policy: DQN | str,
    episodes: int,
    seed: int = 100,
    monitor: ResourceMonitor | None = None,
    stop_requested: Callable[[], bool] | None = None,
) -> dict | None:
    """Evaluate fair worker quotas, or return None when a stop cancels the batch."""
    records = []
    rng = np.random.default_rng(seed)
    vector = env if isinstance(env, VecEnv) else DummyVecEnv([lambda: env])
    obs = vector.reset()
    totals = np.zeros(vector.num_envs)
    lengths = np.zeros(vector.num_envs, dtype=int)
    # Allocate games per worker so quick failures cannot crowd out longer games.
    targets = np.asarray(
        [(episodes + index) // vector.num_envs for index in range(vector.num_envs)]
    )
    counts = np.zeros(vector.num_envs, dtype=int)
    while np.any(counts < targets):
        if stop_requested is not None and stop_requested():
            return None
        if policy == "no-input":
            actions = np.zeros(vector.num_envs, dtype=int)
        elif policy == "random":
            actions = rng.integers(3, size=vector.num_envs)
        else:
            actions = policy.predict(obs, deterministic=True)[0]
        obs, rewards, dones, infos = vector.step(actions)
        totals += rewards
        lengths += 1
        if monitor:
            # These are native-clock games: telemetry must never delay the actor.
            monitor.tick()
        for index, done in enumerate(dones):
            if done:
                if counts[index] < targets[index]:
                    info = {
                        key: value
                        for key, value in infos[index].items()
                        if key not in ("terminal_observation", "episode", "TimeLimit.truncated")
                    }
                    records.append(
                        {
                            "episode": len(records) + 1,
                            **info,
                            "return": float(totals[index]),
                            "length": int(lengths[index]),
                            "truncated": infos[index]["TimeLimit.truncated"],
                        }
                    )
                counts[index] += 1
                totals[index] = 0.0
                lengths[index] = 0
    birds = empty_bird_metrics()
    for record in records:
        for height, counts in record["bird_metrics"].items():
            for name, count in counts.items():
                birds[height][name] += count
    survival = [r["survival_seconds"] for r in records]
    return {
        "episodes": episodes,
        "mean_reward": statistics.mean(r["return"] for r in records),
        "mean_ep_length": statistics.mean(r["length"] for r in records),
        "mean_survival_seconds": statistics.mean(survival),
        "median_survival_seconds": statistics.median(survival),
        "mean_score": statistics.mean(r["score"] for r in records),
        "time_limit_completions": sum(r["truncated"] for r in records),
        "bird_metrics": birds,
        "records": records,
    }


def evaluate(args: Namespace) -> None:
    """Run a standalone test or visible demonstration without updating weights."""
    checkpoint, metadata = resolve_checkpoint(args.checkpoint)
    env = DinoEnv(
        show_browser=args.command == "play" or args.show_browser,
        max_seconds=args.max_seconds,
    )
    try:
        monitor = ResourceMonitor()
        validate_contract(metadata["contract"], env)
        model = DQN.load(checkpoint / "model.zip", device="cpu")
        report = {
            "checkpoint": str(checkpoint),
            "clock_mode": "native",
            "browser_mode": "visible" if env.render_mode == "human" else "headless",
            "max_seconds": args.max_seconds,
            "training_steps": metadata["steps"],
            "contract": contract(env),
            **rollout(env, model, args.episodes),
        }
        report["resources"] = monitor.summary()
        run = Path(args.run).resolve()
        output = (
            args.output
            or run / "reports" / f"{args.command}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
        )
        write_json(output, report)
        if args.command == "evaluate":
            from torch.utils.tensorboard import SummaryWriter

            with SummaryWriter(str(run / "tensorboard" / "dino_0")) as writer:
                prefix = f"test/{report['browser_mode']}"
                for key in (
                    "mean_reward",
                    "mean_ep_length",
                    "mean_survival_seconds",
                    "median_survival_seconds",
                    "mean_score",
                ):
                    writer.add_scalar(f"{prefix}/{key}", report[key], metadata["steps"])
                for name, value in bird_scalars(report["bird_metrics"]).items():
                    writer.add_scalar(f"{prefix}/birds/{name}", value, metadata["steps"])
        print(json.dumps(report, indent=2), flush=True)
        print(f"Saved report: {output}", flush=True)
    finally:
        env.close()
