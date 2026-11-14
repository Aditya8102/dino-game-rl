"""DQN training, progress reporting and checkpointed continuation."""

import json
import logging
import os
import pickle
import random
import signal
import statistics
import time
from argparse import Namespace
from collections import deque
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import LinearSchedule

from .checkpoints import (
    append_json,
    contract,
    resolve_checkpoint,
    save_checkpoint,
    validate_contract,
    write_json,
)
from .config import DQN_DEFAULTS, NETWORK_SHAPE
from .env import empty_bird_metrics
from .evaluation import bird_scalars, rollout
from .parallel import repartition_replay, training_env
from .resources import CpuBudget

logger = logging.getLogger(__name__)


class Progress(BaseCallback):
    """Collect training metrics and honor a checkpointed stop request."""

    def __init__(self, budget: CpuBudget, run: Path) -> None:
        super().__init__()
        self.budget = budget
        self.run = run
        self.stop_requested = False
        self.last_print = 0
        self.survival = deque(maxlen=20)
        self.birds = empty_bird_metrics()

    def _on_training_start(self) -> None:
        self.started = time.monotonic()
        self.start_steps = self.last_print = self.model.num_timesteps

    def _on_step(self) -> bool:
        self.budget.tick(throttle=False)
        for info in self.locals["infos"]:
            if "episode" in info:
                append_json(
                    self.run / "training-episodes.jsonl",
                    {
                        "steps": self.num_timesteps,
                        **info["episode"],
                        "survival_seconds": info["survival_seconds"],
                        "score": info["score"],
                        "bird_metrics": info["bird_metrics"],
                    },
                )
                for height, counts in info["bird_metrics"].items():
                    for name, count in counts.items():
                        self.birds[height][name] += count
                self.survival.append(info["survival_seconds"])
                self.logger.record("rollout/survival_mean_20", statistics.mean(self.survival))
        if self.num_timesteps - self.last_print >= 1000:
            self.last_print = self.num_timesteps
            resources = self.budget.summary()
            speed = (self.num_timesteps - self.start_steps) / max(
                time.monotonic() - self.started, 1e-6
            )
            self.logger.record("resources/transitions_per_second", speed)
            for key in ("mean_machine_cpu_percent", "process_group_rss_mb", "mean_cpu_cores"):
                self.logger.record(f"resources/{key}", resources[key])
            self.logger.record("train/gradient_updates", self.model._n_updates)
            for name, value in bird_scalars(self.birds).items():
                self.logger.record(f"birds/train/{name}", value)
            self.logger.dump(self.num_timesteps)
            logger.info(
                json.dumps(
                    {
                        "steps": self.num_timesteps,
                        "transitions_per_second": round(speed, 1),
                        "exploration": round(self.model.exploration_rate, 3),
                        "bird_metrics": self.birds,
                        "resources": resources,
                    }
                )
            )
        return not self.stop_requested


def train(args: Namespace) -> None:
    """Train or resume one native-clock DQN run using complete checkpoints."""
    # Lower scheduling priority is inherited by Chrome; one torch thread is set in main().
    if args.workers == 1:
        os.nice(10)
    run = Path(args.run).resolve()
    run.mkdir(parents=True, exist_ok=True)
    if not args.resume and (run / "config.json").exists():
        raise ValueError("This run already exists. Use --resume or choose a new --run directory.")
    continuation = False
    if args.resume:
        checkpoint, metadata = resolve_checkpoint(args.resume)
        continuation = Path(args.resume).resolve().parent != run
        if continuation and (run / "config.json").exists():
            raise ValueError("Continuing into another run requires a new --run directory.")
        if metadata["config"].get("clock_mode") != "native":
            raise ValueError(
                "Controlled-clock replay cannot be resumed as native data. "
                "Use --initialize-from with a new --run to transfer policy weights only."
            )
        if metadata.get("artifact_type") == "inference":
            raise ValueError(
                "This inference release has no replay or random-generator state. "
                "Use --initialize-from with a new --run to train from its policy weights."
            )
        browser_mode = "visible" if args.show_browser else "headless"
        if not continuation and metadata["config"].get("browser_mode") != browser_mode:
            raise ValueError(
                "The run's browser rendering mode differs. "
                "Use --resume with a new --run to keep the new evaluation condition separate."
            )
        if not continuation and metadata["config"]["max_seconds"] != args.max_seconds:
            raise ValueError(
                "The run's episode time limit differs. "
                "Use --resume with a new --run to keep evaluations comparable."
            )
    env = training_env(args.workers, args.max_seconds, args.show_browser)
    budget = CpuBudget()
    try:
        config = {
            **DQN_DEFAULTS,
            "seed": args.seed,
            "network": list(NETWORK_SHAPE[1:-1]),
            "gradient_steps": args.workers,
            "torch_threads": torch.get_num_threads(),
            "max_seconds": args.max_seconds,
        }
        best_quality = (-1.0, -1.0)
        if args.resume:
            validate_contract(metadata["contract"], env)
            config = metadata["config"]
            model = DQN.load(
                checkpoint / "model.zip",
                env=env,
                device="cpu",
                tensorboard_log=str(run / "tensorboard"),
            )
            model.load_replay_buffer(checkpoint / "replay.pkl")
            model.replay_buffer, migration = repartition_replay(model.replay_buffer, args.workers)
            migration.update(checkpoint=str(checkpoint), at_steps=model.num_timesteps)
            write_json(run / "replay-migration.json", migration)
            append_json(run / "replay-migrations.jsonl", migration)
            model.gradient_steps = args.workers
            model._n_calls = model.num_timesteps // args.workers
            with (checkpoint / "rng.pkl").open("rb") as handle:
                rng = pickle.load(handle)
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"])
            if continuation:
                config["resume_checkpoint"] = str(checkpoint)
                baselines = {
                    name: rollout(env, name, args.eval_episodes, args.seed + 100, budget)
                    for name in ("no-input", "random")
                }
                write_json(run / "baselines.json", baselines)
            else:
                best_quality = tuple(metadata["best_quality"])
        else:
            baselines = {
                name: rollout(env, name, args.eval_episodes, args.seed + 100, budget)
                for name in ("no-input", "random")
            }
            write_json(run / "baselines.json", baselines)
            model = DQN(
                "MlpPolicy",
                env,
                policy_kwargs={"net_arch": config["network"]},
                learning_rate=config["learning_rate"],
                gamma=config["gamma"],
                batch_size=config["batch_size"],
                buffer_size=config["buffer_size"],
                learning_starts=config["learning_starts"],
                train_freq=config["train_freq"],
                gradient_steps=config["gradient_steps"],
                target_update_interval=config["target_update_interval"],
                exploration_initial_eps=1.0,
                exploration_final_eps=0.05,
                seed=args.seed,
                device="cpu",
                verbose=0,
                tensorboard_log=str(run / "tensorboard"),
            )
            if args.initialize_from:
                source_checkpoint, source_metadata = resolve_checkpoint(args.initialize_from)
                validate_contract(source_metadata["contract"], env)
                source = DQN.load(source_checkpoint / "model.zip", device="cpu")
                model.policy.load_state_dict(source.policy.state_dict())
                config.update(
                    source_checkpoint=str(source_checkpoint),
                    source_training_steps=source_metadata.get(
                        "lifetime_training_steps", source_metadata["steps"]
                    ),
                    initialization="policy_weights_only",
                )
                del source
            logger.info(
                "Baselines: %s",
                json.dumps({k: v["median_survival_seconds"] for k, v in baselines.items()}),
            )
        if args.steps <= model.num_timesteps:
            raise ValueError(
                f"--steps is a total target and must exceed checkpoint step {model.num_timesteps}."
            )
        if args.exploration_final is not None:
            model.exploration_final_eps = args.exploration_final
        config.update(
            max_seconds=args.max_seconds,
            workers=args.workers,
            gradient_steps=args.workers,
            exploration_final_eps=model.exploration_final_eps,
        )
        config["browser_mode"] = "visible" if args.show_browser else "headless"
        write_json(
            run / "config.json",
            {
                **config,
                "contract": contract(env),
                "target_steps": args.steps,
                "checkpoint_every": args.checkpoint_every,
                "eval_episodes": args.eval_episodes,
                "selection_metric": "median_survival_seconds",
            },
        )
        if continuation:
            save_checkpoint(model, env, run, config, None, best_quality, budget)
        callback = Progress(budget, run)
        previous_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, lambda *_: setattr(callback, "stop_requested", True))
        try:
            while model.num_timesteps < args.steps and not callback.stop_requested:
                end = min(args.steps, model.num_timesteps + args.checkpoint_every)
                # Keep epsilon tied to absolute steps, including after a run is extended.
                model.exploration_fraction = config["exploration_steps"] / end
                model.exploration_schedule = LinearSchedule(
                    model.exploration_initial_eps,
                    model.exploration_final_eps,
                    model.exploration_fraction,
                )
                model.exploration_rate = model.exploration_schedule(1 - model.num_timesteps / end)
                # Evaluation shares the worker games: discard stale observations before collecting data.
                model.set_env(model.get_env(), force_reset=True)
                model.learn(
                    total_timesteps=end - model.num_timesteps,
                    reset_num_timesteps=False,
                    callback=callback,
                    tb_log_name="dino",
                    log_interval=10,
                )
                evaluation = None
                if not callback.stop_requested:
                    evaluation = rollout(
                        env,
                        model,
                        args.eval_episodes,
                        budget=budget,
                        stop_requested=lambda: callback.stop_requested,
                    )
                if evaluation is not None:
                    for key in (
                        "mean_reward",
                        "mean_ep_length",
                        "mean_survival_seconds",
                        "median_survival_seconds",
                        "mean_score",
                    ):
                        model.logger.record(f"eval/{key}", evaluation[key])
                    for name, value in bird_scalars(evaluation["bird_metrics"]).items():
                        model.logger.record(f"birds/eval/{name}", value)
                    for name, baseline in json.loads((run / "baselines.json").read_text()).items():
                        model.logger.record(
                            f"baseline/{name}_survival_seconds", baseline["median_survival_seconds"]
                        )
                    model.logger.dump(model.num_timesteps)
                    append_json(
                        run / "evaluations.jsonl",
                        {"steps": model.num_timesteps, "clock_mode": "native", **evaluation},
                    )
                best_quality = save_checkpoint(
                    model, env, run, config, evaluation, best_quality, budget
                )
                if evaluation is not None:
                    logger.info(
                        "Evaluation: %s",
                        json.dumps({k: v for k, v in evaluation.items() if k != "records"}),
                    )
        finally:
            signal.signal(signal.SIGINT, previous_handler)
        write_json(
            run / "summary.json",
            {
                "clock_mode": "native",
                "steps": model.num_timesteps,
                "lifetime_training_steps": model.num_timesteps
                + config.get("source_training_steps", 0),
                "best_median_survival_seconds": best_quality[0],
                "best_mean_survival_seconds": best_quality[1],
                "interrupted": callback.stop_requested,
                "resources": budget.summary(),
            },
        )
    finally:
        env.close()
