"""Compare normal-speed and controlled Dino play, retaining failure diagnostics."""

import argparse
import hashlib
import json
import statistics
import time
from collections import Counter, deque
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import DQN
from torch.utils.tensorboard import SummaryWriter

from dino_rl.checkpoints import resolve_checkpoint, validate_contract, write_json
from dino_rl.env import DinoEnv


def weights_hash(model):
    digest = hashlib.sha256()
    for value in model.policy.state_dict().values():
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def collision_description(raw):
    t = raw["trex"]
    overlapping = [
        o for o in raw["obstacles"] if o["x"] < t["x"] + t["width"] and o["x"] + o["width"] > t["x"]
    ]
    if not overlapping:
        return {"obstacle": None, "motion": "no broad-phase overlap found"}
    obstacle = overlapping[0]
    motion = (
        ("descending" if t["vy"] > 0 else "ascending")
        if t["jumping"]
        else ("grounded-ducking" if t["ducking"] else "grounded-standing")
    )
    return {
        "obstacle": obstacle,
        "motion": motion,
        "dino_height": t["groundY"] - t["y"],
        "vertical_velocity": t["vy"],
    }


def run_mode(model, mode, args, output):
    realtime = mode == "native"
    env = DinoEnv(
        realtime=realtime, show_browser=realtime, max_seconds=args.max_seconds, cpu_cores=6
    )
    records, inference_ms, decision_ms, actions = [], [], [], Counter()
    try:
        validate_contract(args.contract, env)
        for episode in range(1, args.episodes + 1):
            obs, _ = env.reset()
            tail = deque(maxlen=24)
            last_decision = None
            total = 0.0
            episode_started = time.perf_counter()
            episode_intervals = []
            episode_inference = []
            episode_actions = Counter()
            bird_frames = 0
            steps = 0
            while True:
                before = env.raw
                decision = time.perf_counter()
                if last_decision is not None:
                    episode_intervals.append(1000 * (decision - last_decision))
                last_decision = decision
                action = int(model.predict(obs, deterministic=True)[0])
                episode_inference.append(1000 * (time.perf_counter() - decision))
                observation_before = obs.tolist()
                try:
                    obs, reward, terminated, truncated, info = env.step(action)
                except Exception as error:
                    write_json(
                        output / f"{mode}-invalid-episode-{episode:02d}.json",
                        {
                            "error": str(error),
                            "before": before,
                            "after": env.raw,
                            "final_decisions": list(tail),
                        },
                    )
                    raise
                steps += 1
                episode_actions[action] += 1
                bird_frames += any(o["type"] == "pterodactyl" for o in before["obstacles"])
                total += reward
                tail.append(
                    {
                        "observation": observation_before,
                        "action": action,
                        "before": before,
                        "after": env.raw,
                        "reward": reward,
                    }
                )
                if steps % 100 == 0:
                    write_json(
                        output / "progress.json",
                        {"mode": mode, "episode": episode, "episodes": args.episodes, **info},
                    )
                if terminated or truncated:
                    break
            # Diagnose the final decisions after the game ends, adding no live inference delay.
            for frame in tail:
                tensor, _ = model.policy.obs_to_tensor(
                    np.asarray(frame["observation"], dtype=np.float32)
                )
                with torch.no_grad():
                    frame["q_values"] = model.q_net(tensor).cpu().numpy()[0].tolist()
            description = collision_description(env.raw) if terminated else {"motion": "time-limit"}
            elapsed = time.perf_counter() - episode_started
            record = {
                "episode": episode,
                **info,
                "truncated": truncated,
                "return": total,
                "wall_seconds": elapsed,
                "length": steps,
                "action_counts": dict(episode_actions),
                "bird_observation_frames": bird_frames,
                "inference_p95_ms": float(np.percentile(episode_inference, 95)),
                "decision_interval_p95_ms": float(np.percentile(episode_intervals, 95))
                if episode_intervals
                else None,
                "decision_intervals_over_75ms": sum(t > 75 for t in episode_intervals),
                "estimated_game_updates_per_second": (
                    (env.raw["speed"] - 6) / 0.001 / info["survival_seconds"]
                    if env.raw["speed"] < 13
                    else None
                ),
                "collision": description,
            }
            records.append(record)
            actions.update(episode_actions)
            inference_ms.extend(episode_inference)
            decision_ms.extend(episode_intervals)
            write_json(
                output / f"{mode}-episode-{episode:02d}.json",
                {**record, "final_decisions": list(tail)},
            )
            if terminated:
                env.page.screenshot(path=str(output / f"{mode}-episode-{episode:02d}.png"))
            write_json(
                output / "progress.json",
                {"mode": mode, "finished_episode": episode, "episodes": args.episodes, **info},
            )
            print(json.dumps({"mode": mode, **record}), flush=True)
    finally:
        env.close()
    survival = [r["survival_seconds"] for r in records]
    summary = {
        "mode": mode,
        "episodes": len(records),
        "max_seconds": args.max_seconds,
        "mean_reward": statistics.mean(r["return"] for r in records),
        "mean_ep_length": statistics.mean(r["length"] for r in records),
        "mean_survival_seconds": statistics.mean(survival),
        "median_survival_seconds": statistics.median(survival),
        "minimum_survival_seconds": min(survival),
        "maximum_survival_seconds": max(survival),
        "mean_score": statistics.mean(r["score"] for r in records),
        "time_limit_completions": sum(r["truncated"] for r in records),
        "action_counts": dict(actions),
        "inference_p95_ms": float(np.percentile(inference_ms, 95)),
        "decision_interval_p95_ms": float(np.percentile(decision_ms, 95)) if decision_ms else None,
        "decision_intervals_over_75ms": sum(t > 75 for t in decision_ms),
        "records": records,
    }
    write_json(output / f"{mode}-summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="models/best.json")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-seconds", type=float, default=120)
    parser.add_argument("--output", required=True, help="New directory for diagnostic reports.")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("At least one episode is required.")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    checkpoint, metadata = resolve_checkpoint(args.checkpoint)
    args.contract = metadata["contract"]
    model = DQN.load(checkpoint / "model.zip", device="cpu")
    original_weights = weights_hash(model)
    summaries = {}
    for mode in ("native", "controlled"):
        summaries[mode] = run_mode(model, mode, args, output)
        with SummaryWriter(str(output / "tensorboard" / "dino_0")) as writer:
            prefix = "test/visible" if mode == "native" else "diagnostics/controlled"
            for key in (
                "mean_reward",
                "mean_ep_length",
                "mean_survival_seconds",
                "median_survival_seconds",
                "mean_score",
            ):
                writer.add_scalar(f"{prefix}/{key}", summaries[mode][key], metadata["steps"])
    if original_weights != weights_hash(model):
        raise AssertionError("Policy weights changed during evaluation.")
    write_json(
        output / "comparison.json",
        {
            "checkpoint": str(checkpoint),
            "training_steps": metadata["steps"],
            "weights_unchanged": True,
            "exploration_enabled": False,
            **summaries,
        },
    )
    print("Evaluation complete; weights unchanged. Reports: " + str(output), flush=True)


if __name__ == "__main__":
    main()
