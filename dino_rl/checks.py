"""Native Chrome environment and input diagnostics."""

import json
import time
from argparse import Namespace

from stable_baselines3.common.env_checker import check_env

from .checkpoints import contract, write_json
from .env import DinoEnv


def movement_trace(env: DinoEnv, hold_steps: int = 4) -> list[dict]:
    """Measure the native jump arc for one uninterrupted key-hold length."""
    env.reset()
    trace = []
    for action in [1] * hold_steps + [0] * (16 - hold_steps):
        _, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            raise AssertionError("Movement check ended unexpectedly.")
        trace.append(
            {
                "seconds": info["survival_seconds"],
                "height": env.raw["trex"]["groundY"] - env.raw["trex"]["y"],
            }
        )
    return trace


def check(args: Namespace) -> None:
    """Verify native movement and save a reproducible interface report."""
    report = {"format_version": 1, "clock_mode": "native"}
    env = DinoEnv(realtime=True, show_browser=args.show_browser)
    try:
        check_env(env, warn=True, skip_render_check=True)
        assert env.raw["runningMs"] < 10000, "Clock units are incorrect"
        env.reset()
        before = env._read()
        time.sleep(0.25)
        after = env._read()
        assert after["distance"] > before["distance"], "Native game stopped while Python was busy"
        report["observed_update_rate_hz"] = (
            (after["speed"] - before["speed"])
            / 0.001
            / ((after["runningMs"] - before["runningMs"]) / 1000)
        )
        env.reset()
        env.step(2)
        assert env.raw["trex"]["ducking"], "Down did not duck"
        env.reset()
        env.step(1)
        assert env.raw["trex"]["jumping"], "Up did not jump"
        env.step(2)
        assert env.raw["trex"]["speedDrop"], "Down did not trigger a mid-air drop"
        env.max_seconds = 0.1
        env.reset()
        for _ in range(5):
            _, reward, terminated, truncated, _ = env.step(0)
            if terminated or truncated:
                break
        assert truncated and not terminated and reward > 0, "Time cap was treated as a crash"
        env.max_seconds = 300
        env.reset()
        for _ in range(200):
            _, reward, terminated, truncated, info = env.step(0)
            if terminated:
                break
        assert terminated and reward < -9, "Expected a real collision and crash penalty"
        report["no_input_collision"] = info
        report["native_jump_trace"] = movement_trace(env)
        report["contract"] = contract(env)
    finally:
        env.close()
    report["status"] = "PASS"
    write_json(args.output, report)
    print(json.dumps(report, indent=2), flush=True)
