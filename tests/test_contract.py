import json
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.utils import LinearSchedule
from stable_baselines3.common.vec_env import DummyVecEnv

from dino_rl.checkpoints import contract, save_checkpoint, validate_contract
from dino_rl.env import DinoEnv, empty_bird_metrics, observation
from dino_rl.evaluation import bird_scalars, rollout
from dino_rl.parallel import repartition_replay
from dino_rl.training import train


def game_state():
    return {
        "speed": 6.0,
        "width": 600,
        "height": 150,
        "trex": {
            "x": 14,
            "y": 93,
            "groundY": 93,
            "vy": 0,
            "width": 44,
            "jumping": False,
            "ducking": False,
            "speedDrop": False,
            "reachedMinHeight": True,
        },
        "obstacles": [],
    }


def obstacle(x, kind="cactusSmall"):
    return {"x": x, "y": 105, "width": 17, "height": 35, "type": kind, "speedOffset": 0}


class FixedEpisode(gym.Env):
    browser_version = "157.test"
    observation_space = spaces.Box(-1, 1, shape=(28,), dtype=np.float32)
    action_space = spaces.Discrete(3)

    def __init__(self, length):
        self.length = length

    def reset(self, *, seed=None, options=None):
        self.steps = 0
        return np.zeros(28, dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        return (
            np.zeros(28, dtype=np.float32),
            1.0,
            self.steps == self.length,
            False,
            {
                "survival_seconds": self.steps,
                "score": self.steps,
                "bird_metrics": empty_bird_metrics(),
            },
        )


class ContractTests(unittest.TestCase):
    def test_cancelled_evaluation_does_not_publish_partial_results(self):
        environment = FixedEpisode(10)
        result = rollout(
            environment, "no-input", episodes=2, stop_requested=lambda: environment.steps >= 1
        )
        self.assertIsNone(result)
        self.assertEqual(environment.steps, 1)

    def test_inference_release_cannot_resume_before_opening_chrome(self):
        args = SimpleNamespace(workers=8, run="runs/unused-test", resume="models/best.json")
        with TemporaryDirectory() as directory:
            args.run = directory
            with patch("dino_rl.training.training_env") as create:
                with self.assertRaisesRegex(ValueError, "Use --initialize-from"):
                    train(args)
                create.assert_not_called()

    def test_renderer_change_cannot_mix_evaluation_in_the_same_run(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "checkpoint").mkdir()
            (root / "checkpoint/metadata.json").write_text(
                json.dumps({"config": {"clock_mode": "native", "browser_mode": "visible"}})
            )
            pointer = root / "latest.json"
            pointer.write_text(json.dumps({"checkpoint": "checkpoint"}))
            args = SimpleNamespace(
                workers=8, run=str(root), resume=str(pointer), show_browser=False
            )
            with patch("dino_rl.training.training_env") as create:
                with self.assertRaisesRegex(ValueError, "rendering mode differs"):
                    train(args)
                create.assert_not_called()

    def test_native_resume_rejects_controlled_replay_before_opening_chrome(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "checkpoint").mkdir()
            (root / "checkpoint/metadata.json").write_text(
                json.dumps({"config": {"clock_mode": "controlled"}})
            )
            pointer = root / "latest.json"
            pointer.write_text(json.dumps({"checkpoint": "checkpoint"}))
            args = SimpleNamespace(workers=8, run=str(root), resume=str(pointer))
            with patch("dino_rl.training.training_env") as create:
                with self.assertRaisesRegex(ValueError, "Controlled-clock replay"):
                    train(args)
                create.assert_not_called()

    def test_same_run_cannot_change_the_evaluation_time_limit(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "checkpoint").mkdir()
            (root / "checkpoint/metadata.json").write_text(
                json.dumps(
                    {
                        "config": {
                            "clock_mode": "native",
                            "browser_mode": "headless",
                            "max_seconds": 300,
                        }
                    }
                )
            )
            pointer = root / "latest.json"
            pointer.write_text(json.dumps({"checkpoint": "checkpoint"}))
            args = SimpleNamespace(
                workers=8, run=str(root), resume=str(pointer), show_browser=False, max_seconds=120
            )
            with patch("dino_rl.training.training_env") as create:
                with self.assertRaisesRegex(ValueError, "episode time limit differs"):
                    train(args)
                create.assert_not_called()

    def test_interrupt_checkpoint_does_not_replace_evaluated_best(self):
        with TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "best.json").write_text('{"checkpoint":"previous-best"}')
            model = SimpleNamespace(
                num_timesteps=10,
                n_envs=1,
                _n_updates=1,
                replay_buffer=SimpleNamespace(size=lambda: 10, n_envs=1),
                save=lambda p: Path(p).write_bytes(b"weights"),
                save_replay_buffer=lambda p: Path(p).write_bytes(b"replay"),
            )
            quality = save_checkpoint(
                model,
                SimpleNamespace(browser_version="157.test"),
                run,
                {},
                None,
                (12.0, 13.0),
                SimpleNamespace(summary=lambda: {}),
            )
            self.assertEqual(quality, (12.0, 13.0))
            self.assertEqual(
                json.loads((run / "best.json").read_text())["checkpoint"], "previous-best"
            )
            saved = run / json.loads((run / "latest.json").read_text())["checkpoint"]
            self.assertIsNone(json.loads((saved / "metadata.json").read_text())["evaluation"])

    def frame_test_env(self, realtime):
        env = DinoEnv.__new__(DinoEnv)
        env.realtime = realtime
        env.action_space = spaces.Discrete(3)
        env.previous_action = 0
        env._ended = False
        env.elapsed = 0.0
        env.max_seconds = 120
        env.budget = None
        env.frame_stalls = 0
        env._birds = {}
        env.bird_metrics = empty_bird_metrics()
        env._no_progress_since = None
        env.raw = {
            **game_state(),
            "runningMs": 1000,
            "clock": 1000,
            "score": 10,
            "distance": 100,
            "hidden": False,
            "paused": False,
            "crashed": False,
        }
        env._advance = lambda: None
        env._read = lambda: {**env.raw, "clock": env.raw["clock"] + 50}
        return env

    def test_native_missed_frame_is_not_a_collision_or_runner_failure(self):
        env = self.frame_test_env(True)
        _, reward, terminated, truncated, info = env.step(0)
        self.assertEqual(reward, 0)
        self.assertFalse(terminated or truncated)
        self.assertEqual(info["frame_stalls"], 1)
        env._read = lambda: {**env.raw, "runningMs": env.raw["runningMs"] + 50}
        _, reward, _, _, _ = env.step(0)
        self.assertAlmostEqual(reward, 0.05)
        self.assertIsNone(env._no_progress_since)

    def test_controlled_clock_still_requires_progress(self):
        with self.assertRaisesRegex(RuntimeError, "clock did not advance"):
            self.frame_test_env(False).step(0)

    def test_sustained_native_freeze_is_still_an_error(self):
        env = self.frame_test_env(True)
        with patch("dino_rl.env.time.monotonic", side_effect=[10, 11.1]):
            env.step(0)
            with self.assertRaisesRegex(RuntimeError, "over one second"):
                env.step(0)

    def test_parallel_evaluation_does_not_favor_short_episodes(self):
        vector = DummyVecEnv([lambda: FixedEpisode(1), lambda: FixedEpisode(4)])
        try:
            report = rollout(vector, "no-input", 2)
            self.assertEqual(sorted(r["survival_seconds"] for r in report["records"]), [1, 4])
            self.assertEqual(report["mean_survival_seconds"], 2.5)
        finally:
            vector.close()

    def test_unseen_and_passed_obstacles_do_not_leak_into_inputs(self):
        raw = game_state()
        raw["obstacles"] = [obstacle(650), obstacle(-30)]
        obs = observation(raw, 0)
        self.assertEqual(obs.shape, (28,))
        self.assertTrue(np.all(obs[10:] == 0))
        self.assertEqual(list(obs[7:10]), [1, 0, 0])
        self.assertEqual(obs[6], 0)  # Grounded Dino cannot end a jump.

    def test_nearest_two_visible_obstacles_and_bird_speed(self):
        raw = game_state()
        bird = obstacle(400, "pterodactyl")
        bird["speedOffset"] = 0.8
        raw["obstacles"] = [bird, obstacle(500), obstacle(100)]
        obs = observation(raw, 2)
        self.assertAlmostEqual(obs[11], (100 - 14 - 44) / 600)
        self.assertEqual(list(obs[16:19]), [1, 0, 0])
        self.assertAlmostEqual(obs[24], 6.8 / 14)
        self.assertEqual(list(obs[25:28]), [0, 0, 1])

    def test_bird_counts_track_individual_birds_and_separate_height_outcomes(self):
        env = self.frame_test_env(True)
        initial_info = env._info()

        def see(*birds, crashed=False):
            env.raw["obstacles"] = [
                {**obstacle(x, "pterodactyl"), "id": bird_id, "y": y, "width": 46, "height": 40}
                for bird_id, x, y in birds
            ]
            env.raw["crashed"] = crashed
            env._track_birds()

        see((1, 650, 50))
        self.assertEqual(env.bird_metrics["50"]["encounters"], 0)
        for x in (500, 400, -40, -80):
            see((1, x, 50))
        self.assertEqual(env.bird_metrics["50"], {"encounters": 1, "passes": 1, "collisions": 0})
        self.assertEqual(initial_info["bird_metrics"]["50"]["encounters"], 0)
        see((2, 500, 75))
        see()  # A bird can leave the viewport between observations.
        see((3, 30, 100), (4, 450, 75), crashed=True)
        self.assertEqual(env.bird_metrics["100"], {"encounters": 1, "passes": 0, "collisions": 1})
        self.assertEqual(env.bird_metrics["75"], {"encounters": 2, "passes": 1, "collisions": 0})
        self.assertEqual(bird_scalars(env.bird_metrics)["pass_rate_y75"], 1)
        self.assertNotIn("pass_rate_y50", bird_scalars(empty_bird_metrics()))

    def test_resume_keeps_learning_state_and_applies_lower_exploration(self):
        torch.set_num_threads(1)
        with TemporaryDirectory() as directory:
            run = Path(directory)
            source_env = DummyVecEnv([lambda: FixedEpisode(4)])
            source = DQN(
                "MlpPolicy",
                source_env,
                buffer_size=32,
                learning_starts=0,
                batch_size=2,
                train_freq=4,
                gradient_steps=1,
                policy_kwargs={"net_arch": [8]},
                device="cpu",
                seed=7,
            )
            source.learn(8)
            budget = SimpleNamespace(tick=lambda **kw: None, summary=lambda: {})
            config = {
                "clock_mode": "native",
                "browser_mode": "headless",
                "exploration_steps": 4,
                "max_seconds": 300,
            }
            save_checkpoint(source, source_env, run, config, None, (4, 4), budget)
            (run / "baselines.json").write_text("{}")
            args = SimpleNamespace(
                workers=1,
                run=str(run),
                resume=str(run / "latest.json"),
                show_browser=False,
                seed=7,
                max_seconds=300,
                steps=12,
                checkpoint_every=4,
                eval_episodes=1,
                exploration_final=0.01,
            )
            actual_learn = DQN.learn

            def inspect_then_learn(resumed, **kwargs):
                self.assertEqual(resumed.num_timesteps, 8)
                self.assertEqual(resumed._n_updates, source._n_updates)
                self.assertEqual(resumed.replay_buffer.size(), 8)
                self.assertAlmostEqual(resumed.exploration_rate, 0.01)
                torch.testing.assert_close(resumed.policy.state_dict(), source.policy.state_dict())
                torch.testing.assert_close(
                    resumed.policy.optimizer.state_dict(), source.policy.optimizer.state_dict()
                )
                np.testing.assert_array_equal(
                    resumed.replay_buffer.observations, source.replay_buffer.observations
                )
                return actual_learn(resumed, **kwargs)

            resumed_env = DummyVecEnv([lambda: FixedEpisode(4)])
            with (
                patch("dino_rl.training.training_env", return_value=resumed_env),
                patch("dino_rl.training.CpuBudget", return_value=budget),
                patch("dino_rl.training.os.nice"),
                patch.object(DQN, "learn", inspect_then_learn),
            ):
                train(args)
            pointer = json.loads((run / "latest.json").read_text())
            restored = DQN.load(run / pointer["checkpoint"] / "model.zip", device="cpu")
            self.assertEqual(restored.num_timesteps, 12)
            self.assertEqual(restored._n_updates, source._n_updates + 1)
            self.assertAlmostEqual(restored.exploration_final_eps, 0.01)
            self.assertAlmostEqual(
                json.loads((run / "config.json").read_text())["exploration_final_eps"], 0.01
            )
            # A new run may change its cap, but cannot inherit the old selection score.
            continued_run = run / "different-cap"
            args.run = str(continued_run)
            args.max_seconds = 120
            args.steps = 16
            continued_env = DummyVecEnv([lambda: FixedEpisode(2)])
            with (
                patch("dino_rl.training.training_env", return_value=continued_env),
                patch("dino_rl.training.CpuBudget", return_value=budget),
                patch("dino_rl.training.os.nice"),
            ):
                train(args)
            selected = json.loads((continued_run / "best.json").read_text())
            selected_metadata = json.loads(
                (continued_run / selected["checkpoint"] / "metadata.json").read_text()
            )
            self.assertEqual(selected_metadata["best_quality"], [2, 2])
            self.assertEqual(selected_metadata["config"]["max_seconds"], 120)
            self.assertTrue((continued_run / "baselines.json").exists())
            self.assertEqual(json.loads((run / "config.json").read_text())["max_seconds"], 300)
            source_env.close()

    def test_corrupt_or_changed_game_state_stops_instead_of_training(self):
        raw = game_state()
        raw["speed"] = float("nan")
        with self.assertRaises(RuntimeError):
            observation(raw, 0)
        raw = game_state()
        raw["obstacles"] = [obstacle(100, "unexpected-new-obstacle")]
        with self.assertRaises(RuntimeError):
            observation(raw, 0)

    def test_extending_training_does_not_restart_exploration(self):
        for end_step in (10000, 20000, 100000):
            schedule = LinearSchedule(1.0, 0.05, 50000 / end_step)
            self.assertAlmostEqual(schedule(1 - 8000 / end_step), 0.848)

    def test_incompatible_checkpoint_is_rejected(self):
        env = SimpleNamespace(browser_version="157.test")
        saved = contract(env)
        validate_contract(saved, env)
        changed = deepcopy(saved)
        changed["chrome_version"] = "158.test"
        with self.assertRaises(ValueError):
            validate_contract(changed, env)

    def test_replay_migration_preserves_experiences_and_timeout_flags(self):
        old = ReplayBuffer(
            16,
            spaces.Box(-100, 100, shape=(28,), dtype=np.float32),
            spaces.Discrete(3),
            device="cpu",
        )
        for index in range(5):
            old.add(
                np.full((1, 28), index),
                np.full((1, 28), index + 1),
                np.array([index % 3]),
                np.array([index + 0.5]),
                np.array([index % 2 == 0]),
                [{"TimeLimit.truncated": index == 4}],
            )
        migrated, info = repartition_replay(old, 4)
        self.assertEqual(info["retained_transitions"], 5)
        self.assertEqual(info["repeated_padding_transitions"], 3)
        self.assertEqual(old.size(), 5)
        for name in (
            "observations",
            "next_observations",
            "actions",
            "rewards",
            "dones",
            "timeouts",
        ):
            before = getattr(old, name)[:5]
            before = before.reshape((5, *before.shape[2:]))
            after = getattr(migrated, name).reshape((16, *before.shape[1:]))
            np.testing.assert_array_equal(before, after[:5])
            np.testing.assert_array_equal(before[:3], after[5:8])
        unchanged, info = repartition_replay(migrated, 4)
        self.assertIs(unchanged, migrated)
        self.assertEqual(info["repeated_padding_transitions"], 0)

    def test_full_replay_can_return_to_single_worker_without_data_loss(self):
        old = ReplayBuffer(
            16,
            spaces.Box(-100, 100, shape=(28,), dtype=np.float32),
            spaces.Discrete(3),
            device="cpu",
        )
        for index in range(23):
            old.add(
                np.full((1, 28), index),
                np.full((1, 28), index + 1),
                np.array([index % 3]),
                np.array([index]),
                np.array([index % 2 == 0]),
                [{"TimeLimit.truncated": index % 5 == 0}],
            )
        parallel, _ = repartition_replay(old, 4)
        restored, _ = repartition_replay(parallel, 1)
        self.assertTrue(restored.full)
        self.assertEqual(restored.size(), 16)
        for name in (
            "observations",
            "next_observations",
            "actions",
            "rewards",
            "dones",
            "timeouts",
        ):
            values = getattr(old, name)
            expected = np.concatenate((values[old.pos :], values[: old.pos]))
            np.testing.assert_array_equal(expected, getattr(restored, name))
        restored.add(
            np.full((1, 28), 23),
            np.full((1, 28), 24),
            np.array([2]),
            np.array([23]),
            np.array([False]),
            [{}],
        )
        chronological = np.concatenate(
            (restored.observations[restored.pos :], restored.observations[: restored.pos])
        )
        np.testing.assert_array_equal(chronological[:, 0, 0], np.arange(8, 24))


if __name__ == "__main__":
    unittest.main()
