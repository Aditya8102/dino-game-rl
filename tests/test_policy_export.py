"""Cross-runtime parity for the actual released policy, not a surrogate model."""

import json
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from stable_baselines3 import DQN

from dino_rl.checkpoints import resolve_checkpoint
from dino_rl.env import observation
from scripts.export_policy import export_policy

ROOT = Path(__file__).resolve().parents[1]


class PolicyExportTests(unittest.TestCase):
    def test_export_is_deterministic_and_browser_predictions_match_pytorch(self):
        torch.set_num_threads(1)
        checkpoint, _ = resolve_checkpoint(ROOT / "models/best.json")
        policy = DQN.load(checkpoint / "model.zip", device="cpu")
        random = np.random.default_rng(17)
        cases = []
        for index in range(500):
            jumping = index % 3 == 0
            ducking = not jumping and index % 4 == 0
            obstacles = []
            for obstacle_index in range(index % 4):
                kind = ["cactusSmall", "cactusLarge", "pterodactyl"][obstacle_index % 3]
                obstacles.append(
                    {
                        "x": int(random.integers(-100, 750)),
                        "y": int(random.choice([50, 75, 100])) if kind == "pterodactyl" else 90,
                        "width": 46 if kind == "pterodactyl" else 25,
                        "height": 40 if kind == "pterodactyl" else 50,
                        "type": kind,
                        "speedOffset": float(random.choice([-0.8, 0.8]))
                        if kind == "pterodactyl"
                        else 0,
                    }
                )
            raw = {
                "speed": float(random.uniform(6, 13)),
                "width": 600,
                "trex": {
                    "x": 14,
                    "y": float(random.uniform(0, 92)) if jumping else 93,
                    "groundY": 93,
                    "vy": float(random.uniform(-10, 12)) if jumping else 0,
                    "jumping": jumping,
                    "ducking": ducking,
                    "speedDrop": jumping and index % 5 == 0,
                    "reachedMinHeight": index % 2 == 0,
                    "width": 59 if ducking else 44,
                },
                "obstacles": obstacles,
            }
            cases.append({"raw": raw, "action": index % 3})
        expected_observations = np.stack(
            [observation(case["raw"], case["action"]) for case in cases]
        )
        with torch.no_grad():
            expected_q = policy.q_net(torch.as_tensor(expected_observations)).numpy()
        with TemporaryDirectory() as directory:
            first, second = Path(directory) / "first.json", Path(directory) / "second.json"
            exported = export_policy(ROOT / "models/best.json", first)
            export_policy(ROOT / "models/best.json", second)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(first.read_bytes(), (ROOT / "extension/model.json").read_bytes())
            script = """
                import {readFileSync} from 'node:fs';
                import {observe, predictQ, validateModel} from './extension/policy.js';
                const {model, cases} = JSON.parse(readFileSync(0, 'utf8'));
                validateModel(model);
                console.log(JSON.stringify(cases.map(item => {
                    const observation = observe(item.raw, item.action, model.normalization);
                    return {observation: Array.from(observation), q: predictQ(model, observation)};
                })));
            """
            completed = subprocess.run(
                ["node", "--input-type=module", "-e", script],
                input=json.dumps({"model": exported, "cases": cases}),
                text=True,
                capture_output=True,
                check=True,
                cwd=ROOT,
                timeout=30,
            )
        actual = json.loads(completed.stdout)
        np.testing.assert_array_equal(
            np.asarray([item["observation"] for item in actual], dtype=np.float32),
            expected_observations,
        )
        actual_q = np.asarray([item["q"] for item in actual], dtype=np.float32)
        np.testing.assert_allclose(actual_q, expected_q, rtol=2e-5, atol=2e-5)
        np.testing.assert_array_equal(actual_q.argmax(axis=1), expected_q.argmax(axis=1))


if __name__ == "__main__":
    unittest.main()
