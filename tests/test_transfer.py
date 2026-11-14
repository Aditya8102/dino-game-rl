"""Transfer diagnostics must report very short capped games without a crash."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from scripts.evaluate_transfer import run_mode


class CappedTestEnv:
    def __init__(self, decisions):
        self.decisions = decisions
        self.raw = {"speed": 6, "obstacles": []}
        self.closed = False

    def reset(self):
        self.steps = 0
        return np.zeros(28, dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        info = {"survival_seconds": float(self.steps), "score": self.steps}
        return np.zeros(28, dtype=np.float32), 1.0, False, self.steps == self.decisions, info

    def close(self):
        self.closed = True


class TransferDiagnosticsTests(unittest.TestCase):
    def test_capped_single_decision_and_multiple_decisions(self):
        model = SimpleNamespace(
            predict=lambda observation, deterministic: (0, None),
            policy=SimpleNamespace(obs_to_tensor=lambda observation: (torch.zeros(1, 28), False)),
            q_net=lambda tensor: torch.zeros(1, 3),
        )
        args = SimpleNamespace(max_seconds=1, contract={}, episodes=1)
        for decisions in (1, 3):
            with self.subTest(decisions=decisions), TemporaryDirectory() as directory:
                environment = CappedTestEnv(decisions)
                with (
                    patch("scripts.evaluate_transfer.DinoEnv", return_value=environment),
                    patch("scripts.evaluate_transfer.validate_contract"),
                ):
                    report = run_mode(model, "native", args, Path(directory))
                self.assertTrue(environment.closed)
                self.assertEqual(report["time_limit_completions"], 1)
                self.assertEqual(report["records"][0]["length"], decisions)
                if decisions == 1:
                    self.assertIsNone(report["decision_interval_p95_ms"])
                    self.assertIsNone(report["records"][0]["decision_interval_p95_ms"])
                else:
                    self.assertGreaterEqual(report["decision_interval_p95_ms"], 0)


if __name__ == "__main__":
    unittest.main()
