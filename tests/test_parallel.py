"""Regression tests for failures while creating or stepping parallel workers."""

import unittest
from functools import partial
from unittest.mock import patch

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from dino_rl.parallel import ManagedSubprocVecEnv


class WorkerTestEnv(gym.Env):
    observation_space = spaces.Box(-1, 1, (1,), dtype=np.float32)
    action_space = spaces.Discrete(2)

    def __init__(self, fail_start=False, fail_step=False):
        if fail_start:
            raise ValueError("Expected worker startup failure")
        self.fail_step = fail_step

    def reset(self, *, seed=None, options=None):
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        if self.fail_step:
            raise ValueError("Expected worker step failure")
        return np.zeros(1, dtype=np.float32), 0.0, False, False, {}


class ParallelCleanupTests(unittest.TestCase):
    def test_one_worker_failure_does_not_leave_the_other_worker_running(self):
        vector = ManagedSubprocVecEnv([partial(WorkerTestEnv, fail_step=True), WorkerTestEnv])
        try:
            vector.reset()
            with self.assertRaises((EOFError, ConnectionError)):
                vector.step(np.zeros(2, dtype=int))
        finally:
            vector.close()
        self.assertTrue(all(not process.is_alive() for process in vector.processes))
        vector.close()  # Closing an already-closed vector must be safe.

    def test_partial_startup_failure_closes_every_started_worker(self):
        processes = []
        close = ManagedSubprocVecEnv.close

        def record_and_close(vector):
            processes.extend(vector.processes)
            close(vector)

        with patch.object(ManagedSubprocVecEnv, "close", record_and_close):
            with self.assertRaises((EOFError, ConnectionError)):
                ManagedSubprocVecEnv([partial(WorkerTestEnv, fail_start=True), WorkerTestEnv])
        self.assertEqual(len(processes), 2)
        self.assertTrue(all(not process.is_alive() for process in processes))


if __name__ == "__main__":
    unittest.main()
