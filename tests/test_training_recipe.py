"""Check the prepared run through the real CLI without starting any games."""

import unittest
from pathlib import Path
from unittest.mock import patch

from dino_rl import __main__ as cli

RECIPE = Path(__file__).resolve().parents[1] / "configs/scratch-5m.args"


class TrainingRecipeTests(unittest.TestCase):
    def parse_recipe(self, *extra):
        with (
            patch("sys.argv", ["dino", f"@{RECIPE}", *extra]),
            patch.object(cli, "train") as train,
            patch.object(cli.torch, "set_num_threads"),
            patch.object(cli.torch, "set_num_interop_threads"),
        ):
            cli.main()
        return train.call_args.args[0]

    def test_recipe_starts_a_separate_fresh_five_million_step_run(self):
        args = self.parse_recipe()
        self.assertEqual(args.run, "runs/dqn-scratch-5m")
        self.assertEqual(args.steps, 5_000_000)
        self.assertEqual(args.workers, 8)
        self.assertEqual(args.seed, 7)
        self.assertEqual(args.checkpoint_every, 50_000)
        self.assertEqual(args.eval_episodes, 8)
        self.assertEqual(args.max_seconds, 300)
        self.assertEqual(args.exploration_final, 0.01)
        self.assertFalse(args.show_browser)
        self.assertIsNone(args.resume)
        self.assertIsNone(args.initialize_from)

    def test_resume_is_explicit_and_keeps_the_five_million_total_target(self):
        checkpoint = "runs/dqn-scratch-5m/latest.json"
        args = self.parse_recipe("--resume", checkpoint)
        self.assertEqual(args.resume, checkpoint)
        self.assertEqual(args.steps, 5_000_000)
        self.assertIsNone(args.initialize_from)
