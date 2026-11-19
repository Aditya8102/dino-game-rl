"""Export the project's DQN as a small, dependency-free browser policy."""

import argparse
import hashlib
import json
from pathlib import Path

from stable_baselines3 import DQN
from torch import nn

from dino_rl.checkpoints import resolve_checkpoint
from dino_rl.config import (
    ACTIONS,
    GAME_CONFIG,
    GAME_DIMENSIONS,
    NETWORK_SHAPE,
    OBSERVATION_SCALES,
    SCHEMA,
    STEP_MS,
    TREX_CONFIG,
)


def export_policy(checkpoint: str | Path, output: str | Path) -> dict:
    """Export only the supported network and its unchanged observation contract."""
    directory, metadata = resolve_checkpoint(checkpoint)
    expected = {"schema": SCHEMA, "step_ms": STEP_MS, "actions": list(ACTIONS)}
    if any(metadata["contract"].get(key) != value for key, value in expected.items()):
        raise ValueError("The checkpoint input/action contract differs from the browser policy.")
    model = DQN.load(directory / "model.zip", device="cpu")
    network = list(model.q_net.q_net)
    if (
        model.observation_space.shape != (28,)
        or model.action_space.n != 3
        or [type(layer) for layer in network] != [nn.Linear, nn.ReLU, nn.Linear, nn.ReLU, nn.Linear]
        or [
            (layer.in_features, layer.out_features)
            for layer in network
            if isinstance(layer, nn.Linear)
        ]
        != list(zip(NETWORK_SHAPE, NETWORK_SHAPE[1:]))
    ):
        raise ValueError("Browser export supports only this project's 28-64-64-3 ReLU DQN.")
    layers = [
        {
            "weights": layer.weight.detach().cpu().tolist(),
            "bias": layer.bias.detach().cpu().tolist(),
            "activation": "linear" if index == 4 else "relu",
        }
        for index, layer in enumerate(network)
        if isinstance(layer, nn.Linear)
    ]
    result = {
        "format_version": 1,
        "algorithm": "DQN",
        "schema": SCHEMA,
        "step_ms": STEP_MS,
        "checkpoint_step": metadata["steps"],
        "source_sha256": hashlib.sha256((directory / "model.zip").read_bytes()).hexdigest(),
        "actions": ["release", "jump", "duck"],
        "input_size": NETWORK_SHAPE[0],
        "normalization": OBSERVATION_SCALES,
        "game": {
            "width": GAME_DIMENSIONS[0],
            "height": GAME_DIMENSIONS[1],
            "runner": GAME_CONFIG,
            "trex": TREX_CONFIG,
        },
        "layers": layers,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="models/best.json")
    parser.add_argument("--output", default="extension/model.json")
    args = parser.parse_args()
    result = export_policy(args.checkpoint, args.output)
    print(f"Exported DQN at step {result['checkpoint_step']} to {args.output}")
