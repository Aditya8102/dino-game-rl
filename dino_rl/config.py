"""The fixed game, observation and vanilla DQN configuration."""

SCHEMA = "dino-state-28-v1"
STEP_MS = 50
ACTIONS = (None, "ArrowUp", "ArrowDown")
GAME_DIMENSIONS = (600, 150)
GAME_CONFIG = {"speed": 6, "maxSpeed": 13, "acceleration": 0.001, "clearTime": 3000}
TREX_CONFIG = {"width": 44, "widthDuck": 59, "gravity": 0.6, "initialJumpVelocity": -10}
OBSERVATION_SCALES = {"speed": 14, "height": 150, "velocity": 20, "distance": 600}
NETWORK_SHAPE = (28, 64, 64, 3)
DQN_DEFAULTS = {
    "format_version": 1,
    "algorithm": "DQN",
    "clock_mode": "native",
    "learning_rate": 0.0001,
    "gamma": 0.995,
    "batch_size": 64,
    "buffer_size": 100000,
    "learning_starts": 5000,
    "train_freq": 4,
    "target_update_interval": 1000,
    "exploration_steps": 50000,
}
