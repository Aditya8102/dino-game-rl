"""Gymnasium environment backed by Chrome's own chrome://dino game."""

import math
import time

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from playwright.sync_api import sync_playwright

from .config import (
    ACTIONS,
    GAME_CONFIG,
    GAME_DIMENSIONS,
    OBSERVATION_SCALES,
    SCHEMA,
    STEP_MS,
    TREX_CONFIG,
)

READ_STATE = """() => {
  const r = Runner.getInstance(), t = r.tRex;
  return {
    playing: r.playing, paused: r.paused, crashed: r.crashed,
    intro: r.playingIntro, activated: r.activated,
    speed: r.currentSpeed, distance: r.distanceRan, runningMs: r.runningTime,
    clock: performance.now(), width: r.dimensions.width, height: r.dimensions.height,
    trex: {x: t.xPos, y: t.yPos, groundY: t.groundYPos, vy: t.jumpVelocity,
           jumping: t.jumping, ducking: t.ducking, speedDrop: t.speedDrop,
           reachedMinHeight: t.reachedMinHeight,
           width: t.ducking ? t.config.widthDuck : t.config.width},
    obstacles: r.horizon.obstacles.map(o => {
      if (!window.__dinoObstacleIds.has(o)) {
        window.__dinoObstacleIds.set(o, window.__dinoNextObstacleId++);
      }
      return {id: window.__dinoObstacleIds.get(o),
              x: o.xPos, y: o.yPos, width: o.width, height: o.typeConfig.height,
              type: o.typeConfig.type, speedOffset: o.speedOffset};
    }),
    score: r.distanceMeter.getActualDistance(Math.ceil(r.distanceRan)),
    config: {speed: r.config.speed, maxSpeed: r.config.maxSpeed,
             acceleration: r.config.acceleration, clearTime: r.config.clearTime},
    hidden: document.visibilityState !== 'visible'
  };
}"""


def empty_bird_metrics() -> dict[str, dict[str, int]]:
    """Return independent per-height encounter, pass and collision counters."""
    return {str(y): {"encounters": 0, "passes": 0, "collisions": 0} for y in (50, 75, 100)}


def observation(raw: dict, previous_action: int) -> np.ndarray:
    """Only visible, unpassed obstacles enter the model's input."""
    t = raw["trex"]
    scales = OBSERVATION_SCALES
    values = [
        raw["speed"] / scales["speed"],
        (t["groundY"] - t["y"]) / scales["height"],
        t["vy"] / scales["velocity"],
        float(t["jumping"]),
        float(t["ducking"]),
        float(t["speedDrop"]),
        float(t["jumping"] and t["reachedMinHeight"]),
        *[float(previous_action == action) for action in range(3)],
    ]
    obstacles = sorted(
        (o for o in raw["obstacles"] if o["x"] < raw["width"] and o["x"] + o["width"] >= t["x"]),
        key=lambda o: o["x"],
    )
    types = ("cactusSmall", "cactusLarge", "pterodactyl")
    for index in range(2):
        if index >= len(obstacles):
            values.extend([0.0] * 9)
            continue
        o = obstacles[index]
        if o["type"] not in types:
            raise RuntimeError(f"Unsupported Dino obstacle: {o['type']}")
        values.extend(
            [
                1.0,
                (o["x"] - t["x"] - t["width"]) / scales["distance"],
                o["y"] / scales["height"],
                o["width"] / scales["distance"],
                o["height"] / scales["height"],
                (raw["speed"] + o["speedOffset"]) / scales["speed"],
                *[float(o["type"] == name) for name in types],
            ]
        )
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (28,) or not np.isfinite(result).all():
        raise RuntimeError("Chrome returned an invalid observation; training was stopped.")
    return result


class DinoEnv(gym.Env):
    """Expose the native Dino game as numeric observations and held-key actions."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        *,
        show_browser: bool = False,
        max_seconds: float = 300.0,
    ) -> None:
        super().__init__()
        if not math.isfinite(max_seconds) or max_seconds <= 0:
            raise ValueError("Episode limit must be a positive number of seconds.")
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(28,), dtype=np.float32)
        self.render_mode = "human" if show_browser else None
        self.max_seconds = max_seconds
        self.previous_action = 0
        self._ended = True
        self._closed = False
        self.pw = sync_playwright().start()
        self.browser = None
        try:
            self.browser = self.pw.chromium.launch(channel="chrome", headless=not show_browser)
            self.page = self.browser.new_page(
                viewport={"width": 1000, "height": 700}, device_scale_factor=1
            )
            self.cdp = self.page.context.new_cdp_session(self.page)
            navigation = self.cdp.send("Page.navigate", {"url": "chrome://dino/"})
            if navigation.get("errorText") not in (None, "net::ERR_INTERNET_DISCONNECTED"):
                raise RuntimeError(f"Dino navigation failed: {navigation}")
            self.page.wait_for_selector("canvas.runner-canvas", state="attached", timeout=10000)
            self.page.wait_for_function(
                "() => typeof Runner !== 'undefined' && !!Runner.getInstance().tRex"
            )
            # Stable diagnostic IDs do not enter observations or modify game objects.
            self.page.evaluate(
                "() => { window.__dinoObstacleIds = new WeakMap(); window.__dinoNextObstacleId = 0; }"
            )
            self.page.keyboard.press("ArrowUp")
            self.page.wait_for_function(
                "() => { const r = Runner.getInstance(); return r.activated && !r.playingIntro; }",
                timeout=10000,
            )
            self.raw = self._read()
            self._validate_game()
            self._boundary = time.monotonic()
            # Let the real game create its game-over UI through a natural collision.
            deadline = time.monotonic() + 20
            while not self.raw["crashed"]:
                if time.monotonic() > deadline:
                    raise RuntimeError("Dino did not reach its first collision during setup.")
                self._advance()
                self.raw = self._read()
            self.browser_version = self.browser.version
        except BaseException:
            self.close()
            raise

    def _read(self) -> dict:
        return self.page.evaluate(READ_STATE)

    def _validate_game(self) -> None:
        r = self.raw
        if (r["width"], r["height"]) != GAME_DIMENSIONS:
            raise RuntimeError(f"Expected a 600 × 150 Dino game, got {r['width']} × {r['height']}.")
        if r["config"] != GAME_CONFIG:
            raise RuntimeError(f"Unsupported game configuration: {r['config']}")
        trex_config = self.page.evaluate("""() => {
          const config = Runner.getInstance().tRex.config;
          return {width: config.width, widthDuck: config.widthDuck,
            gravity: config.gravity, initialJumpVelocity: config.initialJumpVelocity};
        }""")
        if trex_config != TREX_CONFIG:
            raise RuntimeError(f"Unsupported dinosaur configuration: {trex_config}")
        if not 0 <= r["runningMs"] < 10000:
            raise RuntimeError("Dino's setup clock is outside its expected range.")
        observation(r, 0)

    def _advance(self) -> None:
        remaining = self._boundary + STEP_MS / 1000 - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        self._boundary = time.monotonic()

    def _set_action(self, action: int) -> None:
        if action == self.previous_action:
            return
        old_key, new_key = ACTIONS[self.previous_action], ACTIONS[action]
        if old_key:
            self.page.keyboard.up(old_key)
        if new_key:
            self.page.keyboard.down(new_key)
        self.previous_action = action

    def _track_birds(self) -> None:
        birds = {o["id"]: o for o in self.raw["obstacles"] if o["type"] == "pterodactyl"}
        for bird_id, bird in birds.items():
            if (
                bird["x"] < self.raw["width"]
                and bird["x"] + bird["width"] >= self.raw["trex"]["x"]
                and bird_id not in self._birds
            ):
                self._birds[bird_id] = str(bird["y"])
                self.bird_metrics[str(bird["y"])]["encounters"] += 1
        # Chrome tests collision against the first obstacle in its queue.
        first = self.raw["obstacles"][0] if self.raw["obstacles"] else None
        collision_id = (
            first["id"]
            if self.raw["crashed"] and first and first["type"] == "pterodactyl"
            else None
        )
        for bird_id, height in list(self._birds.items()):
            bird = birds.get(bird_id)
            if bird_id == collision_id:
                self.bird_metrics[height]["collisions"] += 1
            elif bird is None or bird["x"] + bird["width"] < self.raw["trex"]["x"]:
                self.bird_metrics[height]["passes"] += 1
            else:
                continue
            del self._birds[bird_id]

    def _info(self) -> dict:
        return {
            "survival_seconds": self.elapsed,
            "score": self.raw["score"],
            "distance": self.raw["distance"],
            "game_speed": self.raw["speed"],
            "crashed": self.raw["crashed"],
            "schema": SCHEMA,
            "frame_stalls": self.frame_stalls,
            "bird_metrics": {height: dict(counts) for height, counts in self.bird_metrics.items()},
        }

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        """Start an ordinary new attempt and clear episode-only counters."""
        super().reset(seed=seed)
        # Chrome keeps its native Math.random(); Gymnasium's seed governs only our RNG.
        self._set_action(0)
        self.page.evaluate("() => { const r = Runner.getInstance(); r.stop(); r.restart(); }")
        self.raw = self._read()
        if not self.raw["playing"] or self.raw["crashed"] or self.raw["intro"]:
            raise RuntimeError("Dino did not enter a clean running state after reset.")
        self.elapsed = 0.0
        self.frame_stalls = 0
        self._birds = {}
        self.bird_metrics = empty_bird_metrics()
        self._no_progress_since = None
        self._boundary = time.monotonic()
        self._ended = False
        return observation(self.raw, 0), self._info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Hold the chosen key state for one decision interval and score elapsed game time."""
        if self._ended:
            raise RuntimeError("Call reset() before stepping a finished episode.")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        previous = self.raw
        self._set_action(int(action))
        self._advance()
        self.raw = self._read()
        if self.raw["hidden"] or (self.raw["paused"] and not self.raw["crashed"]):
            raise RuntimeError(
                "The game lost visibility or paused unexpectedly. No transition was recorded."
            )
        elapsed = max(0.0, (self.raw["runningMs"] - previous["runningMs"]) / 1000)
        if not self.raw["crashed"] and elapsed <= 0:
            # Native rendering can miss a 50 ms decision interval without pausing.
            self.frame_stalls += 1
            now = time.monotonic()
            if self._no_progress_since is None:
                self._no_progress_since = now
            elif now - self._no_progress_since > 1.0:
                raise RuntimeError("The native game has not rendered for over one second.")
        else:
            self._no_progress_since = None
        self.elapsed += elapsed
        terminated = bool(self.raw["crashed"])
        truncated = bool(not terminated and self.elapsed >= self.max_seconds)
        reward = float(elapsed - 10.0 * terminated)
        self._track_birds()
        self._ended = terminated or truncated
        return (
            observation(self.raw, self.previous_action),
            reward,
            terminated,
            truncated,
            self._info(),
        )

    def close(self) -> None:
        """Release the owned browser and Playwright connection exactly once."""
        if self._closed:
            return
        self._closed = True
        try:
            if self.browser:
                self.browser.close()
        finally:
            self.pw.stop()
