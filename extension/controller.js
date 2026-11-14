// This function runs in the native game page, not in the extension's worker.
export function installController(model, makeObservation, evaluateQ) {
  if (location.href !== "chrome-error://chromewebdata/" || typeof Runner === "undefined") {
    throw new Error("This tab is not Chrome's native offline Dino game.");
  }
  globalThis.__dinoRl?.destroy();
  const r = Runner.getInstance();
  validateGame();
  const CONNECTION_TIMEOUT_MS = 500;
  const STATUS_REFRESH_MS = 1000;
  let enabled = false;
  let previousAction = 0;
  let phase = "paused";
  let reason = "Ready. Press Resume to let the model play.";
  let lastPulse = performance.now();
  let lastPaint = 0;
  let predictions = 0;
  const keys = [null, "ArrowUp", "ArrowDown"];
  const host = document.createElement("div");
  host.id = "dino-rl-controls";
  host.style.cssText = "position:fixed;right:16px;bottom:16px;z-index:2147483647";
  const root = host.attachShadow({mode: "open"});
  const style = document.createElement("style");
  style.textContent = `
    :host { color-scheme: light dark; }
    section { font: 13px/1.45 system-ui,sans-serif; color:CanvasText; background:Canvas;
      border:1px solid GrayText; border-radius:8px; padding:12px; width:248px; }
    strong { font-size:14px; } p { margin:6px 0 10px; }
    button { font:inherit; padding:6px 12px; cursor:pointer; }
    button:focus-visible { outline:2px solid Highlight; outline-offset:3px; }
  `;
  const panel = document.createElement("section");
  panel.setAttribute("aria-label", "Dino RL controls");
  const title = document.createElement("strong");
  title.textContent = "Dino RL";
  const message = document.createElement("p");
  message.setAttribute("role", "status");
  const button = document.createElement("button");
  button.type = "button";
  panel.append(title, message, button);
  root.append(style, panel);
  document.documentElement.append(host);

  function validateGame() {
    const expected = model.game;
    if (r.dimensions.width !== expected.width || r.dimensions.height !== expected.height ||
        Object.entries(expected.runner).some(([key, value]) => r.config[key] !== value) ||
        Object.entries(expected.trex).some(([key, value]) => r.tRex.config[key] !== value)) {
      throw new Error("Unsupported game settings. Use the default Dino game in a wider window.");
    }
  }

  function readState() {
    const t = r.tRex;
    return {speed: r.currentSpeed, width: r.dimensions.width,
      trex: {x: t.xPos, y: t.yPos, groundY: t.groundYPos, vy: t.jumpVelocity,
        jumping: t.jumping, ducking: t.ducking, speedDrop: t.speedDrop,
        reachedMinHeight: t.reachedMinHeight, width: t.ducking ? t.config.widthDuck : t.config.width},
      obstacles: r.horizon.obstacles.map(o => ({x: o.xPos, y: o.yPos, width: o.width,
        height: o.typeConfig.height, type: o.typeConfig.type, speedOffset: o.speedOffset}))};
  }

  function keyEvent(type, key) {
    const keyCode = key === "ArrowUp" ? 38 : 40;
    document.dispatchEvent(new KeyboardEvent(type, {key, code: key, keyCode,
      which: keyCode, bubbles: true, cancelable: true}));
  }

  function setAction(action) {
    if (action === previousAction) return;
    if (keys[previousAction]) keyEvent("keyup", keys[previousAction]);
    if (keys[action]) keyEvent("keydown", keys[action]);
    previousAction = action;
  }

  function paint() {
    message.textContent = enabled ? `Model playing · score ${r.distanceMeter.getActualDistance(Math.ceil(r.distanceRan))}` : reason;
    button.textContent = enabled ? "Pause" : r.crashed ? "Restart with AI" : "Resume";
    button.disabled = phase === "disconnected";
    button.setAttribute("aria-pressed", String(enabled));
    lastPaint = performance.now();
  }

  function status() {
    return {phase, reason, enabled, previousAction, predictions, checkpointStep: model.checkpoint_step,
      score: r.distanceMeter.getActualDistance(Math.ceil(r.distanceRan))};
  }

  function pause(text, nextPhase = "paused") {
    enabled = false;
    setAction(0);
    phase = nextPhase;
    reason = text;
    paint();
    return status();
  }

  function resume() {
    try {
      validateGame();
      if (performance.now() - lastPulse > CONNECTION_TIMEOUT_MS) {
        return pause("Connection ended. Reopen Dino from the extension.", "disconnected");
      }
      if (document.hidden || !document.hasFocus()) return pause("Focus the game and press Resume.");
      setAction(0);
      if (r.crashed) r.restart();
      else if (!r.activated) {
        keyEvent("keydown", "ArrowUp");
        keyEvent("keyup", "ArrowUp");
      } else if (r.paused) r.play();
      // Only this explicit control changes the authorization latch to true.
      enabled = true;
      phase = "playing";
      reason = "Model playing";
      paint();
      return status();
    } catch (error) {
      return pause(error.message, "error");
    }
  }

  function humanInput(event) {
    if (!event.isTrusted) return;
    const activatesButton = event.composedPath().includes(button) &&
      (event.type === "pointerdown" ||
        (event.type === "keydown" && ["Enter", " "].includes(event.key)));
    if (!activatesButton) pause("Paused by your input. Press Resume when ready.");
  }

  function lostFocus() { pause("Paused while the game is unfocused. Press Resume when ready."); }
  function visibilityChanged() { if (document.hidden) lostFocus(); }
  const inputEvents = ["keydown", "pointerdown", "wheel"];
  for (const name of inputEvents) document.addEventListener(name, humanInput, true);
  window.addEventListener("blur", lostFocus);
  document.addEventListener("visibilitychange", visibilityChanged);
  for (const name of ["pointerdown", "click"]) {
    root.addEventListener(name, event => event.stopPropagation());
  }
  for (const name of ["keydown", "keyup"]) {
    root.addEventListener(name, event => {
      if (["Enter", " "].includes(event.key)) event.stopPropagation();
    });
  }
  button.addEventListener("click", () => {
    button.blur();
    if (enabled) pause("Paused. Press Resume when ready.");
    else resume();
  });

  const timer = setInterval(() => {
    if (performance.now() - lastPulse > CONNECTION_TIMEOUT_MS) {
      if (phase !== "disconnected") pause("Connection ended. Reopen Dino from the extension.", "disconnected");
      return;
    }
    if (!enabled) return;
    try {
      if (document.hidden || !document.hasFocus()) return lostFocus();
      if (r.crashed) return pause("Game over. Press Restart with AI to try again.", "game-over");
      if (r.paused) return pause("The game paused. Press Resume when ready.");
      if (!r.activated || r.playingIntro) return;
      validateGame();
      const q = evaluateQ(model, makeObservation(readState(), previousAction, model.normalization));
      if (q.length !== 3 || !q.every(Number.isFinite)) throw new Error("The policy returned invalid Q-values.");
      predictions++;
      setAction(q.indexOf(Math.max(...q)));
      if (performance.now() - lastPaint > STATUS_REFRESH_MS) paint();
    } catch (error) { pause(error.message, "error"); }
  }, model.step_ms);

  function destroy() {
    pause("Disconnected", "disconnected");
    clearInterval(timer);
    for (const name of inputEvents) document.removeEventListener(name, humanInput, true);
    window.removeEventListener("blur", lostFocus);
    document.removeEventListener("visibilitychange", visibilityChanged);
    host.remove();
    delete globalThis.__dinoRl;
  }

  globalThis.__dinoRl = {status, pause, resume, destroy, pulse() { lastPulse = performance.now(); return status(); }};
  // Initial authorization is synchronous with installation; no delayed worker resume can undo input.
  return resume();
}
