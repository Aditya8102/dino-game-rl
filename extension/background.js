import {installController} from "./controller.js";
import {observe, predictQ, validateModel} from "./policy.js";

const GAME_URL = "https://dino.invalid/";
const HEARTBEAT_MS = 200;
const GAME_LOAD_TIMEOUT_MS = 10000;
let session = null;
let opening = null;
let lastStatus = {phase: "disconnected", reason: "Open Dino to play with the trained model."};

async function evaluate(tabId, expression) {
  const response = await chrome.debugger.sendCommand({tabId}, "Runtime.evaluate", {
    expression, returnByValue: true,
  });
  if (response.exceptionDetails) throw new Error(response.exceptionDetails.text);
  return response.result.value;
}

async function updateStatus(status) {
  lastStatus = status;
  const badge = status.enabled ? "AI" : status.phase === "disconnected" ? "" : "II";
  await chrome.action.setBadgeText({text: badge});
}

async function disconnect(reason = "Disconnected. You control the game.") {
  const current = session;
  session = null;
  if (current) {
    clearInterval(current.heartbeat);
    if (current.attached) {
      try {
        await evaluate(current.tabId,
          "if (location.href === 'chrome-error://chromewebdata/') globalThis.__dinoRl?.destroy()");
      } catch (error) {
        console.warn("Game cleanup ended before completion:", error.message);
      }
      try {
        await chrome.debugger.detach({tabId: current.tabId});
      } catch (error) {
        console.warn("Debugger connection already ended:", error.message);
      }
    }
  }
  await updateStatus({phase: "disconnected", reason, enabled: false});
  return lastStatus;
}

async function heartbeat(current) {
  if (current.busy || session !== current) return;
  current.busy = true;
  try {
    const status = await evaluate(current.tabId, "globalThis.__dinoRl.pulse()");
    if (session === current) {
      if (status.phase === "disconnected") {
        await disconnect("Game connection timed out. Open Dino to reconnect.");
        return;
      }
      if (status.phase !== lastStatus.phase) await updateStatus(status);
      else lastStatus = status;
    }
  } catch (error) {
    if (session === current) await disconnect(`Game connection ended: ${error.message}`);
  } finally {
    current.busy = false;
  }
}

async function pauseForFocusChange() {
  const current = session;
  if (!current?.ready) return;
  try {
    const status = await evaluate(current.tabId,
      "globalThis.__dinoRl.pause('Paused while the game is unfocused. Press Resume when ready.')");
    if (session === current) await updateStatus(status);
  } catch (error) {
    if (session === current) await disconnect(`Game connection ended: ${error.message}`);
  }
}

async function createGame() {
  const response = await fetch(chrome.runtime.getURL("model.json"));
  if (!response.ok) throw new Error("The bundled Dino RL model could not be loaded.");
  const model = await response.json();
  validateModel(model);
  const tab = await chrome.tabs.create({url: "about:blank", active: true});
  const current = {tabId: tab.id, windowId: tab.windowId, attached: false, ready: false, busy: false};
  session = current;
  try {
    await chrome.debugger.attach({tabId: tab.id}, "1.3");
    current.attached = true;
    await chrome.debugger.sendCommand({tabId: tab.id}, "Network.enable");
    await chrome.debugger.sendCommand({tabId: tab.id}, "Network.emulateNetworkConditions", {
      offline: true, latency: 0, downloadThroughput: -1, uploadThroughput: -1,
    });
    await chrome.debugger.sendCommand({tabId: tab.id}, "Page.navigate", {url: GAME_URL});
    const deadline = Date.now() + GAME_LOAD_TIMEOUT_MS;
    while (!await evaluate(tab.id, "typeof Runner !== 'undefined' && Boolean(Runner.getInstance().tRex)")) {
      if (Date.now() > deadline) throw new Error("Chrome did not load its native offline Dino game.");
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    const injection = `(${installController.toString()})(${JSON.stringify(model)},${observe.toString()},${predictQ.toString()})`;
    const status = await evaluate(tab.id, injection);
    current.ready = true;
    current.heartbeat = setInterval(() => heartbeat(current), HEARTBEAT_MS);
    await updateStatus(status);
    return status;
  } catch (error) {
    await disconnect(`Could not open Dino: ${error.message}`);
    throw error;
  }
}

async function openGame() {
  if (opening) return opening;
  if (session?.ready) {
    const tab = await chrome.tabs.update(session.tabId, {active: true});
    await chrome.windows.update(tab.windowId, {focused: true});
    return lastStatus;
  }
  opening = createGame();
  try {
    return await opening;
  } finally {
    opening = null;
  }
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (sender.id !== chrome.runtime.id) return false;
  const action = message?.action;
  const operation = action === "open" ? openGame()
    : action === "disconnect" ? Promise.resolve(opening).then(() => disconnect())
      : action === "status" ? Promise.resolve({...lastStatus, gameTabId: session?.tabId ?? null})
        : Promise.reject(new Error("Unknown Dino RL command."));
  operation.then(status => sendResponse({ok: true, status}),
    error => sendResponse({ok: false, error: error.message}));
  return true;
});

chrome.debugger.onDetach.addListener(source => {
  if (source.tabId !== session?.tabId) return;
  clearInterval(session.heartbeat);
  session = null;
  updateStatus({phase: "disconnected", reason: "Debugger connection ended. Reopen Dino to reconnect.", enabled: false});
});

chrome.tabs.onRemoved.addListener(tabId => {
  if (tabId === session?.tabId) disconnect("The Dino tab was closed.");
});

chrome.tabs.onActivated.addListener(active => {
  if (active.windowId === session?.windowId && active.tabId !== session?.tabId) pauseForFocusChange();
});

chrome.windows.onFocusChanged.addListener(windowId => {
  if (session && windowId !== session.windowId) pauseForFocusChange();
});

chrome.tabs.onUpdated.addListener((tabId, change) => {
  if (tabId === session?.tabId && session.ready && change.status === "loading") {
    disconnect("The game tab navigated away. Open Dino to start a new session.");
  }
});
