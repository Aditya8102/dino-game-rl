const status = document.querySelector("#status");
const open = document.querySelector("#open");
const disconnect = document.querySelector("#disconnect");

async function command(action) {
  const response = await chrome.runtime.sendMessage({action});
  if (!response.ok) throw new Error(response.error);
  return response.status;
}

function render(state) {
  status.textContent = state.reason;
  const connected = state.phase !== "disconnected";
  open.textContent = connected ? "Return to Dino" : "Open Dino & play";
  disconnect.disabled = !connected;
}

open.addEventListener("click", async () => {
  open.disabled = true;
  status.textContent = "Opening Chrome’s offline game…";
  try {
    await command("open");
    window.close();
  } catch (error) {
    status.textContent = error.message;
    open.disabled = false;
  }
});

disconnect.addEventListener("click", async () => {
  disconnect.disabled = true;
  try { render(await command("disconnect")); }
  catch (error) { status.textContent = error.message; }
});

command("status").then(render, error => { status.textContent = error.message; });
