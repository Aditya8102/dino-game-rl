"""Opt-in lifecycle checks using the actual installed Chrome extension and game."""

import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


class OnlineHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"normal tab remains online")

    def log_message(self, format, *args):
        return


@unittest.skipUnless(
    os.environ.get("DINO_BROWSER_TESTS") == "1", "Set DINO_BROWSER_TESTS=1 for native Chrome tests"
)
class ExtensionBrowserTests(unittest.TestCase):
    def setUp(self):
        self.profile = TemporaryDirectory(prefix="dino-extension-test-")
        self.addCleanup(self.profile.cleanup)
        self.playwright = sync_playwright().start()
        self.addCleanup(self.playwright.stop)
        self.context = self.playwright.chromium.launch_persistent_context(
            self.profile.name,
            channel="chrome",
            headless=True,
            ignore_default_args=["--disable-extensions"],
            args=["--enable-unsafe-extension-debugging"],
            viewport={"width": 1100, "height": 800},
        )
        self.addCleanup(self.context.close)
        cdp = self.context.browser.new_browser_cdp_session()
        extension_id = cdp.send("Extensions.loadUnpacked", {"path": str(ROOT / "extension")})["id"]
        self.popup = self.context.new_page()
        self.popup.goto(f"chrome-extension://{extension_id}/popup.html")

    def command(self, action):
        response = self.popup.evaluate("action => chrome.runtime.sendMessage({action})", action)
        self.assertTrue(response["ok"], response)
        return response["status"]

    def open_game(self):
        with self.context.expect_page() as created:
            self.command("open")
        game = created.value
        game.bring_to_front()
        game.wait_for_function("() => Boolean(globalThis.__dinoRl)")
        self.assertEqual(game.url, "chrome-error://chromewebdata/")
        return game

    def resume(self, game):
        game.bring_to_front()
        button = game.locator("#dino-rl-controls").get_by_role("button")
        if game.evaluate("!__dinoRl.status().enabled"):
            button.click()
        game.wait_for_function("() => __dinoRl.status().enabled")

    def assert_paused(self, game):
        game.wait_for_function("() => !__dinoRl.status().enabled")
        before = game.evaluate("__dinoRl.status()")
        self.assertEqual(before["previousAction"], 0)
        game.wait_for_timeout(650)
        after = game.evaluate("__dinoRl.status()")
        self.assertFalse(after["enabled"])
        self.assertEqual(after["predictions"], before["predictions"])

    def test_native_autoplay_manual_takeover_focus_and_compatibility(self):
        game = self.open_game()
        self.resume(game)
        game.wait_for_function("() => __dinoRl.status().predictions >= 5", timeout=15000)
        config = game.evaluate(
            "({speed:Runner.getInstance().config.speed, acceleration:Runner.getInstance().config.acceleration})"
        )
        self.assertEqual(config, {"speed": 6, "acceleration": 0.001})
        game.keyboard.press("ArrowDown")
        self.assert_paused(game)
        self.resume(game)
        game.locator("#dino-rl-controls").get_by_role("button").focus()
        game.keyboard.press("ArrowUp")
        self.assert_paused(game)
        self.resume(game)
        game.mouse.click(20, 20)
        self.assert_paused(game)
        self.resume(game)
        game.mouse.wheel(0, 20)
        self.assert_paused(game)
        self.resume(game)
        other = self.context.new_page()
        other.bring_to_front()
        self.assert_paused(game)
        game.bring_to_front()
        self.assert_paused(game)
        other.close()
        self.resume(game)
        artifact = ROOT / "runs/extension-tests"
        artifact.mkdir(parents=True, exist_ok=True)
        game.screenshot(path=str(artifact / "game-playing.png"))
        game.set_viewport_size({"width": 400, "height": 700})
        game.wait_for_function("() => Runner.getInstance().dimensions.width < 600")
        # Resizing can first pause the native game; explicit resume must reject its new dimensions.
        game.locator("#dino-rl-controls").get_by_role("button").click()
        game.wait_for_function("() => __dinoRl.status().phase === 'error'")
        self.assert_paused(game)

    def test_disconnect_navigation_and_tab_scoped_offline_mode(self):
        game = self.open_game()
        server = ThreadingHTTPServer(("127.0.0.1", 0), OnlineHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            other = self.context.new_page()
            response = other.goto(f"http://127.0.0.1:{server.server_port}/")
            self.assertEqual(response.status, 200)
            self.assertIn("normal tab remains online", other.inner_text("body"))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)
        self.command("disconnect")
        game.wait_for_function("() => !globalThis.__dinoRl")
        self.assertEqual(game.locator("#dino-rl-controls").count(), 0)
        game = self.open_game()
        game.goto("about:blank")
        self.popup.wait_for_function(
            "async () => (await chrome.runtime.sendMessage({action:'status'})).status.phase === 'disconnected'"
        )
        self.assertEqual(self.command("status")["phase"], "disconnected")

    def test_debugger_cancellation_stops_predictions_and_releases_input(self):
        game = self.open_game()
        self.resume(game)
        game.wait_for_function("() => __dinoRl.status().predictions >= 3", timeout=15000)
        tab_id = self.command("status")["gameTabId"]
        self.popup.evaluate("tabId => chrome.debugger.detach({tabId})", tab_id)
        game.wait_for_function("() => __dinoRl.status().phase === 'disconnected'", timeout=3000)
        self.assert_paused(game)
        self.assertEqual(self.command("status")["phase"], "disconnected")
        self.assertTrue(game.locator("#dino-rl-controls").get_by_role("button").is_disabled())
        game.evaluate("__dinoRl.resume()")
        self.assert_paused(game)

    def test_manual_input_during_opening_cannot_be_undone_by_a_late_resume(self):
        worker = self.context.service_workers[0]
        worker.evaluate("""() => {
          const original = chrome.debugger.sendCommand;
          chrome.debugger.sendCommand = async (...args) => {
            const result = await original(...args);
            if (args[1] === 'Runtime.evaluate' &&
                args[2].expression.includes('function installController(')) {
              await new Promise(resolve => { globalThis.releaseInstallation = resolve; });
            }
            return result;
          };
        }""")
        with self.context.expect_page() as created:
            self.popup.evaluate(
                "() => { window.openResult = chrome.runtime.sendMessage({action:'open'}); }"
            )
        game = created.value
        game.bring_to_front()
        game.wait_for_function("() => Boolean(globalThis.__dinoRl)")
        game.keyboard.press("ArrowDown")
        self.assertFalse(game.evaluate("__dinoRl.status().enabled"))
        worker.evaluate("() => globalThis.releaseInstallation()")
        self.popup.wait_for_function("async () => (await window.openResult).ok")
        self.assert_paused(game)

    def test_expired_heartbeat_closes_session_without_reenabling_control(self):
        worker = self.context.service_workers[0]
        worker.evaluate("""() => {
          const original = chrome.debugger.sendCommand;
          chrome.debugger.sendCommand = async (...args) => {
            if (args[1] === 'Runtime.evaluate' &&
                args[2].expression === 'globalThis.__dinoRl.pulse()') {
              await new Promise(resolve => { globalThis.releaseHeartbeat = resolve; });
            }
            return original(...args);
          };
        }""")
        game = self.open_game()
        game.wait_for_function("() => __dinoRl.status().phase === 'disconnected'", timeout=3000)
        self.assertFalse(game.evaluate("__dinoRl.status().enabled"))
        worker.evaluate("() => globalThis.releaseHeartbeat()")
        game.wait_for_function("() => !globalThis.__dinoRl")
        self.assertEqual(self.command("status")["phase"], "disconnected")


if __name__ == "__main__":
    unittest.main()
