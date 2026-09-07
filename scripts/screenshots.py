"""Regenerates the screenshots the README embeds.

    uv run python scripts/screenshots.py

Needs the servers, the backend, the demo corpus and the dev server:

    uv run python scripts/start_servers.py
    uv run uvicorn app.main:app --app-dir backend --port 8000
    uv run python scripts/load_demo.py --reset
    npm --prefix frontend run dev

A script rather than a folder of PNGs somebody once took by hand: the interface
changes, and a screenshot nobody can regenerate is a screenshot that quietly
starts lying. Re-run this after a UI change and the README is current again.

It drives Chrome over the DevTools Protocol, using the `websockets` package the
backend already depends on. Playwright would do the same thing and cost a 150 MB
browser download for a task Chrome can already do.

Shot at 4:3 and 1120 CSS pixels wide, at 2x device scale: the narrow viewport
keeps the layout from spreading out, so text and controls read large in the
image, and 2x keeps them sharp on a high-density display.
"""

from __future__ import annotations

import asyncio
import base64
import json
import shutil
import subprocess
import sys
import tempfile
from contextlib import suppress
from pathlib import Path

import httpx
import websockets

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "screenshots"
FRONTEND = "http://localhost:5173"
BACKEND = "http://127.0.0.1:8000"
DEBUG_PORT = 9333

# 4:3. Width in CSS pixels drives how large everything looks; the scale factor
# only decides how sharp it is.
WIDTH, HEIGHT, SCALE = 1120, 840, 2

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "/usr/bin/google-chrome",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
]


def find_chrome() -> str:
    for candidate in CHROME_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    found = shutil.which("chrome") or shutil.which("chromium") or shutil.which("google-chrome")
    if found:
        return found
    raise SystemExit("nenhum Chrome ou Edge encontrado")


class Tab:
    """The few DevTools Protocol calls this script needs, and nothing else."""

    def __init__(self, socket: websockets.ClientConnection) -> None:
        self._socket = socket
        self._next_id = 0

    async def call(self, method: str, **params) -> dict:
        self._next_id += 1
        message_id = self._next_id
        await self._socket.send(json.dumps({"id": message_id, "method": method, "params": params}))
        while True:
            message = json.loads(await self._socket.recv())
            # CDP interleaves events with replies; anything without our id is an
            # event we did not subscribe to and can drop.
            if message.get("id") == message_id:
                if "error" in message:
                    raise RuntimeError(f"{method}: {message['error']}")
                return message.get("result", {})

    async def goto(self, url: str) -> None:
        await self.call("Page.navigate", url=url)
        # Fixed settle time instead of a load event: this is a single-page app, so
        # the document finishes loading long before the data it renders arrives.
        await asyncio.sleep(2.5)

    async def js(self, expression: str) -> object:
        result = await self.call(
            "Runtime.evaluate", expression=expression, awaitPromise=True, returnByValue=True
        )
        return result.get("result", {}).get("value")

    async def shoot(self, name: str) -> Path:
        data = await self.call("Page.captureScreenshot", format="png", captureBeyondViewport=False)
        path = OUT_DIR / f"{name}.png"
        path.write_bytes(base64.b64decode(data["data"]))
        print(f"  {path.relative_to(ROOT)}  ({path.stat().st_size // 1024} KB)")
        return path


async def ask(tab: Tab, question: str) -> None:
    """Types a question into the composer and waits for the answer to finish."""
    await tab.js(f"""
        (() => {{
            const box = document.querySelector('textarea');
            const setter = Object.getOwnPropertyDescriptor(
                window.HTMLTextAreaElement.prototype, 'value').set;
            setter.call(box, {json.dumps(question)});
            box.dispatchEvent(new Event('input', {{ bubbles: true }}));
        }})()
    """)
    await asyncio.sleep(0.3)
    await tab.js(
        "[...document.querySelectorAll('button')].find(b => b.textContent.trim() === 'Send').click()"
    )
    # Polls for the streaming to stop rather than sleeping a guessed amount: the
    # answer takes as long as the model takes.
    for _ in range(60):
        await asyncio.sleep(1.0)
        pending = await tab.js("!!document.querySelector('[data-pending], .pending')")
        settled = await tab.js("document.body.innerText.length")
        await asyncio.sleep(0.6)
        if not pending and settled == await tab.js("document.body.innerText.length"):
            break
    await asyncio.sleep(1.0)


async def main() -> None:
    if httpx.get(f"{FRONTEND}/", timeout=3.0).status_code != 200:
        raise SystemExit(f"frontend nao responde em {FRONTEND}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    profile = tempfile.mkdtemp(prefix="polyrag-shots-")
    chrome = subprocess.Popen(
        [
            find_chrome(),
            "--headless=new",
            f"--remote-debugging-port={DEBUG_PORT}",
            f"--user-data-dir={profile}",
            f"--window-size={WIDTH},{HEIGHT}",
            f"--force-device-scale-factor={SCALE}",
            "--hide-scrollbars",
            "--no-first-run",
            "--disable-extensions",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        endpoint = ""
        for _ in range(40):
            await asyncio.sleep(0.5)
            with suppress(httpx.HTTPError, KeyError, IndexError):
                pages = httpx.get(f"http://127.0.0.1:{DEBUG_PORT}/json", timeout=1.0).json()
                endpoint = next(p["webSocketDebuggerUrl"] for p in pages if p["type"] == "page")
                break
        if not endpoint:
            raise SystemExit("Chrome nao abriu a porta de depuracao")

        async with websockets.connect(endpoint, max_size=32 * 1024 * 1024) as socket:
            tab = Tab(socket)
            await tab.call("Page.enable")
            await tab.call("Runtime.enable")
            await tab.call(
                "Emulation.setDeviceMetricsOverride",
                width=WIDTH,
                height=HEIGHT,
                deviceScaleFactor=SCALE,
                mobile=False,
            )
            print(f"capturando {WIDTH}x{HEIGHT} @{SCALE}x em {OUT_DIR.relative_to(ROOT)}/")

            for name, path, question, repeat in SHOTS:
                # Cleared before every chat shot: the previous shot leaves its
                # question in the cache, and without this the second one is
                # answered from RAM and the screenshot shows an empty pipeline.
                if question:
                    httpx.delete(f"{BACKEND}/api/v1/cache", timeout=5.0)
                await tab.goto(f"{FRONTEND}{path}")
                if question:
                    await ask(tab, question)
                    if repeat:
                        # Asked twice on purpose: the second one is the cache hit.
                        await ask(tab, question)
                await tab.shoot(name)
    finally:
        chrome.terminate()
        with suppress(Exception):
            chrome.wait(timeout=10)
        shutil.rmtree(profile, ignore_errors=True)


# Each shot exists to show something the others do not: a route decided by
# geometry, a route decided by the gray-zone evidence stage, the corpus split,
# and the process controls.
SHOTS = [
    # name, path, question (None = just load the page), ask twice for a cache hit
    #
    # Asked in English against a Portuguese corpus, on purpose. The answer follows
    # the question's language and the sources show the passage as it was written,
    # which is the multilingual behaviour rather than a translation step.
    ("chat-relational", "/", "What was the total revenue of the Sudeste region?", False),
    (
        "chat-graph",
        "/",
        "Which approval policy do the orders processed by Sistema Atlas follow?",
        False,
    ),
    ("chat-evidence", "/", "Which database does the Orion Database replicate to?", False),
    ("chat-cache", "/", "Who can suspend the Contrato Marco 2026?", True),
    ("corpus", "/corpus", None, False),
    ("servers", "/servers", None, False),
    ("telemetry", "/telemetry", None, False),
]


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
