"""An OpenAI-compatible mock provider for tests and the image smoke check.

It answers chat completions with a fixed reply and lists the upstream model ids.
It never needs a real key; the Authorization header it receives is reduced to
whether it was present, so a test can assert that the gateway forwarded one
without the mock ever holding or printing the value.
Run it standalone with: python tests/mock_provider.py --port 8080
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM_MODELS = (
    "nvidia/nemotron-3.5-super-vl-120b-a12b",
    "nvidia/nemotron-3.5-lightning-30b-a3b",
)


class MockProvider(ThreadingHTTPServer):
    """The server plus what it saw: (path, model, authorized) per request."""

    daemon_threads = True

    def __init__(self, address: tuple[str, int]) -> None:
        super().__init__(address, _Handler)
        self.seen: list[tuple[str, str | None, bool]] = []
        self.lock = threading.Lock()
        #: Per upstream model (or "*" for all): {"status", "delay_s", "headers", "times"}.
        #: Set with POST /_control; "times" counts down and the rule ends at 0 (absent = forever).
        self.behaviour: dict[str, dict] = {}

    def take_rule(self, model: str | None) -> dict | None:
        """The rule for ``model`` (or the catch-all), consuming one use of it."""
        with self.lock:
            for key in (model or "", "*"):
                rule = self.behaviour.get(key)
                if rule is None:
                    continue
                if "times" in rule:
                    rule["times"] -= 1
                    if rule["times"] <= 0:
                        del self.behaviour[key]
                return dict(rule)
        return None

    def set_behaviour(self, rules: dict[str, dict], reset: bool = False) -> None:
        """Replace programmable rules without requiring an HTTP control client."""
        with self.lock:
            if reset:
                self.behaviour.clear()
            for model, rule in rules.items():
                self.behaviour[model] = dict(rule)

    @property
    def url(self) -> str:
        host, port = self.server_address[:2]
        if isinstance(host, bytes):
            host = host.decode()
        return f"http://{host}:{port}/v1"


class _Handler(BaseHTTPRequestHandler):
    server: MockProvider

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        """Keep the default per-request log quiet."""

    def _reply(self, status: int, body: dict, headers: dict | None = None) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        for name, value in (headers or {}).items():
            self.send_header(str(name), str(value))
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _record(self, model: str | None) -> None:
        with self.server.lock:
            self.server.seen.append(
                (self.path, model, bool(self.headers.get("Authorization")))
            )

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/_seen":
            # Provider calls per model so far: the smoke test counts attempts with it.
            with self.server.lock:
                counts: dict[str, int] = {}
                for _path, model, _auth in self.server.seen:
                    if model:
                        counts[model] = counts.get(model, 0) + 1
            self._reply(200, {"calls": counts})
            return
        self._record(None)
        if self.path.rstrip("/").endswith("/models"):
            self._reply(
                200,
                {
                    "object": "list",
                    "data": [
                        {"id": m, "object": "model", "owned_by": "mock"}
                        for m in UPSTREAM_MODELS
                    ],
                },
            )
        else:
            self._reply(404, {"error": {"message": "not found"}})

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/_control":
            with self.server.lock:
                if payload.get("reset"):
                    self.server.behaviour.clear()
                for model, rule in (payload.get("rules") or {}).items():
                    self.server.behaviour[model] = dict(rule)
            self._reply(200, {"behaviour": self.server.behaviour})
            return
        model = payload.get("model")
        self._record(model)
        rule = self.server.take_rule(model)
        if rule is not None:
            if rule.get("delay_s"):
                time.sleep(
                    float(rule["delay_s"])
                )  # the mock's simulated provider latency
            status = int(rule.get("status", 200))
            if status != 200:
                self._reply(
                    status,
                    {"error": {"message": f"mock {status}", "type": "mock"}},
                    headers=rule.get("headers") or {},
                )
                return
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self._reply(404, {"error": {"message": "not found"}})
            return
        self._reply(
            200,
            {
                "id": "chatcmpl-mock",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )


def serve(host: str = "127.0.0.1", port: int = 0) -> MockProvider:
    """Start the mock provider in a background thread and return it."""
    server = MockProvider((host, port))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")  # nosec B104 - test container only
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    MockProvider((args.host, args.port)).serve_forever()
