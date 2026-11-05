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

    @property
    def url(self) -> str:
        host, port = self.server_address[:2]
        return f"http://{host}:{port}/v1"


class _Handler(BaseHTTPRequestHandler):
    server: MockProvider

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        """Keep the default per-request log quiet."""

    def _reply(self, status: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
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
        model = payload.get("model")
        self._record(model)
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
