"""``clef-rag serve``: a System One endpoint for Clef's open weights on your own GPU.

    POST /v1/systemone   System One request body -> System One response body
    GET  /health         {"status": "ok", "model": ..., "release": ...}

Point any clef-rag install at it with ``CLEF_BASE_URL=http://gpu-box:8000``
(``clef.backend: self-hosted``). Bearer auth is optional: set ``--api-key-env``
to the name of an environment variable holding the key.

Stdlib HTTP server: requests are serialised on the GPU, one at a time. That
matches a single loaded model; put a reverse proxy in front for TLS.
"""

from __future__ import annotations

import hmac
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

MAX_BODY = 13 * 1024 * 1024  # same request cap as Workers AI


def make_handler(engine, served_name: str, api_key: Optional[str]):
    class Handler(BaseHTTPRequestHandler):
        server_version = "clef-rag-serve"

        def log_message(self, fmt, *args):  # one line per request on stderr
            sys.stderr.write("[serve] %s %s\n" % (self.address_string(), fmt % args))

        def _send(self, status: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _authorised(self) -> bool:
            if not api_key:
                return True
            got = self.headers.get("Authorization", "")
            return hmac.compare_digest(got.encode(), f"Bearer {api_key}".encode())

        def do_GET(self):
            if self.path.rstrip("/") == "/health":
                return self._send(200, {"status": "ok", "model": served_name, "release": str(engine.release)})
            return self._send(404, {"error": {"type": "not_found", "message": self.path}})

        def do_POST(self):
            from .local import RequestError

            if self.path.rstrip("/") != "/v1/systemone":
                return self._send(404, {"error": {"type": "not_found", "message": self.path}})
            if not self._authorised():
                return self._send(401, {"error": {"type": "auth", "message": "missing or wrong bearer token"}})
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > MAX_BODY:
                return self._send(413 if length > MAX_BODY else 400,
                                  {"error": {"type": "invalid_request", "message": "body missing or larger than 13 MiB"}})
            try:
                body = json.loads(self.rfile.read(length))
            except json.JSONDecodeError as exc:
                return self._send(400, {"error": {"type": "invalid_request", "message": f"invalid JSON: {exc}"}})
            try:
                result = engine.answer(body)
            except RequestError as exc:
                return self._send(422, {"error": {"type": "invalid_request", "message": str(exc)}})
            except Exception as exc:  # keep serving after a bad request or OOM
                return self._send(500, {"error": {"type": "server_error", "message": str(exc)[:500]}})
            result["model"] = served_name
            return self._send(200, result)

    return Handler


def serve(model_path: str, *, host: str = "127.0.0.1", port: int = 8000, device: Optional[str] = None,
          dtype: str = "bfloat16", max_length: int = 32_768, served_name: Optional[str] = None,
          api_key_env: Optional[str] = None, revision: Optional[str] = None) -> None:
    from .local import LocalClef

    api_key = os.environ.get(api_key_env, "") if api_key_env else ""
    if api_key_env and not api_key:
        raise SystemExit(f"--api-key-env {api_key_env} is set but the variable is empty")
    name = served_name or ("clef-flash" if "flash" in str(model_path).lower() else "clef")
    print(f"[serve] loading {model_path} ...", file=sys.stderr)
    engine = LocalClef(model_path, device=device, dtype=dtype, max_length=max_length, revision=revision)
    httpd = ThreadingHTTPServer((host, port), make_handler(engine, name, api_key or None))
    print(f"[serve] {name} on http://{host}:{port}/v1/systemone (device={engine.device}, auth={'on' if api_key else 'off'})",
          file=sys.stderr)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
