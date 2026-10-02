"""Local + serve backends against a tiny random Clef-shaped release (CPU, no downloads).

Skipped unless torch + transformers (Qwen3_5) are installed and a tiny release dir is
given in CLEF_RAG_TEST_TINY_RELEASE (build one with `clef-finetune make-tiny DIR`).
Proves the plumbing: Cloudflare's own joint_schema_model.py is imported from the
release dir, answers come back in System One shape, the HTTP server works with and
without auth, and ClefClient talks to both. It does NOT measure answer quality.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import socket
import threading
import urllib.error
import urllib.request

import pytest

TINY = os.environ.get("CLEF_RAG_TEST_TINY_RELEASE")
pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Qwen3_5ForConditionalGeneration"):
    pytest.skip("transformers lacks Qwen3_5ForConditionalGeneration (need >= 5.10.2)", allow_module_level=True)
if not TINY:
    pytest.skip("set CLEF_RAG_TEST_TINY_RELEASE to a tiny release dir", allow_module_level=True)

from clef_rag.clef import ClefClient, ClefConfig, ClefError, Choice, Noul, Score  # noqa: E402
from clef_rag.clef.local import LocalClef, RequestError  # noqa: E402

REQ = {
    "model": "clef",
    "state": "Checkout has been failing for every customer for the last hour.",
    "questions": {
        "urgent": {"type": "noul", "instructions": "Is this support request urgent?"},
        "team": {"type": "choice", "instructions": "Which team?",
                 "criteria": {"technical": "Outages", "billing": "Payments", "sales": "Plans"}},
        "severity": {"type": "score", "instructions": "How severe?", "criteria": ["None", "Minor", "Major", "Critical"]},
    },
}


@pytest.fixture(scope="module")
def engine():
    return LocalClef(TINY, device="cpu", dtype="float32", max_length=4096)


def test_local_answers_system_one_shape(engine):
    out = engine.answer(REQ)
    a = out["answers"]
    assert set(a) == {"urgent", "team", "severity"}
    assert 0 <= a["urgent"]["noul"] <= 1
    assert set(a["team"]["probabilities"]) == {"technical", "billing", "sales"}
    assert abs(sum(a["team"]["probabilities"].values()) - 1) < 1e-3
    assert 0 <= a["severity"]["score"] <= 3 and set(a["severity"]["legend"]) == {"0", "1", "2", "3"}
    assert out["usage"]["input_tokens"] > 0
    assert engine.answer(REQ) == out  # deterministic


def test_local_accepts_data_url_images(engine):
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (200, 30, 30)).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    out = engine.answer({**REQ, "images": [f"data:image/png;base64,{b64}", {"content_type": "image/png", "base64": b64}]})
    assert set(out["answers"]) == {"urgent", "team", "severity"}
    with pytest.raises(RequestError, match="remote URLs"):
        engine.answer({**REQ, "images": ["https://example.com/x.png"]})


def test_local_rejects_bad_requests(engine):
    with pytest.raises(RequestError):
        engine.answer({"model": "clef", "state": "x", "questions": {"q": {"type": "bogus", "instructions": "?"}}})
    with pytest.raises(RequestError, match="64"):
        engine.answer({"model": "clef", "state": "x", "questions": {f"q{i}": {"type": "noul", "instructions": "?"} for i in range(65)}})


def test_local_backend_through_client(monkeypatch):
    cfg = ClefConfig(backend="local", model_path=TINY, device="cpu", dtype="float32", max_length=4096, cache_dir=None)

    async def go():
        async with ClefClient(cfg) as c:
            r = await c.ask(REQ["state"], {"urgent": Noul("Is this urgent?"),
                                           "team": Choice("Which team?", {"technical": "Outages", "billing": "Payments"}),
                                           "sev": Score("How severe?", ["None", "Minor", "Major"])})
            return r, c.usage.to_dict(c.price_per_million)

    r, usage = asyncio.run(go())
    assert r.answers["team"].choice in ("technical", "billing") and usage["cost_usd"] == 0.0


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start(engine, api_key=None):
    from http.server import ThreadingHTTPServer

    from clef_rag.clef.serve import make_handler

    port = _free_port()
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(engine, "clef-flash", api_key))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{port}"


def _post(url, body, key=None):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_serve_http_and_client(engine, monkeypatch):
    httpd, base = _start(engine)
    try:
        with urllib.request.urlopen(base + "/health", timeout=10) as r:
            assert json.loads(r.read())["status"] == "ok"
        status, body = _post(base + "/v1/systemone", REQ)
        assert status == 200 and body["model"] == "clef-flash" and set(body["answers"]) == set(REQ["questions"])
        assert _post(base + "/v1/systemone", {"model": "clef", "state": "x", "questions": {}})[0] == 422
        assert _post(base + "/nope", REQ)[0] == 404

        cfg = ClefConfig(backend="self-hosted", base_url=base, model="clef-flash", cache_dir=None)

        async def go():
            async with ClefClient(cfg) as c:
                return await c.ask(REQ["state"], {"urgent": Noul("Is this urgent?")})

        assert 0 <= asyncio.run(go()).answers["urgent"].value <= 1
    finally:
        httpd.shutdown()


def test_serve_bearer_auth(engine, monkeypatch):
    httpd, base = _start(engine, api_key="s3cret")
    try:
        assert _post(base + "/v1/systemone", REQ)[0] == 401
        assert _post(base + "/v1/systemone", REQ, key="wrong")[0] == 401
        assert _post(base + "/v1/systemone", REQ, key="s3cret")[0] == 200
        monkeypatch.setenv("CLEF_API_KEY", "wrong")
        cfg = ClefConfig(backend="self-hosted", base_url=base, cache_dir=None, max_retries=0)

        async def go():
            async with ClefClient(cfg) as c:
                await c.ask("x", {"q": Noul("?")})

        with pytest.raises(ClefError) as e:
            asyncio.run(go())
        assert e.value.status == 401
    finally:
        httpd.shutdown()
