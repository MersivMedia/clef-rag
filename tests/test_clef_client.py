import asyncio

import httpx
import pytest

from clef_rag.clef import Choice, ClefClient, ClefConfig, ClefError, Noul, Score, SpecError, question_from_dict
from clef_rag.clef.questions import parse_answer


def run(c):
    return asyncio.run(c)


def test_question_specs_validate():
    assert Noul("x?").to_wire() == {"type": "noul", "instructions": "x?"}
    with pytest.raises(SpecError):
        Choice("x?", {"only": None}).to_wire()
    with pytest.raises(SpecError):
        Choice("x?", {str(i): None for i in range(256)}).to_wire()
    with pytest.raises(SpecError):
        Score("x?", ["a"]).to_wire()
    with pytest.raises(SpecError):
        Score("x?", [str(i) for i in range(11)]).to_wire()
    with pytest.raises(SpecError):
        Noul("x?", {"yes": "y"}).to_wire()
    assert question_from_dict({"type": "choice", "instructions": "q", "criteria": {"a": 1, "b": 2}}).to_wire()["type"] == "choice"
    with pytest.raises(SpecError):
        question_from_dict({"type": "choice", "instructions": "q", "criteria": ["a", "b"]})


def test_parse_answer_dialects():
    assert parse_answer({"type": "noul", "noul": 0.8}).value == 0.8
    assert parse_answer({"type": "boolean", "probability": 0.3}).value == 0.3  # Vercel native
    a = parse_answer({"type": "choice", "probabilities": {"x": 0.7, "y": 0.3}})
    assert a.choice == "x" and 0 < a.confidence < 1
    s = parse_answer({"type": "score", "probabilities": {"0": 0.5, "1": 0.5}})
    assert s.score == pytest.approx(0.5)


def test_auto_backend_order(monkeypatch):
    monkeypatch.setenv("CLEF_BASE_URL", "http://gpu-box:8000/")
    r = ClefConfig().resolve()
    assert r["backend"] == "self-hosted" and r["url"] == "http://gpu-box:8000/v1/systemone"
    assert r["ready"] and r["price_per_million"] == 0.0
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "t")
    assert ClefConfig().resolve()["backend"] == "self-hosted"  # token alone is not enough
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acc123")
    r = ClefConfig().resolve()
    assert r["backend"] == "workers-ai" and r["model"] == "clef"
    assert r["url"] == "https://api.cloudflare.com/client/v4/accounts/acc123/ai/run/@cf/cloudflare/clef"
    r = ClefConfig(model="clef-flash").resolve()
    assert r["url"].endswith("/@cf/cloudflare/clef-flash") and r["price_per_million"] == 0.09
    assert ClefConfig().price() == 0.24 and ClefConfig(price_per_million=0.5).price() == 0.5


def test_no_backend_is_a_clear_error():
    with pytest.raises(ClefError) as e:
        ClefConfig().resolve()
    assert "CLOUDFLARE_API_TOKEN" in str(e.value) and "CLEF_BASE_URL" in str(e.value)
    assert ClefConfig().price() == 0.0


def test_model_names_are_validated(monkeypatch):
    with pytest.raises(ClefError, match="clef-flash"):
        ClefConfig(backend="workers-ai", model="jev-1.13.0").resolve()
    r = ClefConfig(backend="local", model="clef-flash").resolve()
    assert r["model_path"] == "Cloudflare/clef-flash" and r["ready"] and r["price_per_million"] == 0.0
    # self-hosted servers may serve any name (e.g. a fine-tuned release)
    assert ClefConfig(backend="self-hosted", base_url="http://x", model="clef-insurance-v1").resolve()["model"] == "clef-insurance-v1"


def test_workers_ai_envelope_and_auth(fake_clef):
    fj, transport = fake_clef

    async def go():
        async with ClefClient(ClefConfig(cache_dir=None, model="clef-flash"), transport=transport) as c:
            return await c.ask("Checkout failing for everyone", {"urgent": Noul("Is this urgent?")})

    r = run(go())
    assert fj.urls[-1] == "https://api.cloudflare.com/client/v4/accounts/test-account/ai/run/@cf/cloudflare/clef-flash"
    assert fj.calls[-1]["model"] == "clef-flash" and set(fj.calls[-1]) == {"model", "state", "questions"}
    assert 0 <= r.answers["urgent"].value <= 1 and r.input_tokens > 0


def test_workers_ai_error_envelope(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "t")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "a")
    t = httpx.MockTransport(lambda r: httpx.Response(
        400, json={"result": None, "success": False, "errors": [{"code": 5006, "message": "bad input"}], "messages": []}))

    async def go():
        async with ClefClient(ClefConfig(cache_dir=None), transport=t) as c:
            await c.ask("x", {"q": Noul("?")})

    with pytest.raises(ClefError, match="5006 bad input") as e:
        run(go())
    assert e.value.status == 400


def test_self_hosted_plain_body_optional_key(monkeypatch):
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((str(request.url), request.headers.get("authorization")))
        return httpx.Response(200, json={"model": "clef", "answers": {"q": {"type": "noul", "noul": 0.25}},
                                         "usage": {"input_tokens": 7, "output_tokens": 0}})

    async def go(cfg):
        async with ClefClient(cfg, transport=httpx.MockTransport(handler)) as c:
            return await c.ask("x", {"q": Noul("?")})

    r = run(go(ClefConfig(backend="self-hosted", base_url="http://gpu:8000", cache_dir=None)))
    assert r.answers["q"].value == 0.25 and seen[-1] == ("http://gpu:8000/v1/systemone", None)
    monkeypatch.setenv("CLEF_API_KEY", "s3cret")
    run(go(ClefConfig(backend="self-hosted", base_url="http://gpu:8000", cache_dir=None)))
    assert seen[-1][1] == "Bearer s3cret"


def test_more_than_64_questions_refused(fake_clef):
    _, transport = fake_clef

    async def go():
        async with ClefClient(ClefConfig(cache_dir=None), transport=transport) as c:
            await c.ask("x", {f"q{i}": Noul("?") for i in range(65)})

    with pytest.raises(ClefError, match="64"):
        run(go())


def test_cache_is_per_backend_and_model(fake_clef, tmp_path):
    fj, transport = fake_clef

    async def go(model):
        async with ClefClient(ClefConfig(cache_dir=str(tmp_path / "c"), model=model), transport=transport) as c:
            return await c.ask("same", {"q": Noul("Is this filler?")})

    assert not run(go("clef")).cached
    assert not run(go("clef-flash")).cached  # a different model must not reuse clef's answers
    assert run(go("clef")).cached


def test_ask_retries_then_caches(fake_clef, tmp_path):
    fj, transport = fake_clef
    fj.fail_status, fj.fail_times = 503, 2
    cfg = ClefConfig(cache_dir=str(tmp_path / "cache"), max_retries=3)

    async def go():
        async with ClefClient(cfg, transport=transport) as c:
            r1 = await c.ask("some text", {"q": Noul("Is this text filler with no usable information?")})
            r2 = await c.ask("some text", {"q": Noul("Is this text filler with no usable information?")})
            return c, r1, r2

    c, r1, r2 = run(go())
    assert not r1.cached and r2.cached
    assert c.usage.retries == 2 and c.usage.requests == 1 and c.usage.cached == 1
    assert len(fj.calls) == 3  # two 503s + one success; the second ask never hit the network
    assert fj.calls[-1]["model"] == "clef"


def test_non_retryable_raises(fake_clef, tmp_path):
    fj, transport = fake_clef
    fj.fail_status, fj.fail_times = 422, 1

    async def go():
        async with ClefClient(ClefConfig(cache_dir=None), transport=transport) as c:
            await c.ask("x", {"q": Noul("?")})

    with pytest.raises(ClefError) as e:
        run(go())
    assert e.value.status == 422


def test_missing_answer_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setenv("CLEF_BASE_URL", "http://gpu:8000")
    t = httpx.MockTransport(lambda r: httpx.Response(200, json={"model": "m", "answers": {}, "usage": {}}))

    async def go():
        async with ClefClient(ClefConfig(cache_dir=None), transport=t) as c:
            await c.ask("x", {"q": Noul("?")})

    with pytest.raises(ClefError, match="missing answers"):
        run(go())


def test_ask_many_returns_errors_in_place(fake_clef):
    fj, transport = fake_clef
    fj.fail_status, fj.fail_times = 401, 1

    async def go():
        async with ClefClient(ClefConfig(cache_dir=None, max_retries=0), transport=transport) as c:
            return await c.ask_many([("a", {"q": Noul("?")}), ("b", {"q": Noul("?")})])

    res = run(go())
    assert sum(isinstance(r, ClefError) for r in res) == 1
    assert sum(not isinstance(r, ClefError) for r in res) == 1


def test_oversize_request_refused(fake_clef):
    _, transport = fake_clef

    async def go():
        async with ClefClient(ClefConfig(cache_dir=None), transport=transport) as c:
            await c.ask("x" * 400_000, {"q": Noul("?")})

    with pytest.raises(ClefError, match="budget"):
        run(go())


def test_clients_sharing_a_key_share_one_rate_budget(monkeypatch):
    """Clef limits are per key: two clients opened at once must not double the request rate."""
    import asyncio
    import time

    import httpx

    from clef_rag.clef.client import ClefClient, ClefConfig
    from clef_rag.clef.questions import Noul

    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "test-key")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acc")
    stamps = []

    def handler(request: httpx.Request) -> httpx.Response:
        stamps.append(time.monotonic())
        return httpx.Response(200, json={"model": "clef", "answers": {"q": {"type": "noul", "noul": 0.5}},
                                         "usage": {"input_tokens": 10, "output_tokens": 1}})

    transport = httpx.MockTransport(handler)
    cfg = ClefConfig(backend="workers-ai", cache_dir=None, max_rps=10, max_concurrency=16)

    async def worker(i: int) -> None:
        async with ClefClient(cfg, transport=transport) as clef:
            for j in range(10):
                await clef.ask({"i": i, "j": j}, {"q": Noul("x?")})

    async def go() -> float:
        t0 = time.monotonic()
        await asyncio.gather(worker(1), worker(2))
        return time.monotonic() - t0

    elapsed = asyncio.run(go())
    assert len(stamps) == 20
    # 20 requests at 10/s with a 10-request burst: at least ~1 s when shared, ~0 s if each client had its own budget
    assert elapsed >= 0.9, elapsed
