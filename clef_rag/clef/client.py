"""Async Clef client.

Three backends, one wire format (a System One request body: ``model``, ``state``, ``questions``):

* ``workers-ai``  - Cloudflare Workers AI, ``@cf/cloudflare/clef`` or ``@cf/cloudflare/clef-flash``
  (``CLOUDFLARE_API_TOKEN`` + ``CLOUDFLARE_ACCOUNT_ID``)
* ``self-hosted`` - any server answering ``POST {base_url}/v1/systemone``: ``clef-rag serve`` on
  your own GPU, or anything else that speaks System One (``CLEF_BASE_URL``, optional ``CLEF_API_KEY``)
* ``local``       - the open weights in this process (``model_path``: a Hugging Face repo id or a
  release directory, e.g. one produced by ``clef-finetune merge``). Needs ``pip install "clef-rag[serve]"``
  and, in practice, a GPU.

``backend: auto`` picks ``workers-ai`` when both Cloudflare variables are set, otherwise
``self-hosted`` when ``CLEF_BASE_URL`` is set.

The client adds what batch RAG work needs on top of the raw API: a shared rate
limiter (requests and tokens per second), bounded concurrency, retries with
backoff that honour ``retry-after``, an on-disk answer cache, and usage
accounting. Any failure raises :class:`ClefError`; callers decide how to fall back.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import time
import weakref
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

import httpx

from .. import __version__
from .cache import AnswerCache, cache_key
from .questions import Answer, Question, parse_answer, questions_to_wire

MODELS = ("clef", "clef-flash")
# Workers AI list prices, USD per million input tokens (developers.cloudflare.com/workers-ai/models/clef[-flash]).
WORKERS_AI_PRICE = {"clef": 0.24, "clef-flash": 0.09}

BACKENDS: Dict[str, Dict[str, Any]] = {
    "workers-ai": {
        "base_url": "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/@cf/cloudflare",
        "api_key_env": "CLOUDFLARE_API_TOKEN",
        "account_id_env": "CLOUDFLARE_ACCOUNT_ID",
        "model": "clef",
        # 65,536-token context window per the Workers AI model page; keep headroom for the schema.
        "request_budget": 64_000,
    },
    "self-hosted": {
        "base_url_env": "CLEF_BASE_URL",
        "api_key_env": "CLEF_API_KEY",
        "model": "clef",
        "request_budget": 30_000,  # `clef-rag serve` default --max-length is 32768
    },
    "local": {
        "model": "clef",
        "request_budget": 30_000,
    },
}
AUTO_ORDER = ("workers-ai", "self-hosted")
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504, 529}

# Packing budget for one request's state (chunk windows, screening batches). Sized so a
# packed request fits every backend, including a self-hosted server at its default length.
STATE_BUDGET_TOKENS = 24_000
REQUEST_BUDGET_TOKENS = 64_000  # largest backend budget; per-backend budgets are in BACKENDS
PRICE_PER_MILLION_INPUT = WORKERS_AI_PRICE["clef"]


class ClefError(RuntimeError):
    """Any failure to obtain answers. Callers treat it as 'no decision'."""

    def __init__(self, message: str, *, status: Optional[int] = None, kind: str = "error"):
        super().__init__(message)
        self.status = status
        self.kind = kind


def estimate_tokens(obj: Any) -> int:
    """Conservative token estimate for budget packing (about 3 characters per token)."""
    text = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
    return len(text) // 3 + 8


@dataclass
class ClefResponse:
    model: str
    answers: Dict[str, Answer]
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    cached: bool = False


@dataclass
class ClefConfig:
    backend: str = "auto"  # auto | workers-ai | self-hosted | local
    model: Optional[str] = None  # clef | clef-flash (default clef)
    base_url: Optional[str] = None  # self-hosted endpoint, or override the Workers AI base
    api_key_env: Optional[str] = None
    account_id: Optional[str] = None  # Workers AI; default from CLOUDFLARE_ACCOUNT_ID
    model_path: Optional[str] = None  # local: HF repo id or release dir (default Cloudflare/<model>)
    device: Optional[str] = None  # local: cuda | cpu (default: cuda if available)
    dtype: str = "bfloat16"  # local
    max_length: int = 32_768  # local: tokens per request incl. schema
    cache_dir: Optional[str] = ".clef-rag/cache"
    max_rps: float = 20.0  # Workers AI limits for Clef are not published; tune to your account
    max_tokens_per_s: float = 80_000.0
    max_concurrency: int = 16
    timeout_s: float = 30.0
    max_retries: int = 5
    price_per_million: Optional[float] = None  # default: Workers AI list price; 0 for self-hosted/local

    def resolve(self) -> Dict[str, Any]:
        backend = self.backend
        if backend == "auto":
            if os.environ.get("CLOUDFLARE_API_TOKEN") and (self.account_id or os.environ.get("CLOUDFLARE_ACCOUNT_ID")):
                backend = "workers-ai"
            elif self.base_url or os.environ.get("CLEF_BASE_URL"):
                backend = "self-hosted"
            else:
                raise ClefError(
                    "no Clef backend configured: set CLOUDFLARE_API_TOKEN + CLOUDFLARE_ACCOUNT_ID (Workers AI), "
                    "or CLEF_BASE_URL (self-hosted), or clef.backend: local",
                    kind="config",
                )
        if backend not in BACKENDS:
            raise ClefError(f"unknown Clef backend {backend!r}; choose from auto, {', '.join(BACKENDS)}", kind="config")
        preset = BACKENDS[backend]
        model = (self.model or preset["model"]).strip()
        if backend in ("workers-ai", "local") and model not in MODELS:
            raise ClefError(f"model must be one of {MODELS} for the {backend} backend, got {model!r}", kind="config")
        key_env = self.api_key_env or preset.get("api_key_env") or ""
        out: Dict[str, Any] = {
            "backend": backend,
            "model": model,
            "api_key_env": key_env,
            "api_key": os.environ.get(key_env, "") if key_env else "",
            "request_budget": preset["request_budget"],
        }
        if backend == "workers-ai":
            account = self.account_id or os.environ.get(preset["account_id_env"], "")
            base = (self.base_url or preset["base_url"]).rstrip("/")
            out["base_url"] = base.replace("{account_id}", account)
            out["url"] = f"{out['base_url']}/{model}"
            out["account_id"] = account
            out["ready"] = bool(out["api_key"] and account)
            out["missing"] = " and ".join(v for v, ok in ((key_env, out["api_key"]), (preset["account_id_env"], account)) if not ok)
            price = WORKERS_AI_PRICE[model]
        elif backend == "self-hosted":
            base = (self.base_url or os.environ.get(preset["base_url_env"], "")).rstrip("/")
            out["base_url"] = base
            out["url"] = f"{base}/v1/systemone" if base else ""
            out["ready"] = bool(base)  # a key is optional for self-hosted servers
            out["missing"] = "" if base else preset["base_url_env"]
            price = 0.0
        else:  # local
            out["model_path"] = self.model_path or f"Cloudflare/{model}"
            out["base_url"] = "local"
            out["url"] = "http://clef.local/v1/systemone"
            out["ready"] = True
            out["missing"] = ""
            out["request_budget"] = max(1_000, self.max_length - 2_000)
            price = 0.0
        out["price_per_million"] = self.price_per_million if self.price_per_million is not None else price
        where = out.get("model_path") or out.get("base_url", "")
        out["cache_model"] = f"{backend}:{model}:{where}"
        return out

    def price(self) -> float:
        """USD per million input tokens for cost reports: Workers AI list price, 0 when self-run."""
        try:
            return float(self.resolve()["price_per_million"])
        except ClefError:
            return float(self.price_per_million or 0.0)


@dataclass
class Usage:
    requests: int = 0
    cached: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    errors: int = 0
    retries: int = 0
    latency_ms: List[float] = field(default_factory=list)

    def add(self, resp: ClefResponse) -> None:
        if resp.cached:
            self.cached += 1
            return
        self.requests += 1
        self.input_tokens += resp.input_tokens
        self.output_tokens += resp.output_tokens
        self.latency_ms.append(resp.latency_ms)

    def cost(self, price_per_million: Optional[float] = PRICE_PER_MILLION_INPUT) -> float:
        return self.input_tokens * (price_per_million or 0.0) / 1_000_000

    def to_dict(self, price_per_million: Optional[float] = PRICE_PER_MILLION_INPUT) -> Dict[str, Any]:
        lat = sorted(self.latency_ms)
        return {
            "requests": self.requests,
            "cached": self.cached,
            "errors": self.errors,
            "retries": self.retries,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(self.cost(price_per_million), 6),
            "latency_ms_p50": round(lat[len(lat) // 2], 1) if lat else None,
            "latency_ms_p90": round(lat[int(len(lat) * 0.9)], 1) if lat else None,
        }


class _Limiter:
    """Token buckets for requests/s and tokens/s, shared by every task on one loop."""

    def __init__(self, rps: float, tps: float) -> None:
        self.rps = max(0.1, rps)
        self.tps = max(1.0, tps)
        self._req = self.rps
        self._tok = self.tps
        self._t = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        dt = now - self._t
        self._t = now
        self._req = min(self.rps, self._req + dt * self.rps)
        self._tok = min(self.tps, self._tok + dt * self.tps)

    async def acquire(self, tokens: int) -> None:
        tokens = min(tokens, int(self.tps))
        while True:
            async with self._lock:
                self._refill()
                if self._req >= 1 and self._tok >= tokens:
                    self._req -= 1
                    self._tok -= tokens
                    return
                wait = max((1 - self._req) / self.rps, (tokens - self._tok) / self.tps, 0.005)
            await asyncio.sleep(wait)

    def penalise(self, seconds: float) -> None:
        """After a 429, drain the request bucket so every task backs off together."""
        self._req = min(self._req, -seconds * self.rps)


_SHARED: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, Dict[Tuple[str, ...], Tuple[_Limiter, asyncio.Semaphore]]]" = \
    weakref.WeakKeyDictionary()


def _shared_limits(key: Tuple[str, ...], rps: float, tps: float, concurrency: int) -> Tuple[_Limiter, asyncio.Semaphore]:
    """One limiter and semaphore per (event loop, backend, endpoint, key fingerprint).

    Rate limits are per API key, not per client object. ``Pipeline`` opens a
    client per call, so without sharing, N concurrent retrievals would each get
    the full budget and together exceed the account's limit.
    """
    loop = asyncio.get_running_loop()
    per_loop = _SHARED.setdefault(loop, {})
    if key not in per_loop:
        per_loop[key] = (_Limiter(rps, tps), asyncio.Semaphore(concurrency))
    return per_loop[key]


class ClefClient:
    """Use as ``async with ClefClient(config) as clef: await clef.ask(state, questions)``.

    Clients on one event loop that use the same backend and key share one rate
    limiter and one concurrency cap (``max_rps``, ``max_tokens_per_s``,
    ``max_concurrency``), so opening many clients doesn't multiply the budget.
    """

    def __init__(self, config: Optional[ClefConfig] = None, *, transport: Optional[httpx.AsyncBaseTransport] = None,
                 cache: Optional[AnswerCache] = None) -> None:
        self.config = config or ClefConfig()
        self._transport = transport
        self._resolved: Optional[Dict[str, Any]] = None
        self.cache = cache if cache is not None else AnswerCache(self.config.cache_dir)
        self.usage = Usage()
        self._http: Optional[httpx.AsyncClient] = None
        self._limiter: Optional[_Limiter] = None
        self._sem: Optional[asyncio.Semaphore] = None

    # -- lifecycle -----------------------------------------------------------

    @property
    def resolved(self) -> Dict[str, Any]:
        if self._resolved is None:
            self._resolved = self.config.resolve()
        return self._resolved

    @property
    def model(self) -> str:
        return self.resolved["model"]

    @property
    def price_per_million(self) -> float:
        try:
            return float(self.resolved["price_per_million"])
        except ClefError:
            return 0.0

    def available(self) -> bool:
        try:
            return bool(self.resolved["ready"])
        except ClefError:
            return False

    async def __aenter__(self) -> "ClefClient":
        transport = self._transport
        if transport is None and self.available() and self.resolved["backend"] == "local":
            from .local import local_transport  # imports torch lazily
            transport = local_transport(self.config, self.resolved)
            self._transport = transport
        self._http = httpx.AsyncClient(
            transport=transport,
            headers={"User-Agent": f"clef-rag/{__version__} (+https://github.com/MersivMedia/clef-rag)"},
            timeout=self.config.timeout_s,
        )
        r = self.resolved if self.available() else {"backend": "none", "url": "", "api_key": ""}
        fingerprint = hashlib.sha256(str(r.get("api_key", "")).encode()).hexdigest()[:16]
        key = (str(r.get("backend", "")), str(r.get("url", "")), fingerprint, str(id(self._transport) if self._transport else ""))
        concurrency = self.config.max_concurrency
        self._limiter, self._sem = _shared_limits(key, self.config.max_rps, self.config.max_tokens_per_s, concurrency)
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    # -- API -----------------------------------------------------------------

    def build_payload(self, state: Any, questions: Mapping[str, Question]) -> Dict[str, Any]:
        return {"model": self.model, "state": state, "questions": questions_to_wire(questions)}

    async def ask(self, state: Any, questions: Mapping[str, Question]) -> ClefResponse:
        payload = self.build_payload(state, questions)
        if len(payload["questions"]) > 64:
            raise ClefError(f"{len(payload['questions'])} questions in one request; the API allows 64",
                            kind="too_large")
        key = cache_key(self.resolved["cache_model"], payload["state"], payload["questions"])
        hit = self.cache.get(key)
        if hit is not None:
            resp = ClefResponse(model=hit["model"], cached=True,
                                answers={q: parse_answer(a) for q, a in hit["answers"].items()})
            self.usage.add(resp)
            return resp
        if not self.resolved["ready"]:
            raise ClefError(f"{self.resolved['missing']} is not set", kind="config")
        if self._http is None or self._limiter is None or self._sem is None:
            raise ClefError("ClefClient must be used inside 'async with'", kind="config")

        est = estimate_tokens(payload["state"]) + estimate_tokens(payload["questions"])
        budget = self.resolved["request_budget"]
        if est > budget:
            raise ClefError(f"request estimated at {est} tokens exceeds the {budget} budget for "
                            f"{self.resolved['backend']}", kind="too_large")
        headers = {"Content-Type": "application/json"}
        if self.resolved["api_key"]:
            headers["Authorization"] = f"Bearer {self.resolved['api_key']}"
        url = self.resolved["url"]

        async with self._sem:
            attempt = 0
            r: Optional[httpx.Response] = None
            t0 = time.monotonic()
            while True:
                await self._limiter.acquire(est)
                t0 = time.monotonic()
                r = None
                try:
                    r = await self._http.post(url, json=payload, headers=headers)
                except httpx.TimeoutException as exc:
                    err: Optional[ClefError] = ClefError(f"timeout: {exc}", kind="timeout")
                    status = None
                except httpx.HTTPError as exc:
                    err = ClefError(f"transport error: {exc}", kind="transport")
                    status = None
                else:
                    status = r.status_code
                    err = None if status == 200 else ClefError(_error_message(r), status=status,
                                                               kind=_error_kind(r))
                if err is None:
                    break
                retryable = status is None or status in RETRYABLE_STATUS
                if not retryable or attempt >= self.config.max_retries:
                    self.usage.errors += 1
                    raise err
                attempt += 1
                self.usage.retries += 1
                delay = (_retry_after(r) if r is not None else None) or \
                    min(8.0, 0.25 * (2 ** attempt)) + random.uniform(0, 0.1)
                if status == 429:
                    self._limiter.penalise(delay)
                await asyncio.sleep(delay)

        assert r is not None
        latency = (time.monotonic() - t0) * 1000.0
        try:
            body = _unwrap(r.json())
            answers = {qid: parse_answer(a) for qid, a in (body.get("answers") or {}).items()}
        except ClefError:
            self.usage.errors += 1
            raise
        except Exception as exc:
            self.usage.errors += 1
            raise ClefError(f"unparseable response: {exc}", kind="parse") from exc
        missing = set(payload["questions"]) - set(answers)
        if missing:
            self.usage.errors += 1
            raise ClefError(f"response missing answers for {sorted(missing)[:5]}", kind="parse")
        usage = body.get("usage") or {}
        resp = ClefResponse(
            model=str(body.get("model") or self.model),
            answers=answers,
            input_tokens=int(usage.get("input_tokens", usage.get("inputTokens", 0)) or 0),
            output_tokens=int(usage.get("output_tokens", usage.get("outputTokens", 0)) or 0),
            latency_ms=latency,
        )
        self.cache.put(key, {"model": resp.model, "answers": body.get("answers")})
        self.usage.add(resp)
        return resp

    async def ask_many(self, jobs: List[Tuple[Any, Mapping[str, Question]]]) -> List[Any]:
        """Run many requests concurrently. Each result is a ClefResponse or a ClefError."""
        async def one(state: Any, qs: Mapping[str, Question]) -> Any:
            try:
                return await self.ask(state, qs)
            except ClefError as exc:
                return exc
        return list(await asyncio.gather(*(one(s, q) for s, q in jobs)))


def _unwrap(body: Any) -> Dict[str, Any]:
    """Workers AI wraps results as ``{"result": {...}, "success": true, "errors": []}``."""
    if not isinstance(body, dict):
        raise ClefError(f"unexpected response body: {str(body)[:200]}", kind="parse")
    if body.get("success") is False:
        raise ClefError(f"API error: {_cf_errors(body)}", kind="api")
    if "answers" not in body and isinstance(body.get("result"), dict):
        return body["result"]
    return body


def _cf_errors(body: Dict[str, Any]) -> str:
    errs = body.get("errors") or []
    parts = []
    for e in errs:
        if isinstance(e, dict):
            parts.append(f"{e.get('code', '')} {e.get('message', '')}".strip())
        else:
            parts.append(str(e))
    return "; ".join(parts) or "unknown error"


def _retry_after(resp: Any) -> Optional[float]:
    try:
        value = resp.headers.get("retry-after")
        return min(30.0, float(value)) if value is not None else None
    except (ValueError, AttributeError):
        return None


def _error_message(resp: httpx.Response) -> str:
    try:
        body = resp.json()
        if isinstance(body, dict) and body.get("errors"):
            return f"HTTP {resp.status_code}: {_cf_errors(body)}"
        err = body.get("error")
        if isinstance(err, dict):
            return f"HTTP {resp.status_code}: {err.get('message') or err}"
        if err:
            return f"HTTP {resp.status_code}: {err}"
    except Exception:
        pass
    return f"HTTP {resp.status_code}: {resp.text[:200]}"


def _error_kind(resp: httpx.Response) -> str:
    try:
        err = resp.json().get("error")
        if isinstance(err, dict) and err.get("type"):
            return str(err["type"])
    except Exception:
        pass
    return {401: "auth", 403: "forbidden", 422: "invalid_request", 429: "rate_limited"}.get(
        resp.status_code, "http")
