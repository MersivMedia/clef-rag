"""Run Clef's open weights in-process and answer System One requests.

Used by the ``local`` backend (through an httpx transport, so the rest of the
client is unchanged) and by ``clef-rag serve``.

The model code is Cloudflare's own ``joint_schema_model.py``, imported from the
release directory (a Hugging Face snapshot of Cloudflare/clef or Cloudflare/clef-flash,
or a directory written by ``clef-finetune merge``). Nothing here re-implements the model.

Requires ``pip install "clef-rag[serve]"`` (torch, transformers>=5.10.2, pillow).
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import importlib.util
import io
import json
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Optional

import httpx

# Hugging Face commits checked on 2026-10-02. A repo id without an explicit
# revision resolves to these so a silent upstream change can't alter answers.
PINNED_REVISIONS = {
    "Cloudflare/clef": "2f3de3dd85f379784083b0814d997ab627200f0c",
    "Cloudflare/clef-flash": "17f0b0ad64efb65d273590632833508766b2aae6",
}
MAX_IMAGES = 4


class RequestError(ValueError):
    """The request is malformed (maps to HTTP 422)."""


def resolve_release(model_path: str, revision: Optional[str] = None) -> Path:
    p = Path(model_path).expanduser()
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model_path, revision=revision or PINNED_REVISIONS.get(model_path)))


def import_release_code(release: Path):
    path = release / "joint_schema_model.py"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found: not a Clef release directory")
    name = "clef_release_" + hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:10]
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses need the module registered
    spec.loader.exec_module(module)
    return module


def decode_images(images: Any) -> list:
    """System One / Workers AI image forms -> PIL images: data URLs or {content_type, base64}."""
    if images is None:
        return []
    if not isinstance(images, list) or len(images) > MAX_IMAGES:
        raise RequestError(f"images must be a list of at most {MAX_IMAGES} items")
    from PIL import Image

    out = []
    for i, item in enumerate(images):
        if isinstance(item, str) and item[:5].lower() == "data:":
            _, _, data = item.partition(",")
        elif isinstance(item, dict) and "base64" in item:
            data = item["base64"]
        else:
            raise RequestError(f"images[{i}]: expected a data: URL or {{content_type, base64}} (remote URLs are not accepted)")
        try:
            out.append(Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB"))
        except Exception as exc:
            raise RequestError(f"images[{i}]: not a decodable image ({exc})") from None
    return out


class LocalClef:
    """One loaded model. ``answer`` is thread-safe (requests are serialised on the device)."""

    def __init__(self, model_path: str, *, device: Optional[str] = None, dtype: str = "bfloat16",
                 max_length: int = 32_768, revision: Optional[str] = None, served_name: Optional[str] = None) -> None:
        import torch

        self.release = resolve_release(model_path, revision)
        self.jsm = import_release_code(self.release)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.model, self.processor = self.jsm.load_release_model(self.release, device=self.device,
                                                                 dtype=getattr(torch, dtype))
        self.served_name = served_name
        self._lock = threading.Lock()

    def answer(self, request: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(request, dict):
            raise RequestError("request body must be a JSON object")
        req = dict(request)
        if self.served_name and str(req.get("model", "")).strip() not in ("", self.served_name):
            raise RequestError(f"this server serves model {self.served_name!r}, not {req.get('model')!r}")
        req.setdefault("model", self.served_name or "clef")
        qs = req.get("questions")
        if isinstance(qs, dict) and len(qs) > 64:
            raise RequestError("at most 64 questions per request")
        req["images"] = decode_images(req.get("images"))
        if not req["images"]:
            req.pop("images")
        req.pop("videos", None)  # not part of the public API; not accepted over HTTP
        with self._lock:
            try:
                return self.jsm.systemone(self.model, self.processor, req, max_length=self.max_length)
            except (ValueError, KeyError, TypeError) as exc:
                raise RequestError(str(exc)) from None


_ENGINES: Dict[tuple, LocalClef] = {}
_ENGINES_LOCK = threading.Lock()


def get_engine(model_path: str, device: Optional[str], dtype: str, max_length: int, served_name: str) -> LocalClef:
    key = (model_path, device, dtype, max_length)
    with _ENGINES_LOCK:
        if key not in _ENGINES:
            _ENGINES[key] = LocalClef(model_path, device=device, dtype=dtype, max_length=max_length,
                                      served_name=None)
        return _ENGINES[key]


class _LocalTransport(httpx.AsyncBaseTransport):
    def __init__(self, engine_factory) -> None:
        self._factory = engine_factory
        self._engine: Optional[LocalClef] = None

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(await request.aread())
        try:
            if self._engine is None:
                self._engine = await asyncio.to_thread(self._factory)
            result = await asyncio.to_thread(self._engine.answer, body)
        except RequestError as exc:
            return httpx.Response(422, json={"error": {"type": "invalid_request", "message": str(exc)}})
        return httpx.Response(200, json=result)


def local_transport(config: Any, resolved: Dict[str, Any]) -> httpx.AsyncBaseTransport:
    """httpx transport that answers in-process. The model loads on first use and is shared per process."""
    return _LocalTransport(lambda: get_engine(resolved["model_path"], config.device, config.dtype,
                                              config.max_length, resolved["model"]))
