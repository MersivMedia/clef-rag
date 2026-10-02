"""Backfill the rank-mode gate score on rows recorded before it existed.

For each row without ``gate.clef_rank``: rebuild the rank-mode passage set from
the recorded Clef scores (same code as the runner), fetch those chunk texts from
the store, ask the gate once, and append an updated row (load_recording keeps
the latest row per id). One Clef request per row.

Usage: python scripts/backfill_rank_gate.py <collection> [--config path] [--out dir]
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clef_rag.config import Config  # noqa: E402
from clef_rag.eval.runner import RANK_CLASSIFY, _gate, load_recording  # noqa: E402
from clef_rag.clef.client import ClefClient  # noqa: E402
from clef_rag.retrieve.stages import evidence_order, route_passage  # noqa: E402
from clef_rag.stores import make_store  # noqa: E402


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("collection")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=".clef-rag/eval")
    ap.add_argument("--concurrency", type=int, default=8)
    a = ap.parse_args()
    cfg = Config.load(a.config)
    store = make_store(cfg.store)
    rows = [r for r in load_recording(Path(a.out), a.collection) if "clef_rank" not in (r.get("gate") or {})]
    print(f"{a.collection}: {len(rows)} rows need a rank-mode gate")
    path = Path(a.out) / f"record_{a.collection}.jsonl"
    sem = asyncio.Semaphore(a.concurrency)
    lock = asyncio.Lock()
    queries = {}
    qfile = Path(".clef-rag/eval_data/prepared") / a.collection.removeprefix("eval_") / "queries.jsonl"
    for line in qfile.read_text().splitlines():
        q = json.loads(line)
        queries[str(q["id"])] = q["query"]
    async with ClefClient(cfg.clef) as clef:
        async def one(row):
            cands = row["cands"]
            inc = [c for c in cands if not c.get("clef") or route_passage(c["clef"], RANK_CLASSIFY) == "include"]
            inc.sort(key=evidence_order)
            con = [c for c in cands if c.get("clef") and route_passage(c["clef"], RANK_CLASSIFY) == "conflict"]
            chosen = inc[:RANK_CLASSIFY.max_passages] + con
            recs = {r.id: r.text for r in await asyncio.to_thread(store.get, a.collection, [c["id"] for c in chosen])}
            texts = [recs[c["id"]] for c in chosen if c["id"] in recs]
            if len(texts) != len(chosen):
                print(f"  {row['id']}: {len(chosen) - len(texts)} chunk texts missing; skipped")
                return
            async with sem:
                try:
                    g = await _gate(clef, queries[row["id"]], texts, 3000)
                except Exception as exc:  # leave the row for a rerun
                    print(f"  {row['id']}: {type(exc).__name__}: {exc}")
                    return
            row = dict(row)
            row["gate"] = dict(row.get("gate") or {}, clef_rank=g, clef_rank_ids=[c["id"] for c in chosen])
            async with lock:
                with path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
        await asyncio.gather(*(one(r) for r in rows))
        print(f"  Clef: {clef.usage.to_dict(cfg.clef.price())}")
    left = [r for r in load_recording(Path(a.out), a.collection) if "clef_rank" not in (r.get("gate") or {})]
    print(f"  rows still missing a rank gate: {len(left)}")


if __name__ == "__main__":
    asyncio.run(main())
