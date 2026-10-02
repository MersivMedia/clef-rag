# Results

## Clef

**No measurements with Clef or Clef-flash yet.** What has been verified is plumbing, not quality:

| Check | Result |
|---|---|
| Offline suite (fake Clef over `httpx.MockTransport`, Workers AI envelope, self-hosted body, auth, retries, cache, budgets) | 187 passed |
| `local` backend + `clef-rag serve` against a tiny **random** Clef-shaped release on CPU (real `Qwen3_5ForConditionalGeneration`, Cloudflare's `joint_schema_model.py`) | 6 passed |
| `clef-rag serve` as a process → `clef-rag check` / `ingest` / `query` against it | worked end to end |

A random model's answers are noise, so none of this says anything about retrieval quality.

Planned: rerun the upstream benchmarks below with `clef` and `clef-flash` on Workers AI, retune the thresholds, and record cost and latency here.

## Upstream (Jev)

clef-rag was converted from [jev-rag-retrieval](https://github.com/MersivMedia/jev-rag-retrieval). Every number there was measured with **Jev** (TypeSafe's System One model), not Clef: [upstream RESULTS.md](https://github.com/MersivMedia/jev-rag-retrieval/blob/main/docs/RESULTS.md). Headline upstream findings, for context only:

- Ranking by evidence score (`select: rank`, top 5) stayed within 2 points of vector top-10 recall on SciFact and QASPER with about 43% less context, and lost 4.9 points on FiQA.
- The answer gate caught 29 to 41% of real unanswerable questions (QASPER), and wrongly refused 26% of SciFact queries, which are claims.
- `structural` chunking matched or beat model-placed chunking on every benchmark.

Clef is a different model with a different architecture and training, so treat these as hypotheses to re-test, not as Clef's performance.
