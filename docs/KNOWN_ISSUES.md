# Known issues

## Not yet measured with Clef

- **Every quality number is upstream and Jev-based.** The questions, thresholds and defaults come from [jev-rag-retrieval](https://github.com/MersivMedia/jev-rag-retrieval/blob/main/docs/RESULTS.md), where they were chosen with Jev. Clef's probabilities may be distributed differently (it is a separate model with a joint schema head and its own calibration training), so `drop_boilerplate: 0.85`, `answer_min: 0.35` and the rest may be too strict or too loose. Run stages in `shadow` mode and use `clef-rag eval` on your data before relying on them.
- **Workers AI hasn't been called with a real key yet.** The client follows Cloudflare's published schema (request body `model`/`state`/`questions`/`images`, response wrapped in `{"result": ..., "success": ...}`), and the tests use that shape. The first live run may surface differences.
- **Rate limits are unknown.** Cloudflare hasn't published Clef's Workers AI limits. `max_rps: 20` is a guess; 429s are retried with backoff.

## Backends

- **`clef-rag serve` handles one request at a time per process.** A loaded model answers requests in sequence. With `top_k: 30` each query makes about 32 requests, so on one GPU a query's latency is about 32 forward passes. Batching several passages into one request (Clef allows 64 questions per request) is on the roadmap.
- **`serve` has no TLS.** It's a stdlib HTTP server meant for a private network or behind a reverse proxy. Bearer auth is optional (`--api-key-env`).
- **`local` and `serve` were tested only on CPU against a tiny random model.** Real Clef-flash (9B) and Clef (27B) haven't been loaded by this code yet. Expect roughly 18 GB and 54 GB of GPU memory for weights in bf16. On GPU, install `flash-linear-attention` and `causal-conv1d`, or transformers falls back to slow reference kernels for Qwen3.5's linear-attention layers.
- **Videos aren't accepted over HTTP.** Cloudflare's code supports video frames in-process, but the public API has no `videos` field, so `serve` drops it.
- **Long state is truncated server-side** by Cloudflare's encoder to fit `max_length`. clef-rag's packing budgets (`STATE_BUDGET_TOKENS`, 24,000 estimated tokens) are sized so packed requests fit every backend at its default length.

## Pipeline (unchanged from upstream)

- **One classification request per candidate.** With `top_k: 30`, a query makes about 32 Clef requests. Lower `top_k` for throughput and cost.
- **Token counts are estimates** unless the `tokens` extra (tiktoken) is installed or you pass your embedder's tokenizer as `token_counter`. Clef uses the Qwen tokenizer, so tiktoken counts are approximate too.
- **Duplicates are removed within a document only.**
- **`semantic-embedding` chunking embeds every sentence.**
- **No `reenrich` or `reembed` commands yet.**
- **Very long passages are truncated in some Clef questions.** Classification sees the first 6,000 characters of a passage, the gate the first 3,000 per passage, and boundary questions the first 1,200 per sentence or table.

## Stores (unchanged from upstream)

- **Pinecone is experimental and untested against a live index.**
- **Chroma** can't express existence tests natively, and its `$ne` / `$nin` also match records without the field; clef-rag over-fetches and filters in Python.
- **LangChain bridge** filters in Python after over-fetching (`overfetch`, default 4 times `top_k`).
- **Qdrant embedded mode** ignores payload indexes; use a Qdrant server for large collections.
- **`--retrieve-only`** expects clef-rag's field layout (text in the store's `text` field).

## Security

- **Injection screening is one layer.** Upstream (with Jev) it caught all 6 planted instructions in the messy benchmark. It hasn't been measured with Clef or on a real attack set. Keep treating retrieved text as data in your LLM's system prompt (clef-rag's `answer()` does).
