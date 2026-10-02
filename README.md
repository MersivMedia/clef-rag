# clef-rag

Clef-steered ingestion and retrieval for any vector database, on Cloudflare Workers AI, on your own GPU, or in-process.

**Keep junk and planted instructions out of your index, send your LLM only the passages that answer the question, and refuse when nothing does, using Cloudflare's open-source Clef or Clef-flash decision models on the vector database you already use.**

- **Three ways to run Clef, one config line.** `backend: workers-ai` calls `@cf/cloudflare/clef` or `@cf/cloudflare/clef-flash` on Cloudflare. `backend: self-hosted` points at any System One endpoint, including the bundled `clef-rag serve` running the Apache-2.0 weights on your GPU. `backend: local` loads the weights in-process. A fine-tuned release from [clef-finetune](https://github.com/MersivMedia/clef-finetune) works with `self-hosted` and `local` unchanged.
- **Clef or Clef-flash.** `model: clef` (27B) for precision, `model: clef-flash` (9B) for latency and price ($0.24 vs $0.09 per million input tokens on Workers AI, nothing for output).
- **A better order for the passages you already retrieve.** Clef scores every retrieved passage as evidence, a conflict with the question, or noise. Code ranks by evidence and keeps the top 5. A final check can abstain before any LLM call when the passages can't answer.
- **A cleaner index.** Every paragraph is screened before chunking: boilerplate is cut, and text that tries to instruct an AI is quarantined on its own, before anything is embedded. Every chunk can be tagged against your own taxonomy, with probabilities.
- **Chunking that respects structure.** The default `structural` chunker never crosses a heading and keeps tables and code whole, with no Clef calls. Model-placed cuts (`method: clef`) are available.
- **Your database.** Adapters for Postgres + pgvector, Qdrant and Chroma, plus a bridge to any LangChain vector store, all held to one conformance suite. A Pinecone adapter is included as experimental.
- **Self-hosting is free per call.** On `self-hosted` and `local` there's no per-token charge, just your GPU. Cost reports show $0 for them and the Workers AI list price otherwise.

> **Status: converted from [jev-rag-retrieval](https://github.com/MersivMedia/jev-rag-retrieval), not yet benchmarked with Clef.** clef-rag is jev-rag-retrieval with its model client swapped for Clef. Clef's API is System One-compatible, so every question, threshold and pipeline stage is unchanged. Tested: 187 offline tests, plus the local and `serve` backends end to end against a tiny random model on CPU. **Not yet run:** Workers AI with a real key, and anything with the real Clef weights. Every quality number in jev-rag-retrieval was measured with **Jev**, not Clef. They are [upstream results](https://github.com/MersivMedia/jev-rag-retrieval/blob/main/docs/RESULTS.md) and are not claimed here. Thresholds tuned on Jev may need retuning for Clef: run `clef-rag eval` on your data. See [Known issues](https://github.com/MersivMedia/clef-rag/blob/main/docs/KNOWN_ISSUES.md).

clef-rag uses [Clef](https://huggingface.co/Cloudflare/clef), Cloudflare's open-source decision model ([launch post](https://blog.cloudflare.com/clef-decision-models), [Workers AI docs](https://developers.cloudflare.com/workers-ai/models/clef/)). Clef never writes text. It answers typed questions (a yes/no probability, one option from a list, or a score on a scale), and plain code with visible thresholds decides what happens. Every decision is logged with its probabilities.

## What it does

| Stage | What Clef decides | What code does |
|---|---|---|
| Chunking (opt-in, `method: clef`) | Does this sentence continue the point of the previous one? Does it depend on it to make sense? | Places cuts where continuity is lowest, within min/target/max token sizes. The default `structural` chunker asks Clef nothing |
| Paragraph screen | Is this paragraph boilerplate? Does it contain instructions aimed at an AI? | Cuts it before chunking, or quarantines it alone |
| Quality screen | Is this chunk filler or boilerplate? Does it contain instructions aimed at an AI? Is it self-contained? | Drops, quarantines or keeps it |
| Tagging | Which option of each taxonomy field fits, including `other` | Stores the tag only when confident; always stores the probability |
| Query routing | Which taxonomy value the question is about | Applies a metadata filter only when confident; retries unfiltered if it returns too little |
| Passage classification | Relevant? Usable evidence? Contradicts the question's premise? Instructions aimed at an AI? | Drops injections, marks conflicts, ranks the rest by evidence score and keeps the top 5 (`select: rank`, the default); `select: threshold` keeps only passages above set scores |
| Answer gate | Can these passages answer the question? | Abstains without calling the LLM, or builds the prompt |

## Contents

- [Install](#install)
- [Choosing a backend](#choosing-a-backend)
- [Quickstart](#quickstart)
- [Configure](#configure)
- [Ingestion guide: processing and storing embeddings](#ingestion-guide-processing-and-storing-embeddings)
- [Retrieval guide: querying, classifying and answering](#retrieval-guide-querying-classifying-and-answering)
- [Vector databases](#vector-databases)
- [CLI reference](#cli-reference)
- [Cost and limits](#cost-and-limits)
- [Testing](#testing)
- [Roadmap](#roadmap)
- [More](#more)

## Install

Python 3.10 to 3.13. The package installs as `clef-rag`; you import it as `clef_rag` and run it as the `clef-rag` command. The core needs only `httpx`, `pydantic` and `pyyaml`; each database, embedder and file format is an extra.

Install from PyPI (add the extras for your database and embedder):

```bash
pip install "clef-rag[qdrant]"
```

| Extra | Installs |
|---|---|
| `qdrant`, `chroma`, `pgvector` | That database's official client |
| `pinecone` | The Pinecone client, for the experimental adapter |
| `langchain` | `langchain-core`, for the bridge to any LangChain vector store |
| `local` | sentence-transformers, for local embeddings |
| `pdf`, `docx` | PyMuPDF and python-docx loaders |
| `tokens` | tiktoken, for exact token counts (otherwise an estimate is used) |
| `serve` | torch + transformers ≥ 5.10.2, for `backend: local` and `clef-rag serve` (GPU recommended) |
| `all` | Everything above except `local` and `serve` |

### Keys

| Backend | Set | Notes |
|---|---|---|
| `workers-ai` | `CLOUDFLARE_API_TOKEN` (Workers AI permission) and `CLOUDFLARE_ACCOUNT_ID` | Billed per input token by Cloudflare |
| `self-hosted` | `CLEF_BASE_URL`, plus `CLEF_API_KEY` if the server requires one | Any `POST /v1/systemone` endpoint |
| `local` | nothing: set `clef.backend: local` (optionally `clef.model_path`) | Needs the `serve` extra; downloads weights from Hugging Face on first use |

With `backend: auto` (the default), clef-rag uses Workers AI when both Cloudflare variables are set, otherwise `CLEF_BASE_URL`.

Then add the key for your embedding provider (`OPENAI_API_KEY`, or `AI_GATEWAY_API_KEY` with the `gateway:` embedder) and your database's connection settings.

Put them in a `.env` file: [`.env.example`](https://github.com/MersivMedia/clef-rag/blob/main/.env.example) lists every variable clef-rag reads, blank, with a note on each. The `clef-rag` CLI loads `./.env` before every command. Variables already set in your shell win, blank lines in the file are ignored, and it warns if the file is readable by other users. Use `--env-file path` for another file or `--no-env-file` to skip it. The Python API doesn't read `.env` on its own; call `clef_rag.envfile.load_env_file()` first if you want the same behaviour.

Without a Clef backend everything still runs, but nothing is screened or tagged, and retrieval returns plain vector ranking marked `degraded`.

## Choosing a backend

```yaml
clef:
  backend: workers-ai     # or self-hosted, local
  model: clef-flash       # or clef
```

**Workers AI**: no infrastructure. Cloudflare says it doesn't store or train on requests.

**Self-hosted**: run the weights on a GPU you control, which matters for regulated data. On the GPU box:

```bash
pip install "clef-rag[serve]"
clef-rag serve Cloudflare/clef-flash --host 0.0.0.0 --port 8000 --api-key-env CLEF_SERVER_KEY
#   or a fine-tuned release:  clef-rag serve /models/clef-flash-insurance-v1
```

Then on any client, set `CLEF_BASE_URL=http://gpu-box:8000` (and `CLEF_API_KEY` to the same key). `serve` loads Cloudflare's own `joint_schema_model.py` from the release, answers `POST /v1/systemone` and `GET /health`, accepts the Workers AI `images` field (data URLs or base64), and handles one request at a time per GPU. Put TLS in front of it with a reverse proxy.

**Local**: the same engine in your Python process (`backend: local`), for notebooks and single-machine batch jobs. The model loads on first use and is shared across clients in the process.

Clef-flash in bf16 needs roughly 18 GB of GPU memory for weights alone, and Clef (27B) roughly 54 GB. Cloudflare tested on one H200.

## Quickstart

```bash
docker run -d -p 6333:6333 qdrant/qdrant       # or any supported database

clef-rag init --store qdrant --embedder openai:text-embedding-3-small
                                               # writes clef-rag.yaml, plus a blank .env (chmod 600) if none exists
# edit .env: set CLOUDFLARE_API_TOKEN + CLOUDFLARE_ACCOUNT_ID (or CLEF_BASE_URL) and OPENAI_API_KEY
clef-rag check                                   # one Clef call, one embedding, one store round trip
clef-rag ingest ./docs --collection handbook --dry-run
clef-rag ingest ./docs --collection handbook
clef-rag query "How long do refresh tokens last?" --collection handbook
```

`query` prints the kept passages with their evidence scores, any conflicts, the filter used and the gate's verdict. Add `-v` to see what was dropped and why, `--json` for machine output, or `--answer` to have an LLM write the answer from the passages.

The same in Python:

```python
from clef_rag import Pipeline

rag = Pipeline.from_config("clef-rag.yaml")
report = rag.ingest(["./docs"], collection="handbook")
print(report.summary())        # docs, chunks, dropped, quarantined, Clef requests, tokens, cost

result = rag.retrieve("How long do refresh tokens last?", collection="handbook")
if result.abstain:
    print("Not in the documents:", result.reason)
else:
    for p in result.passages:
        print(f"{p.scores['contains_answer_evidence']:.2f}  {p.citation}")
        print(p.text[:200])
```

Every sync method has an async twin (`aingest`, `aretrieve`, `aanswer`) for use inside an event loop.

## Configure

Two sample files at the repo root:

- [`clef-rag.example.yaml`](https://github.com/MersivMedia/clef-rag/blob/main/clef-rag.example.yaml): every setting with its default and a comment, plus a ready-to-uncomment block for each database. `clef-rag init --full` writes the same file. Plain `clef-rag init` writes the short version below.
- [`.env.example`](https://github.com/MersivMedia/clef-rag/blob/main/.env.example): every environment variable, blank.

Secrets never go in the YAML; it only names the variable to read (`api_key_env`, `dsn_env`). The short config:

```yaml
clef:
  backend: auto                # auto | workers-ai | self-hosted | local
  model: clef                  # clef | clef-flash
  cache_dir: .clef-rag/cache   # answers are cached by content hash
  max_rps: 20                  # Workers AI limits for Clef are not published: tune

store:
  kind: qdrant
  url: http://localhost:6333
  # api_key_env: QDRANT_API_KEY

embedder:
  model: openai:text-embedding-3-small
  batch_size: 128

chunking:
  method: structural           # structural | clef | fixed | semantic-embedding
  mode: on                     # clef method only: on | shadow | off
  min_tokens: 64
  target_tokens: 350
  max_tokens: 800
  overlap_tokens: 0

enrich:
  mode: on
  drop_low_information: 0.85
  drop_boilerplate: 0.85
  quarantine_instructs_ai: 0.70
  tag_min_confidence: 0.50
  # taxonomy: taxonomy.yaml

retrieve:
  top_k: 30
  expand_neighbours: false
  route: { mode: on, min_confidence: 0.60, top2_mass: 0.80, min_candidates: 5 }
  classify:
    mode: on
    select: rank             # or threshold: keep only passages above min_relevant/min_evidence
    drop_instructs_ai: 0.70
    min_relevant: 0.50
    min_evidence: 0.40
    conflict: 0.60
    max_passages: 5
  gate: { mode: on, answer_min: 0.35 }

answer:                        # only used by answer() / `clef-rag query --answer`
  provider: openai             # openai | anthropic | gateway
  model: gpt-4.1-mini
```

**The thresholds are starting points, not measured defaults.** They were chosen with Jev on small probes in the [upstream project](https://github.com/MersivMedia/jev-rag-retrieval/blob/main/docs/RESULTS.md) and have not been checked against Clef, whose probabilities may be distributed differently. Check them against your own documents with `clef-rag query -v` before relying on them.

Every Clef stage has a `mode`. `shadow` computes and logs Clef's decision but acts on the fallback's result (structural chunks, keep every chunk, vector order, never abstain), so you can compare before switching it on. The shadow decisions appear in the ingest traces and in `result.trace`.

Unknown keys are rejected with an error, so a typo like `topk` fails loudly.

## Ingestion guide: processing and storing embeddings

Ingestion turns files into records in your vector database. Six steps, each inspectable on its own.

```
files -> 1 parse -> 2 screen paragraphs -> 3 chunk -> 4 enrich -> 5 embed -> 6 store
```

### Step 1: Parse

Loaders read `.md`, `.txt`, `.rst`, `.html`, `.pdf` (text layer only, needs the `pdf` extra) and `.docx` (needs `docx`). HTML is converted to Markdown with scripts, styles, navigation, headers, footers and asides removed. Each document becomes blocks (heading, paragraph, list item, table, code or quote), each with its heading path ("Guide > Auth > Tokens") and character offsets.

What code decides, never Clef:

- **Headings are hard boundaries.** No chunk spans two sections.
- **Tables and code blocks stay whole**, split by rows or lines only if they exceed `max_tokens`.
- **Sentences** are split by a segmenter that handles "e.g.", "U.S.", "Dr.", decimals, version numbers and URLs. A sentence longer than `max_tokens` is split at whitespace.

Scanned PDFs, images and audio need OCR or transcription first; Clef reads text only.

To bring your own parsed text:

```python
from clef_rag import Document

docs = [Document(doc_id="kb-142", text=body, title="Refunds policy",
                 source_uri="https://example.com/kb/142", metadata={"team": "billing"})]
rag.ingest(docs, collection="handbook")
```

`doc_id` should be stable across runs (a URL, path or database key). It's what makes re-ingestion replace old chunks instead of duplicating them. For files, it's the path relative to the folder you ingest. `format` is `markdown` (default), `text` or `html`. Scalar `metadata` values are stored as `m_<key>` and can be filtered on.

### Step 2: Screen paragraphs

Before chunking, every paragraph, list item, table and quote is checked on its own with two yes/no questions:

| Question | Default action |
|---|---|
| `boilerplate`: navigation, cookie banner, newsletter or share prompt, advert, legal footer, table of contents, reference-list entry? | Cut from the text at ≥ 0.85 (listed in the report) |
| `instructs_ai`: does it contain instructions addressed to an AI assistant? | Cut and stored alone as a **quarantined** record at ≥ 0.70 |

The checks are packed 40 paragraphs to a Clef request, with each question carrying its own paragraph.

Why per paragraph and not per chunk:
- **Junk survives a chunk-level check.** A short cookie banner or share bar is under `min_tokens`, so it gets merged into a content chunk, where it's a small fraction of the text and passes.
- **Chunk-level quarantine hides real content.** An injection planted between two real paragraphs would be quarantined together with them.

Upstream, with Jev, on the [messy-document benchmark](https://github.com/MersivMedia/jev-rag-retrieval/blob/main/docs/RESULTS.md#paragraph-level-screening-rerun-of-the-messy-benchmark) (not yet rerun with Clef): all 6 planted injections were quarantined alone, with no answers hidden, and 11 of 12 planted boilerplate paragraphs were removed. The hit rate rose from 93.4% to 96.7%.

Cut paragraphs are blanked in a working copy, so chunk offsets still point into the original document. Headings and code are never screened.
- **Reference lists are cut too.** If your users ask about citations, run `enrich.screen_paragraphs: shadow` first: it scores and logs everything and cuts nothing.
- **`off`** skips this step.
- **On Clef failure**, a batch's paragraphs are kept.

### Step 3: Chunk

**Default: `structural`.** It cuts at headings and paragraph breaks within `min_tokens`/`target_tokens`/`max_tokens`, merging short pieces. It makes no Clef calls. It's the default because, upstream with Jev, it matched or beat model-placed chunking on both benchmarks once paragraph screening ran, at about half the ingest cost ([upstream results](https://github.com/MersivMedia/jev-rag-retrieval/blob/main/docs/RESULTS.md#paragraph-level-screening-rerun-of-the-messy-benchmark)); not yet rerun with Clef. Headings are hard boundaries for every method.

**Clef chunking (`method: clef`).** It may help on long unstructured text, such as transcripts or prose without paragraph breaks; this hasn't been shown yet. Compare on your own documents with `clef-rag inspect <file> --compare clef`. Within each section, clef-rag sends Clef the section text (in windows sized to Clef's request budget) plus two yes/no questions for every adjacent pair of sentences, all in **one request per window**:

| Question | Wording |
|---|---|
| `continues` | In the document, does `sentence` continue the specific point that `previous` is making? |
| `refers_back` | Does `sentence` depend on `previous` to be understood, for example by referring back to it with words like this, it, these or such? |

Each question carries its own sentence pair. An earlier design referred to sentences by position (`sentences[4]`), and Jev (upstream) answered those about the wrong sentences: mean error 0.63 on a labelled probe, against 0.07 with the pair inline ([upstream results](https://github.com/MersivMedia/jev-rag-retrieval/blob/main/docs/RESULTS.md#boundary-question-format)). Not re-measured with Clef. Set `chunking.boundary.style: keyed` to use named keys instead, about 25% fewer tokens at slightly lower accuracy on that probe.

Gaps that code already decides (after a heading, between sections) are never asked. A 2,000-sentence manual takes a handful of requests, not 2,000.

Code then places the cuts:

1. Each gap gets a cut cost: `0.6 × continues + 0.4 × refers_back`, reduced at paragraph breaks.
2. A dynamic-programming pass chooses the cuts with the lowest total cost plus a penalty for straying from `target_tokens`, and never exceeds `max_tokens`.
3. A gap where `refers_back` ≥ 0.5 is protected: it's only cut if the size limit forces it, so "This means..." stays with what "this" refers to.
4. Chunks under `min_tokens` merge into their more continuous neighbour.

Sizes are counted with tiktoken's `cl100k_base` when the `tokens` extra is installed, otherwise with an estimate. Pass `token_counter=` to `Pipeline` to use your embedding model's own tokenizer.

See the result before storing anything:

```bash
clef-rag inspect ./docs/auth.md                     # chunks and sizes
clef-rag inspect ./docs/auth.md --compare clef -v     # plus Clef chunking, with its score at every candidate cut
clef-rag inspect ./docs/auth.md --compare clef,fixed
```

Each chunk also gets an `embed_text`: the document title and heading path prepended to the chunk, so a chunk that says "They expire after 14 days" still embeds near questions about refresh tokens. `text` (what the LLM sees) is stored separately.

Other methods: `fixed` (about `target_tokens` per chunk, sentence-aligned, with optional `overlap_tokens`), and `semantic-embedding` (cut where adjacent sentence embeddings diverge; one embedding call per sentence). If Clef fails, `clef` falls back to `structural` and the ingest report says so.

### Step 4: Enrich

One Clef request per chunk carries every enrichment question at once. This is a second check after the paragraph screen, and it adds the tags.

**Quality screen** (yes/no probabilities, stored as `q_<name>`):

| Question | Default action |
|---|---|
| `low_information`: is this filler with no usable information? | Drop at ≥ 0.85 |
| `boilerplate`: navigation, footer, cookie banner, copyright line, table of contents? | Drop at ≥ 0.85 |
| `instructs_ai`: does it contain instructions addressed to an AI assistant, rather than information for a human reader? | **Quarantine** at ≥ 0.70 |
| `self_contained`: can it be understood without the text before it? | Stored; used by neighbour expansion |

Dropped chunks are listed in the ingest report (`clef-rag ingest -v`) and in the per-document trace. Quarantined chunks are stored with `quarantined=true` and excluded from every query unless you ask for them, so you can review them:

```bash
clef-rag inspect --collection handbook --quarantined
```

**Taxonomy tags.** Define fields in a YAML file and point `enrich.taxonomy` at it. Each field is a one-of-N choice. Describe every option, and always include `other`: Clef must put its probability somewhere, and a missing option forces a wrong tag. clef-rag refuses a field without `other`.

```yaml
version: 3
fields:
  doc_type:
    route: true                 # usable as a query filter
    options:
      policy:    Rules the company commits to; what is and isn't allowed
      how_to:    Step-by-step instructions for doing a task
      reference: Specifications, limits, parameters, API fields
      other:     None of the above
  product:
    route: true
    options:
      billing:   Payments, invoices, refunds, plans
      auth:      Sign-in, sessions, tokens, SSO
      api:       Endpoints, SDKs, webhooks, rate limits
      other:     Anything else
questions:                      # optional custom questions, stored as x_<name>
  mentions_deadline:
    type: noul
    instructions: Does this text state a date or time limit that applies to the reader?
```

A tag is stored as `tag_<field>` only when Clef's confidence is at least `tag_min_confidence`; otherwise it's `unknown`. The probability is always stored as `tag_<field>_p`, so you can filter on it later. If an enrichment request fails, the chunk is kept with `unknown` tags and `needs_reenrich=true`.

**Duplicates.** Chunks with identical text within a document are stored once.

### Step 5: Embed

Chunks are embedded in batches with the configured provider.

| Provider | `embedder.model` | Key |
|---|---|---|
| OpenAI | `openai:text-embedding-3-small` | `OPENAI_API_KEY` |
| Vercel AI Gateway | `gateway:openai/text-embedding-3-small` | `AI_GATEWAY_API_KEY` |
| Any OpenAI-compatible API | `openai:<model>` plus `base_url` and `api_key_env` | yours |
| Ollama | `ollama:nomic-embed-text` | none (`OLLAMA_HOST`) |
| Local (sentence-transformers) | `local:BAAI/bge-small-en-v1.5` | none |
| Testing only | `hash:256` (deterministic, no semantic quality) | none |
| Anything else | a Python callable `list[str] -> list[list[float]]` passed as `embedder=` | yours |

Model names are examples; use any model your provider serves. Extra keys under `embedder:` (such as `base_url`, `api_key_env`, `dimensions`, `batch_size`) are passed to the provider.

The first ingest writes a **collection manifest**: embedding provider and model, dimension, distance metric, chunker, taxonomy version and clef-rag version. Later ingests or queries with a different embedding model are refused with a clear error, because mixing embedding models in one collection silently ruins retrieval. To switch models, ingest into a new collection.

### Step 6: Store

Each chunk becomes one record:

| Field | Meaning |
|---|---|
| `id` | UUIDv5 of `doc_id` + chunk content hash: identical on every run and valid in every store |
| vector | The embedding of `embed_text` |
| text | Chunk text shown to the LLM |
| `doc_id`, `source_uri`, `title`, `section_path` | Where it came from |
| `chunk_index`, `char_start`, `char_end`, `tokens` | Position and size, for citations and neighbour expansion |
| `tag_<field>`, `tag_<field>_p` | Taxonomy tag and its probability |
| `q_<name>`, `x_<name>` | Quality and custom question answers |
| `m_<key>` | Your document metadata |
| `quarantined` | Excluded from queries when true |
| `chunker`, `clef_model`, `pipeline_version`, `ingested_at`, `content_hash`, `doc_hash` | Provenance |

Each adapter maps these to native fields (Qdrant payload, Postgres columns and `jsonb`, Chroma metadata, Pinecone metadata) and creates what filtering needs, such as Qdrant payload indexes and Postgres expression indexes.

**Re-ingesting is safe.** Unchanged documents (same text and title) are skipped. A changed document has its current chunks upserted and its stale ones deleted. `--force` reprocesses everything. Removing a document:

```bash
clef-rag delete --collection handbook --doc kb-142
```

### Ingest report

```bash
clef-rag ingest ./docs --collection handbook --dry-run   # parse and estimate Clef requests and cost; no Clef or DB writes
clef-rag ingest ./docs --collection handbook --limit 1   # one document end to end
clef-rag ingest ./docs --collection handbook -v          # everything, with per-document detail
```

Estimate and test one document before a large batch. The report lists documents ingested, skipped and failed, chunks stored, dropped and quarantined, Clef requests (and how many came from the cache), tokens and cost, embedding tokens, time, and any fallback used. A JSON trace per document goes to `.clef-rag/traces/ingest/<collection>/`, including Clef's score at every candidate cut.

## Retrieval guide: querying, classifying and answering

```
query -> 1 route -> 2 recall -> 3 classify -> 4 expand -> 5 gate -> result (-> 6 answer)
```

### Step 1: Route

If your taxonomy has `route: true` fields, one Clef request on the query asks which value of each field the question is about (with an added `any` option), plus whether it's something documents could answer at all rather than small talk or a request to act.

Code turns that into a filter cautiously, because a wrong filter hides the right answer:

- top option's confidence ≥ `min_confidence` → `tag_product = auth`
- else, top two options together ≥ `top2_mass` → `tag_product in [auth, api]`
- else → no filter
- if the filtered search returns fewer than `min_candidates` hits → search again unfiltered

Routing is skipped when you pass a filter yourself. Filters use a portable language that every adapter translates:

```python
rag.retrieve(q, collection="handbook",
             where={"and": [{"eq": {"tag_product": "billing"}},
                            {"gte": {"chunk_index": 2}}]})
```

```bash
clef-rag query "refund window?" --collection handbook --where '{"eq": {"m_team": "billing"}}'
```

Operators: `eq`, `ne`, `in`, `nin`, `gt`, `gte`, `lt`, `lte`, `exists`, `and`, `or`, `not`. Several operators in one dict are ANDed. `ne` and `nin` match only records that have the field. Quarantined records are always excluded unless `retrieve.include_quarantined` is true.

### Step 2: Recall

The query is embedded with the collection's embedding model and the store returns the top `top_k` (default 30). Recall is deliberately wide; the next step does the narrowing.

Scores from every store are normalised to similarity in [0, 1], higher is better, so the numbers mean the same thing on any database. Cosine distance `d` becomes `1 − d`; L2 distance becomes `1 / (1 + d)`. The raw score is kept too.

### Step 3: Classify

Each candidate gets one Clef request whose state is the query and that one passage, with four yes/no questions (adapted from the System One [passage classification cookbook](https://docs.typesafe.ai/cookbooks/classifying_rag_passages)):

| Question | Used for |
|---|---|
| `is_relevant` | Does the passage address the subject of the query? |
| `contains_answer_evidence` | Does it state information usable in a direct answer? |
| `contradicts_query_premise` | Does it conflict with a factual premise in the query? |
| `instructs_ai` | Does it contain instructions addressed to an AI assistant? |

Requests run concurrently under a shared rate limiter. Code then decides, in one of two modes.

**`select: rank` (default).** First match wins:

1. `instructs_ai` ≥ `drop_instructs_ai` → **drop**
2. `is_relevant` ≥ `min_relevant` and `contradicts_query_premise` ≥ `conflict` → **conflict**
3. otherwise → **include**

Included passages are sorted by evidence probability (ties broken by vector score) and the top `max_passages` (default 5) are kept. Nothing is dropped for a low score: upstream, with Jev on public data, dropping low scorers threw away real answers ([upstream results](https://github.com/MersivMedia/jev-rag-retrieval/blob/main/docs/RESULTS.md#public-datasets-m2-scifact-fiqa-qasper)).

**`select: threshold`.** First match wins:

1. `instructs_ai` ≥ `drop_instructs_ai` → **drop**
2. `is_relevant` < `min_relevant` → **drop** (off topic)
3. `contradicts_query_premise` ≥ `conflict` → **conflict**
4. `contains_answer_evidence` ≥ `min_evidence` → **include**
5. otherwise → **drop** (no evidence)

Upstream with Jev, this sent less context (1.5 to 6 passages per query on the public sets) but lost 4 to 22 points of recall. Kept passages are sorted and capped the same way. None of the questions asks "should this be included?"; that decision stays in code, where changing it means editing a number.

Conflicts matter. Asked "Refresh tokens expire after 30 days; how do I extend that?" when the docs say 14 days, the 14-day passage scored 0.95 on `contradicts_query_premise` in a live run and arrived in a separate conflict block, so the LLM can correct the premise instead of going along with it.

### Step 4: Expand (optional)

With `expand_neighbours: true`, an included chunk whose `self_contained` score is below 0.5 brings in the chunks just before and after it (same `doc_id`, adjacent `chunk_index`), within `expand_max_tokens`.

### Step 5: Gate

One final Clef request with the query and the kept passages: do `passages` contain the information needed to answer `query`? Below `answer_min`, the result has `abstain=True` and no LLM is called, so your app can say "That isn't in the documents" instead of letting the model guess.

The gate never abstains because of an error. If Clef fails or times out anywhere in retrieval, you get vector-ranked results marked `degraded=True`.

### The result

```python
result = rag.retrieve("How long do refresh tokens last?", collection="handbook")

result.passages      # included, ranked; each has .text .citation .doc_id .section_path .scores .vector_score
result.conflicts     # passages that contradict the query's premise
result.dropped       # each with .reason ("off_topic", "no_evidence", "instructs_ai", "over_max_passages")
result.abstain       # True if the gate decided the passages can't answer
result.reason        # why it abstained
result.gate_p        # the gate's probability
result.filter_used   # the metadata filter actually applied
result.degraded      # True if a Clef stage failed and fell back
result.trace         # per-stage latency, routing decisions, Clef requests, tokens and cost
result.to_prompt()   # evidence and conflicts as separate, numbered, cited blocks
result.to_dict()     # everything, JSON-serialisable
```

### Step 6: Answer (optional)

```python
answer = rag.answer("How long do refresh tokens last?", collection="handbook")
print(answer.text, answer.citations)
```

`answer()` runs `retrieve()`, returns "I couldn't find that in the documents." if the gate abstained, and otherwise calls the configured LLM with `to_prompt()`. Providers: `openai` (`OPENAI_API_KEY`), `anthropic` (`ANTHROPIC_API_KEY`), `gateway` (`AI_GATEWAY_API_KEY`, models like `openai/gpt-4.1-mini`), or any OpenAI-compatible API via `base_url`. The system prompt tells the model to treat retrieved text as data, not instructions, and to cite passages by number. To use your own LLM stack, call `retrieve()` and pass `to_prompt()` yourself.

### Using an existing collection

```bash
clef-rag query "..." --collection docs --retrieve-only
```

`--retrieve-only` skips the manifest check and routing, so classification and the gate can run on a collection clef-rag didn't build. The collection must use the field layout the adapter expects (for example Qdrant payload `text`, Pinecone metadata `text`); a configurable text field is on the roadmap.

## Vector databases

| Database | `store.kind` | Tested | Notes |
|---|---|---|---|
| Postgres + pgvector | `pgvector` | Conformance suite against Postgres 17 + pgvector | One table per collection; HNSW index with the metric's operator class; expression indexes on filter fields; filters compile to parameterised SQL. `dsn` or `dsn_env` (default `DATABASE_URL`) |
| Qdrant | `qdrant` | Conformance suite in embedded mode | Uses `query_points`; creates payload indexes for filter fields. `url`, or `path` for embedded local mode |
| Chroma | `chroma` | Conformance suite, persistent client | Sets the distance metric explicitly and converts distances to similarity. Existence tests and `ne`/`nin` run in Python after a superset query. `path`, or `host` + `port` |
| Pinecone (experimental) | `pinecone` | **Not yet run against Pinecone; not part of v1.0's tested set** | Collection = namespace in one serverless index (created if missing). Text trimmed to fit the 40 KB metadata limit. Filtered deletes list ids first. `index`, `cloud`, `region` |
| Anything in LangChain | Python only | Conformance suite with `InMemoryVectorStore` | `LangChainStore(your_store)`; filters run in Python over an over-fetched result |
| In-memory | `memory` | Conformance suite | Reference semantics; optional JSON file with `path`. For tests and small demos |

Every adapter except the experimental Pinecone one passes the same conformance suite: round trip, idempotent upsert, replace, delete by id and by filter, the stale-delete pattern, every filter operator (18 cases), score normalisation, empty batches and manifests.

LangChain bridge:

```python
from langchain_core.vectorstores import InMemoryVectorStore   # or any LangChain vector store
from clef_rag import Pipeline
from clef_rag.stores.langchain import LangChainStore, ClefRagEmbeddings
from clef_rag.embed import make_embedder

emb = make_embedder("openai:text-embedding-3-small")
rag = Pipeline(store=LangChainStore(InMemoryVectorStore(ClefRagEmbeddings(emb))), embedder=emb)
```

### Adding a database

Subclass `VectorStore` and implement seven methods:

```python
from clef_rag.stores import VectorStore, Capabilities
from clef_rag.types import Record, Hit

class MyStore(VectorStore):
    kind = "mystore"
    def ensure_collection(self, name, dim, metric, filter_fields): ...
    def upsert(self, collection, records: list[Record]): ...
    def delete(self, collection, ids=None, where=None): ...
    def get(self, collection, ids) -> list[Record]: ...
    def list_ids(self, collection, where=None, limit=100_000) -> list[str]: ...
    def query(self, collection, vector, where, top_k) -> list[Hit]: ...
    def capabilities(self) -> Capabilities: ...
```

`where` arrives in the portable language; `clef_rag.stores.filters.parse()` turns it into a small tree to translate, `push_down_not()` removes `not` for stores that lack it, and `matches()` is the reference evaluator every adapter must agree with. Manifests default to a sidecar file; override `get_manifest`/`put_manifest` to store them natively. Then add your store to the fixture in `tests/stores/test_conformance.py` and make it pass.

## CLI reference

| Command | What it does |
|---|---|
| `clef-rag init --store <kind> --embedder <spec>` | Write a starter `clef-rag.yaml` and, if missing, a blank `.env`. `--full` for every setting; `--force` overwrites the YAML (never `.env`) |
| `clef-rag check` | One Clef call, one embedding, one store round trip |
| `clef-rag ingest <paths...> --collection <name>` | Parse, chunk, enrich, embed, store. `--dry-run`, `--limit N`, `--force`, `-v`, `--json` |
| `clef-rag query "<question>" --collection <name>` | Retrieve. `--where JSON`, `--top-k`, `--no-route`, `--retrieve-only`, `--answer`, `-v`, `--json` |
| `clef-rag inspect <file>` | Show how a file would be chunked. `--compare clef,fixed`, `-v` for Clef cut scores |
| `clef-rag inspect --collection <name>` | List stored records with tags and scores. `--quarantined` |
| `clef-rag delete --collection <name> --doc <doc_id>` | Delete one document's records |
| `clef-rag serve [model]` | Serve Clef weights (HF id or release dir) as a System One endpoint. `--host`, `--port`, `--device`, `--max-length`, `--api-key-env`, `--name` |
| `clef-rag eval <dataset> --collection <name>` | Measure retrieval on labelled data (`beir:<dir>`, `qasper:<file>`, `jsonl:<file>`): records candidates once, then scores vector search, Clef classification, Clef re-ranking and an optional LLM re-ranker (`--llm-rerank MODEL`) on the same candidates. `--dry-run`, `--limit N`, `--split`, `--report-only`, `--compare a,b` |

`-c path/to/clef-rag.yaml` selects a config file; `--env-file path` or `--no-env-file` controls `.env` loading (both go before the command). Set `CLEF_RAG_DEBUG=1` for full tracebacks.

## Cost and limits

| Backend | Price | Limits |
|---|---|---|
| Workers AI `clef` | $0.24 per million input tokens, output free | 65,536-token context; rate limits not published |
| Workers AI `clef-flash` | $0.09 per million input tokens, output free | same |
| `self-hosted` / `local` | your GPU time | one request at a time per loaded model |

Cost scales with `top_k`: each candidate passage is one classification request of roughly 400 tokens plus the passage. With the default `top_k: 30` and passages of about 350 tokens, a query sends roughly 25,000 input tokens. That's about $0.006 on `clef` or $0.002 on `clef-flash` (an estimate from token counts; Clef runs not yet measured). Repeated inputs are served from the on-disk cache at no cost, and the cache is keyed by backend and model, so switching models never reuses another model's answers. `clef-rag ingest --dry-run` estimates a batch before you spend anything.

clef-rag paces requests with a shared limiter (`clef.max_rps`, default 20, since Cloudflare hasn't published Clef's rate limits), retries 408, 429, 5xx and 529 responses with backoff, and honours `retry-after`. Requests over 64 questions or over the backend's token budget are refused before they're sent.

## Testing

```bash
pip install -e ".[qdrant,chroma,pgvector,langchain,dev]"
pytest                                   # offline: fake Clef over httpx.MockTransport, no keys, no network

# local + serve backends against a tiny random Clef-shaped release (CPU, no weight download)
pip install -e ".[serve,dev]" && pip install clef-finetune
clef-finetune make-tiny /tmp/clef-tiny
CLEF_RAG_TEST_TINY_RELEASE=/tmp/clef-tiny pytest tests/test_local_backend.py

# pgvector conformance against a real database
docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=clef_rag pgvector/pgvector:pg17
CLEF_RAG_TEST_PG_DSN=postgresql://postgres:clef_rag@localhost:5432/postgres pytest tests/stores -k pgvector

# live end-to-end against Clef on Workers AI (a fraction of a cent)
CLEF_RAG_LIVE=1 CLOUDFLARE_API_TOKEN=... CLOUDFLARE_ACCOUNT_ID=... OPENAI_API_KEY=... pytest tests/test_live.py
```

CI runs the offline suite on Python 3.10, 3.12 and 3.13, the local/serve backend tests on a tiny random model, and the pgvector conformance suite against a Postgres service container.

## Roadmap

- **Clef numbers.** Rerun the upstream benchmarks (SciFact, FiQA, QASPER, messy documents) with `clef` and `clef-flash`, retune thresholds, and publish them in [Results](https://github.com/MersivMedia/clef-rag/blob/main/docs/RESULTS.md).
- **Packed multi-passage classification.** Clef answers up to 64 questions in one forward pass, so scoring several passages per request should cut latency and tokens.
- **Image-aware ingestion.** Clef reads images; pass page images or figures to screening and tagging.
- **More databases and embedders**, hybrid search, `reenrich` / `reembed`, an MCP server.

## More

- **[Results](https://github.com/MersivMedia/clef-rag/blob/main/docs/RESULTS.md)**: Clef measurements (none yet) and links to the upstream Jev results
- **[Known issues](https://github.com/MersivMedia/clef-rag/blob/main/docs/KNOWN_ISSUES.md)**: current limits
- **[jev-rag-retrieval](https://github.com/MersivMedia/jev-rag-retrieval)**: the upstream project, its PRD and design decisions

clef-rag is not affiliated with Cloudflare. Clef and Clef-flash are © Cloudflare, Inc., released under Apache-2.0. Treat injection screening as one layer of defence, never the only one.

## License

MIT
