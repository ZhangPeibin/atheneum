# Atheneum

A **local-first AI research engine**: hybrid retrieval (Okapi BM25 + dense vectors, fused with
Reciprocal Rank Fusion), an agentic tool loop, and answers that cite the passages they came from.

It installs with one command and needs **nothing else** — no Docker, no Postgres, no Elasticsearch,
no vector database, no GPU, and **no API key**.

```bash
pip install atheneum
ath index ~/notes
ath ask "what did I decide about rate limiting?"
```

The last line works on a laptop on a plane. Atheneum ships a deterministic **offline provider** that
speaks the same protocol as a hosted model: it requests the `search` tool, reads the results, and
writes an extractive answer with numbered citations. Configure a real model when you want generative
fluency; the retrieval, agent loop, memory and citations are identical either way.

## Why this exists

Atheneum is a response to a specific gap, not a general-purpose framework. Researching 34 open-source
AI products and engines surfaced the same three complaints repeatedly:

| What users report | Where | What Atheneum does |
| --- | --- | --- |
| "fresh build, page loads but working icon never finishes" (41 comments), "not working on fresh installation" (33), "Failed to connect to the server" (29) | [Perplexica issues](https://github.com/ItzCrazyKns/Perplexica) · also "r2r container failed to become healthy" in [R2R](https://github.com/SciPhi-AI/R2R) | No containers. One SQLite file. `pip install` and it runs. |
| "[FEAT]: RAG with Hybrid Search" (10 comments) | [anything-llm issues](https://github.com/Mintplex-Labs/anything-llm) | Hybrid retrieval is the default path, not a plugin. |
| "sliding_window compaction ignores percentage, performs full context wipe" | [letta issues](https://github.com/letta-ai/letta) | Context compaction is defined as a fraction of a token budget and pinned by tests. |

And in the code, the same trade-off appears over and over: the engines with the best retrieval are
enormous and service-dependent (RAGFlow ~1.4M lines of code requiring Elasticsearch or Infinity;
Onyx ~1.1M requiring Vespa; Verba requiring Weaviate), while the lean, lovable CLIs do no retrieval
at all (`simonw/llm` is ~36k lines but has no chunker and no agent loop, and does dense search with a
per-row SQLite UDF).

**Atheneum's bet is that the middle of that graph is unoccupied and underserved.**

## What you get

- **Hybrid retrieval.** Okapi BM25 over an inverted index plus dense vector search, fused by RRF, with
  optional re-ranking. Every hit reports which retriever contributed to it.
- **Cited answers.** Every passage carries `source#chunk` coordinates, so an answer can be verified
  rather than trusted.
- **Agentic tool loop.** The model calls `search`, `read_chunk`, `read_source`, `list_sources`,
  `corpus_info`. Tool failures are returned to the model as error results so it can recover, and the
  loop is bounded by `max_turns`.
- **Provider-agnostic.** One OpenAI-compatible implementation covers OpenAI, DeepSeek, Groq, Mistral,
  Together, OpenRouter, Moonshot, DashScope, vLLM, LM Studio and Ollama. Anthropic has its own
  provider because the wire format genuinely differs.
- **Multilingual by default.** CJK text is indexed as character bigrams and the splitter understands
  `。！？；` — most BM25 implementations silently produce garbage for Chinese and Japanese.
- **A real HTTP API** with bearer auth available, and a streaming endpoint.
- **Fully offline-testable.** The deterministic provider means CI needs no secrets.

## Install

```bash
pip install atheneum            # core: click + numpy, nothing else
pip install atheneum[net]       # + httpx, for hosted model providers
pip install atheneum[api]       # + fastapi/uvicorn, for the HTTP server
```

Requires Python 3.11+.

## Quickstart

```bash
# 1. Index a directory
ath index ~/docs --pattern '*.md'

# 2. Look at what retrieval found, without generating anything
ath search "rate limiting" --explain

# 3. Ask a question; the agent searches, reads, and cites
ath ask "how should we throttle the public API?"

# 4. Multi-turn
ath chat

# 5. Serve it
ath serve --port 8077
```

```python
import atheneum

corpus = atheneum.Corpus.open("mycorpus.db")
corpus.add_text("decisions/001.md", open("decisions/001.md").read())

for hit in corpus.search("queue depth limits", top_k=5):
    print(f"{hit.score:.4f}  {hit.chunk.source}#{hit.chunk.ordinal}")

from atheneum.agent.builtin_tools import build_corpus_tools
from atheneum.agent.loop import Agent

agent = Agent(atheneum.get_provider("offline"), build_corpus_tools(corpus))
run = agent.run("what did we decide about queue depth?")
print(run.answer)
print(run.stopped_reason, run.turns, run.usage.total_tokens)
```

## How retrieval works

```
query ──┬──► tokenizer (CJK-aware) ──► BM25 inverted index ──┐
        │                                                     ├──► RRF fusion ──► [rerank] ──► top-k
        └──► embedder ──► dense vector matrix ────────────────┘
```

**BM25.** Okapi with `k1=1.5`, `b=0.75`, and the negative-IDF floor at `epsilon=0.25 × average_idf`
so that terms appearing in more than half the corpus cannot penalise the documents holding them.
Scoring gathers postings for the query terms only, so cost scales with matching chunks rather than
corpus size, and top-k uses `argpartition` instead of a full sort.

**Dense vectors.** Stored as packed float32 rows in SQLite and loaded into one contiguous numpy
matrix, so a query is a single matrix-vector product — no per-row function call per candidate. Rows
are L2-normalized at insert time, which turns cosine similarity into a dot product.

**Fusion.** Reciprocal Rank Fusion with `k=61`. 60 is the constant recommended by Cormack, Clarke and
Büttcher ([SIGIR 2009](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)); one is added
because ranks here are zero-based. Fusing on *rank* rather than *score* is what lets an unbounded BM25
score and a bounded cosine similarity share an ordering without calibration. `dbsf` (distribution-based
score fusion) and `weighted` are available when you do want magnitude to matter.

**Chunking.** Hierarchical: headings → paragraphs → sentences → characters, with a 200-character
overlap and code fences kept intact. Overlap larger than chunk size is rejected at construction,
because that combination makes packing loop forever.

**Re-ranking.** Opt-in. `overlap` is a free deterministic coverage re-scorer; `cross-encoder` uses
sentence-transformers if you installed it.

### Why not SQLite FTS5?

FTS5's `bm25()` is excellent and would remove this module entirely, but its `unicode61` tokenizer does
not segment Han characters into words, and swapping in a real segmenter reintroduces a dependency.
Owning the scorer costs ~250 lines, gives identical ranking for Latin text, and stays correct for
Chinese and Japanese. `index/bm25.py` is where to look if you disagree.

### Why not a vector database?

Brute-force cosine over a normalized float32 matrix is a single BLAS call. On a corpus of 50k chunks
at 512 dimensions that is tens of milliseconds, and it is exact. A vector DB buys you recall at
millions of chunks and costs you a service to operate, which is the thing this project exists to
avoid. Past roughly 200k chunks, swap `index/vectors.py` for `hnswlib` or `sqlite-vec` — the interface
is 30 lines.

## Providers

`ath providers` lists these and reports which are usable right now. Keys are read from the
environment and **never written to disk**.

| Name | Default model | Environment |
| --- | --- | --- |
| `offline` *(default)* | deterministic extractive | — |
| `openai` | `gpt-4o-mini` | `OPENAI_API_KEY` |
| `anthropic` | `claude-sonnet-4-5` | `ANTHROPIC_API_KEY` |
| `deepseek` | `deepseek-chat` | `DEEPSEEK_API_KEY` |
| `groq` | `llama-3.3-70b-versatile` | `GROQ_API_KEY` |
| `mistral` | `mistral-large-latest` | `MISTRAL_API_KEY` |
| `together` | `meta-llama/Llama-3.3-70B-Instruct-Turbo` | `TOGETHER_API_KEY` |
| `openrouter` | `openai/gpt-4o-mini` | `OPENROUTER_API_KEY` |
| `moonshot` | `moonshot-v1-8k` | `MOONSHOT_API_KEY` |
| `dashscope` | `qwen-plus` | `DASHSCOPE_API_KEY` |
| `ollama` | `qwen2.5:7b` @ `127.0.0.1:11434/v1` | — |
| `lmstudio` / `vllm` | local server | — |

Third-party packages can register providers through the `atheneum.providers` entry-point group.
Set `ATHENEUM_NO_PLUGINS=1` to disable discovery.

```bash
export OPENAI_API_KEY=sk-...
ath ask "summarise the incident reviews" -m openai
ath ask "same question" -m ollama --stream
```

## Configuration

Precedence is fixed: **built-in defaults < config file < environment < CLI flags**.

```bash
ath config            # show what won, and from where
ath init              # write a starter config
```

```bash
ATHENEUM_FUSION=dbsf ATHENEUM_CHUNK_SIZE=800 ath index ~/notes
```

Config file lives at `$XDG_CONFIG_HOME/atheneum/config.json` (override with `ATHENEUM_CONFIG_DIR`).
The corpus is one file at `$ATHENEUM_DATA_DIR/corpus.db`.

## HTTP API

```bash
ATHENEUM_API_TOKEN=$(openssl rand -hex 16) ath serve --port 8077
```

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | liveness, no auth |
| GET | `/stats` | corpus size and active settings |
| POST | `/documents` | add a document as text |
| POST | `/search` | ranked passages with score breakdown |
| POST | `/ask` | cited answer, `include_evidence` optional |
| GET | `/ask/stream` | server-sent events |
| GET | `/sources` | list indexed documents |

`/documents` takes text, **not a filesystem path**, on purpose: a server-side path parameter turns a
local tool into a remote file reader. Ingesting from disk is what the CLI is for. Bearer auth is off
until `ATHENEUM_API_TOKEN` is set, which is safe because the default bind is loopback.

## Evaluation

The package ships a labelled retrieval dataset so the hybrid claim is checkable:

```bash
python -m atheneum.evaluate   # or: ath eval
```

Reports recall@k, precision@k, MRR, hit rate and mean latency for `hybrid`, `lexical` and `vector`
over the same queries on the same corpus. Current numbers on the bundled 18-document / 26-query set
(10 of them distractors or paraphrases, added specifically so the modes can disagree):

```
MODE       recall@5   prec@5     MRR     hit
hybrid        0.923    0.200   0.837   0.923
lexical       0.923    0.200   0.833   0.923
vector        0.885    0.192   0.766   0.885
```

The first four columns are deterministic and reproduce exactly on any machine.
`ath eval` also prints a latency column, but that one is machine- and
run-dependent — the same hybrid evaluation measured between 0.16 ms and 0.27 ms
across runs here — so it is reported live rather than quoted as a fixed figure.
For throughput on your own hardware use `ath bench PATH...`; on this machine it
ingested 322 chunks at ~1170 chunks/s with a mean query time of 0.12 ms.

Read that honestly: with the default hashed embedder, hybrid wins by a **small** margin on MRR and
ties on recall. The margin is real but it is not dramatic, and it would vanish entirely on a corpus
whose documents are trivially separable — which is exactly what an earlier version of this dataset
was, and why it now contains distractors. `ath bench PATH...` measures ingestion throughput and query
latency against your own files.

## Design decisions, and what they were taken from

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full evidence chain — every decision there
cites the repository or paper it came from. Highlights:

- **Bounded turns, and tool errors returned as data.** Both are consensus across `pydantic-ai`,
  `openai-agents-python` and `smolagents`; `smolagents` issue #39 ("repeatedly getting Error in code
  parsing", 39 comments) is what happens when they are not.
- **One loop, three views.** `openai-agents-python` carries near-identical `run` / `run_sync` /
  `run_streamed` implementations. Here the streaming and non-streaming paths share one loop body.
- **A deterministic provider as a first-class citizen, not a test fixture.** Modelled on
  `pydantic-ai`'s `TestModel`, promoted from `tests/` to the shipped default.
- **Test coverage as a feature.** `open-webui` has ~348k lines and 3 test files; Perplexica has 0.
  Atheneum's tests run in about a second and need no secrets.

## Limitations

Be clear-eyed before you adopt this:

- **The offline provider does not generate prose.** It selects and cites sentences. It is correct and
  traceable, and it is not a substitute for a language model on questions needing synthesis.
- **`HashingEmbedder` is not semantic.** It scores term and phrase overlap. On the bundled evaluation
  the two queries it fails outright are pure paraphrases with no shared content words ("how do you
  combine results from two different search systems" → the rank-fusion document). A neural embedder
  (`--embedder openai|ollama|sentence-transformers`) is what fixes those.
- **Fusion weights had to be tuned against measurement, not assumed.** Equal-weight RRF scored MRR
  0.802 here against 0.833 for lexical alone — fusing in a weak second opinion made ranking *worse*.
  The shipped default of `bm25=0.7, vector=0.3` scores 0.837. If you swap in a strong neural
  embedder, raise the vector weight; the default is tuned for the embedder that ships with it.
- **No stemming or lemmatization.** `ath search "burst limit"` will not lexically match a document
  saying "bursts", because the tokenizer folds case and accents but does not reduce words to a root.
  Verified: `tokenize("burst size allowed")` and `tokenize("allowing bursts to 200 requests")` share
  **zero** terms, so such a query is carried entirely by the dense retriever. This is a deliberate
  trade — a stemmer is fast and occasionally merges unrelated words, which degrades precision
  silently — but it is a real gap for morphologically rich queries.
- **Vector search is exact brute force.** Fine to a few hundred thousand chunks; beyond that, or at
  high dimension, bring an ANN index.
- **Single-writer SQLite.** Concurrent readers are fine (WAL); concurrent ingesters are not.
- **PDF, DOCX and images are not parsed.** Text formats only. A document parser is a documented
  extension point, not a TODO.
- **No web search tool.** Everything here is over what you have indexed.

## License

Apache-2.0. BM25, RRF and the splitting strategies are published algorithms and academic results,
implemented from their specifications. No code was copied from any repository, and nothing was taken
from the AGPL-3.0 or GPL-3.0 projects examined during design.
