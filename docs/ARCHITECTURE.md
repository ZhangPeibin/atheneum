# Architecture

Every decision below is traced to evidence. The research method was: clone each
candidate repository shallowly, extract its licence, module layout, dependency
weight, test footprint and the presence of specific implementation techniques,
then delete the clone. **34 repositories** were analysed this way; no clone was
retained.

## 1. What was studied

| Repository | Stars | Licence | Code LOC | Test files | Notes |
| --- | --- | --- | --- | --- | --- |
| langgenius/dify | 154k | Apache-2.0 | 2.23M | 4128 | platform, not an engine |
| lobehub/lobehub | 82k | Apache-2.0 | 2.19M | 3234 | chat UI + plugins |
| infiniflow/ragflow | 90k | Apache-2.0 | 1.41M | 1579 | deepest retrieval; needs ES/Infinity |
| CherryHQ/cherry-studio | 51k | **AGPL-3.0** | 1.29M | 2218 | desktop, FTS5 present |
| onyx-dot-app/onyx | 32k | MIT | 1.14M | 1989 | needs Vespa |
| pydantic/pydantic-ai | 20k | MIT | 482k | **1746 (67%)** | best test discipline found |
| run-llama/llama_index | 52k | MIT | 472k | 1659 | pipeline abstractions |
| openai/openai-agents-python | 29k | MIT | 432k | 427 | guardrails=358 files |
| gptme/gptme | 4.4k | MIT | 430k | 484 | lean CLI agent, has FTS5 |
| langchain-ai/langchain | 145k | MIT | 401k | 1053 | |
| open-webui/open-webui | 151k | BSD-3 | 349k | **3** | huge product, ~no tests |
| FlowiseAI/Flowise | 55k | Apache-2.0 | 311k | 153 | |
| All-Hands-AI/OpenHands | 86k | MIT | 283k | 684 | sandboxing |
| Mintplex-Labs/anything-llm | 65k | MIT | 270k | **57** | bm25 in 1 file, no FTS5 |
| chatboxai/chatbox | 41k | **GPL-3.0** | 269k | 421 | |
| langchain-ai/langgraph | 41k | MIT | 188k | 238 | durable graph execution |
| zylon-ai/private-gpt | 57k | Apache-2.0 | 157k | 236 | |
| deepset-ai/haystack | 26k | Apache-2.0 | 146k | 264 | **RRF reference** |
| SciPhi-AI/R2R | 8k | MIT | 89k | 52 | |
| khoj-ai/khoj | 37k | **AGPL-3.0** | 84k | 64 | |
| microsoft/graphrag | 36k | MIT | 51k | 187 | |
| assafelovic/gpt-researcher | 29k | Apache-2.0 | 46k | 130 | |
| simonw/llm | 12k | Apache-2.0 | **36k** | **44 (38%)** | CLI gold standard |
| huggingface/smolagents | 29k | Apache-2.0 | 31k | 28 | |
| weaviate/Verba | 7.7k | BSD-3 | 20k | **2** | requires Weaviate |
| ItzCrazyKns/Perplexica | 36k | MIT | 17.5k | **0** | Docker-first |
| sigoden/aichat | 10k | Apache-2.0 | 17k | **0** | Rust CLI |
| truefoundry/cognita | 4.4k | Apache-2.0 | 17k | **0** | |
| The-Vibe-Company/quivr | 39k | Apache-2.0 | 8k | 37 | pivoted to a library |
| pathwaycom/llm-app | 59k | MIT | 2k | 0 | |
| dorianbrown/rank_bm25 | 1.4k | Apache-2.0 | **402** | 1 | BM25 reference |
| letta-ai/letta | 25k | Apache-2.0 | — | — | `main` is docs-only; code not analysable |

Two columns matter more than stars here: **test files** and **licence**. A 151k-star
project with three test files and a 12k-star project with a 38% test ratio are
very different things to learn from.

## 2. Evidence → conclusions

### The deployment gap

> **Evidence.** The highest-commented open issues on Perplexica are *"fresh build,
> page loads but working icon in the middle never finishes"* (41 comments),
> *"Perplexica not working on fresh installation"* (33), *"Failed to connect to the
> server"* (29). R2R's equivalent is *"r2r container failed to become healthy"*
> (10). Verba requires Weaviate; Onyx requires Vespa; RAGFlow requires
> Elasticsearch or Infinity plus MySQL/Redis.

**Conclusion.** No Docker, no compose file, no external service. One SQLite file,
`pip install`, run. This is the single highest-leverage decision in the project,
because it is the difference between a product someone tries and one they abandon
at step one.

### The hybrid-search gap

> **Evidence.** anything-llm's open feature requests include *"[FEAT]: RAG with
> Hybrid Search"* (10 comments). Its code contains `bm25` in exactly one file and
> `fts5` in zero. Perplexica contains no BM25, no fusion and no reranking at all.

**Conclusion.** Hybrid retrieval is the default path, not an optional extra.

### Fusion: RRF with k=61

> **Evidence.** `haystack/utils/misc.py::_reciprocal_rank_fusion` uses `k = 61`
> with the comment *"60 was suggested by the original paper, plus 1 as python lists
> are 0-based and the paper used 1-based ranking"*, citing Cormack, Clarke and
> Büttcher, SIGIR 2009. Haystack also ships `distribution_based_rank_fusion`
> (`DocumentJoiner.JoinMode`) as an alternative, and normalises weights to sum to 1
> while rejecting all-zero weights.

**Conclusion.** `retrieval/fusion.py` implements RRF at `k=61` with the same
documented rationale, plus DBSF and weighted-sum as alternatives, plus the same
weight validation. Fusing on rank rather than score is what removes the need to
calibrate an unbounded BM25 score against a bounded cosine similarity.

### Fusion weights had to be measured, not assumed

> **Evidence (our own).** On the bundled evaluation set, equal-weight RRF scored
> MRR **0.802** while lexical-only scored **0.833** — fusion made ranking *worse*,
> because the default hashed embedder is a weak second opinion and equal weights
> gave it equal say. Weighting `bm25=0.7, vector=0.3` scored **0.837** at
> identical recall.

**Conclusion.** Ship `bm25=0.7, vector=0.3` as the default and say why in the
config comment. An earlier version of this project shipped equal weights and an
evaluation dataset too easy to discriminate; both were wrong and both were caught
by making the evaluation adversarial. **A retrieval evaluation whose modes cannot
disagree measures nothing.** The dataset now contains 10 distractor documents and
paraphrase queries chosen for low lexical overlap with their answers.

### BM25 parameters and the negative-IDF floor

> **Evidence.** `dorianbrown/rank_bm25.py` (402 lines total, Apache-2.0) implements
> `BM25Okapi(k1=1.5, b=0.75, epsilon=0.25)` with
> `idf = log((N - df + 0.5) / (df + 0.5))`, flooring negative values at
> `epsilon * average_idf`. It also implements BM25L (`delta=0.5`) and BM25Plus
> (`delta=1`), citing Trotman et al.

**Conclusion.** Implement Okapi with the epsilon floor exactly, because a term
present in over half the corpus would otherwise *penalise* the documents
containing it. Skip BM25L and BM25Plus: none of the 30 product repositories
studied use them, so they add surface without demonstrated value.

**Deviations from the reference, both deliberate:**

1. `rank_bm25.get_top_n` uses `np.argsort(scores)[::-1][:n]` — a full O(n log n)
   sort. `index/bm25.py` uses `np.argpartition` for O(n) pre-selection and sorts
   only the winning slice.
2. `rank_bm25.BM25._tokenize_corpus` spawns `multiprocessing.Pool(cpu_count())`.
   For a corpus of a few hundred chunks that costs more than it saves, and a
   tokenizer that is a closure cannot be pickled at all under macOS spawn
   semantics. Tokenization happens in-process.

Both deviations were validated by the test suite: an early version of
`_top_k` used `np.lexsort` for tie-breaking, which returns *indices into the key
arrays* rather than the values, silently scrambling result order. 24 BM25 tests
caught it.

### Dense vectors without a vector database

> **Evidence.** `simonw/llm`'s `Collection.similar_by_vector` registers a Python
> function `distance_score` into SQLite and calls it per row:
> `order by distance_score(embedding) desc limit N`. That is one Python-level call
> per candidate per query. Verba depends on Weaviate (48 files reference a vector
> DB); Onyx on Vespa; RAGFlow on Elasticsearch or Infinity.

**Conclusion.** Store vectors as packed little-endian float32 blobs in an ordinary
column, load them into one contiguous numpy matrix, L2-normalize at insert time so
cosine becomes a dot product, and answer a query with a single matrix-vector
product. Measured on 322 chunks at 512 dimensions: **0.12 ms mean, 0.137 ms p95**.
That is exact search at a latency where an ANN index would be pure overhead.

### Chunking

> **Evidence.** `llama_index/core/node_parser/text/sentence.py` uses
> `SENTENCE_CHUNK_OVERLAP = 200`, `DEFAULT_PARAGRAPH_SEP = "\n\n\n"`, and
> `CHUNKING_REGEX = "[^,.;。？！]+[,.;。？！]?|[,.;。？！]"` — note the CJK
> punctuation. Splitting cascades paragraph → regex → character → sentence
> tokenizer. `SentenceSplitter.__init__` raises if `chunk_overlap > chunk_size`.

**Conclusion.** `text/splitter.py` uses the same cascade (heading → paragraph →
sentence → hard cut), the same 200-character overlap default, and the same
construction-time validation. CJK terminators `。！？；` are first-class in the
sentence regex. Two additions: complete fenced code blocks are emitted whole even
past the budget, because a fragment of a function body is worse than a long chunk;
and `respect_code_fences=False` opts out.

### Tokenization

> **Evidence.** SQLite FTS5's `unicode61` tokenizer splits on whitespace and
> punctuation, so Han text is not segmented at all — it becomes one token per
> contiguous run. `gptme` is the only lean CLI studied that uses FTS5 (2 files).
> `llama_index`'s chunking regex explicitly includes CJK punctuation.

**Conclusion.** Own the tokenizer. `text/tokenizer.py` emits word tokens for Latin
runs and **unigrams plus overlapping bigrams** for CJK runs, because a single Han
character is usually too ambiguous to rank on. Normalization is NFKC (folds
full-width `ＦＴＳ５` → `fts5`) then NFD with combining marks stripped (folds
`CAFÉ` → `cafe`), then NFC. An earlier version used NFKC alone and silently failed
to match accented queries; a test caught it.

Owning ~150 lines of tokenizer buys correct Chinese and Japanese retrieval and
removes the FTS5 dependency. The trade-off is documented in the README.

### The agent loop

> **Evidence.** `openai/openai-agents-python/src/agents/run.py` runs `while True`
> (line 967) with `max_turns: int | None = DEFAULT_MAX_TURNS` and raises
> `MaxTurnsExceeded` (line 1491). It carries `InputGuardrailTripwireTriggered` /
> `OutputGuardrailTripwireTriggered`, and `_safe_redacted_persistence_error` for
> error redaction. But `run`, `run_sync` and `run_streamed` are three
> near-identical copies of a very long function — the docstrings at lines 263, 368
> and 470 are word-for-word duplicates.
>
> `pydantic-ai` models a failed tool call as a `RetryPromptPart` *message*, i.e.
> the error is fed back to the model. `smolagents`' most-commented open issue is
> *"Repeatedly getting Error in code parsing: Your code snippet is invalid"* (39
> comments) — what happens when that feedback path is fragile.
>
> `letta`'s top issue is *"sliding_window compaction ignores percentage, performs
> full context wipe"* (15 comments).
>
> `langgraph`'s most-commented issues concern cancellation losing streamed state
> and long tool calls being silently re-executed from a checkpoint.

**Conclusion.**
- Bounded by `max_turns`; the loop cannot hang. **Deviation:** exhausting turns
  returns an `AgentRun` with `stopped_reason="max_turns"` rather than raising, so a
  caller always gets whatever was produced. The failure is reported in the result,
  not thrown through it.
- Tool errors are data. `ToolRegistry.execute` catches every exception and returns
  a `ToolResult(is_error=True)` whose content names the error type and tells the
  model what to do next. The run continues.
- **One loop body, two views.** `Agent.run` and `Agent.stream` share the same
  termination, tool-execution and memory logic rather than duplicating it.
- Durable checkpointing, graph orchestration and cancellation semantics are
  deliberately **not** built. `langgraph`'s issue backlog shows how much complexity
  they carry, and none of it is needed for a local research assistant.

### Context compaction

> **Evidence.** letta's *"sliding_window compaction ignores percentage, performs
> full context wipe"* is a 15-comment bug report about exactly the failure mode a
> sliding window invites.

**Conclusion.** `agent/memory.py::compact` is defined as: keep a verbatim tail
under `keep_ratio` of `token_budget`, summarize the rest into one synthetic system
message, and **always retain at least the final message** so the window can never
be empty. Both invariants are pinned by tests, including a case where a single
message exceeds the whole budget. When the budget is contested, digest lines are
selected by information value — retrieved findings first, then tool-request
records, then restated prose — because dropping a retrieved number changes the
answer while dropping a restated question does not.

Token estimation weights CJK at one character per token rather than the usual four,
since a Han character is frequently a whole token.

### Tool schemas

> **Evidence.** `pydantic-ai` derives tool schemas from typed signatures (200 files
> touch `tool_schema`). `simonw/llm` discovers plugins through `pluggy` and
> setuptools entry points in the group `"llm"`, and guards test isolation with
> `if not hasattr(sys, "_called_from_test")` — a private attribute the test suite
> monkeypatches.

**Conclusion.** `agent/tools.py` derives JSON Schema from the signature and
annotations, so there is no second place to keep in sync. Discovery uses the
`atheneum.providers` entry-point group, but the test-isolation switch is an
explicit `ATHENEUM_NO_PLUGINS=1` environment variable rather than a patched
`sys` attribute, which does not leak.

Two rules that came out of bugs found during development:
- An **unannotated parameter is a hard error**, not a default to `"string"`.
  Defaulting silently rewrites `4` into `"4"`, which turned `x * 2` into string
  repetition with nothing reporting it.
- If annotations cannot be resolved and are still strings, raise rather than emit a
  schema claiming every argument is a string. A missing `get_type_hints` import was
  swallowed by a broad `except Exception` and degraded **every** tool schema in the
  package to `"string"`; no test noticed until one asserted the type. That is why
  the assertion exists.

### The offline provider

> **Evidence.** `pydantic-ai`'s `TestModel` is a deterministic `Model` subclass
> with `call_tools='all'`, `custom_output_text`, `seed=0` and `__test__ = False`,
> and it ships inside the library rather than in `tests/`. Meanwhile Perplexica has
> 0 test files, Verba 2, open-webui 3, anything-llm 57 — all products that need
> live services to exercise their own main path.

**Conclusion.** Promote that pattern from test fixture to **shipped default
provider**. `providers/offline.py` is a deterministic extractive engine that
implements the real `Provider` protocol: it requests the `search` tool on turn one,
parses the results on turn two, selects sentences by query-term coverage, dedupes
them, and emits numbered citations with a source list.

The payoff is that the entire product — ingest → chunk → tokenize → embed → index
→ fuse → agent loop → cited answer — runs and is regression-tested with no API key,
no network and no secrets in CI. Two runs are byte-identical. This required
replacing a `hash()`-based tool-call id with a blake2b digest, because CPython salts
`hash()` per process and the reproducibility claim was otherwise false.

### Provider abstraction

> **Evidence.** khoj's issue *"DeepSeek-compatible OpenAI API fails due to
> unsupported response_format type"* (7 comments) shows that "OpenAI-compatible"
> means "close enough to break in specific ways". Anthropic's Messages API takes
> `system` as a top-level parameter and returns tool results as content blocks
> inside a user message.

**Conclusion.** One `OpenAICompatibleProvider` covers OpenAI, Azure, DeepSeek,
Groq, Mistral, Together, OpenRouter, Moonshot, DashScope, Ollama, LM Studio and
vLLM, parameterised by base URL and model. Per-vendor quirks are recorded in
`PROVIDER_PROFILES[*].capabilities` rather than discovered at runtime. Anthropic
gets its own provider because the wire format genuinely differs. Keys come from the
environment and are never written to disk.

### HTTP API

> **Evidence.** letta has an issue titled *"MCP SSRF protection needs allow-list
> for Docker/local development"*; khoj has *"[FIX] CORS issue"*. anything-llm's top
> request is *"[FEAT]: Fine-Grained Access controls"* (27 comments).

**Conclusion.** `/documents` accepts **text, not a filesystem path**. A server-side
path parameter turns a local tool into a remote file reader, and ingestion from
disk is already the CLI's job. Bearer auth is available via `ATHENEUM_API_TOKEN`;
the default bind is loopback.

## 3. What was deliberately not built

| Skipped | Why |
| --- | --- |
| Graph retrieval (GraphRAG-style community summaries) | 51k LOC in `microsoft/graphrag` for a technique that needs LLM calls at index time; it cannot work offline |
| Durable graph execution / checkpointing | `langgraph`'s issue backlog is the cost; not needed for a local assistant |
| Web UI | `open-webui` (349k LOC, 3 tests) and `lobehub` (2.19M LOC) show how much surface it adds. The CLI and HTTP API cover the same ground |
| Document parsers (PDF, DOCX, OCR) | RAGFlow's `deepdoc` is a large subsystem of its own. Text-only is stated as a limitation, not hidden |
| Multi-agent orchestration | No evidence it improves single-question grounded answering |
| Vector database integration | See "Dense vectors without a vector database" |
| MCP client | `simonw/llm`'s *"ability to use MCP servers"* is a 17-comment request; it is a clean extension point (`ToolRegistry`) but not v0.1 |
| Stemming / lemmatization | Reduces `bursts`→`burst`, but merges unrelated words and degrades precision silently. **Cost is real and measured**: `tokenize("burst size allowed")` shares zero terms with `tokenize("allowing bursts to 200 requests")`, so morphological variants are carried only by the dense retriever. Documented as a limitation rather than hidden |

## 4. Licence analysis

Of the 34 repositories studied: 15 Apache-2.0, 12 MIT, 3 BSD-3-Clause,
2 AGPL-3.0 (khoj, cherry-studio), 1 GPL-3.0 (chatbox), 2 NOASSERTION/custom.

**Atheneum is Apache-2.0.** No code was copied from any repository. What was taken
is:

- **Published algorithms**: Okapi BM25 and its epsilon floor (Robertson & Walker;
  Trotman et al.), Reciprocal Rank Fusion (Cormack, Clarke & Büttcher, SIGIR 2009),
  hierarchical sentence splitting. These are academic results, not copyrightable
  expression.
- **Architectural patterns**: entry-point plugin discovery, provider profiles,
  error-as-tool-result, deterministic test-double providers.
- **Measured defaults**: `k1=1.5`, `b=0.75`, `epsilon=0.25`, `k=61`,
  `chunk_overlap=200`.

Nothing was taken from the AGPL-3.0 or GPL-3.0 projects beyond observing their
existence, licence, and issue counts.

## 5. Module map

```
src/atheneum/
├── core/types.py          Document, Chunk, Message, ToolCall, ToolResult — no I/O, no deps
├── text/tokenizer.py      CJK-aware terms: word tokens + Han bigrams, NFKC→NFD→NFC folding
├── text/splitter.py       heading → paragraph → sentence → hard cut, code fences atomic
├── index/bm25.py          Okapi BM25 over an inverted index, argpartition top-k
├── index/vectors.py       packed float32 matrix, normalized rows, one matmul per query
├── index/store.py         SQLite schema + migrations, thread-safe, WAL
├── ingest.py              file discovery, encoding detection, binary rejection
├── retrieval/embedders.py hashing (default, offline) + OpenAI/Ollama/ST backends
├── retrieval/fusion.py    RRF (k=61), DBSF, weighted sum
├── retrieval/rerank.py    coverage re-scorer (free) + cross-encoder (opt-in)
├── retrieval/pipeline.py  Corpus: the only class most callers need
├── providers/base.py      Provider protocol, GenerationRequest, Generation, Usage
├── providers/offline.py   deterministic extractive engine — the shipped default
├── providers/openai_compat.py  12 vendors through one implementation
├── providers/anthropic.py Messages API, content blocks, tool_use/tool_result
├── providers/registry.py  profiles + entry-point plugin discovery
├── agent/tools.py         signature → JSON Schema, validation, error-as-result
├── agent/loop.py          bounded turns, approval gate, one body for run + stream
├── agent/memory.py        token estimation, budgeted compaction, value-ranked digest
├── agent/builtin_tools.py search / read_chunk / read_source / list_sources / corpus_info
├── evaluate.py            labelled dataset, recall@k, precision@k, MRR, benchmark
├── config.py              defaults < file < env < flags
├── api/http.py            FastAPI, bearer auth, SSE streaming
└── cli.py                 ath index|search|ask|chat|serve|eval|bench|...
```

Hard dependencies are **click** and **numpy**. `httpx`, `fastapi` and `uvicorn` are
extras. Nothing else is imported at startup — `sentence_transformers` and `torch`
are loaded lazily inside the reranker that needs them, so `import atheneum` never
pays for them.
