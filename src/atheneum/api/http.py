"""HTTP API.

Deliberately narrow: documents are POSTed as text rather than read from a path
by the server. An endpoint that accepts an arbitrary filesystem path turns a
local tool into a remote file reader, and ingestion from disk is already what
the CLI is for.
"""

from __future__ import annotations

import hmac
import os
import re
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

import atheneum
from atheneum.config import Config, load_config
from atheneum.core.types import Document
from atheneum.retrieval.pipeline import Corpus

SearchMode = Literal["hybrid", "lexical", "vector"]

_MAX_DOCUMENT_BYTES = 4 * 1024 * 1024
_MAX_QUERY_CHARS = 8000
_MAX_FIELD_BYTES = 64 * 1024
# \Z, not $: in Python `$` also matches immediately before a trailing newline, so
# a source of "notes.md\n" passed validation and was stored with the newline
# intact, which is exactly the forged-citation case the validator exists to stop.
_SOURCE_RE = re.compile(r"\A[\w][\w./\-+#]{0,511}\Z", re.UNICODE)


class DocumentIn(BaseModel):
    source: str = Field(..., description="Stable identifier for the document, not a server path.")
    content: str = Field(..., min_length=1)
    title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("content")
    @classmethod
    def _validate_content_size(cls, value: str) -> str:
        # Measured in UTF-8 bytes. A character count let 4.19M CJK characters --
        # a 12.6 MB body -- through a limit named _MAX_DOCUMENT_BYTES.
        encoded = len(value.encode("utf-8"))
        if encoded > _MAX_DOCUMENT_BYTES:
            raise ValueError(
                f"content is {encoded} bytes, over the {_MAX_DOCUMENT_BYTES} byte limit"
            )
        return value

    @field_validator("title")
    @classmethod
    def _validate_title_size(cls, value: str | None) -> str | None:
        # The content limit alone let a caller persist an 8 MB title and an 8 MB
        # metadata value; one request grew the database by 34 MB.
        if value is not None and len(value.encode("utf-8")) > _MAX_FIELD_BYTES:
            raise ValueError(f"title must be at most {_MAX_FIELD_BYTES} bytes")
        return value

    @field_validator("metadata")
    @classmethod
    def _validate_metadata_size(cls, value: dict[str, Any]) -> dict[str, Any]:
        import json as _json

        if len(_json.dumps(value, ensure_ascii=False).encode("utf-8")) > _MAX_FIELD_BYTES:
            raise ValueError(f"metadata must be at most {_MAX_FIELD_BYTES} bytes")
        return value

    @field_validator("source")
    @classmethod
    def _validate_source(cls, value: str) -> str:
        # Sources land in citations and logs, so control characters and path
        # separators here would let a client forge or probe identifiers.
        if not _SOURCE_RE.match(value):
            raise ValueError(
                "source must be 1-512 characters of letters, digits, '.', '/', '-', '_' or '+', "
                "starting with a letter or digit"
            )
        if ".." in value:
            raise ValueError("source must not contain '..'")
        if value != value.strip():
            raise ValueError("source must not have leading or trailing whitespace")
        if "//" in value:
            # An empty path segment is non-canonical: "a//b" and "a/b" would be
            # two distinct documents that read as the same source in a citation.
            raise ValueError("source must not contain an empty path segment ('//')")
        return value


class SearchQuery(BaseModel):
    query: str = Field(..., min_length=1, max_length=8000)
    top_k: int = Field(5, ge=1, le=100)
    mode: SearchMode = "hybrid"


class AskQuery(BaseModel):
    query: str = Field(..., min_length=1, max_length=8000)
    top_k: int = Field(5, ge=1, le=100)
    max_turns: int = Field(8, ge=1, le=32)
    provider: str | None = None
    include_evidence: bool = False


class AppContext:
    """Lazily opened corpus, closed when the application shuts down."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._corpus: Corpus | None = None

    @property
    def corpus(self) -> Corpus:
        if self._corpus is None:
            from atheneum.index.bm25 import BM25Params
            from atheneum.retrieval.embedders import build_embedder
            from atheneum.retrieval.pipeline import CorpusConfig
            from atheneum.text.splitter import SplitterConfig

            cfg = self.config
            self._corpus = Corpus.open(
                cfg.db,
                embedder=build_embedder(cfg.embedder, default_dim=cfg.embedder_dim),
                config=CorpusConfig(
                    splitter=SplitterConfig(chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap),
                    bm25=BM25Params(k1=cfg.bm25_k1, b=cfg.bm25_b),
                    fusion=cfg.fusion,
                    fusion_k=cfg.fusion_k,
                    reranker=cfg.reranker,
                ),
            )
        return self._corpus

    def close(self) -> None:
        if self._corpus is not None:
            self._corpus.close()
            self._corpus = None


def create_app(config: Config | None = None) -> Any:
    """Build the FastAPI application.

    Returned as ``Any`` because FastAPI is an optional extra; importing this
    module without it installed raises a clear error at call time rather than
    breaking ``import atheneum``.
    """
    try:
        from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
        from fastapi.exceptions import RequestValidationError
        from fastapi.responses import JSONResponse, StreamingResponse
    except ImportError as exc:  # pragma: no cover - depends on extras
        raise RuntimeError("the HTTP API needs the `api` extra: pip install atheneum[api]") from exc

    settings = config or load_config()
    docs_enabled = not (os.environ.get("ATHENEUM_API_TOKEN") or "").strip()
    app = FastAPI(
        title="Atheneum",
        version=atheneum.__version__,
        summary="Local-first hybrid retrieval and cited answers.",
        lifespan=_lifespan,
        # /docs and /redoc were reachable unauthenticated even with a token set,
        # which publishes the whole schema on any non-loopback bind.
        docs_url="/docs" if docs_enabled else None,
        redoc_url="/redoc" if docs_enabled else None,
        openapi_url="/openapi.json" if docs_enabled else None,
    )
    app.state.context = AppContext(settings)
    # A whitespace-only value would otherwise "enable" auth with a secret of " ",
    # which is worse than no auth because it looks protected.
    token = (os.environ.get("ATHENEUM_API_TOKEN") or "").strip()

    def require_token(authorization: str | None = Header(default=None)) -> None:
        """Optional bearer auth.

        Off by default because the intended deployment is loopback. Setting
        ATHENEUM_API_TOKEN turns it on, which is what a non-loopback bind needs.
        """
        if not token:
            return
        supplied = authorization or ""
        # compare_digest, not `!=`: a plain string comparison short-circuits on
        # the first differing byte, which leaks the prefix length.
        if not hmac.compare_digest(supplied.encode("utf-8"), f"Bearer {token}".encode()):
            raise HTTPException(status_code=401, detail="missing or invalid bearer token")

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        """Report what failed without echoing the payload.

        Pydantic's default handler includes the rejected value, so a 5 MB body
        produced a 5 MB response -- a free amplification primitive, and the whole
        document written into the server's error stream.
        """
        errors = []
        for error in exc.errors():
            location = ".".join(str(part) for part in error.get("loc", ()))
            errors.append(
                {
                    "field": location,
                    "message": error.get("msg", ""),
                    "type": error.get("type", ""),
                }
            )
        return JSONResponse(status_code=422, content={"detail": errors})

    @app.get("/health")
    def health() -> dict[str, Any]:
        # No auth flag: /health is deliberately unauthenticated, so telling an
        # anonymous caller whether the deployment has a token set is free
        # reconnaissance for no operational benefit.
        return {"status": "ok", "version": atheneum.__version__}

    @app.get("/stats", dependencies=[Depends(require_token)])
    def stats() -> dict[str, Any]:
        return dict(app.state.context.corpus.stats())

    @app.post("/documents", status_code=201, dependencies=[Depends(require_token)])
    def add_document(payload: DocumentIn) -> dict[str, Any]:
        document = Document(
            source=payload.source,
            content=payload.content,
            title=payload.title or payload.source,
            metadata=payload.metadata,
        )
        added = app.state.context.corpus.add_document(document)
        return {"source": payload.source, "chunks_added": added, "doc_id": document.id}

    @app.post("/search", dependencies=[Depends(require_token)])
    def search(payload: SearchQuery) -> dict[str, Any]:
        try:
            results = app.state.context.corpus.search(payload.query, top_k=payload.top_k, mode=payload.mode)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"query": payload.query, "mode": payload.mode, "results": [r.as_dict() for r in results]}

    @app.post("/ask", dependencies=[Depends(require_token)])
    def ask(payload: AskQuery) -> dict[str, Any]:
        from atheneum.agent.builtin_tools import build_corpus_tools
        from atheneum.agent.loop import Agent, AgentConfig
        from atheneum.providers.base import ProviderError

        try:
            provider = atheneum.get_provider(payload.provider or settings.provider)
        except KeyError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        agent = Agent(
            provider,
            build_corpus_tools(app.state.context.corpus, default_top_k=payload.top_k),
            config=AgentConfig(max_turns=payload.max_turns, token_budget=settings.token_budget),
        )
        try:
            run = agent.run(payload.query)
        except ProviderError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        body: dict[str, Any] = {
            "answer": run.answer,
            "turns": run.turns,
            "stopped_reason": run.stopped_reason,
            "usage": run.usage.as_dict(),
        }
        if run.error:
            body["error"] = run.error
        if payload.include_evidence:
            body["evidence"] = [
                {"name": result.name, "is_error": result.is_error, "content": result.content}
                for result in run.evidence
            ]
        return body

    # response_class carries the media type into the schema. The function cannot
    # be annotated `-> StreamingResponse`: this module defers annotations and the
    # class is imported inside create_app, so the string never resolves and
    # /openapi.json raised PydanticUserError, which left /docs and /redoc broken
    # in the default no-token configuration.
    @app.get("/ask/stream", dependencies=[Depends(require_token)], response_class=StreamingResponse)
    def ask_stream(
        # Query(...) as a default rather than Annotated: this module defers
        # annotations, and Query is imported inside create_app, so an Annotated
        # form becomes an unresolvable ForwardRef at schema-build time.
        query: str = Query(min_length=1, max_length=_MAX_QUERY_CHARS),
        top_k: int = Query(5, ge=1, le=100),
        max_turns: int = Query(8, ge=1, le=32),
    ) -> Any:
        """Stream an answer.

        The bounds mirror POST /ask on purpose: without them a caller could ask
        for max_turns=999999 and drive an unbounded, client-controlled spend
        against a real provider.
        """
        from atheneum.agent.builtin_tools import build_corpus_tools
        from atheneum.agent.loop import Agent, AgentConfig

        provider = atheneum.get_provider(settings.provider)
        agent = Agent(
            provider,
            build_corpus_tools(app.state.context.corpus, default_top_k=top_k),
            config=AgentConfig(max_turns=max_turns, token_budget=settings.token_budget),
        )
        return StreamingResponse(
            _sse(agent, query), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
        )

    @app.get("/sources", dependencies=[Depends(require_token)])
    def list_sources(limit: int = Query(50, ge=1, le=1000)) -> dict[str, Any]:
        bounded = max(1, min(limit, 1000))
        return {"sources": app.state.context.corpus.sources(limit=bounded)}

    return app


def _sse(agent: Any, query: str) -> Iterator[str]:
    """Render agent events as server-sent events."""
    import json

    for event in agent.stream(query):
        if event.kind == "done":
            payload = {
                "answer": event.run.answer if event.run else "",
                "turns": event.run.turns if event.run else 0,
                "stopped_reason": event.run.stopped_reason if event.run else "",
            }
            yield f"event: done\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
        elif event.kind == "text":
            yield f"event: text\ndata: {json.dumps({'text': event.text}, ensure_ascii=False)}\n\n"
        elif event.kind == "tool_call" and event.tool_call is not None:
            yield (
                "event: tool_call\n"
                f"data: {json.dumps({'name': event.tool_call.name, 'arguments': event.tool_call.arguments}, ensure_ascii=False)}\n\n"
            )


@asynccontextmanager
async def _lifespan(app: Any) -> AsyncIterator[None]:
    """Close the corpus when the server shuts down."""
    try:
        yield
    finally:
        context: AppContext | None = getattr(app.state, "context", None)
        if context is not None:
            context.close()


# Deliberately no module-level `app` object.

# An earlier version had `app = None` here, set only when ATHENEUM_SERVE=1.
# `ath serve` never set that variable, so uvicorn resolved
# "atheneum.api.http:app" to None and every route answered 500 with
# `TypeError: 'NoneType' object is not callable`. The TestClient suite could not
# catch it because it calls create_app() directly and never goes through the
# import string -- the exact gap between "the code is correct" and "the command
# in the README works".
#
# Servers should use the factory: uvicorn --factory atheneum.api.http:create_app
# or, equivalently, `ath serve`.


def build_app() -> Any:
    """Zero-argument ASGI app factory, honouring ATHENEUM_* environment config."""
    return create_app(load_config())
