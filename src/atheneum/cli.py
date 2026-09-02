"""The ``ath`` command line interface.

Designed around one rule: the first command a curious person types should work
and produce something worth reading, with no configuration. That is why
``ath index . && ath ask "..."`` answers from the local ``offline`` provider
instead of failing on a missing API key.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import click

import atheneum
from atheneum.config import Config, load_config, save_config
from atheneum.index.bm25 import BM25Params
from atheneum.retrieval.pipeline import Corpus, CorpusConfig
from atheneum.text.splitter import SplitterConfig

__all__ = ["main"]

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"], "max_content_width": 100}


@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option(atheneum.__version__, prog_name="atheneum")
@click.option("--db", type=click.Path(), default=None, help="Corpus database file.")
@click.option("-v", "--verbose", count=True, help="Increase logging detail.")
@click.pass_context
def cli(ctx: click.Context, db: str | None, verbose: int) -> None:
    """Atheneum: local-first hybrid retrieval and cited answers.

    \b
      ath index PATH...      Build the corpus from files on disk
      ath search QUERY       Retrieve passages only
      ath ask QUERY          Run the agent and answer with citations
      ath chat               Interactive session
      ath serve              Serve the HTTP API
    """
    level = logging.WARNING - min(verbose, 2) * 10
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")
    config = load_config()
    ctx.obj = {"config": config, "db": db or config.db, "verbose": verbose}


def _opts(ctx: click.Context) -> dict[str, Any]:
    return ctx.obj or {}


def _open_corpus(ctx: click.Context, *, create: bool = True) -> Corpus:
    from atheneum.retrieval.embedders import build_embedder

    options = _opts(ctx)
    config: Config = options["config"]
    db = Path(options["db"])
    if not db.exists() and not create:
        raise click.ClickException(f"no corpus at {db}. Run `ath index PATH...` first.")
    embedder = build_embedder(config.embedder, default_dim=config.embedder_dim)
    corpus_config = CorpusConfig(
        splitter=SplitterConfig(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap),
        bm25=BM25Params(k1=config.bm25_k1, b=config.bm25_b),
        fusion=config.fusion,
        fusion_k=config.fusion_k,
        reranker=config.reranker,
    )
    return Corpus.open(db, embedder=embedder, config=corpus_config)


# -- index ------------------------------------------------------------------
@cli.command()
@click.argument("paths", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--pattern", "-p", multiple=True, help="Only index files matching this glob.")
@click.option("--exclude", multiple=True, help="Directory or file name patterns to skip.")
@click.option("--rebuild", is_flag=True, help="Drop existing chunks and re-chunk from stored documents.")
@click.option("--fresh", is_flag=True, help="Delete the database file first and start empty.")
@click.option("--chunk-size", type=int, default=None, show_default=True)
@click.option("--chunk-overlap", type=int, default=None, show_default=True)
@click.option("--limit", type=int, default=None, help="Stop after this many files.")
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable output.")
@click.pass_context
def index(
    ctx: click.Context,
    paths: tuple[str, ...],
    pattern: tuple[str, ...],
    exclude: tuple[str, ...],
    rebuild: bool,
    fresh: bool,
    chunk_size: int | None,
    chunk_overlap: int | None,
    limit: int | None,
    as_json: bool,
) -> None:
    """Index files or directories into the corpus."""
    options = _opts(ctx)
    config: Config = options["config"]
    db = Path(options["db"])

    if fresh and db.exists():
        db.unlink()
        for sidecar in (Path(str(db) + "-wal"), Path(str(db) + "-shm")):
            with contextlib.suppress(FileNotFoundError):
                sidecar.unlink()

    if chunk_size is not None:
        config.chunk_size = chunk_size
        if config.chunk_overlap >= chunk_size:
            # Shrinking below the configured overlap is a normal thing to want, so
            # adapt rather than fail with a validation error that looks like a bug.
            config.chunk_overlap = max(0, chunk_size // 2)
            if not as_json:
                click.echo(
                    f"note: chunk overlap reduced to {config.chunk_overlap} to fit chunk size {chunk_size}",
                    err=True,
                )
    if chunk_overlap is not None:
        config.chunk_overlap = chunk_overlap

    corpus = _open_corpus(ctx)
    try:
        if rebuild:
            stats = corpus.rebuild()
            if not as_json:
                click.echo(f"rebuilt index: {stats['chunks']} chunks from {stats['documents']} documents")
            return
        added = corpus.add_paths(
            paths,
            patterns=tuple(pattern) or ("*",),
            exclude=tuple(exclude),
            limit=limit,
        )
        stats = corpus.stats()
    finally:
        corpus.close()

    if as_json:
        click.echo(json.dumps({"added_chunks": added, **stats}, indent=2, sort_keys=True))
    else:
        click.echo(f"indexed {added} new chunks")
        click.echo(f"corpus: {stats['documents']} documents, {stats['chunks']} chunks")
        click.echo(f"database: {db}")


@cli.command(name="rebuild")
@click.option("--chunk-size", type=int, default=None)
@click.option("--chunk-overlap", type=int, default=None)
@click.pass_context
def rebuild_command(ctx: click.Context, chunk_size: int | None, chunk_overlap: int | None) -> None:
    """Re-chunk and re-embed every stored document."""
    config: Config = _opts(ctx)["config"]
    if chunk_size is not None:
        config.chunk_size = chunk_size
        if config.chunk_overlap >= chunk_size:
            # Shrinking below the configured overlap is a normal thing to want, so
            # adapt rather than fail with a validation error that looks like a bug.
            config.chunk_overlap = max(0, chunk_size // 2)
            click.echo(
                f"note: chunk overlap reduced to {config.chunk_overlap} to fit chunk size {chunk_size}",
                err=True,
            )
    if chunk_overlap is not None:
        config.chunk_overlap = chunk_overlap
    corpus = _open_corpus(ctx, create=False)
    try:
        stats = corpus.rebuild()
    finally:
        corpus.close()
    click.echo(f"rebuilt: {stats['chunks']} chunks from {stats['documents']} documents")


# -- search -----------------------------------------------------------------
@cli.command()
@click.argument("query")
@click.option("--top", "-k", default=None, type=int, help="Number of passages.")
@click.option("--mode", type=click.Choice(["hybrid", "lexical", "vector"]), default="hybrid", show_default=True)
@click.option("--explain", is_flag=True, help="Show each retriever's contribution to the ranking.")
@click.option("--json", "as_json", is_flag=True)
@click.pass_context
def search(ctx: click.Context, query: str, top: int | None, mode: str, explain: bool, as_json: bool) -> None:
    """Retrieve passages from the corpus without generating an answer."""
    config: Config = _opts(ctx)["config"]
    limit = top or config.top_k
    corpus = _open_corpus(ctx, create=False)
    try:
        results = corpus.search(query, top_k=limit, mode=mode)  # type: ignore[arg-type]
    finally:
        corpus.close()

    if as_json:
        click.echo(json.dumps([r.as_dict() for r in results], indent=2, ensure_ascii=False))
        return
    if not results:
        click.echo("no matches")
        return
    for number, result in enumerate(results, start=1):
        click.echo(f"[{number}] {result.chunk.source} (chunk {result.chunk.ordinal}) score={result.score:.5f}")
        click.echo(_indent(_clip(result.text, 300)))
        if explain:
            for name, value in sorted(result.contributions.items(), key=lambda kv: -kv[1]):
                click.echo(_indent(f"{name:<24} {value:.6f}", 4))


def _clip(text: str, limit: int) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[:limit] + "…"


def _indent(text: str, spaces: int = 2) -> str:
    pad = " " * spaces
    return "\n".join(pad + line for line in text.splitlines())


# -- ask --------------------------------------------------------------------
@cli.command()
@click.argument("query")
@click.option("--provider", "-m", default=None, help="Provider name, e.g. offline, openai, ollama.")
@click.option("--top", "-k", default=None, type=int)
@click.option("--max-turns", default=None, type=int, show_default=True)
@click.option("--stream", is_flag=True, help="Print tokens as they arrive.")
@click.option("--json", "as_json", is_flag=True)
@click.pass_context
def ask(
    ctx: click.Context,
    query: str,
    provider: str | None,
    top: int | None,
    max_turns: int | None,
    stream: bool,
    as_json: bool,
) -> None:
    """Answer QUERY from the corpus, with citations."""
    from atheneum.agent.builtin_tools import build_corpus_tools
    from atheneum.agent.loop import Agent, AgentConfig
    from atheneum.providers.base import ProviderError

    config: Config = _opts(ctx)["config"]
    name = provider or config.provider
    corpus = _open_corpus(ctx, create=False)
    try:
        try:
            built = atheneum.get_provider(name)
        except KeyError as exc:
            raise click.ClickException(str(exc)) from exc

        tools = build_corpus_tools(corpus, default_top_k=top or config.top_k)
        runner = Agent(
            built,
            tools,
            config=AgentConfig(
                max_turns=max_turns or config.max_turns,
                token_budget=config.token_budget,
                temperature=config.temperature,
            ),
        )

        if stream and not as_json:
            final = None
            for event in runner.stream(query):
                if event.kind == "text":
                    click.echo(event.text, nl=False)
                elif event.kind == "tool_call" and event.tool_call is not None:
                    if (ctx.obj or {}).get("verbose"):
                        click.echo(
                            f"\n[tool] {event.tool_call.name} {event.tool_call.arguments}", err=True
                        )
                elif event.kind == "done":
                    final = event.run
            if final is not None:
                click.echo()
                _report_run(final, config, verbose=bool((ctx.obj or {}).get("verbose")))
            return

        run = runner.run(query)
    except ProviderError as exc:
        raise click.ClickException(str(exc)) from exc
    finally:
        corpus.close()

    if as_json:
        click.echo(json.dumps(run.as_dict(), indent=2, ensure_ascii=False))
        return
    click.echo(run.answer or "(no answer produced)")
    click.echo()
    _report_run(run, config, verbose=bool((ctx.obj or {}).get("verbose")))


def _report_run(run: Any, config: Config, *, verbose: bool) -> None:
    detail = f"{run.turns} turn(s), {run.tool_call_count} tool call(s), {run.usage.total_tokens} tokens"
    suffix = "" if run.stopped_reason == "final_answer" else f" [{run.stopped_reason}]"
    line = f"provider={config.provider} {detail}{suffix}"
    click.echo(line if verbose or run.stopped_reason != "final_answer" else f"— {detail}", err=True)
    if run.error:
        click.echo(f"error: {run.error}", err=True)


# -- chat -------------------------------------------------------------------
@cli.command()
@click.option("--provider", "-m", default=None)
@click.pass_context
def chat(ctx: click.Context, provider: str | None) -> None:
    """Interactive multi-turn session with history kept in the corpus database."""
    from atheneum.agent.builtin_tools import build_corpus_tools
    from atheneum.agent.loop import Agent, AgentConfig
    from atheneum.core.types import Role
    from atheneum.providers.base import ProviderError

    config: Config = _opts(ctx)["config"]
    name = provider or config.provider
    built = atheneum.get_provider(name)
    corpus = _open_corpus(ctx)
    history: list[Any] = []
    click.echo(f"atheneum chat — provider={built.name}  (ctrl-d or /exit to quit)")
    try:
        while True:
            try:
                line = click.prompt("you", prompt_suffix="> ", default="", show_default=False).strip()
            except (EOFError, click.Abort):
                click.echo()
                break
            if not line:
                continue
            if line in {"/exit", "/quit"}:
                break
            if line == "/history":
                for message in history:
                    click.echo(f"  {message.role.value}: {_clip(message.content, 120)}")
                continue
            tools = build_corpus_tools(corpus, default_top_k=config.top_k)
            runner = Agent(built, tools, config=AgentConfig(max_turns=config.max_turns, token_budget=config.token_budget))
            try:
                run = runner.run(line, history=history)
            except ProviderError as exc:
                click.echo(f"provider error: {exc}", err=True)
                continue
            click.echo(run.answer or "(no answer)")
            history = [m for m in run.messages if m.role is not Role.SYSTEM]
    finally:
        corpus.close()


# -- corpus inspection ------------------------------------------------------
@cli.command(name="sources")
@click.option("--limit", default=30, show_default=True)
@click.option("--json", "as_json", is_flag=True)
@click.pass_context
def sources_command(ctx: click.Context, limit: int, as_json: bool) -> None:
    """List indexed documents."""
    corpus = _open_corpus(ctx, create=False)
    try:
        rows = corpus.sources(limit=limit)
    finally:
        corpus.close()
    if as_json:
        click.echo(json.dumps(rows, indent=2))
        return
    if not rows:
        click.echo("the corpus is empty; run `ath index PATH...`")
        return
    for row in rows:
        click.echo(f"{row['chunk_count']:>6} chunks  {row['source']}")


@cli.command()
@click.argument("source")
@click.option("--chars", default=4000, show_default=True, help="Maximum characters to print.")
@click.pass_context
def show(ctx: click.Context, source: str, chars: int) -> None:
    """Print an indexed document by source path."""
    corpus = _open_corpus(ctx, create=False)
    try:
        document = corpus.store.find_document_by_source(source)
        if document is None:
            matches = [row for row in corpus.sources(limit=5000) if source in row["source"]]
            if not matches:
                raise click.ClickException(f"no indexed document matching {source!r}")
            document = corpus.store.get_document(matches[0]["id"])
    finally:
        corpus.close()
    if document is None:  # pragma: no cover - guarded by the check above
        raise click.ClickException("document could not be read back")
    body = document.content
    click.echo(f"# {document.source}  ({len(body)} chars)")
    click.echo(body[:chars] + ("\n…" if len(body) > chars else ""))


@cli.command()
@click.option("--json", "as_json", is_flag=True)
@click.pass_context
def stats(ctx: click.Context, as_json: bool) -> None:
    """Report corpus size and active retrieval settings."""
    corpus = _open_corpus(ctx)
    try:
        info = corpus.stats()
    finally:
        corpus.close()
    if as_json:
        click.echo(json.dumps(info, indent=2, sort_keys=True))
        return
    for key, value in info.items():
        click.echo(f"{key:>18}  {json.dumps(value) if isinstance(value, dict) else value}")


# -- configuration ----------------------------------------------------------
@cli.command(name="config")
@click.option("--json", "as_json", is_flag=True)
@click.pass_context
def config_command(ctx: click.Context, as_json: bool) -> None:
    """Show the effective configuration and where it came from."""
    config: Config = _opts(ctx)["config"]
    payload = config.to_dict()
    payload["_config_file"] = str(atheneum.config_path())
    if as_json:
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    for key in sorted(payload):
        click.echo(f"{key:>18}  {payload[key]}")


@cli.command()
@click.pass_context
def init(ctx: click.Context) -> None:
    """Write a starter configuration file."""
    config: Config = _opts(ctx)["config"]
    target = save_config(config)
    click.echo(f"wrote {target}")
    click.echo("edit it, or override per-run with environment variables such as ATHENEUM_PROVIDER=openai")


@cli.command()
@click.option("--ready", is_flag=True, help="Only show providers usable right now.")
@click.option("--json", "as_json", is_flag=True)
def providers(ready: bool, as_json: bool) -> None:
    """List model providers and their configuration state."""
    rows = []
    for name, profile in sorted(atheneum.PROVIDER_PROFILES.items()):
        has_key = profile.has_key()
        usable = profile.kind == "offline" or has_key or not profile.api_key_env
        if ready and not usable:
            continue
        rows.append(
            {
                "name": name,
                "kind": profile.kind,
                "default_model": profile.default_model,
                "base_url": profile.default_base_url,
                "key_env": list(profile.api_key_env),
                "ready": usable,
            }
        )
    if as_json:
        click.echo(json.dumps(rows, indent=2))
        return
    click.echo(f"{'NAME':<14}{'KIND':<12}{'READY':<7}DEFAULT MODEL / BASE URL")
    for row in rows:
        click.echo(
            f"{row['name']:<14}{row['kind']:<12}{'yes' if row['ready'] else 'no':<7}"
            f"{row['default_model']}" + (f"  @  {row['base_url']}" if row["base_url"] else "")
        )
    click.echo("\nKeys are read from the environment only and are never written to disk.")


@cli.command()
@click.argument("paths", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--top", "-k", default=5, show_default=True)
@click.option("--mode", type=click.Choice(["hybrid", "lexical", "vector"]), default=None)
@click.pass_context
def bench(ctx: click.Context, paths: tuple[str, ...], top: int, mode: str | None) -> None:
    """Measure ingestion and query latency for the given files."""
    from atheneum.evaluate import benchmark_paths

    options = _opts(ctx)
    report = benchmark_paths(
        paths, top_k=top, mode=mode, db=str(Path(options["db"]).parent / "bench.db")  # type: ignore[arg-type]
    )
    click.echo(json.dumps(report, indent=2, sort_keys=True))


@cli.command(name="eval")
@click.option("--json", "as_json", is_flag=True, help="Emit the full report as JSON.")
@click.option("--top", "-k", default=5, show_default=True, type=int)
def eval_command(as_json: bool, top: int) -> None:
    """Score retrieval quality on the bundled labelled dataset.

    The default output is a table; every other command in this CLI treats --json
    as the machine-readable switch, so this one does too rather than printing
    JSON unconditionally.
    """
    from atheneum.evaluate import run_evaluation

    report = run_evaluation(top_k=top)
    if as_json:
        click.echo(json.dumps(report, indent=2, sort_keys=True))
        return

    click.echo(
        f"corpus: {report['corpus']['documents']} documents, "
        f"{report['corpus']['chunks']} chunks, {report['query_count']} queries, k={report['k']}\n"
    )
    click.echo(f"{'MODE':<9}{'recall@k':>10}{'prec@k':>9}{'MRR':>8}{'hit':>8}{'latency(ms)':>13}")
    for row in report["results"]:
        click.echo(
            f"{row['name']:<9}{row['recall_at_k']:>10.3f}{row['precision_at_k']:>9.3f}"
            f"{row['mrr']:>8.3f}{row['hit_rate']:>8.3f}{row['mean_latency_ms']:>13.4f}"
        )
    verdict = report["hybrid_beats_both_retrievers"]
    click.echo(
        f"\nhybrid beats every single retriever on recall and MRR: {verdict}"
    )
    missed = [row for row in report["queries"] if row["recall_at_k"] < 1.0]
    if missed:
        click.echo(f"{len(missed)} queries missed a labelled passage:")
        for row in missed[:5]:
            click.echo(f"  recall={row['recall_at_k']:.2f}  {row['query'][:60]}")


@cli.command()
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8077, show_default=True, type=int)
@click.option("--reload", is_flag=True)
@click.pass_context
def serve(ctx: click.Context, host: str, port: int, reload: bool) -> None:
    """Start the HTTP API."""
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - depends on extras
        raise click.ClickException(
            "the HTTP API needs the `api` extra: pip install atheneum[api]"
        ) from exc
    options = _opts(ctx)
    os.environ.setdefault("ATHENEUM_DB", str(options["db"]))
    click.echo(f"serving atheneum on http://{host}:{port} (db={options['db']})")
    uvicorn.run("atheneum.api.http:app", host=host, port=port, reload=reload)


def main() -> int:
    try:
        cli(auto_envvar_prefix="ATHENEUM", standalone_mode=False)
    except click.ClickException as exc:
        exc.show()
        return exc.exit_code
    except click.Abort:
        click.echo("aborted", err=True)
        return 130
    except SystemExit as exc:
        return int(exc.code or 0)
    except FileNotFoundError as exc:
        click.echo(f"error: {exc}", err=True)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
