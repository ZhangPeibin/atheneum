from __future__ import annotations

import json

import pytest

fastapi = pytest.importorskip("fastapi", reason="atheneum[api] not installed")

from fastapi.testclient import TestClient  # noqa: E402

from atheneum.api.http import create_app  # noqa: E402
from atheneum.config import Config  # noqa: E402


@pytest.fixture
def client(tmp_path):
    config = Config(db=str(tmp_path / "api.db"), provider="offline", chunk_size=400, chunk_overlap=40)
    app = create_app(config)
    with TestClient(app) as test_client:
        yield test_client


# -- health -----------------------------------------------------------------
def test_health_is_open(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_reports_the_version(client):
    import atheneum

    assert client.get("/health").json()["version"] == atheneum.__version__


# -- documents --------------------------------------------------------------
def test_adding_a_document(client):
    response = client.post(
        "/documents",
        json={"source": "notes/a.md", "content": "# A\n\nReciprocal rank fusion merges ranked lists."},
    )
    assert response.status_code == 201
    assert response.json()["chunks_added"] >= 1


def test_empty_content_is_rejected(client):
    assert client.post("/documents", json={"source": "a.md", "content": ""}).status_code == 422


def test_missing_source_is_rejected(client):
    assert client.post("/documents", json={"content": "x"}).status_code == 422


@pytest.mark.parametrize(
    "source",
    [
        "/etc/passwd",
        "../../secrets",
        "..\\..\\windows",
        "a" * 600,
        "with space",
        "",
        "newline\ninjected",
    ],
)
def test_dangerous_source_values_are_rejected(client, source: str):
    """Sources land in citations and logs, so they must not be paths or control data."""
    assert client.post("/documents", json={"source": source, "content": "body"}).status_code == 422


@pytest.mark.parametrize("source", ["a.md", "docs/note-1.md", "ticket#42", "dir/sub/x.py", "中文.md"])
def test_reasonable_source_values_are_accepted(client, source: str):
    assert client.post("/documents", json={"source": source, "content": "body text"}).status_code == 201


def test_duplicate_document_is_not_double_indexed(client):
    payload = {"source": "a.md", "content": "Same content both times."}
    assert client.post("/documents", json=payload).json()["chunks_added"] >= 1
    assert client.post("/documents", json=payload).json()["chunks_added"] == 0


# -- search -----------------------------------------------------------------
def test_search_returns_ranked_results(client):
    client.post("/documents", json={"source": "a.md", "content": "Token buckets enforce an average rate."})
    client.post("/documents", json={"source": "b.md", "content": "Unrelated prose about gardens."})
    body = client.post("/search", json={"query": "token bucket average rate", "top_k": 2}).json()
    assert body["results"][0]["source"] == "a.md"
    assert body["mode"] == "hybrid"


def test_search_result_shape(client):
    client.post("/documents", json={"source": "a.md", "content": "Fusion merges ranked lists."})
    result = client.post("/search", json={"query": "fusion"}).json()["results"][0]
    assert {"chunk_id", "source", "ordinal", "score", "contributions", "text"} <= set(result)


def test_empty_query_is_rejected(client):
    assert client.post("/search", json={"query": ""}).status_code == 422


def test_top_k_bounds(client):
    assert client.post("/search", json={"query": "x", "top_k": 0}).status_code == 422
    assert client.post("/search", json={"query": "x", "top_k": 101}).status_code == 422


def test_invalid_mode_is_rejected(client):
    assert client.post("/search", json={"query": "x", "mode": "telepathy"}).status_code == 422


def test_search_on_an_empty_corpus(client):
    assert client.post("/search", json={"query": "anything"}).json()["results"] == []


# -- ask --------------------------------------------------------------------
def test_ask_produces_an_answer(client):
    client.post(
        "/documents",
        json={"source": "a.md", "content": "Okapi BM25 uses k1 to control term frequency saturation."},
    )
    body = client.post("/ask", json={"query": "what does k1 control"}).json()
    assert "saturation" in body["answer"] or "k1" in body["answer"]
    assert body["stopped_reason"] == "final_answer"
    assert body["turns"] >= 1


def test_ask_can_include_evidence(client):
    client.post("/documents", json={"source": "a.md", "content": "Fusion merges ranked lists nicely."})
    body = client.post("/ask", json={"query": "what is fusion", "include_evidence": True}).json()
    assert body["evidence"]
    assert json.loads(body["evidence"][0]["content"])["results"]


def test_ask_with_an_unknown_provider_is_a_422(client):
    assert client.post("/ask", json={"query": "x", "provider": "ghost"}).status_code == 422


def test_ask_max_turns_bounds_the_loop(client):
    body = client.post("/ask", json={"query": "fusion", "max_turns": 1}).json()
    assert body["turns"] <= 1


def test_ask_over_a_fresh_corpus_explains_itself(client):
    body = client.post("/ask", json={"query": "anything"}).json()
    assert "No indexed material" in body["answer"] or "index" in body["answer"]


# -- stats and sources ------------------------------------------------------
def test_stats_endpoint(client):
    client.post("/documents", json={"source": "a.md", "content": "Some content here."})
    body = client.get("/stats").json()
    assert body["chunks"] >= 1
    assert body["config"]["fusion_k"] == 61


def test_sources_endpoint(client):
    client.post("/documents", json={"source": "a.md", "content": "Some content here."})
    sources = client.get("/sources").json()["sources"]
    assert sources[0]["source"] == "a.md"


def test_sources_limit_is_bounded(client):
    """limit=0 used to be silently clamped; it is now rejected as out of range."""
    assert client.get("/sources?limit=0").status_code == 422
    assert client.get("/sources?limit=1").status_code == 200
    assert client.get("/sources?limit=5000").status_code == 422


# -- streaming --------------------------------------------------------------
def test_streaming_endpoint(client):
    client.post("/documents", json={"source": "a.md", "content": "Reciprocal rank fusion merges lists."})
    with client.stream("GET", "/ask/stream?query=what+is+fusion") as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        body = "".join(chunk for chunk in response.iter_text())
    assert "event: done" in body
    assert "event:" in body


# -- auth -------------------------------------------------------------------
def test_auth_is_off_by_default(tmp_path):
    client = TestClient(create_app(Config(db=str(tmp_path / "a.db"))))
    with client:
        assert client.get("/stats").status_code == 200


def test_bearer_token_is_enforced_when_configured(tmp_path, monkeypatch):
    monkeypatch.setenv("ATHENEUM_API_TOKEN", "secret-value")
    client = TestClient(create_app(Config(db=str(tmp_path / "b.db"))))
    with client:
        assert client.get("/stats").status_code == 401
        assert client.get("/stats", headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert client.get("/stats", headers={"Authorization": "Bearer secret-value"}).status_code == 200


def test_health_stays_open_under_auth(tmp_path, monkeypatch):
    monkeypatch.setenv("ATHENEUM_API_TOKEN", "secret-value")
    client = TestClient(create_app(Config(db=str(tmp_path / "c.db"))))
    with client:
        assert client.get("/health").status_code == 200


def test_documents_endpoint_requires_auth(tmp_path, monkeypatch):
    monkeypatch.setenv("ATHENEUM_API_TOKEN", "s3cret")
    client = TestClient(create_app(Config(db=str(tmp_path / "d.db"))))
    with client:
        assert (
            client.post("/documents", json={"source": "a.md", "content": "x"}).status_code == 401
        )


# -- app lifecycle ----------------------------------------------------------
def test_the_corpus_is_closed_on_shutdown(tmp_path):
    app = create_app(Config(db=str(tmp_path / "e.db")))
    with TestClient(app) as client:
        client.post("/documents", json={"source": "a.md", "content": "content"})
    # Reopening the same file must work, proving the previous handle was released.
    again = TestClient(create_app(Config(db=str(tmp_path / "e.db"))))
    with again:
        assert again.get("/stats").json()["chunks"] == 1


# ---------------------------------------------------------------------------
# Security checklist, pinned. Each of these was verified by direct probing and
# is asserted here so a regression cannot reintroduce it silently.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source",
    [
        "trail\n", " pad ", "a\rb", "t\tx", "l\u2028s", "p\u2029s",
        # A combining mark is not \w, so "é.md" written as e+U+0301 is refused.
        # That is the safer answer: it cannot alias a precomposed "é.md".
        "e\u0301.md", "a//b", "a//b/", "..", "/", ".", "", "x" * 513, "nul\x00.md",
    ],
)
def test_hostile_sources_are_rejected(client, source: str):
    assert client.post("/documents", json={"source": source, "content": "body text"}).status_code == 422


@pytest.mark.parametrize("source", ["ok.md", "a+b/c-d_e.md", "中文.md", "x" * 512, "ticket#42"])
def test_ordinary_sources_are_accepted(client, source: str):
    assert client.post("/documents", json={"source": source, "content": "body text"}).status_code == 201


def test_the_same_source_with_different_content_coexists(client):
    """Content-addressed ids: this is a documented property, not a collision.

    Two documents under one source make citations ambiguous, which is why
    `source` is documented as a stable identifier the caller owns.
    """
    first = client.post("/documents", json={"source": "ok.md", "content": "AAA"}).json()
    second = client.post("/documents", json={"source": "ok.md", "content": "BBB"}).json()
    assert first["doc_id"] != second["doc_id"]
    assert first["chunks_added"] == second["chunks_added"] == 1


@pytest.mark.parametrize(
    "payload",
    [
        {"source": "t.md", "content": "x", "title": "T" * 70_000},
        {"source": "m.md", "content": "x", "metadata": {"k": "v" * 70_000}},
        {"source": "n.md", "content": "x", "metadata": {"k": {"j": "v" * 70_000}}},
    ],
)
def test_auxiliary_fields_are_capped(client, payload: dict):
    assert client.post("/documents", json=payload).status_code == 422


@pytest.mark.parametrize(
    "call",
    [
        ("post", "/ask", {"query": "x", "provider": "ghost"}),
        ("post", "/search", {"query": "x", "mode": "telepathy"}),
        ("get", "/stats", None),
        ("get", "/sources", None),
    ],
)
def test_no_error_response_leaks_server_internals(client, call):
    # `request` is reserved by pytest and cannot be a parametrize argument name.
    method, path, body = call
    response = client.post(path, json=body) if method == "post" else client.get(path)
    text = response.text
    for needle in ("Traceback", "/Users/", "site-packages", "atheneum.db", "sk-"):
        assert needle not in text, f"{path} leaked {needle!r}: {text[:200]}"


def test_every_sensitive_route_requires_the_token(tmp_path, monkeypatch):
    monkeypatch.setenv("ATHENEUM_API_TOKEN", "s3cret")
    client = TestClient(create_app(Config(db=str(tmp_path / "auth.db"))))
    with client:
        assert client.get("/health").status_code == 200
        for path in ("/stats", "/sources", "/docs", "/redoc", "/openapi.json"):
            assert client.get(path).status_code in (401, 404), path
        assert client.get("/ask/stream?query=hi").status_code == 401
        for path, body in (("/search", {"query": "hi"}), ("/ask", {"query": "hi"})):
            assert client.post(path, json=body).status_code == 401, path
        assert client.post("/documents", json={"source": "z.md", "content": "x"}).status_code == 401

        good = {"Authorization": "Bearer s3cret"}
        assert client.get("/stats", headers=good).status_code == 200
        for bad in (
            {"Authorization": "Bearer nope"},
            {"Authorization": "bearer s3cret"},
            {"Authorization": "s3cret"},
            {},
        ):
            assert client.get("/stats", headers=bad).status_code == 401, bad
