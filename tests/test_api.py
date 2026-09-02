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
    assert client.get("/sources?limit=0").status_code == 200


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
