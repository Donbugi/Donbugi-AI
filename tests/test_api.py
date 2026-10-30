from fastapi.testclient import TestClient

from finedu_ai import main
from finedu_ai.llm import set_client_for_tests

from .conftest import FakeClient, tool_message

client = TestClient(main.app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_quiz_with_content(good_item):
    set_client_for_tests(FakeClient([tool_message("emit_quiz", {"quiz": [good_item]})]))
    r = client.post("/quiz", json={"article_id": 1, "content": "본문 " * 30})
    assert r.status_code == 200
    body = r.json()
    assert body["article_id"] == 1
    assert body["quiz"][0]["correct_index"] == 0
    assert body["model"]


def test_quiz_generation_failure_is_502(good_item):
    bad = dict(good_item, explanation="")
    set_client_for_tests(FakeClient([tool_message("emit_quiz", {"quiz": [bad]})] * 2))
    r = client.post("/quiz", json={"article_id": 1, "content": "본문"})
    assert r.status_code == 502


def test_quiz_missing_article_is_404(monkeypatch):
    monkeypatch.setattr(main.db, "fetch_article_content", lambda _id: None)
    r = client.post("/quiz", json={"article_id": 999})
    assert r.status_code == 404


def test_insight_empty_topics_400():
    r = client.post("/news/interest-insight", json={"topics": [{"name": " ", "count": 1}]})
    assert r.status_code == 400


def test_ready_degraded_without_db(monkeypatch):
    monkeypatch.setattr(main.db, "ping", lambda: False)
    r = client.get("/ready")
    assert r.status_code == 503
    assert r.json()["db"] is False
