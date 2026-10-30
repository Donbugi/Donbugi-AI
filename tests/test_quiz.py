import pytest

from finedu_ai.quiz import QuizGenerationError, generate_insight, generate_quiz, validate_item

from .conftest import FakeClient, text_message, tool_message


def test_validate_ok(good_item):
    assert validate_item(good_item).ok


@pytest.mark.parametrize(
    "mutate, expected",
    [
        (lambda i: i.update(options=["a", "a", "b", "c"]), "서로 달라야"),
        (lambda i: i.update(options=["모두 정답", "x2", "x3", "x4"]), "금지된 보기"),
        (lambda i: i.update(correct_index=4), "correct_index"),
        (lambda i: i.update(explanation=""), "해설"),
        (lambda i: i.update(options=["가" * 100, "나", "다", "라"], correct_index=0), "지나치게 길다"),
    ],
)
def test_validate_problems(good_item, mutate, expected):
    mutate(good_item)
    v = validate_item(good_item)
    assert not v.ok
    assert any(expected in p for p in v.problems)


def test_generate_uses_tool_and_returns_item(good_item):
    client = FakeClient([tool_message("emit_quiz", {"quiz": [good_item]})])
    out = generate_quiz(client, "본문 " * 50)
    assert out == [good_item]
    call = client.messages.calls[0]
    assert call["tool_choice"] == {"type": "tool", "name": "emit_quiz"}
    assert call["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_generate_retries_with_feedback(good_item):
    bad = dict(good_item, options=["a", "a", "b", "c"])
    client = FakeClient([
        tool_message("emit_quiz", {"quiz": [bad]}),
        tool_message("emit_quiz", {"quiz": [good_item]}),
    ])
    out = generate_quiz(client, "본문")
    assert out[0]["options"] == good_item["options"]
    assert "이전 시도의 문제점" in client.messages.calls[1]["messages"][0]["content"]


def test_generate_fails_after_max_attempts(good_item):
    bad = dict(good_item, explanation="")
    client = FakeClient([tool_message("emit_quiz", {"quiz": [bad]})] * 2)
    with pytest.raises(QuizGenerationError):
        generate_quiz(client, "본문")


def test_generate_rejects_missing_tool_call():
    client = FakeClient([text_message("그냥 텍스트")])
    with pytest.raises(QuizGenerationError):
        generate_quiz(client, "본문")


def test_empty_content():
    with pytest.raises(QuizGenerationError):
        generate_quiz(FakeClient([]), "   ")


def test_insight_strips_quotes():
    client = FakeClient([text_message('"최근 금리 뉴스를 많이 보셨어요."\n추가 줄')])
    assert generate_insight(client, [{"name": "금리", "count": 3}]) == "최근 금리 뉴스를 많이 보셨어요."
