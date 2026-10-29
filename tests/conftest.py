import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")


class FakeMessages:
    """client.messages.create 를 흉내낸다. responses 큐에서 순서대로 꺼내 반환."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


class FakeClient:
    def __init__(self, responses):
        self.messages = FakeMessages(responses)


def tool_message(name, payload):
    return SimpleNamespace(content=[SimpleNamespace(type="tool_use", name=name, input=payload)])


def text_message(text):
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])


GOOD_ITEM = {
    "question": "기사에서 한국은행이 기준금리를 동결한 주된 이유는?",
    "options": ["물가 상승 압력 지속", "수출 급증", "부동산 가격 급락", "환율 하락"],
    "correct_index": 0,
    "explanation": "본문은 물가 상승 압력이 여전하다는 점을 동결 이유로 든다.",
    "question_type": "inference",
    "difficulty": "medium",
}


@pytest.fixture
def good_item():
    return dict(GOOD_ITEM, options=list(GOOD_ITEM["options"]))
