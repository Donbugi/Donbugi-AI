"""
기사 기반 객관식 퀴즈 생성.

개선점 (기존 quiz-generation.py 대비)
- 정규식으로 ```json 블록을 뜯던 방식 → tool use 로 JSON 스키마 강제 (파싱 실패 502 제거)
- 규칙 기반 품질 검증(보기 중복·정답 범위·금지 보기 등) 실패 시 피드백을 붙여 1회 재생성
- (선택) LLM-as-judge: 정답 근거가 본문에 있는지, 정답이 하나인지 검증
- 시스템 프롬프트 분리 + cache_control (긴 지침은 프롬프트 캐싱 대상)
- 유형·난이도 메타데이터 추가 (BE 는 모르는 필드를 무시하므로 하위호환)
"""
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .settings import settings

log = logging.getLogger(__name__)

QUESTION_TYPES = ["concept", "inference", "main_idea", "context"]
DIFFICULTIES = ["easy", "medium", "hard"]

QUIZ_TOOL = {
    "name": "emit_quiz",
    "description": "기사 본문에 근거한 금융·경제 교육용 4지선다 퀴즈를 제출한다.",
    "input_schema": {
        "type": "object",
        "properties": {
            "quiz": {
                "type": "array",
                "minItems": 1,
                "maxItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "질문 (한국어)"},
                        "options": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 4,
                            "maxItems": 4,
                            "description": "서로 다른 보기 4개",
                        },
                        "correct_index": {"type": "integer", "minimum": 0, "maximum": 3},
                        "explanation": {"type": "string", "description": "정답 근거 1~2문장 (본문 기반)"},
                        "question_type": {"type": "string", "enum": QUESTION_TYPES},
                        "difficulty": {"type": "string", "enum": DIFFICULTIES},
                    },
                    "required": ["question", "options", "correct_index", "explanation", "question_type", "difficulty"],
                },
            }
        },
        "required": ["quiz"],
    },
}

SYSTEM_PROMPT = """너는 금융·경제 뉴스로 학습용 퀴즈를 만드는 출제자다. 반드시 emit_quiz 도구로만 답한다.

[출제 원칙]
- 기사 본문에 근거가 분명한 내용만 묻는다. 본문에 없는 사실을 만들지 않는다.
- 문항은 정확히 1개, 4지선다. 정답은 정확히 하나여야 한다.
- 비율·금액·연도·정확한 숫자를 맞히는 문제는 피하고, 수치 없이도 이해되는 질문을 우선한다.
- 아래 유형 중 본문에 가장 알맞은 하나를 고른다.
  · concept: 용어·제도·원리·정의
  · inference: 본문 근거 안에서 원인·결과·전망을 추론
  · main_idea: 기사의 중심 이슈·핵심 메시지
  · context: 인과, 주장과 근거, 비교·해석
- 오답 보기는 그럴듯해야 한다(본문 주제와 관련 있지만 틀린 진술). 정답만 유난히 길거나 구체적이지 않게 길이를 비슷하게 맞춘다.
- "모두 정답", "정답 없음", "위의 모든 것" 같은 보기는 쓰지 않는다.
- 해설은 본문 근거를 들어 1~2문장 한국어로 쓴다.
- 투자 권유나 특정 종목 매수·매도 판단을 정답으로 만들지 않는다.
"""

BANNED_OPTION_PATTERNS = [r"모두\s*정답", r"정답\s*없음", r"위의\s*모든", r"모두\s*맞", r"해당\s*없음"]


class QuizGenerationError(Exception):
    pass


@dataclass
class Validation:
    ok: bool
    problems: List[str]


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", (s or "")).strip().lower()


def validate_item(item: Dict[str, Any]) -> Validation:
    problems: List[str] = []
    q = (item.get("question") or "").strip()
    opts = item.get("options")
    idx = item.get("correct_index")
    exp = (item.get("explanation") or "").strip()

    if len(q) < 8:
        problems.append("질문이 너무 짧다")
    if not isinstance(opts, list) or len(opts) != 4 or not all(isinstance(o, str) and o.strip() for o in opts):
        problems.append("보기는 비어 있지 않은 문자열 4개여야 한다")
    else:
        if len({_norm(o) for o in opts}) != 4:
            problems.append("보기 4개가 서로 달라야 한다")
        for o in opts:
            if any(re.search(p, o) for p in BANNED_OPTION_PATTERNS):
                problems.append(f"금지된 보기 형식: {o}")
        lengths = [len(o.strip()) for o in opts]
        if isinstance(idx, int) and 0 <= idx < 4 and max(lengths) > 0:
            others = [l for i, l in enumerate(lengths) if i != idx]
            if lengths[idx] > 2.5 * max(1, max(others)):
                problems.append("정답 보기만 지나치게 길다(길이로 정답이 드러남)")
    if not isinstance(idx, int) or isinstance(idx, bool) or not 0 <= idx <= 3:
        problems.append("correct_index 는 0~3 정수여야 한다")
    if not exp:
        problems.append("해설이 비어 있다")
    return Validation(ok=not problems, problems=problems)


def _tool_input(message) -> Dict[str, Any]:
    for block in getattr(message, "content", []) or []:
        if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == QUIZ_TOOL["name"]:
            return dict(block.input)
    raise QuizGenerationError("모델이 emit_quiz 도구를 호출하지 않았다")


def _call(client, content: str, feedback: Optional[str]) -> Dict[str, Any]:
    user_text = f"다음 기사 본문으로 퀴즈 1문항을 만들어라.\n\n본문:\n---\n{content}\n---"
    if feedback:
        user_text += f"\n\n[이전 시도의 문제점 — 반드시 고쳐서 다시 출제]\n{feedback}"
    msg = client.messages.create(
        model=settings.model,
        max_tokens=1024,
        system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        tools=[QUIZ_TOOL],
        tool_choice={"type": "tool", "name": QUIZ_TOOL["name"]},
        messages=[{"role": "user", "content": user_text}],
    )
    return _tool_input(msg)


JUDGE_TOOL = {
    "name": "emit_verdict",
    "description": "퀴즈 검수 결과를 제출한다.",
    "input_schema": {
        "type": "object",
        "properties": {
            "answer_supported_by_article": {"type": "boolean"},
            "single_correct_answer": {"type": "boolean"},
            "reason": {"type": "string"},
        },
        "required": ["answer_supported_by_article", "single_correct_answer", "reason"],
    },
}


def judge(client, content: str, item: Dict[str, Any]) -> Validation:
    """LLM-as-judge: 정답 근거가 본문에 있고 정답이 하나뿐인지 검수."""
    opts = "\n".join(f"{i}. {o}" for i, o in enumerate(item["options"]))
    msg = client.messages.create(
        model=settings.effective_judge_model,
        max_tokens=512,
        tools=[JUDGE_TOOL],
        tool_choice={"type": "tool", "name": JUDGE_TOOL["name"]},
        messages=[{
            "role": "user",
            "content": (
                "아래 기사와 퀴즈를 검수하라. 정답이 본문으로 뒷받침되는지, 정답으로 볼 수 있는 보기가 하나뿐인지 판단한다.\n\n"
                f"기사:\n---\n{content}\n---\n\n질문: {item['question']}\n보기:\n{opts}\n"
                f"출제자가 지정한 정답: {item['correct_index']}\n해설: {item.get('explanation', '')}"
            ),
        }],
    )
    for block in getattr(msg, "content", []) or []:
        if getattr(block, "type", None) == "tool_use":
            v = dict(block.input)
            problems = []
            if not v.get("answer_supported_by_article"):
                problems.append("정답 근거가 본문에 없다: " + str(v.get("reason", "")))
            if not v.get("single_correct_answer"):
                problems.append("정답이 하나가 아니다: " + str(v.get("reason", "")))
            return Validation(ok=not problems, problems=problems)
    return Validation(ok=True, problems=[])  # 판정 실패 시 생성 결과를 막지 않는다


def generate_quiz(client, content: str, max_attempts: int = 2) -> List[Dict[str, Any]]:
    """
    퀴즈 생성 → 규칙 검증 → (선택) LLM 검수. 실패하면 문제점을 피드백으로 붙여 재생성.
    """
    if not content or not content.strip():
        raise QuizGenerationError("본문이 비어 있다")
    body = content.strip()[: settings.max_content_chars]

    feedback: Optional[str] = None
    last_problems: List[str] = []
    for attempt in range(1, max_attempts + 1):
        data = _call(client, body, feedback)
        items = data.get("quiz")
        if not isinstance(items, list) or len(items) != 1 or not isinstance(items[0], dict):
            last_problems = ["quiz 배열은 객체 1개여야 한다"]
        else:
            item = items[0]
            v = validate_item(item)
            if v.ok and settings.judge_enabled:
                v = judge(client, body, item)
            if v.ok:
                item.setdefault("question_type", "concept")
                item.setdefault("difficulty", "medium")
                return [item]
            last_problems = v.problems
        log.info("퀴즈 품질 검증 실패(%d/%d): %s", attempt, max_attempts, last_problems)
        feedback = "\n".join(f"- {p}" for p in last_problems)
    raise QuizGenerationError("품질 검증 실패: " + "; ".join(last_problems))


INSIGHT_SYSTEM = (
    "너는 금융 학습 앱의 안내자다. 사용자의 최근 관심 뉴스 토픽을 1문장(80자 이내) 한국어로 요약한다. "
    "과장하지 말고, 투자 조언처럼 단정하지 말고, 관심 흐름을 자연스럽게 짚는다. 따옴표 없이 문장만 출력한다."
)


def generate_insight(client, topics: List[Dict[str, Any]]) -> str:
    lines = "\n".join(f"- {t['name'].strip()}: {int(t.get('count', 0))}회" for t in topics[:10])
    msg = client.messages.create(
        model=settings.model,
        max_tokens=200,
        system=INSIGHT_SYSTEM,
        messages=[{"role": "user", "content": f"최근 한 달 많이 본 토픽과 조회 수:\n{lines}"}],
    )
    text = "".join(getattr(b, "text", "") for b in getattr(msg, "content", []) or []).strip()
    # 첫 줄만 쓰고, 그 다음에 앞뒤 따옴표를 제거 (순서가 반대면 여러 줄일 때 닫는 따옴표가 남는다)
    text = text.splitlines()[0].strip().strip('"').strip("“”").strip() if text else ""
    if not text:
        raise QuizGenerationError("빈 인사이트")
    return text[:150]
