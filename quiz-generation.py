"""
DB에서 기사 내용을 읽어 Claude Haiku로 객관식 퀴즈 4문항(4지선다) 생성 후
API 응답으로 반환
"""
import json
import re
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import psycopg2
from anthropic import Anthropic
from fastapi import FastAPI, HTTPException
from psycopg2.extras import RealDictCursor
from pydantic import BaseModel, Field

from config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL, DB_CONFIG

NUM_QUESTIONS = 4
MAX_CONTENT_CHARS = 18_000
MAX_OUTPUT_TOKENS = 4096

app = FastAPI(title="Quiz generation (Claude Haiku)", version="1.0.0")


class QuizRequest(BaseModel):
    article_id: int = Field(..., description="articles.article_id")


class QuizItem(BaseModel):
    question: str
    options: List[str] = Field(..., min_length=4, max_length=4)
    correct_index: int = Field(..., ge=0, le=3)
    explanation: Optional[str] = None


class QuizResponse(BaseModel):
    article_id: int
    quiz: List[QuizItem]


def get_client() -> Anthropic:
    if not ANTHROPIC_API_KEY or not str(ANTHROPIC_API_KEY).strip():
        raise RuntimeError(
            "config.py의 ANTHROPIC_API_KEY(또는 환경 변수)를 설정하세요."
        )
    return Anthropic(api_key=ANTHROPIC_API_KEY.strip())


def get_model_id() -> str:
    return (ANTHROPIC_MODEL or "").strip() or "claude-haiku-4-5"


@contextmanager
def db_conn():
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        yield conn
    finally:
        conn.close()


def fetch_content_by_article_id(article_id: int) -> Optional[str]:
    with db_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT content FROM articles
                WHERE article_id = %s AND content IS NOT NULL AND content != ''
                """,
                (article_id,),
            )
            row = cur.fetchone()
    if not row:
        return None
    return row["content"]


def extract_json_array(text: str) -> List[Any]:
    text = (text or "").strip()
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    start = text.find("[")
    if start < 0:
        raise ValueError("JSON 배열([)을 찾을 수 없습니다.")
    decoded, _ = json.JSONDecoder().raw_decode(text[start:])
    return decoded


def build_prompt(content: str, num_questions: int) -> str:
    body = content.strip()
    if len(body) > MAX_CONTENT_CHARS:
        body = body[:MAX_CONTENT_CHARS]

    return f"""다음은 기사 본문이다. 본문 내용만 근거로 금융·경제 교육용 객관식 퀴즈를 정확히 {num_questions}문제 만들어라.
각 문제는 4지선다(보기 4개)이다. 본문에 답의 근거가 분명해야 하며 본문에 없는 사실을 꾸미지 마라.

[문항 유형 균형 — 반드시 지킬 것]
- 전체 문항이 비율·금액·구체 수치·연도 맞히기에 치우치지 않도록 한다. "정확히 몇 %/얼마/몇 년"류는 전체 중 최대 1문항까지이며, 가능하면 아예 두지 말고 수치 없이도 이해되는 질문으로 쓴다.
- 다음 성격이 골고루 나오게 배분한다 (문항 수가 4개면 (1)~(4)를 각각 정확히 1문항씩. 문항 수가 다르면 (1)~(3)은 반드시 각 1문항 이상 포함하고, 나머지는 (4)와 섞어 균형 있게):
  (1) 개념 이해형: 용어·제도·원리·정의를 본문 맥락에서 묻는 문제
  (2) 추론형: 본문에 직접 쓰인 문장만으로는 단순 암기가 아니라, 논지·원인·결과·전망 등을 종합·추론해야 풀 수 있는 문제 (환각 없이 본문 근거 범위 안에서만)
  (3) 핵심 주제형: 이 기사의 중심 이슈·핵심 메시지·다루는 대상(무엇을 설명하는지)을 묻는 문제
  (4) 나머지 문항: 인과관계, 주장 vs 근거, 맥락 비교, 정책·시장에 대한 해석 등으로 구성하되, 위 세 유형과 겹치지 않게 다양하게 (역시 수치 편중 금지)

출력은 설명 없이 오직 JSON 배열만 출력한다. (앞뒤에 다른 텍스트 금지)
반드시 원소 개수는 {num_questions}개이다.

각 원소 형식:
{{
  "question": "질문 문자열",
  "options": ["보기1", "보기2", "보기3", "보기4"],
  "correct_index": 0,
  "explanation": "정답 근거 한두 문장 (한국어)"
}}

correct_index는 0~3 정수(정답이 options의 몇 번째인지). options는 반드시 정확히 4개 문자열이다.

본문:
---
{body}
---
"""


def generate_quiz_list(client: Anthropic, model: str, content: str) -> List[Dict[str, Any]]:
    if not content or not str(content).strip():
        raise ValueError("본문이 비어 있습니다.")

    prompt = build_prompt(content, NUM_QUESTIONS)
    msg = client.messages.create(
        model=model,
        max_tokens=MAX_OUTPUT_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )

    text_parts: List[str] = []
    for block in msg.content:
        if hasattr(block, "text"):
            text_parts.append(block.text)
    raw = "".join(text_parts)

    quizzes = extract_json_array(raw)
    if not isinstance(quizzes, list):
        raise ValueError("퀴즈 응답이 배열이 아닙니다.")
    if len(quizzes) != NUM_QUESTIONS:
        raise ValueError(f"문항 수는 {NUM_QUESTIONS}개여야 하는데 {len(quizzes)}개입니다.")

    for i, item in enumerate(quizzes):
        if not isinstance(item, dict):
            raise ValueError(f"항목 {i}가 객체가 아닙니다.")
        opts = item.get("options")
        if not isinstance(opts, list) or len(opts) != 4:
            raise ValueError(f"항목 {i}: options는 길이 4인 배열이어야 합니다.")
        idx = item.get("correct_index")
        if not isinstance(idx, int) or idx < 0 or idx > 3:
            raise ValueError(f"항목 {i}: correct_index는 0~3이어야 합니다.")

    return quizzes


@app.post("/quiz", response_model=QuizResponse)
def create_quiz(req: QuizRequest) -> QuizResponse:
    """article_id로 본문을 읽어 퀴즈 4문항을 생성해 JSON으로 반환"""
    content = fetch_content_by_article_id(req.article_id)
    if content is None:
        raise HTTPException(
            status_code=404,
            detail=f"article_id={req.article_id} 기사를 찾을 수 없거나 content가 비어 있습니다.",
        )

    try:
        client = get_client()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    model = get_model_id()
    try:
        raw_list = generate_quiz_list(client, model, content)
    except ValueError as e:
        raise HTTPException(status_code=502, detail=f"퀴즈 생성 실패: {e}") from e
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"퀴즈 생성 실패: {e}") from e

    try:
        items = [QuizItem.model_validate(x) for x in raw_list]
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"응답 형식 오류: {e}") from e

    return QuizResponse(article_id=req.article_id, quiz=items)


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
