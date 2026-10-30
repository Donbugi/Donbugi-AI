"""FastAPI 엔트리포인트. 요청/응답 계약은 기존 quiz-generation.py 와 동일 (BE 수정 불필요)."""
import logging
import time
from typing import List, Optional

from anthropic import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from . import db
from .llm import LlmNotConfigured, get_client
from .quiz import QuizGenerationError, generate_insight, generate_quiz
from .settings import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
log = logging.getLogger("finedu_ai")

app = FastAPI(title="finEdu AI", version="2.0.0")


class QuizRequest(BaseModel):
    article_id: int = Field(..., description="articles.article_id")
    content: Optional[str] = Field(None, description="기사 본문. 있으면 DB 조회 대신 사용")


class QuizItem(BaseModel):
    question: str
    options: List[str] = Field(..., min_length=4, max_length=4)
    correct_index: int = Field(..., ge=0, le=3)
    explanation: Optional[str] = None
    question_type: Optional[str] = None
    difficulty: Optional[str] = None


class QuizResponse(BaseModel):
    article_id: int
    quiz: List[QuizItem]
    model: Optional[str] = None
    latency_ms: Optional[int] = None


class InterestTopic(BaseModel):
    name: str
    count: int = 0


class InterestInsightRequest(BaseModel):
    topics: List[InterestTopic]


class InterestInsightResponse(BaseModel):
    insight: str


def _llm_error(e: Exception) -> HTTPException:
    """BE(AiQuizClient)는 5xx 만 재시도한다 → 일시 장애는 503/504, 영구 오류는 422."""
    if isinstance(e, LlmNotConfigured):
        return HTTPException(status_code=503, detail=str(e))
    if isinstance(e, (APITimeoutError,)):
        return HTTPException(status_code=504, detail="LLM 응답 시간 초과")
    if isinstance(e, (RateLimitError, APIConnectionError)):
        return HTTPException(status_code=503, detail=f"LLM 일시 장애: {e.__class__.__name__}")
    if isinstance(e, APIStatusError):
        code = 503 if e.status_code >= 500 else 502
        return HTTPException(status_code=code, detail=f"LLM 오류 {e.status_code}")
    if isinstance(e, QuizGenerationError):
        return HTTPException(status_code=502, detail=f"퀴즈 생성 실패: {e}")
    return HTTPException(status_code=500, detail=f"알 수 없는 오류: {e.__class__.__name__}")


@app.post("/quiz", response_model=QuizResponse)
def create_quiz(req: QuizRequest) -> QuizResponse:
    content = (req.content or "").strip()
    if not content:
        try:
            content = db.fetch_article_content(req.article_id) or ""
        except Exception as e:  # noqa: BLE001
            log.warning("기사 조회 실패 article_id=%s: %s", req.article_id, e)
            raise HTTPException(status_code=503, detail="DB 조회 실패") from e
    if not content:
        # 본문이 없으면 재시도해도 같으므로 4xx
        raise HTTPException(status_code=404, detail=f"article_id={req.article_id} 본문이 없습니다.")

    started = time.perf_counter()
    try:
        items = generate_quiz(get_client(), content)
    except Exception as e:  # noqa: BLE001
        log.warning("퀴즈 생성 실패 article_id=%s: %s", req.article_id, e)
        raise _llm_error(e) from e
    latency = int((time.perf_counter() - started) * 1000)
    log.info("퀴즈 생성 article_id=%s model=%s %dms", req.article_id, settings.model, latency)
    return QuizResponse(
        article_id=req.article_id,
        quiz=[QuizItem.model_validate(x) for x in items],
        model=settings.model,
        latency_ms=latency,
    )


@app.post("/news/interest-insight", response_model=InterestInsightResponse)
def create_interest_insight(req: InterestInsightRequest) -> InterestInsightResponse:
    topics = [t.model_dump() for t in req.topics if t.name.strip()]
    if not topics:
        raise HTTPException(status_code=400, detail="topics가 비어 있습니다.")
    try:
        return InterestInsightResponse(insight=generate_insight(get_client(), topics))
    except Exception as e:  # noqa: BLE001
        raise _llm_error(e) from e


@app.get("/health")
def health():
    """liveness — 프로세스 생존만 확인 (배포 헬스체크용)."""
    return {"status": "ok"}


@app.get("/ready")
def ready():
    """readiness — DB 연결·API 키 설정 확인."""
    checks = {"db": db.ping(), "anthropic_api_key": bool(settings.anthropic_api_key), "model": settings.model}
    ok = checks["db"] and checks["anthropic_api_key"]
    return JSONResponse(status_code=200 if ok else 503, content={"status": "ok" if ok else "degraded", **checks})


def run() -> None:
    import uvicorn

    uvicorn.run(app, host=settings.app_host, port=settings.app_port)


if __name__ == "__main__":
    run()
