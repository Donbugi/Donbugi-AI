"""Anthropic 클라이언트 (프로세스당 1개, 타임아웃·재시도 설정)."""
import threading
from typing import Optional

from anthropic import Anthropic

from .settings import settings

_client: Optional[Anthropic] = None
_lock = threading.Lock()


class LlmNotConfigured(RuntimeError):
    pass


def get_client() -> Anthropic:
    """요청마다 새 클라이언트(=새 HTTP 커넥션 풀)를 만들던 구조를 싱글톤으로 변경."""
    global _client
    if not settings.anthropic_api_key:
        raise LlmNotConfigured("ANTHROPIC_API_KEY 가 설정되지 않았습니다.")
    if _client is None:
        with _lock:
            if _client is None:
                _client = Anthropic(
                    api_key=settings.anthropic_api_key,
                    timeout=settings.request_timeout_s,
                    max_retries=settings.max_retries,  # 429/5xx/연결 오류 지수 백오프 재시도
                )
    return _client


def set_client_for_tests(client) -> None:
    global _client
    _client = client
