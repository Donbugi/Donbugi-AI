"""PostgreSQL 커넥션 풀 (요청마다 connect/close 하던 구조 대체)."""
import logging
import threading
from contextlib import contextmanager
from typing import Iterator, Optional

import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import ThreadedConnectionPool

from .settings import settings

log = logging.getLogger(__name__)

_pool: Optional[ThreadedConnectionPool] = None
_lock = threading.Lock()


def _get_pool() -> ThreadedConnectionPool:
    global _pool
    if _pool is None:
        with _lock:
            if _pool is None:
                _pool = ThreadedConnectionPool(minconn=1, maxconn=5, connect_timeout=5, **settings.db_config)
    return _pool


@contextmanager
def connection() -> Iterator[psycopg2.extensions.connection]:
    pool = _get_pool()
    conn = pool.getconn()
    try:
        yield conn
    finally:
        pool.putconn(conn)


def fetch_article_content(article_id: int) -> Optional[str]:
    with connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT content FROM articles WHERE article_id = %s AND content IS NOT NULL AND content <> ''",
                (article_id,),
            )
            row = cur.fetchone()
        conn.rollback()  # 읽기 전용 트랜잭션 정리 후 풀 반환
    return row["content"] if row else None


def ping() -> bool:
    try:
        with connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
            conn.rollback()
        return True
    except Exception as e:  # noqa: BLE001
        log.warning("DB ping 실패: %s", e)
        return False
