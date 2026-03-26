"""
PostgreSQL articles 테이블의 content 열을 KoBART로 요약하여 summary 열에 저장
"""
import psycopg2
from psycopg2.extras import RealDictCursor
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from config import DB_CONFIG

MODEL_ID = "gogamza/kobart-summarization"

# 요약 길이 조절 (토큰 기준)
# - SUMMARY_MAX_LENGTH: 출력 최대 길이
# - SUMMARY_MIN_LENGTH: 출력 최소 길이
# - LENGTH_PENALTY: 1.0보다 크면 긴 시퀀스에 유리 (보통 1.0~2.0)
SUMMARY_MAX_LENGTH = 356
SUMMARY_MIN_LENGTH = 128
LENGTH_PENALTY = 2.0


def load_model():
    """KoBART 요약 모델 로드"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
    model.eval()
    return tokenizer, model


def summarize_text(
    text: str,
    tokenizer,
    model,
    device: torch.device,
    max_input_length: int = 1024,
    max_summary_length: int = SUMMARY_MAX_LENGTH,
    min_summary_length: int = SUMMARY_MIN_LENGTH,
    length_penalty: float = LENGTH_PENALTY,
) -> str:
    """
    content를 요약 문자열로 반환.
    빈 텍스트는 빈 문자열 반환.
    """
    if not text or not str(text).strip():
        return ""

    text = str(text).strip()

    with torch.no_grad():
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=max_input_length,
            truncation=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        summary_ids = model.generate(
            **inputs,
            max_length=max_summary_length,
            min_length=min_summary_length,
            length_penalty=length_penalty,
            num_beams=4,
            early_stopping=True,
        )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()


def ensure_summary_column(conn):
    """articles 테이블에 summary 컬럼이 없으면 추가"""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'articles' AND column_name = 'summary'
        """)
        if cur.fetchone() is None:
            cur.execute(
                "ALTER TABLE articles ADD COLUMN IF NOT EXISTS summary TEXT"
            )
            conn.commit()
            print("summary 컬럼이 추가되었습니다.")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("모델 로딩 중...")
    tokenizer, model = load_model()
    model.to(device)
    print(f"모델 로드 완료. device={device}")

    try:
        conn = psycopg2.connect(**DB_CONFIG)
    except psycopg2.Error as e:
        print(f"DB 연결 실패: {e}")
        print("config.py 또는 환경변수(DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD)를 확인하세요.")
        return

    ensure_summary_column(conn)

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT article_id, content FROM articles
                WHERE content IS NOT NULL AND content != ''
                ORDER BY article_id
                """
            )
            rows = cur.fetchall()

        print(f"총 {len(rows)}개 기사 처리 예정")

        update_cur = conn.cursor()
        for i, row in enumerate(rows):
            article_id = row["article_id"]
            content = row["content"]

            try:
                summary = summarize_text(content, tokenizer, model, device)
                update_cur.execute(
                    """
                    UPDATE articles SET summary = %s WHERE article_id = %s
                    """,
                    (summary or None, article_id),
                )
                conn.commit()

                if (i + 1) % 10 == 0 or i == 0:
                    preview = (summary[:60] + "…") if summary and len(summary) > 60 else summary
                    print(f"[{i + 1}/{len(rows)}] article_id={article_id} -> {preview}")
            except Exception as e:
                print(f"article_id={article_id} 처리 실패: {e}")
                conn.rollback()
                continue

        update_cur.close()
        print("요약 저장 완료.")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
