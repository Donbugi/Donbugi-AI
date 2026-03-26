"""
PostgreSQL articles 테이블의 content 열에 KR-FinBert-SC 모델을 적용하여
긍정/부정/중립 감성 라벨링
"""
import psycopg2
from psycopg2.extras import RealDictCursor
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config import DB_CONFIG


def load_model():
    """KR-FinBert-SC 모델 로드"""
    tokenizer = AutoTokenizer.from_pretrained("snunlp/KR-FinBert-SC")
    model = AutoModelForSequenceClassification.from_pretrained("snunlp/KR-FinBert-SC")
    model.eval()
    return tokenizer, model


def predict_sentiment(text: str, tokenizer, model, max_length: int = 512) -> str:
    """
    텍스트의 감성 예측 (긍정/부정/중립)
    - 빈 텍스트: '중립' 반환
    - max_length: BERT 최대 토큰 길이, 긴 텍스트는 앞부분만 사용
    """
    if not text or not str(text).strip():
        return "중립"

    text = str(text).strip()

    with torch.no_grad():
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
            add_special_tokens=True,
        )

        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
            model = model.cuda()

        outputs = model(**inputs)
        logits = outputs.logits
        pred_id = torch.argmax(logits, dim=1).item()

    # 모델의 id2label 사용 (0: negative, 1: neutral, 2: positive)
    label_map = model.config.id2label
    label = label_map.get(pred_id, "neutral")

    # 한글로 변환
    label_ko = {"negative": "부정", "neutral": "중립", "positive": "긍정"}.get(
        label.lower(), "중립"
    )
    return label_ko


def ensure_sentiment_column(conn):
    """articles 테이블에 sentiment 컬럼이 없으면 추가"""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'articles' AND column_name = 'sentiment'
        """)
        if cur.fetchone() is None:
            cur.execute(
                "ALTER TABLE articles ADD COLUMN IF NOT EXISTS sentiment VARCHAR(10)"
            )
            conn.commit()
            print("sentiment 컬럼이 추가되었습니다.")


def main():
    # 모델 로드
    print("모델 로딩 중...")
    tokenizer, model = load_model()
    print("모델 로드 완료.")

    # PostgreSQL 연결
    try:
        conn = psycopg2.connect(**DB_CONFIG)
    except psycopg2.Error as e:
        print(f"DB 연결 실패: {e}")
        print("config.py 또는 환경변수(DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD)를 확인하세요.")
        return

    ensure_sentiment_column(conn)

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # content가 있는 articles 조회 (sentiment가 비어있거나 None인 것만 선택 가능)
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
                sentiment = predict_sentiment(content, tokenizer, model)
                update_cur.execute(
                    """
                    UPDATE articles SET sentiment = %s WHERE article_id = %s
                    """,
                    (sentiment, article_id),
                )
                conn.commit()

                if (i + 1) % 10 == 0 or i == 0:
                    print(f"[{i + 1}/{len(rows)}] article_id={article_id} -> {sentiment}")
            except Exception as e:
                print(f"article_id={article_id} 처리 실패: {e}")
                conn.rollback()
                continue

        update_cur.close()
        print("감성 라벨링 완료.")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
