"""
하위호환 실행 진입점: `python quiz-generation.py`
실제 구현은 finedu_ai 패키지 (finedu_ai/main.py) 로 이동했다.
"""
from finedu_ai.main import app, run  # noqa: F401

if __name__ == "__main__":
    run()
