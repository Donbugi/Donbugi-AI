# finEdu AI — 변경 사항과 로드맵

## 이번 변경 (v2.0)
| 항목 | 이전 | 이후 |
|---|---|---|
| 구조 | `quiz-generation.py` 단일 파일 | `finedu_ai/` 패키지 (settings·db·llm·quiz·main). 기존 실행 명령 유지 |
| 모델 | claude-haiku-4-5 | **claude-haiku-5-5** (`ANTHROPIC_MODEL`로 교체) |
| 출력 파싱 | 정규식으로 \`\`\`json 블록 추출 → 실패 시 502 | **tool use + JSON 스키마 강제** |
| 품질 | 형식 검사만 | 규칙 검증(보기 중복·금지 보기·정답 길이 편향·해설) 실패 시 **피드백 붙여 1회 재생성** + 선택형 LLM-as-judge(`QUIZ_JUDGE_ENABLED`) |
| 메타데이터 | 없음 | `question_type`(concept/inference/main_idea/context), `difficulty` — BE 는 모르는 필드를 무시하므로 하위호환 |
| 클라이언트 | 요청마다 생성 | 싱글톤, 타임아웃 45초, 429/5xx 재시도 |
| DB | 요청마다 connect | `ThreadedConnectionPool` |
| 오류 코드 | 전부 502 | 일시 장애 503/504(BE 재시도 대상), 본문 없음 404, 품질 실패 502 |
| 헬스체크 | `/health` | `/health`(liveness) + `/ready`(DB·API 키) |
| 테스트 | 없음 | pytest 18개 (Anthropic mock) — `pip install -r requirements-dev.txt && pytest` |

> 참고: 시스템 프롬프트에 `cache_control` 을 붙였지만, 프롬프트 캐싱은 모델별 최소 토큰 수 이상일 때만 적용된다.
> 지금 지침 길이로는 캐시가 잡히지 않을 수 있고, 예시 문항(few-shot)을 늘리면 효과가 생긴다.

## 로드맵 (파인튜닝 없이)
1. **few-shot 예시 은행**: 좋은 문항 10~20개를 시스템 프롬프트에 넣어 품질 상향 (+ 프롬프트 캐싱 효과)
2. **사전 생성 + Batch API**: BE 가 크롤링 직후 퀴즈를 미리 만든다(이번 BE 변경). 대량일 때 Message Batches 로 비용 절감
3. **평가셋 회귀 테스트**: 기사 30~50개 + 기대 조건(정답 근거·난이도)을 고정해 모델/프롬프트 변경 시 점수 비교
4. **뉴스 감성 점수**: 기존 KR-FinBERT-SC(사전학습 모델 그대로) 를 크롤링 파이프라인에 붙여 기사별 긍/부정 → 경제날씨 보조 지표
5. **요약**: KoBART 대신 LLM 3줄 요약 (기사 카드·퀴즈 입력 길이 절감)
6. **임베딩 + pgvector**: 유사 기사 중복 제거, 읽은 기사 기반 추천, 의미 검색
7. **개인화 퀴즈**: 오답노트의 `question_type` 분포로 약한 유형 위주 출제
8. **모의투자 코치**: 거래내역 지표(회전율·집중도·손절 지연)를 규칙으로 계산하고 LLM 이 설명만 생성 (투자 권유 금지)
9. **관측성**: 요청별 모델·지연·토큰 사용량을 Prometheus 메트릭으로 노출, BE 와 같은 Grafana 에서 확인

`kobart-summarization.py`, `kr-finbert-sc.py` 는 운영 이미지에 포함되지 않는 오프라인 스크립트로 그대로 둔다.
