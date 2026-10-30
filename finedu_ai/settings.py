"""
런타임 설정. 운영에서는 Docker env-file / GitHub Actions secret 으로 주입한다.
"""
import os
from dataclasses import dataclass, field


def _bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", "").strip())
    # 생성 모델: 최신 Haiku. ANTHROPIC_MODEL 로 교체 가능
    model: str = field(default_factory=lambda: os.getenv("ANTHROPIC_MODEL", "").strip() or "claude-haiku-5-5")
    # 품질 검증(LLM-as-judge) 모델. 비우면 생성 모델 사용
    judge_model: str = field(default_factory=lambda: os.getenv("QUIZ_JUDGE_MODEL", "").strip())
    judge_enabled: bool = field(default_factory=lambda: _bool("QUIZ_JUDGE_ENABLED", False))
    request_timeout_s: float = field(default_factory=lambda: float(os.getenv("ANTHROPIC_TIMEOUT_S", "45")))
    max_retries: int = field(default_factory=lambda: int(os.getenv("ANTHROPIC_MAX_RETRIES", "2")))
    max_content_chars: int = 18_000

    db_host: str = field(default_factory=lambda: os.getenv("DB_HOST", "127.0.0.1"))
    db_port: str = field(default_factory=lambda: os.getenv("DB_PORT", "5432"))
    db_name: str = field(default_factory=lambda: os.getenv("DB_NAME", "finedu"))
    db_user: str = field(default_factory=lambda: os.getenv("DB_USER", ""))
    db_password: str = field(default_factory=lambda: os.getenv("DB_PASSWORD", ""))

    app_host: str = field(default_factory=lambda: os.getenv("APP_HOST", "127.0.0.1"))
    app_port: int = field(default_factory=lambda: int(os.getenv("APP_PORT", "8000")))

    @property
    def effective_judge_model(self) -> str:
        return self.judge_model or self.model

    @property
    def db_config(self) -> dict:
        return {
            "host": self.db_host,
            "port": self.db_port,
            "database": self.db_name,
            "user": self.db_user,
            "password": self.db_password,
        }


settings = Settings()
