"""
하위호환: kobart-summarization.py / kr-finbert-sc.py 오프라인 스크립트용 설정.
API 서버 설정은 finedu_ai/settings.py 참고.
"""
from finedu_ai.settings import settings

DB_CONFIG = settings.db_config
ANTHROPIC_API_KEY = settings.anthropic_api_key
ANTHROPIC_MODEL = settings.model
APP_HOST = settings.app_host
APP_PORT = settings.app_port
