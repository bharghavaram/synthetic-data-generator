import os
from dotenv import load_dotenv
load_dotenv()

class Settings:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o")
    CLAUDE_MODEL: str = os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20241022")
    MAX_BATCH_SIZE: int = int(os.getenv("MAX_BATCH_SIZE", "50"))
    DEFAULT_SAMPLES: int = int(os.getenv("DEFAULT_SAMPLES", "100"))
    OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "outputs")
    QUALITY_CHECK_ENABLED: bool = os.getenv("QUALITY_CHECK_ENABLED", "true").lower() == "true"
    DIVERSITY_THRESHOLD: float = float(os.getenv("DIVERSITY_THRESHOLD", "0.7"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.9"))
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "2048"))
    APP_HOST: str = os.getenv("APP_HOST", "0.0.0.0")
    APP_PORT: int = int(os.getenv("APP_PORT", "8000"))

settings = Settings()
