from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    DEEPSEEK_API_KEY: str
    DEEPSEEK_API_URL: str = "https://api.deepseek.com/chat/completions"
    ALLOWED_ORIGINS: list[str] = ["http://localhost:5173", "http://localhost:8000"]
    RATE_LIMIT_PER_MINUTE: int = 10
    LOG_LEVEL: str = "INFO"

    TOOL_TIMEOUT_SECONDS: float = 15.0
    REFLECTION_MIN_SCORE: int = 7
    MAX_REFLECTIONS: int = 2
    TAVILY_API_KEY: str = ""
    RAG_TOP_K: int = 3
    TRENDING_CANDIDATE_LIMIT: int = 15
    TRENDING_KEYWORD_LIMIT: int = 10

    # RAG v2
    RAG_VERSION: str = "v2"  # "v1" | "v2"
    RAG_HYBRID_TOPK: int = 20  # candidates fed to reranker
    RERANKER_MODEL: str = "BAAI/bge-reranker-base"
    SEMANTIC_CACHE_THRESHOLD: float = 0.92
    SEMANTIC_CACHE_MAX_SIZE: int = 256
    SEMANTIC_CACHE_TTL_SECONDS: int = 3600
    PERSONA_SOFT_BOOST_PER_MATCH: float = 0.05

    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def parse_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v


settings = Settings()
