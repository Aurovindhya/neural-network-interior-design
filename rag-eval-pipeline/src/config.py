from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    google_api_key: str = ""
    llm_model: str = "gpt-4o"
    llm_temperature: float = 0.2

    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    vector_store: str = "chroma"
    chroma_persist_dir: str = "./data/chroma_db"
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    zilliz_uri: str = ""
    zilliz_token: str = ""

    embedding_model: str = "text-embedding-3-small"

    app_host: str = "0.0.0.0"
    app_port: int = 8000
    secret_key: str = "dev-secret-key"
    jwt_expire_minutes: int = 60

    eval_enabled: bool = True
    eval_model: str = "gpt-4o-mini"

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()
