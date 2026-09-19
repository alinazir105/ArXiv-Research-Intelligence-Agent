from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):

    # Define fields with type annotations
    OPENAI_API_KEY: str
    QDRANT_URL: str = "http://localhost:6333"
    REDIS_URL : str = "redis://localhost:6379"
    COLLECTION_NAME: str = "arxiv_papers"
    VECTOR_DIMENSION: int = 1536
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    LLM_MODEL: str = "gpt-4o-mini",
    LANGCHAIN_API_KEY: str = ""
    LANGCHAIN_TRACING_V2: str = "true"
    LANGCHAIN_PROJECT: str = "arxiv-research-agent"

    # Bind the model to read automatically from your .env file
    model_config = SettingsConfigDict(env_file=".env")


# Use lru_cache to cache configuration loading and avoid re-reading files
@lru_cache
def get_settings() -> Settings:
    return Settings()

# Instantiate the settings for quick import across your app
settings = get_settings()