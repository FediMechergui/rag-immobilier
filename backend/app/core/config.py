"""
Configuration settings for the Immobilier RAG Pipeline.
Uses pydantic-settings for environment variable management.
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List, Optional
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Settings
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8080, env="API_PORT")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Ollama Settings
    ollama_host: str = Field(default="ollama", env="OLLAMA_HOST")
    ollama_port: int = Field(default=11434, env="OLLAMA_PORT")
    ollama_model: str = Field(default="qwen2.5:0.5b", env="OLLAMA_MODEL")
    
    @property
    def ollama_base_url(self) -> str:
        return f"http://{self.ollama_host}:{self.ollama_port}"
    
    # PostgreSQL Settings
    postgres_host: str = Field(default="postgres", env="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, env="POSTGRES_PORT")
    postgres_db: str = Field(default="immobilier_rag", env="POSTGRES_DB")
    postgres_user: str = Field(default="raguser", env="POSTGRES_USER")
    postgres_password: str = Field(default="ragpassword", env="POSTGRES_PASSWORD")
    
    @property
    def database_url(self) -> str:
        return f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
    
    @property
    def sync_database_url(self) -> str:
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
    
    # ChromaDB Settings
    chroma_host: str = Field(default="chromadb", env="CHROMA_HOST")
    chroma_port: int = Field(default=8000, env="CHROMA_PORT")
    
    @property
    def chroma_url(self) -> str:
        return f"http://{self.chroma_host}:{self.chroma_port}"
    
    # HuggingFace Settings
    hf_model: str = Field(
        default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        env="HF_MODEL"
    )
    
    # Redis Settings
    redis_host: str = Field(default="redis", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    
    @property
    def redis_url(self) -> str:
        return f"redis://{self.redis_host}:{self.redis_port}"
    
    # RabbitMQ Settings
    rabbitmq_host: str = Field(default="rabbitmq", env="RABBITMQ_HOST")
    rabbitmq_port: int = Field(default=5672, env="RABBITMQ_PORT")
    rabbitmq_user: str = Field(default="raguser", env="RABBITMQ_USER")
    rabbitmq_password: str = Field(default="ragpassword", env="RABBITMQ_PASSWORD")
    
    @property
    def rabbitmq_url(self) -> str:
        return f"amqp://{self.rabbitmq_user}:{self.rabbitmq_password}@{self.rabbitmq_host}:{self.rabbitmq_port}//"
    
    # RAG Settings
    chunk_size: int = Field(default=800, env="RAG_CHUNK_SIZE")
    chunk_overlap: int = Field(default=150, env="RAG_CHUNK_OVERLAP")
    top_k: int = Field(default=5, env="RAG_TOP_K")
    similarity_threshold: float = Field(default=0.3, env="RAG_SIMILARITY_THRESHOLD")  # Lowered from 0.7 to get more results
    temperature: float = Field(default=0.3, env="RAG_TEMPERATURE")
    max_context_tokens: int = Field(default=4000, env="RAG_MAX_CONTEXT_TOKENS")
    
    # Web Search Settings
    web_search_enabled: bool = Field(default=True, env="WEB_SEARCH_ENABLED")
    web_search_max_results: int = Field(default=5, env="WEB_SEARCH_MAX_RESULTS")
    web_search_cache_ttl: int = Field(default=3600, env="WEB_SEARCH_CACHE_TTL")  # 1 hour
    
    # Upload Settings
    max_upload_size_mb: int = Field(default=50, env="MAX_UPLOAD_SIZE_MB")
    upload_directory: str = Field(default="/app/uploads", env="UPLOAD_DIRECTORY")
    allowed_extensions: List[str] = Field(default=[".pdf"], env="ALLOWED_EXTENSIONS")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# Web search whitelisted domains
WHITELISTED_DOMAINS = [
    "seloger.com",
    "bienici.com",
    "notaires.fr",
    "service-public.fr",
    "legifrance.gouv.fr",
    "anil.org",
    "pap.fr",
    "leboncoin.fr",
    "meilleursagents.com",
    "immo.lefigaro.fr"
]


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
