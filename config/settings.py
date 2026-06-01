"""
Configuration settings for the Legal Copilot system.
Loads environment variables and provides typed configuration objects.
"""

import os
from pathlib import Path
from typing import Optional
try:
    from pydantic_settings import BaseSettings
    from pydantic import Field
except ImportError:
    # Fallback for older pydantic versions
    from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """System-wide configuration settings."""

    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")

    # OpenAI Configuration
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-3.5-turbo", env="OPENAI_MODEL")
    use_openai_generator: bool = Field(
        default=False,
        env="USE_OPENAI_GENERATOR",
        description="If True and OPENAI_API_KEY is set, use OpenAI (GPT) for answer generation; else use local LLM (Mistral). Aligns with methodology/Ch6 evaluation (GPT-3.5)."
    )
    openai_max_tokens: int = Field(default=100, env="OPENAI_MAX_TOKENS")  # Default 100 for demo
    openai_temperature: float = Field(default=0.1, env="OPENAI_TEMPERATURE")

    # Model Configuration
    embedding_model: str = Field(
        default="sentence-transformers/all-mpnet-base-v2",  # Primary model used in all evaluations; all-MiniLM-L6-v2 for ablation A5 only
        env="EMBEDDING_MODEL"
    )
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        env="RERANKER_MODEL"
    )
    llm_model: str = Field(
        default="mistralai/Mistral-7B-Instruct-v0.2",
        env="LLM_MODEL"
    )
    nli_model: str = Field(
        default="roberta-large-mnli",
        env="NLI_MODEL"
    )

    # Retrieval Configuration
    bm25_k1: float = Field(default=1.5, env="BM25_K1")
    bm25_b: float = Field(default=0.75, env="BM25_B")
    top_k_retrieval: int = Field(default=20, env="TOP_K_RETRIEVAL")
    top_k_rerank: int = Field(default=5, env="TOP_K_RERANK")
    fusion_alpha: float = Field(default=0.5, env="FUSION_ALPHA")

    # Generation Configuration
    max_tokens: int = Field(default=100, env="MAX_TOKENS")  # Default 100 for demo
    temperature: float = Field(default=0.1, env="TEMPERATURE")
    min_confidence: float = Field(default=0.7, env="MIN_CONFIDENCE")
    abstention_threshold: float = Field(default=0.3, env="ABSTENTION_THRESHOLD")

    # Paths
    data_path: Path = Field(default=Path("./data"), env="DATA_PATH")
    model_path: Path = Field(default=Path("./models"), env="MODEL_PATH")
    index_path: Path = Field(default=Path("./data/indices"), env="INDEX_PATH")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()
