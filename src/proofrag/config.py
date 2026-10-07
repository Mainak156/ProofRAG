from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Application configuration for ProofRAG."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------

    app_name: str = "ProofRAG"
    app_version: str = "0.1.0"
    environment: str = "development"

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    data_dir: Path = PROJECT_ROOT / "data"
    documents_dir: Path = PROJECT_ROOT / "data" / "documents"
    okf_dir: Path = PROJECT_ROOT / "data" / "okf"
    chroma_dir: Path = PROJECT_ROOT / "data" / "chroma"

    # ------------------------------------------------------------------
    # LLM
    # ------------------------------------------------------------------

    groq_api_key: str = Field(default="", repr=False)
    llm_model: str = "openai/gpt-oss-120b"
    llm_temperature: float = 0.0
    llm_timeout: int = 60

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    chunk_size: int = 1000
    chunk_overlap: int = 150

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    retrieval_top_k: int = 5
    retrieval_score_threshold: float = 0.35

    # ------------------------------------------------------------------
    # Citation verification
    # ------------------------------------------------------------------

    citation_keyword_threshold: float = 0.20
    citation_embedding_threshold: float = 0.60

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    api_host: str = "127.0.0.1"
    api_port: int = 8000

    # ------------------------------------------------------------------
    # Safety / limits
    # ------------------------------------------------------------------

    max_question_length: int = 2000
    max_context_chunks: int = 8

    def ensure_directories(self) -> None:
        """Create required runtime directories if they do not exist."""

        for directory in (
            self.data_dir,
            self.documents_dir,
            self.okf_dir,
            self.chroma_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings."""

    settings = Settings()
    settings.ensure_directories()
    return settings