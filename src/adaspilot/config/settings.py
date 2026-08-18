"""Typed application settings loaded from environment variables."""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    """Supported runtime environments."""

    DEVELOPMENT = "development"
    TEST = "test"


class Settings(BaseSettings):
    """ADASPilot runtime configuration.

    All environment variables use the ``ADASPILOT_`` prefix. Settings are
    intentionally centralized so later modules do not hard-code model names,
    thresholds, paths, or infrastructure endpoints.
    """

    model_config = SettingsConfigDict(
        env_prefix="ADASPILOT_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "ADASPilot"
    app_version: str = "0.1.0"
    environment: Environment = Environment.DEVELOPMENT
    log_level: str = "INFO"

    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8000, ge=1, le=65535)

    corpus_manifest_path: Path = Path("data/corpus/manifest.json")
    raw_corpus_dir: Path = Path("data/corpus/raw")
    processed_data_dir: Path = Path("data/processed")
    index_dir: Path = Path("data/index")
    results_dir: Path = Path("data/results")
    benchmark_path: Path = Path("data/evaluation/benchmark.jsonl")
    benchmark_targets_path: Path = Path("data/evaluation/targets.json")

    render_dpi: int = Field(default=120, ge=72, le=300)
    parent_chunk_chars: int = Field(default=1800, ge=500, le=6000)
    child_chunk_chars: int = Field(default=520, ge=150, le=2000)
    child_chunk_overlap: int = Field(default=80, ge=0, le=500)
    minimum_ocr_chars: int = Field(default=40, ge=0, le=1000)

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimension: int = Field(default=384, ge=8, le=8192)
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    qdrant_path: Path = Path("data/index/qdrant")
    text_collection_prefix: str = "adas_text"
    dense_top_k: int = Field(default=8, ge=1, le=100)
    retrieval_candidate_k: int = Field(default=30, ge=2, le=500)
    rrf_constant: int = Field(default=60, ge=1, le=1000)
    hyde_score_threshold: float = Field(default=0.45, ge=-1.0, le=1.0)
    enable_hyde: bool = False
    experiment_minimum_gain: float = Field(default=0.01, ge=0.0, le=1.0)

    evidence_score_threshold: float = Field(default=0.35, ge=-1.0, le=1.0)
    direct_recursion_limit: int = Field(default=10, ge=8, le=20)
    node_timeout_seconds: float = Field(default=15.0, gt=0, le=120)
    react_max_tool_calls: int = Field(default=6, ge=1, le=10)
    react_recursion_limit: int = Field(default=8, ge=4, le=20)
    plan_max_tasks: int = Field(default=5, ge=1, le=10)
    plan_max_depth: int = Field(default=3, ge=1, le=5)
    plan_max_parallel_workers: int = Field(default=2, ge=1, le=4)
    plan_max_replans: int = Field(default=1, ge=0, le=2)
    max_input_chars: int = Field(default=4000, ge=100, le=20000)
    max_image_bytes: int = Field(default=12_000_000, ge=100_000, le=50_000_000)
    max_image_pixels: int = Field(default=20_000_000, ge=1_000_000, le=100_000_000)

    llm_base_url: str | None = None
    llm_api_key: SecretStr | None = None
    llm_model: str | None = None
    llm_timeout_seconds: float = Field(default=60.0, gt=0, le=300)
    llm_input_cost_per_million: float = Field(default=0.0, ge=0)
    llm_output_cost_per_million: float = Field(default=0.0, ge=0)
    tavily_api_key: SecretStr | None = None

    langsmith_tracing: bool = False
    langsmith_api_key: SecretStr | None = None
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    langsmith_project: str = "adaspilot-local"

    ragas_judge_model: str | None = None
    evaluation_report_path: Path = Path("data/results/system_evaluation.json")
    upload_dir: Path = Path("data/corpus/uploads")
    max_upload_bytes: int = Field(default=25_000_000, ge=100_000, le=100_000_000)
    stream_chunk_chars: int = Field(default=80, ge=20, le=500)

    visual_backend: str = "clip"
    visual_image_model: str = "Qdrant/clip-ViT-B-32-vision"
    visual_text_model: str = "Qdrant/clip-ViT-B-32-text"
    colpali_model: str = "vidore/colSmol-256M"
    visual_top_k: int = Field(default=5, ge=1, le=50)

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        """Normalize and validate standard logging levels."""

        normalized = value.upper()
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if normalized not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}")
        return normalized

    @field_validator("visual_backend")
    @classmethod
    def validate_visual_backend(cls, value: str) -> str:
        normalized = value.lower()
        if normalized not in {"clip", "colpali"}:
            raise ValueError("visual_backend must be 'clip' or 'colpali'")
        return normalized


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached settings instance for the process."""

    return Settings()
