from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    openrag_url: str = Field(
        default="http://localhost:3000",
        validation_alias=AliasChoices("FASTPDF_OPENRAG_OPENRAG_URL", "OPENRAG_URL"),
    )
    openrag_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("FASTPDF_OPENRAG_OPENRAG_API_KEY", "OPENRAG_API_KEY"),
    )
    langflow_url: str = Field(
        default="http://localhost:7860",
        validation_alias=AliasChoices("FASTPDF_OPENRAG_LANGFLOW_URL", "LANGFLOW_URL"),
    )
    langflow_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("FASTPDF_OPENRAG_LANGFLOW_API_KEY", "LANGFLOW_KEY"),
    )
    langflow_flows_root: Path = Field(
        default=Path.home() / ".openrag" / "flows",
        validation_alias=AliasChoices("FASTPDF_OPENRAG_LANGFLOW_FLOWS_ROOT"),
    )
    agent_flow_filename: str = Field(
        default="openrag_agent.json",
        validation_alias=AliasChoices("FASTPDF_OPENRAG_AGENT_FLOW_FILENAME"),
    )
    ingestion_flow_filename: str = Field(
        default="ingestion_flow.json",
        validation_alias=AliasChoices("FASTPDF_OPENRAG_INGESTION_FLOW_FILENAME"),
    )
    materialized_root: Path = Field(
        default=Path("data/materialized"),
        validation_alias=AliasChoices("FASTPDF_OPENRAG_MATERIALIZED_ROOT"),
    )
    output_root: Path = Field(
        default=Path("outputs"),
        validation_alias=AliasChoices("FASTPDF_OPENRAG_OUTPUT_ROOT"),
    )
    extraction_root: Path = Field(
        default=Path("data/extracted"),
        validation_alias=AliasChoices("FASTPDF_OPENRAG_EXTRACTION_ROOT"),
    )
    trace_root: Path = Field(
        default=Path("outputs/traces"),
        validation_alias=AliasChoices("FASTPDF_OPENRAG_TRACE_ROOT"),
    )
    default_chunk_size: int = Field(
        default=1200,
        validation_alias=AliasChoices("FASTPDF_OPENRAG_DEFAULT_CHUNK_SIZE"),
    )
    default_chunk_overlap: int = Field(
        default=150,
        validation_alias=AliasChoices("FASTPDF_OPENRAG_DEFAULT_CHUNK_OVERLAP"),
    )
    structure_chunk_target_chars: int = Field(
        default=1400,
        validation_alias=AliasChoices("FASTPDF_OPENRAG_STRUCTURE_CHUNK_TARGET_CHARS"),
    )
    structure_chunk_overlap_blocks: int = Field(
        default=1,
        validation_alias=AliasChoices("FASTPDF_OPENRAG_STRUCTURE_CHUNK_OVERLAP_BLOCKS"),
    )
    retrieval_rerank_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("FASTPDF_OPENRAG_RETRIEVAL_RERANK_ENABLED"),
    )
    retrieval_rerank_candidate_limit: int = Field(
        default=24,
        validation_alias=AliasChoices("FASTPDF_OPENRAG_RETRIEVAL_RERANK_CANDIDATE_LIMIT"),
    )
    retrieval_rerank_top_k: int = Field(
        default=8,
        validation_alias=AliasChoices("FASTPDF_OPENRAG_RETRIEVAL_RERANK_TOP_K"),
    )
    backend_rerank_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("FASTPDF_OPENRAG_BACKEND_RERANK_ENABLED"),
    )
    backend_rerank_provider: str = Field(
        default="cross_encoder",
        validation_alias=AliasChoices("FASTPDF_OPENRAG_BACKEND_RERANK_PROVIDER"),
    )
    backend_rerank_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        validation_alias=AliasChoices("FASTPDF_OPENRAG_BACKEND_RERANK_MODEL"),
    )
    backend_rerank_top_n: int = Field(
        default=8,
        validation_alias=AliasChoices("FASTPDF_OPENRAG_BACKEND_RERANK_TOP_N"),
    )
    backend_rerank_candidate_limit: int = Field(
        default=16,
        validation_alias=AliasChoices("FASTPDF_OPENRAG_BACKEND_RERANK_CANDIDATE_LIMIT"),
    )
    backend_search_rerank_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("FASTPDF_OPENRAG_BACKEND_SEARCH_RERANK_ENABLED"),
    )
    backend_search_rerank_candidate_limit: int = Field(
        default=24,
        validation_alias=AliasChoices("FASTPDF_OPENRAG_BACKEND_SEARCH_RERANK_CANDIDATE_LIMIT"),
    )
    verification_limit: int = Field(
        default=3,
        validation_alias=AliasChoices("FASTPDF_OPENRAG_VERIFICATION_LIMIT"),
    )
    verification_concurrency: int = Field(
        default=2,
        validation_alias=AliasChoices("FASTPDF_OPENRAG_VERIFICATION_CONCURRENCY"),
    )
    verification_score_threshold: float = Field(
        default=0.15,
        validation_alias=AliasChoices("FASTPDF_OPENRAG_VERIFICATION_SCORE_THRESHOLD"),
    )
    filter_prefix: str = Field(
        default="fastpdf-openrag-native",
        validation_alias=AliasChoices("FASTPDF_OPENRAG_FILTER_PREFIX"),
    )
    page_summary_concurrency: int = Field(
        default=4,
        validation_alias=AliasChoices("FASTPDF_OPENRAG_PAGE_SUMMARY_CONCURRENCY"),
    )
    google_application_credentials: Path | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "FASTPDF_OPENRAG_GOOGLE_APPLICATION_CREDENTIALS",
            "GOOGLE_APPLICATION_CREDENTIALS",
        ),
    )
    pdf_render_dpi: int = Field(
        default=144,
        validation_alias=AliasChoices("FASTPDF_OPENRAG_PDF_RENDER_DPI"),
    )
    opensearch_url: str = Field(
        default="https://127.0.0.1:9200",
        validation_alias=AliasChoices("FASTPDF_OPENRAG_OPENSEARCH_URL"),
    )
    opensearch_username: str | None = Field(
        default=None,
        validation_alias=AliasChoices("FASTPDF_OPENRAG_OPENSEARCH_USERNAME"),
    )
    opensearch_password: str | None = Field(
        default=None,
        validation_alias=AliasChoices("FASTPDF_OPENRAG_OPENSEARCH_PASSWORD"),
    )
    opensearch_index_name: str = Field(
        default="documents",
        validation_alias=AliasChoices("FASTPDF_OPENRAG_OPENSEARCH_INDEX_NAME"),
    )
    opensearch_verify_ssl: bool = Field(
        default=False,
        validation_alias=AliasChoices("FASTPDF_OPENRAG_OPENSEARCH_VERIFY_SSL"),
    )
    ui_host: str = Field(
        default="127.0.0.1",
        validation_alias=AliasChoices("FASTPDF_OPENRAG_UI_HOST"),
    )
    ui_port: int = Field(
        default=8077,
        validation_alias=AliasChoices("FASTPDF_OPENRAG_UI_PORT"),
    )
    openrag_ingest_wait_timeout: float = Field(
        default=180.0,
        validation_alias=AliasChoices("FASTPDF_OPENRAG_INGEST_WAIT_TIMEOUT"),
    )
    openrag_request_timeout: float = Field(
        default=600.0,
        validation_alias=AliasChoices("FASTPDF_OPENRAG_REQUEST_TIMEOUT"),
    )
    extractor_llm_provider: str | None = Field(
        default=None,
        validation_alias=AliasChoices("FASTPDF_OPENRAG_EXTRACTOR_LLM_PROVIDER"),
    )
    extractor_llm_model: str | None = Field(
        default=None,
        validation_alias=AliasChoices("FASTPDF_OPENRAG_EXTRACTOR_LLM_MODEL"),
    )
    renderer_llm_provider: str | None = Field(
        default=None,
        validation_alias=AliasChoices("FASTPDF_OPENRAG_RENDERER_LLM_PROVIDER"),
    )
    renderer_llm_model: str | None = Field(
        default=None,
        validation_alias=AliasChoices("FASTPDF_OPENRAG_RENDERER_LLM_MODEL"),
    )
    renderer_disable_retrieval: bool = Field(
        default=True,
        validation_alias=AliasChoices("FASTPDF_OPENRAG_RENDERER_DISABLE_RETRIEVAL"),
    )
    editor_llm_provider: str | None = Field(
        default=None,
        validation_alias=AliasChoices("FASTPDF_OPENRAG_EDITOR_LLM_PROVIDER"),
    )
    editor_llm_model: str | None = Field(
        default=None,
        validation_alias=AliasChoices("FASTPDF_OPENRAG_EDITOR_LLM_MODEL"),
    )
    editor_disable_retrieval: bool = Field(
        default=True,
        validation_alias=AliasChoices("FASTPDF_OPENRAG_EDITOR_DISABLE_RETRIEVAL"),
    )


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()


def fresh_settings() -> AppSettings:
    get_settings.cache_clear()
    return get_settings()
