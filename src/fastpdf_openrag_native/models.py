from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class MaterializedPage(BaseModel):
    run_id: str
    pdf_id: str
    page: int
    order_index: int
    source_filename: str
    relative_path: str
    document_type: str = "text/markdown"
    label: str | None = None
    service_date: str | None = None
    patient_name: str | None = None
    text_length: int
    text_preview: str
    source_fields: list[str] = Field(default_factory=list)
    artifacts: dict[str, str] = Field(default_factory=dict)
    retrieval_filenames: list[str] = Field(default_factory=list)
    retrieval_relative_paths: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def retrieval_sources(self) -> list[str]:
        return list(self.retrieval_filenames or [self.source_filename])


class MaterializedRetrievalDocument(BaseModel):
    run_id: str
    pdf_id: str
    page: int
    order_index: int
    chunk_index: int
    source_filename: str
    relative_path: str
    document_type: str = "text/markdown"
    text_length: int
    text_preview: str
    parent_source_filename: str | None = None
    section_title: str | None = None
    source_fields: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MaterializationManifest(BaseModel):
    run_id: str
    created_at: datetime = Field(default_factory=utc_now)
    source_kind: str
    patient_name: str | None = None
    total_pages: int
    materialized_pages: int
    retrieval_document_count: int = 0
    requested_max_pages: int | None = None
    is_partial_run: bool = False
    page_documents: list[MaterializedPage] = Field(default_factory=list)
    retrieval_documents: list[MaterializedRetrievalDocument] = Field(default_factory=list)

    def page_lookup(self) -> dict[tuple[str, int], MaterializedPage]:
        return {(page.pdf_id, page.page): page for page in self.page_documents}

    def ingest_documents(self) -> list[MaterializedRetrievalDocument]:
        derived_from_pages = [
            MaterializedRetrievalDocument(
                run_id=page.run_id,
                pdf_id=page.pdf_id,
                page=page.page,
                order_index=page.order_index,
                chunk_index=1,
                source_filename=page.source_filename,
                relative_path=page.relative_path,
                document_type=page.document_type,
                text_length=page.text_length,
                text_preview=page.text_preview,
                parent_source_filename=page.source_filename,
                section_title=None,
                source_fields=list(page.source_fields),
                metadata=dict(page.metadata),
            )
            for page in self.page_documents
        ]
        if not self.retrieval_documents:
            return derived_from_pages

        documents = list(self.retrieval_documents)
        existing_filenames = {document.source_filename for document in documents}
        for page_document, derived_document in zip(self.page_documents, derived_from_pages, strict=False):
            if page_document.retrieval_filenames:
                continue
            if derived_document.source_filename in existing_filenames:
                continue
            documents.append(derived_document)
        return documents

    def all_source_filenames(self) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for page in self.page_documents:
            if page.source_filename not in seen:
                seen.add(page.source_filename)
                ordered.append(page.source_filename)
            for filename in page.retrieval_sources():
                if filename not in seen:
                    seen.add(filename)
                    ordered.append(filename)
        for doc in self.retrieval_documents:
            if doc.source_filename not in seen:
                seen.add(doc.source_filename)
                ordered.append(doc.source_filename)
        return ordered


class PageRef(BaseModel):
    pdf_id: str
    page: int


class SummaryScope(BaseModel):
    scope_id: str
    title: str
    objective: str = (
        "Produce a grounded medical/legal summary of the selected pages. "
        "Do not invent facts and keep chronology intact."
    )
    page_refs: list[PageRef] = Field(default_factory=list)


class EvidenceHit(BaseModel):
    filename: str
    text: str
    score: float
    page: int | None = None
    mimetype: str | None = None
    base_score: float | None = None
    rerank_score: float | None = None
    retrieval_rank: int | None = None


class PageMapSummary(BaseModel):
    pdf_id: str
    page: int
    source_filename: str
    summary: str
    key_facts: list[str] = Field(default_factory=list)
    raw_response: str
    retrieved_sources: list[EvidenceHit] = Field(default_factory=list)


class VerifiedSentence(BaseModel):
    sentence: str
    supported: bool
    evidence: list[EvidenceHit] = Field(default_factory=list)


class ScopedSummaryResult(BaseModel):
    run_id: str
    scope: SummaryScope
    created_at: datetime = Field(default_factory=utc_now)
    source_filenames: list[str] = Field(default_factory=list)
    page_summaries: list[PageMapSummary] = Field(default_factory=list)
    draft_title: str
    draft_summary: str
    chronology: list[str] = Field(default_factory=list)
    verified_sentences: list[VerifiedSentence] = Field(default_factory=list)
    supported_summary: str = ""
    unsupported_sentences: list[str] = Field(default_factory=list)
    debug: dict[str, Any] = Field(default_factory=dict)


class IngestedDocumentResult(BaseModel):
    filename: str
    status: str
    successful_files: int = 0
    failed_files: int = 0
    task_id: str | None = None
    delete_error: str | None = None


class KnowledgeFilterResult(BaseModel):
    filter_id: str
    filter_name: str
    data_sources: list[str] = Field(default_factory=list)


class TraceEvent(BaseModel):
    timestamp: datetime = Field(default_factory=utc_now)
    stage: str
    service: str
    action: str
    status: str = "ok"
    request: dict[str, Any] = Field(default_factory=dict)
    response: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    output_files: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class OpenSearchChunkRecord(BaseModel):
    id: str | None = None
    filename: str
    text: str
    score: float | None = None
    page: int | None = None
    mimetype: str | None = None
    embedding_model: str | None = None
    embedding_dimensions: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkStats(BaseModel):
    total_chunks: int = 0
    documents_with_chunks: int = 0
    chunks_per_document: dict[str, int] = Field(default_factory=dict)
    single_chunk_documents: list[str] = Field(default_factory=list)
    multi_chunk_documents: list[str] = Field(default_factory=list)
    empty_documents: list[str] = Field(default_factory=list)


class OpenSearchIndexDiagnostics(BaseModel):
    database_backend: str | None = None
    cluster_name: str | None = None
    documents_index_name: str | None = None
    knowledge_filter_index_name: str | None = None
    document_count: int | None = None
    retrieval_mode: str | None = None
    application_retrieval_mode: str | None = None
    semantic_weight: float | None = None
    keyword_weight: float | None = None
    reranking_enabled: bool | None = None
    reranker_location: str | None = None
    reranker_type: str | None = None
    reranker_model: str | None = None
    chunking_enabled: bool | None = None
    structure_chunking_strategy: str | None = None
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    prechunk_target_chars: int | None = None
    prechunk_overlap_blocks: int | None = None
    embedding_provider: str | None = None
    embedding_model: str | None = None
    vector_fields: list[str] = Field(default_factory=list)
    cluster_health: dict[str, Any] = Field(default_factory=dict)
    allocation_explain: dict[str, Any] = Field(default_factory=dict)
    index_mapping: dict[str, Any] = Field(default_factory=dict)
    index_settings: dict[str, Any] = Field(default_factory=dict)
    cat_indices: str | None = None
    cat_shards: str | None = None
    index_exists: bool | None = None
    index_search_error: dict[str, Any] | None = None


class LangflowFlowDiagnostics(BaseModel):
    langflow_url: str | None = None
    has_api_key: bool = False
    flows_root: str | None = None
    agent_flow_path: str | None = None
    agent_flow_id: str | None = None
    agent_flow_name: str | None = None
    agent_flow_locked: bool | None = None
    agent_flow_rerank_marker_present: bool | None = None
    agent_flow_prompt_upgraded: bool | None = None
    ingestion_flow_path: str | None = None
    ingestion_flow_id: str | None = None
    ingestion_chunk_size: int | None = None
    ingestion_chunk_overlap: int | None = None
    backend_reranking_enabled: bool | None = None
    backend_reranker_provider: str | None = None
    backend_reranker_model: str | None = None
    backend_reranker_top_n: int | None = None


class FlowUpgradeResult(BaseModel):
    created_at: datetime = Field(default_factory=utc_now)
    backup_dir: str | None = None
    agent_flow_path: str | None = None
    ingestion_flow_path: str | None = None
    agent_flow_live_patch_applied: bool = False
    ingestion_flow_live_patch_applied: bool = False
    agent_flow_marker_present: bool = False
    ingestion_settings_updated: bool = False
    notes: list[str] = Field(default_factory=list)


class OpenSearchRepairResult(BaseModel):
    created_at: datetime = Field(default_factory=utc_now)
    index_name: str
    output_dir: str | None = None
    target_embedding_provider: str | None = None
    target_embedding_model: str | None = None
    deleted_existing_index: bool = False
    normalized_indices: list[str] = Field(default_factory=list)
    before: OpenSearchIndexDiagnostics
    after: OpenSearchIndexDiagnostics


class PdfPipelineResult(BaseModel):
    run_id: str
    source_pdf: str
    extraction_dir: str
    manifest_path: str
    trace_path: str
    trace_summary_path: str
    trace_dir: str
    chunk_dump_dir: str
    total_pages: int
    materialized_pages: int
    requested_max_pages: int | None = None
    is_partial_run: bool = False
    summary_path: str | None = None
    summary: ScopedSummaryResult | None = None
    summary_error: str | None = None
    ingestion_results: list[IngestedDocumentResult] = Field(default_factory=list)
    knowledge_filter: KnowledgeFilterResult | None = None
    chunk_stats: ChunkStats | None = None
    opensearch_diagnostics: OpenSearchIndexDiagnostics | None = None
