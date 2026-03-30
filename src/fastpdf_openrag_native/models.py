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
    verified_sentences: list[VerifiedSentence] = Field(default_factory=list)
    verified_key_facts: list[VerifiedSentence] = Field(default_factory=list)
    supported_summary: str = ""
    supported_key_facts: list[str] = Field(default_factory=list)
    unsupported_sentences: list[str] = Field(default_factory=list)
    unsupported_key_facts: list[str] = Field(default_factory=list)
    passed_verification: bool = False


class VerifiedSentence(BaseModel):
    sentence: str
    supported: bool
    evidence: list[EvidenceHit] = Field(default_factory=list)


TRUTH_FIELD_NAMES = (
    "date_of_service",
    "facility",
    "provider",
    "patient_reference",
    "note_type",
    "chief_complaint",
    "hpi",
    "pmh",
    "psh",
    "social_history",
    "allergies",
    "medications",
    "vitals",
    "abnormal_labs",
    "diagnoses",
    "assessment",
    "treatment",
    "plan",
    "follow_up",
    "positive_ros",
    "positive_physical_exam",
    "residual_supported_facts",
)

TRUTH_FIELD_LABELS = {
    "date_of_service": "Date of Service",
    "facility": "Facility",
    "provider": "Provider",
    "patient_reference": "Patient Reference",
    "note_type": "Note Type",
    "chief_complaint": "Chief Complaint",
    "hpi": "History of Present Illness",
    "pmh": "Past Medical History",
    "psh": "Past Surgical History",
    "social_history": "Social History",
    "allergies": "Allergies",
    "medications": "Medications",
    "vitals": "Vitals",
    "abnormal_labs": "Abnormal Labs",
    "diagnoses": "Diagnoses",
    "assessment": "Assessment",
    "treatment": "Treatment",
    "plan": "Plan",
    "follow_up": "Follow Up",
    "positive_ros": "Positive Review of Systems",
    "positive_physical_exam": "Positive Physical Exam",
    "residual_supported_facts": "Additional Supported Facts",
}


class SupportedFact(BaseModel):
    value: str
    evidence_sentences: list[str] = Field(default_factory=list)
    evidence: list[EvidenceHit] = Field(default_factory=list)


class TruthLayerNote(BaseModel):
    note_id: str
    pdf_ids: list[str] = Field(default_factory=list)
    pages: list[int] = Field(default_factory=list)
    source_filenames: list[str] = Field(default_factory=list)
    date_of_service: list[SupportedFact] = Field(default_factory=list)
    facility: list[SupportedFact] = Field(default_factory=list)
    provider: list[SupportedFact] = Field(default_factory=list)
    patient_reference: list[SupportedFact] = Field(default_factory=list)
    note_type: list[SupportedFact] = Field(default_factory=list)
    chief_complaint: list[SupportedFact] = Field(default_factory=list)
    hpi: list[SupportedFact] = Field(default_factory=list)
    pmh: list[SupportedFact] = Field(default_factory=list)
    psh: list[SupportedFact] = Field(default_factory=list)
    social_history: list[SupportedFact] = Field(default_factory=list)
    allergies: list[SupportedFact] = Field(default_factory=list)
    medications: list[SupportedFact] = Field(default_factory=list)
    vitals: list[SupportedFact] = Field(default_factory=list)
    abnormal_labs: list[SupportedFact] = Field(default_factory=list)
    diagnoses: list[SupportedFact] = Field(default_factory=list)
    assessment: list[SupportedFact] = Field(default_factory=list)
    treatment: list[SupportedFact] = Field(default_factory=list)
    plan: list[SupportedFact] = Field(default_factory=list)
    follow_up: list[SupportedFact] = Field(default_factory=list)
    positive_ros: list[SupportedFact] = Field(default_factory=list)
    positive_physical_exam: list[SupportedFact] = Field(default_factory=list)
    residual_supported_facts: list[SupportedFact] = Field(default_factory=list)
    debug: dict[str, Any] = Field(default_factory=dict)

    def populated_fields(self) -> list[str]:
        return [field_name for field_name in TRUTH_FIELD_NAMES if getattr(self, field_name, [])]

    def field_values(self, field_name: str) -> list[str]:
        return [item.value for item in getattr(self, field_name, [])]


class ValidationCheck(BaseModel):
    field_name: str
    label: str
    required: bool = False
    populated: bool = False
    source_detected: bool = False
    missing_values: list[str] = Field(default_factory=list)
    message: str = ""


class NoteValidationLayer(BaseModel):
    note_id: str
    passed: bool = False
    requested_fields: list[str] = Field(default_factory=list)
    populated_fields: list[str] = Field(default_factory=list)
    missing_required_fields: list[str] = Field(default_factory=list)
    checks: list[ValidationCheck] = Field(default_factory=list)
    debug: dict[str, Any] = Field(default_factory=dict)


class ValidationLayer(BaseModel):
    passed: bool = False
    notes: list[NoteValidationLayer] = Field(default_factory=list)
    debug: dict[str, Any] = Field(default_factory=dict)


class PresentationItem(BaseModel):
    item_id: str
    text: str
    field_name: str | None = None
    note_id: str | None = None
    fact_ids: list[str] = Field(default_factory=list)
    rendered_by_model: bool = False
    evidence: list[EvidenceHit] = Field(default_factory=list)
    candidate_filenames: list[str] = Field(default_factory=list)
    pdf_ids: list[str] = Field(default_factory=list)
    pages: list[int] = Field(default_factory=list)


class PresentationSection(BaseModel):
    section_id: str
    title: str
    note_id: str | None = None
    items: list[PresentationItem] = Field(default_factory=list)


class PresentationLayer(BaseModel):
    title: str
    narrative: str = ""
    sections: list[PresentationSection] = Field(default_factory=list)
    debug: dict[str, Any] = Field(default_factory=dict)


class CitationBox(BaseModel):
    text: str = ""
    block_index: int | None = None
    paragraph_index: int | None = None
    page_paragraph_index: int | None = None
    bbox: dict[str, int] = Field(default_factory=dict)


class CitationIndexEntry(BaseModel):
    id: str
    number: int
    chunk_id: str | None = None
    label: str
    pdf_id: str
    page: int
    snippet: str
    anchor: str | None = None
    page_key: str | None = None
    source_filename: str | None = None
    page_source_filename: str | None = None
    page_image_path: str | None = None
    source_pdf_path: str | None = None
    page_width: int | None = None
    page_height: int | None = None
    block_start: int | None = None
    block_end: int | None = None
    paragraph_start: int | None = None
    paragraph_end: int | None = None
    page_paragraph_start: int | None = None
    page_paragraph_end: int | None = None
    bbox: dict[str, int] = Field(default_factory=dict)
    boxes: list[CitationBox] = Field(default_factory=list)
    degraded: bool = False
    degraded_reason: str | None = None


class CitationInstance(CitationIndexEntry):
    sentence_id: str
    sentence_text: str
    primary_evidence_id: str | None = None
    secondary_evidence_ids: list[str] = Field(default_factory=list)


class CitationSentenceItem(BaseModel):
    item_id: str
    text: str
    citation_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    supported: bool | None = None
    degraded: bool = False
    degraded_reason: str | None = None
    pdf_id: str | None = None
    page: int | None = None
    debug: dict[str, Any] = Field(default_factory=dict)


class CitationSection(BaseModel):
    section_id: str
    title: str
    kind: str
    debug_only: bool = False
    pdf_id: str | None = None
    page: int | None = None
    items: list[CitationSentenceItem] = Field(default_factory=list)


class CitationSourcePage(BaseModel):
    page_key: str
    pdf_id: str
    page: int
    source_filename: str
    page_source_filename: str | None = None
    image_path: str | None = None
    html_path: str | None = None
    source_pdf_path: str | None = None
    width: int | None = None
    height: int | None = None
    paragraph_count: int | None = None


class ResolvedCitations(BaseModel):
    version: str = "citation_v2"
    generated_at: datetime = Field(default_factory=utc_now)
    citation_index: list[CitationIndexEntry] = Field(default_factory=list)
    citation_instances: list[CitationInstance] = Field(default_factory=list)
    sections: list[CitationSection] = Field(default_factory=list)
    source_pages: list[CitationSourcePage] = Field(default_factory=list)
    debug: dict[str, Any] = Field(default_factory=dict)


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
    truth_layer: list[TruthLayerNote] = Field(default_factory=list)
    validation_layer: ValidationLayer | None = None
    presentation_plan: PresentationLayer | None = None
    presentation_draft: PresentationLayer | None = None
    presentation_layer: PresentationLayer | None = None
    citation_index: list[CitationIndexEntry] = Field(default_factory=list)
    citation_instances: list[CitationInstance] = Field(default_factory=list)
    resolved_citations: ResolvedCitations | None = None
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
    citation_index_path: str | None = None
    citation_instances_path: str | None = None
    resolved_citations_path: str | None = None
    source_pdf_copy_path: str | None = None
    summary: ScopedSummaryResult | None = None
    summary_error: str | None = None
    ingestion_results: list[IngestedDocumentResult] = Field(default_factory=list)
    knowledge_filter: KnowledgeFilterResult | None = None
    chunk_stats: ChunkStats | None = None
    opensearch_diagnostics: OpenSearchIndexDiagnostics | None = None
