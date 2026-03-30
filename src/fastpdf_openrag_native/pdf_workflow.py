from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import (
    ChunkStats,
    OpenSearchIndexDiagnostics,
    PdfPipelineResult,
    SummaryScope,
)
from .citations import ensure_summary_citations
from .ocr_extract import extract_pdf_to_html, slugify_filename
from .openrag import OpenRAGGateway
from .opensearch import OpenSearchInspector
from .settings import AppSettings, get_settings
from .summarizer import summarize_scope
from .trace import TraceRecorder


def build_pipeline_id(pdf_path: Path) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = slugify_filename(pdf_path.stem.lower())
    return f"{stem}-{timestamp}"


def _all_pages_scope(manifest, objective: str | None = None) -> SummaryScope:
    return SummaryScope.model_validate(
        {
            "scope_id": "all-pages",
            "title": "All Pages Summary",
            "objective": objective
            or (
                "Summarize each page independently, then build a grounded overall summary for the full PDF. "
                "Preserve chronology and do not merge unrelated procedures."
            ),
            "page_refs": [
                {"pdf_id": page.pdf_id, "page": page.page}
                for page in manifest.page_documents
            ],
        }
    )


def _build_chunk_stats(chunk_counts: dict[str, int]) -> ChunkStats:
    ordered = dict(sorted(chunk_counts.items()))
    return ChunkStats(
        total_chunks=sum(ordered.values()),
        documents_with_chunks=sum(1 for count in ordered.values() if count > 0),
        chunks_per_document=ordered,
        single_chunk_documents=[name for name, count in ordered.items() if count == 1],
        multi_chunk_documents=[name for name, count in ordered.items() if count > 1],
        empty_documents=[name for name, count in ordered.items() if count == 0],
    )


def _build_retrieval_debug_payload(
    *,
    diagnostics: OpenSearchIndexDiagnostics | None,
    chunk_stats: ChunkStats,
    source_pages: int,
    retrieval_documents: int,
) -> dict[str, Any]:
    payload = {
        "database_backend": diagnostics.database_backend if diagnostics else None,
        "cluster_name": diagnostics.cluster_name if diagnostics else None,
        "documents_index_name": diagnostics.documents_index_name if diagnostics else None,
        "knowledge_filter_index_name": (
            diagnostics.knowledge_filter_index_name if diagnostics else None
        ),
        "document_count_in_index": diagnostics.document_count if diagnostics else None,
        "retrieval_mode": diagnostics.retrieval_mode if diagnostics else None,
        "application_retrieval_mode": (
            diagnostics.application_retrieval_mode if diagnostics else None
        ),
        "semantic_weight": diagnostics.semantic_weight if diagnostics else None,
        "keyword_weight": diagnostics.keyword_weight if diagnostics else None,
        "reranking_enabled": diagnostics.reranking_enabled if diagnostics else None,
        "reranker_location": diagnostics.reranker_location if diagnostics else None,
        "reranker_type": diagnostics.reranker_type if diagnostics else None,
        "reranker_model": diagnostics.reranker_model if diagnostics else None,
        "chunking_enabled": diagnostics.chunking_enabled if diagnostics else None,
        "structure_chunking_strategy": (
            diagnostics.structure_chunking_strategy if diagnostics else None
        ),
        "chunk_size": diagnostics.chunk_size if diagnostics else None,
        "chunk_overlap": diagnostics.chunk_overlap if diagnostics else None,
        "prechunk_target_chars": diagnostics.prechunk_target_chars if diagnostics else None,
        "prechunk_overlap_blocks": diagnostics.prechunk_overlap_blocks if diagnostics else None,
        "embedding_provider": diagnostics.embedding_provider if diagnostics else None,
        "embedding_model": diagnostics.embedding_model if diagnostics else None,
        "vector_fields": diagnostics.vector_fields if diagnostics else [],
        "source_pages_in_run": source_pages,
        "retrieval_documents_in_run": retrieval_documents,
        "chunk_stats": chunk_stats.model_dump(mode="json"),
        "notes": [],
    }

    notes = payload["notes"]
    if retrieval_documents > source_pages:
        notes.append(
            "Structure-aware pre-chunking split pages into dedicated retrieval documents before OpenRAG ingest."
        )
    if chunk_stats.documents_with_chunks == retrieval_documents and chunk_stats.total_chunks == retrieval_documents:
        notes.append(
            "Each retrieval document stayed intact as a single stored chunk in OpenSearch for this run."
        )
    if diagnostics and diagnostics.reranking_enabled is False:
        notes.append(
            "Results are ranked by OpenSearch hybrid _score; no separate reranker is enabled in the current retrieval path."
        )
    if diagnostics and diagnostics.reranker_location == "application":
        notes.append(
            "OpenRAG hybrid retrieval is followed by an application-side reranking pass before summary generation and verification."
        )
    if diagnostics and diagnostics.reranker_location == "langflow_agent_tool":
        notes.append(
            "OpenRAG chat now depends on a Langflow-backed retrieval tool that reranks hybrid OpenSearch hits before they reach generation."
        )
    return payload


async def _safe_diagnostics(
    inspector: OpenSearchInspector,
    trace: TraceRecorder,
) -> OpenSearchIndexDiagnostics | None:
    try:
        diagnostics = await inspector.diagnostics()
    except Exception as exc:
        trace.record(
            stage="opensearch",
            service="opensearch",
            action="diagnostics",
            status="error",
            response={"error": str(exc)},
        )
        return None
    trace.record(
        stage="opensearch",
        service="opensearch",
        action="diagnostics",
        response={
            "database_backend": diagnostics.database_backend,
            "cluster_name": diagnostics.cluster_name,
            "documents_index_name": diagnostics.documents_index_name,
            "knowledge_filter_index_name": diagnostics.knowledge_filter_index_name,
            "document_count": diagnostics.document_count,
            "retrieval_mode": diagnostics.retrieval_mode,
            "application_retrieval_mode": diagnostics.application_retrieval_mode,
            "semantic_weight": diagnostics.semantic_weight,
            "keyword_weight": diagnostics.keyword_weight,
            "reranking_enabled": diagnostics.reranking_enabled,
            "reranker_location": diagnostics.reranker_location,
            "reranker_type": diagnostics.reranker_type,
            "reranker_model": diagnostics.reranker_model,
            "structure_chunking_strategy": diagnostics.structure_chunking_strategy,
            "chunk_size": diagnostics.chunk_size,
            "chunk_overlap": diagnostics.chunk_overlap,
            "prechunk_target_chars": diagnostics.prechunk_target_chars,
            "prechunk_overlap_blocks": diagnostics.prechunk_overlap_blocks,
            "embedding_provider": diagnostics.embedding_provider,
            "embedding_model": diagnostics.embedding_model,
            "vector_fields": diagnostics.vector_fields,
            "cluster_health": diagnostics.cluster_health,
            "allocation_explain": diagnostics.allocation_explain,
        },
        notes=[
            "index_search_error present means the documents index was not queryable during this run"
            if diagnostics.index_search_error
            else "documents index query succeeded during diagnostics"
        ],
    )
    return diagnostics


async def run_pdf_pipeline(
    *,
    pdf_path: Path,
    credentials_path: Path | None = None,
    settings: AppSettings | None = None,
    question: str | None = None,
    max_pages: int | None = None,
    apply_recommended_settings: bool = True,
) -> PdfPipelineResult:
    effective_settings = settings or get_settings()
    pipeline_id = build_pipeline_id(pdf_path)
    extraction_dir = effective_settings.extraction_root / pipeline_id
    trace_dir = effective_settings.trace_root / pipeline_id
    summary_dir = effective_settings.output_root / pipeline_id
    summary_dir.mkdir(parents=True, exist_ok=True)
    trace = TraceRecorder(trace_dir)

    trace.record(
        stage="pipeline",
        service="fastpdf-openrag-native",
        action="start_pdf_pipeline",
        request={
            "pdf_path": pdf_path.as_posix(),
            "credentials_path": credentials_path.as_posix() if credentials_path else None,
            "question": question,
            "max_pages": max_pages,
        },
        response={"pipeline_id": pipeline_id},
        notes=(
            [
                f"Pipeline limited to max_pages={max_pages}. This run will not cover the full PDF."
            ]
            if max_pages
            else []
        ),
    )

    manifest = await asyncio.to_thread(
        extract_pdf_to_html,
        pdf_path,
        output_dir=extraction_dir,
        trace=trace,
        settings=effective_settings,
        credentials_path=credentials_path,
        max_pages=max_pages,
        run_id=pipeline_id,
    )
    manifest_path = extraction_dir / "manifest.json"

    gateway = OpenRAGGateway(effective_settings)
    inspector = OpenSearchInspector(effective_settings)
    ingest_documents = manifest.ingest_documents()
    diagnostics_before = await _safe_diagnostics(inspector, trace)
    settings_payload: dict[str, Any] | None = None

    if apply_recommended_settings:
        settings_payload = await gateway.apply_recommended_settings()
        trace.record(
            stage="openrag",
            service="openrag",
            action="apply_recommended_settings",
            response=settings_payload,
        )

    ingestion_results = await gateway.ingest_manifest(
        manifest,
        manifest_dir=extraction_dir,
        replace_existing=True,
    )
    for row in ingestion_results:
        task_status: dict[str, Any] | None = None
        if row.task_id:
            try:
                task_status = await gateway.task_status(row.task_id)
            except Exception as exc:
                task_status = {"error": str(exc)}
        trace.record(
            stage="openrag",
            service="openrag",
            action="ingest_document",
            status="ok" if "completed" in row.status else "warning",
            request={"filename": row.filename},
            response={
                "status": row.status,
                "task_id": row.task_id,
                "successful_files": row.successful_files,
                "failed_files": row.failed_files,
                "delete_error": row.delete_error,
                "task_status": task_status,
            },
        )

    scope = _all_pages_scope(manifest, objective=question)
    knowledge_filter = None
    try:
        knowledge_filter = await gateway.upsert_scope_filter(
            manifest=manifest,
            scope=scope,
            data_sources=[document.source_filename for document in ingest_documents],
        )
        trace.record(
            stage="openrag",
            service="openrag",
            action="upsert_knowledge_filter",
            response=knowledge_filter.model_dump(),
        )
    except Exception as exc:
        trace.record(
            stage="openrag",
            service="openrag",
            action="upsert_knowledge_filter",
            status="error",
            response={"error": str(exc)},
        )

    summary = None
    summary_path = None
    summary_error = None
    citation_index_path = None
    resolved_citations_path = None
    source_pdf_copy_path = None
    try:
        summary = await summarize_scope(
            gateway,
            manifest=manifest,
            scope=scope,
            settings=effective_settings,
        )
        summary_path = summary_dir / "all-pages.summary.json"
        try:
            (
                summary,
                citation_index_path,
                resolved_citations_path,
                source_pdf_copy_path,
            ) = ensure_summary_citations(
                summary_path=summary_path,
                manifest_path=manifest_path,
                source_pdf=pdf_path,
                summary=summary,
                manifest=manifest,
            )
            trace.record(
                stage="citations",
                service="fastpdf-openrag-native",
                action="build_summary_citations",
                response={
                    "citation_index_path": citation_index_path.as_posix(),
                    "resolved_citations_path": resolved_citations_path.as_posix(),
                    "citation_count": len(summary.citation_index),
                    "section_count": len(summary.resolved_citations.sections) if summary.resolved_citations else 0,
                    "source_pdf_copy_path": source_pdf_copy_path.as_posix() if source_pdf_copy_path else None,
                },
                output_files=[
                    citation_index_path.as_posix(),
                    resolved_citations_path.as_posix(),
                    *([source_pdf_copy_path.as_posix()] if source_pdf_copy_path else []),
                ],
            )
        except Exception as exc:
            summary_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
            trace.record(
                stage="citations",
                service="fastpdf-openrag-native",
                action="build_summary_citations",
                status="error",
                response={"error": str(exc)},
                notes=[
                    "Summary succeeded but citation artifacts could not be generated. The summary route can retry citation backfill later."
                ],
            )
        trace.record(
            stage="summary",
            service="openrag",
            action="summarize_scope",
            response={
                "summary_path": summary_path.as_posix(),
                "citation_index_path": citation_index_path.as_posix() if citation_index_path else None,
                "resolved_citations_path": resolved_citations_path.as_posix() if resolved_citations_path else None,
                "page_summary_count": len(summary.page_summaries),
                "unsupported_sentences": summary.unsupported_sentences,
            },
            output_files=[
                summary_path.as_posix(),
                *([citation_index_path.as_posix()] if citation_index_path else []),
                *([resolved_citations_path.as_posix()] if resolved_citations_path else []),
            ],
        )
    except Exception as exc:
        summary_error = str(exc)
        trace.record(
            stage="summary",
            service="openrag",
            action="summarize_scope",
            status="error",
            response={"error": str(exc)},
            notes=[
                "Summary failed after ingestion. Inspect OpenRAG chat/search health and the OpenSearch diagnostics captured in this run."
            ],
        )

    chunk_dump_dir = trace_dir / "chunks"
    chunk_dump_dir.mkdir(parents=True, exist_ok=True)
    chunk_counts: dict[str, int] = {}
    for document in ingest_documents:
        try:
            chunks = await inspector.list_chunks_for_filename(document.source_filename)
            chunk_counts[document.source_filename] = len(chunks)
            chunk_path = chunk_dump_dir / f"{document.source_filename}.chunks.json"
            chunk_path.write_text(
                json.dumps([row.model_dump() for row in chunks], indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            trace.record(
                stage="opensearch",
                service="opensearch",
                action="dump_chunks_for_document",
                request={"filename": document.source_filename},
                response={"chunk_count": len(chunks), "chunk_dump_path": chunk_path.as_posix()},
                output_files=[chunk_path.as_posix()],
            )
        except Exception as exc:
            trace.record(
                stage="opensearch",
                service="opensearch",
                action="dump_chunks_for_document",
                status="error",
                request={"filename": document.source_filename},
                response={"error": str(exc)},
                notes=[
                    "OpenSearch chunk inspection failed. Check cluster health and documents index allocation."
                ],
            )
            break

    chunk_stats = _build_chunk_stats(chunk_counts)
    trace.record(
        stage="opensearch",
        service="opensearch",
        action="summarize_chunk_inventory",
        response=chunk_stats.model_dump(mode="json"),
    )

    diagnostics_after = await _safe_diagnostics(inspector, trace)
    diagnostics = diagnostics_after or diagnostics_before
    if diagnostics and settings_payload:
        knowledge_settings = settings_payload.get("knowledge") or {}
        diagnostics.chunk_size = knowledge_settings.get("chunk_size", diagnostics.chunk_size)
        diagnostics.chunk_overlap = knowledge_settings.get("chunk_overlap", diagnostics.chunk_overlap)
        diagnostics.embedding_provider = knowledge_settings.get(
            "embedding_provider",
            diagnostics.embedding_provider,
        )
        diagnostics.embedding_model = knowledge_settings.get(
            "embedding_model",
            diagnostics.embedding_model,
        )
    if diagnostics:
        diagnostics.application_retrieval_mode = (
            "hybrid_backend_reranked"
            if effective_settings.backend_rerank_enabled
            else (
                "hybrid_application_reranked"
                if effective_settings.retrieval_rerank_enabled
                else diagnostics.retrieval_mode
            )
        )
        diagnostics.reranking_enabled = (
            effective_settings.backend_rerank_enabled or effective_settings.retrieval_rerank_enabled
        )
        diagnostics.reranker_location = (
            "langflow_agent_tool"
            if effective_settings.backend_rerank_enabled
            else ("application" if effective_settings.retrieval_rerank_enabled else None)
        )
        diagnostics.reranker_type = (
            effective_settings.backend_rerank_provider
            if effective_settings.backend_rerank_enabled
            else ("deterministic_hybrid_v1" if effective_settings.retrieval_rerank_enabled else None)
        )
        diagnostics.reranker_model = (
            effective_settings.backend_rerank_model if effective_settings.backend_rerank_enabled else None
        )
        diagnostics.structure_chunking_strategy = "structure_aware_blocks"
        diagnostics.prechunk_target_chars = effective_settings.structure_chunk_target_chars
        diagnostics.prechunk_overlap_blocks = effective_settings.structure_chunk_overlap_blocks

    if summary is not None:
        summary.debug["retrieval_runtime"] = _build_retrieval_debug_payload(
            diagnostics=diagnostics,
            chunk_stats=chunk_stats,
            source_pages=len(manifest.page_documents),
            retrieval_documents=len(ingest_documents),
        )
        if summary_path:
            summary_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")

    result = PdfPipelineResult(
        run_id=pipeline_id,
        source_pdf=pdf_path.as_posix(),
        extraction_dir=extraction_dir.as_posix(),
        manifest_path=manifest_path.as_posix(),
        trace_path=trace.trace_path.as_posix(),
        trace_summary_path=trace.summary_path.as_posix(),
        trace_dir=trace_dir.as_posix(),
        chunk_dump_dir=chunk_dump_dir.as_posix(),
        total_pages=manifest.total_pages,
        materialized_pages=manifest.materialized_pages,
        requested_max_pages=manifest.requested_max_pages,
        is_partial_run=manifest.is_partial_run,
        summary_path=summary_path.as_posix() if summary_path else None,
        citation_index_path=citation_index_path.as_posix() if citation_index_path else None,
        resolved_citations_path=resolved_citations_path.as_posix() if resolved_citations_path else None,
        source_pdf_copy_path=source_pdf_copy_path.as_posix() if source_pdf_copy_path else None,
        summary=summary,
        summary_error=summary_error,
        ingestion_results=ingestion_results,
        knowledge_filter=knowledge_filter,
        chunk_stats=chunk_stats,
        opensearch_diagnostics=diagnostics,
    )
    trace.write_summary(result.model_dump(mode="json"))
    return result


def load_pipeline_summary(path: Path) -> PdfPipelineResult:
    return PdfPipelineResult.model_validate_json(path.read_text(encoding="utf-8"))
