from __future__ import annotations

from types import SimpleNamespace

from fastpdf_openrag_native.models import (
    DEFAULT_MEDICAL_NOTE_OBJECTIVE,
    MaterializedPage,
    MaterializedRetrievalDocument,
    OpenSearchIndexDiagnostics,
)
from fastpdf_openrag_native.opensearch import OpenSearchInspector
from fastpdf_openrag_native.pdf_workflow import (
    _all_pages_scope,
    _build_chunk_audit,
    _build_chunk_stats,
    _build_page_chunk_audit,
    _build_retrieval_debug_payload,
)


def test_build_chunk_stats_tracks_single_multi_and_empty_documents() -> None:
    stats = _build_chunk_stats(
        {
            "alpha.html": 1,
            "beta.html": 3,
            "gamma.html": 0,
        }
    )

    assert stats.total_chunks == 4
    assert stats.documents_with_chunks == 2
    assert stats.single_chunk_documents == ["alpha.html"]
    assert stats.multi_chunk_documents == ["beta.html"]
    assert stats.empty_documents == ["gamma.html"]


def test_build_retrieval_debug_payload_calls_out_single_chunk_runs_and_no_reranker() -> None:
    diagnostics = OpenSearchIndexDiagnostics(
        database_backend="OpenSearch",
        cluster_name="docker-cluster",
        documents_index_name="documents",
        knowledge_filter_index_name="knowledge_filters",
        document_count=12,
        retrieval_mode="hybrid",
        application_retrieval_mode="hybrid",
        semantic_weight=0.7,
        keyword_weight=0.3,
        reranking_enabled=False,
        reranker_location=None,
        chunking_enabled=True,
        structure_chunking_strategy="page_level",
        chunk_size=6000,
        chunk_overlap=0,
        embedding_provider="openai",
        embedding_model="text-embedding-3-large",
        vector_fields=["chunk_embedding_text_embedding_3_large"],
    )
    chunk_stats = _build_chunk_stats(
        {
            "page-1.html": 1,
            "page-2.html": 1,
        }
    )

    payload = _build_retrieval_debug_payload(
        diagnostics=diagnostics,
        chunk_stats=chunk_stats,
        chunk_audit={},
        page_chunk_audit=[],
        source_pages=2,
        retrieval_documents=2,
    )

    assert payload["documents_index_name"] == "documents"
    assert payload["knowledge_filter_index_name"] == "knowledge_filters"
    assert payload["retrieval_mode"] == "hybrid"
    assert payload["reranking_enabled"] is False
    assert len(payload["notes"]) == 2


def test_build_retrieval_debug_payload_calls_out_structure_aware_and_reranked_runs() -> None:
    diagnostics = OpenSearchIndexDiagnostics(
        database_backend="OpenSearch",
        cluster_name="docker-cluster",
        documents_index_name="documents",
        knowledge_filter_index_name="knowledge_filters",
        document_count=12,
        retrieval_mode="hybrid",
        application_retrieval_mode="hybrid_backend_reranked",
        semantic_weight=0.7,
        keyword_weight=0.3,
        reranking_enabled=True,
        reranker_location="langflow_agent_tool",
        reranker_type="cross_encoder",
        reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        chunking_enabled=True,
        structure_chunking_strategy="structure_aware_blocks",
        chunk_size=6000,
        chunk_overlap=0,
        prechunk_target_chars=1400,
        prechunk_overlap_blocks=1,
        embedding_provider="openai",
        embedding_model="text-embedding-3-large",
        vector_fields=["chunk_embedding_text_embedding_3_large"],
    )
    chunk_stats = _build_chunk_stats(
        {
            "page-1__c0001.md": 1,
            "page-1__c0002.md": 1,
            "page-2__c0001.md": 1,
        }
    )

    payload = _build_retrieval_debug_payload(
        diagnostics=diagnostics,
        chunk_stats=chunk_stats,
        chunk_audit={},
        page_chunk_audit=[],
        source_pages=2,
        retrieval_documents=3,
    )

    assert payload["application_retrieval_mode"] == "hybrid_backend_reranked"
    assert payload["reranker_location"] == "langflow_agent_tool"
    assert payload["reranker_type"] == "cross_encoder"
    assert payload["reranker_model"] == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert payload["source_pages_in_run"] == 2
    assert payload["retrieval_documents_in_run"] == 3
    assert len(payload["notes"]) == 3


def test_build_page_chunk_audit_reports_native_text_guardrails() -> None:
    page_audit = _build_page_chunk_audit(
        [
            MaterializedPage(
                run_id="run-1",
                pdf_id="alpha.pdf",
                page=1,
                order_index=0,
                source_filename="alpha__p0001.html",
                relative_path="pages/alpha__p0001.html",
                document_type="text/html",
                text_length=1200,
                text_preview="History of present illness.",
                retrieval_filenames=["alpha__p0001__c0001.md", "alpha__p0001__c0002.md"],
                metadata={
                    "source_text_strategy": "best_text_with_ocr_assets",
                    "chunk_text_strategy": "best_text",
                    "native_text_chars": 1400,
                    "ocr_paragraph_count": 2,
                    "indexed_chunk_count": 2,
                    "native_text_guardrail_triggered": True,
                    "native_text_guardrail_reason": "born_digital_single_header_chunk",
                    "chunk_previews": [
                        "HISTORY OF PRESENTING COMPLAINT: She rates pain as 8/10.",
                        "ASSESSMENT: V89.2 injured in MVA traffic.",
                    ],
                },
            )
        ]
    )

    payload = _build_retrieval_debug_payload(
        diagnostics=None,
        chunk_stats=_build_chunk_stats({"alpha__p0001__c0001.md": 1, "alpha__p0001__c0002.md": 1}),
        chunk_audit={},
        page_chunk_audit=page_audit,
        source_pages=1,
        retrieval_documents=2,
    )

    assert payload["page_chunk_audit"][0]["native_text_chars"] == 1400
    assert payload["page_chunk_audit"][0]["chunk_previews"][0].startswith("HISTORY OF PRESENTING COMPLAINT")
    assert any("Native-text guardrail rebuilt sparse born-digital page chunks" in note for note in payload["notes"])
    assert any("fewer than 3 indexed chunks" in note for note in payload["notes"])


def test_all_pages_scope_defaults_to_strong_medical_objective() -> None:
    manifest = SimpleNamespace(page_documents=[SimpleNamespace(pdf_id="alpha.pdf", page=1)])

    scope = _all_pages_scope(manifest)

    assert scope.objective == DEFAULT_MEDICAL_NOTE_OBJECTIVE


def test_build_chunk_audit_reports_short_chunk_counts_and_samples() -> None:
    audit = _build_chunk_audit(
        [
            MaterializedRetrievalDocument(
                run_id="run-1",
                pdf_id="alpha.pdf",
                page=1,
                order_index=0,
                chunk_index=1,
                source_filename="alpha__c0001.md",
                relative_path="retrieval/alpha__c0001.md",
                text_length=24,
                text_preview="with",
            ),
            MaterializedRetrievalDocument(
                run_id="run-1",
                pdf_id="alpha.pdf",
                page=1,
                order_index=0,
                chunk_index=2,
                source_filename="alpha__c0002.md",
                relative_path="retrieval/alpha__c0002.md",
                text_length=68,
                text_preview="Plan: Physical therapy was recommended.",
                section_title="Plan",
            ),
            MaterializedRetrievalDocument(
                run_id="run-1",
                pdf_id="alpha.pdf",
                page=1,
                order_index=0,
                chunk_index=3,
                source_filename="alpha__c0003.md",
                relative_path="retrieval/alpha__c0003.md",
                text_length=240,
                text_preview="History of present illness documented worsening right knee pain with bending.",
                section_title="History of Present Illness",
            ),
        ]
    )

    assert audit["total_chunks"] == 3
    assert audit["median_chunk_length"] == 68
    assert audit["chunks_under_30_chars"] == 1
    assert audit["chunks_under_50_chars"] == 1
    assert audit["chunks_under_100_chars"] == 2
    assert audit["weak_chunk_samples"][0]["filename"] == "alpha__c0001.md"


def test_parse_cat_indices_doc_count_reads_target_index_row() -> None:
    cat_indices = "\n".join(
        [
            "health status index pri rep docs.count store.size",
            "green open documents 1 0 42 128kb",
            "green open knowledge_filters 1 0 3 32kb",
        ]
    )

    assert OpenSearchInspector._parse_cat_indices_doc_count(cat_indices, "documents") == 42
    assert OpenSearchInspector._parse_cat_indices_doc_count(cat_indices, "missing") is None


def test_extract_vector_fields_finds_knn_vectors() -> None:
    index_mapping = {
        "documents": {
            "mappings": {
                "properties": {
                    "text": {"type": "text"},
                    "chunk_embedding": {"type": "knn_vector"},
                    "nested": {
                        "properties": {
                            "chunk_embedding_text_embedding_3_large": {"type": "knn_vector"},
                        }
                    },
                }
            }
        }
    }

    assert OpenSearchInspector._extract_vector_fields(index_mapping, "documents") == [
        "chunk_embedding",
        "nested.chunk_embedding_text_embedding_3_large",
    ]
