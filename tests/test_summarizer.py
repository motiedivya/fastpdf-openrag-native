from __future__ import annotations

import json
import re
from pathlib import Path

from fastpdf_openrag_native.fastpdf_loader import materialize_summary_payload
from fastpdf_openrag_native.models import EvidenceHit, SummaryScope
from fastpdf_openrag_native.settings import AppSettings
from fastpdf_openrag_native.summarizer import load_manifest, summarize_scope


class FakeGateway:
    async def chat_on_sources(
        self,
        *,
        message: str,
        data_sources: list[str],
        limit: int = 6,
        score_threshold: float = 0,
    ):
        if "single page" in message:
            source_name = data_sources[0]
            match = re.search(r"p(\d+)", source_name)
            page_number = int(match.group(1)) if match else 0
            return (
                json.dumps(
                    {
                        "summary": f"Grounded summary for page {page_number}.",
                        "key_facts": [f"fact-page-{page_number}"],
                    }
                ),
                [EvidenceHit(filename=source_name, text=f"text for page {page_number}", score=0.9)],
            )

        return (
            json.dumps(
                {
                    "title": "Operative Sequence",
                    "summary": "Supported procedure detail. Unsupported leap.",
                    "chronology": ["Page 2 procedure", "Page 3 procedure"],
                }
            ),
            [EvidenceHit(filename=data_sources[0], text="support", score=0.8)],
        )

    async def search_on_sources(
        self,
        *,
        query: str,
        data_sources: list[str],
        limit: int | None = None,
        score_threshold: float | None = None,
    ):
        if query == "Supported procedure detail.":
            return [EvidenceHit(filename=data_sources[0], text="support", score=0.7)]
        return []


class FakeRetryGateway:
    def __init__(self) -> None:
        self.calls = 0

    async def chat_on_sources(
        self,
        *,
        message: str,
        data_sources: list[str],
        limit: int = 6,
        score_threshold: float = 0,
    ):
        self.calls += 1
        if self.calls == 1:
            return ("Please provide the URL you want summarized.", [])
        if self.calls == 2:
            source_name = data_sources[0]
            return (
                json.dumps(
                    {
                        "summary": "Grounded retry page summary.",
                        "key_facts": ["retry-fact"],
                    }
                ),
                [EvidenceHit(filename=source_name, text="page support", score=0.9)],
            )
        if self.calls == 3:
            return ("No relevant supporting sources were found for that request.", [])
        return (
            json.dumps(
                {
                    "title": "Retry Summary",
                    "summary": "Supported summary. Unsupported leap.",
                    "chronology": ["Chronology item"],
                }
            ),
            [EvidenceHit(filename=data_sources[0], text="reduce support", score=0.8)],
        )

    async def search_on_sources(
        self,
        *,
        query: str,
        data_sources: list[str],
        limit: int | None = None,
        score_threshold: float | None = None,
    ):
        if query.startswith("page summary ") or query.startswith("overall chronology summary "):
            return [EvidenceHit(filename=data_sources[0], text="preflight support", score=0.75)]
        if query == "Supported summary.":
            return [EvidenceHit(filename=data_sources[0], text="verified support", score=0.7)]
        return []


class CaptureRerankGateway:
    def __init__(self, preferred_filename: str) -> None:
        self.preferred_filename = preferred_filename
        self.chat_calls: list[list[str]] = []

    async def chat_on_sources(
        self,
        *,
        message: str,
        data_sources: list[str],
        limit: int = 6,
        score_threshold: float = 0,
    ):
        self.chat_calls.append(list(data_sources))
        if "Current page:" in message:
            return (
                json.dumps(
                    {
                        "summary": "Grounded page summary.",
                        "key_facts": ["page-fact"],
                    }
                ),
                [EvidenceHit(filename=data_sources[0], text="page support", score=0.9)],
            )
        return (
            json.dumps(
                {
                    "title": "Reranked Summary",
                    "summary": "Supported summary. Unsupported leap.",
                    "chronology": ["Chronology item"],
                }
            ),
            [EvidenceHit(filename=data_sources[0], text="reduce support", score=0.8)],
        )

    async def search_on_sources(
        self,
        *,
        query: str,
        data_sources: list[str],
        limit: int | None = None,
        score_threshold: float | None = None,
    ):
        if query.startswith("page summary "):
            hits: list[EvidenceHit] = []
            for index, filename in enumerate(data_sources):
                if filename == self.preferred_filename:
                    hits.append(
                        EvidenceHit(
                            filename=filename,
                            text="rare diagnosis with focused supporting evidence",
                            score=0.2,
                        )
                    )
                    continue
                hits.append(
                    EvidenceHit(
                        filename=filename,
                        text=f"generic context block {index}",
                        score=1.0 - (index * 0.1),
                    )
                )
            return hits
        if query.startswith("overall chronology summary "):
            return [EvidenceHit(filename=data_sources[0], text="reduce support", score=0.8)]
        if query == "Supported summary.":
            return [EvidenceHit(filename=data_sources[0], text="verified support", score=0.7)]
        return []


class CaptureBackendRerankGateway:
    def __init__(self) -> None:
        self.chat_calls: list[list[str]] = []

    async def chat_on_sources(
        self,
        *,
        message: str,
        data_sources: list[str],
        limit: int = 6,
        score_threshold: float = 0,
    ):
        self.chat_calls.append(list(data_sources))
        if "Current page:" in message:
            return (
                json.dumps(
                    {
                        "summary": "Grounded page summary.",
                        "key_facts": ["page-fact"],
                    }
                ),
                [EvidenceHit(filename=data_sources[0], text="page support", score=0.9)],
            )
        return (
            json.dumps(
                {
                    "title": "Backend Reranked Summary",
                    "summary": "Supported summary. Unsupported leap.",
                    "chronology": ["Chronology item"],
                }
            ),
            [EvidenceHit(filename=data_sources[0], text="reduce support", score=0.8)],
        )

    async def search_on_sources(
        self,
        *,
        query: str,
        data_sources: list[str],
        limit: int | None = None,
        score_threshold: float | None = None,
    ):
        if query.startswith("page summary ") or query.startswith("overall chronology summary "):
            return [EvidenceHit(filename=name, text=f"candidate for {name}", score=0.5) for name in data_sources]
        if query == "Supported summary.":
            return [EvidenceHit(filename=data_sources[0], text="verified support", score=0.7)]
        return []


class VerificationFallbackGateway:
    def __init__(self, relevant_filename: str) -> None:
        self.relevant_filename = relevant_filename

    async def chat_on_sources(
        self,
        *,
        message: str,
        data_sources: list[str],
        limit: int = 6,
        score_threshold: float = 0,
    ):
        if "Current page:" in message:
            return (
                json.dumps(
                    {
                        "summary": "Grounded page summary.",
                        "key_facts": ["page-fact"],
                    }
                ),
                [EvidenceHit(filename=data_sources[0], text="page support", score=0.9)],
            )
        return (
            json.dumps(
                {
                    "title": "Verification Fallback Summary",
                    "summary": "Supported summary.",
                    "chronology": ["Chronology item"],
                }
            ),
            [EvidenceHit(filename=data_sources[0], text="reduce support", score=0.8)],
        )

    async def search_on_sources(
        self,
        *,
        query: str,
        data_sources: list[str],
        limit: int | None = None,
        score_threshold: float | None = None,
    ):
        if query.startswith("page summary ") or query.startswith("overall chronology summary "):
            return [EvidenceHit(filename=name, text=f"candidate for {name}", score=0.5) for name in data_sources]
        if query == "Supported summary.":
            irrelevant = next(name for name in data_sources if name != self.relevant_filename)
            return [
                EvidenceHit(filename=irrelevant, text="demographic header noise only", score=0.95),
                EvidenceHit(
                    filename=self.relevant_filename,
                    text="supported summary with grounded operative detail",
                    score=0.35,
                ),
            ]
        return []


def test_summarize_scope_filters_unsupported_sentences(tmp_path: Path) -> None:
    payload = {
        "pdfs": [
            {
                "pdf_id": "pdf_1",
                "pages": [
                    {"page": 2, "pdf2html_text": "Procedure one"},
                    {"page": 3, "pdf2html_text": "Procedure two"},
                ],
            }
        ]
    }
    materialize_summary_payload(
        run_id="sample-run",
        summary_payload=payload,
        source_kind="summary_payload",
        output_dir=tmp_path,
    )
    manifest = load_manifest(tmp_path / "manifest.json")
    scope = SummaryScope.model_validate(
        {
            "scope_id": "operative-sequence",
            "title": "Operative Sequence",
            "objective": "Keep procedures separate.",
            "page_refs": [{"pdf_id": "pdf_1", "page": 2}, {"pdf_id": "pdf_1", "page": 3}],
        }
    )

    result = __import__("asyncio").run(
        summarize_scope(FakeGateway(), manifest=manifest, scope=scope)
    )

    assert result.draft_title == "Operative Sequence"
    assert result.supported_summary == "Supported procedure detail."
    assert result.unsupported_sentences == ["Unsupported leap."]
    assert len(result.page_summaries) == 2


def test_summarize_scope_retries_when_chat_skips_retrieval(tmp_path: Path) -> None:
    payload = {
        "pdfs": [
            {
                "pdf_id": "pdf_1",
                "pages": [
                    {"page": 1, "pdf2html_text": "Retry target page"},
                ],
            }
        ]
    }
    materialize_summary_payload(
        run_id="retry-run",
        summary_payload=payload,
        source_kind="summary_payload",
        output_dir=tmp_path,
    )
    manifest = load_manifest(tmp_path / "manifest.json")
    scope = SummaryScope.model_validate(
        {
            "scope_id": "retry-scope",
            "title": "Retry Scope",
            "objective": "Summarize with retrieval.",
            "page_refs": [{"pdf_id": "pdf_1", "page": 1}],
        }
    )

    result = __import__("asyncio").run(
        summarize_scope(FakeRetryGateway(), manifest=manifest, scope=scope)
    )

    assert result.page_summaries[0].summary == "Grounded retry page summary."
    assert result.debug["page_requests"][0]["retrieval_retry_used"] is True
    assert result.debug["reduce_retry_used"] is True
    assert result.supported_summary == "Supported summary."
    assert result.unsupported_sentences == ["Unsupported leap."]


def test_summarize_scope_uses_reranked_chunk_sources(tmp_path: Path) -> None:
    payload = {
        "pdfs": [
            {
                "pdf_id": "pdf_1",
                "pages": [
                    {
                        "page": 1,
                        "ocr_text": (
                            "HISTORY:\n\n"
                            "Patient presents with chronic back pain and muscle spasm.\n\n"
                            "ASSESSMENT:\n\n"
                            "Rare diagnosis with focused supporting evidence.\n\n"
                            "PLAN:\n\n"
                            "Continue therapy and monitor response."
                        ),
                    }
                ],
            }
        ]
    }
    settings = AppSettings(
        structure_chunk_target_chars=90,
        structure_chunk_overlap_blocks=0,
        backend_rerank_enabled=False,
        retrieval_rerank_enabled=True,
        retrieval_rerank_top_k=1,
        retrieval_rerank_candidate_limit=8,
    )
    manifest = materialize_summary_payload(
        run_id="rerank-run",
        summary_payload=payload,
        source_kind="summary_payload",
        output_dir=tmp_path,
        settings=settings,
    )
    scope = SummaryScope.model_validate(
        {
            "scope_id": "rerank-scope",
            "title": "Rerank Scope",
            "objective": "rare diagnosis",
            "page_refs": [{"pdf_id": "pdf_1", "page": 1}],
        }
    )
    preferred_filename = next(
        document.source_filename
        for document in manifest.retrieval_documents
        if "rare diagnosis" in (tmp_path / document.relative_path).read_text(encoding="utf-8").lower()
    )
    gateway = CaptureRerankGateway(preferred_filename)

    result = __import__("asyncio").run(
        summarize_scope(gateway, manifest=manifest, scope=scope, settings=settings)
    )

    assert gateway.chat_calls[0] == [preferred_filename]
    assert result.debug["page_requests"][0]["selected_source_filenames"] == [preferred_filename]
    assert result.debug["reranking_enabled"] is True


def test_summarize_scope_uses_backend_reranked_chat_flow_without_local_source_pruning(tmp_path: Path) -> None:
    payload = {
        "pdfs": [
            {
                "pdf_id": "pdf_1",
                "pages": [
                    {
                        "page": 1,
                        "ocr_text": (
                            "HISTORY:\n\n"
                            "Patient presents with chronic back pain and muscle spasm.\n\n"
                            "ASSESSMENT:\n\n"
                            "Rare diagnosis with focused supporting evidence.\n\n"
                            "PLAN:\n\n"
                            "Continue therapy and monitor response."
                        ),
                    }
                ],
            }
        ]
    }
    settings = AppSettings(
        structure_chunk_target_chars=90,
        structure_chunk_overlap_blocks=0,
        backend_rerank_enabled=True,
        backend_search_rerank_enabled=True,
        backend_rerank_candidate_limit=6,
        retrieval_rerank_enabled=False,
    )
    manifest = materialize_summary_payload(
        run_id="backend-rerank-run",
        summary_payload=payload,
        source_kind="summary_payload",
        output_dir=tmp_path,
        settings=settings,
    )
    scope = SummaryScope.model_validate(
        {
            "scope_id": "backend-rerank-scope",
            "title": "Backend Rerank Scope",
            "objective": "rare diagnosis",
            "page_refs": [{"pdf_id": "pdf_1", "page": 1}],
        }
    )
    gateway = CaptureBackendRerankGateway()

    result = __import__("asyncio").run(
        summarize_scope(gateway, manifest=manifest, scope=scope, settings=settings)
    )

    expected_sources = manifest.page_documents[0].retrieval_sources()
    assert len(expected_sources) > 1
    assert gateway.chat_calls[0] == expected_sources
    assert result.debug["page_requests"][0]["selected_source_filenames"] == expected_sources
    assert result.debug["reranking_enabled"] is True
    assert result.debug["reranker_location"] == "langflow_agent_tool"
    assert result.debug["reranker_type"] == "cross_encoder"
    assert result.debug["verification_reranker_location"] == "openrag_backend_search_api"


def test_summarize_scope_uses_application_fallback_for_verification_when_backend_search_rerank_is_disabled(
    tmp_path: Path,
) -> None:
    payload = {
        "pdfs": [
            {
                "pdf_id": "pdf_1",
                "pages": [
                    {
                        "page": 1,
                        "ocr_text": (
                            "HISTORY:\n\n"
                            "Demographic header and generic context.\n\n"
                            "PROCEDURE:\n\n"
                            "Supported summary with grounded operative detail.\n\n"
                            "PLAN:\n\n"
                            "Continue therapy and monitor response."
                        ),
                    }
                ],
            }
        ]
    }
    settings = AppSettings(
        structure_chunk_target_chars=90,
        structure_chunk_overlap_blocks=0,
        backend_rerank_enabled=True,
        backend_search_rerank_enabled=False,
        backend_rerank_candidate_limit=6,
        retrieval_rerank_enabled=False,
    )
    manifest = materialize_summary_payload(
        run_id="verification-fallback-run",
        summary_payload=payload,
        source_kind="summary_payload",
        output_dir=tmp_path,
        settings=settings,
    )
    relevant_filename = next(
        document.source_filename
        for document in manifest.retrieval_documents
        if "grounded operative detail" in (tmp_path / document.relative_path).read_text(encoding="utf-8").lower()
    )
    scope = SummaryScope.model_validate(
        {
            "scope_id": "verification-fallback-scope",
            "title": "Verification Fallback Scope",
            "objective": "supported summary",
            "page_refs": [{"pdf_id": "pdf_1", "page": 1}],
        }
    )
    gateway = VerificationFallbackGateway(relevant_filename)

    result = __import__("asyncio").run(
        summarize_scope(gateway, manifest=manifest, scope=scope, settings=settings)
    )

    assert result.verified_sentences[0].evidence[0].filename == relevant_filename
    assert result.debug["verification_reranking_enabled"] is True
    assert result.debug["verification_reranker_location"] == "application_verification_fallback"
    assert result.debug["verification_reranker_type"] == "deterministic_hybrid_v1"
