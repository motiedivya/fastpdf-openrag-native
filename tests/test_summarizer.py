from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from types import SimpleNamespace

from fastpdf_openrag_native.fastpdf_loader import materialize_summary_payload
from fastpdf_openrag_native.models import EvidenceHit, SummaryScope
from fastpdf_openrag_native.settings import AppSettings
from fastpdf_openrag_native.summarizer import (
    _document_priority_score,
    _expand_page_selected_sources,
    _filter_page_retrieval_documents,
    load_manifest,
    summarize_scope,
)


class FakeGateway:
    async def chat_on_sources(
        self,
        *,
        message: str,
        data_sources: list[str] | None = None,
        limit: int = 6,
        score_threshold: float = 0,
        llm_model: str | None = None,
        llm_provider: str | None = None,
        disable_retrieval: bool = False,
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
        if query.startswith("Grounded summary for page "):
            return [EvidenceHit(filename=data_sources[0], text="page verification support", score=0.7)]
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
        data_sources: list[str] | None = None,
        limit: int = 6,
        score_threshold: float = 0,
        llm_model: str | None = None,
        llm_provider: str | None = None,
        disable_retrieval: bool = False,
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
        if query == "Grounded retry page summary.":
            return [EvidenceHit(filename=data_sources[0], text="page verification support", score=0.72)]
        if query == "Supported summary.":
            return [EvidenceHit(filename=data_sources[0], text="verified support", score=0.7)]
        return []
class LocalInventoryFallbackGateway:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def chat_on_sources(
        self,
        *,
        message: str,
        data_sources: list[str] | None = None,
        limit: int = 6,
        score_threshold: float = 0,
        llm_model: str | None = None,
        llm_provider: str | None = None,
        disable_retrieval: bool = False,
    ):
        self.calls.append({
            "message": message,
            "data_sources": list(data_sources or []),
            "disable_retrieval": disable_retrieval,
        })
        if "Page-local evidence excerpts:" in message:
            return (
                json.dumps(
                    {
                        "summary": "Patient reported frequent sharp pain rated 8 out of 10 with increased pain on bending and improvement with injections and therapy.",
                        "key_facts": [
                            "Past medical history included hypertension and high cholesterol.",
                            "Social history included tobacco use and alcohol use.",
                        ],
                    }
                ),
                [],
            )
        return ("No relevant supporting sources were found for this page.", [])

    async def search_on_sources(
        self,
        *,
        query: str,
        data_sources: list[str],
        limit: int | None = None,
        score_threshold: float | None = None,
    ):
        return []




class CaptureRerankGateway:
    def __init__(self, preferred_filename: str) -> None:
        self.preferred_filename = preferred_filename
        self.chat_calls: list[list[str]] = []

    async def chat_on_sources(
        self,
        *,
        message: str,
        data_sources: list[str] | None = None,
        limit: int = 6,
        score_threshold: float = 0,
        llm_model: str | None = None,
        llm_provider: str | None = None,
        disable_retrieval: bool = False,
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
        if query == "Grounded page summary.":
            return [EvidenceHit(filename=self.preferred_filename, text="page verification support", score=0.74)]
        if query.startswith("overall chronology summary "):
            return [EvidenceHit(filename=data_sources[0], text="reduce support", score=0.8)]
        if query == "Supported summary.":
            return [EvidenceHit(filename=data_sources[0], text="verified support", score=0.7)]
        return []


class CaptureBackendRerankGateway:
    def __init__(self) -> None:
        self.chat_calls: list[list[str]] = []
        self.search_queries: list[str] = []

    async def chat_on_sources(
        self,
        *,
        message: str,
        data_sources: list[str] | None = None,
        limit: int = 6,
        score_threshold: float = 0,
        llm_model: str | None = None,
        llm_provider: str | None = None,
        disable_retrieval: bool = False,
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
        self.search_queries.append(query)
        if query.startswith("page summary ") or query.startswith("overall chronology summary "):
            return [EvidenceHit(filename=name, text=f"candidate for {name}", score=0.5) for name in data_sources]
        if query == "Grounded page summary.":
            return [EvidenceHit(filename=data_sources[0], text="page verification support", score=0.72)]
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
        data_sources: list[str] | None = None,
        limit: int = 6,
        score_threshold: float = 0,
        llm_model: str | None = None,
        llm_provider: str | None = None,
        disable_retrieval: bool = False,
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
        if query == "Grounded page summary.":
            return [EvidenceHit(filename=self.relevant_filename, text="page verification support", score=0.72)]
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


class VerificationConcurrencyGateway:
    def __init__(self) -> None:
        self.current_searches = 0
        self.max_searches = 0

    async def chat_on_sources(
        self,
        *,
        message: str,
        data_sources: list[str] | None = None,
        limit: int = 6,
        score_threshold: float = 0,
        llm_model: str | None = None,
        llm_provider: str | None = None,
        disable_retrieval: bool = False,
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
                    "title": "Verification Concurrency Summary",
                    "summary": "Sentence one. Sentence two. Sentence three.",
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
        self.current_searches += 1
        self.max_searches = max(self.max_searches, self.current_searches)
        try:
            await asyncio.sleep(0.01)
            return [EvidenceHit(filename=data_sources[0], text=f"support for {query}", score=0.7)]
        finally:
            self.current_searches -= 1


class MixedPageVerificationGateway:
    def __init__(self) -> None:
        self.reduce_calls: list[list[str]] = []

    async def chat_on_sources(
        self,
        *,
        message: str,
        data_sources: list[str] | None = None,
        limit: int = 6,
        score_threshold: float = 0,
        llm_model: str | None = None,
        llm_provider: str | None = None,
        disable_retrieval: bool = False,
    ):
        if "presentation-layer renderer" in message:
            return (json.dumps({}), [])
        if "Current page:" in message and "page 1" in message:
            return (
                json.dumps(
                    {
                        "summary": "Grounded summary for page 1.",
                        "key_facts": ["page-1-fact"],
                    }
                ),
                [EvidenceHit(filename=data_sources[0], text="page 1 support", score=0.9)],
            )
        if "Current page:" in message and "page 2" in message:
            return (
                json.dumps(
                    {
                        "summary": "Unsupported summary for page 2.",
                        "key_facts": ["page-2-fact"],
                    }
                ),
                [EvidenceHit(filename=data_sources[0], text="page 2 support", score=0.9)],
            )
        self.reduce_calls.append(list(data_sources))
        return (
            json.dumps(
                {
                    "title": "Verified Pages Only",
                    "summary": "Grounded summary for page 1.",
                    "chronology": ["Page 1 event"],
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
        if query == "Grounded summary for page 1.":
            return [EvidenceHit(filename=data_sources[0], text="page 1 verification support", score=0.7)]
        if query == "Unsupported summary for page 2.":
            return []
        if query == "page-2-fact.":
            return []
        return [EvidenceHit(filename=data_sources[0], text=f"support for {query}", score=0.7)]


class InformativePageGateway:
    async def chat_on_sources(
        self,
        *,
        message: str,
        data_sources: list[str] | None = None,
        limit: int = 6,
        score_threshold: float = 0,
        llm_model: str | None = None,
        llm_provider: str | None = None,
        disable_retrieval: bool = False,
    ):
        if "Current page:" in message:
            return (
                json.dumps(
                    {
                        "summary": "This page is primarily header and administrative material.",
                        "key_facts": [
                            "Patient called about diabetes clearance for surgery",
                            "Requested callback at 404-555-1212",
                        ],
                    }
                ),
                [EvidenceHit(filename=data_sources[0], text="page support", score=0.9)],
            )
        return (
            json.dumps(
                {
                    "title": "Informative Scope",
                    "summary": "Supported informative summary.",
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
        supported = {
            "Patient called about diabetes clearance for surgery.",
            "Requested callback at 404-555-1212.",
            "Supported informative summary.",
        }
        if query in supported:
            return [EvidenceHit(filename=data_sources[0], text=f"support for {query}", score=0.8)]
        if query.startswith("page summary ") or query.startswith("overall chronology summary "):
            return [EvidenceHit(filename=data_sources[0], text="preflight support", score=0.75)]
        return []


class GenericReduceGateway:
    def __init__(self) -> None:
        self.reduce_prompts: list[str] = []

    async def chat_on_sources(
        self,
        *,
        message: str,
        data_sources: list[str] | None = None,
        limit: int = 6,
        score_threshold: float = 0,
        llm_model: str | None = None,
        llm_provider: str | None = None,
        disable_retrieval: bool = False,
    ):
        if "Current page:" in message and "page 1" in message:
            return (
                json.dumps(
                    {
                        "summary": "This page is primarily administrative material.",
                        "key_facts": [
                            "Patient requested diabetes clearance for surgery",
                            "Callback number listed as 404-555-1212",
                        ],
                    }
                ),
                [EvidenceHit(filename=data_sources[0], text="page 1 support", score=0.9)],
            )
        if "Current page:" in message and "page 2" in message:
            return (
                json.dumps(
                    {
                        "summary": "This page identifies the provider.",
                        "key_facts": [
                            "Selective nerve block documented",
                            "Indication for procedure documented",
                        ],
                    }
                ),
                [EvidenceHit(filename=data_sources[0], text="page 2 support", score=0.9)],
            )
        self.reduce_prompts.append(message)
        return (
            json.dumps(
                {
                    "title": "Generic Reduce",
                    "summary": "The scope contains an administrative page and a procedure page.",
                    "chronology": ["Page 1 administrative note", "Page 2 procedure note"],
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
        supported = {
            "Patient requested diabetes clearance for surgery.",
            "Callback number listed as 404-555-1212.",
            "Selective nerve block documented.",
            "Indication for procedure documented.",
            "The scope contains an administrative page and a procedure page.",
        }
        if query in supported:
            return [EvidenceHit(filename=data_sources[0], text=f"support for {query}", score=0.8)]
        if query.startswith("page summary ") or query.startswith("overall chronology summary "):
            return [EvidenceHit(filename=data_sources[0], text="preflight support", score=0.75)]
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
    assert result.debug["layered_output_used"] is True
    assert "Grounded summary for page 2." in result.supported_summary
    assert "fact-page-3." in result.supported_summary
    assert result.unsupported_sentences == ["Unsupported leap."]


def test_summarize_scope_uses_only_verified_page_summaries_for_reduce(tmp_path: Path) -> None:
    payload = {
        "pdfs": [
            {
                "pdf_id": "pdf_1",
                "pages": [
                    {"page": 1, "pdf2html_text": "Grounded procedure detail on page one."},
                    {"page": 2, "pdf2html_text": "Weak or noisy content on page two."},
                ],
            }
        ]
    }
    manifest = materialize_summary_payload(
        run_id="verified-pages-run",
        summary_payload=payload,
        source_kind="summary_payload",
        output_dir=tmp_path,
    )
    scope = SummaryScope.model_validate(
        {
            "scope_id": "verified-pages-scope",
            "title": "Verified Pages Scope",
            "objective": "Use only verified page summaries.",
            "page_refs": [
                {"pdf_id": "pdf_1", "page": 1},
                {"pdf_id": "pdf_1", "page": 2},
            ],
        }
    )
    gateway = MixedPageVerificationGateway()

    result = asyncio.run(
        summarize_scope(gateway, manifest=manifest, scope=scope)
    )

    page_1 = next(page for page in result.page_summaries if page.page == 1)
    page_2 = next(page for page in result.page_summaries if page.page == 2)
    page_1_sources = next(page.retrieval_sources() for page in manifest.page_documents if page.page == 1)

    assert page_1.passed_verification is True
    assert page_2.passed_verification is False
    assert result.debug["verified_page_count"] == 1
    assert result.debug["verified_page_source_filenames"] == page_1_sources
    assert gateway.reduce_calls == [page_1_sources]
    assert len(result.page_summaries) == 2


def test_summarize_scope_prefers_supported_informative_page_content(tmp_path: Path) -> None:
    payload = {
        "pdfs": [
            {
                "pdf_id": "pdf_1",
                "pages": [
                    {"page": 1, "pdf2html_text": "Administrative note with clinically important callback details."},
                ],
            }
        ]
    }
    manifest = materialize_summary_payload(
        run_id="informative-page-run",
        summary_payload=payload,
        source_kind="summary_payload",
        output_dir=tmp_path,
    )
    scope = SummaryScope.model_validate(
        {
            "scope_id": "informative-page-scope",
            "title": "Informative Page Scope",
            "objective": "Keep the clinically meaningful callback details.",
            "page_refs": [{"pdf_id": "pdf_1", "page": 1}],
        }
    )

    result = asyncio.run(
        summarize_scope(InformativePageGateway(), manifest=manifest, scope=scope)
    )

    page_summary = result.page_summaries[0]
    assert page_summary.passed_verification is True
    assert "diabetes clearance for surgery" in page_summary.supported_summary.lower()
    assert "404-555-1212" in page_summary.supported_summary
    assert "header and administrative material" not in page_summary.supported_summary.lower()
    assert page_summary.supported_key_facts == [
        "Patient called about diabetes clearance for surgery.",
        "Requested callback at 404-555-1212.",
    ]


def test_summarize_scope_uses_supported_page_fallback_when_reduce_summary_is_too_generic(tmp_path: Path) -> None:
    payload = {
        "pdfs": [
            {
                "pdf_id": "pdf_1",
                "pages": [
                    {"page": 1, "pdf2html_text": "Administrative note with surgery clearance request."},
                    {"page": 2, "pdf2html_text": "Procedure note with selective nerve block detail."},
                ],
            }
        ]
    }
    manifest = materialize_summary_payload(
        run_id="generic-reduce-run",
        summary_payload=payload,
        source_kind="summary_payload",
        output_dir=tmp_path,
    )
    scope = SummaryScope.model_validate(
        {
            "scope_id": "generic-reduce-scope",
            "title": "Generic Reduce Scope",
            "objective": "Prefer specific supported facts.",
            "page_refs": [
                {"pdf_id": "pdf_1", "page": 1},
                {"pdf_id": "pdf_1", "page": 2},
            ],
        }
    )
    gateway = GenericReduceGateway()

    result = asyncio.run(
        summarize_scope(gateway, manifest=manifest, scope=scope)
    )

    assert "diabetes clearance for surgery" in result.supported_summary.lower()
    assert "selective nerve block documented" in result.supported_summary.lower()
    assert result.debug["scope_supported_fallback_used"] is True
    assert "Supported key facts:" in gateway.reduce_prompts[0]


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
    assert result.debug["layered_output_used"] is True
    assert result.supported_summary == "Grounded retry page summary. retry-fact."
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
    assert any(query.startswith("page summary ") for query in gateway.search_queries)
    assert any(query.startswith("overall chronology summary ") for query in gateway.search_queries)
    assert result.debug["reranking_enabled"] is True
    assert result.debug["reranker_location"] == "langflow_agent_tool"
    assert result.debug["reranker_type"] == "cross_encoder"
    assert result.debug["verification_reranker_location"] == "openrag_backend_search_api"


def test_summarize_scope_surfaces_rerank_metadata_and_reuses_generation_pool_for_verification(
    tmp_path: Path,
) -> None:
    class MetadataGateway:
        def __init__(self) -> None:
            self.search_queries: list[str] = []
            self.detail_filename: str | None = None

        async def chat_on_sources(
            self,
            *,
            message: str,
        data_sources: list[str] | None = None,
            limit: int = 6,
            score_threshold: float = 0,
        llm_model: str | None = None,
        llm_provider: str | None = None,
        disable_retrieval: bool = False,
        ):
            if "Current page:" in message:
                return (
                    json.dumps(
                        {
                            "summary": "Possible right ear infection with right ear pain.",
                            "key_facts": ["Chief complaint: possible right ear infection."],
                        }
                    ),
                    [EvidenceHit(filename=data_sources[0], text="noisy chat source", score=9.5)],
                )
            return (
                json.dumps(
                    {
                        "title": "Metadata Summary",
                        "summary": "Possible right ear infection with right ear pain.",
                        "chronology": ["Possible right ear infection with right ear pain."],
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
            self.search_queries.append(query)
            detail_filename = self.detail_filename or data_sources[-1]
            if query.startswith("page summary "):
                return [
                    EvidenceHit(
                        filename=detail_filename,
                        text="### Subjective Possible right ear infection with right ear pain.",
                        score=0.91,
                        base_score=0.41,
                        rerank_score=0.91,
                        retrieval_rank=1,
                    ),
                    EvidenceHit(
                        filename=data_sources[0],
                        text="demographic header and address block",
                        score=0.62,
                        base_score=0.58,
                        rerank_score=0.62,
                        retrieval_rank=2,
                    ),
                ]
            if query.startswith("overall chronology summary "):
                return [
                    EvidenceHit(
                        filename=detail_filename,
                        text="summary level support",
                        score=0.84,
                        base_score=0.44,
                        rerank_score=0.84,
                        retrieval_rank=1,
                    )
                ]
            return []

    payload = {
        "pdfs": [
            {
                "pdf_id": "pdf_1",
                "pages": [
                    {
                        "page": 1,
                        "ocr_text": (
                            "HEADER\n\n"
                            "Demographic header and address block.\n\n"
                            "SUBJECTIVE\n\n"
                            "Possible right ear infection with right ear pain.\n\n"
                            "PLAN\n\n"
                            "Follow up with ENT."
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
        retrieval_rerank_enabled=False,
    )
    manifest = materialize_summary_payload(
        run_id="metadata-rerank-run",
        summary_payload=payload,
        source_kind="summary_payload",
        output_dir=tmp_path,
        settings=settings,
    )
    scope = SummaryScope.model_validate(
        {
            "scope_id": "metadata-rerank-scope",
            "title": "Metadata Rerank Scope",
            "objective": "right ear infection",
            "page_refs": [{"pdf_id": "pdf_1", "page": 1}],
        }
    )
    gateway = MetadataGateway()
    gateway.detail_filename = next(
        document.source_filename
        for document in manifest.retrieval_documents
        if "subjective" in (tmp_path / document.relative_path).read_text(encoding="utf-8").lower()
    )

    result = asyncio.run(
        summarize_scope(gateway, manifest=manifest, scope=scope, settings=settings)
    )

    page_summary = result.page_summaries[0]
    assert page_summary.retrieved_sources[0].filename == gateway.detail_filename
    assert page_summary.retrieved_sources[0].base_score == 0.41
    assert page_summary.retrieved_sources[0].rerank_score == 0.91
    assert page_summary.retrieved_sources[0].retrieval_rank == 1
    assert page_summary.verified_sentences[0].evidence[0].filename == gateway.detail_filename
    assert result.debug["page_requests"][0]["retrieved_source_strategy"] == "preflight_pool"
    assert result.debug["page_requests"][0]["preflight_sources_before_rerank"][0]["base_score"] == 0.41
    assert any(query.startswith("page summary ") for query in gateway.search_queries)


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


def test_summarize_scope_limits_verification_search_concurrency(tmp_path: Path) -> None:
    payload = {
        "pdfs": [
            {
                "pdf_id": "pdf_1",
                "pages": [
                    {
                        "page": 1,
                        "ocr_text": "Procedure note with three supported chronology sentences.",
                    }
                ],
            }
        ]
    }
    settings = AppSettings(
        backend_rerank_enabled=True,
        backend_search_rerank_enabled=True,
        retrieval_rerank_enabled=False,
        verification_concurrency=2,
    )
    manifest = materialize_summary_payload(
        run_id="verification-concurrency-run",
        summary_payload=payload,
        source_kind="summary_payload",
        output_dir=tmp_path,
        settings=settings,
    )
    scope = SummaryScope.model_validate(
        {
            "scope_id": "verification-concurrency-scope",
            "title": "Verification Concurrency Scope",
            "objective": "Verify supported facts.",
            "page_refs": [{"pdf_id": "pdf_1", "page": 1}],
        }
    )
    gateway = VerificationConcurrencyGateway()

    result = __import__("asyncio").run(
        summarize_scope(gateway, manifest=manifest, scope=scope, settings=settings)
    )

    assert gateway.max_searches <= 2
    assert result.debug["verification_concurrency"] == 2



def test_summarize_scope_sanitizes_source_strings_and_keeps_clean_claims(tmp_path: Path) -> None:
    class CleanClaimsGateway:
        async def chat_on_sources(
            self,
            *,
            message: str,
        data_sources: list[str] | None = None,
            limit: int = 6,
            score_threshold: float = 0,
        llm_model: str | None = None,
        llm_provider: str | None = None,
        disable_retrieval: bool = False,
        ):
            if "Current page:" in message:
                return (
                    json.dumps(
                        {
                            "summary": (
                                "Procedure performed by Georges F. Elkhoury, M.D. using 1% Xylocaine. "
                                "(Source: clean_claims__p0001__c0005.md)"
                            ),
                            "key_facts": [
                                "Provider: Georges F. Elkhoury, M.D. (Source: clean_claims__p0001__c0003.md)",
                                "Local anesthetic: 1% Xylocaine. (Source: clean_claims__p0001__c0005.md)",
                            ],
                        }
                    ),
                    [EvidenceHit(filename=data_sources[0], text="page support", score=0.9)],
                )
            return (
                json.dumps(
                    {
                        "title": "Clean Claims Scope",
                        "summary": (
                            "Procedure performed by Georges F. Elkhoury, M.D. using 1% Xylocaine. "
                            "(Source: clean_claims__p0001__c0005.md)"
                        ),
                        "chronology": [
                            "Procedure note recorded. (Source: clean_claims__p0001__c0005.md)"
                        ],
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
            supported_queries = {
                "Procedure performed by Georges F. Elkhoury, M.D. using 1% Xylocaine.",
                "Provider: Georges F. Elkhoury, M.D.",
                "Local anesthetic: 1% Xylocaine.",
            }
            if query in supported_queries:
                return [EvidenceHit(filename=data_sources[0], text=f"support for {query}", score=0.7)]
            return []

    payload = {
        "pdfs": [
            {
                "pdf_id": "pdf_1",
                "pages": [
                    {
                        "page": 1,
                        "ocr_text": "Procedure note with provider and anesthetic details.",
                    }
                ],
            }
        ]
    }
    manifest = materialize_summary_payload(
        run_id="clean-claims-run",
        summary_payload=payload,
        source_kind="summary_payload",
        output_dir=tmp_path,
    )
    scope = SummaryScope.model_validate(
        {
            "scope_id": "clean-claims-scope",
            "title": "Clean Claims Scope",
            "objective": "Summarize the supported procedure claims cleanly.",
            "page_refs": [{"pdf_id": "pdf_1", "page": 1}],
        }
    )

    result = asyncio.run(
        summarize_scope(CleanClaimsGateway(), manifest=manifest, scope=scope)
    )

    page_summary = result.page_summaries[0]
    assert page_summary.summary == "Procedure performed by Georges F. Elkhoury, M.D. using 1% Xylocaine."
    assert "Source:" not in page_summary.summary
    assert page_summary.supported_key_facts == [
        "Provider: Georges F. Elkhoury, M.D.",
        "Local anesthetic: 1% Xylocaine.",
    ]
    assert result.draft_summary == "Procedure performed by Georges F. Elkhoury, M.D. using 1% Xylocaine."
    assert "Procedure performed by Georges F. Elkhoury, M.D. using 1% Xylocaine." in result.supported_summary
    assert "Source:" not in result.supported_summary
    assert result.debug["page_verification"][0]["verification_queries"] == [
        "Procedure performed by Georges F. Elkhoury, M.D. using 1% Xylocaine.",
    ]
    assert result.debug["page_verification"][0]["key_fact_verification_queries"] == [
        "Provider: Georges F. Elkhoury, M.D.",
        "Local anesthetic: 1% Xylocaine.",
    ]
    assert all("Source:" not in query for query in result.debug["verification_queries"])



def test_expand_page_selected_sources_backfills_informative_chunks() -> None:
    page_sources = [
        "page__c0001.md",
        "page__c0007.md",
        "page__c0009.md",
        "page__c0010.md",
    ]
    documents = [
        SimpleNamespace(
            source_filename="page__c0001.md",
            chunk_index=1,
            section_title="PhoneMsg",
            text_preview="Final Report",
            parent_source_filename="page.html",
        ),
        SimpleNamespace(
            source_filename="page__c0007.md",
            chunk_index=7,
            section_title="Clinical Concern",
            text_preview="patient needs an appointment for surgery clearance",
            parent_source_filename="page.html",
        ),
        SimpleNamespace(
            source_filename="page__c0009.md",
            chunk_index=9,
            section_title="CLINICAL CONCERN",
            text_preview="Concern / Question : Pt is req a diabetes clearance for surgery , please assist",
            parent_source_filename="page.html",
        ),
        SimpleNamespace(
            source_filename="page__c0010.md",
            chunk_index=10,
            section_title="Callback",
            text_preview="Preferred Phone # for call back : ( 605 ) 521-6331 or 404 483 3753",
            parent_source_filename="page.html",
        ),
    ]

    expanded, debug = _expand_page_selected_sources(
        selected_sources=["page__c0001.md"],
        page_sources=page_sources,
        page_retrieval_documents=documents,
    )

    assert expanded[0] == "page__c0001.md"
    assert "page__c0009.md" in expanded
    assert "page__c0010.md" in expanded
    assert debug["context_expanded"] is True
    assert debug["informative_backfill_sources"]



def test_filter_page_retrieval_documents_drops_banners_and_boosts_sections() -> None:
    documents = [
        SimpleNamespace(
            source_filename="page__c0001.md",
            chunk_index=1,
            section_title="Fax Header",
            text_preview="From SiliconMesa fax server page 1 of 2",
            parent_source_filename="page.html",
        ),
        SimpleNamespace(
            source_filename="page__c0002.md",
            chunk_index=2,
            section_title="Subjective",
            text_preview="Patient requested diabetes clearance for surgery and callback number 404-555-1212.",
            parent_source_filename="page.html",
        ),
        SimpleNamespace(
            source_filename="page__c0003.md",
            chunk_index=3,
            section_title="Contact Information",
            text_preview="Address on file and phone number updated for insurance routing.",
            parent_source_filename="page.html",
        ),
    ]

    filtered, debug = _filter_page_retrieval_documents(documents)

    assert [document.source_filename for document in filtered] == ["page__c0002.md"]
    assert debug["dropped_count"] == 2
    assert {row["reason"] for row in debug["dropped"]} == {"header_footer_artifact"}
    assert _document_priority_score(documents[1]) > _document_priority_score(documents[2])


class LayeredOutputGateway:
    def __init__(self) -> None:
        self.override_calls: list[dict[str, object]] = []

    async def chat_on_sources(
        self,
        *,
        message: str,
        data_sources: list[str] | None = None,
        limit: int = 6,
        score_threshold: float = 0,
        llm_model: str | None = None,
        llm_provider: str | None = None,
        disable_retrieval: bool = False,
    ):
        self.override_calls.append(
            {
                "message": message,
                "data_sources": list(data_sources or []),
                "llm_model": llm_model,
                "llm_provider": llm_provider,
                "disable_retrieval": disable_retrieval,
            }
        )
        source_name = (data_sources or ["fallback-source.md"])[0]
        if "strict supported fact sheet" in message:
            return (
                json.dumps(
                    {
                        "date_of_service": ["09/12/2018"],
                        "facility": ["Clinica La Esperanza in Albuquerque, New Mexico"],
                        "provider": ["Susette Eaves CFNP"],
                        "patient_reference": ["Mr. Reazin"],
                        "note_type": ["visit note"],
                        "chief_complaint": ["possible right ear infection with right ear pain for a couple of weeks"],
                        "hpi": ["right ear pain for 2 weeks with prior upper respiratory infection resolved"],
                        "allergies": ["nkda"],
                        "medications": ["hydrocodone acetaminophen 5-325 mg tablet", "oxycodone 5 mg tablet"],
                        "diagnoses": ["otalgia right ear H92.01"],
                        "treatment": ["possible surgery or laser treatment"],
                        "plan": ["ENT follow up with possible surgery or laser treatment"],
                        "follow_up": ["prn"],
                        "positive_ros": ["muscle or joint pain"],
                        "positive_physical_exam": ["well-appearing with normal ear and oropharyngeal findings"],
                        "residual_supported_facts": ["Right ear hearing loss could not be corrected with aides"],
                    }
                ),
                [EvidenceHit(filename=source_name, text="structured support", score=0.9)],
            )
        if "presentation-layer renderer" in message:
            return (
                json.dumps(
                    {
                        "title": "Layered Scope",
                        "narrative": "On 09/12/2018, at Clinica La Esperanza in Albuquerque, New Mexico, Susette Eaves CFNP documented that Mr. Reazin presented for evaluation. Chief complaint was possible right ear infection with right ear pain for a couple of weeks. History of present illness documented right ear pain for 2 weeks with prior upper respiratory infection resolved. Allergies were documented as nkda. Medications included hydrocodone acetaminophen 5-325 mg tablet and oxycodone 5 mg tablet. Diagnoses included otalgia right ear H92.01. Treatment included possible surgery or laser treatment. Plan included ENT follow up with possible surgery or laser treatment. Follow-up included prn.",
                        "sections": [
                            {
                                "title": "On 09/12/2018",
                                "note_id": "note-001",
                                "items": [
                                    {"text": "On 09/12/2018, at Clinica La Esperanza in Albuquerque, New Mexico, Susette Eaves CFNP documented that Mr. Reazin presented for evaluation.", "fact_ids": ["note-001__intro__01"]},
                                    {"text": "Chief complaint was possible right ear infection with right ear pain for a couple of weeks.", "fact_ids": ["note-001__chief_complaint__01"]},
                                    {"text": "History of present illness documented right ear pain for 2 weeks with prior upper respiratory infection resolved.", "fact_ids": ["note-001__hpi__01"]},
                                    {"text": "Allergies were documented as nkda.", "fact_ids": ["note-001__allergies__01"]},
                                    {"text": "Medications included hydrocodone acetaminophen 5-325 mg tablet and oxycodone 5 mg tablet.", "fact_ids": ["note-001__medications__01"]},
                                    {"text": "Diagnoses included otalgia right ear H92.01.", "fact_ids": ["note-001__diagnoses__01"]},
                                    {"text": "Treatment included possible surgery or laser treatment.", "fact_ids": ["note-001__treatment__01"]},
                                    {"text": "Plan included ENT follow up with possible surgery or laser treatment.", "fact_ids": ["note-001__plan__01"]},
                                    {"text": "Follow-up included prn.", "fact_ids": ["note-001__follow_up__01"]}
                                ]
                            }
                        ]
                    }
                ),
                [],
            )
        if "presentation-layer editor" in message:
            return (
                json.dumps(
                    {
                        "title": "Layered Scope",
                        "narrative": "On 09/12/2018, at Clinica La Esperanza in Albuquerque, New Mexico, Susette Eaves CFNP documented that Mr. Reazin presented for evaluation. He reported possible right ear infection with right ear pain for a couple of weeks, and the history of present illness noted a resolved prior upper respiratory infection. Allergies were documented as nkda. Current medications included hydrocodone acetaminophen 5-325 mg tablet and oxycodone 5 mg tablet. Diagnoses included otalgia right ear H92.01. Treatment included possible surgery or laser treatment. Plan included ENT follow up with possible surgery or laser treatment. Follow-up included prn.",
                        "sections": [
                            {
                                "title": "On 09/12/2018",
                                "note_id": "note-001",
                                "items": [
                                    {"text": "On 09/12/2018, at Clinica La Esperanza in Albuquerque, New Mexico, Susette Eaves CFNP documented that Mr. Reazin presented for evaluation.", "fact_ids": ["note-001__intro__01"]},
                                    {"text": "He reported possible right ear infection with right ear pain for a couple of weeks, and the history of present illness noted a resolved prior upper respiratory infection.", "fact_ids": ["note-001__chief_complaint__01", "note-001__hpi__01"]},
                                    {"text": "Allergies were documented as nkda.", "fact_ids": ["note-001__allergies__01"]},
                                    {"text": "Current medications included hydrocodone acetaminophen 5-325 mg tablet and oxycodone 5 mg tablet.", "fact_ids": ["note-001__medications__01"]},
                                    {"text": "Diagnoses included otalgia right ear H92.01.", "fact_ids": ["note-001__diagnoses__01"]},
                                    {"text": "Treatment included possible surgery or laser treatment.", "fact_ids": ["note-001__treatment__01"]},
                                    {"text": "Plan included ENT follow up with possible surgery or laser treatment.", "fact_ids": ["note-001__plan__01"]},
                                    {"text": "Follow-up included prn.", "fact_ids": ["note-001__follow_up__01"]}
                                ]
                            }
                        ]
                    }
                ),
                [],
            )
        if "single page" in message:
            return (
                json.dumps(
                    {
                        "summary": "Visit note documenting right ear pain, medications, diagnosis, and ENT follow up.",
                        "key_facts": [
                            "Date of service 09/12/2018 at Clinica La Esperanza in Albuquerque, New Mexico.",
                            "Provider Susette Eaves CFNP.",
                            "Chief complaint possible right ear infection with right ear pain for a couple of weeks.",
                            "Allergies listed as nkda.",
                            "Medications hydrocodone acetaminophen 5-325 mg tablet and oxycodone 5 mg tablet.",
                            "Diagnosis otalgia right ear H92.01.",
                            "Plan ENT follow up with possible surgery or laser treatment.",
                        ],
                    }
                ),
                [EvidenceHit(filename=source_name, text="page support", score=0.9)],
            )
        return (
            json.dumps(
                {
                    "title": "Visit Note Summary",
                    "summary": "Visit note documenting right ear pain and follow up planning.",
                    "chronology": ["09/12/2018 visit note documented right ear pain and ENT follow up."],
                }
            ),
            [EvidenceHit(filename=source_name, text="reduce support", score=0.8)],
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
            return [EvidenceHit(filename=data_sources[0], text="preflight support", score=0.8)]
        supported_queries = {
            "Visit note documenting right ear pain, medications, diagnosis, and ENT follow up.",
            "Date of service 09/12/2018 at Clinica La Esperanza in Albuquerque, New Mexico.",
            "Provider Susette Eaves CFNP.",
            "Chief complaint possible right ear infection with right ear pain for a couple of weeks.",
            "Allergies listed as nkda.",
            "Medications hydrocodone acetaminophen 5-325 mg tablet and oxycodone 5 mg tablet.",
            "Diagnosis otalgia right ear H92.01.",
            "Plan ENT follow up with possible surgery or laser treatment.",
            "Visit note documenting right ear pain and follow up planning.",
        }
        if query in supported_queries:
            return [EvidenceHit(filename=data_sources[0], text="verified support", score=0.8)]
        return []


class PageFirstLayeredGateway(LayeredOutputGateway):
    async def chat_on_sources(
        self,
        *,
        message: str,
        data_sources: list[str] | None = None,
        limit: int = 6,
        score_threshold: float = 0,
        llm_model: str | None = None,
        llm_provider: str | None = None,
        disable_retrieval: bool = False,
    ):
        if "presentation-layer editor" in message:
            self.override_calls.append(
                {
                    "message": message,
                    "data_sources": list(data_sources or []),
                    "llm_model": llm_model,
                    "llm_provider": llm_provider,
                    "disable_retrieval": disable_retrieval,
                }
            )
            return (
                json.dumps(
                    {
                        "title": "Layered Scope",
                        "narrative": "This page documented the date of service and provider information for the visit note. The page states that right ear pain was discussed.",
                        "sections": [
                            {
                                "title": "Page 1",
                                "note_id": "note-001",
                                "items": [
                                    {
                                        "text": "This page documented the date of service and provider information for the visit note.",
                                        "fact_ids": ["note-001__intro__01"],
                                    },
                                    {
                                        "text": "The page states that right ear pain was discussed.",
                                        "fact_ids": ["note-001__chief_complaint__01"],
                                    }
                                ],
                            }
                        ],
                    }
                ),
                [],
            )
        return await super().chat_on_sources(
            message=message,
            data_sources=data_sources,
            limit=limit,
            score_threshold=score_threshold,
            llm_model=llm_model,
            llm_provider=llm_provider,
            disable_retrieval=disable_retrieval,
        )



class ResidualOnlyLayeredGateway:
    def __init__(self) -> None:
        self.override_calls: list[dict[str, object]] = []

    async def chat_on_sources(
        self,
        *,
        message: str,
        data_sources: list[str] | None = None,
        limit: int = 6,
        score_threshold: float = 0,
        llm_model: str | None = None,
        llm_provider: str | None = None,
        disable_retrieval: bool = False,
    ):
        self.override_calls.append(
            {
                "message": message,
                "data_sources": list(data_sources or []),
                "llm_model": llm_model,
                "llm_provider": llm_provider,
                "disable_retrieval": disable_retrieval,
            }
        )
        source_name = (data_sources or ["fallback-source.md"])[0]
        if "strict supported fact sheet" in message:
            return (
                json.dumps(
                    {
                        "date_of_service": [],
                        "facility": [],
                        "provider": [],
                        "patient_reference": [],
                        "note_type": [],
                        "chief_complaint": [],
                        "hpi": [],
                        "pmh": [],
                        "psh": [],
                        "social_history": [],
                        "allergies": [],
                        "medications": [],
                        "vitals": [],
                        "abnormal_labs": [],
                        "diagnoses": [],
                        "assessment": [],
                        "treatment": [],
                        "plan": [],
                        "follow_up": [],
                        "positive_ros": [],
                        "positive_physical_exam": [],
                        "residual_supported_facts": ["Right knee pain improved with injections and therapy."],
                    }
                ),
                [EvidenceHit(filename=source_name, text="structured residual support", score=0.9)],
            )
        if "presentation-layer renderer" in message:
            return (
                json.dumps(
                    {
                        "title": "Residual Layered Scope",
                        "narrative": "The note documented right knee pain improved with injections and therapy.",
                        "sections": [
                            {
                                "title": "Note 1",
                                "note_id": "note-001",
                                "items": [
                                    {
                                        "text": "The note documented right knee pain improved with injections and therapy.",
                                        "fact_ids": ["note-001__residual_supported_facts__01"],
                                    }
                                ],
                            }
                        ],
                    }
                ),
                [],
            )
        if "presentation-layer editor" in message:
            return (
                json.dumps(
                    {
                        "title": "Residual Layered Scope",
                        "narrative": "The note documented right knee pain improved with injections and therapy.",
                        "sections": [
                            {
                                "title": "Note 1",
                                "note_id": "note-001",
                                "items": [
                                    {
                                        "text": "The note documented right knee pain improved with injections and therapy.",
                                        "fact_ids": ["note-001__residual_supported_facts__01"],
                                    }
                                ],
                            }
                        ],
                    }
                ),
                [],
            )
        if "single page" in message:
            return (
                json.dumps(
                    {
                        "summary": "The note documented right knee pain improved with injections and therapy.",
                        "key_facts": ["Right knee pain improved with injections and therapy."],
                    }
                ),
                [EvidenceHit(filename=source_name, text="page support", score=0.9)],
            )
        return (
            json.dumps(
                {
                    "title": "Residual Summary",
                    "summary": "The note documented right knee pain improved with injections and therapy.",
                    "chronology": ["The note documented right knee pain improved with injections and therapy."],
                }
            ),
            [EvidenceHit(filename=source_name, text="reduce support", score=0.8)],
        )

    async def search_on_sources(
        self,
        *,
        query: str,
        data_sources: list[str],
        limit: int | None = None,
        score_threshold: float | None = None,
    ):
        if data_sources:
            return [EvidenceHit(filename=data_sources[0], text="verified support", score=0.8)]
        return []


def test_summarize_scope_emits_truth_validation_and_presentation_layers(tmp_path: Path) -> None:
    payload = {
        "pdfs": [
            {
                "pdf_id": "visit-note.pdf",
                "pages": [
                    {"page": 1, "pdf2html_text": "Visit note page one", "service_date": "09/12/2018"},
                ],
            }
        ]
    }
    materialize_summary_payload(
        run_id="layered-run",
        summary_payload=payload,
        source_kind="summary_payload",
        output_dir=tmp_path,
    )
    manifest = load_manifest(tmp_path / "manifest.json")
    scope = SummaryScope.model_validate(
        {
            "scope_id": "layered-scope",
            "title": "Layered Scope",
            "objective": "Produce a detailed grounded visit note narrative.",
            "page_refs": [{"pdf_id": "visit-note.pdf", "page": 1}],
        }
    )

    gateway = LayeredOutputGateway()
    settings = AppSettings(
        extractor_llm_provider="openai",
        extractor_llm_model="gpt-5-extractor",
        renderer_llm_provider="openai",
        renderer_llm_model="gpt-5-renderer",
        editor_llm_provider="openai",
        editor_llm_model="gpt-5-editor",
    )

    result = asyncio.run(
        summarize_scope(gateway, manifest=manifest, scope=scope, settings=settings)
    )

    assert result.truth_layer
    truth_note = result.truth_layer[0]
    assert truth_note.date_of_service[0].value == "09/12/2018"
    assert truth_note.provider[0].value == "Susette Eaves CFNP"
    assert any("H92.01" in fact.value for fact in truth_note.diagnoses)
    assert result.validation_layer is not None
    assert result.validation_layer.passed is True
    assert result.presentation_plan is not None
    assert result.presentation_draft is not None
    assert result.presentation_layer is not None
    assert result.presentation_plan.debug["renderer"] == "deterministic_plan"
    assert result.presentation_draft.debug["renderer"] == "llm_draft"
    assert result.presentation_layer.debug["renderer"] == "llm_editor"
    assert result.presentation_layer.debug["rendered_by_model"] is True
    assert result.debug["layered_output_used"] is True
    assert result.supported_summary.startswith("On 09/12/2018")
    assert "Susette Eaves CFNP" in result.supported_summary
    assert "hydrocodone acetaminophen 5-325 mg tablet" in result.supported_summary
    assert "well-appearing" not in result.supported_summary.lower()
    assert not any(item.field_name == "positive_physical_exam" for section in result.presentation_plan.sections for item in section.items)
    assert any("note-001__diagnoses__01" in item.fact_ids for section in result.presentation_layer.sections for item in section.items)
    assert any(call["llm_model"] == "gpt-5-extractor" for call in gateway.override_calls if "strict supported fact sheet" in str(call["message"]))
    assert any(call["llm_model"] == "gpt-5-renderer" and call["disable_retrieval"] is True for call in gateway.override_calls if "presentation-layer renderer" in str(call["message"]))
    assert any(call["llm_model"] == "gpt-5-editor" and call["disable_retrieval"] is True for call in gateway.override_calls if "presentation-layer editor" in str(call["message"]))
    assert result.debug["presentation_omission_audit"]["notes_with_missing_fields"] == 0
    assert result.debug["presentation_quality_gate"]["selected_layer"] == "candidate"


def test_summarize_scope_falls_back_to_deterministic_note_plan_when_editor_is_page_first(tmp_path: Path) -> None:
    payload = {
        "pdfs": [
            {
                "pdf_id": "visit-note.pdf",
                "pages": [
                    {
                        "page": 1,
                        "pdf2html_text": (
                            "Date of Service 09/12/2018. Provider Susette Eaves CFNP. "
                            "Possible right ear infection with right ear pain for a couple of weeks. "
                            "NKDA. Hydrocodone acetaminophen 5-325 mg tablet. "
                            "Diagnosis otalgia right ear H92.01. "
                            "Possible surgery or laser treatment. ENT follow up PRN."
                        ),
                    },
                ],
            }
        ]
    }
    materialize_summary_payload(
        run_id="layered-page-first-run",
        summary_payload=payload,
        source_kind="summary_payload",
        output_dir=tmp_path,
    )
    manifest = load_manifest(tmp_path / "manifest.json")
    scope = SummaryScope.model_validate(
        {
            "scope_id": "layered-page-first-scope",
            "title": "Layered Scope",
            "objective": "Produce a detailed grounded visit note narrative.",
            "page_refs": [{"pdf_id": "visit-note.pdf", "page": 1}],
        }
    )

    result = asyncio.run(
        summarize_scope(PageFirstLayeredGateway(), manifest=manifest, scope=scope)
    )

    assert result.presentation_plan is not None
    assert result.presentation_layer is not None
    assert result.presentation_layer.debug["renderer"] == "deterministic_plan"
    assert result.debug["layered_output_used"] is True
    assert result.debug["layered_output_fallback_reason"] == "forbidden_page_language"
    assert result.debug["presentation_quality_gate"]["selected_layer"] == "deterministic_plan"
    assert result.supported_summary.startswith("On 09/12/2018")
    assert "this page" not in result.supported_summary.lower()
    assert "nkda" in result.supported_summary.lower()
    assert "follow-up included prn" in result.supported_summary.lower()



def test_summarize_scope_uses_layered_output_for_residual_only_notes(tmp_path: Path) -> None:
    payload = {
        "pdfs": [
            {
                "pdf_id": "visit-note.pdf",
                "pages": [
                    {"page": 1, "pdf2html_text": "Visit note page one"},
                ],
            }
        ]
    }
    materialize_summary_payload(
        run_id="layered-residual-run",
        summary_payload=payload,
        source_kind="summary_payload",
        output_dir=tmp_path,
    )
    manifest = load_manifest(tmp_path / "manifest.json")
    scope = SummaryScope.model_validate(
        {
            "scope_id": "layered-residual-scope",
            "title": "Layered Residual Scope",
            "objective": "Produce a detailed grounded visit note narrative.",
            "page_refs": [{"pdf_id": "visit-note.pdf", "page": 1}],
        }
    )

    result = asyncio.run(
        summarize_scope(ResidualOnlyLayeredGateway(), manifest=manifest, scope=scope)
    )

    assert result.truth_layer
    assert result.presentation_layer is not None
    assert result.debug["layered_output_structured"] is False
    assert result.debug["layered_output_used"] is True
    assert result.supported_summary == "The note documented right knee pain improved with injections and therapy."
    assert result.debug["note_group_debug"]["note_count"] == 1
    assert result.debug["presentation_omission_audit"]["notes_with_missing_fields"] == 0


def test_summarize_scope_uses_local_inventory_fallback_when_retrieval_returns_empty(tmp_path: Path) -> None:
    payload = {
        "pdfs": [
            {
                "pdf_id": "pdf_1",
                "pages": [
                    {
                        "page": 1,
                        "ocr_text": (
                            "Date of Service: 8/16/2024 HISTORY OF PRESENTING COMPLAINT: "
                            "She rates her pain as a 8 out of 10. The patient notes this pain as frequent and sharp in nature. "
                            "She reports increased pain with bending. She states that her pain is improved with injections and therapy. "
                            "Past Medical History: Positive and includes hypertension and high cholesterol. "
                            "Social History: Reports tobacco use. Reports alcohol use."
                        ),
                    }
                ],
            }
        ]
    }

    manifest = materialize_summary_payload(
        run_id="local-inventory-run",
        summary_payload=payload,
        source_kind="summary_payload",
        output_dir=tmp_path,
        settings=AppSettings(
            structure_chunk_target_chars=160,
            structure_chunk_overlap_blocks=0,
        ),
    )
    scope = SummaryScope.model_validate(
        {
            "scope_id": "local-inventory-scope",
            "title": "Local Inventory Scope",
            "objective": "Summarize the date of service, pain description, history, and social history.",
            "page_refs": [{"pdf_id": "pdf_1", "page": 1}],
        }
    )

    result = asyncio.run(
        summarize_scope(LocalInventoryFallbackGateway(), manifest=manifest, scope=scope)
    )

    page_summary = result.page_summaries[0]
    assert page_summary.summary.startswith("Patient reported frequent sharp pain")
    assert page_summary.supported_summary
    assert page_summary.passed_verification is True
    assert page_summary.retrieved_sources
    assert result.debug["page_requests"][0]["retrieved_source_strategy"] == "local_page_documents"
    assert result.debug["page_requests"][0]["local_inventory_fallback_used"] is True
