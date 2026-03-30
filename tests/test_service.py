from __future__ import annotations

import asyncio
from pathlib import Path

from fastpdf_openrag_native.models import PdfPipelineResult
from fastpdf_openrag_native.service import (
    _debug_url_for_path,
    _job_debug_artifacts,
    _merge_sections_for_display,
    _render_summary_page,
)
from fastpdf_openrag_native import service


def test_debug_url_for_repo_paths() -> None:
    assert _debug_url_for_path("data/extracted/run-1/manifest.json") == "/debug-data/extracted/run-1/manifest.json"
    assert _debug_url_for_path("outputs/traces/run-1/events.jsonl") == "/debug-outputs/traces/run-1/events.jsonl"


def test_job_debug_artifacts_exposes_primary_debug_links() -> None:
    payload = {
        "extraction_dir": "data/extracted/run-1",
        "manifest_path": "data/extracted/run-1/manifest.json",
        "trace_path": "outputs/traces/run-1/events.jsonl",
        "trace_summary_path": "outputs/traces/run-1/summary.json",
        "summary_path": "outputs/run-1/all-pages.summary.json",
        "trace_dir": "outputs/traces/run-1",
        "chunk_dump_dir": "outputs/traces/run-1/chunks",
        "citation_index_path": "outputs/run-1/citation_index.json",
        "resolved_citations_path": "outputs/run-1/resolved_citations.json",
        "source_pdf_copy_path": "data/extracted/run-1/artifacts/source.pdf",
    }

    artifacts = _job_debug_artifacts(payload)

    assert artifacts["manifest"]["url"] == "/debug-data/extracted/run-1/manifest.json"
    assert artifacts["pages_dir"]["url"] == "/debug-data/extracted/run-1/pages"
    assert artifacts["artifacts_dir"]["url"] == "/debug-data/extracted/run-1/artifacts"
    assert artifacts["trace_events"]["url"] == "/debug-outputs/traces/run-1/events.jsonl"
    assert artifacts["trace_summary"]["url"] == "/debug-outputs/traces/run-1/summary.json"
    assert artifacts["pipeline_summary"]["url"] == "/debug-outputs/run-1/all-pages.summary.json"
    assert artifacts["chunk_dump_dir"]["url"] == "/debug-outputs/traces/run-1/chunks"
    assert artifacts["citation_index"]["url"] == "/debug-outputs/run-1/citation_index.json"
    assert artifacts["resolved_citations"]["url"] == "/debug-outputs/run-1/resolved_citations.json"
    assert artifacts["source_pdf_copy"]["url"] == "/debug-data/extracted/run-1/artifacts/source.pdf"


def test_run_job_marks_summary_failure_as_completed_with_errors(monkeypatch) -> None:
    async def fake_run_pdf_pipeline(**_kwargs):
        return PdfPipelineResult(
            run_id="run-1",
            source_pdf="/tmp/doc.pdf",
            extraction_dir="data/extracted/run-1",
            manifest_path="data/extracted/run-1/manifest.json",
            trace_path="outputs/traces/run-1/events.jsonl",
            trace_summary_path="outputs/traces/run-1/summary.json",
            trace_dir="outputs/traces/run-1",
            chunk_dump_dir="outputs/traces/run-1/chunks",
            total_pages=1,
            materialized_pages=1,
            summary_error="summary crashed",
        )

    monkeypatch.setattr(service, "run_pdf_pipeline", fake_run_pdf_pipeline)
    service._jobs.clear()

    asyncio.run(
        service._run_job(
            job_id="job-1",
            pdf_path=Path("/tmp/doc.pdf"),
            credentials_path=None,
            question="summarize",
            max_pages=None,
        )
    )

    assert service._jobs["job-1"]["status"] == "completed_with_errors"
    assert service._jobs["job-1"]["summary_error"] == "summary crashed"

def test_render_summary_page_uses_batch_style_pdf_viewer_and_click_only_markers() -> None:
    html = _render_summary_page("job-1")

    assert "pdf-viewer" in html
    assert "pdf-page" in html
    assert "pdf-text-layer" in html
    assert "citation-box" in html
    assert "computeDisplayScale" in html
    assert "sentence-meta" in html
    assert "Select a summary sentence to open the cited evidence." in html
    assert "Hover preview" not in html




def test_merge_sections_for_display_builds_single_detailed_summary_without_chronology() -> None:
    sections = [
        {
            "section_id": "supported-summary",
            "title": "Supported Summary",
            "kind": "supported_summary",
            "items": [{"item_id": "a", "text": "Overall supported claim.", "citation_ids": ["c1"]}],
        },
        {
            "section_id": "page-1-summary",
            "title": "Page 1 Summary",
            "kind": "page_summary",
            "pdf_id": "doc.pdf",
            "page": 1,
            "items": [{"item_id": "b", "text": "Page 1 summary line.", "citation_ids": ["c2"]}],
        },
        {
            "section_id": "page-1-key-facts",
            "title": "Page 1 Key Facts",
            "kind": "key_facts",
            "pdf_id": "doc.pdf",
            "page": 1,
            "items": [
                {"item_id": "c", "text": "Page 1 summary line.", "citation_ids": ["c2"]},
                {"item_id": "d", "text": "Page 1 fact line.", "citation_ids": ["c3"]},
            ],
        },
        {
            "section_id": "chronology",
            "title": "Chronology",
            "kind": "chronology",
            "items": [{"item_id": "e", "text": "Chronology line.", "citation_ids": ["c4"]}],
        },
        {
            "section_id": "draft-summary",
            "title": "Draft Summary",
            "kind": "draft_summary",
            "items": [{"item_id": "f", "text": "Draft line.", "citation_ids": []}],
        },
    ]

    merged = _merge_sections_for_display(sections)

    assert [section["kind"] for section in merged] == ["detailed_summary", "draft_summary"]
    detailed_section = merged[0]
    assert detailed_section["title"] == "Detailed Summary"
    assert [item["text"] for item in detailed_section["items"]] == [
        "Overall supported claim.",
        "doc.pdf · Page 1: Page 1 summary line.",
        "doc.pdf · Page 1: Page 1 fact line.",
    ]
