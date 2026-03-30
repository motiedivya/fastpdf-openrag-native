from __future__ import annotations

import asyncio
from pathlib import Path

from fastpdf_openrag_native.models import PdfPipelineResult
from fastpdf_openrag_native.service import _debug_url_for_path, _job_debug_artifacts
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
    }

    artifacts = _job_debug_artifacts(payload)

    assert artifacts["manifest"]["url"] == "/debug-data/extracted/run-1/manifest.json"
    assert artifacts["pages_dir"]["url"] == "/debug-data/extracted/run-1/pages"
    assert artifacts["artifacts_dir"]["url"] == "/debug-data/extracted/run-1/artifacts"
    assert artifacts["trace_events"]["url"] == "/debug-outputs/traces/run-1/events.jsonl"
    assert artifacts["trace_summary"]["url"] == "/debug-outputs/traces/run-1/summary.json"
    assert artifacts["pipeline_summary"]["url"] == "/debug-outputs/run-1/all-pages.summary.json"
    assert artifacts["chunk_dump_dir"]["url"] == "/debug-outputs/traces/run-1/chunks"


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
