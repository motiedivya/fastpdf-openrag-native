from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from fastpdf_openrag_native.models import PdfPipelineResult, ScopedSummaryResult, SummaryScope
from fastpdf_openrag_native.service import (
    _backfill_summary_artifacts,
    _debug_url_for_path,
    _job_debug_artifacts,
    _job_payload,
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
        "citation_instances_path": "outputs/run-1/sentence_citation_instances.json",
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
    assert artifacts["citation_instances"]["url"] == "/debug-outputs/run-1/sentence_citation_instances.json"
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
    assert "citationHasRegionMapping" in html
    assert "Number(page?.page || 0) === Number(citation?.page || 0)" in html
    assert "sentence-meta" in html
    assert 'href="/ui/runs/job-1"' in html
    assert 'href="/ui/runs/job-1/trace"' in html
    assert 'href="/ui/runs/job-1/summary-data"' in html
    assert "Select a summary sentence to open the cited evidence." in html
    assert "Hover preview" not in html



def test_job_payload_recovers_artifacts_from_store_and_run_id(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    output_root = tmp_path / "outputs"
    extraction_dir = data_root / "extracted" / "run-1"
    artifacts_dir = extraction_dir / "artifacts"
    trace_dir = output_root / "traces" / "run-1"
    run_output_dir = output_root / "run-1"
    artifacts_dir.mkdir(parents=True)
    trace_dir.mkdir(parents=True)
    run_output_dir.mkdir(parents=True)
    (extraction_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (trace_dir / "events.jsonl").write_text("", encoding="utf-8")
    (trace_dir / "summary.json").write_text("{}", encoding="utf-8")
    (run_output_dir / "all-pages.summary.json").write_text("{}", encoding="utf-8")
    (run_output_dir / "citation_index.json").write_text("[]", encoding="utf-8")
    (run_output_dir / "sentence_citation_instances.json").write_text("[]", encoding="utf-8")
    (run_output_dir / "resolved_citations.json").write_text("{}", encoding="utf-8")
    (artifacts_dir / "source.pdf").write_text("pdf", encoding="utf-8")

    jobs_path = tmp_path / "fastpdf-ui-jobs.json"
    jobs_path.write_text('{"runs": [{"job_id": "job-1", "status": "completed", "run_id": "run-1"}]}', encoding="utf-8")

    monkeypatch.setattr(service, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(service, "DATA_ROOT", data_root)
    monkeypatch.setattr(service, "OUTPUT_ROOT", output_root)
    monkeypatch.setattr(service, "UI_JOBS_PATH", jobs_path)
    service._jobs.clear()

    payload = _job_payload("job-1")

    assert payload["manifest_path"] == (extraction_dir / "manifest.json").as_posix()
    assert payload["summary_path"] == (run_output_dir / "all-pages.summary.json").as_posix()
    assert payload["trace_path"] == (trace_dir / "events.jsonl").as_posix()
    assert payload["trace_summary_path"] == (trace_dir / "summary.json").as_posix()
    assert payload["citation_instances_path"] == (run_output_dir / "sentence_citation_instances.json").as_posix()
    assert payload["source_pdf_copy_path"] == (artifacts_dir / "source.pdf").as_posix()
    assert payload["debug_artifacts"]["pipeline_summary"]["url"] == "/debug-outputs/run-1/all-pages.summary.json"


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


def test_backfill_summary_artifacts_rebuilds_missing_summary_files(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    output_root = tmp_path / "outputs"
    extraction_dir = data_root / "extracted" / "run-1"
    extraction_dir.mkdir(parents=True)
    manifest_path = extraction_dir / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    jobs_path = tmp_path / "fastpdf-ui-jobs.json"

    monkeypatch.setattr(service, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(service, "DATA_ROOT", data_root)
    monkeypatch.setattr(service, "OUTPUT_ROOT", output_root)
    monkeypatch.setattr(service, "UI_JOBS_PATH", jobs_path)
    service._jobs.clear()

    fake_manifest = SimpleNamespace(run_id="run-1", page_documents=[])
    fake_scope = SummaryScope(scope_id="all-pages", title="All Pages Summary", page_refs=[])
    fake_summary = ScopedSummaryResult(
        run_id="run-1",
        scope=fake_scope,
        draft_title="All Pages Summary",
        draft_summary="Supported narrative.",
        supported_summary="Supported narrative.",
    )

    monkeypatch.setattr(service, "load_manifest", lambda _path: fake_manifest)
    monkeypatch.setattr(service, "_all_pages_scope", lambda manifest, objective=None: fake_scope)

    async def fake_summarize_scope(*_args, **_kwargs):
        return fake_summary

    monkeypatch.setattr(service, "summarize_scope", fake_summarize_scope)

    def fake_ensure_summary_citations(*, summary_path, manifest_path, source_pdf, summary=None, manifest=None):
        summary_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
        citation_index_path = summary_path.parent / "citation_index.json"
        citation_instances_path = summary_path.parent / "sentence_citation_instances.json"
        resolved_citations_path = summary_path.parent / "resolved_citations.json"
        citation_index_path.write_text("[]", encoding="utf-8")
        citation_instances_path.write_text("[]", encoding="utf-8")
        resolved_citations_path.write_text('{"sections": [], "source_pages": []}', encoding="utf-8")
        return summary, citation_index_path, citation_instances_path, resolved_citations_path, None

    monkeypatch.setattr(service, "ensure_summary_citations", fake_ensure_summary_citations)

    payload = {
        "job_id": "job-1",
        "status": "completed",
        "run_id": "run-1",
        "manifest_path": manifest_path.as_posix(),
        "pdf_path": (tmp_path / "source.pdf").as_posix(),
        "question": "summarize",
    }

    recovered = asyncio.run(_backfill_summary_artifacts("job-1", payload))

    assert recovered["summary_path"] == (output_root / "run-1" / "all-pages.summary.json").as_posix()
    assert recovered["citation_index_path"] == (output_root / "run-1" / "citation_index.json").as_posix()
    assert recovered["citation_instances_path"] == (output_root / "run-1" / "sentence_citation_instances.json").as_posix()
    assert recovered["resolved_citations_path"] == (output_root / "run-1" / "resolved_citations.json").as_posix()
    assert recovered["summary_error"] is None
    assert (output_root / "run-1" / "all-pages.summary.json").exists()


def test_build_summary_view_model_surfaces_summary_error_over_missing_artifacts() -> None:
    payload = {
        "job_id": "job-1",
        "status": "completed_with_errors",
        "summary_error": "RuntimeError: summary stage failed",
        "manifest_path": "data/extracted/run-1/manifest.json",
        "summary_path": None,
    }

    try:
        service._build_summary_view_model("job-1", payload)
    except Exception as exc:  # HTTPException
        assert getattr(exc, "status_code", None) == 500
        assert getattr(exc, "detail", None) == "RuntimeError: summary stage failed"
    else:
        raise AssertionError("expected summary view model to surface summary_error")
