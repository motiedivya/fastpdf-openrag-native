from __future__ import annotations

from pathlib import Path

from fastpdf_openrag_native.models import MaterializationManifest
from fastpdf_openrag_native.ocr_extract import build_html_document
from fastpdf_openrag_native.trace import TraceRecorder


def test_trace_recorder_writes_jsonl(tmp_path: Path) -> None:
    recorder = TraceRecorder(tmp_path)
    recorder.record(
        stage="ocr",
        service="google_vision",
        action="document_text_detection",
        request={"page": 1},
        response={"text_length": 120},
    )
    events = recorder.load_events()
    assert len(events) == 1
    assert events[0].service == "google_vision"
    assert events[0].request["page"] == 1


def test_build_html_document_includes_ocr_paragraphs() -> None:
    html = build_html_document(
        source_pdf="merged_notes.pdf",
        page_number=1,
        image_filename="../artifacts/page.png",
        width=800,
        height=1200,
        paragraphs=[
            {
                "block_index": 1,
                "paragraph_index": 1,
                "text": "Procedure note on page 1.",
                "bbox": {"left": 10, "top": 20, "width": 300, "height": 40},
            }
        ],
        full_text="Procedure note on page 1.",
    )

    assert "Google Vision OCR paragraphs" in html
    assert "Procedure note on page 1." in html
    assert "data-left=\"10\"" in html


def test_manifest_partial_run_flags_default_and_explicit_values() -> None:
    manifest = MaterializationManifest(
        run_id="run-1",
        source_kind="pdf_google_vision_html",
        total_pages=40,
        materialized_pages=1,
        requested_max_pages=1,
        is_partial_run=True,
    )

    assert manifest.requested_max_pages == 1
    assert manifest.is_partial_run is True
