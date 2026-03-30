from __future__ import annotations

from pathlib import Path

from fastpdf_openrag_native.models import MaterializationManifest
from fastpdf_openrag_native.ocr_extract import (
    _build_indexable_full_text,
    _prepare_paragraphs_for_indexing,
    build_html_document,
)
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
                "page_paragraph_index": 7,
                "text": "Procedure note on page 1.",
                "bbox": {"left": 10, "top": 20, "width": 300, "height": 40},
            }
        ],
        full_text="Procedure note on page 1.",
    )

    assert "Google Vision OCR paragraphs" in html
    assert "Procedure note on page 1." in html
    assert "data-left=\"10\"" in html
    assert "data-page-paragraph=\"7\"" in html


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



def test_prepare_paragraphs_for_indexing_filters_noise_and_normalizes_common_ocr_errors() -> None:
    paragraphs = [
        {
            "block_index": 1,
            "paragraph_index": 1,
            "page_paragraph_index": 1,
            "text": "From SiliconMesa fax server page 1 of 2",
            "bbox": {"left": 0, "top": 0, "width": 100, "height": 10},
        },
        {
            "block_index": 1,
            "paragraph_index": 2,
            "page_paragraph_index": 2,
            "text": "OTC Alive twice daily",
            "bbox": {"left": 0, "top": 12, "width": 160, "height": 10},
        },
        {
            "block_index": 1,
            "paragraph_index": 3,
            "page_paragraph_index": 3,
            "text": "OTC Alive twice daily",
            "bbox": {"left": 0, "top": 24, "width": 160, "height": 10},
        },
        {
            "block_index": 1,
            "paragraph_index": 4,
            "page_paragraph_index": 4,
            "text": "D0B 01/05/2021 DOS 01/06/2021",
            "bbox": {"left": 0, "top": 36, "width": 180, "height": 10},
        },
    ]

    prepared = _prepare_paragraphs_for_indexing(paragraphs)
    full_text = _build_indexable_full_text(prepared, "")

    assert [row["text"] for row in prepared] == [
        "OTC Aleve twice daily",
        "DOB 01/05/2021 DOS 01/06/2021",
    ]
    assert "SiliconMesa" not in full_text
    assert "OTC Aleve" in full_text
