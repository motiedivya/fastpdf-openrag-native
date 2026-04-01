from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from fastpdf_openrag_native.fastpdf_loader import (
    _should_rebuild_from_best_text_guardrail,
    materialize_summary_payload,
)
from fastpdf_openrag_native.settings import AppSettings
from fastpdf_openrag_native.summarizer import load_manifest, load_scopes


def test_materialize_summary_payload_writes_page_docs(tmp_path: Path) -> None:
    payload = {
        "pdfs": [
            {
                "pdf_id": "alpha.pdf",
                "pages": [
                    {"page": 1, "pdf2html_text": "Procedure note page 1", "model_label": "Operative_Report"},
                    {"page": 2, "ocr_text": "Google Vision Extract\nProcedure note page 2", "model_label": "Operative_Report"},
                    {"page": 3, "is_blank": True, "ocr_text": "Should be skipped"},
                ],
            }
        ]
    }

    manifest = materialize_summary_payload(
        run_id="run-1",
        summary_payload=payload,
        source_kind="summary_payload",
        output_dir=tmp_path,
    )

    assert manifest.materialized_pages == 2
    manifest_path = tmp_path / "manifest.json"
    assert manifest_path.exists()

    first_page = tmp_path / manifest.page_documents[0].relative_path
    second_page = tmp_path / manifest.page_documents[1].relative_path
    assert "Procedure note page 1" in first_page.read_text(encoding="utf-8")
    assert "Google Vision Extract" not in second_page.read_text(encoding="utf-8")
    assert manifest.retrieval_document_count == len(manifest.retrieval_documents)
    assert manifest.page_documents[0].retrieval_filenames


def test_materialize_summary_payload_builds_structure_aware_retrieval_docs(tmp_path: Path) -> None:
    payload = {
        "pdfs": [
            {
                "pdf_id": "alpha.pdf",
                "pages": [
                    {
                        "page": 1,
                        "ocr_text": (
                            "HISTORY:\n\n"
                            "Patient reports ongoing left knee pain and swelling after a fall. "
                            "Pain worsens with stairs and prolonged standing.\n\n"
                            "IMAGING:\n\n"
                            "MRI shows medial meniscus tear with joint effusion and marrow edema.\n\n"
                            "PLAN:\n\n"
                            "Recommend orthopedic follow-up, brace use, and physical therapy."
                        ),
                    }
                ],
            }
        ]
    }

    manifest = materialize_summary_payload(
        run_id="run-structured",
        summary_payload=payload,
        source_kind="summary_payload",
        output_dir=tmp_path,
        settings=AppSettings(
            structure_chunk_target_chars=120,
            structure_chunk_overlap_blocks=0,
        ),
    )

    assert manifest.retrieval_document_count >= 2
    assert len(manifest.page_documents[0].retrieval_filenames) >= 2
    first_chunk = tmp_path / manifest.retrieval_documents[0].relative_path
    assert "FastPDF Retrieval Chunk" in first_chunk.read_text(encoding="utf-8")


def test_load_scopes_accepts_monitor_style_payload(tmp_path: Path) -> None:
    scope_file = tmp_path / "scope.json"
    scope_file.write_text(
        json.dumps(
            {
                "scopes": [
                    {
                        "scope_id": "operative-report",
                        "label": "Operative Report",
                        "instructions": "Focus on procedure details.",
                        "page_refs": [{"pdf_id": "alpha.pdf", "page": 2}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    scopes = load_scopes(scope_file)
    assert scopes[0].scope_id == "operative-report"
    assert scopes[0].title == "Operative Report"
    assert scopes[0].objective == "Focus on procedure details."


def test_load_manifest_round_trips(tmp_path: Path) -> None:
    payload = {
        "pdfs": [
            {
                "pdf_id": "alpha.pdf",
                "pages": [{"page": 1, "ocr_text": "Page one"}],
            }
        ]
    }
    materialize_summary_payload(
        run_id="run-2",
        summary_payload=payload,
        source_kind="summary_payload",
        output_dir=tmp_path,
    )
    manifest = load_manifest(tmp_path / "manifest.json")
    assert manifest.run_id == "run-2"
    assert manifest.page_documents[0].pdf_id == "alpha.pdf"
    assert manifest.retrieval_document_count == len(manifest.retrieval_documents)


def test_materialize_summary_payload_prefers_labeled_service_date_over_dob(tmp_path: Path) -> None:
    payload = {
        "pdfs": [
            {
                "pdf_id": "alpha.pdf",
                "pages": [
                    {
                        "page": 1,
                        "ocr_text": "DOB: 11/13/1958 Date of Service: 8/16/2024 Procedure note.",
                    }
                ],
            }
        ]
    }

    manifest = materialize_summary_payload(
        run_id="run-service-date",
        summary_payload=payload,
        source_kind="summary_payload",
        output_dir=tmp_path,
    )

    assert manifest.page_documents[0].service_date == "8/16/2024"


def test_materialize_summary_payload_uses_ocr_blocks_for_page_assets_and_chunk_metadata(tmp_path: Path) -> None:
    payload = {
        "pdfs": [
            {
                "pdf_id": "alpha.pdf",
                "pages": [
                    {
                        "page": 1,
                        "pdf2html_text": "Header line\n\nLocalized medial knee pain was documented.\n\nPhysical therapy was recommended.",
                        "ocr_blocks": [
                            {"text": "Localized medial knee pain was documented.", "bbox": [100, 100, 300, 180]},
                            {"text": "Physical therapy was recommended.", "bbox": [100, 220, 320, 300]},
                        ],
                    }
                ],
            }
        ]
    }

    manifest = materialize_summary_payload(
        run_id="run-ocr-blocks",
        summary_payload=payload,
        source_kind="summary_payload",
        output_dir=tmp_path,
    )

    page = manifest.page_documents[0]
    assert page.document_type == "text/html"
    assert page.relative_path.endswith(".html")
    assert page.metadata["paragraph_count"] == 2
    assert page.metadata["bbox_space"] == "normalized_1000"
    assert "ocr_blocks" in page.source_fields
    assert page.artifacts["page_markdown"].endswith(".md")

    html_path = tmp_path / page.relative_path
    assert html_path.exists()
    html_text = html_path.read_text(encoding="utf-8")
    assert 'data-page-paragraph="1"' in html_text
    assert 'data-left="100"' in html_text

    first_chunk = manifest.retrieval_documents[0]
    assert first_chunk.parent_source_filename == page.source_filename
    assert first_chunk.metadata["structure_chunking_strategy"] == "ocr_paragraph_blocks"
    assert first_chunk.metadata["page_paragraph_start"] == 1
    assert first_chunk.metadata["page_paragraph_end"] >= 1
    assert first_chunk.metadata["paragraph_refs"][0]["page_paragraph_index"] == 1


def test_native_text_guardrail_rebuilds_single_header_chunk_inventory() -> None:
    should_rebuild, reason = _should_rebuild_from_best_text_guardrail(
        best_text=(
            "HISTORY OF PRESENTING COMPLAINT: She rates her pain as 8/10 and reports increased pain with bending. "
            "She states that the pain is frequent and sharp and improved with injections and therapy. "
            "PATIENT HISTORY: Treatment since current injury includes St. Luke's South, Acute Accident Urgent Care, and Acute Spinal Rehab. "
            "Past Medical History includes hypertension and high cholesterol. Past Surgical History includes gallbladder, hysterectomy, and lap band. "
            "Current Medications include Aleve. Medication Allergies: NKDA. "
            "ASSESSMENT: V89.2 injured in MVA traffic, M54.2 cervicalgia, M54.5 low back pain, and M25.562 pain in left knee."
        ),
        source_fields=["rich_text", "ocr_blocks"],
        chunking_strategy="ocr_paragraph_blocks",
        chunks=[
            SimpleNamespace(
                text="Provider: Everett Wilkinson, DO Date of Service: 8/16/2024 Phone: 816-216-7054 Fax: 816-216-6010 File #: 18995"
            )
        ],
    )

    assert should_rebuild is True
    assert reason == "born_digital_single_header_chunk"


def test_materialize_summary_payload_prefers_best_text_chunks_when_ocr_paragraphs_are_sparse(tmp_path: Path) -> None:
    payload = {
        "pdfs": [
            {
                "pdf_id": "alpha.pdf",
                "pages": [
                    {
                        "page": 1,
                        "rich_text": """Spine and Joint Centers of Missouri
Provider: Everett Wilkinson, DO
Date of Service: 8/16/2024

HISTORY OF PRESENTING COMPLAINT:
She rates her pain as a 8 out of 10. The patient notes this pain as frequent and sharp in nature.
She reports increased pain with bending. She states that her pain is improved with injections and therapy.

PATIENT HISTORY:
Past Medical History: Positive and includes hypertension and high cholesterol.
Past Surgical History: Positive and includes gallbladder, hysterectomy and lap band.
Current Medications: Positive and includes Aleve.
Medication Allergies: NKDA.

ASSESSMENT:
V89.2 Injured in MVA traffic, M54.2 Cervicalgia, M54.5 Low back pain, M25.562 Pain in left knee.
""",
                        "ocr_blocks": [
                            {
                                "text": "Spine and Joint Centers of Missouri Provider: Everett Wilkinson, DO File #: 18995 Date of Service: 8/16/2024",
                                "bbox": [20, 25, 850, 127],
                            },
                            {"text": "Jorgensen, Colleen", "bbox": [413, 199, 586, 218]},
                        ],
                    }
                ],
            }
        ]
    }

    manifest = materialize_summary_payload(
        run_id="run-native-first",
        summary_payload=payload,
        source_kind="summary_payload",
        output_dir=tmp_path,
        settings=AppSettings(
            structure_chunk_target_chars=180,
            structure_chunk_overlap_blocks=0,
        ),
    )

    page = manifest.page_documents[0]
    assert page.document_type == "text/html"
    assert page.metadata["paragraph_count"] == 2
    assert page.metadata["ocr_paragraph_count"] == 2
    assert page.metadata["chunk_text_strategy"] == "best_text"
    assert page.metadata["source_text_strategy"] == "best_text_with_ocr_assets"
    assert page.metadata["native_text_chars"] > 300
    assert page.metadata["indexed_chunk_count"] >= 3
    assert page.metadata["chunk_previews"]
    assert page.metadata["native_text_guardrail_triggered"] is False
    assert manifest.retrieval_document_count >= 3

    retrieval_texts = [
        (tmp_path / row.relative_path).read_text(encoding="utf-8")
        for row in manifest.retrieval_documents
    ]
    assert any("HISTORY OF PRESENTING COMPLAINT" in chunk for chunk in retrieval_texts)
    assert any("hypertension and high cholesterol" in chunk for chunk in retrieval_texts)
    assert any("V89.2 Injured in MVA traffic" in chunk for chunk in retrieval_texts)
    assert all(row.metadata["structure_chunking_strategy"] == "structure_aware_blocks" for row in manifest.retrieval_documents)


def test_materialize_summary_payload_merges_word_level_ocr_blocks_into_highlightable_sections(tmp_path: Path) -> None:
    payload = {
        "pdfs": [
            {
                "pdf_id": "alpha.pdf",
                "pages": [
                    {
                        "page": 1,
                        "pdf2html_text": (
                            "Chief complaint was frequent pain. Pain worsened with bending.\n\n"
                            "Plan: Physical therapy."
                        ),
                        "ocr_blocks": [
                            {"text": "Chief", "bbox": [100, 100, 145, 120]},
                            {"text": "complaint", "bbox": [150, 100, 245, 120]},
                            {"text": "was", "bbox": [250, 100, 282, 120]},
                            {"text": "frequent", "bbox": [287, 100, 360, 120]},
                            {"text": "pain.", "bbox": [365, 100, 410, 120]},
                            {"text": "Pain", "bbox": [100, 128, 138, 148]},
                            {"text": "worsened", "bbox": [143, 128, 225, 148]},
                            {"text": "with", "bbox": [230, 128, 264, 148]},
                            {"text": "bending.", "bbox": [269, 128, 344, 148]},
                            {"text": "Plan:", "bbox": [100, 196, 145, 216]},
                            {"text": "Physical", "bbox": [100, 226, 176, 246]},
                            {"text": "therapy.", "bbox": [181, 226, 252, 246]},
                        ],
                    }
                ],
            }
        ]
    }

    manifest = materialize_summary_payload(
        run_id="run-word-ocr",
        summary_payload=payload,
        source_kind="summary_payload",
        output_dir=tmp_path,
    )

    page = manifest.page_documents[0]
    assert page.document_type == "text/html"
    assert page.metadata["paragraph_count"] == 3

    html_path = tmp_path / page.relative_path
    html_text = html_path.read_text(encoding="utf-8")
    assert "Chief complaint was frequent pain. Pain worsened with bending." in html_text
    assert html_text.count("data-page-paragraph=") == 3

    retrieval_chunks = manifest.retrieval_documents
    assert retrieval_chunks
    assert retrieval_chunks[0].metadata["page_paragraph_start"] == 1
    assert retrieval_chunks[0].metadata["page_paragraph_end"] >= 1
    assert retrieval_chunks[0].metadata["paragraph_refs"][0]["page_paragraph_index"] == 1
