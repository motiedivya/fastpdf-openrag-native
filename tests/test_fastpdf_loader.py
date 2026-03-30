from __future__ import annotations

import json
from pathlib import Path

from fastpdf_openrag_native.fastpdf_loader import materialize_summary_payload
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
