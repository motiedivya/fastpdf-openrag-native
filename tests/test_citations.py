from __future__ import annotations

import json
from pathlib import Path

from fastpdf_openrag_native import service
from fastpdf_openrag_native.citations import ensure_summary_citations
from fastpdf_openrag_native.models import (
    EvidenceHit,
    MaterializationManifest,
    MaterializedPage,
    MaterializedRetrievalDocument,
    PageMapSummary,
    PageRef,
    ScopedSummaryResult,
    SummaryScope,
    VerifiedSentence,
)
from fastpdf_openrag_native.ocr_extract import build_html_document


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    extraction_dir = tmp_path / "data" / "extracted" / "run-1"
    output_dir = tmp_path / "outputs" / "run-1"
    pages_dir = extraction_dir / "pages"
    retrieval_dir = extraction_dir / "retrieval"
    artifacts_dir = extraction_dir / "artifacts"
    pages_dir.mkdir(parents=True, exist_ok=True)
    retrieval_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_pdf = tmp_path / "fixtures" / "source.pdf"
    source_pdf.parent.mkdir(parents=True, exist_ok=True)
    source_pdf.write_bytes(b"%PDF-1.4\n% citation test\n")

    page_image_path = artifacts_dir / "run-1__p0001.png"
    page_image_path.write_bytes(b"png")

    page_html_path = pages_dir / "run-1__p0001.html"
    page_html_path.write_text(
        build_html_document(
            source_pdf="source.pdf",
            page_number=1,
            image_filename="../artifacts/run-1__p0001.png",
            width=800,
            height=1200,
            paragraphs=[
                {
                    "block_index": 1,
                    "paragraph_index": 1,
                    "page_paragraph_index": 1,
                    "text": "Procedure note on page 1.",
                    "bbox": {"left": 10, "top": 20, "width": 320, "height": 40},
                },
                {
                    "block_index": 1,
                    "paragraph_index": 2,
                    "page_paragraph_index": 2,
                    "text": "Local anesthetic was used.",
                    "bbox": {"left": 12, "top": 72, "width": 340, "height": 42},
                },
            ],
            full_text="Procedure note on page 1. Local anesthetic was used.",
        ),
        encoding="utf-8",
    )

    chunk_filename = "run-1__p0001__c0001.md"
    chunk_path = retrieval_dir / chunk_filename
    chunk_path.write_text(
        "\n".join(
            [
                "# Retrieval Chunk",
                "## Evidence Text",
                "Procedure note on page 1.",
                "Local anesthetic was used.",
            ]
        ),
        encoding="utf-8",
    )

    page_hit = EvidenceHit(
        filename=chunk_filename,
        text="Procedure note on page 1. Local anesthetic was used.",
        score=0.91,
        page=1,
    )
    verified_sentences = [
        VerifiedSentence(sentence="Procedure note on page 1.", supported=True, evidence=[page_hit]),
        VerifiedSentence(sentence="Local anesthetic was used.", supported=True, evidence=[page_hit]),
    ]

    manifest = MaterializationManifest(
        run_id="run-1",
        source_kind="pdf_google_vision_html",
        total_pages=1,
        materialized_pages=1,
        retrieval_document_count=1,
        page_documents=[
            MaterializedPage(
                run_id="run-1",
                pdf_id="run-1-pdf",
                page=1,
                order_index=1,
                source_filename="run-1__p0001.html",
                relative_path="pages/run-1__p0001.html",
                text_length=52,
                text_preview="Procedure note on page 1. Local anesthetic was used.",
                artifacts={"page_image": "artifacts/run-1__p0001.png"},
                retrieval_filenames=[chunk_filename],
                retrieval_relative_paths=["retrieval/run-1__p0001__c0001.md"],
                metadata={"page_width": 800, "page_height": 1200, "paragraph_count": 2},
            )
        ],
        retrieval_documents=[
            MaterializedRetrievalDocument(
                run_id="run-1",
                pdf_id="run-1-pdf",
                page=1,
                order_index=1,
                chunk_index=1,
                source_filename=chunk_filename,
                relative_path="retrieval/run-1__p0001__c0001.md",
                text_length=52,
                text_preview="Procedure note on page 1. Local anesthetic was used.",
                parent_source_filename="run-1__p0001.html",
                metadata={
                    "block_start": 1,
                    "block_end": 1,
                    "paragraph_start": 1,
                    "paragraph_end": 2,
                    "page_paragraph_start": 1,
                    "page_paragraph_end": 2,
                    "paragraph_refs": [
                        {"block_index": 1, "paragraph_index": 1, "page_paragraph_index": 1},
                        {"block_index": 1, "paragraph_index": 2, "page_paragraph_index": 2},
                    ],
                },
            )
        ],
    )
    manifest_path = extraction_dir / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    summary = ScopedSummaryResult(
        run_id="run-1",
        scope=SummaryScope(
            scope_id="all-pages",
            title="All Pages",
            objective="Produce a grounded summary.",
            page_refs=[PageRef(pdf_id="run-1-pdf", page=1)],
        ),
        source_filenames=[chunk_filename],
        page_summaries=[
            PageMapSummary(
                pdf_id="run-1-pdf",
                page=1,
                source_filename="run-1__p0001.html",
                summary="Procedure note on page 1. Local anesthetic was used.",
                key_facts=["Local anesthetic was used."],
                raw_response="Procedure note on page 1. Local anesthetic was used.",
                retrieved_sources=[page_hit],
                verified_sentences=verified_sentences,
                supported_summary="Procedure note on page 1. Local anesthetic was used.",
                unsupported_sentences=["Page 1: Follow-up instructions were reviewed."],
                passed_verification=True,
            )
        ],
        draft_title="All Pages Summary",
        draft_summary="Procedure note on page 1; local anesthetic was used.",
        chronology=["Page 1: Procedure note on page 1.", "Page 1: Local anesthetic was used."],
        verified_sentences=verified_sentences,
        supported_summary="Procedure note on page 1. Local anesthetic was used.",
        unsupported_sentences=["Page 1: Follow-up instructions were reviewed."],
        debug={"verified_page_count": 1},
    )
    summary_path = output_dir / "all-pages.summary.json"
    summary_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    return manifest_path, summary_path, source_pdf, extraction_dir


def test_ensure_summary_citations_writes_artifacts_and_grounding(tmp_path: Path) -> None:
    manifest_path, summary_path, source_pdf, extraction_dir = _write_fixture(tmp_path)

    summary, citation_index_path, citation_instances_path, resolved_citations_path, source_pdf_copy_path = ensure_summary_citations(
        summary_path=summary_path,
        manifest_path=manifest_path,
        source_pdf=source_pdf,
    )

    assert citation_index_path.exists()
    assert citation_instances_path.exists()
    assert resolved_citations_path.exists()
    assert source_pdf_copy_path == extraction_dir / "artifacts" / source_pdf.name
    assert source_pdf_copy_path.exists()
    assert summary.resolved_citations is not None
    assert summary.citation_index
    assert summary.citation_instances

    first = summary.citation_index[0]
    assert first.chunk_id == "run-1__p0001__c0001.md"
    assert first.page == 1
    assert first.page_key == "run-1-pdf::1"
    assert first.page_source_filename == "run-1__p0001.html"
    assert len(first.boxes) == 1
    assert first.anchor == "run-1-pdf:1:p1-1"
    assert first.bbox == {"left": 10, "top": 20, "right": 330, "bottom": 60, "width": 320, "height": 40}
    assert first.degraded is False

    second = summary.citation_index[1]
    assert len(second.boxes) == 1
    assert second.anchor == "run-1-pdf:1:p2-2"
    assert second.bbox == {"left": 12, "top": 72, "right": 352, "bottom": 114, "width": 340, "height": 42}

    persisted_summary = ScopedSummaryResult.model_validate_json(summary_path.read_text(encoding="utf-8"))
    assert persisted_summary.citation_index == summary.citation_index
    assert persisted_summary.citation_instances == summary.citation_instances
    persisted_index = json.loads(citation_index_path.read_text(encoding="utf-8"))
    persisted_instances = json.loads(citation_instances_path.read_text(encoding="utf-8"))
    assert [row["number"] for row in persisted_index] == list(range(1, len(persisted_index) + 1))
    assert [row["number"] for row in persisted_instances] == list(range(1, len(persisted_instances) + 1))
    assert len(persisted_instances) >= 2
    assert len(persisted_instances) >= len(persisted_index)
    persisted_resolved = json.loads(resolved_citations_path.read_text(encoding="utf-8"))
    section_kinds = {section["kind"] for section in persisted_resolved["sections"]}
    assert {"supported_summary", "chronology", "page_summary", "draft_summary", "unsupported_summary"} <= section_kinds
    draft_section = next(section for section in persisted_resolved["sections"] if section["kind"] == "draft_summary")
    assert len(draft_section["items"]) == 2


def test_ensure_summary_citations_prefers_presentation_layer_sections(tmp_path: Path) -> None:
    manifest_path, summary_path, source_pdf, _extraction_dir = _write_fixture(tmp_path)
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["presentation_layer"] = {
        "title": "All Pages Summary",
        "narrative": "On 09/12/2018, the procedure note documented page 1 findings.",
        "sections": [
            {
                "section_id": "presentation-001",
                "title": "09/12/2018",
                "note_id": "note-001",
                "items": [
                    {
                        "item_id": "note-001__intro__01",
                        "text": "On 09/12/2018, the procedure note documented page 1 findings.",
                        "field_name": "intro",
                        "note_id": "note-001",
                        "evidence": [
                            {
                                "filename": "run-1__p0001__c0001.md",
                                "text": "Procedure note on page 1. Local anesthetic was used.",
                                "score": 0.91,
                                "page": 1,
                            }
                        ],
                        "candidate_filenames": ["run-1__p0001__c0001.md"],
                        "pdf_ids": ["run-1-pdf"],
                        "pages": [1],
                    }
                ],
            }
        ],
        "debug": {"note_count": 1},
    }
    payload["verified_sentences"] = []
    payload["chronology"] = []
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    summary, _citation_index_path, _citation_instances_path, _resolved_citations_path, _source_pdf_copy_path = ensure_summary_citations(
        summary_path=summary_path,
        manifest_path=manifest_path,
        source_pdf=source_pdf,
    )

    assert summary.resolved_citations is not None
    kinds = [section.kind for section in summary.resolved_citations.sections]
    assert "supported_summary" in kinds
    assert "page_summary" not in kinds
    assert summary.resolved_citations.sections[0].items[0].text.startswith("On 09/12/2018")


def test_build_summary_view_model_exposes_citation_urls_and_numbers(tmp_path: Path, monkeypatch) -> None:
    manifest_path, summary_path, source_pdf, _extraction_dir = _write_fixture(tmp_path)
    payload = {
        "status": "completed",
        "run_id": "run-1",
        "pdf_path": source_pdf.as_posix(),
        "question": "Summarize everything.",
        "max_pages": 1,
        "summary_path": summary_path.as_posix(),
        "manifest_path": manifest_path.as_posix(),
    }

    monkeypatch.setattr(service, "_debug_url_for_path", lambda value: f"/debug/{Path(value).name}" if value else None)

    model = service._build_summary_view_model("job-1", payload)

    assert model["job"]["job_id"] == "job-1"
    assert model["urls"]["trace"] == "/ui/runs/job-1/trace"
    assert model["urls"]["status"] == "/ui/runs/job-1"
    assert model["urls"]["summary_json"] == "/debug/all-pages.summary.json"
    assert model["urls"]["citation_index"] == "/debug/citation_index.json"
    assert model["urls"]["resolved_citations"] == "/debug/resolved_citations.json"
    assert model["urls"]["source_pdf"] == "/debug/source.pdf"
    assert payload["citation_index_path"].endswith("citation_index.json")
    assert payload["citation_instances_path"].endswith("sentence_citation_instances.json")
    assert payload["resolved_citations_path"].endswith("resolved_citations.json")
    assert payload["source_pdf_copy_path"].endswith("source.pdf")

    citation_numbers = {
        number
        for section in model["sections"]
        for item in section.get("items", [])
        for number in item.get("citation_numbers", [])
    }
    assert citation_numbers == set(range(1, len(model["citation_instances"]) + 1))
    assert all(entry["page_image_url"] == "/debug/run-1__p0001.png" for entry in model["citation_instances"])
    assert model["artifacts"]["citation_index"]["url"] == "/debug/citation_index.json"
    assert model["artifacts"]["citation_instances"]["url"] == "/debug/sentence_citation_instances.json"
    assert model["artifacts"]["resolved_citations"]["url"] == "/debug/resolved_citations.json"
    assert model["artifacts"]["source_pdf_copy"]["url"] == "/debug/source.pdf"


def test_ensure_summary_citations_scales_normalized_boxes(tmp_path: Path) -> None:
    extraction_dir = tmp_path / "data" / "extracted" / "run-norm"
    output_dir = tmp_path / "outputs" / "run-norm"
    pages_dir = extraction_dir / "pages"
    retrieval_dir = extraction_dir / "retrieval"
    artifacts_dir = extraction_dir / "artifacts"
    pages_dir.mkdir(parents=True, exist_ok=True)
    retrieval_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_pdf = tmp_path / "fixtures" / "source.pdf"
    source_pdf.parent.mkdir(parents=True, exist_ok=True)
    source_pdf.write_bytes(b"%PDF-1.4\n% citation test\n")

    page_html_path = pages_dir / "run-norm__p0001.html"
    page_html_path.write_text(
        build_html_document(
            source_pdf="source.pdf",
            page_number=1,
            image_filename="../artifacts/run-norm__p0001.png",
            width=1000,
            height=1000,
            paragraphs=[
                {
                    "block_index": 1,
                    "paragraph_index": 1,
                    "page_paragraph_index": 1,
                    "text": "Localized medial knee pain was documented.",
                    "bbox": {"left": 100, "top": 100, "width": 200, "height": 100},
                }
            ],
            full_text="Localized medial knee pain was documented.",
        ),
        encoding="utf-8",
    )

    chunk_filename = "run-norm__p0001__c0001.md"
    chunk_path = retrieval_dir / chunk_filename
    chunk_path.write_text(
        "\n".join(["# Retrieval Chunk", "## Evidence Text", "Localized medial knee pain was documented."]),
        encoding="utf-8",
    )

    hit = EvidenceHit(
        filename=chunk_filename,
        text="Localized medial knee pain was documented.",
        score=0.93,
        page=1,
    )

    manifest = MaterializationManifest(
        run_id="run-norm",
        source_kind="summary_payload",
        total_pages=1,
        materialized_pages=1,
        retrieval_document_count=1,
        page_documents=[
            MaterializedPage(
                run_id="run-norm",
                pdf_id="run-norm-pdf",
                page=1,
                order_index=1,
                source_filename="run-norm__p0001.html",
                relative_path="pages/run-norm__p0001.html",
                document_type="text/html",
                text_length=40,
                text_preview="Localized medial knee pain was documented.",
                retrieval_filenames=[chunk_filename],
                retrieval_relative_paths=["retrieval/run-norm__p0001__c0001.md"],
                metadata={
                    "paragraph_count": 1,
                    "bbox_space": "normalized_1000",
                    "page_width": 800,
                    "page_height": 1200,
                },
            )
        ],
        retrieval_documents=[
            MaterializedRetrievalDocument(
                run_id="run-norm",
                pdf_id="run-norm-pdf",
                page=1,
                order_index=1,
                chunk_index=1,
                source_filename=chunk_filename,
                relative_path="retrieval/run-norm__p0001__c0001.md",
                text_length=40,
                text_preview="Localized medial knee pain was documented.",
                parent_source_filename="run-norm__p0001.html",
                metadata={
                    "block_start": 1,
                    "block_end": 1,
                    "paragraph_start": 1,
                    "paragraph_end": 1,
                    "page_paragraph_start": 1,
                    "page_paragraph_end": 1,
                    "paragraph_refs": [
                        {"block_index": 1, "paragraph_index": 1, "page_paragraph_index": 1},
                    ],
                },
            )
        ],
    )
    manifest_path = extraction_dir / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    summary = ScopedSummaryResult(
        run_id="run-norm",
        scope=SummaryScope(
            scope_id="all-pages",
            title="All Pages",
            objective="Produce a grounded summary.",
            page_refs=[PageRef(pdf_id="run-norm-pdf", page=1)],
        ),
        source_filenames=[chunk_filename],
        page_summaries=[
            PageMapSummary(
                pdf_id="run-norm-pdf",
                page=1,
                source_filename="run-norm__p0001.html",
                summary="Localized medial knee pain was documented.",
                key_facts=[],
                raw_response="Localized medial knee pain was documented.",
                retrieved_sources=[hit],
                verified_sentences=[VerifiedSentence(sentence="Localized medial knee pain was documented.", supported=True, evidence=[hit])],
                supported_summary="Localized medial knee pain was documented.",
                passed_verification=True,
            )
        ],
        draft_title="All Pages Summary",
        draft_summary="Localized medial knee pain was documented.",
        chronology=[],
        verified_sentences=[VerifiedSentence(sentence="Localized medial knee pain was documented.", supported=True, evidence=[hit])],
        supported_summary="Localized medial knee pain was documented.",
    )
    summary_path = output_dir / "all-pages.summary.json"
    summary_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")

    resolved_summary, *_ = ensure_summary_citations(
        summary_path=summary_path,
        manifest_path=manifest_path,
        source_pdf=source_pdf,
    )

    first = resolved_summary.citation_index[0]
    assert first.degraded is False
    assert len(first.boxes) == 1
    assert first.bbox == {"left": 80, "top": 120, "right": 240, "bottom": 240, "width": 160, "height": 120}
