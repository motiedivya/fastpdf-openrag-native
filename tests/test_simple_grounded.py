from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastpdf_openrag_native.simple_grounded import generate_grounded_response, load_scope_chunks


class FakeGateway:
    async def chat_request(self, **_: object) -> dict[str, object]:
        return {
            "response": "- Patient name is Maria Thompson.",
            "chat_id": "chat-test-1",
        }

    def export_debug_events(self) -> list[dict[str, object]]:
        return []


def _write_manifest(root: Path) -> Path:
    pages_dir = root / "pages"
    retrieval_dir = root / "retrieval"
    pages_dir.mkdir(parents=True, exist_ok=True)
    retrieval_dir.mkdir(parents=True, exist_ok=True)

    html_path = pages_dir / "pdf_1__p0001.html"
    html_path.write_text(
        """
<!DOCTYPE html>
<html><body>
  <p data-block="1" data-paragraph="1" data-page-paragraph="1" data-left="10" data-top="20" data-width="110" data-height="28">Patient Name: Maria Thompson.</p>
  <p data-block="1" data-paragraph="2" data-page-paragraph="2" data-left="10" data-top="60" data-width="180" data-height="28">Assessment: Knee pain.</p>
</body></html>
        """.strip(),
        encoding="utf-8",
    )

    chunk_path = retrieval_dir / "pdf_1__p0001__c0001.md"
    chunk_path.write_text(
        """
# FastPDF Retrieval Chunk

- PDF ID: pdf_1
- Page: 1
- Retrieval chunk: 1 of 1

## Evidence Text

Patient Name: Maria Thompson.
        """.strip(),
        encoding="utf-8",
    )

    manifest = {
        "run_id": "run-test-1",
        "source_kind": "batch_monitor_tool",
        "total_pages": 1,
        "materialized_pages": 1,
        "retrieval_document_count": 1,
        "page_documents": [
            {
                "run_id": "run-test-1",
                "pdf_id": "pdf_1",
                "page": 1,
                "order_index": 1,
                "source_filename": "pdf_1__p0001.html",
                "relative_path": "pages/pdf_1__p0001.html",
                "document_type": "text/html",
                "text_length": 52,
                "text_preview": "Patient Name: Maria Thompson. Assessment: Knee pain.",
                "source_fields": ["ocr_blocks"],
                "artifacts": {},
                "retrieval_filenames": ["pdf_1__p0001__c0001.md"],
                "retrieval_relative_paths": ["retrieval/pdf_1__p0001__c0001.md"],
                "metadata": {
                    "page_width": 1000,
                    "page_height": 1000,
                    "bbox_space": "normalized_1000",
                },
            }
        ],
        "retrieval_documents": [
            {
                "run_id": "run-test-1",
                "pdf_id": "pdf_1",
                "page": 1,
                "order_index": 1,
                "chunk_index": 1,
                "source_filename": "pdf_1__p0001__c0001.md",
                "relative_path": "retrieval/pdf_1__p0001__c0001.md",
                "document_type": "text/markdown",
                "text_length": 29,
                "text_preview": "Patient Name: Maria Thompson.",
                "parent_source_filename": "pdf_1__p0001.html",
                "section_title": None,
                "source_fields": ["ocr_blocks"],
                "metadata": {
                    "paragraph_refs": [
                        {
                            "block_index": 1,
                            "paragraph_index": 1,
                            "page_paragraph_index": 1,
                        }
                    ]
                },
            }
        ],
    }

    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_load_scope_chunks_keeps_chunk_rectangles(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path)
    chunks, page_sources, _manifest = load_scope_chunks(manifest_path=manifest_path, page_refs=[("pdf_1", 1)])

    assert len(chunks) == 1
    assert len(chunks[0].rects) == 2
    assert chunks[0].rects[0].left == 10
    assert chunks[0].rects[0].top == 20
    assert page_sources[("pdf_1", 1)].page_width == 1000


def test_generate_grounded_response_returns_inline_citations_and_boxes(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path)
    result = asyncio.run(
        generate_grounded_response(
            FakeGateway(),
            manifest_path=manifest_path,
            message="Patient name?",
            title="Summary",
            page_refs=[("pdf_1", 1)],
            history=[],
            max_context_chunks=4,
        )
    )

    assert result["answer"].endswith("[1]")
    citations = result["citations"]
    assert len(citations) == 1
    assert citations[0]["pdf_id"] == "pdf_1"
    assert citations[0]["page"] == 1
    assert citations[0]["boxes"][0]["bbox"]["left"] == 10
    assert citations[0]["number"] == 1
    assert result["native_summary_data"]["sections"][0]["items"][0]["citation_numbers"] == [1]
