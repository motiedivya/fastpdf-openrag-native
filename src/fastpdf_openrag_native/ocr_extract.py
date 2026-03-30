from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import fitz
from google.cloud import vision
from google.protobuf.json_format import MessageToDict

from .chunking import blocks_from_ocr_paragraphs, build_structured_chunks, render_retrieval_markdown, text_to_blocks
from .models import MaterializationManifest, MaterializedPage, MaterializedRetrievalDocument
from .settings import AppSettings, get_settings
from .trace import TraceRecorder


def slugify_filename(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-") or "document"


def build_run_id(pdf_path: Path) -> str:
    return slugify_filename(pdf_path.stem.lower())


def _normalize_text(text: str) -> str:
    clean = text.replace("\u00a0", " ")
    clean = re.sub(r"\s+", " ", clean)
    return clean.strip()


def _bbox_from_vertices(vertices: list[Any]) -> tuple[int, int, int, int]:
    xs = [int(getattr(vertex, "x", 0) or 0) for vertex in vertices]
    ys = [int(getattr(vertex, "y", 0) or 0) for vertex in vertices]
    if not xs or not ys:
        return 0, 0, 0, 0
    return min(xs), min(ys), max(xs), max(ys)


def _paragraph_text(paragraph: Any) -> str:
    words: list[str] = []
    for word in paragraph.words:
        letters = "".join(symbol.text for symbol in word.symbols)
        if letters:
            words.append(letters)
        if word.symbols and getattr(word.symbols[-1].property, "detected_break", None):
            break_type = word.symbols[-1].property.detected_break.type_
            if break_type in (1, 3):
                words.append("\n")
    return _normalize_text(" ".join(words))


def _page_paragraphs(page_annotation: Any) -> list[dict[str, Any]]:
    paragraphs: list[dict[str, Any]] = []
    for block_index, block in enumerate(page_annotation.blocks, start=1):
        for paragraph_index, paragraph in enumerate(block.paragraphs, start=1):
            text = _paragraph_text(paragraph)
            if not text:
                continue
            left, top, right, bottom = _bbox_from_vertices(paragraph.bounding_box.vertices)
            paragraphs.append(
                {
                    "block_index": block_index,
                    "paragraph_index": paragraph_index,
                    "text": text,
                    "bbox": {
                        "left": left,
                        "top": top,
                        "right": right,
                        "bottom": bottom,
                        "width": max(0, right - left),
                        "height": max(0, bottom - top),
                    },
                }
            )
    return paragraphs


def build_html_document(
    *,
    source_pdf: str,
    page_number: int,
    image_filename: str,
    width: int,
    height: int,
    paragraphs: list[dict[str, Any]],
    full_text: str,
) -> str:
    paragraph_html = []
    for item in paragraphs:
        bbox = item["bbox"]
        paragraph_html.append(
            (
                "<p "
                f"data-block=\"{item['block_index']}\" "
                f"data-paragraph=\"{item['paragraph_index']}\" "
                f"data-left=\"{bbox['left']}\" "
                f"data-top=\"{bbox['top']}\" "
                f"data-width=\"{bbox['width']}\" "
                f"data-height=\"{bbox['height']}\""
                f">{item['text']}</p>"
            )
        )

    return "\n".join(
        [
            "<!DOCTYPE html>",
            "<html lang=\"en\">",
            "<head>",
            "  <meta charset=\"utf-8\" />",
            f"  <title>{source_pdf} page {page_number}</title>",
            "  <style>",
            "    body { font-family: Georgia, serif; margin: 24px; color: #1b1b1b; }",
            "    header { margin-bottom: 16px; }",
            "    .page-image img { max-width: 100%; border: 1px solid #d9d9d9; }",
            "    .ocr-paragraphs { margin-top: 24px; }",
            "    .ocr-paragraphs p { margin: 0 0 12px; line-height: 1.45; }",
            "    .raw-text { margin-top: 24px; padding: 12px; background: #f7f3eb; white-space: pre-wrap; }",
            "  </style>",
            "</head>",
            "<body>",
            "  <header>",
            f"    <h1>{source_pdf}</h1>",
            f"    <p>Page {page_number} rendered size: {width}x{height}px</p>",
            "  </header>",
            "  <section class=\"page-image\">",
            f"    <img src=\"{image_filename}\" alt=\"{source_pdf} page {page_number}\" />",
            "  </section>",
            "  <section class=\"ocr-paragraphs\">",
            "    <h2>Google Vision OCR paragraphs</h2>",
            *[f"    {row}" for row in paragraph_html],
            "  </section>",
            "  <section>",
            "    <h2>Google Vision full text</h2>",
            f"    <div class=\"raw-text\">{full_text}</div>",
            "  </section>",
            "</body>",
            "</html>",
        ]
    )


def _build_client(credentials_path: Path | None) -> vision.ImageAnnotatorClient:
    if credentials_path:
        return vision.ImageAnnotatorClient.from_service_account_file(credentials_path.as_posix())
    return vision.ImageAnnotatorClient()


def extract_pdf_to_html(
    pdf_path: Path,
    *,
    output_dir: Path,
    trace: TraceRecorder,
    settings: AppSettings | None = None,
    credentials_path: Path | None = None,
    max_pages: int | None = None,
    run_id: str | None = None,
) -> MaterializationManifest:
    effective_settings = settings or get_settings()
    resolved_credentials = credentials_path or effective_settings.google_application_credentials
    if not resolved_credentials:
        raise ValueError(
            "Google Vision credentials are required. Set GOOGLE_APPLICATION_CREDENTIALS or "
            "FASTPDF_OPENRAG_GOOGLE_APPLICATION_CREDENTIALS."
        )
    if not resolved_credentials.exists():
        raise FileNotFoundError(f"Google Vision credentials file not found: {resolved_credentials}")

    output_dir.mkdir(parents=True, exist_ok=True)
    pages_dir = output_dir / "pages"
    artifacts_dir = output_dir / "artifacts"
    retrieval_dir = output_dir / "retrieval"
    pages_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    retrieval_dir.mkdir(parents=True, exist_ok=True)

    run_id = run_id or build_run_id(pdf_path)
    client = _build_client(resolved_credentials)
    document = fitz.open(pdf_path)
    page_limit = min(document.page_count, max_pages) if max_pages else document.page_count

    trace.record(
        stage="extract",
        service="pymupdf",
        action="open_pdf",
        request={"pdf_path": pdf_path.as_posix()},
        response={"page_count": document.page_count},
        metrics={"page_limit": page_limit, "render_dpi": effective_settings.pdf_render_dpi},
        notes=(
            [
                "PyMuPDF is used only for PDF rendering. Google Vision is the only OCR/text source emitted by "
                "this pipeline.",
                (
                    f"Partial extraction requested via max_pages={max_pages}; only the first {page_limit} page(s) "
                    f"out of {document.page_count} will be materialized and ingested."
                ),
            ]
            if max_pages and page_limit < document.page_count
            else [
                "PyMuPDF is used only for PDF rendering. Google Vision is the only OCR/text source emitted by "
                "this pipeline."
            ]
        ),
    )

    page_documents: list[MaterializedPage] = []
    retrieval_documents: list[MaterializedRetrievalDocument] = []
    zoom = effective_settings.pdf_render_dpi / 72
    matrix = fitz.Matrix(zoom, zoom)

    for page_index in range(page_limit):
        page_number = page_index + 1
        page = document.load_page(page_index)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        image_filename = f"{run_id}__p{page_number:04d}.png"
        image_path = artifacts_dir / image_filename
        pix.save(image_path.as_posix())

        trace.record(
            stage="extract",
            service="pymupdf",
            action="render_page_image",
            request={"page": page_number},
            response={"image_path": image_path.as_posix()},
            metrics={"width": pix.width, "height": pix.height},
            output_files=[image_path.as_posix()],
            notes=["Rendered page image for Google Vision OCR. No native PyMuPDF text is emitted or ingested."],
        )

        with image_path.open("rb") as handle:
            image_bytes = handle.read()

        response = client.document_text_detection(image=vision.Image(content=image_bytes))
        if response.error.message:
            trace.record(
                stage="ocr",
                service="google_vision",
                action="document_text_detection",
                status="error",
                request={"page": page_number, "image_path": image_path.as_posix()},
                response={"error": response.error.message},
            )
            raise RuntimeError(f"Google Vision OCR failed for page {page_number}: {response.error.message}")

        response_dict = MessageToDict(response._pb, preserving_proto_field_name=True)
        raw_json_path = artifacts_dir / f"{run_id}__p{page_number:04d}.google_vision.json"
        raw_json_path.write_text(
            json.dumps(response_dict, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        full_text_annotation = response.full_text_annotation
        annotation_pages = list(full_text_annotation.pages)
        page_annotation = annotation_pages[0] if annotation_pages else None
        paragraphs = _page_paragraphs(page_annotation) if page_annotation else []
        full_text = _normalize_text(full_text_annotation.text or "")

        text_path = artifacts_dir / f"{run_id}__p{page_number:04d}.txt"
        text_path.write_text(full_text, encoding="utf-8")

        html_filename = f"{run_id}__p{page_number:04d}.html"
        html_path = pages_dir / html_filename
        html_path.write_text(
            build_html_document(
                source_pdf=pdf_path.name,
                page_number=page_number,
                image_filename=f"../artifacts/{image_filename}",
                width=pix.width,
                height=pix.height,
                paragraphs=paragraphs,
                full_text=full_text,
            ),
            encoding="utf-8",
        )

        blocks = blocks_from_ocr_paragraphs(paragraphs) or text_to_blocks(
            full_text,
            target_chars=effective_settings.structure_chunk_target_chars,
        )
        chunks = build_structured_chunks(
            blocks,
            target_chars=effective_settings.structure_chunk_target_chars,
            overlap_blocks=effective_settings.structure_chunk_overlap_blocks,
        )
        retrieval_filenames: list[str] = []
        retrieval_relative_paths: list[str] = []
        for chunk in chunks:
            retrieval_filename = f"{run_id}__p{page_number:04d}__c{chunk.chunk_index:04d}.md"
            retrieval_path = retrieval_dir / retrieval_filename
            retrieval_path.write_text(
                render_retrieval_markdown(
                    run_id=run_id,
                    pdf_id=pdf_path.name,
                    page=page_number,
                    page_source_filename=html_filename,
                    chunk=chunk,
                    chunk_total=len(chunks),
                ),
                encoding="utf-8",
            )
            retrieval_filenames.append(retrieval_filename)
            retrieval_relative_paths.append(retrieval_path.relative_to(output_dir).as_posix())
            retrieval_documents.append(
                MaterializedRetrievalDocument(
                    run_id=run_id,
                    pdf_id=pdf_path.name,
                    page=page_number,
                    order_index=page_index,
                    chunk_index=chunk.chunk_index,
                    source_filename=retrieval_filename,
                    relative_path=retrieval_path.relative_to(output_dir).as_posix(),
                    text_length=len(chunk.text),
                    text_preview=chunk.text[:240],
                    parent_source_filename=html_filename,
                    section_title=chunk.section_title,
                    source_fields=["google_vision_document_text_detection"],
                    metadata={
                        "structure_chunking_strategy": "structure_aware_blocks",
                        "ocr_provider": "google_vision_document_text_detection",
                        **chunk.metadata,
                    },
                )
            )

        trace.record(
            stage="materialize",
            service="fastpdf-openrag-native",
            action="build_retrieval_chunks",
            request={"page": page_number, "source_filename": html_filename},
            response={
                "retrieval_chunk_count": len(chunks),
                "target_chars": effective_settings.structure_chunk_target_chars,
                "overlap_blocks": effective_settings.structure_chunk_overlap_blocks,
            },
            output_files=[(retrieval_dir / name).as_posix() for name in retrieval_filenames],
        )

        trace.record(
            stage="ocr",
            service="google_vision",
            action="document_text_detection",
            request={
                "page": page_number,
                "image_path": image_path.as_posix(),
                "credential_path": resolved_credentials.as_posix(),
            },
            response={
                "text_path": text_path.as_posix(),
                "raw_json_path": raw_json_path.as_posix(),
                "paragraph_count": len(paragraphs),
            },
            metrics={"text_length": len(full_text)},
            output_files=[raw_json_path.as_posix(), text_path.as_posix()],
            notes=["Google Vision OCR output is the only text source written for this page."],
        )

        page_documents.append(
            MaterializedPage(
                run_id=run_id,
                pdf_id=pdf_path.name,
                page=page_number,
                order_index=page_index,
                source_filename=html_filename,
                relative_path=html_path.relative_to(output_dir).as_posix(),
                document_type="text/html",
                label="google_vision_html_page",
                text_length=len(full_text),
                text_preview=full_text[:240],
                source_fields=["google_vision_document_text_detection"],
                artifacts={
                    "page_image": image_path.relative_to(output_dir).as_posix(),
                    "ocr_json": raw_json_path.relative_to(output_dir).as_posix(),
                    "ocr_text": text_path.relative_to(output_dir).as_posix(),
                },
                retrieval_filenames=retrieval_filenames,
                retrieval_relative_paths=retrieval_relative_paths,
                metadata={
                    "source_pdf": pdf_path.as_posix(),
                    "page_width": pix.width,
                    "page_height": pix.height,
                    "paragraph_count": len(paragraphs),
                    "ocr_provider": "google_vision_document_text_detection",
                },
            )
        )

    manifest = MaterializationManifest(
        run_id=run_id,
        source_kind="pdf_google_vision_html",
        patient_name=None,
        total_pages=document.page_count,
        materialized_pages=len(page_documents),
        retrieval_document_count=len(retrieval_documents),
        requested_max_pages=max_pages,
        is_partial_run=page_limit < document.page_count,
        page_documents=page_documents,
        retrieval_documents=retrieval_documents,
    )
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    trace.record(
        stage="extract",
        service="fastpdf-openrag-native",
        action="write_manifest",
        response={"manifest_path": manifest_path.as_posix()},
        metrics={"materialized_pages": manifest.materialized_pages},
        output_files=[manifest_path.as_posix()],
        notes=(
            [
                f"Manifest is partial: materialized {manifest.materialized_pages} of {manifest.total_pages} total pages."
            ]
            if manifest.is_partial_run
            else []
        ),
    )
    return manifest
