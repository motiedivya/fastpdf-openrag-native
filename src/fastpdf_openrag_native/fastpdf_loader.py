from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pymongo import MongoClient

from .chunking import build_structured_chunks, render_retrieval_markdown, text_to_blocks
from .models import MaterializationManifest, MaterializedPage, MaterializedRetrievalDocument
from .settings import AppSettings, get_settings

TEXT_FIELD_PRIORITY = (
    "pdf2html_text",
    "rich_text",
    "page_html_text",
    "text_layer_text",
    "ocr_text",
    "native_text",
    "text",
)

INLINE_NOISE_PATTERNS = (
    r"google\s+vision\s+extract",
    r"^ocr\s+output$",
)


TEXT_PREVIEW_CHARS = 1200


def _clean_scalar(value: Any) -> str:
    text = str(value or "")
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\r\n?", "\n", text)
    return text.strip()


def _normalize_text(value: Any) -> str:
    text = _clean_scalar(value)
    if not text:
        return ""

    for pattern in INLINE_NOISE_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE | re.MULTILINE)

    lines: list[str] = []
    previous = ""
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue
        if line.lower() == previous.lower():
            continue
        previous = line
        lines.append(line)

    if lines:
        return "\n".join(lines)
    return re.sub(r"\s+", " ", text).strip()


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except Exception:
        return None
    if parsed <= 0:
        return None
    return parsed


def _extract_best_page_text(page_row: dict[str, Any]) -> tuple[str, list[str]]:
    collected_fields: list[str] = []
    for field_name in TEXT_FIELD_PRIORITY:
        text = _normalize_text(page_row.get(field_name))
        if text:
            collected_fields.append(field_name)
            return text, collected_fields
    return "", collected_fields


def _extract_patient_name(payload: dict[str, Any]) -> str | None:
    candidates: dict[str, int] = {}
    pattern = re.compile(r"\b(?:patient\s+name|name)\s*:\s*([A-Z][A-Za-z' -]{1,80})", re.IGNORECASE)
    for pdf_row in payload.get("pdfs") if isinstance(payload.get("pdfs"), list) else []:
        if not isinstance(pdf_row, dict):
            continue
        for page_row in pdf_row.get("pages") if isinstance(pdf_row.get("pages"), list) else []:
            if not isinstance(page_row, dict):
                continue
            text, _ = _extract_best_page_text(page_row)
            if not text:
                continue
            for match in pattern.finditer(text):
                name = re.sub(r"\s+", " ", match.group(1)).strip(" .,:;")
                if len(name) < 3:
                    continue
                candidates[name] = candidates.get(name, 0) + 1
    if not candidates:
        return None
    return sorted(candidates.items(), key=lambda item: (-item[1], item[0].lower()))[0][0]


def _extract_service_date(text: str) -> str | None:
    labeled_patterns = (
        r"\b(?:date\s+of\s+service|service\s+date|dos)\s*[:#-]?\s*(\d{1,2}/\d{1,2}/\d{2,4})\b",
        r"\b(?:date\s+of\s+service|service\s+date|dos)\s*[:#-]?\s*(\d{4}-\d{2}-\d{2})\b",
    )
    for pattern in labeled_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1)

    generic_patterns = (
        r"\b(\d{4}-\d{2}-\d{2})\b",
        r"\b(\d{1,2}/\d{1,2}/\d{2,4})\b",
    )
    for pattern in generic_patterns:
        for match in re.finditer(pattern, text):
            context = text[max(0, match.start() - 32):match.start()].lower()
            if re.search(r"\b(?:dob|date\s+of\s+birth|birth)\b", context):
                continue
            return match.group(1)
    return None


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    cleaned = cleaned.strip("-")
    return cleaned or "item"


def _render_page_markdown(
    *,
    run_id: str,
    pdf_id: str,
    page: int,
    label: str | None,
    patient_name: str | None,
    service_date: str | None,
    text: str,
) -> str:
    metadata_lines = [
        f"- Run ID: {run_id}",
        f"- PDF ID: {pdf_id}",
        f"- Page: {page}",
    ]
    if label:
        metadata_lines.append(f"- Model label: {label}")
    if patient_name:
        metadata_lines.append(f"- Patient name: {patient_name}")
    if service_date:
        metadata_lines.append(f"- Service date: {service_date}")

    return "\n".join(
        [
            "# FastPDF Page Evidence",
            "",
            *metadata_lines,
            "",
            "## Evidence Text",
            "",
            text,
            "",
        ]
    )


def load_run_json(path: Path, *, run_id: str | None = None) -> tuple[str, dict[str, Any], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("summary_payload"), dict):
        source_kind = "fastpdf_run_document"
        summary_payload = payload["summary_payload"]
        resolved_run_id = str(run_id or payload.get("run_id") or payload.get("_id") or path.stem)
    else:
        source_kind = "summary_payload"
        summary_payload = payload
        resolved_run_id = str(run_id or payload.get("run_id") or path.stem)
    return resolved_run_id, summary_payload, source_kind


def load_run_from_mongo(
    *,
    run_id: str,
    mongo_uri: str,
    mongo_database: str,
    mongo_collection: str = "runs",
) -> tuple[str, dict[str, Any], str]:
    client = MongoClient(mongo_uri)
    try:
        collection = client[mongo_database][mongo_collection]
        document = collection.find_one({"_id": run_id}) or collection.find_one({"run_id": run_id})
        if not isinstance(document, dict):
            raise ValueError(f"run_id {run_id!r} was not found in {mongo_database}.{mongo_collection}")
        payload = document.get("summary_payload")
        if not isinstance(payload, dict):
            raise ValueError(f"run_id {run_id!r} does not contain a usable summary_payload")
        return run_id, payload, "fastpdf_run_document"
    finally:
        client.close()


def materialize_summary_payload(
    *,
    run_id: str,
    summary_payload: dict[str, Any],
    source_kind: str,
    output_dir: Path,
    include_non_survivors: bool = False,
    settings: AppSettings | None = None,
) -> MaterializationManifest:
    effective_settings = settings or get_settings()
    output_dir.mkdir(parents=True, exist_ok=True)
    pages_dir = output_dir / "pages"
    retrieval_dir = output_dir / "retrieval"
    pages_dir.mkdir(parents=True, exist_ok=True)
    retrieval_dir.mkdir(parents=True, exist_ok=True)

    patient_name = _extract_patient_name(summary_payload)
    page_documents: list[MaterializedPage] = []
    retrieval_documents: list[MaterializedRetrievalDocument] = []
    total_pages = 0

    raw_pdfs = summary_payload.get("pdfs")
    if not isinstance(raw_pdfs, list):
        raise ValueError("summary_payload.pdfs must be a list")

    order_index = 0
    for pdf_row in raw_pdfs:
        if not isinstance(pdf_row, dict):
            continue
        pdf_id = _clean_scalar(pdf_row.get("pdf_id"))
        if not pdf_id:
            continue
        raw_pages = pdf_row.get("pages")
        if not isinstance(raw_pages, list):
            continue
        for page_row in raw_pages:
            if not isinstance(page_row, dict):
                continue
            total_pages += 1
            page_no = _positive_int(page_row.get("page"))
            if page_no is None:
                continue
            if not include_non_survivors:
                if bool(page_row.get("is_blank")) or bool(page_row.get("is_duplicate")):
                    continue
                if page_row.get("is_survivor") is False:
                    continue
            text, source_fields = _extract_best_page_text(page_row)
            if not text:
                continue

            label = _clean_scalar(page_row.get("effective_model_label") or page_row.get("model_label")) or None
            service_date = _extract_service_date(text)
            filename = f"{_slugify(run_id)}__{_slugify(pdf_id)}__p{page_no:04d}.md"
            file_path = pages_dir / filename
            file_path.write_text(
                _render_page_markdown(
                    run_id=run_id,
                    pdf_id=pdf_id,
                    page=page_no,
                    label=label,
                    patient_name=patient_name,
                    service_date=service_date,
                    text=text,
                ),
                encoding="utf-8",
            )

            blocks = text_to_blocks(
                text,
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
                retrieval_filename = (
                    f"{_slugify(run_id)}__{_slugify(pdf_id)}__p{page_no:04d}__c{chunk.chunk_index:04d}.md"
                )
                retrieval_path = retrieval_dir / retrieval_filename
                retrieval_path.write_text(
                    render_retrieval_markdown(
                        run_id=run_id,
                        pdf_id=pdf_id,
                        page=page_no,
                        page_source_filename=filename,
                        chunk=chunk,
                        chunk_total=len(chunks),
                        label=label,
                        patient_name=patient_name,
                        service_date=service_date,
                    ),
                    encoding="utf-8",
                )
                retrieval_filenames.append(retrieval_filename)
                retrieval_relative_paths.append(str(retrieval_path.relative_to(output_dir)))
                retrieval_documents.append(
                    MaterializedRetrievalDocument(
                        run_id=run_id,
                        pdf_id=pdf_id,
                        page=page_no,
                        order_index=order_index,
                        chunk_index=chunk.chunk_index,
                        source_filename=retrieval_filename,
                        relative_path=str(retrieval_path.relative_to(output_dir)),
                        text_length=len(chunk.text),
                        text_preview=chunk.text[:TEXT_PREVIEW_CHARS],
                        parent_source_filename=filename,
                        section_title=chunk.section_title,
                        source_fields=list(source_fields),
                        metadata={
                            "structure_chunking_strategy": "structure_aware_blocks",
                            "source_kind": source_kind,
                            **chunk.metadata,
                        },
                    )
                )

            page_documents.append(
                MaterializedPage(
                    run_id=run_id,
                    pdf_id=pdf_id,
                    page=page_no,
                    order_index=order_index,
                    source_filename=filename,
                    relative_path=str(file_path.relative_to(output_dir)),
                    label=label,
                    service_date=service_date,
                    patient_name=patient_name,
                    text_length=len(text),
                    text_preview=text[:TEXT_PREVIEW_CHARS],
                    source_fields=source_fields,
                    retrieval_filenames=retrieval_filenames,
                    retrieval_relative_paths=retrieval_relative_paths,
                )
            )
            order_index += 1

    manifest = MaterializationManifest(
        run_id=run_id,
        source_kind=source_kind,
        patient_name=patient_name,
        total_pages=total_pages,
        materialized_pages=len(page_documents),
        retrieval_document_count=len(retrieval_documents),
        page_documents=page_documents,
        retrieval_documents=retrieval_documents,
    )
    (output_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return manifest
