from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pymongo import MongoClient

from .chunking import blocks_from_ocr_paragraphs, build_structured_chunks, render_retrieval_markdown, text_to_blocks
from .models import MaterializationManifest, MaterializedPage, MaterializedRetrievalDocument
from .ocr_extract import _build_indexable_full_text, _prepare_paragraphs_for_indexing, build_html_document
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
NATIVE_CHUNK_TEXT_FIELDS = frozenset({
    "pdf2html_text",
    "rich_text",
    "page_html_text",
    "text_layer_text",
    "native_text",
})
WORD_RE = re.compile(r"[A-Za-z0-9]+")
CLINICAL_DETAIL_SIGNAL_RE = re.compile(
    r"\b(?:history of present(?:ing)? complaint|history of present illness|past medical history|past surgical history|social history|allerg(?:y|ies)|current medications?|review of systems|exam findings|physical exam(?:ination)?|assessment|diagnostic studies|plan|treatment|mechanism of injury|pain|therapy|hypertension|cholesterol|effusion|meniscus|mri)\b",
    flags=re.IGNORECASE,
)
HEADER_ADMIN_SIGNAL_RE = re.compile(
    r"\b(?:provider|patient|dob|date of birth|doa|date of accident|date of service|service date|phone|fax|email|website|file\s*#?|page\s+\d+\s+of\s+\d+)\b",
    flags=re.IGNORECASE,
)

INLINE_NOISE_PATTERNS = (
    r"google\s+vision\s+extract",
    r"^ocr\s+output$",
)

TEXT_PREVIEW_CHARS = 1200
NORMALIZED_BBOX_CANVAS = 1000


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


def _word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def _should_prefer_best_text_chunking(
    *,
    best_text: str,
    source_fields: list[str],
    ocr_paragraphs: list[dict[str, Any]],
) -> bool:
    if not best_text or not source_fields or not ocr_paragraphs:
        return False

    primary_source = next((field for field in source_fields if field in TEXT_FIELD_PRIORITY), "")
    if primary_source not in NATIVE_CHUNK_TEXT_FIELDS:
        return False

    ocr_text = _build_indexable_full_text(ocr_paragraphs, "")
    if not ocr_text:
        return True

    best_word_count = _word_count(best_text)
    ocr_word_count = _word_count(ocr_text)
    if best_word_count <= 0:
        return False
    if ocr_word_count <= 0:
        return True

    if best_word_count >= max(ocr_word_count * 2, ocr_word_count + 120):
        return True

    best_has_clinical_detail = bool(CLINICAL_DETAIL_SIGNAL_RE.search(best_text))
    ocr_has_clinical_detail = bool(CLINICAL_DETAIL_SIGNAL_RE.search(ocr_text))
    if len(ocr_paragraphs) <= 2 and best_has_clinical_detail and best_word_count >= max(80, ocr_word_count + 40):
        return True
    if best_has_clinical_detail and not ocr_has_clinical_detail and best_word_count >= max(
        ocr_word_count + 60,
        int(ocr_word_count * 1.35),
    ):
        return True
    return False


def _is_header_like_chunk_text(text: str) -> bool:
    clean = _normalize_text(text)
    if not clean:
        return True
    admin_hits = len(HEADER_ADMIN_SIGNAL_RE.findall(clean))
    if admin_hits <= 0:
        return False
    if CLINICAL_DETAIL_SIGNAL_RE.search(clean):
        return False
    token_count = _word_count(clean)
    if admin_hits >= 2:
        return True
    return token_count <= 40


def _build_chunk_previews(chunks: list[Any], *, limit: int = 5, preview_chars: int = 160) -> list[str]:
    previews: list[str] = []
    for chunk in chunks[:limit]:
        chunk_text = _normalize_text(getattr(chunk, "text", ""))
        if not chunk_text:
            continue
        previews.append(chunk_text[:preview_chars])
    return previews


def _should_rebuild_from_best_text_guardrail(
    *,
    best_text: str,
    source_fields: list[str],
    chunking_strategy: str,
    chunks: list[Any],
) -> tuple[bool, str | None]:
    if chunking_strategy != "ocr_paragraph_blocks" or not best_text or not source_fields:
        return False, None

    primary_source = next((field for field in source_fields if field in TEXT_FIELD_PRIORITY), "")
    if primary_source not in NATIVE_CHUNK_TEXT_FIELDS:
        return False, None

    native_text_chars = len(best_text)
    if native_text_chars < 400:
        return False, None

    chunk_count = len(chunks)
    header_like_count = sum(1 for chunk in chunks if _is_header_like_chunk_text(getattr(chunk, "text", "")))
    if chunk_count == 0:
        return True, "born_digital_empty_chunk_inventory"
    if chunk_count == 1 and header_like_count == 1:
        return True, "born_digital_single_header_chunk"
    if native_text_chars >= 900 and chunk_count < 3:
        return True, "born_digital_low_chunk_count"
    if chunk_count < 3 and header_like_count == chunk_count:
        return True, "born_digital_header_only_chunk_inventory"
    return False, None


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


def _unique_preserve(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        clean = str(value or "").strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        ordered.append(clean)
    return ordered


def _coerce_bbox_value(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _median_float(values: list[float], default: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return default
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def _join_ocr_tokens(tokens: list[str]) -> str:
    text = " ".join(token for token in tokens if str(token or "").strip()).strip()
    if not text:
        return ""
    text = re.sub(r"\s+([,.;:!?%])", r"\1", text)
    text = re.sub(r"([(/$#])\s+", r"\1", text)
    text = re.sub(r"\s+([)\]])", r"\1", text)
    text = re.sub(r"(?<=\d)\s*/\s*(?=\d)", "/", text)
    text = re.sub(r"(?<=\d)\s*-\s*(?=\d)", "-", text)
    text = re.sub(r"(?<=\w)\s+'(?=\w)", "'", text)
    return _normalize_text(text)


def _looks_like_word_level_ocr(entries: list[dict[str, Any]]) -> bool:
    if len(entries) < 8:
        return False

    shortish = 0
    longish = 0
    for entry in entries:
        text = str(entry.get("text") or "")
        word_count = len(text.split())
        if len(text) <= 20 and word_count <= 3:
            shortish += 1
        if len(text) >= 36 or word_count >= 6:
            longish += 1

    total = len(entries)
    return (shortish / total) >= 0.72 and (longish / total) <= 0.18


def _paragraph_row_from_entries(entries: list[dict[str, Any]], *, paragraph_index: int) -> dict[str, Any]:
    left = int(round(min(float(entry["left"]) for entry in entries)))
    top = int(round(min(float(entry["top"]) for entry in entries)))
    right = int(round(max(float(entry["right"]) for entry in entries)))
    bottom = int(round(max(float(entry["bottom"]) for entry in entries)))
    text = _join_ocr_tokens([str(entry.get("text") or "") for entry in entries])
    return {
        "block_index": 1,
        "paragraph_index": paragraph_index,
        "page_paragraph_index": paragraph_index,
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


def _extract_ocr_block_paragraphs(page_row: dict[str, Any]) -> list[dict[str, Any]]:
    raw_blocks = page_row.get("ocr_blocks")
    if not isinstance(raw_blocks, list):
        return []

    entries: list[dict[str, Any]] = []
    for item in raw_blocks:
        if not isinstance(item, dict):
            continue
        text = _normalize_text(item.get("text"))
        bbox = item.get("bbox")
        if not text or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        x0 = _coerce_bbox_value(bbox[0])
        y0 = _coerce_bbox_value(bbox[1])
        x1 = _coerce_bbox_value(bbox[2])
        y1 = _coerce_bbox_value(bbox[3])
        if None in {x0, y0, x1, y1}:
            continue
        assert x0 is not None and y0 is not None and x1 is not None and y1 is not None
        if x1 <= x0 or y1 <= y0:
            continue
        entries.append(
            {
                "text": text,
                "left": x0,
                "top": y0,
                "right": x1,
                "bottom": y1,
                "width": x1 - x0,
                "height": y1 - y0,
                "center_y": (y0 + y1) / 2.0,
            }
        )

    if not entries:
        return []

    entries.sort(key=lambda row: (round(float(row["top"]), 3), round(float(row["left"]), 3), str(row["text"])))
    if not _looks_like_word_level_ocr(entries):
        return [_paragraph_row_from_entries([entry], paragraph_index=index) for index, entry in enumerate(entries, start=1)]

    median_height = _median_float([float(entry["height"]) for entry in entries], 16.0)
    line_tolerance = max(6.0, median_height * 0.65)

    lines: list[list[dict[str, Any]]] = []
    current_line: list[dict[str, Any]] = []
    for entry in entries:
        if not current_line:
            current_line = [entry]
            continue

        line_center = sum(float(row["center_y"]) for row in current_line) / len(current_line)
        line_top = min(float(row["top"]) for row in current_line)
        line_bottom = max(float(row["bottom"]) for row in current_line)
        vertical_gap = max(0.0, float(entry["top"]) - line_bottom, line_top - float(entry["bottom"]))
        same_line = abs(float(entry["center_y"]) - line_center) <= line_tolerance or vertical_gap <= max(2.0, median_height * 0.2)
        if same_line:
            current_line.append(entry)
            continue

        lines.append(sorted(current_line, key=lambda row: (float(row["left"]), float(row["top"]))))
        current_line = [entry]

    if current_line:
        lines.append(sorted(current_line, key=lambda row: (float(row["left"]), float(row["top"]))))

    line_rows: list[dict[str, Any]] = []
    for row in lines:
        paragraph = _paragraph_row_from_entries(row, paragraph_index=len(line_rows) + 1)
        bbox = paragraph["bbox"]
        line_rows.append(
            {
                "text": paragraph["text"],
                "left": float(bbox["left"]),
                "top": float(bbox["top"]),
                "right": float(bbox["right"]),
                "bottom": float(bbox["bottom"]),
                "height": float(bbox["height"]),
                "entries": row,
            }
        )

    positive_gaps = [
        max(0.0, float(line_rows[index + 1]["top"]) - float(line_rows[index]["bottom"]))
        for index in range(len(line_rows) - 1)
        if line_rows[index + 1]["top"] >= line_rows[index]["bottom"]
    ]
    median_gap = _median_float(positive_gaps, max(4.0, median_height * 0.4))
    paragraph_gap_threshold = max(8.0, median_height * 1.15, median_gap * 2.2)
    indent_tolerance = max(18.0, median_height * 1.1)

    paragraphs: list[dict[str, Any]] = []
    current_paragraph_lines: list[dict[str, Any]] = []

    def flush_paragraph() -> None:
        if not current_paragraph_lines:
            return
        merged_entries: list[dict[str, Any]] = []
        for line in current_paragraph_lines:
            merged_entries.extend(line["entries"])
        paragraphs.append(_paragraph_row_from_entries(merged_entries, paragraph_index=len(paragraphs) + 1))
        current_paragraph_lines.clear()

    for line in line_rows:
        if not current_paragraph_lines:
            current_paragraph_lines.append(line)
            continue

        previous = current_paragraph_lines[-1]
        current_indent = float(current_paragraph_lines[0]["left"])
        line_gap = max(0.0, float(line["top"]) - float(previous["bottom"]))
        start_new_paragraph = False
        if line_gap > paragraph_gap_threshold:
            start_new_paragraph = True
        elif previous["text"].endswith(":") and len(previous["text"].split()) <= 8:
            start_new_paragraph = True
        elif abs(float(line["left"]) - current_indent) > indent_tolerance and line_gap > max(4.0, median_gap * 0.8):
            start_new_paragraph = True

        if start_new_paragraph:
            flush_paragraph()
        current_paragraph_lines.append(line)

    flush_paragraph()
    return paragraphs


def _extract_page_dimensions(page_row: dict[str, Any]) -> tuple[int | None, int | None]:
    width = _positive_int(page_row.get("page_width")) or _positive_int(page_row.get("width"))
    height = _positive_int(page_row.get("page_height")) or _positive_int(page_row.get("height"))
    return width, height


def _infer_bbox_space(
    *,
    paragraphs: list[dict[str, Any]],
    page_row: dict[str, Any],
) -> tuple[str, int | None, int | None]:
    explicit_width, explicit_height = _extract_page_dimensions(page_row)
    max_right = max((int((paragraph.get("bbox") or {}).get("right") or 0) for paragraph in paragraphs), default=0)
    max_bottom = max((int((paragraph.get("bbox") or {}).get("bottom") or 0) for paragraph in paragraphs), default=0)
    max_coord = max(max_right, max_bottom)

    if max_coord <= NORMALIZED_BBOX_CANVAS and not (explicit_width and explicit_height and max(explicit_width, explicit_height) > NORMALIZED_BBOX_CANVAS):
        return "normalized_1000", explicit_width, explicit_height

    return "page_pixels", explicit_width or max_right or None, explicit_height or max_bottom or None


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

            best_text, source_fields = _extract_best_page_text(page_row)
            ocr_paragraphs = _prepare_paragraphs_for_indexing(_extract_ocr_block_paragraphs(page_row))
            paragraph_text = _build_indexable_full_text(ocr_paragraphs, best_text) if ocr_paragraphs else ""
            text = best_text or paragraph_text
            if not text:
                continue

            label = _clean_scalar(page_row.get("effective_model_label") or page_row.get("model_label")) or None
            service_date = _extract_service_date(text)
            base_filename = f"{_slugify(run_id)}__{_slugify(pdf_id)}__p{page_no:04d}"
            markdown_filename = f"{base_filename}.md"
            markdown_path = pages_dir / markdown_filename
            markdown_path.write_text(
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

            page_source_filename = markdown_filename
            page_relative_path = str(markdown_path.relative_to(output_dir))
            page_document_type = "text/markdown"
            page_artifacts: dict[str, str] = {}
            page_metadata: dict[str, Any] = {}
            chunking_strategy = "structure_aware_blocks"
            primary_source = next((field for field in source_fields if field in TEXT_FIELD_PRIORITY), "")
            native_text_chars = len(best_text) if primary_source in NATIVE_CHUNK_TEXT_FIELDS else 0
            prefer_best_text_chunking = _should_prefer_best_text_chunking(
                best_text=best_text,
                source_fields=source_fields,
                ocr_paragraphs=ocr_paragraphs,
            )
            guardrail_triggered = False
            guardrail_reason: str | None = None

            if ocr_paragraphs:
                bbox_space, page_width, page_height = _infer_bbox_space(paragraphs=ocr_paragraphs, page_row=page_row)
                html_filename = f"{base_filename}.html"
                html_path = pages_dir / html_filename
                html_canvas_width = page_width or NORMALIZED_BBOX_CANVAS
                html_canvas_height = page_height or NORMALIZED_BBOX_CANVAS
                html_path.write_text(
                    build_html_document(
                        source_pdf=pdf_id,
                        page_number=page_no,
                        image_filename="",
                        width=html_canvas_width,
                        height=html_canvas_height,
                        paragraphs=ocr_paragraphs,
                        full_text=text,
                    ),
                    encoding="utf-8",
                )
                page_source_filename = html_filename
                page_relative_path = str(html_path.relative_to(output_dir))
                page_document_type = "text/html"
                page_artifacts["page_markdown"] = str(markdown_path.relative_to(output_dir))
                page_metadata.update(
                    {
                        "paragraph_count": len(ocr_paragraphs),
                        "ocr_paragraph_count": len(ocr_paragraphs),
                        "bbox_space": bbox_space,
                    }
                )
                if page_width:
                    page_metadata["page_width"] = page_width
                if page_height:
                    page_metadata["page_height"] = page_height
                source_fields = _unique_preserve([*source_fields, "ocr_blocks"])
                if prefer_best_text_chunking:
                    blocks = text_to_blocks(
                        text,
                        target_chars=effective_settings.structure_chunk_target_chars,
                    )
                else:
                    blocks = blocks_from_ocr_paragraphs(ocr_paragraphs)
                    chunking_strategy = "ocr_paragraph_blocks"
                page_metadata["indexed_paragraph_count"] = len(blocks)
            else:
                page_metadata.update(
                    {
                        "source_text_strategy": primary_source or "text",
                        "chunk_text_strategy": "best_text",
                        "ocr_paragraph_count": 0,
                    }
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
            if ocr_paragraphs:
                guardrail_triggered, guardrail_reason = _should_rebuild_from_best_text_guardrail(
                    best_text=best_text,
                    source_fields=source_fields,
                    chunking_strategy=chunking_strategy,
                    chunks=chunks,
                )
                if guardrail_triggered:
                    blocks = text_to_blocks(
                        text,
                        target_chars=effective_settings.structure_chunk_target_chars,
                    )
                    chunks = build_structured_chunks(
                        blocks,
                        target_chars=effective_settings.structure_chunk_target_chars,
                        overlap_blocks=effective_settings.structure_chunk_overlap_blocks,
                    )
                    chunking_strategy = "structure_aware_blocks"
                    prefer_best_text_chunking = True
            page_metadata.update(
                {
                    "source_text_strategy": (
                        "best_text_with_ocr_assets"
                        if (ocr_paragraphs and prefer_best_text_chunking)
                        else ("ocr_blocks_with_best_text_fallback" if ocr_paragraphs and best_text else ("ocr_blocks" if ocr_paragraphs else (primary_source or "text")))
                    ),
                    "chunk_text_strategy": "best_text" if chunking_strategy == "structure_aware_blocks" else "ocr_blocks",
                    "native_text_chars": native_text_chars,
                    "indexed_chunk_count": len(chunks),
                    "chunk_previews": _build_chunk_previews(chunks),
                    "native_text_guardrail_triggered": guardrail_triggered,
                }
            )
            if guardrail_reason:
                page_metadata["native_text_guardrail_reason"] = guardrail_reason

            retrieval_filenames: list[str] = []
            retrieval_relative_paths: list[str] = []
            for chunk in chunks:
                retrieval_filename = f"{base_filename}__c{chunk.chunk_index:04d}.md"
                retrieval_path = retrieval_dir / retrieval_filename
                retrieval_path.write_text(
                    render_retrieval_markdown(
                        run_id=run_id,
                        pdf_id=pdf_id,
                        page=page_no,
                        page_source_filename=page_source_filename,
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
                        parent_source_filename=page_source_filename,
                        section_title=chunk.section_title,
                        source_fields=list(source_fields),
                        metadata={
                            "structure_chunking_strategy": chunking_strategy,
                            "source_kind": source_kind,
                            **({"bbox_space": page_metadata.get("bbox_space")} if page_metadata.get("bbox_space") else {}),
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
                    source_filename=page_source_filename,
                    relative_path=page_relative_path,
                    document_type=page_document_type,
                    label=label,
                    service_date=service_date,
                    patient_name=patient_name,
                    text_length=len(text),
                    text_preview=text[:TEXT_PREVIEW_CHARS],
                    source_fields=source_fields,
                    artifacts=page_artifacts,
                    retrieval_filenames=retrieval_filenames,
                    retrieval_relative_paths=retrieval_relative_paths,
                    metadata=page_metadata,
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
