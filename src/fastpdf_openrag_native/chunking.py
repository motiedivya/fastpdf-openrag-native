from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(slots=True)
class StructuredBlock:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StructuredChunk:
    chunk_index: int
    section_title: str | None
    blocks: list[StructuredBlock]
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _normalize_text(value: str) -> str:
    clean = value.replace("\u00a0", " ")
    clean = clean.replace("\r\n", "\n").replace("\r", "\n")
    clean = re.sub(r"[ \t]+", " ", clean)
    clean = re.sub(r"\n{3,}", "\n\n", clean)
    return clean.strip()


def _normalize_block_text(value: str) -> str:
    text = _normalize_text(value)
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines).strip()


def _is_probable_heading(text: str) -> bool:
    line = _normalize_block_text(text)
    if not line or "\n" in line or len(line) > 80:
        return False
    if line.endswith((".", "!", "?")):
        return False
    letters = [char for char in line if char.isalpha()]
    if not letters:
        return False
    uppercase_ratio = sum(1 for char in letters if char.isupper()) / len(letters)
    titlecase_words = sum(1 for word in line.split() if word[:1].isupper())
    if line.endswith(":"):
        return True
    if uppercase_ratio >= 0.7:
        return True
    if titlecase_words == len(line.split()) and len(line.split()) <= 6:
        return True
    return bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 /(),&'-]{0,79}", line))


def text_to_blocks(text: str, *, target_chars: int) -> list[StructuredBlock]:
    normalized = _normalize_text(text)
    if not normalized:
        return []

    raw_parts = [part for part in re.split(r"\n\s*\n", normalized) if part.strip()]
    if len(raw_parts) <= 1:
        raw_parts = [line for line in normalized.splitlines() if line.strip()]
    blocks = [StructuredBlock(text=_normalize_block_text(part)) for part in raw_parts if _normalize_block_text(part)]
    if len(blocks) == 1 and len(blocks[0].text) > max(target_chars, 600):
        blocks = [
            StructuredBlock(text=part, metadata=dict(blocks[0].metadata))
            for part in _split_long_text(blocks[0].text, max(250, target_chars // 2))
        ]
    return blocks


def blocks_from_ocr_paragraphs(paragraphs: list[dict[str, Any]]) -> list[StructuredBlock]:
    blocks: list[StructuredBlock] = []
    for paragraph in paragraphs:
        text = _normalize_block_text(str(paragraph.get("text") or ""))
        if not text:
            continue
        metadata = {
            "block_index": paragraph.get("block_index"),
            "paragraph_index": paragraph.get("paragraph_index"),
            "bbox": paragraph.get("bbox") or {},
        }
        blocks.append(StructuredBlock(text=text, metadata=metadata))
    return blocks


def _split_long_text(text: str, target_chars: int) -> list[str]:
    sentences = [part.strip() for part in SENTENCE_BOUNDARY_RE.split(_normalize_text(text)) if part.strip()]
    if len(sentences) <= 1:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    current_chars = 0
    for sentence in sentences:
        projected = current_chars + len(sentence) + (1 if current else 0)
        if current and projected > target_chars:
            chunks.append(" ".join(current).strip())
            current = [sentence]
            current_chars = len(sentence)
            continue
        current.append(sentence)
        current_chars = projected
    if current:
        chunks.append(" ".join(current).strip())
    return chunks or [text]


def _split_sections(blocks: list[StructuredBlock]) -> list[tuple[str | None, list[StructuredBlock]]]:
    sections: list[tuple[str | None, list[StructuredBlock]]] = []
    current_title: str | None = None
    current_blocks: list[StructuredBlock] = []

    for block in blocks:
        if _is_probable_heading(block.text):
            if current_blocks:
                sections.append((current_title, current_blocks))
                current_blocks = []
            current_title = block.text.rstrip(":").strip() or None
            continue
        current_blocks.append(block)

    if current_blocks or current_title:
        sections.append((current_title, current_blocks))

    if not sections:
        return [(None, blocks)]
    return sections


def _metadata_range(blocks: list[StructuredBlock]) -> dict[str, Any]:
    block_indexes = [value for value in (block.metadata.get("block_index") for block in blocks) if isinstance(value, int)]
    paragraph_indexes = [
        value for value in (block.metadata.get("paragraph_index") for block in blocks) if isinstance(value, int)
    ]
    metadata: dict[str, Any] = {"block_count": len(blocks)}
    if block_indexes:
        metadata["block_start"] = min(block_indexes)
        metadata["block_end"] = max(block_indexes)
    if paragraph_indexes:
        metadata["paragraph_start"] = min(paragraph_indexes)
        metadata["paragraph_end"] = max(paragraph_indexes)
    return metadata


def build_structured_chunks(
    blocks: list[StructuredBlock],
    *,
    target_chars: int,
    overlap_blocks: int = 1,
) -> list[StructuredChunk]:
    normalized_blocks = [
        StructuredBlock(text=_normalize_block_text(block.text), metadata=dict(block.metadata))
        for block in blocks
        if _normalize_block_text(block.text)
    ]
    if not normalized_blocks:
        return []

    min_chunk_chars = max(350, target_chars // 3)
    sections = _split_sections(normalized_blocks)
    chunks: list[StructuredChunk] = []
    chunk_index = 1

    for section_title, section_blocks in sections:
        if not section_blocks and section_title:
            chunks.append(
                StructuredChunk(
                    chunk_index=chunk_index,
                    section_title=section_title,
                    blocks=[],
                    text=section_title,
                    metadata={},
                )
            )
            chunk_index += 1
            continue

        start = 0
        previous_window: list[StructuredBlock] = []
        while start < len(section_blocks):
            overlap_tail = previous_window[-overlap_blocks:] if overlap_blocks > 0 else []
            window = list(overlap_tail)
            window_chars = sum(len(block.text) for block in window)
            added_new_blocks = 0

            while start < len(section_blocks):
                block = section_blocks[start]
                projected = window_chars + len(block.text) + (2 if window else 0)
                if window and added_new_blocks > 0 and projected > target_chars and window_chars >= min_chunk_chars:
                    break
                if window and added_new_blocks == 0 and projected > target_chars and overlap_tail:
                    window = []
                    window_chars = 0
                    overlap_tail = []
                    continue
                window.append(block)
                window_chars = projected
                start += 1
                added_new_blocks += 1

            if added_new_blocks == 0 and start < len(section_blocks):
                block = section_blocks[start]
                split_blocks = [
                    StructuredBlock(text=part, metadata=dict(block.metadata))
                    for part in _split_long_text(block.text, target_chars)
                ]
                for split_block in split_blocks:
                    chunks.append(
                        StructuredChunk(
                            chunk_index=chunk_index,
                            section_title=section_title,
                            blocks=[split_block],
                            text=split_block.text,
                            metadata=_metadata_range([split_block]),
                        )
                    )
                    chunk_index += 1
                start += 1
                previous_window = []
                continue

            chunks.append(
                StructuredChunk(
                    chunk_index=chunk_index,
                    section_title=section_title,
                    blocks=list(window),
                    text="\n\n".join(block.text for block in window if block.text).strip(),
                    metadata=_metadata_range(window),
                )
            )
            chunk_index += 1
            previous_window = list(window)

    return chunks


def render_retrieval_markdown(
    *,
    run_id: str,
    pdf_id: str,
    page: int,
    page_source_filename: str,
    chunk: StructuredChunk,
    chunk_total: int,
    label: str | None = None,
    patient_name: str | None = None,
    service_date: str | None = None,
) -> str:
    metadata_lines = [
        f"- Run ID: {run_id}",
        f"- PDF ID: {pdf_id}",
        f"- Page: {page}",
        f"- Page source filename: {page_source_filename}",
        f"- Retrieval chunk: {chunk.chunk_index} of {chunk_total}",
    ]
    if chunk.section_title:
        metadata_lines.append(f"- Section title: {chunk.section_title}")
    if label:
        metadata_lines.append(f"- Model label: {label}")
    if patient_name:
        metadata_lines.append(f"- Patient name: {patient_name}")
    if service_date:
        metadata_lines.append(f"- Service date: {service_date}")
    if chunk.metadata.get("block_start") is not None and chunk.metadata.get("block_end") is not None:
        metadata_lines.append(
            f"- OCR block span: {chunk.metadata['block_start']} to {chunk.metadata['block_end']}"
        )
    if (
        chunk.metadata.get("paragraph_start") is not None
        and chunk.metadata.get("paragraph_end") is not None
    ):
        metadata_lines.append(
            f"- OCR paragraph span: {chunk.metadata['paragraph_start']} to {chunk.metadata['paragraph_end']}"
        )

    body_lines: list[str] = []
    if chunk.section_title:
        body_lines.extend([f"### {chunk.section_title}", ""])

    block_texts = [block.text for block in chunk.blocks if block.text] or [chunk.text]
    for index, block_text in enumerate(block_texts):
        if index:
            body_lines.append("")
        body_lines.append(block_text)

    return "\n".join(
        [
            "# FastPDF Retrieval Chunk",
            "",
            *metadata_lines,
            "",
            "## Evidence Text",
            "",
            *body_lines,
            "",
        ]
    )
