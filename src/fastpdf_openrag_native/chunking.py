from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")
WORD_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
EMBEDDED_SECTION_SPLIT_RE = re.compile(
    r"(?=\b(?:chief complaint|history of present(?:ing)? complaint|history of present illness|past medical history|past surgical history|social history|allerg(?:y|ies)|current medications?|review of systems|physical exam(?:ination)?|assessment|diagnos(?:is|es)|plan|treatment|mechanism of injury|impression|findings)\s*:)",
    flags=re.IGNORECASE,
)
CLINICAL_SECTION_SIGNAL_RE = re.compile(
    r"\b(?:chief complaint|history of present(?:ing)? complaint|history of present illness|past medical history|past surgical history|social history|allerg(?:y|ies)|current medications?|review of systems|physical exam(?:ination)?|assessment|diagnos(?:is|es)|plan|treatment|mechanism of injury|impression|findings|pain|tender(?:ness)?|effusion|mri|tear|meniscus|icd|follow[- ]?up|therapy|injection|accident|mva)\b",
    flags=re.IGNORECASE,
)
LOW_VALUE_ADMIN_SIGNAL_RE = re.compile(
    r"\b(?:phone|fax|email|website|page\s+\d+\s+of\s+\d+|confidential(?:ity)?|routing|printed\s+(?:by|on)|cover sheet|member id|guarantor|policy|subscriber|dob|mrn|ssn|po box|file\s*#?)\b",
    flags=re.IGNORECASE,
)
ADDRESS_SPAN_RE = re.compile(
    r"\b\d{1,5}\s+(?:[NEWS]\.?\s+)?(?:[A-Za-z0-9#.'-]+\s+){0,6}(?:St\.?|Street|Ave\.?|Avenue|Rd\.?|Road|Blvd\.?|Boulevard|Dr\.?|Drive|Ln\.?|Lane|Ct\.?|Court|Way|Pkwy\.?|Parkway)(?:,?\s*(?:Ste\.?|Suite)\s*[A-Za-z0-9-]+)?(?:\s+[A-Za-z .'-]+,?\s*[A-Z]{2}\s+\d{5}(?:-\d{4})?)?",
    flags=re.IGNORECASE,
)
CONTACT_SPAN_PATTERNS = tuple(
    re.compile(pattern, flags=re.IGNORECASE)
    for pattern in (
        r"\bPhone:?\s*(?:\(?\d{3}\)?[-.\s]*)?\d{3}[-.\s]*\d{4}",
        r"\bFax:?\s*(?:\(?\d{3}\)?[-.\s]*)?\d{3}[-.\s]*\d{4}",
        r"\bEmail:?\s*\S+@\S+",
        r"\bWebsite:?\s*(?:https?://|www\.)\S+",
        r"\bPage\s+\d+\s+of\s+\d+\b",
        r"\bFile\s*#?:?\s*[A-Z0-9-]+\b",
    )
)


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


def _split_embedded_sections(text: str) -> list[str]:
    clean = _normalize_block_text(text)
    if not clean:
        return []
    parts = re.split(EMBEDDED_SECTION_SPLIT_RE, clean)
    return [part.strip(" -;") for part in parts if part.strip(" -;")]


def _token_count(text: str) -> int:
    return len(WORD_TOKEN_RE.findall(text))


def _merge_short_fragments(fragments: list[str]) -> list[str]:
    merged: list[str] = []
    pending_prefix: str | None = None
    for raw_fragment in fragments:
        fragment = _normalize_block_text(raw_fragment)
        if not fragment:
            continue
        token_count = _token_count(fragment)
        looks_like_heading = fragment.endswith(":") or (
            fragment[:1].isupper() and _is_probable_heading(fragment) and token_count <= 5
        )
        if looks_like_heading and len(fragment) <= 48:
            pending_prefix = fragment if fragment.endswith(":") else f"{fragment}:"
            continue
        if pending_prefix:
            fragment = _normalize_block_text(f"{pending_prefix} {fragment}")
            pending_prefix = None
        if token_count <= 1 and not CLINICAL_SECTION_SIGNAL_RE.search(fragment):
            continue
        merged.append(fragment)

    if pending_prefix and merged:
        merged[-1] = _normalize_block_text(f"{merged[-1]} {pending_prefix}")
    return merged


def _strip_low_value_admin_spans(text: str) -> str:
    clean = _normalize_block_text(text)
    if not clean:
        return ""
    for pattern in CONTACT_SPAN_PATTERNS:
        clean = pattern.sub(" ", clean)
    clean = ADDRESS_SPAN_RE.sub(" ", clean)
    clean = LOW_VALUE_ADMIN_SIGNAL_RE.sub(" ", clean)
    clean = re.sub(r"\s{2,}", " ", clean)
    clean = re.sub(r"\s+([,.;:!?])", r"\1", clean)
    clean = re.sub(r"^[,;:\s]+|[,;:\s]+$", "", clean)
    return _normalize_block_text(clean)


def _is_low_value_admin_text(text: str) -> bool:
    clean = _normalize_block_text(text)
    if not clean:
        return True
    has_admin_signal = bool(LOW_VALUE_ADMIN_SIGNAL_RE.search(clean) or ADDRESS_SPAN_RE.search(clean))
    has_clinical_signal = bool(CLINICAL_SECTION_SIGNAL_RE.search(clean))
    if has_admin_signal and not has_clinical_signal:
        return True
    if len(clean.split()) <= 5 and (LOW_VALUE_ADMIN_SIGNAL_RE.search(clean) or ADDRESS_SPAN_RE.search(clean)):
        return True
    return False


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
        base_metadata = {
            "block_index": paragraph.get("block_index"),
            "paragraph_index": paragraph.get("paragraph_index"),
            "page_paragraph_index": paragraph.get("page_paragraph_index"),
            "bbox": paragraph.get("bbox") or {},
        }
        fragments: list[str] = []
        raw_lines = [line for line in text.splitlines() if line.strip()] or [text]
        for raw_line in raw_lines:
            for segment in _split_embedded_sections(raw_line):
                cleaned_segment = _strip_low_value_admin_spans(segment)
                if not cleaned_segment or _is_low_value_admin_text(cleaned_segment):
                    continue
                fragments.append(cleaned_segment)
        fragments = _merge_short_fragments(fragments)
        if not fragments:
            fallback = _strip_low_value_admin_spans(text)
            if fallback and not _is_low_value_admin_text(fallback):
                fragments.append(fallback)
        for fragment_index, fragment in enumerate(fragments):
            metadata = dict(base_metadata)
            metadata["fragment_index"] = fragment_index
            blocks.append(StructuredBlock(text=fragment, metadata=metadata))
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

    if current_blocks:
        sections.append((current_title, current_blocks))

    if not sections and any(not _is_probable_heading(block.text) for block in blocks):
        return [(None, blocks)]
    if not sections and len(blocks) == 1 and _token_count(blocks[0].text) >= 3:
        return [(None, blocks)]
    return sections


def _render_chunk_text(blocks: list[StructuredBlock]) -> str:
    return "\n\n".join(block.text for block in blocks if block.text).strip()


def _make_chunk(section_title: str | None, blocks: list[StructuredBlock], *, chunk_index: int) -> StructuredChunk:
    return StructuredChunk(
        chunk_index=chunk_index,
        section_title=section_title,
        blocks=list(blocks),
        text=_render_chunk_text(blocks),
        metadata=_metadata_range(blocks),
    )


def _is_banned_retrieval_chunk_text(text: str) -> bool:
    clean = _normalize_block_text(text)
    if not clean:
        return True
    token_count = _token_count(clean)
    if _is_probable_heading(clean) and token_count <= 3:
        return True
    if token_count <= 1:
        return True
    if token_count <= 2 and len(clean) <= 12 and not CLINICAL_SECTION_SIGNAL_RE.search(clean):
        return True
    return False


def _should_merge_short_chunk(text: str, *, min_chunk_chars: int) -> bool:
    clean = _normalize_block_text(text)
    if not clean:
        return False
    token_count = _token_count(clean)
    if token_count <= 3 and len(clean) < max(80, min_chunk_chars // 2):
        return True
    if len(clean) < max(60, min_chunk_chars // 3) and token_count <= 8:
        return True
    return False


def _consolidate_section_chunks(
    section_title: str | None,
    chunks: list[StructuredChunk],
    *,
    target_chars: int,
    min_chunk_chars: int,
) -> list[StructuredChunk]:
    if not chunks:
        return []
    if len(chunks) == 1:
        chunk = chunks[0]
        return [] if _is_banned_retrieval_chunk_text(chunk.text) else [chunk]

    consolidated: list[StructuredChunk] = []
    cursor = 0
    merge_target_chars = max(target_chars, min_chunk_chars)
    while cursor < len(chunks):
        current = chunks[cursor]
        if _should_merge_short_chunk(current.text, min_chunk_chars=min_chunk_chars):
            previous = consolidated[-1] if consolidated else None
            following = chunks[cursor + 1] if cursor + 1 < len(chunks) else None
            merged_into_previous = False
            if previous is not None:
                previous_text = _normalize_block_text(previous.text)
                current_text = _normalize_block_text(current.text)
                if len(previous_text) + len(current_text) <= int(merge_target_chars * 1.35):
                    consolidated[-1] = _make_chunk(
                        section_title,
                        [*previous.blocks, *current.blocks],
                        chunk_index=previous.chunk_index,
                    )
                    merged_into_previous = True
            if merged_into_previous:
                cursor += 1
                continue
            if following is not None:
                following_text = _normalize_block_text(following.text)
                current_text = _normalize_block_text(current.text)
                if len(following_text) + len(current_text) <= int(merge_target_chars * 1.35):
                    chunks[cursor + 1] = _make_chunk(
                        section_title,
                        [*current.blocks, *following.blocks],
                        chunk_index=following.chunk_index,
                    )
                    cursor += 1
                    continue
        if not _is_banned_retrieval_chunk_text(current.text):
            consolidated.append(current)
        cursor += 1
    return consolidated


def _metadata_range(blocks: list[StructuredBlock]) -> dict[str, Any]:
    block_indexes = [value for value in (block.metadata.get("block_index") for block in blocks) if isinstance(value, int)]
    paragraph_indexes = [
        value for value in (block.metadata.get("paragraph_index") for block in blocks) if isinstance(value, int)
    ]
    page_paragraph_indexes = [
        value for value in (block.metadata.get("page_paragraph_index") for block in blocks) if isinstance(value, int)
    ]
    metadata: dict[str, Any] = {"block_count": len(blocks)}
    if block_indexes:
        metadata["block_start"] = min(block_indexes)
        metadata["block_end"] = max(block_indexes)
    if paragraph_indexes:
        metadata["paragraph_start"] = min(paragraph_indexes)
        metadata["paragraph_end"] = max(paragraph_indexes)
    if page_paragraph_indexes:
        metadata["page_paragraph_start"] = min(page_paragraph_indexes)
        metadata["page_paragraph_end"] = max(page_paragraph_indexes)

    paragraph_refs: list[dict[str, int]] = []
    seen_refs: set[tuple[int, int, int | None]] = set()
    for block in blocks:
        block_index = block.metadata.get("block_index")
        paragraph_index = block.metadata.get("paragraph_index")
        page_paragraph_index = block.metadata.get("page_paragraph_index")
        if not isinstance(block_index, int) or not isinstance(paragraph_index, int):
            continue
        ref_key = (block_index, paragraph_index, page_paragraph_index if isinstance(page_paragraph_index, int) else None)
        if ref_key in seen_refs:
            continue
        seen_refs.add(ref_key)
        ref = {
            "block_index": block_index,
            "paragraph_index": paragraph_index,
        }
        if isinstance(page_paragraph_index, int):
            ref["page_paragraph_index"] = page_paragraph_index
        paragraph_refs.append(ref)
    if paragraph_refs:
        metadata["paragraph_refs"] = paragraph_refs
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
        if not section_blocks:
            continue

        section_chunks: list[StructuredChunk] = []
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
                    section_chunks.append(
                        _make_chunk(
                            section_title,
                            [split_block],
                            chunk_index=chunk_index,
                        )
                    )
                start += 1
                previous_window = []
                continue

            section_chunks.append(
                _make_chunk(
                    section_title,
                    list(window),
                    chunk_index=chunk_index,
                )
            )
            previous_window = list(window)

        section_chunks = _consolidate_section_chunks(
            section_title,
            section_chunks,
            target_chars=target_chars,
            min_chunk_chars=min_chunk_chars,
        )
        for section_chunk in section_chunks:
            chunks.append(
                StructuredChunk(
                    chunk_index=chunk_index,
                    section_title=section_chunk.section_title,
                    blocks=list(section_chunk.blocks),
                    text=section_chunk.text,
                    metadata=dict(section_chunk.metadata),
                )
            )
            chunk_index += 1

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
    if (
        chunk.metadata.get("page_paragraph_start") is not None
        and chunk.metadata.get("page_paragraph_end") is not None
    ):
        metadata_lines.append(
            "- OCR page paragraph span: "
            f"{chunk.metadata['page_paragraph_start']} to {chunk.metadata['page_paragraph_end']}"
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
