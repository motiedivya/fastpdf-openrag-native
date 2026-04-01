from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable, Sequence

from .models import MaterializationManifest

MAX_HISTORY_ITEMS = 6
MAX_EVIDENCE_TEXT = 1800
DEFAULT_CHAT_CONTEXT_CHUNKS = 10
DEFAULT_SUMMARY_CONTEXT_CHUNKS = 14
DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 150

STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "with",
}

SUMMARY_QUERY_RE = re.compile(
    r"\b(summary|summarize|overview|gist|key points|main points|highlights|chronology|timeline|what is this document|document about)\b",
    flags=re.IGNORECASE,
)
SOURCE_PAREN_RE = re.compile(r"\s*\((?:source|sources):[^)]*\)", flags=re.IGNORECASE)
INLINE_MARKER_RE = re.compile(r"\[(\d+)\]")


@dataclass(slots=True)
class ChunkRect:
    left: float
    top: float
    width: float
    height: float

    def as_bbox(self) -> tuple[float, float, float, float]:
        return (self.left, self.top, self.left + self.width, self.top + self.height)


@dataclass(slots=True)
class PageParagraph:
    text: str
    block_index: int | None = None
    paragraph_index: int | None = None
    page_paragraph_index: int | None = None
    rect: ChunkRect | None = None


@dataclass(slots=True)
class PageSource:
    pdf_id: str
    page: int
    page_key: str
    source_filename: str
    page_source_filename: str | None
    html_path: Path | None
    image_path: Path | None
    source_pdf_path: Path | None
    page_width: int | None = None
    page_height: int | None = None
    paragraphs: list[PageParagraph] = field(default_factory=list)


@dataclass(slots=True)
class GroundedChunk:
    id: str
    chunk_id: str
    pdf_id: str
    page: int
    label: str
    filename: str
    text: str
    chunk_index: int
    chunk_count: int
    source_filename: str
    page_source_filename: str | None
    page_width: int | None = None
    page_height: int | None = None
    rects: list[ChunkRect] = field(default_factory=list)
    score: float = 0.0
    reference_number: int = 0


def _clone_chunk(chunk: GroundedChunk, **updates: Any) -> GroundedChunk:
    return GroundedChunk(
        id=updates.get('id', chunk.id),
        chunk_id=updates.get('chunk_id', chunk.chunk_id),
        pdf_id=updates.get('pdf_id', chunk.pdf_id),
        page=updates.get('page', chunk.page),
        label=updates.get('label', chunk.label),
        filename=updates.get('filename', chunk.filename),
        text=updates.get('text', chunk.text),
        chunk_index=updates.get('chunk_index', chunk.chunk_index),
        chunk_count=updates.get('chunk_count', chunk.chunk_count),
        source_filename=updates.get('source_filename', chunk.source_filename),
        page_source_filename=updates.get('page_source_filename', chunk.page_source_filename),
        page_width=updates.get('page_width', chunk.page_width),
        page_height=updates.get('page_height', chunk.page_height),
        rects=list(updates.get('rects', chunk.rects)),
        score=float(updates.get('score', chunk.score)),
        reference_number=int(updates.get('reference_number', chunk.reference_number)),
    )


@dataclass(slots=True)
class SummaryItem:
    item_id: str
    text: str
    reference_numbers: list[int]
    title: str


class _ParagraphParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.paragraphs: list[PageParagraph] = []
        self._active_attrs: dict[str, str] | None = None
        self._buffer: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "p":
            return
        attr_map = {str(key): str(value or "") for key, value in attrs}
        if not {"data-left", "data-top", "data-width", "data-height"}.issubset(attr_map):
            return
        self._active_attrs = attr_map
        self._buffer = []

    def handle_data(self, data: str) -> None:
        if self._active_attrs is not None:
            self._buffer.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "p" or self._active_attrs is None:
            return
        text = _normalize_whitespace("".join(self._buffer))
        if text:
            rect = _coerce_rect(
                self._active_attrs.get("data-left"),
                self._active_attrs.get("data-top"),
                self._active_attrs.get("data-width"),
                self._active_attrs.get("data-height"),
            )
            self.paragraphs.append(
                PageParagraph(
                    text=text,
                    block_index=_coerce_int(self._active_attrs.get("data-block")),
                    paragraph_index=_coerce_int(self._active_attrs.get("data-paragraph")),
                    page_paragraph_index=_coerce_int(self._active_attrs.get("data-page-paragraph")),
                    rect=rect,
                )
            )
        self._active_attrs = None
        self._buffer = []


def _coerce_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except Exception:
        return None
    return parsed if parsed > 0 else None


def _coerce_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except Exception:
        return None
    return parsed if math.isfinite(parsed) else None


def _coerce_rect(left: Any, top: Any, width: Any, height: Any) -> ChunkRect | None:
    x = _coerce_float(left)
    y = _coerce_float(top)
    w = _coerce_float(width)
    h = _coerce_float(height)
    if None in {x, y, w, h}:
        return None
    assert x is not None and y is not None and w is not None and h is not None
    if w <= 0 or h <= 0:
        return None
    return ChunkRect(left=max(0.0, x), top=max(0.0, y), width=w, height=h)


def _normalize_whitespace(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("\r", " ").replace("\n", " ")).strip()


def _normalize_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", _normalize_whitespace(value).lower()).strip()


def _tokenize(value: Any) -> list[str]:
    normalized = _normalize_text(value)
    if not normalized:
        return []
    return [
        token
        for token in dict.fromkeys(normalized.split(" "))
        if len(token) > 2 and token not in STOP_WORDS
    ]


def _clean_answer_segment(value: str) -> str:
    return re.sub(r"[ \t]{2,}", " ", INLINE_MARKER_RE.sub("", SOURCE_PAREN_RE.sub("", str(value or "")))).strip()


def _format_citation_suffix(reference_numbers: Sequence[int]) -> str:
    return "".join(f"[{number}]" for number in reference_numbers if isinstance(number, int) and number > 0)


def _extract_referenced_numbers(text: str, max_reference: int) -> list[int]:
    ordered: list[int] = []
    for match in INLINE_MARKER_RE.finditer(str(text or "")):
        value = _coerce_int(match.group(1))
        if value is None or value > max_reference or value in ordered:
            continue
        ordered.append(value)
    return ordered


def _parse_requested_page(query: str) -> int | None:
    numeric_match = re.search(r"\bpage\s+(\d+)\b", str(query or ""), flags=re.IGNORECASE)
    if numeric_match:
        return _coerce_int(numeric_match.group(1))

    normalized = str(query or "").lower()
    ordinal_pages = {
        "first page": 1,
        "second page": 2,
        "third page": 3,
        "fourth page": 4,
        "fifth page": 5,
    }
    for phrase, page in ordinal_pages.items():
        if phrase in normalized:
            return page
    return None


def _build_retrieval_query(message: str, history: Sequence[dict[str, str]] | None) -> str:
    trimmed = str(message or "").strip()
    if len(trimmed) > 36 and not re.search(r"\b(this|that|it|he|she|they|those|these|there)\b", trimmed, flags=re.IGNORECASE):
        return trimmed
    recent_turns = []
    for row in list(history or [])[-MAX_HISTORY_ITEMS:]:
        role = str((row or {}).get("role") or "").strip().lower()
        content = str((row or {}).get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            recent_turns.append(f"{role.title()}: {content}")
    return f"{'\n'.join(recent_turns)}\nUser: {trimmed}".strip() if recent_turns else trimmed


def is_broad_summary_query(query: str) -> bool:
    return bool(SUMMARY_QUERY_RE.search(str(query or "")))


def _score_chunk(query: str, chunk: GroundedChunk) -> float:
    requested_page = _parse_requested_page(query)
    query_tokens = _tokenize(query)
    chunk_text = _normalize_text(chunk.text)
    score = 0.0

    if requested_page is not None:
        score += 8.0 if chunk.page == requested_page else max(-1.0, 2.0 - abs(chunk.page - requested_page))
    else:
        score += max(0.0, 1.2 - (chunk.page - 1) * 0.18)

    for token in query_tokens:
        if token not in chunk_text:
            continue
        if len(token) >= 8:
            score += 2.6
        elif len(token) >= 5:
            score += 1.8
        else:
            score += 1.1

    compact_query = _normalize_text(query)
    if compact_query and compact_query[: min(80, len(compact_query))] in chunk_text:
        score += 1.8

    numeric_tokens = list(dict.fromkeys(re.findall(r"\b\d[\d/:-]*\b", str(query or ""))))
    for token in numeric_tokens:
        if token in chunk.text:
            score += 2.4

    if is_broad_summary_query(query):
        score += 1.2 if chunk.page == (requested_page or 1) else 0.2

    score += max(0.0, 0.4 - (chunk.chunk_index - 1) * 0.01)
    return score


def _coverage_selection(candidates: Sequence[GroundedChunk], limit: int) -> list[GroundedChunk]:
    groups: dict[str, list[GroundedChunk]] = {}
    for candidate in candidates:
        key = f"{candidate.pdf_id}::{candidate.page}"
        groups.setdefault(key, []).append(candidate)

    ordered_groups = [
        group
        for _key, group in sorted(
            groups.items(),
            key=lambda row: (row[0].split("::")[0], int(row[0].split("::")[1])),
        )
    ]

    selected: list[GroundedChunk] = []
    seen: set[str] = set()
    offset = 0
    while len(selected) < limit:
        added = False
        for group in ordered_groups:
            if offset >= len(group):
                continue
            candidate = group[offset]
            dedupe_key = f"{candidate.pdf_id}::{candidate.page}::{_normalize_text(candidate.text)[:180]}"
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            selected.append(candidate)
            added = True
            if len(selected) >= limit:
                break
        if not added:
            break
        offset += 1

    if len(selected) >= limit:
        return selected

    for candidate in candidates:
        dedupe_key = f"{candidate.pdf_id}::{candidate.page}::{_normalize_text(candidate.text)[:180]}"
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        selected.append(candidate)
        if len(selected) >= limit:
            break
    return selected


def select_relevant_chunks(
    *,
    message: str,
    history: Sequence[dict[str, str]] | None,
    chunks: Sequence[GroundedChunk],
    limit: int,
) -> list[GroundedChunk]:
    candidates = list(chunks)
    if not candidates:
        return []

    retrieval_query = _build_retrieval_query(message, history)
    requested_page = _parse_requested_page(message)
    summary_query = is_broad_summary_query(message) and requested_page is None

    scored = sorted(
        (
            _clone_chunk(candidate, score=_score_chunk(retrieval_query, candidate))
            for candidate in candidates
        ),
        key=lambda candidate: (-candidate.score, candidate.page, candidate.chunk_index),
    )

    selected = (
        _coverage_selection(scored, limit)
        if summary_query
        else _dedupe_candidates(scored, limit)
    )

    if not selected:
        selected = sorted(
            candidates,
            key=lambda candidate: (
                abs(candidate.page - requested_page) if requested_page is not None else candidate.page,
                candidate.chunk_index,
            ),
        )[:limit]

    output: list[GroundedChunk] = []
    for index, chunk in enumerate(selected, start=1):
        output.append(_clone_chunk(chunk, reference_number=index))
    return output


def _dedupe_candidates(candidates: Sequence[GroundedChunk], limit: int) -> list[GroundedChunk]:
    selected: list[GroundedChunk] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = f"{candidate.pdf_id}::{candidate.page}::{_normalize_text(candidate.text)[:180]}"
        if key in seen:
            continue
        seen.add(key)
        selected.append(candidate)
        if len(selected) >= limit:
            break
    return selected


def _build_prompt(*, message: str, history: Sequence[dict[str, str]] | None, evidence: Sequence[GroundedChunk], summary_title: str | None = None) -> str:
    history_block = "\n\n".join(
        f"{'User' if str((entry or {}).get('role') or '').strip().lower() == 'user' else 'Assistant'}: {str((entry or {}).get('content') or '').strip()}"
        for entry in list(history or [])[-MAX_HISTORY_ITEMS:]
        if str((entry or {}).get('content') or '').strip()
    )

    evidence_block = "\n\n".join(
        f"[{chunk.reference_number}] Document: {chunk.label} | Page {chunk.page} | Chunk {chunk.chunk_index} of {chunk.chunk_count}\n{chunk.text[:MAX_EVIDENCE_TEXT]}"
        for chunk in evidence
    )

    lines = [
        "You answer questions about uploaded PDF evidence.",
        "Use only the evidence chunks provided below.",
        "If the answer is not supported by the evidence, say you do not see it in the uploaded PDFs.",
        "Use grouped citations in the normal OpenRAG style: put citation markers at the end of each bullet or paragraph, not after every sentence.",
        "Use citation markers like [1], [2], or [1][2]. Do not write raw '(Source: ...)' strings or raw filenames inside the answer body.",
        "Prefer concise markdown bullets for summaries unless the user explicitly asks for a paragraph answer.",
    ]
    if summary_title:
        lines.append(f"Summary title: {summary_title}")
    if history_block:
        lines.append(f"Conversation history:\n{history_block}")
    lines.append(f"Evidence chunks:\n{evidence_block}")
    lines.append(f"User question:\n{str(message or '').strip()}")
    lines.append("Answer:")
    return "\n\n".join(part for part in lines if part)


def _score_segment_against_chunk(segment: str, chunk: GroundedChunk) -> float:
    segment_tokens = _tokenize(segment)
    chunk_text = _normalize_text(chunk.text)
    score = 0.0
    for token in segment_tokens:
        if token not in chunk_text:
            continue
        if len(token) >= 8:
            score += 2.4
        elif len(token) >= 5:
            score += 1.6
        else:
            score += 1.0
    numeric_tokens = list(dict.fromkeys(re.findall(r"\b\d[\d/:-]*\b", str(segment or ""))))
    for token in numeric_tokens:
        if token in chunk.text:
            score += 2.6
    normalized_segment = _normalize_text(segment)
    if normalized_segment and normalized_segment[: min(120, len(normalized_segment))] in chunk_text:
        score += 2.0
    score += max(0.0, chunk.score * 0.1)
    return score


def _find_best_citation_numbers(segment: str, evidence: Sequence[GroundedChunk]) -> list[int]:
    scored = sorted(
        (
            {"reference_number": chunk.reference_number, "score": _score_segment_against_chunk(segment, chunk)}
            for chunk in evidence
        ),
        key=lambda row: (-float(row["score"]), int(row["reference_number"])),
    )
    top = scored[0] if scored else None
    if not top or float(top["score"]) <= 0:
        return []
    selected = [int(top["reference_number"])]
    second = scored[1] if len(scored) > 1 else None
    if second and float(second["score"]) >= max(2.25, float(top["score"]) * 0.72):
        second_number = int(second["reference_number"])
        if second_number not in selected:
            selected.append(second_number)
    return sorted(selected)


def annotate_answer_with_citations(answer: str, evidence: Sequence[GroundedChunk]) -> str:
    blocks = re.split(r"\n\n+", str(answer or ""))
    annotated_blocks: list[str] = []
    for block in blocks:
        trimmed = block.strip()
        if not trimmed:
            continue
        if re.match(r"^#{1,6}\s", trimmed):
            annotated_blocks.append(trimmed)
            continue

        lines = trimmed.split("\n")
        if lines and all(re.match(r"^\s*(?:[-*+]|\d+\.)\s+", line or "") for line in lines):
            rendered_lines: list[str] = []
            for line in lines:
                cleaned_line = _clean_answer_segment(line.strip())
                reference_numbers = _extract_referenced_numbers(line, len(evidence))
                if not reference_numbers:
                    reference_numbers = _find_best_citation_numbers(cleaned_line, evidence)
                suffix = _format_citation_suffix(reference_numbers)
                rendered_lines.append(f"{cleaned_line} {suffix}".strip() if suffix else cleaned_line)
            annotated_blocks.append("\n".join(rendered_lines))
            continue

        cleaned_block = _clean_answer_segment(trimmed)
        reference_numbers = _extract_referenced_numbers(trimmed, len(evidence))
        if not reference_numbers:
            reference_numbers = _find_best_citation_numbers(cleaned_block, evidence)
        suffix = _format_citation_suffix(reference_numbers)
        annotated_blocks.append(f"{cleaned_block} {suffix}".strip() if suffix else cleaned_block)

    return "\n\n".join(annotated_blocks).strip()


def _resolve_artifact_path(root: Path, value: Any) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = (root / path).resolve()
    return path if path.exists() else None


def _parse_evidence_text(markdown_text: str) -> str:
    raw = str(markdown_text or "")
    marker = re.search(r"^##\s+Evidence\s+Text\s*$", raw, flags=re.IGNORECASE | re.MULTILINE)
    if not marker:
        return _normalize_whitespace(raw)
    text = raw[marker.end() :].strip()
    return text.strip()


def _load_paragraphs_from_html(path: Path | None) -> list[PageParagraph]:
    if path is None or not path.exists() or path.suffix.lower() != ".html":
        return []
    parser = _ParagraphParser()
    parser.feed(path.read_text(encoding="utf-8", errors="replace"))
    return parser.paragraphs


def _page_source_from_manifest(extraction_dir: Path, page_doc: Any) -> PageSource:
    html_path = _resolve_artifact_path(extraction_dir, page_doc.relative_path)
    image_path = _resolve_artifact_path(extraction_dir, (page_doc.artifacts or {}).get("page_image"))
    source_pdf_path = _resolve_artifact_path(extraction_dir, (page_doc.metadata or {}).get("source_pdf"))
    page_width = _coerce_int((page_doc.metadata or {}).get("page_width"))
    page_height = _coerce_int((page_doc.metadata or {}).get("page_height"))
    paragraphs = _load_paragraphs_from_html(html_path)
    return PageSource(
        pdf_id=str(page_doc.pdf_id),
        page=int(page_doc.page),
        page_key=f"{page_doc.pdf_id}:{page_doc.page}",
        source_filename=str(page_doc.source_filename),
        page_source_filename=str(page_doc.source_filename),
        html_path=html_path,
        image_path=image_path,
        source_pdf_path=source_pdf_path,
        page_width=page_width,
        page_height=page_height,
        paragraphs=paragraphs,
    )


def _score_text_similarity(left: str, right: str) -> float:
    normalized_left = _normalize_text(left)
    normalized_right = _normalize_text(right)
    if not normalized_left or not normalized_right:
        return 0.0
    if normalized_left == normalized_right:
        return 1.4
    shorter, longer = (
        (normalized_left, normalized_right)
        if len(normalized_left) <= len(normalized_right)
        else (normalized_right, normalized_left)
    )
    if shorter and shorter in longer:
        return 0.95 * (len(shorter) / max(len(longer), 1)) + 0.25
    return _token_coverage_score(normalized_left, normalized_right) * 0.7 + _token_coverage_score(normalized_right, normalized_left) * 0.3


def _token_coverage_score(source: str, target: str) -> float:
    source_tokens = list({token for token in source.split() if len(token) > 1})
    target_tokens = {token for token in target.split() if len(token) > 1}
    if not source_tokens or not target_tokens:
        return 0.0
    matched_weight = 0
    total_weight = 0
    for token in source_tokens:
        total_weight += len(token)
        if token in target_tokens:
            matched_weight += len(token)
    return float(matched_weight) / float(total_weight) if total_weight > 0 else 0.0


def _paragraph_lookup(paragraphs: Sequence[PageParagraph]) -> tuple[dict[int, PageParagraph], dict[tuple[int, int], PageParagraph]]:
    by_page_paragraph: dict[int, PageParagraph] = {}
    by_block_paragraph: dict[tuple[int, int], PageParagraph] = {}
    for paragraph in paragraphs:
        if paragraph.page_paragraph_index is not None:
            by_page_paragraph[paragraph.page_paragraph_index] = paragraph
        if paragraph.block_index is not None and paragraph.paragraph_index is not None:
            by_block_paragraph[(paragraph.block_index, paragraph.paragraph_index)] = paragraph
    return by_page_paragraph, by_block_paragraph


def _match_chunk_to_paragraphs(text: str, paragraphs: Sequence[PageParagraph]) -> list[PageParagraph]:
    if not text or not paragraphs:
        return []
    snippet = _normalize_whitespace(text)
    snippet_word_count = len(_normalize_text(snippet).split())
    max_window = min(len(paragraphs), max(1, math.ceil(snippet_word_count / 45) + 3))
    best: tuple[float, int, int] | None = None
    runner_up = 0.0
    for window_size in range(1, max_window + 1):
        for start in range(0, len(paragraphs) - window_size + 1):
            combined = " ".join(paragraph.text for paragraph in paragraphs[start : start + window_size])
            score = _score_text_similarity(snippet, combined)
            if score > 0.98:
                return list(paragraphs[start : start + window_size])
            if best is None or score > best[0] + 0.02 or (abs(score - best[0]) <= 0.02 and window_size < (best[2] - best[1] + 1)):
                runner_up = best[0] if best else runner_up
                best = (score, start, start + window_size - 1)
            elif score > runner_up:
                runner_up = score
    if best and best[0] >= 0.50 and best[0] - runner_up >= 0.04:
        return list(paragraphs[best[1] : best[2] + 1])
    return []


def _resolve_chunk_rects(doc: Any, page_source: PageSource) -> list[ChunkRect]:
    paragraph_refs = (doc.metadata or {}).get("paragraph_refs") if isinstance(doc.metadata, dict) else None
    matched: list[PageParagraph] = []
    if isinstance(paragraph_refs, list) and paragraph_refs:
        by_page_paragraph, by_block_paragraph = _paragraph_lookup(page_source.paragraphs)
        seen: set[tuple[int | None, int | None, int | None]] = set()
        for ref in paragraph_refs:
            if not isinstance(ref, dict):
                continue
            page_paragraph_index = _coerce_int(ref.get("page_paragraph_index"))
            block_index = _coerce_int(ref.get("block_index"))
            paragraph_index = _coerce_int(ref.get("paragraph_index"))
            paragraph = by_page_paragraph.get(page_paragraph_index) if page_paragraph_index is not None else None
            if paragraph is None and block_index is not None and paragraph_index is not None:
                paragraph = by_block_paragraph.get((block_index, paragraph_index))
            key = (block_index, paragraph_index, page_paragraph_index)
            if paragraph is None or paragraph.rect is None or key in seen:
                continue
            seen.add(key)
            matched.append(paragraph)
    if not matched:
        matched = _match_chunk_to_paragraphs(str(getattr(doc, "text_preview", "") or ""), page_source.paragraphs)
    if not matched:
        chunk_path = _resolve_artifact_path(page_source.html_path.parent if page_source.html_path else Path.cwd(), getattr(doc, "relative_path", None))
        if chunk_path and chunk_path.exists():
            matched = _match_chunk_to_paragraphs(_parse_evidence_text(chunk_path.read_text(encoding="utf-8", errors="replace")), page_source.paragraphs)
    return [paragraph.rect for paragraph in matched if paragraph.rect is not None]




def _join_chunk_text(parts: Sequence[str]) -> str:
    fragments: list[str] = []
    for raw in parts:
        value = _normalize_whitespace(raw)
        if not value:
            continue
        previous = fragments[-1] if fragments else ""
        needs_space = bool(previous) and not previous.endswith("\n") and not re.match(r'^[,.;:!?)]', value) and not re.search(r'[(/-]$', previous)
        fragments.append(f" {value}" if needs_space else value)
        fragments.append("\n")
    return "".join(fragments).replace(" \n", "\n").replace("\n\n\n", "\n\n").strip()

def _merge_chunk_rects(rects: Sequence[ChunkRect]) -> list[ChunkRect]:
    ordered = sorted(rects, key=lambda rect: (rect.top, rect.left))
    merged: list[ChunkRect] = []
    for rect in ordered:
        previous = merged[-1] if merged else None
        if previous is not None and abs(previous.top - rect.top) <= max(6.0, rect.height * 0.55) and rect.left <= previous.left + previous.width + 20.0:
            right = max(previous.left + previous.width, rect.left + rect.width)
            bottom = max(previous.top + previous.height, rect.top + rect.height)
            previous.left = min(previous.left, rect.left)
            previous.top = min(previous.top, rect.top)
            previous.width = right - previous.left
            previous.height = bottom - previous.top
            continue
        merged.append(ChunkRect(left=rect.left, top=rect.top, width=rect.width, height=rect.height))
    return merged


def _build_page_chunks(page_source: PageSource, *, chunk_size: int, chunk_overlap: int) -> list[GroundedChunk]:
    paragraphs = [paragraph for paragraph in page_source.paragraphs if _normalize_whitespace(paragraph.text)]
    if not paragraphs:
        return []

    size = max(300, int(chunk_size or DEFAULT_CHUNK_SIZE))
    overlap = max(0, min(size - 50, int(chunk_overlap or DEFAULT_CHUNK_OVERLAP)))
    chunks: list[GroundedChunk] = []
    start_index = 0
    chunk_offset = 0

    while start_index < len(paragraphs):
        end_index = start_index
        current_chars = 0
        while end_index < len(paragraphs) and current_chars < size:
            current_chars += len(_normalize_whitespace(paragraphs[end_index].text)) + 1
            end_index += 1
        if end_index <= start_index:
            end_index = min(len(paragraphs), start_index + 1)

        chunk_paragraphs = paragraphs[start_index:end_index]
        chunk_text = _join_chunk_text([paragraph.text for paragraph in chunk_paragraphs])
        if chunk_text:
            rects = _merge_chunk_rects([paragraph.rect for paragraph in chunk_paragraphs if paragraph.rect is not None])
            chunk_offset += 1
            chunk_id = f"{page_source.pdf_id}:{page_source.page}:chunk:{chunk_offset}"
            chunks.append(
                GroundedChunk(
                    id=chunk_id,
                    chunk_id=chunk_id,
                    pdf_id=page_source.pdf_id,
                    page=page_source.page,
                    label=f"{page_source.pdf_id} p.{page_source.page}",
                    filename=page_source.source_filename,
                    text=chunk_text,
                    chunk_index=chunk_offset,
                    chunk_count=0,
                    source_filename=page_source.source_filename,
                    page_source_filename=page_source.page_source_filename,
                    page_width=page_source.page_width,
                    page_height=page_source.page_height,
                    rects=rects,
                )
            )

        if end_index >= len(paragraphs):
            break

        overlap_chars = 0
        next_start = end_index
        while next_start > start_index and overlap_chars < overlap:
            next_start -= 1
            overlap_chars += len(_normalize_whitespace(paragraphs[next_start].text)) + 1
        start_index = max(start_index + 1, next_start)

    return chunks


def _materialized_manifest_chunks(
    *,
    manifest: MaterializationManifest,
    manifest_path: Path,
    page_sources: dict[tuple[str, int], PageSource],
    allowed_pages: set[tuple[str, int]],
) -> list[GroundedChunk]:
    extraction_dir = manifest_path.parent
    filtered_docs = [
        doc
        for doc in manifest.ingest_documents()
        if not allowed_pages or (str(doc.pdf_id), int(doc.page)) in allowed_pages
    ]

    chunk_counts: dict[tuple[str, int], int] = {}
    for doc in filtered_docs:
        key = (str(doc.pdf_id), int(doc.page))
        chunk_counts[key] = chunk_counts.get(key, 0) + 1

    chunks: list[GroundedChunk] = []
    for doc in filtered_docs:
        chunk_path = _resolve_artifact_path(extraction_dir, doc.relative_path)
        if chunk_path is None or not chunk_path.exists():
            continue
        raw_text = chunk_path.read_text(encoding="utf-8", errors="replace")
        text = _parse_evidence_text(raw_text).strip()
        if not text:
            continue
        key = (str(doc.pdf_id), int(doc.page))
        page_source = page_sources.get(key)
        rects = _resolve_chunk_rects(doc, page_source) if page_source else []
        label = f"{doc.pdf_id} p.{doc.page}"
        chunk_index = _coerce_int(getattr(doc, "chunk_index", None)) or len([row for row in chunks if row.pdf_id == str(doc.pdf_id) and row.page == int(doc.page)]) + 1
        chunk_count = chunk_counts.get(key, chunk_index)
        chunk_id = str(doc.source_filename)
        chunks.append(
            GroundedChunk(
                id=chunk_id,
                chunk_id=chunk_id,
                pdf_id=str(doc.pdf_id),
                page=int(doc.page),
                label=label,
                filename=str(doc.source_filename),
                text=text,
                chunk_index=chunk_index,
                chunk_count=chunk_count,
                source_filename=str(doc.source_filename),
                page_source_filename=page_source.page_source_filename if page_source else None,
                page_width=page_source.page_width if page_source else None,
                page_height=page_source.page_height if page_source else None,
                rects=rects,
            )
        )
    return chunks
def load_scope_chunks(
    *,
    manifest_path: Path,
    page_refs: Iterable[tuple[str, int]] | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> tuple[list[GroundedChunk], dict[tuple[str, int], PageSource], MaterializationManifest]:
    manifest = MaterializationManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    extraction_dir = manifest_path.parent
    allowed_pages = {(str(pdf_id), int(page)) for pdf_id, page in (page_refs or []) if str(pdf_id).strip() and _coerce_int(page)}
    page_lookup = manifest.page_lookup()

    page_sources: dict[tuple[str, int], PageSource] = {}
    for key, page_doc in page_lookup.items():
        normalized_key = (str(key[0]), int(key[1]))
        if allowed_pages and normalized_key not in allowed_pages:
            continue
        page_sources[normalized_key] = _page_source_from_manifest(extraction_dir, page_doc)

    grouped_page_chunks: dict[str, list[GroundedChunk]] = {}
    built_from_pages = False
    for key in sorted(page_sources.keys(), key=lambda row: (row[0], row[1])):
        page_chunks = _build_page_chunks(page_sources[key], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if not page_chunks:
            continue
        built_from_pages = True
        grouped_page_chunks.setdefault(key[0], []).extend(page_chunks)

    if built_from_pages:
        chunks: list[GroundedChunk] = []
        for pdf_id in sorted(grouped_page_chunks.keys()):
            pdf_chunks = grouped_page_chunks[pdf_id]
            chunk_count = len(pdf_chunks)
            for chunk_index, chunk in enumerate(pdf_chunks, start=1):
                chunks.append(_clone_chunk(chunk, chunk_index=chunk_index, chunk_count=chunk_count))
        return chunks, page_sources, manifest

    chunks = _materialized_manifest_chunks(
        manifest=manifest,
        manifest_path=manifest_path,
        page_sources=page_sources,
        allowed_pages=allowed_pages,
    )
    return chunks, page_sources, manifest


def _rect_to_box(rect: ChunkRect, *, text: str = "", paragraph_index: int | None = None) -> dict[str, Any]:
    bbox = {
        "left": int(round(rect.left)),
        "top": int(round(rect.top)),
        "right": int(round(rect.left + rect.width)),
        "bottom": int(round(rect.top + rect.height)),
        "width": int(round(rect.width)),
        "height": int(round(rect.height)),
    }
    payload: dict[str, Any] = {"text": text, "bbox": bbox}
    if paragraph_index is not None:
        payload["page_paragraph_index"] = paragraph_index
    return payload


def _union_rect(rects: Sequence[ChunkRect]) -> dict[str, int]:
    if not rects:
        return {}
    x0 = min(rect.left for rect in rects)
    y0 = min(rect.top for rect in rects)
    x1 = max(rect.left + rect.width for rect in rects)
    y1 = max(rect.top + rect.height for rect in rects)
    return {
        "left": int(round(x0)),
        "top": int(round(y0)),
        "right": int(round(x1)),
        "bottom": int(round(y1)),
        "width": int(round(max(0.0, x1 - x0))),
        "height": int(round(max(0.0, y1 - y0))),
    }


def build_citation_catalog(
    *,
    evidence: Sequence[GroundedChunk],
    page_sources: dict[tuple[str, int], PageSource],
) -> list[dict[str, Any]]:
    catalog: list[dict[str, Any]] = []
    for chunk in evidence:
        page_source = page_sources.get((chunk.pdf_id, chunk.page))
        boxes = [_rect_to_box(rect, text=chunk.text[:240]) for rect in chunk.rects]
        catalog.append(
            {
                "id": chunk.id,
                "number": chunk.reference_number,
                "chunk_id": chunk.chunk_id,
                "label": chunk.label,
                "pdf_id": chunk.pdf_id,
                "page": chunk.page,
                "snippet": _normalize_whitespace(chunk.text[:240]),
                "anchor": f"{chunk.pdf_id}:{chunk.page}:{chunk.chunk_id}",
                "page_key": f"{chunk.pdf_id}:{chunk.page}",
                "source_filename": chunk.source_filename,
                "page_source_filename": chunk.page_source_filename,
                "page_image_path": page_source.image_path.as_posix() if page_source and page_source.image_path else None,
                "source_pdf_path": page_source.source_pdf_path.as_posix() if page_source and page_source.source_pdf_path else None,
                "page_width": chunk.page_width,
                "page_height": chunk.page_height,
                "bbox": _union_rect(chunk.rects),
                "boxes": boxes,
                "degraded": False,
                "degraded_reason": None,
            }
        )
    return catalog


def _extract_summary_items(answer: str) -> list[SummaryItem]:
    blocks = re.split(r"\n\n+", str(answer or "").strip())
    items: list[SummaryItem] = []
    current_title = "Supported Summary"
    item_index = 1
    for block in blocks:
        trimmed = block.strip()
        if not trimmed:
            continue
        if re.match(r"^#{1,6}\s", trimmed):
            current_title = re.sub(r"^#{1,6}\s*", "", trimmed).strip() or current_title
            continue
        lines = [line.strip() for line in trimmed.split("\n") if line.strip()]
        bullet_lines = [line for line in lines if re.match(r"^(?:[-*+]|\d+\.)\s+", line)]
        target_lines = bullet_lines if bullet_lines and len(bullet_lines) == len(lines) else [trimmed]
        for line in target_lines:
            cleaned = re.sub(r"^(?:[-*+]|\d+\.)\s+", "", line).strip()
            ref_numbers = _extract_referenced_numbers(cleaned, 999)
            cleaned = _clean_answer_segment(cleaned)
            if not cleaned:
                continue
            items.append(
                SummaryItem(
                    item_id=f"item-{item_index}",
                    text=cleaned,
                    reference_numbers=ref_numbers,
                    title=current_title,
                )
            )
            item_index += 1
    return items


def build_native_summary_data(
    *,
    title: str,
    answer: str,
    evidence: Sequence[GroundedChunk],
    page_sources: dict[tuple[str, int], PageSource],
    artifacts: dict[str, Any] | None = None,
    urls: dict[str, Any] | None = None,
) -> dict[str, Any]:
    citation_index = build_citation_catalog(evidence=evidence, page_sources=page_sources)
    citation_by_number = {int(entry["number"]): entry for entry in citation_index}
    items = _extract_summary_items(answer)

    sections_by_title: dict[str, dict[str, Any]] = {}
    citation_instances: list[dict[str, Any]] = []
    page_summaries_by_key: dict[tuple[str, int], list[str]] = {}

    for item in items:
        section = sections_by_title.setdefault(
            item.title,
            {
                "section_id": re.sub(r"[^a-z0-9]+", "-", item.title.lower()).strip("-") or "supported-summary",
                "title": item.title,
                "kind": "supported_summary",
                "debug_only": False,
                "items": [],
            },
        )
        citation_ids: list[str] = []
        citation_numbers: list[int] = []
        primary_pdf_id = None
        primary_page = None
        for number in item.reference_numbers:
            citation = citation_by_number.get(number)
            if not citation:
                continue
            citation_ids.append(str(citation["id"]))
            citation_numbers.append(int(number))
            if primary_pdf_id is None:
                primary_pdf_id = str(citation["pdf_id"])
                primary_page = int(citation["page"])
            citation_instances.append(
                {
                    **citation,
                    "sentence_id": item.item_id,
                    "sentence_text": item.text,
                    "primary_evidence_id": citation["id"],
                    "secondary_evidence_ids": [],
                }
            )
        if primary_pdf_id and primary_page:
            page_summaries_by_key.setdefault((primary_pdf_id, primary_page), []).append(item.text)
        section["items"].append(
            {
                "item_id": item.item_id,
                "text": item.text,
                "citation_ids": citation_ids,
                "citation_numbers": citation_numbers,
                "pdf_id": primary_pdf_id,
                "page": primary_page,
            }
        )

    source_pages: list[dict[str, Any]] = []
    seen_pages: set[tuple[str, int]] = set()
    for citation in citation_index:
        key = (str(citation["pdf_id"]), int(citation["page"]))
        if key in seen_pages:
            continue
        seen_pages.add(key)
        page_source = page_sources.get(key)
        source_pages.append(
            {
                "page_key": f"{key[0]}:{key[1]}",
                "pdf_id": key[0],
                "page": key[1],
                "source_filename": page_source.source_filename if page_source else citation.get("page_source_filename") or citation.get("source_filename"),
                "page_source_filename": page_source.page_source_filename if page_source else citation.get("page_source_filename"),
                "image_path": page_source.image_path.as_posix() if page_source and page_source.image_path else None,
                "html_path": page_source.html_path.as_posix() if page_source and page_source.html_path else None,
                "source_pdf_path": page_source.source_pdf_path.as_posix() if page_source and page_source.source_pdf_path else citation.get("source_pdf_path"),
                "width": page_source.page_width if page_source else citation.get("page_width"),
                "height": page_source.page_height if page_source else citation.get("page_height"),
                "paragraph_count": len(page_source.paragraphs) if page_source else None,
            }
        )

    page_summaries = [
        {
            "pdf_id": pdf_id,
            "page": page,
            "supported_summary": " ".join(texts).strip(),
            "summary": " ".join(texts).strip(),
            "supported_key_facts": [],
            "key_facts": [],
        }
        for (pdf_id, page), texts in sorted(page_summaries_by_key.items(), key=lambda row: (row[0][0], row[0][1]))
        if texts
    ]

    return {
        "title": title,
        "summary": {
            "draft_title": title,
            "draft_summary": answer,
            "supported_summary": answer,
            "page_summaries": page_summaries,
            "presentation_layer": None,
            "presentation_draft": None,
        },
        "citation_index": citation_index,
        "citation_instances": citation_instances,
        "sections": list(sections_by_title.values()),
        "source_pages": source_pages,
        "artifacts": dict(artifacts or {}),
        "urls": dict(urls or {}),
    }


async def generate_grounded_response(
    gateway: Any,
    *,
    manifest_path: Path,
    message: str,
    title: str,
    page_refs: Iterable[tuple[str, int]] | None = None,
    history: Sequence[dict[str, str]] | None = None,
    max_context_chunks: int | None = None,
    llm_model: str | None = None,
    llm_provider: str | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> dict[str, Any]:
    chunks, page_sources, _manifest = load_scope_chunks(
        manifest_path=manifest_path,
        page_refs=page_refs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    if not chunks:
        raise RuntimeError("No grounded retrieval chunks were materialized for the selected pages.")

    selected = select_relevant_chunks(
        message=message,
        history=history,
        chunks=chunks,
        limit=max_context_chunks or (DEFAULT_SUMMARY_CONTEXT_CHUNKS if is_broad_summary_query(message) else DEFAULT_CHAT_CONTEXT_CHUNKS),
    )
    if not selected:
        raise RuntimeError("No grounded retrieval chunks matched the selected pages.")

    prompt = _build_prompt(message=message, history=history, evidence=selected, summary_title=title)
    payload = await gateway.chat_request(
        message=prompt,
        limit=0,
        score_threshold=999,
        llm_model=llm_model,
        llm_provider=llm_provider,
    )
    raw_answer = str((payload or {}).get("response") or "").replace("\r", "").strip()
    if not raw_answer:
        raw_answer = "I do not see that in the uploaded PDFs."
    annotated_answer = annotate_answer_with_citations(raw_answer, selected)
    referenced_numbers = _extract_referenced_numbers(annotated_answer, len(selected))
    ordered_citations = [selected[number - 1] for number in referenced_numbers if 1 <= number <= len(selected)] if referenced_numbers else list(selected)

    native_summary_data = build_native_summary_data(
        title=title,
        answer=annotated_answer,
        evidence=ordered_citations,
        page_sources=page_sources,
        artifacts={"manifest_path": manifest_path.as_posix()},
    )

    citation_catalog = native_summary_data.get("citation_index", []) if isinstance(native_summary_data, dict) else []
    return {
        "answer": annotated_answer,
        "raw_answer": raw_answer,
        "chat_id": str((payload or {}).get("chat_id") or "").strip() or None,
        "selected_chunks": [
            {
                "reference_number": chunk.reference_number,
                "chunk_id": chunk.chunk_id,
                "pdf_id": chunk.pdf_id,
                "page": chunk.page,
                "chunk_index": chunk.chunk_index,
                "chunk_count": chunk.chunk_count,
                "label": chunk.label,
                "score": chunk.score,
                "text_preview": _normalize_whitespace(chunk.text[:240]),
            }
            for chunk in selected
        ],
        "citations": citation_catalog,
        "native_summary_data": native_summary_data,
        "source_count": len(ordered_citations),
        "debug": {
            "selected_chunk_count": len(selected),
            "all_chunk_count": len(chunks),
            "selected_chunks": [
                {
                    "reference_number": chunk.reference_number,
                    "chunk_id": chunk.chunk_id,
                    "pdf_id": chunk.pdf_id,
                    "page": chunk.page,
                    "chunk_index": chunk.chunk_index,
                    "chunk_count": chunk.chunk_count,
                    "score": chunk.score,
                }
                for chunk in selected
            ],
            "openrag_http_debug": gateway.export_debug_events() if hasattr(gateway, "export_debug_events") else [],
        },
    }
