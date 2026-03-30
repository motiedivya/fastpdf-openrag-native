from __future__ import annotations

import hashlib
import re
import shutil
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable

from .models import (
    CitationBox,
    CitationIndexEntry,
    CitationSection,
    CitationSentenceItem,
    CitationSourcePage,
    MaterializationManifest,
    ResolvedCitations,
    ScopedSummaryResult,
    VerifiedSentence,
)

TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_/-]*")
STOPWORDS = {
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
    "with",
}
SOURCE_TAIL_RE = re.compile(r"\s*\((?:Source|Sources)\s*:[^)]+\)\s*$", flags=re.IGNORECASE)
PAGE_AS_ABOVE_RE = re.compile(r"\s*\(Sources?\s+as\s+above\)\s*$", flags=re.IGNORECASE)
PAGE_PREFIX_RE = re.compile(r"^Page\s+(\d+)\s*(?:\([^)]*\))?\s*:\s*", flags=re.IGNORECASE)


@dataclass(slots=True)
class _GroundingItem:
    item_id: str
    text: str
    section_id: str
    candidate_filenames: list[str]
    preferred_filenames: list[str] = field(default_factory=list)
    expected_pdf_id: str | None = None
    expected_page: int | None = None
    supported: bool | None = None
    debug_only: bool = False


@dataclass(slots=True)
class _ChunkDocument:
    source_filename: str
    pdf_id: str
    page: int
    relative_path: str
    page_source_filename: str | None
    text: str
    metadata: dict[str, Any]


@dataclass(slots=True)
class _PageAsset:
    page_key: str
    pdf_id: str
    page: int
    source_filename: str
    image_path: str | None
    html_path: str | None
    source_pdf_path: str | None
    width: int | None
    height: int | None
    paragraph_count: int | None
    paragraphs: list[dict[str, Any]] = field(default_factory=list)
    by_block_paragraph: dict[tuple[int, int], dict[str, Any]] = field(default_factory=dict)
    by_page_paragraph: dict[int, dict[str, Any]] = field(default_factory=dict)


class _ParagraphHtmlParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.paragraphs: list[dict[str, Any]] = []
        self._active_attrs: dict[str, str] | None = None
        self._active_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "p":
            return
        attr_map = {key: value or "" for key, value in attrs}
        if "data-block" not in attr_map and "data-paragraph" not in attr_map:
            return
        self._active_attrs = attr_map
        self._active_parts = []

    def handle_data(self, data: str) -> None:
        if self._active_attrs is not None:
            self._active_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "p" or self._active_attrs is None:
            return
        attrs = self._active_attrs
        text = _normalize_space("".join(self._active_parts))
        if text:
            bbox = {
                "left": _safe_int(attrs.get("data-left")),
                "top": _safe_int(attrs.get("data-top")),
                "width": _safe_int(attrs.get("data-width")),
                "height": _safe_int(attrs.get("data-height")),
            }
            bbox["right"] = bbox["left"] + bbox["width"]
            bbox["bottom"] = bbox["top"] + bbox["height"]
            self.paragraphs.append(
                {
                    "block_index": _safe_int(attrs.get("data-block")),
                    "paragraph_index": _safe_int(attrs.get("data-paragraph")),
                    "page_paragraph_index": _safe_int(attrs.get("data-page-paragraph")) or (len(self.paragraphs) + 1),
                    "text": text,
                    "bbox": bbox,
                }
            )
        self._active_attrs = None
        self._active_parts = []


def _safe_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _clean_grounding_text(text: str) -> str:
    cleaned = PAGE_AS_ABOVE_RE.sub("", SOURCE_TAIL_RE.sub("", str(text or ""))).strip()
    cleaned = PAGE_PREFIX_RE.sub("", cleaned).strip()
    return _normalize_space(cleaned)


def _normalize_for_score(text: str) -> str:
    return re.sub(r"[^a-z0-9\s]+", " ", _clean_grounding_text(text).lower()).strip()


def _tokenize(text: str) -> list[str]:
    return [token for token in TOKEN_RE.findall(_normalize_for_score(text)) if len(token) > 1 and token not in STOPWORDS]


def _token_coverage(source: str, target: str) -> float:
    source_tokens = list(dict.fromkeys(_tokenize(source)))
    target_set = set(_tokenize(target))
    if not source_tokens or not target_set:
        return 0.0
    matched = sum(len(token) for token in source_tokens if token in target_set)
    total = sum(len(token) for token in source_tokens)
    return matched / max(total, 1)


def _score_text_similarity(left: str, right: str) -> float:
    normalized_left = _normalize_for_score(left)
    normalized_right = _normalize_for_score(right)
    if not normalized_left or not normalized_right:
        return 0.0
    if normalized_left == normalized_right:
        return 1.5
    shorter, longer = (
        (normalized_left, normalized_right)
        if len(normalized_left) <= len(normalized_right)
        else (normalized_right, normalized_left)
    )
    if shorter and shorter in longer:
        return 0.9 * (len(shorter) / max(len(longer), 1)) + 0.25
    forward = _token_coverage(normalized_left, normalized_right)
    reverse = _token_coverage(normalized_right, normalized_left)
    return (forward * 0.7) + (reverse * 0.3)


def _split_sentences(text: str) -> list[str]:
    clean = _normalize_space(text)
    if not clean:
        return []
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", clean) if part.strip()]


def _split_grounding_units(text: str) -> list[str]:
    units: list[str] = []
    for sentence in _split_sentences(text):
        sentence = sentence.strip()
        if sentence.count(";") >= 1:
            parts = [part.strip(" ;") for part in sentence.split(";") if part.strip(" ;")]
            if len(parts) > 1:
                for part in parts:
                    unit = part.strip()
                    if unit and unit[0].islower():
                        unit = unit[0].upper() + unit[1:]
                    if unit and not re.search(r"[.!?]$", unit):
                        unit = f"{unit}."
                    if unit:
                        units.append(unit)
                continue
        units.append(sentence)
    return units


def _page_key(pdf_id: str, page: int) -> str:
    return f"{pdf_id}::{page}"


def _extract_page_hint(text: str) -> int | None:
    match = re.search(r"\bpage\s+(\d+)\b", str(text or ""), flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _extract_evidence_text(markdown: str) -> str:
    marker = re.search(r"^## Evidence Text\s*$", markdown, flags=re.MULTILINE)
    if not marker:
        return _normalize_space(markdown)
    body = markdown[marker.end() :].strip()
    body = re.sub(r"^###\s+", "", body, flags=re.MULTILINE)
    return body.strip()


def _load_html_paragraphs(html_path: Path) -> list[dict[str, Any]]:
    parser = _ParagraphHtmlParser()
    parser.feed(html_path.read_text(encoding="utf-8"))
    return parser.paragraphs


def _copy_source_pdf(source_pdf: Path | None, extraction_dir: Path) -> Path | None:
    if source_pdf is None or not source_pdf.exists():
        return None
    artifacts_dir = extraction_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    destination = artifacts_dir / source_pdf.name
    if destination.resolve() != source_pdf.resolve():
        shutil.copy2(source_pdf, destination)
    return destination


def _build_page_assets(
    manifest: MaterializationManifest,
    *,
    extraction_dir: Path,
    source_pdf_copy_path: Path | None,
) -> tuple[dict[tuple[str, int], _PageAsset], dict[str, _PageAsset], list[CitationSourcePage]]:
    by_key: dict[tuple[str, int], _PageAsset] = {}
    by_source: dict[str, _PageAsset] = {}
    source_pages: list[CitationSourcePage] = []

    for page in manifest.page_documents:
        html_path = (extraction_dir / page.relative_path).resolve()
        image_path = None
        if page.artifacts.get("page_image"):
            image_path = (extraction_dir / page.artifacts["page_image"]).resolve().as_posix()
        paragraphs = _load_html_paragraphs(html_path) if html_path.exists() else []
        by_block_paragraph = {
            (paragraph["block_index"], paragraph["paragraph_index"]): paragraph
            for paragraph in paragraphs
            if paragraph.get("block_index") and paragraph.get("paragraph_index")
        }
        by_page_paragraph = {
            paragraph["page_paragraph_index"]: paragraph
            for paragraph in paragraphs
            if paragraph.get("page_paragraph_index")
        }
        asset = _PageAsset(
            page_key=_page_key(page.pdf_id, page.page),
            pdf_id=page.pdf_id,
            page=page.page,
            source_filename=page.source_filename,
            image_path=image_path,
            html_path=html_path.as_posix(),
            source_pdf_path=source_pdf_copy_path.as_posix() if source_pdf_copy_path else None,
            width=int(page.metadata.get("page_width") or 0) or None,
            height=int(page.metadata.get("page_height") or 0) or None,
            paragraph_count=int(page.metadata.get("paragraph_count") or 0) or len(paragraphs) or None,
            paragraphs=paragraphs,
            by_block_paragraph=by_block_paragraph,
            by_page_paragraph=by_page_paragraph,
        )
        by_key[(page.pdf_id, page.page)] = asset
        by_source[page.source_filename] = asset
        source_pages.append(
            CitationSourcePage(
                page_key=asset.page_key,
                pdf_id=asset.pdf_id,
                page=asset.page,
                source_filename=asset.source_filename,
                page_source_filename=asset.source_filename,
                image_path=asset.image_path,
                html_path=asset.html_path,
                source_pdf_path=asset.source_pdf_path,
                width=asset.width,
                height=asset.height,
                paragraph_count=asset.paragraph_count,
            )
        )
    source_pages.sort(key=lambda row: (row.pdf_id, row.page))
    return by_key, by_source, source_pages


def _build_chunk_lookup(manifest: MaterializationManifest, extraction_dir: Path) -> dict[str, _ChunkDocument]:
    lookup: dict[str, _ChunkDocument] = {}
    for document in manifest.ingest_documents():
        path = (extraction_dir / document.relative_path).resolve()
        text = document.text_preview
        if path.exists():
            text = _extract_evidence_text(path.read_text(encoding="utf-8"))
        lookup[document.source_filename] = _ChunkDocument(
            source_filename=document.source_filename,
            pdf_id=document.pdf_id,
            page=document.page,
            relative_path=document.relative_path,
            page_source_filename=document.parent_source_filename,
            text=text,
            metadata=dict(document.metadata),
        )
    return lookup


def _dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _page_sources_by_key(manifest: MaterializationManifest) -> dict[tuple[str, int], list[str]]:
    mapping: dict[tuple[str, int], list[str]] = {}
    for page in manifest.page_documents:
        mapping[(page.pdf_id, page.page)] = list(page.retrieval_sources())
    return mapping


def _section_item_id(section_id: str, index: int, part_index: int = 1) -> str:
    return f"{section_id}__i{index:04d}__p{part_index:02d}"


def _expand_verified_items(
    *,
    section_id: str,
    values: list[VerifiedSentence],
    candidate_filenames: list[str],
    default_pdf_id: str | None = None,
    default_page: int | None = None,
    debug_only: bool = False,
) -> list[_GroundingItem]:
    items: list[_GroundingItem] = []
    for sentence_index, value in enumerate(values, start=1):
        for part_index, text in enumerate(_split_grounding_units(value.sentence), start=1):
            items.append(
                _GroundingItem(
                    item_id=_section_item_id(section_id, sentence_index, part_index),
                    text=text,
                    section_id=section_id,
                    candidate_filenames=list(candidate_filenames),
                    preferred_filenames=_dedupe_preserve_order(hit.filename for hit in value.evidence),
                    expected_pdf_id=default_pdf_id,
                    expected_page=_extract_page_hint(text) or default_page,
                    supported=value.supported,
                    debug_only=debug_only,
                )
            )
    return items


def _expand_text_items(
    *,
    section_id: str,
    values: Iterable[str],
    candidate_filenames: list[str],
    default_pdf_id: str | None = None,
    default_page: int | None = None,
    supported: bool | None = None,
    preferred_filenames: list[str] | None = None,
    debug_only: bool = False,
) -> list[_GroundingItem]:
    items: list[_GroundingItem] = []
    counter = 0
    preferred = list(preferred_filenames or [])
    for value in values:
        for part_index, text in enumerate(_split_grounding_units(str(value or "")), start=1):
            counter += 1
            items.append(
                _GroundingItem(
                    item_id=_section_item_id(section_id, counter, part_index),
                    text=text,
                    section_id=section_id,
                    candidate_filenames=list(candidate_filenames),
                    preferred_filenames=list(preferred),
                    expected_pdf_id=default_pdf_id,
                    expected_page=_extract_page_hint(text) or default_page,
                    supported=supported,
                    debug_only=debug_only,
                )
            )
    return items


def _build_sections(
    *,
    manifest: MaterializationManifest,
    summary: ScopedSummaryResult,
) -> list[tuple[CitationSection, list[_GroundingItem]]]:
    sections: list[tuple[CitationSection, list[_GroundingItem]]] = []
    page_source_map = _page_sources_by_key(manifest)
    manifest_pages = {(page.pdf_id, page.page): page for page in manifest.page_documents}
    default_pdf_id = manifest.page_documents[0].pdf_id if manifest.page_documents else None
    scoped_sources = list(summary.source_filenames)

    if summary.verified_sentences:
        section = CitationSection(section_id="supported-summary", title="Supported Summary", kind="supported_summary")
        items = _expand_verified_items(
            section_id=section.section_id,
            values=summary.verified_sentences,
            candidate_filenames=scoped_sources,
            default_pdf_id=default_pdf_id,
        )
        if items:
            sections.append((section, items))

    if summary.chronology:
        section = CitationSection(section_id="chronology", title="Chronology", kind="chronology")
        items: list[_GroundingItem] = []
        for item in _expand_text_items(
            section_id=section.section_id,
            values=summary.chronology,
            candidate_filenames=scoped_sources,
            default_pdf_id=default_pdf_id,
        ):
            if item.expected_page is not None and item.expected_pdf_id is not None:
                item.candidate_filenames = page_source_map.get((item.expected_pdf_id, item.expected_page), item.candidate_filenames)
            items.append(item)
        if items:
            sections.append((section, items))

    for page_summary in sorted(summary.page_summaries, key=lambda row: (row.pdf_id, row.page)):
        page_key = (page_summary.pdf_id, page_summary.page)
        candidate_filenames = page_source_map.get(page_key, [])
        page_title = f"Page {page_summary.page} Summary"
        if page_summary.verified_sentences:
            section = CitationSection(
                section_id=f"page-{page_summary.page}-summary",
                title=page_title,
                kind="page_summary",
                pdf_id=page_summary.pdf_id,
                page=page_summary.page,
            )
            items = _expand_verified_items(
                section_id=section.section_id,
                values=page_summary.verified_sentences,
                candidate_filenames=candidate_filenames,
                default_pdf_id=page_summary.pdf_id,
                default_page=page_summary.page,
            )
            if items:
                sections.append((section, items))
        elif page_summary.supported_summary or page_summary.summary:
            section = CitationSection(
                section_id=f"page-{page_summary.page}-summary",
                title=page_title,
                kind="page_summary",
                pdf_id=page_summary.pdf_id,
                page=page_summary.page,
            )
            items = _expand_text_items(
                section_id=section.section_id,
                values=[page_summary.supported_summary or page_summary.summary],
                candidate_filenames=candidate_filenames,
                default_pdf_id=page_summary.pdf_id,
                default_page=page_summary.page,
                supported=page_summary.passed_verification,
                preferred_filenames=[hit.filename for hit in page_summary.retrieved_sources],
            )
            if items:
                sections.append((section, items))

        visible_key_facts = page_summary.supported_key_facts or page_summary.key_facts
        if visible_key_facts:
            section = CitationSection(
                section_id=f"page-{page_summary.page}-key-facts",
                title=f"Page {page_summary.page} Key Facts",
                kind="key_facts",
                pdf_id=page_summary.pdf_id,
                page=page_summary.page,
            )
            items = _expand_text_items(
                section_id=section.section_id,
                values=visible_key_facts,
                candidate_filenames=candidate_filenames,
                default_pdf_id=page_summary.pdf_id,
                default_page=page_summary.page,
                supported=True,
                preferred_filenames=[hit.filename for hit in page_summary.retrieved_sources],
            )
            if items:
                sections.append((section, items))

        if page_summary.unsupported_key_facts:
            section = CitationSection(
                section_id=f"page-{page_summary.page}-unsupported-key-facts",
                title=f"Page {page_summary.page} Unsupported Key Facts",
                kind="unsupported_key_facts",
                pdf_id=page_summary.pdf_id,
                page=page_summary.page,
                debug_only=True,
            )
            items = _expand_text_items(
                section_id=section.section_id,
                values=page_summary.unsupported_key_facts,
                candidate_filenames=candidate_filenames,
                default_pdf_id=page_summary.pdf_id,
                default_page=page_summary.page,
                supported=False,
                preferred_filenames=[hit.filename for hit in page_summary.retrieved_sources],
                debug_only=True,
            )
            if items:
                sections.append((section, items))

        if page_summary.unsupported_sentences:
            section = CitationSection(
                section_id=f"page-{page_summary.page}-unsupported",
                title=f"Page {page_summary.page} Unsupported",
                kind="unsupported_page_summary",
                pdf_id=page_summary.pdf_id,
                page=page_summary.page,
                debug_only=True,
            )
            items = _expand_text_items(
                section_id=section.section_id,
                values=page_summary.unsupported_sentences,
                candidate_filenames=candidate_filenames,
                default_pdf_id=page_summary.pdf_id,
                default_page=page_summary.page,
                supported=False,
                preferred_filenames=[hit.filename for hit in page_summary.retrieved_sources],
                debug_only=True,
            )
            if items:
                sections.append((section, items))

    if summary.draft_summary:
        section = CitationSection(
            section_id="draft-summary",
            title="Draft Summary",
            kind="draft_summary",
            debug_only=True,
        )
        items = _expand_text_items(
            section_id=section.section_id,
            values=[summary.draft_summary],
            candidate_filenames=scoped_sources,
            default_pdf_id=default_pdf_id,
            supported=None,
            debug_only=True,
        )
        if items:
            sections.append((section, items))

    if summary.unsupported_sentences:
        section = CitationSection(
            section_id="unsupported-summary",
            title="Unsupported Sentences",
            kind="unsupported_summary",
            debug_only=True,
        )
        items = _expand_text_items(
            section_id=section.section_id,
            values=summary.unsupported_sentences,
            candidate_filenames=scoped_sources,
            default_pdf_id=default_pdf_id,
            supported=False,
            debug_only=True,
        )
        if items:
            sections.append((section, items))

    return sections


def _truncate_snippet(text: str, *, limit: int = 320) -> str:
    clean = _normalize_space(text)
    if len(clean) <= limit:
        return clean
    return clean[: max(0, limit - 1)].rstrip() + "…"


def _merge_bbox(paragraphs: list[dict[str, Any]]) -> dict[str, int]:
    if not paragraphs:
        return {}
    left = min(paragraph["bbox"]["left"] for paragraph in paragraphs)
    top = min(paragraph["bbox"]["top"] for paragraph in paragraphs)
    right = max(paragraph["bbox"]["right"] for paragraph in paragraphs)
    bottom = max(paragraph["bbox"]["bottom"] for paragraph in paragraphs)
    return {
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
        "width": max(0, right - left),
        "height": max(0, bottom - top),
    }


def _paragraphs_from_exact_refs(asset: _PageAsset, paragraph_refs: Any) -> list[dict[str, Any]]:
    if not isinstance(paragraph_refs, list):
        return []
    matches: list[dict[str, Any]] = []
    seen: set[int] = set()
    for ref in paragraph_refs:
        if not isinstance(ref, dict):
            continue
        paragraph = None
        page_paragraph_index = ref.get("page_paragraph_index")
        if isinstance(page_paragraph_index, int) and page_paragraph_index in asset.by_page_paragraph:
            paragraph = asset.by_page_paragraph[page_paragraph_index]
        else:
            block_index = ref.get("block_index")
            paragraph_index = ref.get("paragraph_index")
            if isinstance(block_index, int) and isinstance(paragraph_index, int):
                paragraph = asset.by_block_paragraph.get((block_index, paragraph_index))
        if paragraph is None:
            continue
        paragraph_key = int(paragraph.get("page_paragraph_index") or 0)
        if paragraph_key and paragraph_key in seen:
            continue
        if paragraph_key:
            seen.add(paragraph_key)
        matches.append(paragraph)
    return sorted(matches, key=lambda row: (row.get("page_paragraph_index") or 0, row.get("block_index") or 0))


def _paragraphs_from_ranges(asset: _PageAsset, metadata: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    page_start = metadata.get("page_paragraph_start")
    page_end = metadata.get("page_paragraph_end")
    if isinstance(page_start, int) and isinstance(page_end, int):
        matches = [
            paragraph
            for paragraph in asset.paragraphs
            if page_start <= int(paragraph.get("page_paragraph_index") or 0) <= page_end
        ]
        if matches:
            return matches, "page_paragraph_range"

    block_start = metadata.get("block_start")
    block_end = metadata.get("block_end")
    paragraph_start = metadata.get("paragraph_start")
    paragraph_end = metadata.get("paragraph_end")
    if isinstance(block_start, int) and isinstance(block_end, int):
        matches = []
        for paragraph in asset.paragraphs:
            block_index = int(paragraph.get("block_index") or 0)
            paragraph_index = int(paragraph.get("paragraph_index") or 0)
            if not (block_start <= block_index <= block_end):
                continue
            if isinstance(paragraph_start, int) and isinstance(paragraph_end, int):
                if not (paragraph_start <= paragraph_index <= paragraph_end):
                    continue
            matches.append(paragraph)
        if matches:
            return matches, "block_paragraph_range"
    return [], "no_range_match"


def _paragraphs_from_fuzzy_match(asset: _PageAsset, sentence: str, chunk_text: str) -> list[dict[str, Any]]:
    scored: list[tuple[float, dict[str, Any]]] = []
    for paragraph in asset.paragraphs:
        score = max(
            _score_text_similarity(sentence, paragraph.get("text") or ""),
            _score_text_similarity(chunk_text, paragraph.get("text") or ""),
        )
        if score <= 0:
            continue
        scored.append((score, paragraph))
    if not scored:
        return []
    scored.sort(key=lambda row: row[0], reverse=True)
    top_score = scored[0][0]
    threshold = max(0.42, top_score * 0.78)
    matches = [paragraph for score, paragraph in scored[:5] if score >= threshold]
    matches.sort(key=lambda row: int(row.get("page_paragraph_index") or 0))
    return matches


def _resolve_paragraphs(
    *,
    chunk: _ChunkDocument,
    asset: _PageAsset | None,
    sentence: str,
) -> tuple[list[dict[str, Any]], str]:
    if asset is None or not asset.paragraphs:
        return [], "page_only"
    exact = _paragraphs_from_exact_refs(asset, chunk.metadata.get("paragraph_refs"))
    if exact:
        return exact, "exact_refs"
    ranged, range_strategy = _paragraphs_from_ranges(asset, chunk.metadata)
    if ranged:
        return ranged, range_strategy
    fuzzy = _paragraphs_from_fuzzy_match(asset, sentence, chunk.text)
    if fuzzy:
        return fuzzy, "fuzzy_paragraph_match"
    return [], "page_only"


def _citation_key(
    *,
    chunk_id: str | None,
    pdf_id: str,
    page: int,
    anchor: str | None,
    snippet: str,
) -> str:
    raw = "|".join([
        chunk_id or "",
        pdf_id,
        str(page),
        anchor or "",
        _truncate_snippet(snippet, limit=180),
    ])
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"cite-{digest}"


def _ground_item(
    item: _GroundingItem,
    *,
    chunk_lookup: dict[str, _ChunkDocument],
    page_assets_by_key: dict[tuple[str, int], _PageAsset],
    page_assets_by_source: dict[str, _PageAsset],
    page_sources: dict[tuple[str, int], list[str]],
) -> tuple[CitationIndexEntry | None, dict[str, Any]]:
    candidate_filenames = list(item.candidate_filenames)
    if item.expected_page is not None and item.expected_pdf_id is not None:
        candidate_filenames = page_sources.get((item.expected_pdf_id, item.expected_page), candidate_filenames)
    ordered_candidates = _dedupe_preserve_order([*item.preferred_filenames, *candidate_filenames])
    sentence_text = _clean_grounding_text(item.text)
    candidates: list[tuple[float, _ChunkDocument]] = []
    preferred_rank = {filename: index for index, filename in enumerate(item.preferred_filenames)}
    for filename in ordered_candidates:
        chunk = chunk_lookup.get(filename)
        if chunk is None:
            continue
        score = _score_text_similarity(sentence_text, chunk.text)
        if filename in preferred_rank:
            score += max(0.08, 0.18 - (preferred_rank[filename] * 0.03))
        if item.expected_pdf_id:
            score += 0.08 if chunk.pdf_id == item.expected_pdf_id else -0.15
        if item.expected_page is not None:
            score += 0.2 if chunk.page == item.expected_page else -0.24
        candidates.append((score, chunk))
    candidates.sort(key=lambda row: row[0], reverse=True)

    debug: dict[str, Any] = {
        "candidate_count": len(candidates),
        "candidates": [
            {
                "filename": chunk.source_filename,
                "page": chunk.page,
                "score": round(score, 4),
            }
            for score, chunk in candidates[:5]
        ],
        "expected_pdf_id": item.expected_pdf_id,
        "expected_page": item.expected_page,
        "preferred_filenames": list(item.preferred_filenames),
    }

    best_score = candidates[0][0] if candidates else 0.0
    best_chunk = candidates[0][1] if candidates else None
    if best_chunk is None:
        asset = None
        if item.expected_pdf_id and item.expected_page is not None:
            asset = page_assets_by_key.get((item.expected_pdf_id, item.expected_page))
        if asset is None and len(page_assets_by_key) == 1:
            asset = next(iter(page_assets_by_key.values()))
        if asset is None:
            return None, {**debug, "match_strategy": "no_candidates"}
        snippet = _truncate_snippet(sentence_text)
        anchor = f"{asset.pdf_id}:{asset.page}:page"
        citation = CitationIndexEntry(
            id=_citation_key(chunk_id=None, pdf_id=asset.pdf_id, page=asset.page, anchor=anchor, snippet=snippet),
            number=0,
            chunk_id=None,
            label=f"{asset.pdf_id} p.{asset.page}",
            pdf_id=asset.pdf_id,
            page=asset.page,
            snippet=snippet,
            anchor=anchor,
            page_key=asset.page_key,
            source_filename=None,
            page_source_filename=asset.source_filename,
            page_image_path=asset.image_path,
            source_pdf_path=asset.source_pdf_path,
            page_width=asset.width,
            page_height=asset.height,
            degraded=True,
            degraded_reason="snippet_card_page_level_fallback",
        )
        return citation, {**debug, "match_strategy": "page_level_fallback"}

    asset = None
    if best_chunk.page_source_filename:
        asset = page_assets_by_source.get(best_chunk.page_source_filename)
    if asset is None:
        asset = page_assets_by_key.get((best_chunk.pdf_id, best_chunk.page))

    paragraphs, match_strategy = _resolve_paragraphs(chunk=best_chunk, asset=asset, sentence=sentence_text)
    boxes = [
        CitationBox(
            text=str(paragraph.get("text") or ""),
            block_index=paragraph.get("block_index"),
            paragraph_index=paragraph.get("paragraph_index"),
            page_paragraph_index=paragraph.get("page_paragraph_index"),
            bbox=dict(paragraph.get("bbox") or {}),
        )
        for paragraph in paragraphs
    ]
    bbox = _merge_bbox(paragraphs)
    snippet = _truncate_snippet(" ".join(box.text for box in boxes) if boxes else best_chunk.text)

    degraded = False
    degraded_reason = None
    if match_strategy == "page_only":
        degraded = True
        degraded_reason = "snippet_card_page_level_fallback"
    elif best_score < 0.36 and not item.preferred_filenames:
        degraded = True
        degraded_reason = "low_similarity_fallback"

    page_paragraph_indices = [box.page_paragraph_index for box in boxes if box.page_paragraph_index is not None]
    anchor_suffix = (
        f"p{min(page_paragraph_indices)}-{max(page_paragraph_indices)}"
        if page_paragraph_indices
        else "page"
    )
    anchor = f"{best_chunk.pdf_id}:{best_chunk.page}:{anchor_suffix}"
    citation = CitationIndexEntry(
        id=_citation_key(
            chunk_id=best_chunk.source_filename,
            pdf_id=best_chunk.pdf_id,
            page=best_chunk.page,
            anchor=anchor,
            snippet=snippet,
        ),
        number=0,
        chunk_id=best_chunk.source_filename,
        label=f"{best_chunk.pdf_id} p.{best_chunk.page}",
        pdf_id=best_chunk.pdf_id,
        page=best_chunk.page,
        snippet=snippet,
        anchor=anchor,
        page_key=_page_key(best_chunk.pdf_id, best_chunk.page),
        source_filename=best_chunk.source_filename,
        page_source_filename=asset.source_filename if asset else best_chunk.page_source_filename,
        page_image_path=asset.image_path if asset else None,
        source_pdf_path=asset.source_pdf_path if asset else None,
        page_width=asset.width if asset else None,
        page_height=asset.height if asset else None,
        block_start=best_chunk.metadata.get("block_start"),
        block_end=best_chunk.metadata.get("block_end"),
        paragraph_start=best_chunk.metadata.get("paragraph_start"),
        paragraph_end=best_chunk.metadata.get("paragraph_end"),
        page_paragraph_start=min(page_paragraph_indices) if page_paragraph_indices else best_chunk.metadata.get("page_paragraph_start"),
        page_paragraph_end=max(page_paragraph_indices) if page_paragraph_indices else best_chunk.metadata.get("page_paragraph_end"),
        bbox=bbox,
        boxes=boxes,
        degraded=degraded,
        degraded_reason=degraded_reason,
    )
    debug.update(
        {
            "chosen_filename": best_chunk.source_filename,
            "chosen_score": round(best_score, 4),
            "match_strategy": match_strategy,
            "matched_block_indices": [box.block_index for box in boxes if box.block_index is not None],
            "matched_page_paragraph_indices": [
                box.page_paragraph_index for box in boxes if box.page_paragraph_index is not None
            ],
            "fallback_reason": degraded_reason,
        }
    )
    return citation, debug


def _register_citation(
    citation: CitationIndexEntry,
    *,
    citation_lookup: dict[str, CitationIndexEntry],
    citation_order: list[CitationIndexEntry],
) -> CitationIndexEntry:
    existing = citation_lookup.get(citation.id)
    if existing is not None:
        return existing
    numbered = citation.model_copy(update={"number": len(citation_order) + 1})
    citation_lookup[numbered.id] = numbered
    citation_order.append(numbered)
    return numbered


def build_resolved_citations(
    *,
    manifest: MaterializationManifest,
    summary: ScopedSummaryResult,
    extraction_dir: Path,
    source_pdf: Path | None = None,
) -> tuple[ScopedSummaryResult, ResolvedCitations, Path | None]:
    source_pdf_copy_path = _copy_source_pdf(source_pdf, extraction_dir)
    page_assets_by_key, page_assets_by_source, source_pages = _build_page_assets(
        manifest,
        extraction_dir=extraction_dir,
        source_pdf_copy_path=source_pdf_copy_path,
    )
    page_sources = _page_sources_by_key(manifest)
    chunk_lookup = _build_chunk_lookup(manifest, extraction_dir)
    sections_with_items = _build_sections(manifest=manifest, summary=summary)

    citation_lookup: dict[str, CitationIndexEntry] = {}
    citation_order: list[CitationIndexEntry] = []
    resolved_sections: list[CitationSection] = []
    strategy_counts: dict[str, int] = {}

    for section, items in sections_with_items:
        resolved_items: list[CitationSentenceItem] = []
        for item in items:
            citation, debug = _ground_item(
                item,
                chunk_lookup=chunk_lookup,
                page_assets_by_key=page_assets_by_key,
                page_assets_by_source=page_assets_by_source,
                page_sources=page_sources,
            )
            if citation is not None:
                registered = _register_citation(citation, citation_lookup=citation_lookup, citation_order=citation_order)
                citation_ids = [registered.id]
                strategy = debug.get("match_strategy") or "unknown"
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
                degraded = registered.degraded
                degraded_reason = registered.degraded_reason
                pdf_id = registered.pdf_id
                page = registered.page
            else:
                citation_ids = []
                degraded = True
                degraded_reason = debug.get("match_strategy") or "no_citation"
                pdf_id = item.expected_pdf_id
                page = item.expected_page
            resolved_items.append(
                CitationSentenceItem(
                    item_id=item.item_id,
                    text=item.text,
                    citation_ids=citation_ids,
                    supported=item.supported,
                    degraded=degraded,
                    degraded_reason=degraded_reason,
                    pdf_id=pdf_id,
                    page=page,
                    debug=debug,
                )
            )
        if resolved_items:
            resolved_sections.append(section.model_copy(update={"items": resolved_items}))

    resolved = ResolvedCitations(
        citation_index=list(citation_order),
        sections=resolved_sections,
        source_pages=source_pages,
        debug={
            "grounding_backend": "local_manifest_chunk_matcher",
            "section_count": len(resolved_sections),
            "citation_count": len(citation_order),
            "strategy_counts": strategy_counts,
        },
    )
    summary_with_citations = summary.model_copy(
        update={
            "citation_index": list(citation_order),
            "resolved_citations": resolved,
        }
    )
    return summary_with_citations, resolved, source_pdf_copy_path


def ensure_summary_citations(
    *,
    summary_path: Path,
    manifest_path: Path,
    source_pdf: Path | None = None,
    summary: ScopedSummaryResult | None = None,
    manifest: MaterializationManifest | None = None,
) -> tuple[ScopedSummaryResult, Path, Path, Path | None]:
    resolved_summary = summary or ScopedSummaryResult.model_validate_json(summary_path.read_text(encoding="utf-8"))
    resolved_manifest = manifest or MaterializationManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    extraction_dir = manifest_path.parent.resolve()
    output_dir = summary_path.parent.resolve()
    citation_index_path = output_dir / "citation_index.json"
    resolved_citations_path = output_dir / "resolved_citations.json"

    summary_with_citations, resolved_citations, source_pdf_copy_path = build_resolved_citations(
        manifest=resolved_manifest,
        summary=resolved_summary,
        extraction_dir=extraction_dir,
        source_pdf=source_pdf,
    )
    summary_path.write_text(summary_with_citations.model_dump_json(indent=2), encoding="utf-8")
    citation_index_path.write_text(
        "\n".join(
            [
                "[",
                *[
                    entry.model_dump_json(indent=2) + ("," if index < len(summary_with_citations.citation_index) - 1 else "")
                    for index, entry in enumerate(summary_with_citations.citation_index)
                ],
                "]",
            ]
        ),
        encoding="utf-8",
    )
    resolved_citations_path.write_text(resolved_citations.model_dump_json(indent=2), encoding="utf-8")
    return summary_with_citations, citation_index_path, resolved_citations_path, source_pdf_copy_path
