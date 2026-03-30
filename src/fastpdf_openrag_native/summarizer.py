from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .models import (
    TRUTH_FIELD_LABELS,
    TRUTH_FIELD_NAMES,
    MaterializationManifest,
    NoteValidationLayer,
    PageMapSummary,
    PresentationItem,
    PresentationLayer,
    PresentationSection,
    ScopedSummaryResult,
    SummaryScope,
    SupportedFact,
    TruthLayerNote,
    ValidationCheck,
    ValidationLayer,
    VerifiedSentence,
)
from .prompts import build_page_map_prompt, build_reduce_prompt, build_truth_layer_prompt
from .reranking import RERANKER_TYPE, attach_rank_metadata, rerank_hits, select_top_source_filenames
from .settings import AppSettings, get_settings


UNIT_TOKEN_RE = re.compile(r"[a-z0-9]+")
SOURCE_PAREN_RE = re.compile(r"\s*\((?:Source|Sources)\s*:[^)]+\)", flags=re.IGNORECASE)
SOURCE_BRACKET_RE = re.compile(r"\s*\[(?:Source|Sources)\s*:[^\]]+\]", flags=re.IGNORECASE)
SOURCE_CLAUSE_RE = re.compile(
    r"(?:^|[;,\s])(?:Source|Sources)\s*:\s*[A-Za-z0-9_.:/-]+(?:\s*(?:;|,)\s*[A-Za-z0-9_.:/-]+)*",
    flags=re.IGNORECASE,
)
SOURCE_FILE_RE = re.compile(r"\b[a-z0-9_.-]+__(?:c\d{4}\.md|p\d{4}\.html)\b", flags=re.IGNORECASE)
PROTECTED_ABBREVIATION_PATTERNS = tuple(
    re.compile(pattern, flags=re.IGNORECASE)
    for pattern in (
        r"\bM\.D\.",
        r"\bD\.O\.",
        r"\bPh\.D\.",
        r"\bDr\.",
        r"\bMr\.",
        r"\bMrs\.",
        r"\bMs\.",
        r"\bProf\.",
        r"\bNo\.(?=\s*\d)",
        r"\bSt\.",
        r"\bvs\.",
        r"\betc\.",
    )
)
INITIAL_PATTERN = re.compile(r"\b[A-Z]\.(?=\s+[A-Z][a-z])")
INITIAL_SERIES_PATTERN = re.compile(r"\b(?:[A-Z]\.){2,}")
HIGH_VALUE_TERMS = (
    "procedure",
    "indication",
    "operative",
    "operation",
    "selective nerve block",
    "nerve block",
    "stellate ganglion",
    "anesthetic",
    "anesthesia",
    "injection",
    "xylocaine",
    "aspiration",
    "antibiotic",
    "diagnosis",
    "assessment",
    "plan",
    "treatment",
    "pain",
    "surgery",
    "surgical",
    "clearance",
    "clearance appointment",
    "callback",
    "call back",
    "diabetes",
    "medication",
    "clinical concern",
    "follow-up",
    "follow up",
    "provider action",
    "requested",
    "request",
)
HEADER_BIAS_TERMS = (
    "header",
    "administrative",
    "demographic",
    "final report",
    "page ",
    "fax",
    "address",
    "dob",
    "mrn",
    "confidential",
    "confidentiality",
    "cover sheet",
    "routing",
    "print metadata",
    "printed by",
    "printed on",
    "document status",
    "phone msg",
    "forwarded by",
)
HIGH_VALUE_SECTION_TERMS = (
    "subjective",
    "chief complaint",
    "history",
    "history of present illness",
    "hpi",
    "assessment",
    "plan",
    "care plan",
    "impression",
    "procedure",
    "indication for procedure",
    "operation performed",
    "operative note",
    "operative diagnosis",
    "findings",
    "recommendation",
)
LOW_VALUE_SECTION_TERMS = (
    "fax",
    "cover sheet",
    "header",
    "footer",
    "routing",
    "contact",
    "contact information",
    "insurance",
    "demographics",
    "patient information",
    "billing",
    "printed by",
    "printed on",
    "page",
    "confidentiality",
    "address",
    "phone",
    "dob",
    "mrn",
)
HARD_DROP_PATTERNS = tuple(
    re.compile(pattern, flags=re.IGNORECASE)
    for pattern in (
        r"\bfrom\s+siliconmesa\b",
        r"\bfax\s+(?:server|summary|from|to|number)\b",
        r"\bpage\s+\d+\s+of\s+\d+\b",
        r"\bconfidentiality\s+notice\b",
        r"\bprinted\s+(?:by|on)\b",
        r"\bwww\.",
        r"https?://",
    )
)
LOW_VALUE_PREVIEW_PATTERNS = tuple(
    re.compile(pattern, flags=re.IGNORECASE)
    for pattern in (
        r"\b(?:dob|dos|mrn|ssn)\b",
        r"\baddress\b",
        r"\bphone\b",
        r"\bmember\s+id\b",
        r"\binsurance\b",
        r"\bguarantor\b",
    )
)


@dataclass(slots=True)
class _SupportedCandidate:
    text: str
    kind: str
    order: int
    evidence: list[Any]


class RetrievalGateway(Protocol):
    async def chat_on_sources(
        self,
        *,
        message: str,
        data_sources: list[str],
        limit: int = 6,
        score_threshold: float = 0,
    ) -> tuple[str, list[Any]]: ...

    async def search_on_sources(
        self,
        *,
        query: str,
        data_sources: list[str],
        limit: int | None = None,
        score_threshold: float | None = None,
    ) -> list[Any]: ...


def load_manifest(path: Path) -> MaterializationManifest:
    return MaterializationManifest.model_validate_json(path.read_text(encoding="utf-8"))


def load_scopes(path: Path) -> list[SummaryScope]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("scopes"), list):
        raw_scopes = payload["scopes"]
    elif isinstance(payload, list):
        raw_scopes = payload
    else:
        raw_scopes = [payload]

    scopes: list[SummaryScope] = []
    for index, item in enumerate(raw_scopes, start=1):
        if not isinstance(item, dict):
            continue
        page_refs = item.get("page_refs")
        if not isinstance(page_refs, list):
            pdf_id = str(item.get("pdf_id", "") or "").strip()
            pages = item.get("pages")
            if pdf_id and isinstance(pages, list):
                page_refs = [{"pdf_id": pdf_id, "page": page} for page in pages]
            else:
                continue

        scope_id = str(item.get("scope_id") or item.get("group_key") or f"scope-{index}").strip()
        title = str(item.get("title") or item.get("label") or item.get("name") or scope_id).strip()
        objective = str(
            item.get("objective")
            or item.get("question")
            or item.get("instructions")
            or (
                "Produce a grounded medical/legal summary of the selected pages. "
                "Do not invent facts and keep chronology intact."
            )
        ).strip()
        scopes.append(
            SummaryScope.model_validate(
                {
                    "scope_id": scope_id,
                    "title": title,
                    "objective": objective,
                    "page_refs": page_refs,
                }
            )
        )
    if not scopes:
        raise ValueError(f"no valid scopes were found in {path}")
    return scopes


def resolve_scope_pages(manifest: MaterializationManifest, scope: SummaryScope):
    lookup = manifest.page_lookup()
    pages = []
    for ref in scope.page_refs:
        page = lookup.get((ref.pdf_id, ref.page))
        if page:
            pages.append(page)
    pages.sort(key=lambda page: (page.order_index, page.pdf_id, page.page))
    if not pages:
        raise ValueError(f"scope {scope.scope_id!r} does not match any materialized pages")
    return pages


def _extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    candidates = [text]
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _sanitize_generated_text(text: str) -> str:
    clean = str(text or "")
    if not clean:
        return ""
    clean = clean.replace("```json", "").replace("```", "")
    clean = SOURCE_PAREN_RE.sub("", clean)
    clean = SOURCE_BRACKET_RE.sub("", clean)
    clean = SOURCE_CLAUSE_RE.sub(" ", clean)
    clean = SOURCE_FILE_RE.sub(" ", clean)
    clean = re.sub(r"\s+", " ", clean)
    clean = re.sub(r"\s+([,.;:!?])", r"\1", clean)
    clean = re.sub(r"([([\{])\s+", r"\1", clean)
    clean = re.sub(r"\s+([)\]\}])", r"\1", clean)
    clean = re.sub(r"\(\s*\)", "", clean)
    clean = clean.strip(" \t\r\n-–—;")
    return clean


def _protect_sentence_tokens(text: str) -> tuple[str, dict[str, str]]:
    protected = text
    replacements: dict[str, str] = {}
    counter = 0

    def reserve(raw: str) -> str:
        nonlocal counter
        token = f"__FASTPDF_SENT_{counter}__"
        counter += 1
        replacements[token] = raw
        return token

    for pattern in PROTECTED_ABBREVIATION_PATTERNS:
        protected = pattern.sub(lambda match: reserve(match.group(0)), protected)
    protected = INITIAL_SERIES_PATTERN.sub(lambda match: reserve(match.group(0)), protected)
    protected = INITIAL_PATTERN.sub(lambda match: reserve(match.group(0)), protected)
    return protected, replacements


def _restore_sentence_tokens(text: str, replacements: dict[str, str]) -> str:
    restored = text
    for token, raw in replacements.items():
        restored = restored.replace(token, raw)
    return restored


def _split_sentences(text: str) -> list[str]:
    clean = _sanitize_generated_text(text)
    if not clean:
        return []
    clean = re.sub(r"\s*[•;]\s*", ". ", clean)
    clean = re.sub(r"\s*\n+\s*", " ", clean)
    protected, replacements = _protect_sentence_tokens(clean)
    parts = re.split(r"(?<=[.!?])\s+", protected)
    sentences: list[str] = []
    for part in parts:
        restored = _restore_sentence_tokens(part, replacements)
        sentence = _sanitize_generated_text(restored)
        if sentence:
            sentences.append(sentence)
    return [part.strip() for part in sentences if part.strip()]


def _split_claim_units(text: str) -> list[str]:
    return _dedupe_preserve_order(
        [_ensure_sentence(sentence) for sentence in _split_sentences(text) if _ensure_sentence(sentence)]
    )


def _ensure_sentence(text: str) -> str:
    clean = re.sub(r"^\s*[-*•]+\s*", "", _sanitize_generated_text(text)).strip()
    clean = re.sub(r"\s+", " ", clean)
    if not clean:
        return ""
    if not re.search(r"[.!?]$", clean):
        clean = f"{clean}."
    return clean


def _split_fact_units(values: list[str]) -> list[str]:
    units: list[str] = []
    for value in values:
        clean = _sanitize_generated_text(value)
        if not clean:
            continue
        parts = [part.strip() for part in re.split(r"\s*[•]\s*", clean) if part.strip()] or [clean]
        for part in parts:
            units.extend(_split_claim_units(part))
    return _dedupe_preserve_order(units)


def _normalize_unit_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", _ensure_sentence(text).lower()).strip()


def _content_priority_score(
    text: str,
    *,
    evidence: list[Any] | None = None,
    kind: str = "summary",
) -> float:
    clean = re.sub(r"\s+", " ", str(text or "")).strip()
    if not clean:
        return 0.0
    lowered = clean.lower()
    tokens = UNIT_TOKEN_RE.findall(lowered)
    token_count = len(tokens)
    unique_tokens = len(set(tokens))

    score = min(token_count, 24) * 0.06
    score += min(len(clean), 240) / 240.0
    if unique_tokens >= 8:
        score += 0.2
    if any(char.isdigit() for char in clean):
        score += 0.18
    if kind == "key_fact":
        score += 0.22
    if evidence:
        score += min(0.36, len(evidence) * 0.08)

    if any(term in lowered for term in HIGH_VALUE_TERMS):
        score += 1.35
    if any(term in lowered for term in HEADER_BIAS_TERMS):
        score -= 0.45
    if "header/administrative material" in lowered or "mostly demographic/header material" in lowered:
        score -= 1.3
    if lowered.startswith("this page is primarily") or lowered.startswith("this page is mostly"):
        score -= 0.75
    if "provider" in lowered and not any(term in lowered for term in HIGH_VALUE_TERMS):
        score -= 0.1

    return score


def _select_informative_candidate_texts(
    candidates: list[_SupportedCandidate],
    *,
    max_items: int,
) -> list[str]:
    if not candidates or max_items <= 0:
        return []

    deduped: dict[str, tuple[float, int, str]] = {}
    for candidate in candidates:
        clean = _ensure_sentence(candidate.text)
        if not clean:
            continue
        key = _normalize_unit_key(clean)
        if not key:
            continue
        score = _content_priority_score(clean, evidence=candidate.evidence, kind=candidate.kind)
        existing = deduped.get(key)
        if existing is None or score > existing[0]:
            deduped[key] = (score, candidate.order, clean)

    if not deduped:
        return []

    ranked = sorted(deduped.values(), key=lambda item: (item[0], -item[1]), reverse=True)
    top_score = ranked[0][0]
    if top_score >= 1.5:
        ranked = [item for item in ranked if item[0] >= max(0.5, top_score - 1.0)]
    selected = ranked[:max_items]
    selected.sort(key=lambda item: item[1])
    return [item[2] for item in selected]


def _supported_candidates_from_verified(
    verified_sentences: list[VerifiedSentence],
    *,
    kind: str,
    order_offset: int = 0,
) -> list[_SupportedCandidate]:
    candidates: list[_SupportedCandidate] = []
    for index, row in enumerate(verified_sentences):
        if not row.supported:
            continue
        candidates.append(
            _SupportedCandidate(
                text=_ensure_sentence(row.sentence),
                kind=kind,
                order=order_offset + index,
                evidence=list(row.evidence),
            )
        )
    return candidates


def _compose_supported_page_content(
    *,
    verified_summary_sentences: list[VerifiedSentence],
    verified_key_facts: list[VerifiedSentence],
) -> tuple[str, list[str], list[str]]:
    supported_key_facts = _dedupe_preserve_order(
        [_ensure_sentence(row.sentence) for row in verified_key_facts if row.supported]
    )
    unsupported_key_facts = _dedupe_preserve_order(
        [_ensure_sentence(row.sentence) for row in verified_key_facts if not row.supported]
    )
    candidates = [
        *_supported_candidates_from_verified(verified_summary_sentences, kind="summary", order_offset=0),
        *_supported_candidates_from_verified(
            verified_key_facts,
            kind="key_fact",
            order_offset=len(verified_summary_sentences),
        ),
    ]
    supported_summary = " ".join(
        _select_informative_candidate_texts(candidates, max_items=4)
    ).strip()
    return supported_summary, supported_key_facts, unsupported_key_facts


def _compose_scope_supported_fallback(page_summaries: list[PageMapSummary]) -> str:
    candidates: list[_SupportedCandidate] = []
    order = 0
    for page_summary in page_summaries:
        for sentence in _split_sentences(page_summary.supported_summary):
            candidates.append(_SupportedCandidate(text=sentence, kind="summary", order=order, evidence=[]))
            order += 1
        for fact in page_summary.supported_key_facts:
            candidates.append(_SupportedCandidate(text=fact, kind="key_fact", order=order, evidence=[]))
            order += 1
    return " ".join(_select_informative_candidate_texts(candidates, max_items=6)).strip()


def _should_use_scope_supported_fallback(*, current: str, fallback: str) -> bool:
    current_clean = re.sub(r"\s+", " ", current or "").strip()
    fallback_clean = re.sub(r"\s+", " ", fallback or "").strip()
    if not fallback_clean:
        return False
    if not current_clean:
        return True
    current_score = _content_priority_score(current_clean)
    fallback_score = _content_priority_score(fallback_clean)
    current_sentences = len(_split_sentences(current_clean))
    fallback_sentences = len(_split_sentences(fallback_clean))
    return fallback_score >= (current_score + 0.85) or (
        fallback_score >= (current_score + 0.35) and fallback_sentences > current_sentences
    )


def _normalized_retrieval_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _contains_any_term(text: str, terms: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in terms)


def _looks_like_ocr_junk(text: str) -> bool:
    clean = _normalized_retrieval_text(text)
    if not clean:
        return True
    alpha_count = sum(char.isalpha() for char in clean)
    digit_count = sum(char.isdigit() for char in clean)
    token_count = len(UNIT_TOKEN_RE.findall(clean.lower()))
    if alpha_count == 0 and digit_count > 0:
        return True
    if alpha_count < 4 and digit_count >= 4 and len(clean) < 36:
        return True
    if token_count <= 2 and len(clean) <= 12:
        return True
    if re.fullmatch(r"[\W_]+", clean):
        return True
    return False


def _looks_like_header_or_footer_artifact(title: str, preview: str) -> bool:
    combined = " ".join(part for part in [title, preview] if part).strip()
    if not combined:
        return False
    if _contains_any_term(combined, HIGH_VALUE_TERMS + HIGH_VALUE_SECTION_TERMS):
        return False
    if any(pattern.search(combined) for pattern in HARD_DROP_PATTERNS):
        return True
    token_count = len(UNIT_TOKEN_RE.findall(combined.lower()))
    low_value_hits = sum(1 for pattern in LOW_VALUE_PREVIEW_PATTERNS if pattern.search(combined))
    if low_value_hits >= 2 and token_count <= 16:
        return True
    if _contains_any_term(title, LOW_VALUE_SECTION_TERMS) and token_count <= 18:
        return True
    return False


def _filter_page_retrieval_documents(page_retrieval_documents: list[Any]) -> tuple[list[Any], dict[str, Any]]:
    if not page_retrieval_documents:
        return [], {
            "filtered": False,
            "fallback_used": False,
            "kept_count": 0,
            "dropped_count": 0,
            "dropped": [],
        }

    kept: list[Any] = []
    dropped: list[dict[str, str]] = []
    for document in page_retrieval_documents:
        title = _normalized_retrieval_text(getattr(document, "section_title", ""))
        preview = _normalized_retrieval_text(getattr(document, "text_preview", ""))
        combined = " ".join(part for part in [title, preview] if part).strip()
        drop_reason: str | None = None
        if _looks_like_header_or_footer_artifact(title, preview):
            drop_reason = "header_footer_artifact"
        elif _looks_like_ocr_junk(combined) and not _contains_any_term(combined, HIGH_VALUE_TERMS + HIGH_VALUE_SECTION_TERMS):
            drop_reason = "ocr_junk"
        if drop_reason:
            dropped.append({"filename": getattr(document, "source_filename", ""), "reason": drop_reason})
            continue
        kept.append(document)

    if kept:
        return kept, {
            "filtered": bool(dropped),
            "fallback_used": False,
            "kept_count": len(kept),
            "dropped_count": len(dropped),
            "dropped": dropped,
        }

    return list(page_retrieval_documents), {
        "filtered": False,
        "fallback_used": True,
        "kept_count": len(page_retrieval_documents),
        "dropped_count": len(dropped),
        "dropped": dropped,
    }


def _document_priority_score(document: Any) -> float:
    title = _normalized_retrieval_text(getattr(document, "section_title", ""))
    preview = _normalized_retrieval_text(getattr(document, "text_preview", ""))
    title_lower = title.lower()
    preview_lower = preview.lower()
    combined = " ".join(part for part in [title, preview] if part)
    score = _content_priority_score(combined, kind="retrieval_hint")
    if _contains_any_term(title_lower, HIGH_VALUE_SECTION_TERMS):
        score += 1.45
    if _contains_any_term(preview_lower, HIGH_VALUE_SECTION_TERMS):
        score += 0.45
    if _contains_any_term(title_lower, LOW_VALUE_SECTION_TERMS):
        score -= 0.95
    if any(pattern.search(preview) for pattern in LOW_VALUE_PREVIEW_PATTERNS):
        score -= 0.25
    if _looks_like_header_or_footer_artifact(title, preview):
        score -= 2.2
    if _looks_like_ocr_junk(preview) and not _contains_any_term(preview_lower, HIGH_VALUE_TERMS + HIGH_VALUE_SECTION_TERMS):
        score -= 1.25
    return score


def _build_page_retrieval_hints(page_retrieval_documents: list[Any], *, max_items: int = 6) -> list[str]:
    if not page_retrieval_documents:
        return []
    ranked: list[tuple[float, int, int, str]] = []
    rendered_by_filename: dict[str, str] = {}
    for index, document in enumerate(page_retrieval_documents):
        title = str(getattr(document, "section_title", "") or "").strip()
        preview = re.sub(r"\s+", " ", str(getattr(document, "text_preview", "") or "")).strip()
        preview = preview[:160].rstrip() + "…" if len(preview) > 160 else preview
        label = title or preview or document.source_filename
        rendered = f"- {label} [{document.source_filename}]"
        if preview and preview.lower() != label.lower():
            rendered = f"{rendered}: {preview}"
        rendered_by_filename[document.source_filename] = rendered
        score = _document_priority_score(document)
        ranked.append((score, index, getattr(document, "chunk_index", index), document.source_filename))
    ranked.sort(key=lambda item: (item[0], -item[1], -item[2]), reverse=True)
    selected_filenames = [filename for _, _, _, filename in ranked[:max_items]]
    return [rendered_by_filename[filename] for filename in selected_filenames]


def _expand_page_selected_sources(
    *,
    selected_sources: list[str],
    page_sources: list[str],
    page_retrieval_documents: list[Any],
) -> tuple[list[str], dict[str, Any]]:
    base_sources = _dedupe_preserve_order(selected_sources or page_sources)
    if not page_retrieval_documents:
        return base_sources, {
            "context_expanded": False,
            "same_section_sources": [],
            "informative_backfill_sources": [],
        }

    ordered_docs = sorted(
        page_retrieval_documents,
        key=lambda document: (getattr(document, "chunk_index", 0), getattr(document, "source_filename", "")),
    )
    index_by_filename = {document.source_filename: index for index, document in enumerate(ordered_docs)}
    doc_by_filename = {document.source_filename: document for document in ordered_docs}
    max_total = max(len(base_sources), min(len(page_sources), 6))
    expanded: list[str] = []
    same_section_sources: list[str] = []
    informative_backfill_sources: list[str] = []

    def add_source(filename: str, *, bucket: str | None = None) -> None:
        if filename not in page_sources or filename in expanded or len(expanded) >= max_total:
            return
        expanded.append(filename)
        if bucket == "same_section":
            same_section_sources.append(filename)
        elif bucket == "informative_backfill":
            informative_backfill_sources.append(filename)

    for filename in base_sources:
        add_source(filename)

    for filename in list(expanded):
        index = index_by_filename.get(filename)
        if index is None:
            continue
        document = ordered_docs[index]
        section_title = re.sub(r"\s+", " ", str(getattr(document, "section_title", "") or "")).strip().lower()
        parent_source = getattr(document, "parent_source_filename", None)
        if not section_title:
            continue
        for step in (-1, 1):
            cursor = index + step
            while 0 <= cursor < len(ordered_docs) and len(expanded) < max_total:
                neighbor = ordered_docs[cursor]
                neighbor_section = re.sub(
                    r"\s+", " ", str(getattr(neighbor, "section_title", "") or "")
                ).strip().lower()
                if getattr(neighbor, "parent_source_filename", None) != parent_source or neighbor_section != section_title:
                    break
                add_source(neighbor.source_filename, bucket="same_section")
                cursor += step

    selected_scores = [_document_priority_score(doc_by_filename[filename]) for filename in expanded if filename in doc_by_filename]
    needs_backfill = (not selected_scores) or max(selected_scores) < 1.65 or all(score < 1.3 for score in selected_scores)
    if needs_backfill and len(expanded) < max_total:
        ranked_unselected = sorted(
            (
                (_document_priority_score(document), getattr(document, "chunk_index", 0), document.source_filename)
                for document in ordered_docs
                if document.source_filename not in expanded
            ),
            key=lambda item: (item[0], -item[1]),
            reverse=True,
        )
        for score, _, filename in ranked_unselected:
            if len(expanded) >= max_total:
                break
            if score < 1.25:
                continue
            add_source(filename, bucket="informative_backfill")

    if not expanded:
        expanded = base_sources
    return expanded, {
        "context_expanded": expanded != base_sources,
        "same_section_sources": same_section_sources,
        "informative_backfill_sources": informative_backfill_sources,
    }


def _looks_like_retrieval_failure(text: str) -> bool:
    lowered = (text or "").lower()
    return any(
        marker in lowered
        for marker in (
            "no relevant supporting sources were found",
            "please provide the url",
            "paste the page content",
            "file upload",
        )
    )


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _page_retrieval_sources(page) -> list[str]:
    return _dedupe_preserve_order(page.retrieval_sources())


def resolve_scope_retrieval_sources(
    manifest: MaterializationManifest,
    scope: SummaryScope,
) -> list[str]:
    pages = resolve_scope_pages(manifest, scope)
    return _dedupe_preserve_order(
        [source for page in pages for source in _page_retrieval_sources(page)]
    )


def _candidate_limit(requested_limit: int, settings: AppSettings) -> int:
    if not settings.retrieval_rerank_enabled:
        return requested_limit
    return max(requested_limit, settings.retrieval_rerank_candidate_limit)


def _chat_limit(requested_limit: int, settings: AppSettings) -> int:
    if settings.backend_rerank_enabled:
        return max(requested_limit, settings.backend_rerank_candidate_limit)
    return requested_limit


def _verification_candidate_limit(settings: AppSettings) -> int:
    if settings.retrieval_rerank_enabled:
        return max(settings.verification_limit, settings.retrieval_rerank_candidate_limit)
    if settings.backend_search_rerank_enabled:
        return settings.verification_limit
    if settings.backend_rerank_enabled:
        return max(settings.verification_limit, settings.backend_rerank_candidate_limit)
    return settings.verification_limit


def _selected_source_limit(
    *,
    requested_limit: int,
    settings: AppSettings,
    source_groups: list[list[str]] | None = None,
) -> int:
    if requested_limit <= 0:
        return 0
    if not settings.retrieval_rerank_enabled:
        return requested_limit
    group_count = len([group for group in source_groups or [] if group])
    return max(1, max(group_count, min(requested_limit, settings.retrieval_rerank_top_k)))


def _preflight_skip_reason(
    *,
    data_source_count: int,
    settings: AppSettings,
) -> str | None:
    if data_source_count <= 1:
        return "single_source"
    return None


def _rank_generation_evidence(
    *,
    query: str,
    hits: list[Any],
    settings: AppSettings,
) -> list[Any]:
    if settings.retrieval_rerank_enabled or (settings.backend_rerank_enabled and not settings.backend_search_rerank_enabled):
        return rerank_hits(query, list(hits))
    return attach_rank_metadata(list(hits))


def _prepare_selected_sources(
    *,
    query: str,
    hits: list[Any],
    settings: AppSettings,
    requested_limit: int,
    source_groups: list[list[str]] | None = None,
) -> tuple[list[Any], list[str]]:
    reranked_hits = _rank_generation_evidence(query=query, hits=list(hits), settings=settings)
    top_k = _selected_source_limit(
        requested_limit=requested_limit,
        settings=settings,
        source_groups=source_groups,
    )
    selected_sources = select_top_source_filenames(
        reranked_hits,
        top_k=top_k,
        source_groups=source_groups,
    )
    if not selected_sources and reranked_hits:
        selected_sources = _dedupe_preserve_order([hit.filename for hit in reranked_hits[:requested_limit]])
    return reranked_hits, selected_sources


def _seed_verification_evidence(
    *,
    query: str,
    candidate_hits: list[Any] | None,
    settings: AppSettings,
) -> list[Any]:
    if not candidate_hits:
        return []
    normalized_query = re.sub(r"\s+", " ", query.lower()).strip()
    query_tokens = {token for token in UNIT_TOKEN_RE.findall(normalized_query) if len(token) > 1}
    ranked_hits = rerank_hits(query, list(candidate_hits))
    seeded_hits: list[Any] = []
    minimum_score = max(settings.verification_score_threshold, 0.2)
    required_overlap = 1 if len(query_tokens) <= 2 else 2
    for hit in ranked_hits:
        normalized_text = re.sub(r"\s+", " ", str(hit.text or "").lower()).strip()
        text_tokens = {token for token in UNIT_TOKEN_RE.findall(normalized_text) if len(token) > 1}
        overlap_count = len(query_tokens.intersection(text_tokens))
        has_overlap = overlap_count >= required_overlap
        has_phrase_match = bool(normalized_query and normalized_query in normalized_text)
        if not has_overlap and not has_phrase_match:
            continue
        hit_score = float(hit.rerank_score if hit.rerank_score is not None else hit.score or 0.0)
        if hit_score < minimum_score:
            continue
        seeded_hits.append(hit)
        if len(seeded_hits) >= _verification_candidate_limit(settings):
            break
    return seeded_hits


def _reranking_enabled(settings: AppSettings) -> bool:
    return settings.backend_rerank_enabled or settings.retrieval_rerank_enabled


def _reranker_location(settings: AppSettings) -> str | None:
    if settings.backend_rerank_enabled:
        return "langflow_agent_tool"
    if settings.retrieval_rerank_enabled:
        return "application"
    return None


def _reranker_type(settings: AppSettings) -> str | None:
    if settings.backend_rerank_enabled:
        return settings.backend_rerank_provider
    if settings.retrieval_rerank_enabled:
        return RERANKER_TYPE
    return None


def _verification_reranker_location(settings: AppSettings) -> str | None:
    if settings.retrieval_rerank_enabled:
        return "application"
    if settings.backend_search_rerank_enabled:
        return "openrag_backend_search_api"
    if settings.backend_rerank_enabled:
        return "application_verification_fallback"
    return None


def _verification_reranker_type(settings: AppSettings) -> str | None:
    if settings.retrieval_rerank_enabled or settings.backend_search_rerank_enabled or settings.backend_rerank_enabled:
        return RERANKER_TYPE
    return None


async def _summarize_page(
    gateway: RetrievalGateway,
    scope: SummaryScope,
    page,
    settings: AppSettings,
    page_retrieval_documents: list[Any] | None = None,
) -> tuple[PageMapSummary, dict[str, Any]]:
    local_page_documents = list(page_retrieval_documents or [])
    filtered_page_documents, retrieval_filter_debug = _filter_page_retrieval_documents(local_page_documents)
    all_page_sources = _page_retrieval_sources(page)
    filtered_page_sources = _dedupe_preserve_order(
        [document.source_filename for document in filtered_page_documents if document.source_filename in all_page_sources]
    )
    page_sources = filtered_page_sources or all_page_sources
    active_page_documents = [
        document for document in filtered_page_documents if document.source_filename in page_sources
    ] or local_page_documents
    retrieval_hints = _build_page_retrieval_hints(active_page_documents)
    prompt = build_page_map_prompt(
        scope,
        pdf_id=page.pdf_id,
        page=page.page,
        source_filename=page.source_filename,
        retrieval_source_count=len(page_sources),
        retrieval_source_hints=retrieval_hints,
    )
    preflight_query = f"page summary {scope.objective}"
    preflight_skip_reason = _preflight_skip_reason(
        data_source_count=len(page_sources),
        settings=settings,
    )
    preflight_search_used = False
    retry_preflight_search_used = False
    preflight_sources: list[Any] = []
    reranked_preflight_sources: list[Any] = []
    selected_page_sources = list(page_sources)
    if preflight_skip_reason is None:
        preflight_sources = await gateway.search_on_sources(
            query=preflight_query,
            data_sources=page_sources,
            limit=_candidate_limit(8, settings),
            score_threshold=0,
        )
        preflight_search_used = True
        reranked_preflight_sources, selected_page_sources = _prepare_selected_sources(
            query=preflight_query,
            hits=preflight_sources,
            settings=settings,
            requested_limit=8,
            source_groups=[page_sources],
        )
    chat_data_sources, source_expansion_debug = _expand_page_selected_sources(
        selected_sources=selected_page_sources,
        page_sources=page_sources,
        page_retrieval_documents=active_page_documents,
    )
    response_text, sources = await gateway.chat_on_sources(
        message=prompt,
        data_sources=chat_data_sources,
        limit=max(_chat_limit(8, settings), len(chat_data_sources)),
        score_threshold=0,
    )
    retry_used = False
    if not sources or _looks_like_retrieval_failure(response_text):
        if not reranked_preflight_sources:
            preflight_sources = await gateway.search_on_sources(
                query=preflight_query,
                data_sources=page_sources,
                limit=_candidate_limit(8, settings),
                score_threshold=0,
            )
            reranked_preflight_sources, selected_page_sources = _prepare_selected_sources(
                query=preflight_query,
                hits=preflight_sources,
                settings=settings,
                requested_limit=8,
                source_groups=[page_sources],
            )
            retry_preflight_search_used = True
        retry_prompt = "\n\n".join(
            [
                prompt,
                (
                    f"OpenRAG search preflight found {len(reranked_preflight_sources)} candidate indexed hit(s) "
                    f"across {len(chat_data_sources)} retrieval chunk file(s) for "
                    f"{page.source_filename}. Use the OpenSearch Retrieval Tool and answer from those indexed "
                    "results only."
                ),
            ]
        )
        response_text, sources = await gateway.chat_on_sources(
            message=retry_prompt,
            data_sources=chat_data_sources,
            limit=max(_chat_limit(8, settings), len(chat_data_sources)),
            score_threshold=0,
        )
        retry_used = True
    parsed = _extract_json_object(response_text) or {}
    summary = _sanitize_generated_text(str(parsed.get("summary") or response_text).strip())
    key_facts = parsed.get("key_facts")
    if not isinstance(key_facts, list):
        key_facts = []
    clean_key_facts = _split_fact_units(
        [_sanitize_generated_text(str(item).strip()) for item in key_facts if str(item).strip()]
    )

    generation_evidence_pool = (
        list(reranked_preflight_sources)
        if reranked_preflight_sources
        else _rank_generation_evidence(query=preflight_query, hits=list(sources), settings=settings)
    )
    retrieved_source_strategy = "preflight_pool" if reranked_preflight_sources else "chat_sources"

    page_summary = PageMapSummary(
        pdf_id=page.pdf_id,
        page=page.page,
        source_filename=page.source_filename,
        summary=summary,
        key_facts=clean_key_facts,
        raw_response=response_text,
        retrieved_sources=list(generation_evidence_pool[: max(1, len(chat_data_sources))]),
    )
    return page_summary, {
        "prompt": prompt,
        "retrieval_hints": retrieval_hints,
        "preflight_query": preflight_query,
        "preflight_sources_before_rerank": [source.model_dump() for source in preflight_sources],
        "preflight_sources": [source.model_dump() for source in reranked_preflight_sources],
        "preflight_sources_after_rerank": [source.model_dump() for source in reranked_preflight_sources],
        "retrieved_source_strategy": retrieved_source_strategy,
        "generation_evidence_pool": [source.model_dump() for source in generation_evidence_pool],
        "all_data_sources": all_page_sources,
        "data_sources": page_sources,
        "retrieval_filter_debug": retrieval_filter_debug,
        "selected_source_filenames": selected_page_sources or page_sources,
        "expanded_selected_source_filenames": chat_data_sources,
        "source_expansion_debug": source_expansion_debug,
        "clean_summary": summary,
        "clean_key_facts": clean_key_facts,
        "limit": 8,
        "score_threshold": 0,
        "preflight_search_used": preflight_search_used,
        "preflight_skip_reason": preflight_skip_reason,
        "retry_preflight_search_used": retry_preflight_search_used,
        "retrieval_retry_used": retry_used,
        "reranking_enabled": _reranking_enabled(settings),
        "reranker_location": _reranker_location(settings),
        "reranker_type": _reranker_type(settings),
        "response": response_text,
        "sources": [source.model_dump() for source in sources],
    }


async def _verify_sentence(
    gateway: RetrievalGateway,
    *,
    sentence: str,
    data_sources: list[str],
    settings: AppSettings,
    candidate_hits: list[Any] | None = None,
) -> VerifiedSentence:
    clean_sentence = _ensure_sentence(sentence)
    seeded_evidence = _seed_verification_evidence(
        query=clean_sentence,
        candidate_hits=candidate_hits,
        settings=settings,
    )
    if seeded_evidence:
        return VerifiedSentence(
            sentence=clean_sentence,
            supported=True,
            evidence=list(seeded_evidence[: settings.verification_limit]),
        )

    evidence = await gateway.search_on_sources(
        query=clean_sentence,
        data_sources=data_sources,
        limit=_verification_candidate_limit(settings),
        score_threshold=settings.verification_score_threshold,
    )
    reranked_evidence = _rank_generation_evidence(
        query=clean_sentence,
        hits=list(evidence),
        settings=settings,
    )
    return VerifiedSentence(
        sentence=clean_sentence,
        supported=bool(reranked_evidence),
        evidence=list(reranked_evidence[: settings.verification_limit]),
    )


async def _verify_units(
    gateway: RetrievalGateway,
    *,
    units: list[str],
    data_sources: list[str],
    settings: AppSettings,
    verification_semaphore: asyncio.Semaphore | None = None,
    candidate_hits: list[Any] | None = None,
) -> tuple[list[str], list[VerifiedSentence]]:
    verification_queries = _dedupe_preserve_order(
        [claim for unit in units for claim in _split_claim_units(unit)]
    )
    if not verification_queries:
        return [], []

    semaphore = verification_semaphore or asyncio.Semaphore(max(1, settings.verification_concurrency))

    async def bounded_verify_sentence(sentence: str) -> VerifiedSentence:
        async with semaphore:
            return await _verify_sentence(
                gateway,
                sentence=sentence,
                data_sources=data_sources,
                settings=settings,
                candidate_hits=candidate_hits,
            )

    verified_sentences = await asyncio.gather(
        *(bounded_verify_sentence(sentence) for sentence in verification_queries)
    )
    return verification_queries, list(verified_sentences)


async def _verify_text(
    gateway: RetrievalGateway,
    *,
    text: str,
    data_sources: list[str],
    settings: AppSettings,
    verification_semaphore: asyncio.Semaphore | None = None,
    candidate_hits: list[Any] | None = None,
) -> tuple[list[str], list[VerifiedSentence], str, list[str]]:
    verification_queries, verified_sentences = await _verify_units(
        gateway,
        units=_split_claim_units(text),
        data_sources=data_sources,
        settings=settings,
        verification_semaphore=verification_semaphore,
        candidate_hits=candidate_hits,
    )
    if not verification_queries:
        return [], [], "", []
    supported_summary = " ".join(
        row.sentence for row in verified_sentences if row.supported
    ).strip()
    unsupported_sentences = [row.sentence for row in verified_sentences if not row.supported]
    return verification_queries, list(verified_sentences), supported_summary, unsupported_sentences


def _page_summary_passes_verification(page_summary: PageMapSummary) -> bool:
    if _looks_like_retrieval_failure(page_summary.summary):
        return False
    return bool(page_summary.supported_summary)


async def summarize_scope(
    gateway: RetrievalGateway,
    *,
    manifest: MaterializationManifest,
    scope: SummaryScope,
    settings: AppSettings | None = None,
) -> ScopedSummaryResult:
    effective_settings = settings or get_settings()
    pages = resolve_scope_pages(manifest, scope)
    retrieval_docs_by_page_source: dict[str, list[Any]] = {}
    for document in manifest.retrieval_documents:
        if document.parent_source_filename:
            retrieval_docs_by_page_source.setdefault(document.parent_source_filename, []).append(document)
    all_page_source_groups = [_page_retrieval_sources(page) for page in pages]
    all_page_sources = _dedupe_preserve_order(
        [source for group in all_page_source_groups for source in group]
    )
    semaphore = asyncio.Semaphore(max(1, effective_settings.page_summary_concurrency))

    async def bounded_page_summary(page):
        async with semaphore:
            return await _summarize_page(
                gateway,
                scope,
                page,
                effective_settings,
                page_retrieval_documents=retrieval_docs_by_page_source.get(page.source_filename, []),
            )

    page_results = await asyncio.gather(*(bounded_page_summary(page) for page in pages))
    raw_page_summaries = [item[0] for item in page_results]
    page_debug = [item[1] for item in page_results]
    page_verification_semaphore = asyncio.Semaphore(max(1, effective_settings.verification_concurrency))

    async def verify_page_summary(page, page_summary: PageMapSummary, request_debug: dict[str, Any]):
        candidate_hits = (
            page_summary.retrieved_sources
            if request_debug.get("retrieved_source_strategy") == "preflight_pool"
            else None
        )
        return await _verify_text(
            gateway,
            text=page_summary.summary,
            data_sources=_page_retrieval_sources(page),
            settings=effective_settings,
            verification_semaphore=page_verification_semaphore,
            candidate_hits=candidate_hits,
        )

    async def verify_page_key_facts(page, page_summary: PageMapSummary, request_debug: dict[str, Any]):
        candidate_hits = (
            page_summary.retrieved_sources
            if request_debug.get("retrieved_source_strategy") == "preflight_pool"
            else None
        )
        return await _verify_units(
            gateway,
            units=page_summary.key_facts,
            data_sources=_page_retrieval_sources(page),
            settings=effective_settings,
            verification_semaphore=page_verification_semaphore,
            candidate_hits=candidate_hits,
        )

    page_verifications = await asyncio.gather(
        *(
            verify_page_summary(page, page_summary, request_debug)
            for page, page_summary, request_debug in zip(pages, raw_page_summaries, page_debug, strict=False)
        )
    )
    page_key_fact_verifications = await asyncio.gather(
        *(
            verify_page_key_facts(page, page_summary, request_debug)
            for page, page_summary, request_debug in zip(pages, raw_page_summaries, page_debug, strict=False)
        )
    )

    page_summaries: list[PageMapSummary] = []
    verified_page_summaries: list[PageMapSummary] = []
    verified_page_source_groups: list[list[str]] = []
    page_verification_debug: list[dict[str, Any]] = []
    for page, page_summary, request_debug, page_verification, page_key_fact_verification in zip(
        pages,
        raw_page_summaries,
        page_debug,
        page_verifications,
        page_key_fact_verifications,
        strict=False,
    ):
        page_verification_queries, verified_page_sentences, summary_sentence_support, page_unsupported_sentences = (
            page_verification
        )
        page_key_fact_queries, verified_key_facts = page_key_fact_verification
        page_supported_summary, supported_key_facts, unsupported_key_facts = _compose_supported_page_content(
            verified_summary_sentences=verified_page_sentences,
            verified_key_facts=verified_key_facts,
        )
        combined_unsupported_units = _dedupe_preserve_order(
            [*page_unsupported_sentences, *unsupported_key_facts]
        )
        verified_page_summary = page_summary.model_copy(
            update={
                "verified_sentences": verified_page_sentences,
                "verified_key_facts": verified_key_facts,
                "supported_summary": page_supported_summary,
                "supported_key_facts": supported_key_facts,
                "unsupported_sentences": combined_unsupported_units,
                "unsupported_key_facts": unsupported_key_facts,
            }
        )
        verified_page_summary = verified_page_summary.model_copy(
            update={"passed_verification": _page_summary_passes_verification(verified_page_summary)}
        )
        request_debug.update(
            {
                "verification_queries": page_verification_queries,
                "verified_sentences": [row.model_dump() for row in verified_page_sentences],
                "key_fact_verification_queries": page_key_fact_queries,
                "verified_key_facts": [row.model_dump() for row in verified_key_facts],
                "summary_sentence_support": summary_sentence_support,
                "supported_summary": page_supported_summary,
                "supported_key_facts": supported_key_facts,
                "unsupported_sentences": combined_unsupported_units,
                "unsupported_key_facts": unsupported_key_facts,
                "passed_verification": verified_page_summary.passed_verification,
            }
        )
        if verified_page_summary.passed_verification:
            page_sources = _page_retrieval_sources(page)
            verified_page_summaries.append(verified_page_summary)
            verified_page_source_groups.append(page_sources)
        page_summaries.append(verified_page_summary)
        page_verification_debug.append(
            {
                "pdf_id": page.pdf_id,
                "page": page.page,
                "source_filename": page.source_filename,
                "verification_queries": page_verification_queries,
                "key_fact_verification_queries": page_key_fact_queries,
                "supported_summary": page_supported_summary,
                "supported_key_facts": supported_key_facts,
                "unsupported_sentences": combined_unsupported_units,
                "unsupported_key_facts": unsupported_key_facts,
                "passed_verification": verified_page_summary.passed_verification,
            }
        )

    reduce_prompt = build_reduce_prompt(scope, list(verified_page_summaries)) if verified_page_summaries else ""
    reduce_limit = min(50, max(10, len(verified_page_summaries))) if verified_page_summaries else 0
    reduce_preflight_query = f"overall chronology summary {scope.objective}"
    reduce_page_sources = _dedupe_preserve_order(
        [source for group in verified_page_source_groups for source in group]
    )
    reduce_preflight_skip_reason = _preflight_skip_reason(
        data_source_count=len(reduce_page_sources),
        settings=effective_settings,
    )
    reduce_preflight_search_used = False
    reduce_retry_preflight_search_used = False
    reduce_preflight_sources: list[Any] = []
    reranked_reduce_sources: list[Any] = []
    selected_reduce_sources = list(reduce_page_sources)
    reduce_response = ""
    reduce_sources: list[Any] = []
    reduce_retry_used = False
    reduce_skipped_reason: str | None = None
    draft_title = scope.title
    draft_summary = ""
    chronology: list[str] = []
    verification_queries: list[str] = []
    verified_sentences: list[VerifiedSentence] = []
    supported_summary = ""
    unsupported_sentences: list[str] = []
    scope_supported_fallback = ""
    scope_supported_fallback_used = False

    if not verified_page_summaries:
        reduce_skipped_reason = "no_verified_page_summaries"
        chronology = _dedupe_preserve_order(
            [
                _sanitize_generated_text(page_summary.supported_summary)
                for page_summary in page_summaries
                if _sanitize_generated_text(page_summary.supported_summary)
            ]
        )
    elif reduce_preflight_skip_reason is None:
        reduce_preflight_sources = await gateway.search_on_sources(
            query=reduce_preflight_query,
            data_sources=reduce_page_sources,
            limit=_candidate_limit(reduce_limit, effective_settings),
            score_threshold=0,
        )
        reduce_preflight_search_used = True
        reranked_reduce_sources, selected_reduce_sources = _prepare_selected_sources(
            query=reduce_preflight_query,
            hits=reduce_preflight_sources,
            settings=effective_settings,
            requested_limit=reduce_limit,
            source_groups=verified_page_source_groups,
        )
    if verified_page_summaries:
        reduce_data_sources = selected_reduce_sources or reduce_page_sources
        reduce_response, reduce_sources = await gateway.chat_on_sources(
            message=reduce_prompt,
            data_sources=reduce_data_sources,
            limit=max(_chat_limit(reduce_limit, effective_settings), len(reduce_data_sources)),
            score_threshold=0,
        )
        if not reduce_sources or _looks_like_retrieval_failure(reduce_response):
            if not reranked_reduce_sources:
                reduce_preflight_sources = await gateway.search_on_sources(
                    query=reduce_preflight_query,
                    data_sources=reduce_page_sources,
                    limit=_candidate_limit(reduce_limit, effective_settings),
                    score_threshold=0,
                )
                reranked_reduce_sources, selected_reduce_sources = _prepare_selected_sources(
                    query=reduce_preflight_query,
                    hits=reduce_preflight_sources,
                    settings=effective_settings,
                    requested_limit=reduce_limit,
                    source_groups=verified_page_source_groups,
                )
                reduce_retry_preflight_search_used = True
            retry_prompt = "\n\n".join(
                [
                    reduce_prompt,
                    (
                        f"OpenRAG search preflight found {len(reranked_reduce_sources)} candidate indexed hit(s) across "
                        f"{len(reduce_data_sources)} retrieval chunk file(s). Use the OpenSearch "
                        "Retrieval Tool and answer from the selected indexed files only."
                    ),
                ]
            )
            reduce_response, reduce_sources = await gateway.chat_on_sources(
                message=retry_prompt,
                data_sources=reduce_data_sources,
                limit=max(_chat_limit(reduce_limit, effective_settings), len(reduce_data_sources)),
                score_threshold=0,
            )
            reduce_retry_used = True
        reduce_payload = _extract_json_object(reduce_response) or {}
        draft_title = _sanitize_generated_text(str(reduce_payload.get("title") or scope.title).strip()) or scope.title
        draft_summary = _sanitize_generated_text(str(reduce_payload.get("summary") or reduce_response).strip())
        chronology_payload = reduce_payload.get("chronology")
        if isinstance(chronology_payload, list):
            chronology = _dedupe_preserve_order(
                [
                    _sanitize_generated_text(str(item).strip())
                    for item in chronology_payload
                    if _sanitize_generated_text(str(item).strip())
                ]
            )
        else:
            chronology = _dedupe_preserve_order(
                [
                    _sanitize_generated_text(page_summary.supported_summary or page_summary.summary)
                    for page_summary in verified_page_summaries
                    if _sanitize_generated_text(page_summary.supported_summary or page_summary.summary)
                ]
            )
        reduce_generation_evidence_pool = list(reranked_reduce_sources) if reranked_reduce_sources else []
        (
            verification_queries,
            verified_sentences,
            supported_summary,
            unsupported_sentences,
        ) = await _verify_text(
            gateway,
            text=draft_summary,
            data_sources=reduce_page_sources,
            settings=effective_settings,
            candidate_hits=reduce_generation_evidence_pool,
        )
        scope_supported_fallback = _compose_scope_supported_fallback(list(verified_page_summaries))
        if _should_use_scope_supported_fallback(
            current=supported_summary,
            fallback=scope_supported_fallback,
        ):
            supported_summary = scope_supported_fallback
            scope_supported_fallback_used = True

    return ScopedSummaryResult(
        run_id=manifest.run_id,
        scope=scope,
        source_filenames=reduce_page_sources,
        page_summaries=list(page_summaries),
        draft_title=draft_title,
        draft_summary=draft_summary,
        chronology=[str(item).strip() for item in chronology if str(item).strip()],
        verified_sentences=list(verified_sentences),
        supported_summary=supported_summary,
        unsupported_sentences=unsupported_sentences,
        debug={
            "page_requests": page_debug,
            "page_verification": page_verification_debug,
            "all_scope_source_filenames": all_page_sources,
            "verified_page_source_filenames": reduce_page_sources,
            "verified_page_count": len(verified_page_summaries),
            "total_page_count": len(page_summaries),
            "reduce_prompt": reduce_prompt,
            "reduce_preflight_query": reduce_preflight_query,
            "reduce_preflight_sources_before_rerank": [source.model_dump() for source in reduce_preflight_sources],
            "reduce_preflight_sources": [source.model_dump() for source in reranked_reduce_sources],
            "reduce_preflight_sources_after_rerank": [source.model_dump() for source in reranked_reduce_sources],
            "reduce_selected_source_filenames": selected_reduce_sources or reduce_page_sources,
            "reduce_limit": reduce_limit,
            "reduce_preflight_search_used": reduce_preflight_search_used,
            "reduce_preflight_skip_reason": reduce_preflight_skip_reason,
            "reduce_retry_preflight_search_used": reduce_retry_preflight_search_used,
            "reduce_retry_used": reduce_retry_used,
            "reduce_skipped_reason": reduce_skipped_reason,
            "reranking_enabled": _reranking_enabled(effective_settings),
            "reranker_location": _reranker_location(effective_settings),
            "reranker_type": _reranker_type(effective_settings),
            "reduce_response": reduce_response,
            "reduce_sources": [source.model_dump() for source in reduce_sources],
            "verification_queries": verification_queries,
            "verification_concurrency": effective_settings.verification_concurrency,
            "verification_candidate_limit": _verification_candidate_limit(effective_settings),
            "verification_reranking_enabled": bool(_verification_reranker_location(effective_settings)),
            "verification_reranker_location": _verification_reranker_location(effective_settings),
            "verification_reranker_type": _verification_reranker_type(effective_settings),
            "scope_supported_fallback": scope_supported_fallback,
            "scope_supported_fallback_used": scope_supported_fallback_used,
        },
    )
