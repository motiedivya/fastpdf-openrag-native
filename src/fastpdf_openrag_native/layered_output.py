from __future__ import annotations

import json
import re
from typing import Any, Protocol

from .models import (
    TRUTH_FIELD_LABELS,
    TRUTH_FIELD_NAMES,
    NoteValidationLayer,
    PageMapSummary,
    PresentationItem,
    PresentationLayer,
    PresentationSection,
    SupportedFact,
    SummaryScope,
    TruthLayerNote,
    ValidationCheck,
    ValidationLayer,
    VerifiedSentence,
)
from .prompts import build_presentation_editor_prompt, build_presentation_layer_prompt, build_truth_layer_prompt
from .settings import AppSettings

UNIT_TOKEN_RE = re.compile(r"[a-z0-9]+")
SOURCE_PAREN_RE = re.compile(r"\s*\((?:Source|Sources)\s*:[^)]+\)", flags=re.IGNORECASE)
SOURCE_BRACKET_RE = re.compile(r"\s*\[(?:Source|Sources)\s*:[^\]]+\]", flags=re.IGNORECASE)
SOURCE_CLAUSE_RE = re.compile(
    r"(?:^|[;,\s])(?:Source|Sources)\s*:\s*[A-Za-z0-9_.:/-]+(?:\s*(?:;|,)\s*[A-Za-z0-9_.:/-]+)*",
    flags=re.IGNORECASE,
)
SOURCE_FILE_RE = re.compile(r"\b[a-z0-9_.-]+__(?:c\d{4}\.md|p\d{4}\.html)\b", flags=re.IGNORECASE)
DATE_RE = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")
ICD_RE = re.compile(r"\b[A-TV-Z][0-9][0-9](?:\.[A-Z0-9]{1,4})?\b")
MEDICATION_SIGNAL_RE = re.compile(
    r"\b(?:mg|mcg|g|ml|tablet|capsule|patch|injection|po|bid|tid|qid|daily|nightly|prn|tylenol|aleve|hydrocodone|oxycodone|xylocaine|marcaine|depomedrol|morphine)\b",
    flags=re.IGNORECASE,
)
VITAL_SIGNAL_RE = re.compile(
    r"\b(?:bp|blood pressure|pulse|heart rate|resp(?:irations?)?|temperature|temp\.?|weight|height|bmi|spo2|oxygen saturation)\b",
    flags=re.IGNORECASE,
)
FOLLOW_UP_SIGNAL_RE = re.compile(r"\b(?:follow[- ]?up|return|prn|appointment)\b", flags=re.IGNORECASE)
ALLERGY_SIGNAL_RE = re.compile(r"\b(?:allerg|nkda|nka)\b", flags=re.IGNORECASE)
PLAN_SIGNAL_RE = re.compile(r"\b(?:plan|care plan|recommend|callback|clearance|follow[- ]?up)\b", flags=re.IGNORECASE)
ASSESSMENT_SIGNAL_RE = re.compile(r"\b(?:assessment|impression)\b", flags=re.IGNORECASE)
TREATMENT_SIGNAL_RE = re.compile(
    r"\b(?:treat(?:ed|ment)|procedure|block|injection|anesthetic|anesthesia|medication administered|xylocaine|marcaine|depomedrol|morphine)\b",
    flags=re.IGNORECASE,
)
CHIEF_COMPLAINT_SIGNAL_RE = re.compile(r"\b(?:chief complaint|complaint|reason for visit|possible|pain)\b", flags=re.IGNORECASE)
HPI_SIGNAL_RE = re.compile(r"\b(?:history of present illness|\bhpi\b|reported|noted|for \d+ weeks)\b", flags=re.IGNORECASE)
ROS_SIGNAL_RE = re.compile(r"\b(?:review of systems|ros|muscle or joint pain|hearing loss|ear pain)\b", flags=re.IGNORECASE)
PE_SIGNAL_RE = re.compile(
    r"\b(?:physical exam|exam|tenderness|adenopathy|murmur|gait|cranial nerves|well perfused)\b",
    flags=re.IGNORECASE,
)
SOCIAL_SIGNAL_RE = re.compile(r"\b(?:social history|smokes|tobacco|alcohol|drug use|marital status|employment)\b", flags=re.IGNORECASE)
PMH_SIGNAL_RE = re.compile(r"\b(?:past medical history|\bpmh\b|history of)\b", flags=re.IGNORECASE)
PSH_SIGNAL_RE = re.compile(r"\b(?:past surgical history|\bpsh\b|surgical history)\b", flags=re.IGNORECASE)
LAB_SIGNAL_RE = re.compile(r"\b(?:lab|labs|glucose|a1c|creatinine|wbc|hgb|platelet|abnormal)\b", flags=re.IGNORECASE)
FACILITY_SIGNAL_RE = re.compile(r"\b(?:clinic|medical|center|hospital|office|health|esperanza|oceanview|emory)\b", flags=re.IGNORECASE)
PROVIDER_SIGNAL_RE = re.compile(r"\b(?:m\.d\.|d\.o\.|np\b|pa\b|cfnp|physician|provider|dr\.)\b", flags=re.IGNORECASE)
NOTE_TYPE_SIGNAL_RE = re.compile(r"\b(?:visit note|progress note|operative note|phone ?msg|final report|procedure note|history and physical)\b", flags=re.IGNORECASE)
PATIENT_SIGNAL_RE = re.compile(r"\b(?:mr\.|mrs\.|ms\.|patient|dob)\b", flags=re.IGNORECASE)
NORMAL_FINDING_SIGNAL_RE = re.compile(
    r"\b(?:well[- ]appearing|not in acute distress|normal|regular rate and rhythm|s1 and s2|no murmurs|no rubs|no gallops|well perfused|radial pulses(?:\s*2\+)?|no cervical adenopathy|normal mood and affect|appropriate judgment and insight|grossly intact memory|alert and oriented(?: status)?|cranial nerves grossly intact|normal gait|equal movement of all extremities|normal ear|normal oropharyngeal|oropharyngeal findings)\b",
    flags=re.IGNORECASE,
)
NEGATIVE_FINDING_SIGNAL_RE = re.compile(
    r"\b(?:denies|negative for|without distress|without abnormality|no\s+(?:murmurs?|rubs?|gallops?|cervical adenopathy|acute distress|focal deficits?|edema))\b",
    flags=re.IGNORECASE,
)
POSITIVE_PRESENTATION_SIGNAL_RE = re.compile(
    r"\b(?:pain|tender|swelling|infection|hearing loss|otalgia|abnormal|positive|limited|decreased|effusion|erythema|lesion|mass|ulcer|callback|clearance|procedure|block|injection|diagnosis|plan|follow[- ]?up|medication|allerg(?:y|ies)|icd|hpi|chief complaint)\b",
    flags=re.IGNORECASE,
)

PRESENTATION_FIELD_ORDER = (
    "chief_complaint",
    "hpi",
    "pmh",
    "psh",
    "social_history",
    "allergies",
    "medications",
    "vitals",
    "abnormal_labs",
    "diagnoses",
    "assessment",
    "treatment",
    "plan",
    "follow_up",
    "positive_ros",
    "positive_physical_exam",
    "residual_supported_facts",
)

FIELD_SIGNAL_RULES = {
    "date_of_service": lambda text, hints: bool(DATE_RE.search(text) or hints.get("service_date")),
    "facility": lambda text, hints: bool(FACILITY_SIGNAL_RE.search(text) or hints.get("label")),
    "provider": lambda text, hints: bool(PROVIDER_SIGNAL_RE.search(text)),
    "patient_reference": lambda text, hints: bool(hints.get("patient_name") or PATIENT_SIGNAL_RE.search(text)),
    "note_type": lambda text, hints: bool(NOTE_TYPE_SIGNAL_RE.search(text) or hints.get("label")),
    "chief_complaint": lambda text, hints: bool(CHIEF_COMPLAINT_SIGNAL_RE.search(text)),
    "hpi": lambda text, hints: bool(HPI_SIGNAL_RE.search(text)),
    "pmh": lambda text, hints: bool(PMH_SIGNAL_RE.search(text)),
    "psh": lambda text, hints: bool(PSH_SIGNAL_RE.search(text)),
    "social_history": lambda text, hints: bool(SOCIAL_SIGNAL_RE.search(text)),
    "allergies": lambda text, hints: bool(ALLERGY_SIGNAL_RE.search(text)),
    "medications": lambda text, hints: bool(MEDICATION_SIGNAL_RE.search(text)),
    "vitals": lambda text, hints: bool(VITAL_SIGNAL_RE.search(text)),
    "abnormal_labs": lambda text, hints: bool(LAB_SIGNAL_RE.search(text)),
    "diagnoses": lambda text, hints: bool(ICD_RE.search(text) or "diagnosis" in text or "diagnoses" in text),
    "assessment": lambda text, hints: bool(ASSESSMENT_SIGNAL_RE.search(text)),
    "treatment": lambda text, hints: bool(TREATMENT_SIGNAL_RE.search(text)),
    "plan": lambda text, hints: bool(PLAN_SIGNAL_RE.search(text)),
    "follow_up": lambda text, hints: bool(FOLLOW_UP_SIGNAL_RE.search(text)),
    "positive_ros": lambda text, hints: bool(ROS_SIGNAL_RE.search(text)),
    "positive_physical_exam": lambda text, hints: bool(PE_SIGNAL_RE.search(text)),
    "residual_supported_facts": lambda text, hints: False,
}


class StructuredOutputGateway(Protocol):
    async def chat_on_sources(
        self,
        *,
        message: str,
        data_sources: list[str] | None = None,
        limit: int = 6,
        score_threshold: float = 0,
        llm_model: str | None = None,
        llm_provider: str | None = None,
        disable_retrieval: bool = False,
    ) -> tuple[str, list[Any]]: ...


def _extract_json_object(text: str) -> dict[str, Any] | None:
    clean = text.strip()
    if not clean:
        return None
    candidates = [clean]
    match = re.search(r"\{.*\}", clean, flags=re.DOTALL)
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
    clean = clean.strip(" \t\r\n-–—;")
    return clean


def _ensure_sentence(text: str) -> str:
    clean = re.sub(r"^\s*[-*•]+\s*", "", _sanitize_generated_text(text)).strip()
    clean = re.sub(r"\s+", " ", clean)
    if not clean:
        return ""
    if not re.search(r"[.!?]$", clean):
        clean = f"{clean}."
    return clean


def _split_sentences(text: str) -> list[str]:
    clean = _sanitize_generated_text(text)
    if not clean:
        return []
    clean = re.sub(r"\s*[•;]\s*", ". ", clean)
    clean = re.sub(r"\s*\n+\s*", " ", clean)
    parts = re.split(r"(?<=[.!?])\s+", clean)
    return [sentence for sentence in (_ensure_sentence(part) for part in parts) if sentence]


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _clean_fact_value(text: str) -> str:
    clean = _sanitize_generated_text(text)
    clean = clean.rstrip(".")
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def _extract_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list):
        values = [item for item in value if isinstance(item, str)]
    else:
        return []
    return _dedupe_preserve_order([_clean_fact_value(item) for item in values if _clean_fact_value(item)])


def _token_set(text: str) -> set[str]:
    return {token for token in UNIT_TOKEN_RE.findall(_sanitize_generated_text(text).lower()) if len(token) > 1}


def _fact_overlap_score(left: str, right: str) -> float:
    left_clean = _sanitize_generated_text(left).lower()
    right_clean = _sanitize_generated_text(right).lower()
    if not left_clean or not right_clean:
        return 0.0
    if left_clean in right_clean or right_clean in left_clean:
        return 1.0
    left_tokens = _token_set(left_clean)
    right_tokens = _token_set(right_clean)
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens)
    union = len(left_tokens | right_tokens)
    return overlap / max(1, union)


def _collect_supported_rows(page_summary: PageMapSummary) -> list[VerifiedSentence]:
    rows = [row for row in page_summary.verified_sentences if row.supported]
    rows.extend(row for row in page_summary.verified_key_facts if row.supported)
    if rows:
        return rows
    fallback_hits = list(page_summary.retrieved_sources[:3])
    fallback_rows: list[VerifiedSentence] = []
    for sentence in _split_sentences(page_summary.supported_summary or page_summary.summary):
        fallback_rows.append(VerifiedSentence(sentence=sentence, supported=True, evidence=fallback_hits))
    for fact in page_summary.supported_key_facts or page_summary.key_facts:
        sentence = _ensure_sentence(fact)
        if sentence:
            fallback_rows.append(VerifiedSentence(sentence=sentence, supported=True, evidence=fallback_hits))
    return fallback_rows


def _match_fact_rows(value: str, verified_rows: list[VerifiedSentence]) -> list[VerifiedSentence]:
    scored: list[tuple[float, int, VerifiedSentence]] = []
    for index, row in enumerate(verified_rows):
        score = _fact_overlap_score(value, row.sentence)
        if score <= 0:
            continue
        scored.append((score, index, row))
    if not scored:
        return []
    scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    best_score = scored[0][0]
    selected = [row for score, _, row in scored if score >= max(0.24, best_score - 0.28)]
    return selected[:3]


def _merge_hits(rows: list[VerifiedSentence], fallback_hits: list[Any]) -> list[Any]:
    merged: list[Any] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        for hit in row.evidence:
            key = (str(getattr(hit, "filename", "")), str(getattr(hit, "text", "")))
            if key in seen:
                continue
            seen.add(key)
            merged.append(hit)
    if merged:
        return merged
    for hit in fallback_hits:
        key = (str(getattr(hit, "filename", "")), str(getattr(hit, "text", "")))
        if key in seen:
            continue
        seen.add(key)
        merged.append(hit)
    return merged


def _build_supported_fact(value: str, verified_rows: list[VerifiedSentence], fallback_hits: list[Any]) -> SupportedFact:
    matched_rows = _match_fact_rows(value, verified_rows)
    evidence_sentences = _dedupe_preserve_order([_ensure_sentence(row.sentence) for row in matched_rows])
    evidence = _merge_hits(matched_rows, fallback_hits)
    return SupportedFact(value=_clean_fact_value(value), evidence_sentences=evidence_sentences, evidence=evidence)


def _append_fact(target: list[SupportedFact], fact: SupportedFact) -> None:
    key = _clean_fact_value(fact.value).lower()
    if not key:
        return
    for existing in target:
        if _clean_fact_value(existing.value).lower() != key:
            continue
        existing.evidence_sentences = _dedupe_preserve_order([*existing.evidence_sentences, *fact.evidence_sentences])
        existing.evidence = _merge_hits([], [*existing.evidence, *fact.evidence])
        return
    target.append(fact)


def _metadata_hints_for_page(page: Any) -> dict[str, str]:
    return {
        "service_date": _clean_fact_value(getattr(page, "service_date", "") or ""),
        "patient_name": _clean_fact_value(getattr(page, "patient_name", "") or ""),
        "label": _clean_fact_value(getattr(page, "label", "") or ""),
    }


def _fallback_truth_layer(page: Any, page_summary: PageMapSummary, verified_rows: list[VerifiedSentence]) -> TruthLayerNote:
    fallback_hits = list(page_summary.retrieved_sources[:3])
    note = TruthLayerNote(
        note_id=f"{page.pdf_id}-p{page.page:04d}",
        pdf_ids=[page.pdf_id],
        pages=[page.page],
        source_filenames=list(page.retrieval_sources()),
    )
    metadata_hints = _metadata_hints_for_page(page)
    if metadata_hints.get("service_date"):
        _append_fact(note.date_of_service, SupportedFact(value=metadata_hints["service_date"], evidence=list(fallback_hits)))
    if metadata_hints.get("patient_name"):
        _append_fact(note.patient_reference, SupportedFact(value=metadata_hints["patient_name"], evidence=list(fallback_hits)))
    for unit in _dedupe_preserve_order([
        *_split_sentences(page_summary.supported_summary or page_summary.summary),
        *[_ensure_sentence(item) for item in (page_summary.supported_key_facts or page_summary.key_facts)],
    ]):
        _append_fact(note.residual_supported_facts, _build_supported_fact(unit, verified_rows, fallback_hits))
    note.debug = {"metadata_hints": metadata_hints, "source_units": note.field_values("residual_supported_facts")}
    return note


def _note_from_payload(
    payload: dict[str, Any],
    *,
    page: Any,
    page_summary: PageMapSummary,
    verified_rows: list[VerifiedSentence],
) -> TruthLayerNote:
    note = TruthLayerNote(
        note_id=f"{page.pdf_id}-p{page.page:04d}",
        pdf_ids=[page.pdf_id],
        pages=[page.page],
        source_filenames=list(page.retrieval_sources()),
    )
    fallback_hits = list(page_summary.retrieved_sources[:3])
    metadata_hints = _metadata_hints_for_page(page)
    for field_name in TRUTH_FIELD_NAMES:
        values = _extract_string_list(payload.get(field_name))
        for value in values:
            _append_fact(getattr(note, field_name), _build_supported_fact(value, verified_rows, fallback_hits))
    if metadata_hints.get("service_date") and not note.date_of_service:
        _append_fact(note.date_of_service, SupportedFact(value=metadata_hints["service_date"], evidence=list(fallback_hits)))
    if metadata_hints.get("patient_name") and not note.patient_reference:
        _append_fact(note.patient_reference, SupportedFact(value=metadata_hints["patient_name"], evidence=list(fallback_hits)))

    source_units = _dedupe_preserve_order([
        *_split_sentences(page_summary.supported_summary or page_summary.summary),
        *[_ensure_sentence(item) for item in (page_summary.supported_key_facts or page_summary.key_facts)],
    ])
    existing_values = [value for field_name in TRUTH_FIELD_NAMES for value in note.field_values(field_name)]
    for unit in source_units:
        if any(_fact_overlap_score(unit, existing) >= 0.72 for existing in existing_values):
            continue
        _append_fact(note.residual_supported_facts, _build_supported_fact(unit, verified_rows, fallback_hits))
        existing_values.append(unit)
    note.debug = {"metadata_hints": metadata_hints, "source_units": source_units}
    return note


async def extract_truth_layer_note(
    gateway: StructuredOutputGateway,
    *,
    scope: SummaryScope,
    page: Any,
    page_summary: PageMapSummary,
    settings: AppSettings,
) -> tuple[TruthLayerNote, dict[str, Any]]:
    verified_rows = _collect_supported_rows(page_summary)
    metadata_hints = _metadata_hints_for_page(page)
    prompt = build_truth_layer_prompt(
        scope,
        pdf_id=page.pdf_id,
        page=page.page,
        source_filename=page.source_filename,
        supported_summary=page_summary.supported_summary or page_summary.summary,
        supported_key_facts=page_summary.supported_key_facts or page_summary.key_facts,
        verified_sentences=[row.sentence for row in verified_rows],
        metadata_hints=metadata_hints,
    )
    data_sources = list(page.retrieval_sources())
    response_text, sources = await gateway.chat_on_sources(
        message=prompt,
        data_sources=data_sources,
        limit=max(6, len(data_sources)),
        score_threshold=0,
        llm_model=settings.extractor_llm_model,
        llm_provider=settings.extractor_llm_provider,
    )
    payload = _extract_json_object(response_text) or {}
    truth_note = _note_from_payload(payload, page=page, page_summary=page_summary, verified_rows=verified_rows)
    if not truth_note.populated_fields():
        truth_note = _fallback_truth_layer(page, page_summary, verified_rows)
    truth_note.debug = {
        **dict(truth_note.debug),
        "truth_prompt": prompt,
        "truth_response": response_text,
        "truth_payload": payload,
        "truth_sources": [source.model_dump() if hasattr(source, "model_dump") else source for source in sources],
    }
    return truth_note, {
        "truth_prompt": prompt,
        "truth_response": response_text,
        "truth_payload": payload,
        "source_units": truth_note.debug.get("source_units", []),
        "metadata_hints": metadata_hints,
        "populated_fields": truth_note.populated_fields(),
    }


def _normalize_group_part(values: list[str]) -> str:
    if not values:
        return ""
    clean = _clean_fact_value(values[0]).lower()
    clean = re.sub(r"[^a-z0-9]+", "-", clean).strip("-")
    return clean


def _group_key_for_note(note: TruthLayerNote) -> str:
    parts = [
        _normalize_group_part(note.field_values("date_of_service")),
        _normalize_group_part(note.field_values("facility")),
        _normalize_group_part(note.field_values("provider")),
        _normalize_group_part(note.field_values("patient_reference")),
        _normalize_group_part(note.field_values("note_type")),
    ]
    informative = [part for part in parts if part]
    if len(informative) >= 2:
        return "|".join(parts)
    pdf_id = note.pdf_ids[0] if note.pdf_ids else "note"
    first_page = note.pages[0] if note.pages else 0
    return f"{pdf_id}|page|{first_page}"


def _merge_fact_lists(target: list[SupportedFact], incoming: list[SupportedFact]) -> None:
    for fact in incoming:
        _append_fact(target, fact)


def group_truth_layer_notes(notes: list[TruthLayerNote]) -> list[TruthLayerNote]:
    grouped: dict[str, TruthLayerNote] = {}
    for note in sorted(notes, key=lambda row: (row.pages[0] if row.pages else 0, row.note_id)):
        group_key = _group_key_for_note(note)
        current = grouped.get(group_key)
        if current is None:
            current = note.model_copy(deep=True)
            current.debug = {**dict(current.debug), "group_key": group_key}
            grouped[group_key] = current
            continue
        current.pdf_ids = _dedupe_preserve_order([*current.pdf_ids, *note.pdf_ids])
        current.pages = sorted(set([*current.pages, *note.pages]))
        current.source_filenames = _dedupe_preserve_order([*current.source_filenames, *note.source_filenames])
        for field_name in TRUTH_FIELD_NAMES:
            _merge_fact_lists(getattr(current, field_name), getattr(note, field_name))
        current.debug = {
            **dict(current.debug),
            "group_key": group_key,
            "source_units": _dedupe_preserve_order([
                *list(current.debug.get("source_units", [])),
                *list(note.debug.get("source_units", [])),
            ]),
            "metadata_hints": {
                **dict(current.debug.get("metadata_hints", {})),
                **dict(note.debug.get("metadata_hints", {})),
            },
        }
    ordered = sorted(grouped.values(), key=lambda row: (row.pages[0] if row.pages else 0, row.note_id))
    for index, note in enumerate(ordered, start=1):
        note.note_id = f"note-{index:03d}"
    return ordered


def _source_detected(field_name: str, note: TruthLayerNote) -> bool:
    source_text = " ".join(note.debug.get("source_units", [])).lower()
    metadata_hints = {key: str(value or "") for key, value in dict(note.debug.get("metadata_hints", {})).items()}
    detector = FIELD_SIGNAL_RULES.get(field_name)
    if detector is None:
        return False
    return bool(detector(source_text, metadata_hints))


def validate_truth_layer(notes: list[TruthLayerNote]) -> ValidationLayer:
    note_validations: list[NoteValidationLayer] = []
    for note in notes:
        checks: list[ValidationCheck] = []
        missing_required_fields: list[str] = []
        requested_fields: list[str] = []
        populated_fields = note.populated_fields()
        for field_name in TRUTH_FIELD_NAMES:
            label = TRUTH_FIELD_LABELS[field_name]
            source_detected = _source_detected(field_name, note)
            populated = bool(getattr(note, field_name))
            required = source_detected and field_name != "residual_supported_facts"
            if required:
                requested_fields.append(field_name)
            if required and not populated:
                missing_required_fields.append(field_name)
            if required and not populated:
                message = f"Supported {label.lower()} content appears to be present in the verified evidence but is missing from the truth layer."
            elif populated:
                message = f"Captured supported {label.lower()} content."
            else:
                message = f"No supported {label.lower()} content was detected."
            checks.append(
                ValidationCheck(
                    field_name=field_name,
                    label=label,
                    required=required,
                    populated=populated,
                    source_detected=source_detected,
                    message=message,
                )
            )
        note_validations.append(
            NoteValidationLayer(
                note_id=note.note_id,
                passed=not missing_required_fields,
                requested_fields=requested_fields,
                populated_fields=populated_fields,
                missing_required_fields=missing_required_fields,
                checks=checks,
                debug={
                    "group_key": note.debug.get("group_key"),
                    "source_units": note.debug.get("source_units", []),
                },
            )
        )
    return ValidationLayer(
        passed=all(row.passed for row in note_validations) if note_validations else True,
        notes=note_validations,
        debug={"note_count": len(note_validations)},
    )


def _format_fact_values(facts: list[SupportedFact]) -> str:
    values = [_clean_fact_value(fact.value) for fact in facts if _clean_fact_value(fact.value)]
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return "; ".join(values[:-1]) + f"; and {values[-1]}"


def _aggregate_evidence(facts: list[SupportedFact]) -> list[Any]:
    return _merge_hits([], [hit for fact in facts for hit in fact.evidence])


def _item(text: str, *, field_name: str, note: TruthLayerNote, facts: list[SupportedFact], index: int) -> PresentationItem:
    item_id = f"{note.note_id}__{field_name}__{index:02d}"
    return PresentationItem(
        item_id=item_id,
        text=_ensure_sentence(text),
        field_name=field_name,
        note_id=note.note_id,
        fact_ids=[item_id],
        rendered_by_model=False,
        evidence=_aggregate_evidence(facts),
        candidate_filenames=list(note.source_filenames),
        pdf_ids=list(note.pdf_ids),
        pages=list(note.pages),
    )


def _first_value(note: TruthLayerNote, field_name: str) -> str:
    values = note.field_values(field_name)
    return values[0] if values else ""


def _looks_like_normal_finding(text: str) -> bool:
    clean = _sanitize_generated_text(text)
    if not clean:
        return False
    return bool(
        (NORMAL_FINDING_SIGNAL_RE.search(clean) or NEGATIVE_FINDING_SIGNAL_RE.search(clean))
        and not POSITIVE_PRESENTATION_SIGNAL_RE.search(clean)
    )


def _filter_presentable_facts(field_name: str, facts: list[SupportedFact]) -> list[SupportedFact]:
    if field_name not in {"positive_ros", "positive_physical_exam", "residual_supported_facts"}:
        return facts
    filtered: list[SupportedFact] = []
    for fact in facts:
        clean = _clean_fact_value(fact.value)
        if not clean:
            continue
        if _looks_like_normal_finding(clean):
            if field_name in {"positive_ros", "positive_physical_exam"}:
                continue
            if PE_SIGNAL_RE.search(clean) or ROS_SIGNAL_RE.search(clean):
                continue
        filtered.append(fact)
    return filtered


def _render_intro_items(note: TruthLayerNote) -> list[PresentationItem]:
    date_of_service = _first_value(note, "date_of_service")
    facility = _first_value(note, "facility")
    provider = _first_value(note, "provider")
    patient = _first_value(note, "patient_reference")
    note_type = _first_value(note, "note_type")
    facts: list[SupportedFact] = [
        *note.date_of_service,
        *note.facility,
        *note.provider,
        *note.patient_reference,
        *note.note_type,
    ]
    if not any([date_of_service, facility, provider, patient, note_type]):
        return []
    start = f"On {date_of_service}" if date_of_service else "In this note"
    if facility:
        start = f"{start}, at {facility}"
    if provider:
        start = f"{start}, {provider}"
    detail = f"documented {note_type}" if note_type else "documented the note"
    sentence = f"{start} {detail}"
    if patient:
        sentence = f"{sentence} for {patient}"
    return [_item(sentence, field_name="intro", note=note, facts=facts, index=1)]


def _render_field_items(note: TruthLayerNote, field_name: str) -> list[PresentationItem]:
    facts = _filter_presentable_facts(field_name, list(getattr(note, field_name)))
    if not facts:
        return []
    if field_name == "residual_supported_facts":
        return [_item(fact.value, field_name=field_name, note=note, facts=[fact], index=index) for index, fact in enumerate(facts, start=1)]
    value_text = _format_fact_values(facts)
    templates = {
        "chief_complaint": f"Chief complaint was {value_text}",
        "hpi": f"History of present illness documented {value_text}",
        "pmh": f"Past medical history included {value_text}",
        "psh": f"Past surgical history included {value_text}",
        "social_history": f"Social history documented {value_text}",
        "allergies": f"Allergies were {value_text}",
        "medications": f"Medications included {value_text}",
        "vitals": f"Vitals included {value_text}",
        "abnormal_labs": f"Abnormal labs included {value_text}",
        "diagnoses": f"Diagnoses included {value_text}",
        "assessment": f"Assessment documented {value_text}",
        "treatment": f"Treatment included {value_text}",
        "plan": f"Plan included {value_text}",
        "follow_up": f"Follow up included {value_text}",
        "positive_ros": f"Positive review of systems findings included {value_text}",
        "positive_physical_exam": f"Positive physical exam findings included {value_text}",
    }
    sentence = templates.get(field_name)
    if not sentence:
        return []
    return [_item(sentence, field_name=field_name, note=note, facts=facts, index=1)]


def _build_candidate_presentation_sections(notes: list[TruthLayerNote]) -> list[PresentationSection]:
    sections: list[PresentationSection] = []
    for index, note in enumerate(notes, start=1):
        date_label = _first_value(note, "date_of_service")
        title = f"On {date_label}" if date_label else f"Note {index}"
        items: list[PresentationItem] = []
        items.extend(_render_intro_items(note))
        for field_name in PRESENTATION_FIELD_ORDER:
            items.extend(_render_field_items(note, field_name))
        if not items and note.residual_supported_facts:
            items.extend(_render_field_items(note, "residual_supported_facts"))
        sections.append(
            PresentationSection(
                section_id=f"presentation-{index:03d}",
                title=title,
                note_id=note.note_id,
                items=items,
            )
        )
    return sections


def _presentation_from_sections(
    *,
    scope: SummaryScope,
    sections: list[PresentationSection],
    rendered_by_model: bool,
    debug: dict[str, Any] | None = None,
) -> tuple[PresentationLayer, list[VerifiedSentence], str]:
    verified_sentences: list[VerifiedSentence] = []
    normalized_sections: list[PresentationSection] = []
    paragraphs: list[str] = []
    for section in sections:
        items: list[PresentationItem] = []
        for item in section.items:
            normalized_item = item.model_copy(
                update={
                    "text": _ensure_sentence(item.text),
                    "rendered_by_model": rendered_by_model or item.rendered_by_model,
                }
            )
            items.append(normalized_item)
            verified_sentences.append(
                VerifiedSentence(
                    sentence=_ensure_sentence(normalized_item.text),
                    supported=True,
                    evidence=list(normalized_item.evidence),
                )
            )
        normalized_sections.append(section.model_copy(update={"items": items}))
        paragraph = " ".join(item.text for item in items).strip()
        if paragraph:
            paragraphs.append(paragraph)
    narrative = "\n\n".join(paragraphs).strip()
    return (
        PresentationLayer(
            title=scope.title,
            narrative=narrative,
            sections=normalized_sections,
            debug={**(debug or {}), "note_count": len(normalized_sections), "rendered_by_model": rendered_by_model},
        ),
        verified_sentences,
        narrative,
    )

def _truth_layer_payload(notes: list[TruthLayerNote]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for note in notes:
        row: dict[str, Any] = {
            "note_id": note.note_id,
            "pdf_ids": list(note.pdf_ids),
            "pages": list(note.pages),
            "source_filenames": list(note.source_filenames),
        }
        for field_name in TRUTH_FIELD_NAMES:
            row[field_name] = note.field_values(field_name)
        payload.append(row)
    return payload


def _candidate_section_payload(sections: list[PresentationSection]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for section in sections:
        payload.append(
            {
                "title": section.title,
                "note_id": section.note_id,
                "items": [
                    {
                        "item_id": item.item_id,
                        "field_name": item.field_name,
                        "text": item.text,
                        "fact_ids": item.fact_ids or [item.item_id],
                    }
                    for item in section.items
                ],
            }
        )
    return payload


def _aggregate_items(items: list[PresentationItem]) -> tuple[list[Any], list[str], list[str], list[int]]:
    evidence = _merge_hits([], [hit for item in items for hit in item.evidence])
    candidate_filenames = _dedupe_preserve_order([filename for item in items for filename in item.candidate_filenames])
    pdf_ids = _dedupe_preserve_order([pdf_id for item in items for pdf_id in item.pdf_ids])
    pages = sorted(set(page for item in items for page in item.pages))
    return evidence, candidate_filenames, pdf_ids, pages


def _rendered_sections_from_payload(
    payload: dict[str, Any],
    *,
    candidate_sections: list[PresentationSection],
) -> list[PresentationSection]:
    fact_index: dict[str, PresentationItem] = {}
    for section in candidate_sections:
        for item in section.items:
            fact_index[item.item_id] = item
            for fact_id in item.fact_ids or []:
                fact_index.setdefault(fact_id, item)
    rendered_sections: list[PresentationSection] = []
    for section_index, row in enumerate(payload.get("sections", []), start=1):
        if not isinstance(row, dict):
            continue
        items_payload = row.get("items")
        if not isinstance(items_payload, list):
            continue
        note_id = row.get("note_id") if isinstance(row.get("note_id"), str) else None
        rendered_items: list[PresentationItem] = []
        for item_index, item_row in enumerate(items_payload, start=1):
            if not isinstance(item_row, dict):
                continue
            text = _ensure_sentence(str(item_row.get("text") or ""))
            requested_fact_ids = [fact_id for fact_id in item_row.get("fact_ids", []) if isinstance(fact_id, str) and fact_id in fact_index]
            if not text or not requested_fact_ids:
                continue
            seen_item_ids: set[str] = set()
            facts: list[PresentationItem] = []
            for fact_id in requested_fact_ids:
                fact = fact_index[fact_id]
                if fact.item_id in seen_item_ids:
                    continue
                seen_item_ids.add(fact.item_id)
                facts.append(fact)
            evidence, candidate_filenames, pdf_ids, pages = _aggregate_items(facts)
            field_names = _dedupe_preserve_order([fact.field_name for fact in facts if fact.field_name])
            inferred_note_id = note_id or next((fact.note_id for fact in facts if fact.note_id), None)
            rendered_items.append(
                PresentationItem(
                    item_id=f"rendered-{section_index:03d}-{item_index:03d}",
                    text=text,
                    field_name=field_names[0] if len(field_names) == 1 else None,
                    note_id=inferred_note_id,
                    fact_ids=requested_fact_ids,
                    rendered_by_model=True,
                    evidence=evidence,
                    candidate_filenames=candidate_filenames,
                    pdf_ids=pdf_ids,
                    pages=pages,
                )
            )
        if not rendered_items:
            continue
        rendered_sections.append(
            PresentationSection(
                section_id=f"rendered-section-{section_index:03d}",
                title=str(row.get("title") or f"Note {section_index}").strip() or f"Note {section_index}",
                note_id=note_id or next((item.note_id for item in rendered_items if item.note_id), None),
                items=rendered_items,
            )
        )
    return rendered_sections


async def render_presentation_layer(
    gateway: StructuredOutputGateway,
    *,
    notes: list[TruthLayerNote],
    scope: SummaryScope,
    settings: AppSettings,
) -> tuple[PresentationLayer, PresentationLayer | None, PresentationLayer, list[VerifiedSentence], str]:
    candidate_sections = _build_candidate_presentation_sections(notes)
    plan_layer, plan_verified_sentences, plan_narrative = _presentation_from_sections(
        scope=scope,
        sections=candidate_sections,
        rendered_by_model=False,
        debug={"renderer": "deterministic_plan", "stage": "plan"},
    )
    if not candidate_sections:
        final_plan = plan_layer.model_copy(
            update={
                "debug": {
                    **dict(plan_layer.debug),
                    "stage": "final",
                    "editor": "deterministic_plan",
                    "editor_fallback_reason": "no_candidate_sections",
                }
            }
        )
        return plan_layer, None, final_plan, plan_verified_sentences, plan_narrative

    prompt = build_presentation_layer_prompt(
        scope,
        truth_payload={"notes": _truth_layer_payload(notes)},
        candidate_sections=_candidate_section_payload(candidate_sections),
    )
    renderer_data_sources = ["__fastpdf_openrag_renderer_no_sources__"] if settings.renderer_disable_retrieval else None
    response_text, sources = await gateway.chat_on_sources(
        message=prompt,
        data_sources=renderer_data_sources,
        limit=1,
        score_threshold=0,
        llm_model=settings.renderer_llm_model,
        llm_provider=settings.renderer_llm_provider,
        disable_retrieval=settings.renderer_disable_retrieval,
    )
    payload = _extract_json_object(response_text) or {}
    rendered_sections = _rendered_sections_from_payload(payload, candidate_sections=candidate_sections)
    if not rendered_sections:
        final_plan = plan_layer.model_copy(
            update={
                "debug": {
                    **dict(plan_layer.debug),
                    "renderer_prompt": prompt,
                    "renderer_response": response_text,
                    "renderer_payload": payload,
                    "renderer_sources": [source.model_dump() if hasattr(source, "model_dump") else source for source in sources],
                    "renderer": "deterministic_plan",
                    "stage": "final",
                    "renderer_fallback_reason": "empty_or_invalid_draft",
                }
            }
        )
        return plan_layer, None, final_plan, plan_verified_sentences, plan_narrative

    draft_layer, draft_verified_sentences, draft_narrative = _presentation_from_sections(
        scope=scope,
        sections=rendered_sections,
        rendered_by_model=True,
        debug={
            "renderer": "llm_draft",
            "stage": "draft",
            "renderer_prompt": prompt,
            "renderer_response": response_text,
            "renderer_payload": payload,
            "renderer_sources": [source.model_dump() if hasattr(source, "model_dump") else source for source in sources],
        },
    )

    editor_prompt = build_presentation_editor_prompt(
        scope,
        truth_payload={"notes": _truth_layer_payload(notes)},
        draft_sections=_candidate_section_payload(draft_layer.sections),
    )
    editor_model = settings.editor_llm_model or settings.renderer_llm_model
    editor_provider = settings.editor_llm_provider or settings.renderer_llm_provider
    editor_data_sources = ["__fastpdf_openrag_editor_no_sources__"] if settings.editor_disable_retrieval else None
    editor_response_text, editor_sources = await gateway.chat_on_sources(
        message=editor_prompt,
        data_sources=editor_data_sources,
        limit=1,
        score_threshold=0,
        llm_model=editor_model,
        llm_provider=editor_provider,
        disable_retrieval=settings.editor_disable_retrieval,
    )
    editor_payload = _extract_json_object(editor_response_text) or {}
    edited_sections = _rendered_sections_from_payload(editor_payload, candidate_sections=draft_layer.sections)
    if not edited_sections:
        final_draft = draft_layer.model_copy(
            update={
                "debug": {
                    **dict(draft_layer.debug),
                    "stage": "final",
                    "editor": "draft_fallback",
                    "editor_prompt": editor_prompt,
                    "editor_response": editor_response_text,
                    "editor_payload": editor_payload,
                    "editor_sources": [source.model_dump() if hasattr(source, "model_dump") else source for source in editor_sources],
                    "editor_fallback_reason": "empty_or_invalid_payload",
                }
            }
        )
        return plan_layer, draft_layer, final_draft, draft_verified_sentences, draft_narrative

    final_layer, verified_sentences, narrative = _presentation_from_sections(
        scope=scope,
        sections=edited_sections,
        rendered_by_model=True,
        debug={
            "renderer": "llm_editor",
            "stage": "final",
            "draft_renderer_prompt": prompt,
            "draft_renderer_response": response_text,
            "draft_renderer_payload": payload,
            "draft_renderer_sources": [source.model_dump() if hasattr(source, "model_dump") else source for source in sources],
            "editor_prompt": editor_prompt,
            "editor_response": editor_response_text,
            "editor_payload": editor_payload,
            "editor_sources": [source.model_dump() if hasattr(source, "model_dump") else source for source in editor_sources],
        },
    )
    return plan_layer, draft_layer, final_layer, verified_sentences, narrative
