from __future__ import annotations

import json

from .models import PageMapSummary, SummaryScope


def build_page_map_prompt(
    scope: SummaryScope,
    *,
    pdf_id: str,
    page: int,
    source_filename: str,
    retrieval_source_count: int = 1,
    retrieval_source_hints: list[str] | None = None,
) -> str:
    source_hint_block = ""
    if retrieval_source_hints:
        rendered_hints = "\n".join(retrieval_source_hints)
        source_hint_block = f"\n\nLocal retrieval inventory for this page (titles/previews from the current run only):\n{rendered_hints}"
    return f"""
You are summarizing a single page from an already indexed medical/legal document.

Scope title: {scope.title}
Scope objective: {scope.objective}
Current page: {pdf_id} page {page}
Page artifact filename: {source_filename}
Indexed retrieval chunk files for this page: {retrieval_source_count}

The relevant evidence is already indexed in OpenRAG and pre-filtered to this page's retrieval chunk file set.
Use the OpenSearch Retrieval Tool now.
If the Retrieval Tool returns no hits, you may use the local retrieval inventory below as trusted same-page fallback evidence.
Do not ask for a URL, file upload, or pasted content.
Use only the retrieved page evidence for {source_filename}.
{source_hint_block}

Return valid JSON only with this shape:
{{
  "summary": "2-4 clean sentences of page summary text",
  "key_facts": ["atomic claim 1", "atomic claim 2"]
}}

Rules:
- summary and key_facts must be clean prose only
- do not include citations, markdown, code fences, source filenames, chunk ids, or `(Source: ...)` text anywhere in the JSON values
- key_facts must be short atomic claims that can be verified independently
- do not merge this page with any other page
- do not guess missing details
- preserve medication, procedure, provider, date, callback, clearance, and provider-action details when present
- if a page mixes headers with a clinically or operationally important request, concern, callback, clearance, indication, or procedure detail, summarize that substantive detail instead of collapsing the page to generic header language
- prefer specific indication/procedure/assessment/plan/callback/clearance/provider-action details over facility headers when both are present
- ignore any retrieved result that does not belong to this page artifact or its retrieval chunk files
- if the page is mostly demographic/header material, say that directly
- if both the Retrieval Tool and the local retrieval inventory are empty, say that no supporting sources were found for this page
""".strip()


def build_reduce_prompt(scope: SummaryScope, page_summaries: list[PageMapSummary]) -> str:
    rendered_page_summaries = "\n".join(
        "\n".join(
            [
                f"- {page.pdf_id} p.{page.page}",
                f"  Supported summary: {page.supported_summary or page.summary}",
                *(
                    [
                        "  Supported key facts:",
                        *[f"    - {fact}" for fact in (page.supported_key_facts or page.key_facts)],
                    ]
                    if (page.supported_key_facts or page.key_facts)
                    else []
                ),
            ]
        )
        for page in page_summaries
    )
    rendered_sources = ", ".join(page.source_filename for page in page_summaries)
    return f"""
You are combining verified page-level grounded summaries into a final scoped summary.

Scope title: {scope.title}
Scope objective: {scope.objective}
Indexed source filenames: {rendered_sources}

The relevant evidence is already indexed in OpenRAG and pre-filtered to retrieval chunk files derived from the selected pages.
Use the OpenSearch Retrieval Tool now.
Do not ask for a URL, file upload, or pasted content.

Verified page summaries:
{rendered_page_summaries}

Return valid JSON only with this shape:
{{
  "title": "{scope.title}",
  "summary": "single clean summary paragraph",
  "chronology": ["chronology item 1", "chronology item 2"]
}}

Rules:
- summary and chronology must be clean prose only
- do not include citations, markdown, code fences, source filenames, chunk ids, or `(Source: ...)` text anywhere in the JSON values
- keep chronology correct
- keep procedures separate if the page summaries describe different procedures
- preserve specific supported clinical or operational details when they exist
- if a page is mostly administrative but includes a clinically meaningful request, clearance, callback, or provider action, retain that specific detail in the final summary
- omit unsupported speculation
""".strip()



def build_truth_layer_prompt(
    scope: SummaryScope,
    *,
    pdf_id: str,
    page: int,
    source_filename: str,
    supported_summary: str,
    supported_key_facts: list[str],
    verified_sentences: list[str],
    metadata_hints: dict[str, str] | None = None,
) -> str:
    metadata_hints = metadata_hints or {}
    rendered_summary = supported_summary.strip() or "No supported summary was produced."
    rendered_key_facts = "\n".join(f"- {item}" for item in supported_key_facts) or "- None"
    rendered_verified = "\n".join(f"- {item}" for item in verified_sentences) or "- None"
    metadata_block = "\n".join(
        f"- {label}: {value}"
        for label, value in (
            ("Service date hint", metadata_hints.get("service_date") or ""),
            ("Patient hint", metadata_hints.get("patient_name") or ""),
            ("Page label", metadata_hints.get("label") or ""),
        )
        if value
    ) or "- None"
    return f"""
You are extracting a strict supported fact sheet for a single verified medical/legal page.

Scope title: {scope.title}
Scope objective: {scope.objective}
Current page: {pdf_id} page {page}
Page artifact filename: {source_filename}

Use the verified supported content below as the canonical source of truth for this extraction.
Do not write narrative prose.
Do not include citations, source filenames, markdown, or commentary.
Do not invent or infer missing facts.
Include only positive or abnormal findings; exclude normal findings unless they are the only supported content on the page.
Preserve exact dates, callback numbers, medication doses/frequencies, units, ICD codes, and procedure details when present.
If a supported detail does not fit a named field, place it in residual_supported_facts.

Metadata hints:
{metadata_block}

Verified supported summary:
{rendered_summary}

Verified supported key facts:
{rendered_key_facts}

Verified atomic evidence sentences:
{rendered_verified}

Return valid JSON only with this shape:
{{
  "date_of_service": ["..."],
  "facility": ["..."],
  "provider": ["..."],
  "patient_reference": ["..."],
  "note_type": ["..."],
  "chief_complaint": ["..."],
  "hpi": ["..."],
  "pmh": ["..."],
  "psh": ["..."],
  "social_history": ["..."],
  "allergies": ["..."],
  "medications": ["..."],
  "vitals": ["..."],
  "abnormal_labs": ["..."],
  "diagnoses": ["..."],
  "assessment": ["..."],
  "treatment": ["..."],
  "plan": ["..."],
  "follow_up": ["..."],
  "positive_ros": ["..."],
  "positive_physical_exam": ["..."],
  "residual_supported_facts": ["..."]
}}
""".strip()


def build_presentation_layer_prompt(
    scope: SummaryScope,
    *,
    truth_payload: dict[str, object],
    candidate_sections: list[dict[str, object]],
) -> str:
    rendered_truth = json.dumps(truth_payload, ensure_ascii=True, indent=2)
    rendered_candidates = json.dumps(candidate_sections, ensure_ascii=True, indent=2)
    return f"""
You are a presentation-layer renderer for a grounded medical/legal summarization pipeline.

Scope title: {scope.title}
Scope objective: {scope.objective}

Use only the verified truth-layer JSON and candidate fact inventory below.
Do not retrieve any new evidence.
Do not invent facts.
Do not add citations, markdown, code fences, source filenames, or commentary.
Return valid JSON only.
Every rendered item must include one or more fact_ids drawn from the candidate inventory.
You may reorder and combine candidate facts for readability, but you must not omit clinically or operationally meaningful populated fields.
Skip empty fields.
Use narrative prose, one section per note/date when possible.

Truth layer JSON:
{rendered_truth}

Candidate presentation facts:
{rendered_candidates}

Return valid JSON only with this shape:
{{
  "title": "{scope.title}",
  "narrative": "overall narrative text",
  "sections": [
    {{
      "title": "On 09/12/2018",
      "note_id": "note-001",
      "items": [
        {{"text": "Sentence text.", "fact_ids": ["note-001__intro__01", "note-001__chief_complaint__01"]}}
      ]
    }}
  ]
}}
""".strip()


def build_presentation_editor_prompt(
    scope: SummaryScope,
    *,
    truth_payload: dict[str, object],
    draft_sections: list[dict[str, object]],
) -> str:
    rendered_truth = json.dumps(truth_payload, ensure_ascii=True, indent=2)
    rendered_draft = json.dumps(draft_sections, ensure_ascii=True, indent=2)
    return f"""
You are a presentation-layer editor for a grounded medical/legal summarization pipeline.

Scope title: {scope.title}
Scope objective: {scope.objective}

You are polishing a draft that was already generated from verified facts.
Use only the verified truth-layer JSON and the renderer draft below.
Do not retrieve any new evidence.
Do not invent facts.
Do not omit clinically or operationally meaningful populated fields.
Do not add citations, markdown, code fences, source filenames, commentary, or unsupported conclusions.
Omit normal or negative findings unless they are the only supported content for that note/date.
Render one polished paragraph per note/date with stable chronological ordering when possible.
Every rendered item must include one or more fact_ids already present in the draft.
You may merge or split draft sentences for readability, but you must preserve all supported required details.
Return valid JSON only.

Truth layer JSON:
{rendered_truth}

Renderer draft sections:
{rendered_draft}

Return valid JSON only with this shape:
{{
  "title": "{scope.title}",
  "narrative": "overall narrative text",
  "sections": [
    {{
      "title": "On 09/12/2018",
      "note_id": "note-001",
      "items": [
        {{"text": "Sentence text.", "fact_ids": ["note-001__intro__01", "note-001__chief_complaint__01"]}}
      ]
    }}
  ]
}}
""".strip()




def build_page_local_fallback_prompt(
    scope: SummaryScope,
    *,
    pdf_id: str,
    page: int,
    source_filename: str,
    evidence_excerpts: list[str],
) -> str:
    rendered_excerpts = "\n\n".join(
        f"Excerpt {index}:\n{excerpt}"
        for index, excerpt in enumerate(evidence_excerpts, start=1)
        if str(excerpt).strip()
    ) or "No page-local excerpts were available."
    return f"""
You are summarizing a single page from an already materialized medical/legal document.

Scope title: {scope.title}
Scope objective: {scope.objective}
Current page: {pdf_id} page {page}
Page artifact filename: {source_filename}

Use only the page-local evidence excerpts below.
They were derived from this exact page during materialization and are allowed evidence for this page summary.
Do not retrieve new evidence.
Do not say that supporting sources are missing if the excerpt block below contains page evidence.

Page-local evidence excerpts:
{rendered_excerpts}

Return valid JSON only with this shape:
{{
  "summary": "2-4 clean sentences of page summary text",
  "key_facts": ["atomic claim 1", "atomic claim 2"]
}}

Rules:
- summary and key_facts must be clean prose only
- do not include citations, markdown, code fences, source filenames, chunk ids, or `(Source: ...)` text anywhere in the JSON values
- key_facts must be short atomic claims that can be verified independently
- do not merge this page with any other page
- do not guess missing details
- preserve medication, procedure, provider, date, callback, clearance, and provider-action details when present
- prefer specific clinical or operational details over generic header language
- if the excerpt block is empty, say that no supporting sources were found for this page
""".strip()
