from __future__ import annotations

from .models import PageMapSummary, SummaryScope


def build_page_map_prompt(
    scope: SummaryScope,
    *,
    pdf_id: str,
    page: int,
    source_filename: str,
    retrieval_source_count: int = 1,
) -> str:
    return f"""
You are summarizing a single page from an already indexed medical/legal document.

Scope title: {scope.title}
Scope objective: {scope.objective}
Current page: {pdf_id} page {page}
Page artifact filename: {source_filename}
Indexed retrieval chunk files for this page: {retrieval_source_count}

The relevant evidence is already indexed in OpenRAG and pre-filtered to this page's retrieval chunk file set.
Use the OpenSearch Retrieval Tool now.
Do not ask for a URL, file upload, or pasted content.
Use only the retrieved page evidence for {source_filename}.

Return valid JSON with this shape:
{{
  "summary": "2-4 sentence page summary",
  "key_facts": ["fact 1", "fact 2"]
}}

Rules:
- do not merge this page with any other page
- do not guess missing details
- preserve medication, procedure, provider, and date details when present
- if the page is mostly demographic/header material, say that directly
- if retrieval still fails, say that no supporting sources were found for this page
""".strip()


def build_reduce_prompt(scope: SummaryScope, page_summaries: list[PageMapSummary]) -> str:
    rendered_page_summaries = "\n".join(
        f"- {page.pdf_id} p.{page.page}: {page.summary}"
        for page in page_summaries
    )
    rendered_sources = ", ".join(page.source_filename for page in page_summaries)
    return f"""
You are combining page-level grounded summaries into a final scoped summary.

Scope title: {scope.title}
Scope objective: {scope.objective}
Indexed source filenames: {rendered_sources}

The relevant evidence is already indexed in OpenRAG and pre-filtered to retrieval chunk files derived from the selected pages.
Use the OpenSearch Retrieval Tool now.
Do not ask for a URL, file upload, or pasted content.

Page summaries:
{rendered_page_summaries}

Return valid JSON with this shape:
{{
  "title": "{scope.title}",
  "summary": "single summary paragraph",
  "chronology": ["chronology item 1", "chronology item 2"]
}}

Rules:
- use only the page summaries that were provided
- keep chronology correct
- keep procedures separate if the page summaries describe different procedures
- omit unsupported speculation
""".strip()
