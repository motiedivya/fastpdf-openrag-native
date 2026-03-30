from __future__ import annotations

from fastpdf_openrag_native.models import SummaryScope
from fastpdf_openrag_native.prompts import build_page_map_prompt


def test_page_map_prompt_requires_indexed_retrieval() -> None:
    scope = SummaryScope.model_validate(
        {
            "scope_id": "scope-1",
            "title": "Medical Summary",
            "objective": "Summarize the selected page.",
            "page_refs": [{"pdf_id": "merged_notes.pdf", "page": 1}],
        }
    )

    prompt = build_page_map_prompt(
        scope,
        pdf_id="merged_notes.pdf",
        page=1,
        source_filename="merged_notes__p0001.html",
        retrieval_source_count=3,
    )

    assert "Page artifact filename: merged_notes__p0001.html" in prompt
    assert "Indexed retrieval chunk files for this page: 3" in prompt
    assert "Use the OpenSearch Retrieval Tool now." in prompt
    assert "Do not ask for a URL, file upload, or pasted content." in prompt
    assert "do not include citations, markdown, code fences, source filenames" in prompt.lower()
    assert "key_facts must be short atomic claims" in prompt
