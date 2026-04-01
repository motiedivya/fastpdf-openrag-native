from __future__ import annotations

from types import SimpleNamespace

from fastpdf_openrag_native.chunking import StructuredBlock, blocks_from_ocr_paragraphs, build_structured_chunks
from fastpdf_openrag_native.layered_output import (
    _extract_string_list,
    _filter_presentable_facts,
    _note_from_payload,
    _split_sentences,
)
from fastpdf_openrag_native.models import EvidenceHit, PageMapSummary, SupportedFact, VerifiedSentence


def test_split_sentences_protects_address_abbreviations() -> None:
    text = "Clinic address is 800 W. 47th St., Ste. 100 Kansas City, MO 64112. Chief complaint was left knee pain."

    assert _split_sentences(text) == [
        "Clinic address is 800 W. 47th St., Ste. 100 Kansas City, MO 64112.",
        "Chief complaint was left knee pain.",
    ]


def test_extract_string_list_drops_admin_fragments_from_residual_facts() -> None:
    values = [
        "Clinic address is 800 W. 47th St., Ste. 100 Kansas City, MO 64112",
        "Phone: 816-216-7054",
        "47th St., Ste.",
        "Pain worsened with bending and improved with injections and therapy",
    ]

    assert _extract_string_list(values, field_name="residual_supported_facts") == [
        "Pain worsened with bending and improved with injections and therapy"
    ]


def test_filter_presentable_facts_strips_normal_subclauses() -> None:
    facts = [
        SupportedFact(value="mild to moderate knee effusion with medial tenderness and sensation intact to light touch"),
        SupportedFact(value="appropriate mood and affect"),
        SupportedFact(value="reflexes 2+ bilaterally"),
    ]

    filtered = _filter_presentable_facts("positive_physical_exam", facts)

    assert [fact.value for fact in filtered] == [
        "mild to moderate knee effusion with medial tenderness"
    ]


def test_blocks_from_ocr_paragraphs_splits_embedded_sections_and_strips_contact_footer() -> None:
    paragraphs = [
        {
            "text": (
                "Spine and Joint Centers of Missouri Provider: Everett Wilkinson, DO Date of Service: 8/16/2024 "
                "Phone: 816-216-7054 Fax: 816-216-6010 800 W. 47th St., Ste. 100 Kansas City, MO 64112 "
                "HISTORY OF PRESENT ILLNESS: She reported frequent sharp pain that worsened with bending and improved with injections and therapy."
            ),
            "block_index": 0,
            "paragraph_index": 0,
            "page_paragraph_index": 0,
            "bbox": {"x": 1, "y": 1, "w": 10, "h": 10},
        }
    ]

    blocks = blocks_from_ocr_paragraphs(paragraphs)
    texts = [block.text for block in blocks]

    assert any("Provider: Everett Wilkinson, DO" in text for text in texts)
    assert any(text.startswith("HISTORY OF PRESENT ILLNESS:") for text in texts)
    assert all("Phone:" not in text for text in texts)
    assert all("47th St." not in text for text in texts)

    chunks = build_structured_chunks(blocks, target_chars=180, overlap_blocks=0)
    assert any("HISTORY OF PRESENT ILLNESS" in chunk.text for chunk in chunks)
    assert all("Phone:" not in chunk.text for chunk in chunks)
    assert all("47th St." not in chunk.text for chunk in chunks)


def test_blocks_from_ocr_paragraphs_drops_isolated_one_word_fragments_and_merges_short_headings() -> None:
    paragraphs = [
        {
            "text": "with\nPlan:\nPhysical therapy was recommended.\nappropriate",
            "block_index": 0,
            "paragraph_index": 0,
            "page_paragraph_index": 0,
            "bbox": {"x": 1, "y": 1, "w": 10, "h": 10},
        }
    ]

    blocks = blocks_from_ocr_paragraphs(paragraphs)

    assert [block.text for block in blocks] == ["Plan: Physical therapy was recommended."]


def test_build_structured_chunks_skips_trailing_heading_only_sections() -> None:
    chunks = build_structured_chunks(
        [
            StructuredBlock(text="HISTORY OF PRESENT ILLNESS"),
            StructuredBlock(text="Pain improved with injections and therapy."),
            StructuredBlock(text="ASSESSMENT"),
        ],
        target_chars=160,
        overlap_blocks=0,
    )

    assert len(chunks) == 1
    assert chunks[0].section_title == "HISTORY OF PRESENT ILLNESS"
    assert chunks[0].text == "Pain improved with injections and therapy."


def test_note_from_payload_filters_admin_and_normal_leakage() -> None:
    page = SimpleNamespace(
        pdf_id="pdf_0001",
        page=1,
        service_date="8/16/2024",
        patient_name="Ms. Jorgensen",
        label="Visit Note",
        retrieval_sources=lambda: ["chunk-1.md"],
    )
    hit = EvidenceHit(filename="chunk-1.md", text="support", score=0.9)
    page_summary = PageMapSummary(
        pdf_id="pdf_0001",
        page=1,
        source_filename="pdf_0001__p0001.md",
        summary="Clinic address is 800 W. 47th St., Ste. 100. Pain worsened with bending and improved with injections and therapy.",
        key_facts=["Phone: 816-216-7054", "Pain worsened with bending and improved with injections and therapy"],
        raw_response="{}",
        retrieved_sources=[hit],
        verified_sentences=[
            VerifiedSentence(
                sentence="Pain worsened with bending and improved with injections and therapy.",
                supported=True,
                evidence=[hit],
            )
        ],
        verified_key_facts=[],
        supported_summary="Pain worsened with bending and improved with injections and therapy.",
        supported_key_facts=["Pain worsened with bending and improved with injections and therapy"],
        unsupported_sentences=[],
        unsupported_key_facts=[],
        passed_verification=True,
    )

    note = _note_from_payload(
        {
            "date_of_service": ["8/16/2024"],
            "facility": ["Spine and Joint Centers of Missouri"],
            "provider": ["Everett Wilkinson, DO"],
            "chief_complaint": ["frequent sharp pain rated 8 out of 10"],
            "positive_physical_exam": [
                "mild to moderate knee effusion with medial tenderness and sensation intact to light touch",
                "appropriate mood and affect",
            ],
            "residual_supported_facts": [
                "Clinic address is 800 W. 47th St., Ste. 100 Kansas City, MO 64112",
                "Pain worsened with bending and improved with injections and therapy",
                "47th St., Ste.",
            ],
        },
        page=page,
        page_summary=page_summary,
        verified_rows=page_summary.verified_sentences,
    )

    assert note.residual_supported_facts[0].value == "Pain worsened with bending and improved with injections and therapy"
    assert all("47th St." not in fact.value for fact in note.residual_supported_facts)
    assert all("Phone:" not in fact.value for fact in note.residual_supported_facts)
    assert note.positive_physical_exam[0].value == "mild to moderate knee effusion with medial tenderness"
