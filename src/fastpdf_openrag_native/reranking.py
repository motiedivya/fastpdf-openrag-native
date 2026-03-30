from __future__ import annotations

import math
import re
from collections import Counter

from .models import EvidenceHit

RERANKER_TYPE = "deterministic_hybrid_v1"
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


def _normalize_text(value: str) -> str:
    clean = (value or "").lower()
    clean = clean.replace("\u00a0", " ")
    clean = re.sub(r"\s+", " ", clean)
    return clean.strip()


def _tokenize(value: str) -> list[str]:
    return [token for token in TOKEN_RE.findall(_normalize_text(value)) if token not in STOPWORDS and len(token) > 1]


def rerank_hits(query: str, hits: list[EvidenceHit]) -> list[EvidenceHit]:
    if not hits:
        return []

    query_tokens = _tokenize(query)
    base_scores = [
        float(hit.base_score if hit.base_score is not None else hit.score or 0.0)
        for hit in hits
    ]
    max_base_score = max(base_scores) if base_scores else 0.0

    normalized_texts = [_normalize_text(hit.text) for hit in hits]
    token_sets = [set(_tokenize(text)) for text in normalized_texts]
    document_frequency = Counter(
        token
        for token in set(query_tokens)
        for token_set in token_sets
        if token in token_set
    )

    total_docs = max(1, len(hits))
    token_weights = {
        token: 1.0 + math.log1p(total_docs / max(1, document_frequency.get(token, 1)))
        for token in set(query_tokens)
    }
    total_token_weight = sum(token_weights.values()) or 1.0
    normalized_query = _normalize_text(query)
    query_bigrams = [
        " ".join(query_tokens[index : index + 2])
        for index in range(max(0, len(query_tokens) - 1))
    ]

    reranked: list[EvidenceHit] = []
    for hit, base_score, normalized_text, token_set in zip(
        hits,
        base_scores,
        normalized_texts,
        token_sets,
        strict=False,
    ):
        matched_tokens = [token for token in set(query_tokens) if token in token_set]
        idf_overlap = sum(token_weights[token] for token in matched_tokens) / total_token_weight
        coverage = len(matched_tokens) / max(1, len(set(query_tokens))) if query_tokens else 0.0
        phrase_match = 1.0 if normalized_query and normalized_query in normalized_text else 0.0
        bigram_match = (
            1.0
            if query_bigrams and any(query_bigram in normalized_text for query_bigram in query_bigrams)
            else 0.0
        )
        lead_overlap = 1.0 if matched_tokens and any(token in normalized_text[:240] for token in matched_tokens) else 0.0
        normalized_base = (base_score / max_base_score) if max_base_score > 0 else 0.0
        rerank_score = (
            0.35 * normalized_base
            + 0.25 * idf_overlap
            + 0.15 * coverage
            + 0.20 * max(phrase_match, bigram_match)
            + 0.05 * lead_overlap
        )
        reranked.append(
            hit.model_copy(
                update={
                    "score": rerank_score,
                    "base_score": base_score,
                    "rerank_score": rerank_score,
                }
            )
        )

    reranked.sort(
        key=lambda hit: (
            hit.rerank_score if hit.rerank_score is not None else float("-inf"),
            hit.base_score if hit.base_score is not None else float("-inf"),
            hit.filename,
        ),
        reverse=True,
    )
    return [
        hit.model_copy(update={"retrieval_rank": index})
        for index, hit in enumerate(reranked, start=1)
    ]


def select_top_source_filenames(
    hits: list[EvidenceHit],
    *,
    top_k: int,
    source_groups: list[list[str]] | None = None,
) -> list[str]:
    if top_k <= 0:
        return []

    selected: list[str] = []
    seen: set[str] = set()
    for group in source_groups or []:
        for hit in hits:
            if hit.filename not in group or hit.filename in seen:
                continue
            selected.append(hit.filename)
            seen.add(hit.filename)
            break
        if len(selected) >= top_k:
            return selected[:top_k]

    for hit in hits:
        if hit.filename in seen:
            continue
        selected.append(hit.filename)
        seen.add(hit.filename)
        if len(selected) >= top_k:
            break
    return selected
