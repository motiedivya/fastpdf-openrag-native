"""
Public API v1 Search endpoint.

Provides semantic search functionality.
Uses API key authentication.
Adds deterministic reranking over a larger hybrid candidate pool so public
/api/v1/search behaves more like the backend-reranked chat path.
"""
import math
import os
import re
from collections import Counter
from typing import Any, Dict, Optional

from fastapi import Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from dependencies import get_api_key_user_async, get_search_service
from session_manager import User
from utils.logging_config import get_logger
from utils.opensearch_utils import DISK_SPACE_ERROR_MESSAGE, OpenSearchDiskSpaceError

logger = get_logger(__name__)

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


class SearchV1Body(BaseModel):
    query: str
    filters: Optional[Dict[str, Any]] = None
    limit: int = 10
    score_threshold: float = 0


def _rerank_enabled() -> bool:
    value = (
        os.getenv("FASTPDF_OPENRAG_BACKEND_SEARCH_RERANK_ENABLED", "true") or ""
    ).strip().lower()
    return value not in {"0", "false", "no", "off", ""}


def _candidate_limit(requested_limit: int) -> int:
    raw_value = os.getenv("FASTPDF_OPENRAG_BACKEND_SEARCH_RERANK_CANDIDATE_LIMIT")
    try:
        configured = int(raw_value) if raw_value else 24
    except (TypeError, ValueError):
        configured = 24
    return max(1, max(requested_limit, configured))


def _normalize_text(value: str) -> str:
    clean = (value or "").lower()
    clean = clean.replace("\u00a0", " ")
    clean = re.sub(r"\s+", " ", clean)
    return clean.strip()


def _tokenize(value: str) -> list[str]:
    return [
        token
        for token in TOKEN_RE.findall(_normalize_text(value))
        if token not in STOPWORDS and len(token) > 1
    ]


def _attach_rank_metadata(
    results: list[dict[str, Any]],
    *,
    requested_limit: int,
) -> list[dict[str, Any]]:
    final_results = [dict(row) for row in results[:requested_limit]]
    for index, row in enumerate(final_results, start=1):
        base_score = float(row.get("base_score", row.get("score") or 0.0) or 0.0)
        rerank_score = float(row.get("rerank_score", row.get("score") or 0.0) or 0.0)
        row["base_score"] = base_score
        row["rerank_score"] = rerank_score
        row["score"] = rerank_score
        row["retrieval_rank"] = index
    return final_results


def _rerank_results(
    *,
    query: str,
    results: list[dict[str, Any]],
    requested_limit: int,
) -> list[dict[str, Any]]:
    if not _rerank_enabled() or len(results) < 2:
        return _attach_rank_metadata(results, requested_limit=requested_limit)

    query_tokens = _tokenize(query)
    if not query_tokens:
        return _attach_rank_metadata(results, requested_limit=requested_limit)

    base_scores = [float(row.get("score") or 0.0) for row in results]
    max_base_score = max(base_scores) if base_scores else 0.0
    normalized_texts = [_normalize_text(str(row.get("text") or "")) for row in results]
    token_sets = [set(_tokenize(text)) for text in normalized_texts]
    document_frequency = Counter(
        token
        for token in set(query_tokens)
        for token_set in token_sets
        if token in token_set
    )
    total_docs = max(1, len(results))
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

    reranked: list[dict[str, Any]] = []
    for original_index, (row, base_score, normalized_text, token_set) in enumerate(
        zip(results, base_scores, normalized_texts, token_sets, strict=False),
        start=1,
    ):
        matched_tokens = [token for token in set(query_tokens) if token in token_set]
        idf_overlap = sum(token_weights[token] for token in matched_tokens) / total_token_weight
        coverage = len(matched_tokens) / max(1, len(set(query_tokens)))
        phrase_match = 1.0 if normalized_query and normalized_query in normalized_text else 0.0
        bigram_match = (
            1.0
            if query_bigrams and any(query_bigram in normalized_text for query_bigram in query_bigrams)
            else 0.0
        )
        lead_overlap = (
            1.0
            if matched_tokens and any(token in normalized_text[:240] for token in matched_tokens)
            else 0.0
        )
        normalized_base = (base_score / max_base_score) if max_base_score > 0 else 0.0
        rerank_score = (
            0.35 * normalized_base
            + 0.25 * idf_overlap
            + 0.15 * coverage
            + 0.20 * max(phrase_match, bigram_match)
            + 0.05 * lead_overlap
        )
        enriched = dict(row)
        enriched["base_score"] = base_score
        enriched["rerank_score"] = rerank_score
        enriched["_original_index"] = original_index
        reranked.append(enriched)

    reranked.sort(
        key=lambda row: (
            float(row.get("rerank_score") or float("-inf")),
            float(row.get("base_score") or float("-inf")),
            -int(row.get("_original_index") or 0),
        ),
        reverse=True,
    )
    for row in reranked:
        row.pop("_original_index", None)
    return _attach_rank_metadata(reranked, requested_limit=requested_limit)


async def search_endpoint(
    body: SearchV1Body,
    search_service=Depends(get_search_service),
    user: User = Depends(get_api_key_user_async),
):
    """Perform semantic search on documents. POST /v1/search"""
    query = body.query.strip()
    if not query:
        return JSONResponse({"error": "Query is required"}, status_code=400)

    requested_limit = max(1, int(body.limit))
    rerank_active = _rerank_enabled() and query != "*"
    effective_limit = _candidate_limit(requested_limit) if rerank_active else requested_limit

    logger.debug(
        "Public API search request",
        user_id=user.user_id,
        query=query,
        filters=body.filters,
        limit=requested_limit,
        effective_limit=effective_limit,
        score_threshold=body.score_threshold,
        rerank_active=rerank_active,
    )

    try:
        result = await search_service.search(
            query,
            user_id=user.user_id,
            jwt_token=None,
            filters=body.filters or {},
            limit=effective_limit,
            score_threshold=body.score_threshold,
        )

        raw_results = [
            {
                "filename": item.get("filename"),
                "text": item.get("text"),
                "score": item.get("score"),
                "page": item.get("page"),
                "mimetype": item.get("mimetype"),
                "base_score": item.get("base_score"),
                "rerank_score": item.get("rerank_score"),
                "retrieval_rank": item.get("retrieval_rank"),
            }
            for item in result.get("results", [])
        ]
        results = (
            _rerank_results(query=query, results=raw_results, requested_limit=requested_limit)
            if rerank_active
            else _attach_rank_metadata(raw_results, requested_limit=requested_limit)
        )

        return JSONResponse({"results": results})

    except OpenSearchDiskSpaceError as e:
        logger.error("Search blocked by disk space constraint", error=str(e), user_id=user.user_id)
        return JSONResponse({"error": DISK_SPACE_ERROR_MESSAGE}, status_code=507)
    except Exception as e:
        error_msg = str(e)
        logger.error("Search failed", error=error_msg, user_id=user.user_id)
        if "AuthenticationException" in error_msg or "access denied" in error_msg.lower():
            return JSONResponse({"error": error_msg}, status_code=403)
        return JSONResponse({"error": error_msg}, status_code=500)
