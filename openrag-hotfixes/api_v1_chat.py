"""
Public API v1 Chat endpoint.

Provides chat functionality with streaming support and conversation history.
Uses API key authentication. Routes through Langflow (chat_service.langflow_chat).
"""
import asyncio
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter
from difflib import SequenceMatcher
from typing import Optional, Any, Dict

from fastapi import Depends
from pydantic import BaseModel
from fastapi.responses import JSONResponse, StreamingResponse
from utils.logging_config import get_logger
from auth_context import set_search_filters, set_search_limit, set_score_threshold, set_auth_context
from dependencies import get_chat_service, get_search_service, get_session_manager, get_api_key_user_async
from session_manager import User

logger = get_logger(__name__)


_DEBUG_ROOT = Path(os.getenv("FASTPDF_OPENRAG_DEBUG_ROOT", "/app/openrag-documents/debug"))
_REDACTED_HEADER_KEYS = {"authorization", "x-api-key", "api-key"}


def _json_safe(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        try:
            value = value.model_dump(mode="json")
        except TypeError:
            value = value.model_dump()
        except Exception:
            value = str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _append_debug_record(stream_name: str, payload: dict[str, Any]) -> None:
    try:
        _DEBUG_ROOT.mkdir(parents=True, exist_ok=True)
        file_path = _DEBUG_ROOT / f"{stream_name}.jsonl"
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            **_json_safe(payload),
        }
        with file_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed writing %s debug record", stream_name=stream_name, error=str(exc))


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
HIGH_VALUE_TERMS = (
    "subjective",
    "assessment",
    "plan",
    "care plan",
    "chief complaint",
    "history of present illness",
    "procedure",
    "operation performed",
    "indication for procedure",
    "diagnosis",
    "impression",
    "follow-up",
)
LOW_VALUE_TERMS = (
    "fax",
    "phone",
    "email",
    "address",
    "insurance",
    "member id",
    "guarantor",
    "dob",
    "pid",
    "printed by",
    "confidential",
    "page ",
)


def _rerank_enabled() -> bool:
    value = (os.getenv("FASTPDF_OPENRAG_BACKEND_SEARCH_RERANK_ENABLED", "true") or "").strip().lower()
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
    clean = clean.replace(" ", " ")
    clean = re.sub(r"\s+", " ", clean)
    return clean.strip()


def _tokenize(value: str) -> list[str]:
    return [token for token in TOKEN_RE.findall(_normalize_text(value)) if token not in STOPWORDS and len(token) > 1]


def _role_signal(value: str) -> tuple[float, float]:
    normalized = _normalize_text(value)
    if not normalized:
        return 0.0, 0.0
    high_value = 1.0 if any(term in normalized for term in HIGH_VALUE_TERMS) else 0.0
    low_value = 1.0 if any(term in normalized for term in LOW_VALUE_TERMS) else 0.0
    if high_value and low_value:
        low_value *= 0.35
    return high_value, low_value


def _attach_rank_metadata(results: list[dict[str, Any]], *, requested_limit: int) -> list[dict[str, Any]]:
    final_results = [dict(row) for row in results[:requested_limit]]
    for index, row in enumerate(final_results, start=1):
        base_score = float(row.get("base_score", row.get("score") or 0.0) or 0.0)
        rerank_score = float(row.get("rerank_score", row.get("score") or 0.0) or 0.0)
        row["base_score"] = base_score
        row["rerank_score"] = rerank_score
        row["score"] = rerank_score
        row["retrieval_rank"] = row.get("retrieval_rank") or index
    return final_results


def _rerank_results(*, query: str, results: list[dict[str, Any]], requested_limit: int) -> list[dict[str, Any]]:
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
        lead_overlap = 1.0 if matched_tokens and any(token in normalized_text[:240] for token in matched_tokens) else 0.0
        normalized_base = (base_score / max_base_score) if max_base_score > 0 else 0.0
        high_value_signal, low_value_signal = _role_signal(normalized_text)
        rerank_score = (
            0.31 * normalized_base
            + 0.25 * idf_overlap
            + 0.15 * coverage
            + 0.18 * max(phrase_match, bigram_match)
            + 0.05 * lead_overlap
            + 0.10 * high_value_signal
            - 0.08 * low_value_signal
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

class ChatV1Body(BaseModel):
    message: str
    stream: bool = False
    chat_id: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None
    limit: int = 10
    score_threshold: float = 0
    filter_id: Optional[str] = None


def _coerce_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _allowed_source_names(filters: Optional[Dict[str, Any]]) -> set[str]:
    data_sources = filters.get("data_sources") if isinstance(filters, dict) else None
    if not isinstance(data_sources, list):
        return set()
    return {str(item).strip() for item in data_sources if str(item).strip()}


def _filter_allowed_sources(sources: list[dict], filters: Optional[Dict[str, Any]]) -> list[dict]:
    allowed_sources = _allowed_source_names(filters)
    if not allowed_sources:
        return list(sources)
    return [
        source
        for source in sources
        if str(source.get("filename") or "").strip() in allowed_sources
    ]


def _extract_sources(item: Any) -> list[dict]:
    """Extract sources from a retrieval tool call item."""
    if not isinstance(item, dict):
        return []

    sources = []
    for result in _coerce_list(item.get("results")):
        if not isinstance(result, dict):
            continue
        payload = result.get("data") if isinstance(result.get("data"), dict) else result
        if not isinstance(payload, dict):
            continue
        if "text" in payload:
            sources.append({
                "filename": payload.get("filename", ""),
                "text": payload.get("text", ""),
                "score": payload.get("score", 0),
                "page": payload.get("page"),
                "mimetype": payload.get("mimetype"),
                "base_score": payload.get("base_score"),
                "rerank_score": payload.get("rerank_score"),
                "retrieval_rank": payload.get("retrieval_rank"),
            })
    return sources


async def _recover_history_context(user_id: str, chat_id: str) -> tuple[list[dict], list[str]]:
    """Fetch the latest Langflow session history and recover sources plus tool queries."""
    from services.langflow_history_service import langflow_history_service

    seen: set[tuple[str, object, str]] = set()
    sources: list[dict] = []
    queries: list[str] = []
    seen_queries: set[str] = set()

    for _attempt in range(5):
        messages = await langflow_history_service.get_session_messages(user_id, chat_id)
        for message in reversed(_coerce_list(messages)):
            if not isinstance(message, dict):
                continue
            if message.get("role") != "assistant":
                continue
            for chunk in _coerce_list(message.get("chunks")):
                if not isinstance(chunk, dict):
                    continue
                item = chunk.get("item") or {}
                if not isinstance(item, dict):
                    continue
                inputs = item.get("inputs", {})
                query = inputs.get("search_query") if isinstance(inputs, dict) else None
                if query and query not in seen_queries:
                    seen_queries.add(query)
                    queries.append(query)
                extracted = _extract_sources(item)
                for source in extracted:
                    dedupe_key = (
                        source.get("filename", ""),
                        source.get("page"),
                        source.get("text", ""),
                    )
                    if dedupe_key in seen:
                        continue
                    seen.add(dedupe_key)
                    sources.append(source)
            if sources:
                return sources, queries
        await asyncio.sleep(0.5)

    return sources, queries


async def _enrich_sources_from_history(user_id: str, chat_id: str) -> list[dict]:
    """Fetch the latest Langflow session history and recover tool sources."""
    sources, _queries = await _recover_history_context(user_id, chat_id)
    return sources


def _best_candidate_for_source(source: dict, candidates: list[dict]) -> Optional[dict]:
    """Join a history-derived source with a freshly scored search result."""
    if not candidates:
        return None

    same_filename = [
        candidate for candidate in candidates
        if candidate.get("filename") == source.get("filename")
    ]
    if not same_filename:
        return None

    source_text = source.get("text", "")
    source_page = source.get("page")

    for candidate in same_filename:
        if (
            candidate.get("page") == source_page
            and candidate.get("text") == source_text
            and candidate.get("score") is not None
        ):
            return candidate

    best_candidate = None
    best_similarity = -1.0
    for candidate in same_filename:
        candidate_text = candidate.get("text", "")
        if not source_text or not candidate_text:
            similarity = 0.0
        elif source_text == candidate_text:
            similarity = 1.0
        elif source_text in candidate_text or candidate_text in source_text:
            shorter = min(len(source_text), len(candidate_text))
            longer = max(len(source_text), len(candidate_text))
            similarity = shorter / longer if longer else 0.0
        else:
            similarity = SequenceMatcher(
                None,
                source_text[:1200],
                candidate_text[:1200],
            ).ratio()

        if similarity > best_similarity:
            best_similarity = similarity
            best_candidate = candidate

    return best_candidate if best_candidate and best_candidate.get("score") is not None else None


def _sources_need_backfill(sources: list[dict]) -> bool:
    for source in sources:
        if not isinstance(source, dict):
            return True
        if source.get("score") in (None, 0):
            return True
        if source.get("base_score") is None or source.get("rerank_score") is None or source.get("retrieval_rank") is None:
            return True
    return False


async def _backfill_source_scores(
    *,
    sources: list[dict],
    tool_queries: list[str],
    search_service,
    user_id: str,
    jwt_token: str,
    filters: Optional[Dict[str, Any]],
    limit: int,
) -> list[dict]:
    """Run fresh scored searches and attach scores to history-derived sources."""
    if not sources or not tool_queries or not _sources_need_backfill(sources):
        return sources

    query_cache: dict[str, list[dict]] = {}
    for query in tool_queries:
        if query in query_cache:
            continue
        try:
            requested_limit = max(limit, len(sources), 10)
            search_response = await search_service.search(
                query,
                user_id=user_id,
                jwt_token=jwt_token,
                filters=filters or {},
                limit=_candidate_limit(requested_limit),
                score_threshold=0,
            )
            raw_results = _filter_allowed_sources(_coerce_list(search_response.get("results", [])), filters)
            query_cache[query] = _rerank_results(
                query=query,
                results=raw_results,
                requested_limit=requested_limit,
            )
        except Exception as exc:
            logger.warning(
                "Failed to backfill source scores",
                query=query,
                error=str(exc),
            )
            query_cache[query] = []

    for source in sources:
        best_candidate = None
        for query in tool_queries:
            candidate = _best_candidate_for_source(source, query_cache.get(query, []))
            if candidate is None:
                continue
            if best_candidate is None or float(candidate.get("score") or 0.0) > float(best_candidate.get("score") or 0.0):
                best_candidate = candidate
        if best_candidate is not None:
            source["score"] = best_candidate.get("score")
            source["base_score"] = best_candidate.get("base_score")
            source["rerank_score"] = best_candidate.get("rerank_score")
            source["retrieval_rank"] = best_candidate.get("retrieval_rank")

    return sources


async def _transform_stream_to_sse(raw_stream, chat_id_container: dict):
    """Transform raw Langflow streaming format to clean SSE events for v1 API."""
    full_text = ""
    chat_id = None

    async for chunk in raw_stream:
        try:
            if isinstance(chunk, bytes):
                chunk_str = chunk.decode("utf-8").strip()
            else:
                chunk_str = str(chunk).strip()

            if not chunk_str:
                continue

            chunk_data = json.loads(chunk_str)
            delta_text = ""

            if "delta" in chunk_data:
                delta = chunk_data["delta"]
                if isinstance(delta, dict):
                    delta_text = delta.get("content", "") or delta.get("text", "")
                elif isinstance(delta, str):
                    delta_text = delta

            if not delta_text and chunk_data.get("output_text"):
                delta_text = chunk_data["output_text"]
            if not delta_text and chunk_data.get("text"):
                delta_text = chunk_data["text"]
            if not delta_text and chunk_data.get("content"):
                delta_text = chunk_data["content"]

            if delta_text:
                full_text += delta_text
                yield f"data: {json.dumps({'type': 'content', 'delta': delta_text})}\n\n"

            # Emit sources from retrieval tool calls
            item = chunk_data.get("item") or {}
            if not isinstance(item, dict):
                item = {}
            if item.get("type") in ("retrieval_call", "tool_call") and item.get("results"):
                sources = _extract_sources(item)
                if sources:
                    yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

            if not chat_id:
                chat_id = chunk_data.get("id") or chunk_data.get("response_id")

        except json.JSONDecodeError:
            if chunk_str:
                yield f"data: {json.dumps({'type': 'content', 'delta': chunk_str})}\n\n"
                full_text += chunk_str
        except Exception as e:
            logger.warning("Error processing stream chunk", error=str(e))

    yield f"data: {json.dumps({'type': 'done', 'chat_id': chat_id})}\n\n"
    chat_id_container["chat_id"] = chat_id


async def chat_create_endpoint(
    body: ChatV1Body,
    chat_service=Depends(get_chat_service),
    search_service=Depends(get_search_service),
    session_manager=Depends(get_session_manager),
    user: User = Depends(get_api_key_user_async),
):
    """Send a chat message via Langflow. POST /v1/chat"""
    message = body.message.strip()
    if not message:
        return JSONResponse({"error": "Message is required"}, status_code=400)

    user_id = user.user_id
    jwt_token = user.jwt_token

    _append_debug_record(
        "api_v1_chat_requests",
        {
            "user_id": user_id,
            "message": message,
            "stream": body.stream,
            "chat_id": body.chat_id,
            "filter_id": body.filter_id,
            "filters": body.filters,
            "limit": body.limit,
            "score_threshold": body.score_threshold,
        },
    )

    if body.filters:
        set_search_filters(body.filters)
    set_search_limit(body.limit)
    set_score_threshold(body.score_threshold)
    set_auth_context(user_id, jwt_token)

    if body.stream:
        raw_stream = await chat_service.langflow_chat(
            prompt=message,
            user_id=user_id,
            jwt_token=jwt_token,
            previous_response_id=body.chat_id,
            stream=True,
            filter_id=body.filter_id,
        )
        chat_id_container = {}
        return StreamingResponse(
            _transform_stream_to_sse(raw_stream, chat_id_container),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )
    else:
        result = await chat_service.langflow_chat(
            prompt=message,
            user_id=user_id,
            jwt_token=jwt_token,
            previous_response_id=body.chat_id,
            stream=False,
            filter_id=body.filter_id,
        )
        sources = _filter_allowed_sources(_coerce_list(result.get("sources")), body.filters)
        chat_id = result.get("response_id")
        tool_queries: list[str] = []
        if not sources and chat_id:
            sources, tool_queries = await _recover_history_context(user_id, chat_id)
            sources = _filter_allowed_sources(sources, body.filters)
        if sources and chat_id and _sources_need_backfill(sources):
            if not tool_queries:
                _history_sources, tool_queries = await _recover_history_context(user_id, chat_id)
            if tool_queries:
                sources = await _backfill_source_scores(
                    sources=sources,
                    tool_queries=tool_queries,
                    search_service=search_service,
                    user_id=user_id,
                    jwt_token=jwt_token,
                    filters=body.filters,
                    limit=body.limit,
                )
                sources = _filter_allowed_sources(sources, body.filters)
        response_payload = {
            "response": result.get("response", ""),
            "chat_id": chat_id,
            "sources": sources,
        }
        _append_debug_record(
            "api_v1_chat_responses",
            {
                "user_id": user_id,
                "chat_id": chat_id,
                "filter_id": body.filter_id,
                "filters": body.filters,
                "limit": body.limit,
                "score_threshold": body.score_threshold,
                "tool_queries": tool_queries,
                "source_count": len(sources),
                "response": response_payload,
            },
        )
        return JSONResponse(response_payload)


async def chat_list_endpoint(
    chat_service=Depends(get_chat_service),
    user: User = Depends(get_api_key_user_async),
):
    """List all conversations for the authenticated user. GET /v1/chat"""
    try:
        history = await chat_service.get_langflow_history(user.user_id)
        conversations = [
            {
                "chat_id": conv.get("response_id"),
                "title": conv.get("title", ""),
                "created_at": conv.get("created_at"),
                "last_activity": conv.get("last_activity"),
                "message_count": conv.get("total_messages", 0),
            }
            for conv in history.get("conversations", [])
        ]
        return JSONResponse({"conversations": conversations})
    except Exception as e:
        logger.error("Failed to list conversations", error=str(e), user_id=user.user_id)
        return JSONResponse({"error": f"Failed to list conversations: {str(e)}"}, status_code=500)


async def chat_get_endpoint(
    chat_id: str,
    chat_service=Depends(get_chat_service),
    user: User = Depends(get_api_key_user_async),
):
    """Get a specific conversation with full message history. GET /v1/chat/{chat_id}"""
    try:
        history = await chat_service.get_langflow_history(user.user_id)

        conversation = None
        for conv in history.get("conversations", []):
            if conv.get("response_id") == chat_id:
                conversation = conv
                break

        if not conversation:
            return JSONResponse({"error": "Conversation not found"}, status_code=404)

        # Transform to public API format
        messages = []
        for msg in conversation.get("messages", []):
            message_data = {
                "role": msg.get("role"),
                "content": msg.get("content"),
                "timestamp": msg.get("timestamp"),
            }
            # Include token usage if available (from Responses API)
            usage = msg.get("response_data", {}).get("usage") if isinstance(msg.get("response_data"), dict) else None
            if usage:
                message_data["usage"] = usage
            messages.append(message_data)

        return JSONResponse({
            "chat_id": conversation.get("response_id"),
            "title": conversation.get("title", ""),
            "created_at": conversation.get("created_at"),
            "last_activity": conversation.get("last_activity"),
            "messages": messages,
        })
    except Exception as e:
        logger.error("Failed to get conversation", error=str(e), user_id=user.user_id, chat_id=chat_id)
        return JSONResponse({"error": f"Failed to get conversation: {str(e)}"}, status_code=500)


async def chat_delete_endpoint(
    chat_id: str,
    chat_service=Depends(get_chat_service),
    user: User = Depends(get_api_key_user_async),
):
    """Delete a conversation. DELETE /v1/chat/{chat_id}"""
    try:
        result = await chat_service.delete_session(user.user_id, chat_id)
        if result.get("success"):
            return JSONResponse({"success": True})
        else:
            return JSONResponse(
                {"error": result.get("error", "Failed to delete conversation")},
                status_code=500,
            )
    except Exception as e:
        logger.error("Failed to delete conversation", error=str(e), user_id=user.user_id, chat_id=chat_id)
        return JSONResponse({"error": f"Failed to delete conversation: {str(e)}"}, status_code=500)
