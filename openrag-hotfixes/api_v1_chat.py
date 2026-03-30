"""
Public API v1 Chat endpoint.

Provides chat functionality with streaming support and conversation history.
Uses API key authentication. Routes through Langflow (chat_service.langflow_chat).
"""
import asyncio
import json
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


def _best_score_for_source(source: dict, candidates: list[dict]) -> Optional[float]:
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
            return candidate.get("score")

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

    if best_candidate and best_candidate.get("score") is not None:
        return best_candidate.get("score")
    return None


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
    if not sources or not tool_queries:
        return sources

    query_cache: dict[str, list[dict]] = {}
    for query in tool_queries:
        if query in query_cache:
            continue
        try:
            search_response = await search_service.search(
                query,
                user_id=user_id,
                jwt_token=jwt_token,
                filters=filters or {},
                limit=max(limit, len(sources), 10),
                score_threshold=0,
            )
            query_cache[query] = search_response.get("results", [])
        except Exception as exc:
            logger.warning(
                "Failed to backfill source scores",
                query=query,
                error=str(exc),
            )
            query_cache[query] = []

    for source in sources:
        if source.get("score") not in (None, 0):
            continue
        best_score = None
        for query in tool_queries:
            score = _best_score_for_source(source, query_cache.get(query, []))
            if score is None:
                continue
            if best_score is None or score > best_score:
                best_score = score
        if best_score is not None:
            source["score"] = best_score

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
        sources = _coerce_list(result.get("sources"))
        chat_id = result.get("response_id")
        tool_queries: list[str] = []
        if not sources and chat_id:
            sources, tool_queries = await _recover_history_context(user_id, chat_id)
        if sources and chat_id:
            if not tool_queries:
                _history_sources, tool_queries = await _recover_history_context(user_id, chat_id)
            if message not in tool_queries:
                tool_queries.append(message)
            sources = await _backfill_source_scores(
                sources=sources,
                tool_queries=tool_queries,
                search_service=search_service,
                user_id=user_id,
                jwt_token=jwt_token,
                filters=body.filters,
                limit=body.limit,
            )
        return JSONResponse({
            "response": result.get("response", ""),
            "chat_id": chat_id,
            "sources": sources,
        })


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
