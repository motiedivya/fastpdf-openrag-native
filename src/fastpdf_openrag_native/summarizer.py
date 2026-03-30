from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any, Protocol

from .models import (
    MaterializationManifest,
    PageMapSummary,
    ScopedSummaryResult,
    SummaryScope,
    VerifiedSentence,
)
from .prompts import build_page_map_prompt, build_reduce_prompt
from .reranking import RERANKER_TYPE, rerank_hits, select_top_source_filenames
from .settings import AppSettings, get_settings


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


def _split_sentences(text: str) -> list[str]:
    clean = re.sub(r"\s+", " ", text or "").strip()
    if not clean:
        return []
    parts = re.split(r"(?<=[.!?])\s+", clean)
    return [part.strip() for part in parts if part.strip()]


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


def _prepare_selected_sources(
    *,
    query: str,
    hits: list[Any],
    settings: AppSettings,
    requested_limit: int,
    source_groups: list[list[str]] | None = None,
) -> tuple[list[Any], list[str]]:
    reranked_hits = rerank_hits(query, list(hits)) if settings.retrieval_rerank_enabled else list(hits)
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
) -> tuple[PageMapSummary, dict[str, Any]]:
    page_sources = _page_retrieval_sources(page)
    prompt = build_page_map_prompt(
        scope,
        pdf_id=page.pdf_id,
        page=page.page,
        source_filename=page.source_filename,
        retrieval_source_count=len(page_sources),
    )
    preflight_query = f"page summary {scope.objective}"
    preflight_sources = await gateway.search_on_sources(
        query=preflight_query,
        data_sources=page_sources,
        limit=_candidate_limit(8, settings),
        score_threshold=0,
    )
    if settings.backend_rerank_enabled:
        reranked_preflight_sources = list(preflight_sources)
        selected_page_sources = list(page_sources)
    else:
        reranked_preflight_sources, selected_page_sources = _prepare_selected_sources(
            query=preflight_query,
            hits=preflight_sources,
            settings=settings,
            requested_limit=8,
            source_groups=[page_sources],
        )
    response_text, sources = await gateway.chat_on_sources(
        message=prompt,
        data_sources=selected_page_sources or page_sources,
        limit=max(_chat_limit(8, settings), len(selected_page_sources)),
        score_threshold=0,
    )
    retry_used = False
    if not sources or _looks_like_retrieval_failure(response_text):
        retry_prompt = "\n\n".join(
            [
                prompt,
                (
                    f"OpenRAG search preflight found {len(reranked_preflight_sources)} candidate indexed hit(s) "
                    f"across {len(selected_page_sources or page_sources)} retrieval chunk file(s) for "
                    f"{page.source_filename}. Use the OpenSearch Retrieval Tool and answer from those indexed "
                    "results only."
                ),
            ]
        )
        response_text, sources = await gateway.chat_on_sources(
            message=retry_prompt,
            data_sources=selected_page_sources or page_sources,
            limit=max(_chat_limit(8, settings), len(selected_page_sources or page_sources)),
            score_threshold=0,
        )
        retry_used = True
    parsed = _extract_json_object(response_text) or {}
    summary = str(parsed.get("summary") or response_text).strip()
    key_facts = parsed.get("key_facts")
    if not isinstance(key_facts, list):
        key_facts = []

    page_summary = PageMapSummary(
        pdf_id=page.pdf_id,
        page=page.page,
        source_filename=page.source_filename,
        summary=summary,
        key_facts=[str(item).strip() for item in key_facts if str(item).strip()],
        raw_response=response_text,
        retrieved_sources=list(sources or reranked_preflight_sources[: max(1, len(selected_page_sources))]),
    )
    return page_summary, {
        "prompt": prompt,
        "preflight_query": preflight_query,
        "preflight_sources": [source.model_dump() for source in reranked_preflight_sources],
        "data_sources": page_sources,
        "selected_source_filenames": selected_page_sources,
        "limit": 8,
        "score_threshold": 0,
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
) -> VerifiedSentence:
    evidence = await gateway.search_on_sources(
        query=sentence,
        data_sources=data_sources,
        limit=_verification_candidate_limit(settings),
        score_threshold=settings.verification_score_threshold,
    )
    reranked_evidence = (
        rerank_hits(sentence, list(evidence))
        if settings.retrieval_rerank_enabled or (settings.backend_rerank_enabled and not settings.backend_search_rerank_enabled)
        else list(evidence)
    )
    return VerifiedSentence(
        sentence=sentence,
        supported=bool(reranked_evidence),
        evidence=list(reranked_evidence[: settings.verification_limit]),
    )


async def summarize_scope(
    gateway: RetrievalGateway,
    *,
    manifest: MaterializationManifest,
    scope: SummaryScope,
    settings: AppSettings | None = None,
) -> ScopedSummaryResult:
    effective_settings = settings or get_settings()
    pages = resolve_scope_pages(manifest, scope)
    page_source_groups = [_page_retrieval_sources(page) for page in pages]
    page_sources = _dedupe_preserve_order(
        [source for group in page_source_groups for source in group]
    )
    semaphore = asyncio.Semaphore(max(1, effective_settings.page_summary_concurrency))

    async def bounded_page_summary(page):
        async with semaphore:
            return await _summarize_page(gateway, scope, page, effective_settings)

    page_results = await asyncio.gather(*(bounded_page_summary(page) for page in pages))
    page_summaries = [item[0] for item in page_results]
    page_debug = [item[1] for item in page_results]
    reduce_prompt = build_reduce_prompt(scope, list(page_summaries))
    reduce_limit = min(50, max(10, len(pages)))
    reduce_preflight_query = f"overall chronology summary {scope.objective}"
    reduce_preflight_sources = await gateway.search_on_sources(
        query=reduce_preflight_query,
        data_sources=page_sources,
        limit=_candidate_limit(reduce_limit, effective_settings),
        score_threshold=0,
    )
    if effective_settings.backend_rerank_enabled:
        reranked_reduce_sources = list(reduce_preflight_sources)
        selected_reduce_sources = list(page_sources)
    else:
        reranked_reduce_sources, selected_reduce_sources = _prepare_selected_sources(
            query=reduce_preflight_query,
            hits=reduce_preflight_sources,
            settings=effective_settings,
            requested_limit=reduce_limit,
            source_groups=page_source_groups,
        )
    reduce_response, reduce_sources = await gateway.chat_on_sources(
        message=reduce_prompt,
        data_sources=selected_reduce_sources or page_sources,
        limit=max(_chat_limit(reduce_limit, effective_settings), len(selected_reduce_sources)),
        score_threshold=0,
    )
    reduce_retry_used = False
    if not reduce_sources or _looks_like_retrieval_failure(reduce_response):
        retry_prompt = "\n\n".join(
            [
                reduce_prompt,
                (
                    f"OpenRAG search preflight found {len(reranked_reduce_sources)} candidate indexed hit(s) across "
                    f"{len(selected_reduce_sources or page_sources)} retrieval chunk file(s). Use the OpenSearch "
                    "Retrieval Tool and answer from the selected indexed files only."
                ),
            ]
        )
        reduce_response, reduce_sources = await gateway.chat_on_sources(
            message=retry_prompt,
            data_sources=selected_reduce_sources or page_sources,
            limit=max(_chat_limit(reduce_limit, effective_settings), len(selected_reduce_sources or page_sources)),
            score_threshold=0,
        )
        reduce_retry_used = True
    reduce_payload = _extract_json_object(reduce_response) or {}
    draft_title = str(reduce_payload.get("title") or scope.title).strip() or scope.title
    draft_summary = str(reduce_payload.get("summary") or reduce_response).strip()
    chronology = reduce_payload.get("chronology")
    if not isinstance(chronology, list):
        chronology = [page_summary.summary for page_summary in page_summaries]

    verification_queries = _split_sentences(draft_summary)
    verified_sentences = await asyncio.gather(
        *(
            _verify_sentence(
                gateway,
                sentence=sentence,
                data_sources=page_sources,
                settings=effective_settings,
            )
            for sentence in verification_queries
        )
    )
    supported_summary = " ".join(
        row.sentence for row in verified_sentences if row.supported
    ).strip()
    unsupported_sentences = [row.sentence for row in verified_sentences if not row.supported]

    return ScopedSummaryResult(
        run_id=manifest.run_id,
        scope=scope,
        source_filenames=page_sources,
        page_summaries=list(page_summaries),
        draft_title=draft_title,
        draft_summary=draft_summary,
        chronology=[str(item).strip() for item in chronology if str(item).strip()],
        verified_sentences=list(verified_sentences),
        supported_summary=supported_summary or draft_summary,
        unsupported_sentences=unsupported_sentences,
        debug={
            "page_requests": page_debug,
            "reduce_prompt": reduce_prompt,
            "reduce_preflight_query": reduce_preflight_query,
            "reduce_preflight_sources": [source.model_dump() for source in reranked_reduce_sources],
            "reduce_selected_source_filenames": selected_reduce_sources,
            "reduce_limit": reduce_limit,
            "reduce_retry_used": reduce_retry_used,
            "reranking_enabled": _reranking_enabled(effective_settings),
            "reranker_location": _reranker_location(effective_settings),
            "reranker_type": _reranker_type(effective_settings),
            "reduce_response": reduce_response,
            "reduce_sources": [source.model_dump() for source in reduce_sources],
            "verification_queries": verification_queries,
            "verification_candidate_limit": _verification_candidate_limit(effective_settings),
            "verification_reranking_enabled": bool(_verification_reranker_location(effective_settings)),
            "verification_reranker_location": _verification_reranker_location(effective_settings),
            "verification_reranker_type": _verification_reranker_type(effective_settings),
        },
    )
