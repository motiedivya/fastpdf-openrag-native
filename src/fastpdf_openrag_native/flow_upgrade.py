from __future__ import annotations

from copy import deepcopy
import re
from typing import Any

from .settings import AppSettings

AGENT_FLOW_RERANK_MARKER = "fastpdf_openrag_native_backend_rerank_v1"
AGENT_FLOW_PROMPT_MARKER = "Treat the knowledge filter as hard scope"
OPENSEARCH_NODE_NAME = "OpenSearch (Multi-Model Multi-Embedding)"
PROMPT_TEMPLATE_NODE_NAME = "Prompt Template"
SPLIT_TEXT_NODE_NAME = "Split Text"
DEFAULT_UPGRADED_PROMPT_TEMPLATE = """Knowledge filter context: {filter}

User request: {input}

Use the OpenSearch Retrieval Tool whenever the request depends on indexed documents.
Treat the knowledge filter as hard scope and do not ask the user to upload, paste, or re-ingest indexed content.
When the user asks for structured JSON or machine-readable output, keep response fields clean. Do not place inline citations, source filenames, chunk ids, markdown, or `(Source: ...)` strings inside JSON values; citations are attached downstream.
"""


def build_backend_rerank_helpers(settings: AppSettings) -> str:
    enabled_default = "true" if settings.backend_rerank_enabled else "false"
    provider_default = settings.backend_rerank_provider or "cross_encoder"
    model_default = settings.backend_rerank_model or ""
    if provider_default == "cohere":
        model_default = model_default or "rerank-english-v3.0"
    else:
        model_default = model_default or "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_n_default = max(1, int(settings.backend_rerank_top_n))
    return f"""
FASTPDF_BACKEND_RERANK_MARKER = "{AGENT_FLOW_RERANK_MARKER}"
_FASTPDF_RERANKER_CACHE = {{}}

def _fastpdf_backend_rerank_enabled() -> bool:
    import os

    value = (os.getenv("FASTPDF_OPENRAG_BACKEND_RERANK_ENABLED", {enabled_default!r}) or "").strip().lower()
    return value not in {{"0", "false", "no", "off", ""}}


def _fastpdf_backend_reranker_provider() -> str:
    import os

    return (
        os.getenv("FASTPDF_OPENRAG_BACKEND_RERANK_PROVIDER", {provider_default!r}) or {provider_default!r}
    ).strip().lower()


def _fastpdf_backend_reranker_model() -> str:
    import os

    default_model = {model_default!r}
    return (os.getenv("FASTPDF_OPENRAG_BACKEND_RERANK_MODEL", default_model) or default_model).strip()


def _fastpdf_backend_reranker_top_n(default_top_n: int | None = None) -> int:
    import os

    raw_value = os.getenv("FASTPDF_OPENRAG_BACKEND_RERANK_TOP_N")
    try:
        parsed = int(raw_value) if raw_value else int(default_top_n or {top_n_default})
    except (TypeError, ValueError):
        parsed = int(default_top_n or {top_n_default})
    return max(1, parsed)


def _fastpdf_load_cross_encoder(model_name: str):
    cached = _FASTPDF_RERANKER_CACHE.get(model_name)
    if cached is not None:
        return cached
    from sentence_transformers.cross_encoder import CrossEncoder

    logger.info("Loading FastPDF backend reranker", provider="cross_encoder", model=model_name)
    encoder = CrossEncoder(model_name)
    _FASTPDF_RERANKER_CACHE[model_name] = encoder
    return encoder


def _fastpdf_backend_rerank_hits(
    *,
    query: str,
    hits: list[dict[str, Any]],
    top_n: int | None = None,
) -> list[dict[str, Any]]:
    if not _fastpdf_backend_rerank_enabled():
        return hits

    query = (query or "").strip()
    if not query or len(hits) < 2:
        return hits[:top_n] if top_n else hits

    provider = _fastpdf_backend_reranker_provider()
    resolved_top_n = min(len(hits), _fastpdf_backend_reranker_top_n(top_n or len(hits)))

    try:
        if provider == "cohere":
            import os
            import cohere

            api_key = os.getenv("COHERE_API_KEY")
            if not api_key:
                logger.warning("COHERE_API_KEY is missing; skipping backend rerank")
                return hits[:resolved_top_n]

            client = _FASTPDF_RERANKER_CACHE.get("__cohere_client__")
            if client is None:
                client = cohere.ClientV2(api_key=api_key)
                _FASTPDF_RERANKER_CACHE["__cohere_client__"] = client

            documents = [str(hit.get("_source", {{}}).get("text") or "") for hit in hits]
            response = client.rerank(
                model=_fastpdf_backend_reranker_model(),
                query=query,
                documents=documents,
                top_n=resolved_top_n,
            )

            reranked_hits = []
            seen_indexes = set()
            for row in getattr(response, "results", []) or []:
                index = getattr(row, "index", None)
                if index is None or index >= len(hits):
                    continue
                seen_indexes.add(index)
                enriched = dict(hits[index])
                enriched["base_score"] = enriched.get("_score")
                enriched["rerank_score"] = float(getattr(row, "relevance_score", 0.0) or 0.0)
                reranked_hits.append(enriched)

            for index, hit in enumerate(hits):
                if index in seen_indexes:
                    continue
                enriched = dict(hit)
                enriched["base_score"] = enriched.get("_score")
                enriched["rerank_score"] = enriched.get("_score")
                reranked_hits.append(enriched)

            return reranked_hits[:resolved_top_n]

        model_name = _fastpdf_backend_reranker_model()
        encoder = _fastpdf_load_cross_encoder(model_name)
        pairs = [(query, str(hit.get("_source", {{}}).get("text") or "")) for hit in hits]
        scores = encoder.predict(pairs)

        reranked_hits = []
        for hit, score in zip(hits, scores, strict=False):
            enriched = dict(hit)
            enriched["base_score"] = enriched.get("_score")
            enriched["rerank_score"] = float(score)
            reranked_hits.append(enriched)

        reranked_hits.sort(
            key=lambda hit: (
                hit.get("rerank_score", float("-inf")),
                hit.get("_score", float("-inf")),
            ),
            reverse=True,
        )
        return reranked_hits[:resolved_top_n]
    except Exception as exc:
        logger.warning(
            "FastPDF backend rerank failed; returning hybrid order",
            provider=provider,
            error=str(exc),
        )
        return hits[:resolved_top_n]
""".strip()


def _flow_nodes(flow: dict[str, Any]) -> list[dict[str, Any]]:
    nodes = flow.get("data", {}).get("nodes", [])
    if not isinstance(nodes, list):
        raise ValueError("flow.data.nodes must be a list")
    return nodes


def find_node(flow: dict[str, Any], display_name: str) -> dict[str, Any]:
    for node in _flow_nodes(flow):
        current_name = node.get("data", {}).get("node", {}).get("display_name")
        if current_name == display_name:
            return node
    raise ValueError(f"flow does not contain node {display_name!r}")


def node_template(flow: dict[str, Any], display_name: str) -> dict[str, Any]:
    return find_node(flow, display_name).get("data", {}).get("node", {}).get("template", {})


def prompt_template_is_upgraded(flow: dict[str, Any]) -> bool:
    template = node_template(flow, PROMPT_TEMPLATE_NODE_NAME)
    current = str(template.get("template", {}).get("value") or "")
    return AGENT_FLOW_PROMPT_MARKER in current


def agent_flow_has_backend_rerank(flow: dict[str, Any]) -> bool:
    template = node_template(flow, OPENSEARCH_NODE_NAME)
    code = str(template.get("code", {}).get("value") or "")
    return AGENT_FLOW_RERANK_MARKER in code


def _replace_once(text: str, old: str, new: str, *, context: str) -> str:
    if old not in text:
        raise ValueError(f"could not find expected {context} anchor")
    return text.replace(old, new, 1)


def _replace_pattern_once(text: str, pattern: str, replacement: str, *, context: str) -> str:
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise ValueError(f"could not find expected {context} anchor")
    return updated


def patch_opensearch_component_code(code: str, settings: AppSettings) -> str:
    if AGENT_FLOW_RERANK_MARKER not in code:
        code = _replace_once(
            code,
            "REQUEST_TIMEOUT = 60\nMAX_RETRIES = 5",
            "REQUEST_TIMEOUT = 60\nMAX_RETRIES = 5\n\n" + build_backend_rerank_helpers(settings),
            context="backend rerank helper insertion",
        )

    if '_fastpdf_backend_rerank_hits(query=query, hits=hits)' not in code:
        code = _replace_pattern_once(
            code,
            (
                r'(?P<indent>[ \t]*)return \[\n'
                r'(?P<body>(?:(?P=indent)[ \t]+.*\n)+?)'
                r'(?P=indent)\]\n'
            ),
            (
                r'\g<indent>hits = _fastpdf_backend_rerank_hits(query=query, hits=hits)\n'
                r'\g<indent>return [\n'
                r'\g<indent>    {\n'
                r'\g<indent>        "page_content": hit["_source"].get("text", ""),\n'
                r'\g<indent>        "metadata": {k: v for k, v in hit["_source"].items() if k != "text"},\n'
                r'\g<indent>        "score": hit.get("rerank_score", hit.get("_score")),\n'
                r'\g<indent>        "base_score": hit.get("base_score", hit.get("_score")),\n'
                r'\g<indent>        "rerank_score": hit.get("rerank_score"),\n'
                r'\g<indent>    }\n'
                r'\g<indent>    for hit in hits\n'
                r'\g<indent>]\n'
            ),
            context="search result transformation",
        )

    if 'score=hit.get("score")' not in code:
        code = _replace_pattern_once(
            code,
            (
                r'(?P<indent>[ \t]*)raw = self\.search\(search_query\)\n'
                r'(?P=indent)return \[Data\(text=hit\["page_content"\], \*\*hit\["metadata"\]\) for hit in raw\]'
            ),
            (
                r'\g<indent>raw = self.search(search_query)\n'
                r'\g<indent>return [\n'
                r'\g<indent>    Data(\n'
                r'\g<indent>        text=hit["page_content"],\n'
                r'\g<indent>        score=hit.get("score"),\n'
                r'\g<indent>        base_score=hit.get("base_score"),\n'
                r'\g<indent>        rerank_score=hit.get("rerank_score"),\n'
                r'\g<indent>        **hit["metadata"],\n'
                r'\g<indent>    )\n'
                r'\g<indent>    for hit in raw\n'
                r'\g<indent>]'
            ),
            context="search_documents data mapping",
        )
    return code


def upgrade_agent_flow(flow: dict[str, Any], settings: AppSettings) -> dict[str, Any]:
    upgraded = deepcopy(flow)
    prompt = node_template(upgraded, PROMPT_TEMPLATE_NODE_NAME)
    prompt["template"]["value"] = DEFAULT_UPGRADED_PROMPT_TEMPLATE

    opensearch_template = node_template(upgraded, OPENSEARCH_NODE_NAME)
    current_code = str(opensearch_template.get("code", {}).get("value") or "")
    opensearch_template["code"]["value"] = patch_opensearch_component_code(current_code, settings)
    if "number_of_results" in opensearch_template:
        opensearch_template["number_of_results"]["value"] = max(
            int(opensearch_template["number_of_results"].get("value") or 0),
            settings.backend_rerank_candidate_limit,
        )
    return upgraded


def upgrade_ingestion_flow(flow: dict[str, Any], settings: AppSettings) -> dict[str, Any]:
    upgraded = deepcopy(flow)
    split_template = node_template(upgraded, SPLIT_TEXT_NODE_NAME)
    if "chunk_size" in split_template:
        split_template["chunk_size"]["value"] = settings.default_chunk_size
    if "chunk_overlap" in split_template:
        split_template["chunk_overlap"]["value"] = settings.default_chunk_overlap
    if "separator" in split_template:
        split_template["separator"]["value"] = "\\n\\n"
    return upgraded


def summarize_flow_upgrade(flow: dict[str, Any]) -> dict[str, bool]:
    return {
        "prompt_upgraded": prompt_template_is_upgraded(flow),
        "backend_rerank_present": agent_flow_has_backend_rerank(flow),
    }
