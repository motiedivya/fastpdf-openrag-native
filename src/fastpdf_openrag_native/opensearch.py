from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import httpx

from .models import OpenSearchChunkRecord, OpenSearchIndexDiagnostics, OpenSearchRepairResult, utc_now
from .reranking import RERANKER_TYPE
from .settings import AppSettings, get_settings

KNOWLEDGE_FILTER_INDEX_NAME = "knowledge_filters"
HYBRID_RETRIEVAL_MODE = "hybrid"
HYBRID_SEMANTIC_WEIGHT = 0.7
HYBRID_KEYWORD_WEIGHT = 0.3


class OpenSearchInspector:
    def __init__(self, settings: AppSettings | None = None):
        self.settings = settings or get_settings()
        self._cached_credentials: tuple[str, str] | None = None
        self._cached_target_index_body: dict[str, Any] | None = None

    def _discover_local_credentials(self) -> tuple[str, str] | None:
        try:
            result = subprocess.run(
                [
                    "docker",
                    "inspect",
                    "openrag-backend",
                    "--format",
                    "{{json .Config.Env}}",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            return None
        if result.returncode != 0:
            return None
        try:
            env_rows = json.loads(result.stdout)
        except json.JSONDecodeError:
            return None

        values: dict[str, str] = {}
        for row in env_rows:
            if "=" not in row:
                continue
            key, value = row.split("=", 1)
            values[key] = value
        username = values.get("OPENSEARCH_USERNAME")
        password = values.get("OPENSEARCH_PASSWORD")
        if username and password:
            return username, password
        return None

    def _resolve_auth(self) -> tuple[str, str]:
        if self.settings.opensearch_username and self.settings.opensearch_password:
            return self.settings.opensearch_username, self.settings.opensearch_password
        if self._cached_credentials:
            return self._cached_credentials
        discovered = self._discover_local_credentials()
        if discovered:
            self._cached_credentials = discovered
            return discovered
        raise ValueError(
            "OpenSearch credentials are required. Set FASTPDF_OPENRAG_OPENSEARCH_USERNAME and "
            "FASTPDF_OPENRAG_OPENSEARCH_PASSWORD or run against the local openrag-backend container."
        )

    @staticmethod
    def _extract_last_json_line(stdout: str) -> dict[str, Any]:
        for line in reversed(stdout.splitlines()):
            candidate = line.strip()
            if not candidate:
                continue
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
        raise ValueError("No JSON object found in command output")

    def _run_backend_python(self, script: str) -> dict[str, Any]:
        command = (
            "PYTHONPATH=/app/src /app/.venv/bin/python - <<\"PY\"\n"
            f"{script}\n"
            "PY"
        )
        try:
            result = subprocess.run(
                ["docker", "exec", "openrag-backend", "sh", "-lc", command],
                check=False,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("docker is required to inspect the local openrag-backend container") from exc
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "openrag-backend helper failed")
        return self._extract_last_json_line(result.stdout)

    def discover_target_index_body(self) -> dict[str, Any]:
        if self._cached_target_index_body is not None:
            return self._cached_target_index_body
        script = """
import asyncio
import json
from config.settings import get_openrag_config
from utils.embeddings import create_dynamic_index_body

async def main():
    config = get_openrag_config()
    provider = config.knowledge.embedding_provider
    model = config.knowledge.embedding_model
    provider_config = config.get_embedding_provider_config()
    body = await create_dynamic_index_body(
        model,
        provider=provider,
        endpoint=getattr(provider_config, "endpoint", None),
    )
    body.setdefault("settings", {})["number_of_replicas"] = 0
    body["settings"]["auto_expand_replicas"] = "0-0"
    print(json.dumps({
        "embedding_provider": provider,
        "embedding_model": model,
        "body": body,
    }))

asyncio.run(main())
""".strip()
        self._cached_target_index_body = self._run_backend_python(script)
        return self._cached_target_index_body

    @staticmethod
    def _parse_cat_indices_doc_count(cat_indices: str | None, index_name: str) -> int | None:
        if not cat_indices:
            return None
        lines = [line.strip() for line in cat_indices.splitlines() if line.strip()]
        if len(lines) < 2:
            return None
        header = lines[0].split()
        try:
            index_column = header.index("index")
            docs_column = header.index("docs.count")
        except ValueError:
            return None

        for line in lines[1:]:
            columns = line.split()
            if len(columns) <= max(index_column, docs_column):
                continue
            if columns[index_column] != index_name:
                continue
            raw_value = columns[docs_column].replace(",", "")
            if not raw_value or raw_value == "-":
                return None
            try:
                return int(raw_value)
            except ValueError:
                return None
        return None

    @classmethod
    def _extract_vector_fields(cls, index_mapping: dict[str, Any], index_name: str) -> list[str]:
        root = index_mapping.get(index_name, {}).get("mappings", {})
        fields: list[str] = []

        def visit(node: dict[str, Any], prefix: str = "") -> None:
            properties = node.get("properties")
            if not isinstance(properties, dict):
                return
            for name, spec in properties.items():
                full_name = f"{prefix}.{name}" if prefix else name
                if isinstance(spec, dict) and spec.get("type") in {"knn_vector", "dense_vector"}:
                    fields.append(full_name)
                if isinstance(spec, dict):
                    visit(spec, full_name)

        visit(root)
        return fields

    async def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        username, password = self._resolve_auth()
        async with httpx.AsyncClient(
            auth=(username, password),
            verify=self.settings.opensearch_verify_ssl,
            timeout=60.0,
        ) as client:
            response = await client.request(
                method,
                f"{self.settings.opensearch_url.rstrip('/')}{path}",
                **kwargs,
            )
        return response

    async def _request_text(self, method: str, path: str, **kwargs) -> str:
        response = await self._request(method, path, **kwargs)
        response.raise_for_status()
        return response.text

    async def index_exists(self) -> bool:
        response = await self._request("HEAD", f"/{self.settings.opensearch_index_name}")
        if response.status_code == 404:
            return False
        response.raise_for_status()
        return True

    async def cluster_health(self) -> dict[str, Any]:
        response = await self._request("GET", "/_cluster/health")
        response.raise_for_status()
        return response.json()

    async def allocation_explain(self) -> dict[str, Any]:
        response = await self._request("GET", "/_cluster/allocation/explain")
        if response.status_code >= 400:
            return {
                "status_code": response.status_code,
                "body": response.json(),
            }
        return response.json()

    async def index_mapping(self) -> dict[str, Any]:
        response = await self._request("GET", f"/{self.settings.opensearch_index_name}/_mapping")
        if response.status_code == 404:
            return {}
        response.raise_for_status()
        return response.json()

    async def index_settings(self) -> dict[str, Any]:
        response = await self._request("GET", f"/{self.settings.opensearch_index_name}/_settings")
        if response.status_code == 404:
            return {}
        response.raise_for_status()
        return response.json()

    async def cat_indices(self) -> str:
        return await self._request_text(
            "GET",
            "/_cat/indices?v&h=health,status,index,pri,rep,docs.count,store.size",
        )

    async def cat_shards(self) -> str:
        return await self._request_text(
            "GET",
            f"/_cat/shards/{self.settings.opensearch_index_name}?v&h=index,shard,prirep,state,unassigned.reason,node",
        )

    async def list_chunks_for_filename(self, filename: str, *, size: int = 50) -> list[OpenSearchChunkRecord]:
        response = await self._request(
            "POST",
            f"/{self.settings.opensearch_index_name}/_search",
            json={
                "size": size,
                "query": {"term": {"filename": filename}},
                "_source": [
                    "filename",
                    "text",
                    "page",
                    "mimetype",
                    "embedding_model",
                    "embedding_dimensions",
                    "metadata",
                ],
            },
        )
        response.raise_for_status()
        payload = response.json()
        rows: list[OpenSearchChunkRecord] = []
        for hit in payload.get("hits", {}).get("hits", []):
            source = hit.get("_source", {})
            rows.append(
                OpenSearchChunkRecord(
                    id=hit.get("_id"),
                    filename=source.get("filename", ""),
                    text=source.get("text", ""),
                    score=hit.get("_score"),
                    page=source.get("page"),
                    mimetype=source.get("mimetype"),
                    embedding_model=source.get("embedding_model"),
                    embedding_dimensions=source.get("embedding_dimensions"),
                    metadata=source.get("metadata") or {},
                )
            )
        return rows

    async def normalize_single_node_replicas(self, index_name: str) -> bool:
        response = await self._request("HEAD", f"/{index_name}")
        if response.status_code == 404:
            return False
        response.raise_for_status()
        response = await self._request(
            "PUT",
            f"/{index_name}/_settings",
            json={"index": {"number_of_replicas": 0, "auto_expand_replicas": "0-0"}},
        )
        response.raise_for_status()
        return True

    async def diagnostics(self) -> OpenSearchIndexDiagnostics:
        cluster_health = await self.cluster_health()
        allocation_explain = await self.allocation_explain()
        index_exists = await self.index_exists()
        index_mapping = await self.index_mapping() if index_exists else {}
        index_settings = await self.index_settings() if index_exists else {}
        cat_indices = await self.cat_indices()
        cat_shards = await self.cat_shards() if index_exists else None
        index_search_error: dict[str, Any] | None = None
        embedding_provider: str | None = None
        embedding_model: str | None = None

        try:
            target = self.discover_target_index_body()
            embedding_provider = target.get("embedding_provider")
            embedding_model = target.get("embedding_model")
        except Exception:
            embedding_provider = None
            embedding_model = None

        if index_exists:
            response = await self._request(
                "POST",
                f"/{self.settings.opensearch_index_name}/_search",
                json={"size": 1, "query": {"match_all": {}}},
            )
            if response.status_code >= 400:
                index_search_error = {
                    "status_code": response.status_code,
                    "body": response.json(),
                }

        return OpenSearchIndexDiagnostics(
            database_backend="OpenSearch",
            cluster_name=cluster_health.get("cluster_name"),
            documents_index_name=self.settings.opensearch_index_name,
            knowledge_filter_index_name=KNOWLEDGE_FILTER_INDEX_NAME,
            document_count=self._parse_cat_indices_doc_count(
                cat_indices,
                self.settings.opensearch_index_name,
            ),
            retrieval_mode=HYBRID_RETRIEVAL_MODE,
            application_retrieval_mode=(
                "hybrid_backend_reranked"
                if self.settings.backend_rerank_enabled
                else (
                    "hybrid_application_reranked"
                    if self.settings.retrieval_rerank_enabled
                    else HYBRID_RETRIEVAL_MODE
                )
            ),
            semantic_weight=HYBRID_SEMANTIC_WEIGHT,
            keyword_weight=HYBRID_KEYWORD_WEIGHT,
            reranking_enabled=self.settings.backend_rerank_enabled or self.settings.retrieval_rerank_enabled,
            reranker_location=(
                "langflow_agent_tool"
                if self.settings.backend_rerank_enabled
                else ("application" if self.settings.retrieval_rerank_enabled else None)
            ),
            reranker_type=(
                self.settings.backend_rerank_provider
                if self.settings.backend_rerank_enabled
                else (RERANKER_TYPE if self.settings.retrieval_rerank_enabled else None)
            ),
            reranker_model=self.settings.backend_rerank_model if self.settings.backend_rerank_enabled else None,
            chunking_enabled=True,
            structure_chunking_strategy="structure_aware_blocks",
            chunk_size=self.settings.default_chunk_size,
            chunk_overlap=self.settings.default_chunk_overlap,
            prechunk_target_chars=self.settings.structure_chunk_target_chars,
            prechunk_overlap_blocks=self.settings.structure_chunk_overlap_blocks,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            vector_fields=self._extract_vector_fields(index_mapping, self.settings.opensearch_index_name),
            cluster_health=cluster_health,
            allocation_explain=allocation_explain,
            index_mapping=index_mapping,
            index_settings=index_settings,
            cat_indices=cat_indices,
            cat_shards=cat_shards,
            index_exists=index_exists,
            index_search_error=index_search_error,
        )

    async def repair_documents_index(
        self,
        *,
        output_dir: Path | None = None,
    ) -> OpenSearchRepairResult:
        before = await self.diagnostics()
        target = self.discover_target_index_body()
        target_body = target["body"]
        target_embedding_provider = target.get("embedding_provider")
        target_embedding_model = target.get("embedding_model")

        deleted_existing_index = False
        if before.index_exists:
            response = await self._request("DELETE", f"/{self.settings.opensearch_index_name}")
            if response.status_code not in {200, 404}:
                response.raise_for_status()
            deleted_existing_index = response.status_code == 200

        response = await self._request(
            "PUT",
            f"/{self.settings.opensearch_index_name}",
            json=target_body,
        )
        response.raise_for_status()

        normalized_indices: list[str] = []
        for index_name in (self.settings.opensearch_index_name, KNOWLEDGE_FILTER_INDEX_NAME):
            if await self.normalize_single_node_replicas(index_name):
                normalized_indices.append(index_name)

        after = await self.diagnostics()

        resolved_output_dir = output_dir
        if resolved_output_dir is None:
            resolved_output_dir = (
                self.settings.output_root
                / "opensearch-repair"
                / utc_now().strftime("%Y%m%dT%H%M%SZ")
            )
        resolved_output_dir.mkdir(parents=True, exist_ok=True)
        (resolved_output_dir / "before.json").write_text(
            before.model_dump_json(indent=2),
            encoding="utf-8",
        )
        (resolved_output_dir / "target-index-body.json").write_text(
            json.dumps(target, indent=2),
            encoding="utf-8",
        )
        (resolved_output_dir / "after.json").write_text(
            after.model_dump_json(indent=2),
            encoding="utf-8",
        )

        return OpenSearchRepairResult(
            index_name=self.settings.opensearch_index_name,
            output_dir=resolved_output_dir.as_posix(),
            target_embedding_provider=target_embedding_provider,
            target_embedding_model=target_embedding_model,
            deleted_existing_index=deleted_existing_index,
            normalized_indices=normalized_indices,
            before=before,
            after=after,
        )
