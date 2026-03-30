from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from openrag_sdk import OpenRAGClient, OpenRAGError

from .models import EvidenceHit, IngestedDocumentResult, KnowledgeFilterResult, MaterializationManifest, SummaryScope
from .settings import AppSettings, get_settings


class OpenRAGGateway:
    def __init__(self, settings: AppSettings | None = None):
        self.settings = settings or get_settings()
        self._cached_api_key: str | None = None

    def _discover_local_api_key(self) -> str | None:
        script = (
            "import asyncio\n"
            "from config.settings import clients\n"
            "from services.api_key_service import APIKeyService\n"
            "async def main():\n"
            "    await clients.initialize()\n"
            "    result = await APIKeyService().create_key(\n"
            "        user_id='fastpdf-openrag-native',\n"
            "        user_email='fastpdf-openrag-native@local',\n"
            "        name='fastpdf-openrag-native local cli',\n"
            "    )\n"
            "    print('OPENRAG_API_KEY=' + (result.get('api_key') or ''))\n"
            "asyncio.run(main())\n"
        )
        try:
            result = subprocess.run(
                [
                    "docker",
                    "exec",
                    "openrag-backend",
                    "env",
                    "PYTHONPATH=/app/src",
                    "/app/.venv/bin/python",
                    "-c",
                    script,
                ],
                check=False,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            return None
        if result.returncode != 0:
            return None
        for line in reversed(result.stdout.splitlines()):
            line = line.strip()
            if line.startswith("OPENRAG_API_KEY="):
                value = line.split("=", 1)[1].strip()
                return value or None
        return None

    def _resolve_api_key(self) -> str:
        if self.settings.openrag_api_key:
            return self.settings.openrag_api_key
        if self._cached_api_key:
            return self._cached_api_key
        discovered = self._discover_local_api_key()
        if discovered:
            self._cached_api_key = discovered
            return discovered
        raise ValueError(
            "OPENRAG_API_KEY is required. Export it, place it in .env, or run against a local "
            "openrag-backend container so the repo can auto-discover the key."
        )

    def _client(self) -> OpenRAGClient:
        return OpenRAGClient(
            api_key=self._resolve_api_key(),
            base_url=self.settings.openrag_url,
            timeout=120.0,
        )

    @staticmethod
    def _filter_allowed_sources(
        sources: list[EvidenceHit],
        *,
        data_sources: list[str] | None,
    ) -> list[EvidenceHit]:
        allowed_sources = {source for source in data_sources or [] if source}
        if not allowed_sources:
            return list(sources)
        return [source for source in sources if source.filename in allowed_sources]

    @staticmethod
    def _coerce_float(value: Any) -> float:
        try:
            return float(value or 0.0)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _coerce_int(value: Any) -> int | None:
        if value in (None, ""):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _coerce_evidence_hits(cls, rows: list[Any] | None) -> list[EvidenceHit]:
        hits: list[EvidenceHit] = []
        for row in rows or []:
            if hasattr(row, "model_dump"):
                row = row.model_dump()
            if not isinstance(row, dict):
                continue
            hits.append(
                EvidenceHit(
                    filename=str(row.get("filename") or ""),
                    text=str(row.get("text") or ""),
                    score=cls._coerce_float(row.get("score")),
                    page=cls._coerce_int(row.get("page")),
                    mimetype=(str(row.get("mimetype")) if row.get("mimetype") is not None else None),
                    base_score=(cls._coerce_float(row.get("base_score")) if row.get("base_score") is not None else None),
                    rerank_score=(cls._coerce_float(row.get("rerank_score")) if row.get("rerank_score") is not None else None),
                    retrieval_rank=cls._coerce_int(row.get("retrieval_rank")),
                )
            )
        return hits

    async def health(self) -> dict[str, object]:
        async with self._client() as client:
            settings = await client.settings.get()
        return {
            "openrag_url": self.settings.openrag_url,
            "has_api_key": True,
            "knowledge": settings.knowledge.model_dump(),
            "agent": settings.agent.model_dump(),
        }

    async def apply_recommended_settings(self) -> dict[str, object]:
        async with self._client() as client:
            await client.settings.update(
                {
                    "chunk_size": self.settings.default_chunk_size,
                    "chunk_overlap": self.settings.default_chunk_overlap,
                    "table_structure": True,
                    "ocr": False,
                    "picture_descriptions": False,
                }
            )
            settings = await client.settings.get()
        return settings.model_dump()

    async def ingest_manifest(
        self,
        manifest: MaterializationManifest,
        *,
        manifest_dir: Path,
        replace_existing: bool = True,
    ) -> list[IngestedDocumentResult]:
        results: list[IngestedDocumentResult] = []
        ingest_documents = manifest.ingest_documents()
        stale_filenames = set(manifest.all_source_filenames()) if replace_existing else set()
        async with self._client() as client:
            for document in ingest_documents:
                delete_error: str | None = None
                if stale_filenames:
                    delete_errors: list[str] = []
                    for filename in sorted(stale_filenames):
                        try:
                            await client.documents.delete(filename)
                        except OpenRAGError as exc:
                            # OpenRAG currently returns 500/503 for some deletes when the filename
                            # is not yet indexed or OpenSearch is still warming. Ingest should still proceed.
                            delete_errors.append(f"{filename}: {exc}")
                    if delete_errors:
                        delete_error = "; ".join(delete_errors)
                    stale_filenames = set()
                try:
                    task = await client.documents.ingest(
                        file_path=manifest_dir / document.relative_path,
                        wait=True,
                        timeout=self.settings.openrag_ingest_wait_timeout,
                    )
                    task_status = task.status
                    task_id = task.task_id
                    successful_files = task.successful_files
                    failed_files = task.failed_files
                except TimeoutError as exc:
                    task_status = f"timeout after {self.settings.openrag_ingest_wait_timeout}s"
                    task_id = None
                    successful_files = 0
                    failed_files = 1
                    delete_error = f"{delete_error}; {exc}" if delete_error else str(exc)
                results.append(
                    IngestedDocumentResult(
                        filename=document.source_filename,
                        status=f"{task_status} (delete skipped)" if delete_error else task_status,
                        successful_files=successful_files,
                        failed_files=failed_files,
                        task_id=task_id,
                        delete_error=delete_error,
                    )
                )
        return results

    async def task_status(self, task_id: str) -> dict[str, object]:
        async with self._client() as client:
            task = await client.documents.get_task_status(task_id)
        return task.model_dump()

    async def chat_on_sources(
        self,
        *,
        message: str,
        data_sources: list[str] | None = None,
        document_types: list[str] | None = None,
        limit: int = 6,
        score_threshold: float = 0,
        filter_id: str | None = None,
    ) -> tuple[str, list[EvidenceHit]]:
        filters = None
        if data_sources:
            filters = {
                "data_sources": data_sources,
                "document_types": document_types or ["text/markdown", "text/html"],
            }
        body: dict[str, Any] = {
            "message": message,
            "stream": False,
            "limit": limit,
            "score_threshold": score_threshold,
        }
        if filters:
            body["filters"] = filters
        if filter_id:
            body["filter_id"] = filter_id
        async with self._client() as client:
            response = await client._request("POST", "/api/v1/chat", json=body)
        data = response.json()
        sources = self._coerce_evidence_hits(data.get("sources", []))
        filtered_sources = self._filter_allowed_sources(sources, data_sources=data_sources)
        return str(data.get("response") or ""), filtered_sources

    async def search_on_sources(
        self,
        *,
        query: str,
        data_sources: list[str] | None = None,
        document_types: list[str] | None = None,
        limit: int | None = None,
        score_threshold: float | None = None,
        filter_id: str | None = None,
    ) -> list[EvidenceHit]:
        filters = None
        if data_sources:
            filters = {
                "data_sources": data_sources,
                "document_types": document_types or ["text/markdown", "text/html"],
            }
        body: dict[str, Any] = {
            "query": query,
            "limit": limit if limit is not None else self.settings.verification_limit,
            "score_threshold": (
                score_threshold
                if score_threshold is not None
                else self.settings.verification_score_threshold
            ),
        }
        if filters:
            body["filters"] = filters
        if filter_id:
            body["filter_id"] = filter_id
        async with self._client() as client:
            response = await client._request("POST", "/api/v1/search", json=body)
        data = response.json()
        results = self._coerce_evidence_hits(data.get("results", []))
        return self._filter_allowed_sources(results, data_sources=data_sources)

    async def upsert_scope_filter(
        self,
        *,
        manifest: MaterializationManifest,
        scope: SummaryScope,
        data_sources: list[str],
    ) -> KnowledgeFilterResult:
        filter_name = f"{self.settings.filter_prefix}:{manifest.run_id}:{scope.scope_id}"
        async with self._client() as client:
            existing_id = None
            for candidate in await client.knowledge_filters.search(filter_name, limit=20):
                if candidate.name == filter_name:
                    existing_id = candidate.id
                    break

            query_data = {
                "query": scope.objective,
                "filters": {
                    "data_sources": data_sources,
                    "document_types": ["text/markdown", "text/html"],
                },
                "limit": 10,
                "scoreThreshold": 0,
            }

            if existing_id:
                await client.knowledge_filters.update(
                    existing_id,
                    {
                        "description": scope.title,
                        "queryData": query_data,
                    },
                )
                filter_id = existing_id
            else:
                created = await client.knowledge_filters.create(
                    {
                        "name": filter_name,
                        "description": scope.title,
                        "queryData": query_data,
                    }
                )
                if not created.id:
                    raise RuntimeError("OpenRAG did not return a knowledge filter id")
                filter_id = created.id

        return KnowledgeFilterResult(
            filter_id=filter_id,
            filter_name=filter_name,
            data_sources=data_sources,
        )
