from __future__ import annotations

import json
import subprocess
from pathlib import Path

import httpx

from .flow_upgrade import (
    agent_flow_has_backend_rerank,
    prompt_template_is_upgraded,
    upgrade_agent_flow,
    upgrade_ingestion_flow,
)
from .models import FlowUpgradeResult, LangflowFlowDiagnostics, utc_now
from .settings import AppSettings, get_settings


class LangflowGateway:
    def __init__(self, settings: AppSettings | None = None):
        self.settings = settings or get_settings()
        self._cached_api_key: str | None = None

    @property
    def agent_flow_path(self) -> Path:
        return self.settings.langflow_flows_root / self.settings.agent_flow_filename

    @property
    def ingestion_flow_path(self) -> Path:
        return self.settings.langflow_flows_root / self.settings.ingestion_flow_filename

    def _discover_local_api_key(self) -> str | None:
        script = (
            "import asyncio\n"
            "from config.settings import get_langflow_api_key\n"
            "async def main():\n"
            "    key = await get_langflow_api_key()\n"
            "    print('LANGFLOW_API_KEY=' + (key or ''))\n"
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
            if not line.startswith("LANGFLOW_API_KEY="):
                continue
            value = line.split("=", 1)[1].strip()
            return value or None
        return None

    def _resolve_api_key(self) -> str:
        if self.settings.langflow_api_key:
            return self.settings.langflow_api_key
        if self._cached_api_key:
            return self._cached_api_key
        discovered = self._discover_local_api_key()
        if discovered:
            self._cached_api_key = discovered
            return discovered
        raise ValueError(
            "LANGFLOW_KEY is required. Export it, place it in .env, or run against the local "
            "openrag-backend container so the repo can auto-discover the key."
        )

    async def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        headers = {"x-api-key": self._resolve_api_key(), "Content-Type": "application/json"}
        existing_headers = kwargs.pop("headers", {})
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.request(
                method,
                f"{self.settings.langflow_url.rstrip('/')}{path}",
                headers={**headers, **existing_headers},
                **kwargs,
            )
        return response

    @staticmethod
    def _load_flow(path: Path) -> dict[str, object]:
        if not path.exists():
            raise FileNotFoundError(path)
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _write_flow(path: Path, flow: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(flow, indent=2), encoding="utf-8")

    async def patch_live_flow(self, flow: dict[str, object]) -> None:
        flow_id = str(flow.get("id") or "").strip()
        if not flow_id:
            raise ValueError("flow id is required to patch Langflow")
        response = await self._request("PATCH", f"/api/v1/flows/{flow_id}", json=flow)
        response.raise_for_status()

    async def diagnostics(self) -> LangflowFlowDiagnostics:
        agent_flow = self._load_flow(self.agent_flow_path)
        ingestion_flow = self._load_flow(self.ingestion_flow_path)
        has_api_key = False
        try:
            self._resolve_api_key()
        except Exception:
            has_api_key = False
        else:
            has_api_key = True

        ingestion_nodes = ingestion_flow.get("data", {}).get("nodes", [])
        chunk_size = None
        chunk_overlap = None
        for node in ingestion_nodes if isinstance(ingestion_nodes, list) else []:
            if node.get("data", {}).get("node", {}).get("display_name") != "Split Text":
                continue
            template = node.get("data", {}).get("node", {}).get("template", {})
            chunk_size = template.get("chunk_size", {}).get("value")
            chunk_overlap = template.get("chunk_overlap", {}).get("value")
            break

        return LangflowFlowDiagnostics(
            langflow_url=self.settings.langflow_url,
            has_api_key=has_api_key,
            flows_root=self.settings.langflow_flows_root.as_posix(),
            agent_flow_path=self.agent_flow_path.as_posix(),
            agent_flow_id=str(agent_flow.get("id") or ""),
            agent_flow_name=str(agent_flow.get("name") or ""),
            agent_flow_locked=bool(agent_flow.get("locked")),
            agent_flow_rerank_marker_present=agent_flow_has_backend_rerank(agent_flow),
            agent_flow_prompt_upgraded=prompt_template_is_upgraded(agent_flow),
            ingestion_flow_path=self.ingestion_flow_path.as_posix(),
            ingestion_flow_id=str(ingestion_flow.get("id") or ""),
            ingestion_chunk_size=chunk_size if isinstance(chunk_size, int) else None,
            ingestion_chunk_overlap=chunk_overlap if isinstance(chunk_overlap, int) else None,
            backend_reranking_enabled=self.settings.backend_rerank_enabled,
            backend_reranker_provider=self.settings.backend_rerank_provider,
            backend_reranker_model=self.settings.backend_rerank_model,
            backend_reranker_top_n=self.settings.backend_rerank_top_n,
        )

    async def upgrade_flows(
        self,
        *,
        output_dir: Path | None = None,
        patch_live: bool = True,
    ) -> FlowUpgradeResult:
        agent_flow = self._load_flow(self.agent_flow_path)
        ingestion_flow = self._load_flow(self.ingestion_flow_path)
        upgraded_agent_flow = upgrade_agent_flow(agent_flow, self.settings)
        upgraded_ingestion_flow = upgrade_ingestion_flow(ingestion_flow, self.settings)

        backup_root = output_dir or (
            self.settings.output_root / "langflow-flow-backups" / utc_now().strftime("%Y%m%dT%H%M%SZ")
        )
        backup_root.mkdir(parents=True, exist_ok=True)
        (backup_root / self.agent_flow_path.name).write_text(
            json.dumps(agent_flow, indent=2),
            encoding="utf-8",
        )
        (backup_root / self.ingestion_flow_path.name).write_text(
            json.dumps(ingestion_flow, indent=2),
            encoding="utf-8",
        )

        self._write_flow(self.agent_flow_path, upgraded_agent_flow)
        self._write_flow(self.ingestion_flow_path, upgraded_ingestion_flow)

        agent_live_patch_applied = False
        ingestion_live_patch_applied = False
        notes: list[str] = []

        if patch_live:
            try:
                await self.patch_live_flow(upgraded_agent_flow)
            except Exception as exc:
                notes.append(
                    f"Agent flow file was upgraded on disk but live Langflow patch failed: {exc}"
                )
            else:
                agent_live_patch_applied = True

            try:
                await self.patch_live_flow(upgraded_ingestion_flow)
            except Exception as exc:
                notes.append(
                    f"Ingestion flow file was upgraded on disk but live Langflow patch failed: {exc}"
                )
            else:
                ingestion_live_patch_applied = True

        if not patch_live:
            notes.append("Flow files were upgraded on disk. Restart Langflow to load the new definitions.")

        return FlowUpgradeResult(
            backup_dir=backup_root.as_posix(),
            agent_flow_path=self.agent_flow_path.as_posix(),
            ingestion_flow_path=self.ingestion_flow_path.as_posix(),
            agent_flow_live_patch_applied=agent_live_patch_applied,
            ingestion_flow_live_patch_applied=ingestion_live_patch_applied,
            agent_flow_marker_present=agent_flow_has_backend_rerank(upgraded_agent_flow),
            ingestion_settings_updated=True,
            notes=notes,
        )
