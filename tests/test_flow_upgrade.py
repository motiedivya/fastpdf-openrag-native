from __future__ import annotations

import asyncio
import json

from fastpdf_openrag_native.flow_upgrade import (
    AGENT_FLOW_RERANK_MARKER,
    agent_flow_has_backend_rerank,
    patch_opensearch_component_code,
    prompt_template_is_upgraded,
)
from fastpdf_openrag_native.langflow import LangflowGateway
from fastpdf_openrag_native.settings import AppSettings


def _minimal_agent_flow() -> dict[str, object]:
    return {
        "id": "agent-flow-id",
        "name": "OpenRAG OpenSearch Agent Flow",
        "locked": True,
        "data": {
            "nodes": [
                {
                    "data": {
                        "node": {
                            "display_name": "Prompt Template",
                            "template": {
                                "template": {
                                    "value": "This is Knowledge filter - use it as a context of what to search on the database, unless it's empty: {filter}\n\nChat input: {input}"
                                }
                            },
                        }
                    }
                },
                {
                    "data": {
                        "node": {
                            "display_name": "OpenSearch (Multi-Model Multi-Embedding)",
                            "template": {
                                "number_of_results": {"value": 10},
                                "code": {
                                    "value": (
                                        "REQUEST_TIMEOUT = 60\n"
                                        "MAX_RETRIES = 5\n\n"
                                        "def search(self, query):\n"
                                        "    hits = []\n"
                                        "    return [\n"
                                        "        {\n"
                                        '            "page_content": hit["_source"].get("text", ""),\n'
                                        '            "metadata": {k: v for k, v in hit["_source"].items() if k != "text"},\n'
                                        '            "score": hit.get("_score"),\n'
                                        "        }\n"
                                        "        for hit in hits\n"
                                        "    ]\n\n"
                                        "def search_documents(self) -> list[Data]:\n"
                                        "    search_query = (self.search_query or '').strip()\n"
                                        "    raw = self.search(search_query)\n"
                                        '    return [Data(text=hit["page_content"], **hit["metadata"]) for hit in raw]\n'
                                    )
                                },
                            },
                        }
                    }
                },
            ]
        },
    }


def _minimal_ingestion_flow() -> dict[str, object]:
    return {
        "id": "ingestion-flow-id",
        "name": "OpenSearch Ingestion Flow",
        "locked": True,
        "data": {
            "nodes": [
                {
                    "data": {
                        "node": {
                            "display_name": "Split Text",
                            "template": {
                                "chunk_size": {"value": 1000},
                                "chunk_overlap": {"value": 200},
                                "separator": {"value": "\\n"},
                            },
                        }
                    }
                }
            ]
        },
    }


def test_patch_opensearch_component_code_injects_backend_rerank_logic() -> None:
    code = _minimal_agent_flow()["data"]["nodes"][1]["data"]["node"]["template"]["code"]["value"]  # type: ignore[index]
    settings = AppSettings(
        backend_rerank_enabled=True,
        backend_rerank_provider="cross_encoder",
        backend_rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        backend_rerank_top_n=8,
    )

    patched = patch_opensearch_component_code(code, settings)

    assert AGENT_FLOW_RERANK_MARKER in patched
    assert '_fastpdf_backend_rerank_hits(query=query, hits=hits)' in patched
    assert 'score=hit.get("score")' in patched
    assert 'FASTPDF_OPENRAG_BACKEND_RERANK_ENABLED' in patched
    assert 'FASTPDF_OPENRAG_BACKEND_RERANK_PROVIDER' in patched


def test_langflow_gateway_upgrade_flows_updates_mounted_files(tmp_path, monkeypatch) -> None:
    flows_root = tmp_path / "flows"
    flows_root.mkdir()
    agent_path = flows_root / "openrag_agent.json"
    ingestion_path = flows_root / "ingestion_flow.json"
    agent_path.write_text(json.dumps(_minimal_agent_flow(), indent=2), encoding="utf-8")
    ingestion_path.write_text(json.dumps(_minimal_ingestion_flow(), indent=2), encoding="utf-8")

    async def _fake_patch_live_flow(self, flow):  # noqa: ANN001
        return None

    monkeypatch.setattr(LangflowGateway, "patch_live_flow", _fake_patch_live_flow)

    settings = AppSettings(
        langflow_api_key="lf_test_key",
        langflow_flows_root=flows_root,
        output_root=tmp_path / "outputs",
        default_chunk_size=1200,
        default_chunk_overlap=150,
        backend_rerank_enabled=True,
        backend_rerank_candidate_limit=16,
    )
    gateway = LangflowGateway(settings)

    result = asyncio.run(gateway.upgrade_flows(patch_live=True))
    diagnostics = asyncio.run(gateway.diagnostics())

    upgraded_agent = json.loads(agent_path.read_text(encoding="utf-8"))
    upgraded_ingestion = json.loads(ingestion_path.read_text(encoding="utf-8"))

    assert result.agent_flow_live_patch_applied is True
    assert result.ingestion_flow_live_patch_applied is True
    assert agent_flow_has_backend_rerank(upgraded_agent) is True
    assert prompt_template_is_upgraded(upgraded_agent) is True
    split_template = upgraded_ingestion["data"]["nodes"][0]["data"]["node"]["template"]
    assert split_template["chunk_size"]["value"] == 1200
    assert split_template["chunk_overlap"]["value"] == 150
    assert split_template["separator"]["value"] == "\\n\\n"
    assert diagnostics.agent_flow_rerank_marker_present is True
    assert diagnostics.ingestion_chunk_size == 1200
    assert diagnostics.backend_reranker_provider == "cross_encoder"
