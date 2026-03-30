from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path
from uuid import uuid4


HOTFIX_PATH = (
    Path(__file__).resolve().parents[1]
    / "openrag-hotfixes"
    / "api_v1_chat.py"
)
SEARCH_HOTFIX_PATH = (
    Path(__file__).resolve().parents[1]
    / "openrag-hotfixes"
    / "api_v1_search.py"
)


def _load_chat_hotfix(monkeypatch, history_messages):
    logger = types.SimpleNamespace(
        warning=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
    )

    utils_pkg = types.ModuleType("utils")
    utils_pkg.__path__ = []
    logging_config = types.ModuleType("utils.logging_config")
    logging_config.get_logger = lambda _name: logger
    monkeypatch.setitem(sys.modules, "utils", utils_pkg)
    monkeypatch.setitem(sys.modules, "utils.logging_config", logging_config)

    def _noop(*_args, **_kwargs):
        return None

    auth_context = types.ModuleType("auth_context")
    auth_context.set_search_filters = _noop
    auth_context.set_search_limit = _noop
    auth_context.set_score_threshold = _noop
    auth_context.set_auth_context = _noop
    monkeypatch.setitem(sys.modules, "auth_context", auth_context)

    dependencies = types.ModuleType("dependencies")
    dependencies.get_chat_service = _noop
    dependencies.get_search_service = _noop
    dependencies.get_session_manager = _noop
    dependencies.get_api_key_user_async = _noop
    monkeypatch.setitem(sys.modules, "dependencies", dependencies)

    session_manager = types.ModuleType("session_manager")
    session_manager.User = type("User", (), {})
    monkeypatch.setitem(sys.modules, "session_manager", session_manager)

    services_pkg = types.ModuleType("services")
    services_pkg.__path__ = []

    class FakeHistoryService:
        async def get_session_messages(self, _user_id, _chat_id):
            return history_messages

    history_module = types.ModuleType("services.langflow_history_service")
    history_module.langflow_history_service = FakeHistoryService()
    monkeypatch.setitem(sys.modules, "services", services_pkg)
    monkeypatch.setitem(sys.modules, "services.langflow_history_service", history_module)

    module_name = f"test_api_v1_chat_{uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, HOTFIX_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def _load_search_hotfix(monkeypatch):
    logger = types.SimpleNamespace(
        debug=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
    )

    utils_pkg = types.ModuleType("utils")
    utils_pkg.__path__ = []
    logging_config = types.ModuleType("utils.logging_config")
    logging_config.get_logger = lambda _name: logger
    opensearch_utils = types.ModuleType("utils.opensearch_utils")
    opensearch_utils.OpenSearchDiskSpaceError = type("OpenSearchDiskSpaceError", (Exception,), {})
    opensearch_utils.DISK_SPACE_ERROR_MESSAGE = "disk space blocked"
    monkeypatch.setitem(sys.modules, "utils", utils_pkg)
    monkeypatch.setitem(sys.modules, "utils.logging_config", logging_config)
    monkeypatch.setitem(sys.modules, "utils.opensearch_utils", opensearch_utils)

    def _noop(*_args, **_kwargs):
        return None

    dependencies = types.ModuleType("dependencies")
    dependencies.get_search_service = _noop
    dependencies.get_api_key_user_async = _noop
    monkeypatch.setitem(sys.modules, "dependencies", dependencies)

    session_manager = types.ModuleType("session_manager")
    session_manager.User = type("User", (), {})
    monkeypatch.setitem(sys.modules, "session_manager", session_manager)

    module_name = f"test_api_v1_search_{uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, SEARCH_HOTFIX_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def test_extract_sources_tolerates_none_results(monkeypatch):
    module = _load_chat_hotfix(monkeypatch, history_messages=[])

    assert module._extract_sources(None) == []
    assert module._extract_sources({"results": None}) == []
    assert module._extract_sources(
        {
            "results": {
                "data": {
                    "filename": "doc.html",
                    "text": "example",
                    "score": 1.25,
                    "page": 3,
                }
            }
        }
    ) == [
        {
            "filename": "doc.html",
            "text": "example",
            "score": 1.25,
            "page": 3,
            "mimetype": None,
        }
    ]


def test_recover_history_context_skips_none_items_and_results(monkeypatch):
    module = _load_chat_hotfix(
        monkeypatch,
        history_messages=[
            {
                "role": "assistant",
                "chunks": [
                    {"item": None},
                    {"item": {"inputs": {"search_query": "ignored"}, "results": None}},
                    {
                        "item": {
                            "inputs": {"search_query": "patient visit"},
                            "results": [
                                {
                                    "data": {
                                        "filename": "doc.html",
                                        "text": "Patient visit summary",
                                        "score": 2.75,
                                        "page": 1,
                                        "mimetype": "text/html",
                                    }
                                }
                            ],
                        }
                    },
                ],
            }
        ],
    )

    sources, queries = asyncio.run(module._recover_history_context("user-1", "chat-1"))

    assert queries == ["ignored", "patient visit"]
    assert sources == [
        {
            "filename": "doc.html",
            "text": "Patient visit summary",
            "score": 2.75,
            "page": 1,
            "mimetype": "text/html",
        }
    ]


def test_search_hotfix_reranks_public_results_and_expands_candidate_pool(monkeypatch):
    module = _load_search_hotfix(monkeypatch)
    monkeypatch.setenv("FASTPDF_OPENRAG_BACKEND_SEARCH_RERANK_ENABLED", "true")
    monkeypatch.setenv("FASTPDF_OPENRAG_BACKEND_SEARCH_RERANK_CANDIDATE_LIMIT", "5")

    class FakeSearchService:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        async def search(self, query, user_id=None, jwt_token=None, filters=None, limit=10, score_threshold=0):  # noqa: ANN001
            self.calls.append(
                {
                    "query": query,
                    "user_id": user_id,
                    "filters": filters,
                    "limit": limit,
                    "score_threshold": score_threshold,
                }
            )
            return {
                "results": [
                    {
                        "filename": "header.md",
                        "text": "demographic header only",
                        "score": 0.95,
                        "page": 1,
                        "mimetype": "text/markdown",
                    },
                    {
                        "filename": "procedure.md",
                        "text": "supported summary grounded operative detail",
                        "score": 0.35,
                        "page": 1,
                        "mimetype": "text/markdown",
                    },
                ]
            }

    search_service = FakeSearchService()
    user = types.SimpleNamespace(user_id="user-1")
    body = module.SearchV1Body(query="operative detail", limit=1, filters={"data_sources": ["doc.md"]})

    response = asyncio.run(
        module.search_endpoint(body, search_service=search_service, user=user)
    )
    payload = json.loads(response.body)

    assert search_service.calls[0]["limit"] == 5
    assert payload["results"][0]["filename"] == "procedure.md"
    assert payload["results"][0]["base_score"] == 0.35
    assert payload["results"][0]["retrieval_rank"] == 1
