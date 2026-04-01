from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOG = logging.getLogger("openrag_override")
SRC_DIR = Path("/app/src")
_SUMMARY_NO_TOOLS_SENTINEL = "openrag_monitor_no_tools_summary_v1"

if SRC_DIR.is_dir():
    src_dir_str = str(SRC_DIR)
    if src_dir_str not in sys.path:
        sys.path.insert(0, src_dir_str)


_DEBUG_ROOT = Path(os.getenv("FASTPDF_OPENRAG_DEBUG_ROOT", "/app/openrag-documents/debug"))
_REDACTED_HEADER_KEYS = {"authorization", "x-api-key", "api-key"}
_REDACTED_HEADER_TOKENS = ("authorization", "api-key", "api_key", "key", "token", "secret", "jwt")


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


def _sanitize_headers(value: Any) -> Any:
    if not isinstance(value, dict):
        return _json_safe(value)
    sanitized: dict[str, Any] = {}
    for key, item in value.items():
        key_text = str(key)
        lowered = key_text.lower()
        if key_text.lower() in _REDACTED_HEADER_KEYS or any(token in lowered for token in _REDACTED_HEADER_TOKENS):
            sanitized[key_text] = "<redacted>"
        else:
            sanitized[key_text] = _json_safe(item)
    return sanitized


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
        LOG.warning("OpenRAG override: failed writing %s debug record: %s", stream_name, exc)


def _patch_providers_config() -> None:
    try:
        from config.config_manager import ProvidersConfig
    except Exception as exc:  # noqa: BLE001
        LOG.warning("OpenRAG override: failed to import ProvidersConfig: %s", exc)
        return

    if hasattr(ProvidersConfig, "any_configured"):
        return

    def any_configured(self: ProvidersConfig) -> bool:
        return bool(
            getattr(getattr(self, "openai", None), "configured", False)
            or getattr(getattr(self, "anthropic", None), "configured", False)
            or getattr(getattr(self, "watsonx", None), "configured", False)
            or getattr(getattr(self, "ollama", None), "configured", False)
        )

    ProvidersConfig.any_configured = any_configured  # type: ignore[attr-defined]
    LOG.info("OpenRAG override: patched ProvidersConfig.any_configured")


def _patch_flows_service() -> None:
    try:
        from services.flows_service import FlowsService
    except Exception as exc:  # noqa: BLE001
        LOG.warning("OpenRAG override: failed to import FlowsService: %s", exc)
        return

    original = getattr(FlowsService, "_enable_model_in_langflow", None)
    if original is None or getattr(original, "__openrag_override__", False):
        return

    async def _patched_enable_model_in_langflow(
        self: FlowsService,
        provider_name: str,
        model_value: str,
    ) -> None:
        if not str(model_value or "").strip():
            LOG.info(
                "OpenRAG override: skipping Langflow enable-model call because model_id is empty"
            )
            return
        await original(self, provider_name, model_value)

    _patched_enable_model_in_langflow.__openrag_override__ = True  # type: ignore[attr-defined]
    FlowsService._enable_model_in_langflow = _patched_enable_model_in_langflow
    LOG.info("OpenRAG override: patched FlowsService._enable_model_in_langflow")


def _should_disable_tools_for_summary(prompt: object) -> bool:
    text = str(prompt or "").strip().lower()
    if not text:
        return False
    if _SUMMARY_NO_TOOLS_SENTINEL in text:
        return True
    return (
        "you are an expert clinical document summarizer." in text
        and "use only the provided chunks and chronology." in text
        and "input_json:" in text
    )


def _patch_agent_async_response() -> None:
    try:
        import agent as agent_module
    except Exception as exc:  # noqa: BLE001
        LOG.warning("OpenRAG override: failed to import agent: %s", exc)
        return

    original = getattr(agent_module, "async_response", None)
    if original is None or getattr(original, "__openrag_override__", False):
        return

    logger = getattr(agent_module, "logger", LOG)

    async def _patched_async_response(
        client,
        prompt: str,
        model: str,
        extra_headers: dict | None = None,
        previous_response_id: str | None = None,
        log_prefix: str = "response",
    ):
        try:
            logger.info("User prompt received", prompt=prompt)

            request_params = {
                "model": model,
                "input": prompt,
                "stream": False,
                "include": ["tool_call.results"],
            }
            if previous_response_id is not None:
                request_params["previous_response_id"] = previous_response_id
            if extra_headers:
                request_params["extra_headers"] = extra_headers

            if "x-api-key" not in client.default_headers:
                if hasattr(client, "api_key") and extra_headers is not None:
                    extra_headers["x-api-key"] = client.api_key

            if _should_disable_tools_for_summary(prompt):
                request_params["tool_choice"] = "none"
                logger.info(
                    "OpenRAG override: forcing tool_choice=none for provided-chunks summary prompt",
                    log_prefix=log_prefix,
                )

            _append_debug_record(
                "openai_model_requests",
                {
                    "kind": "request",
                    "log_prefix": log_prefix,
                    "model": model,
                    "previous_response_id": previous_response_id,
                    "prompt": prompt,
                    "request_params": {
                        **_json_safe(request_params),
                        "extra_headers": _sanitize_headers(request_params.get("extra_headers")),
                    },
                    "client_default_headers": _sanitize_headers(getattr(client, "default_headers", None)),
                },
            )

            response = await client.responses.create(**request_params)
            response_dump = _json_safe(response)

            response_text = None
            try:
                response_text = response.output_text
            except Exception as output_text_error:  # noqa: BLE001
                logger.warning(
                    "Failed reading response.output_text directly; falling back to output parsing",
                    error=str(output_text_error),
                )

            if not response_text:
                try:
                    output_items = getattr(response, "output", None)
                    if isinstance(output_items, list):
                        text_parts = []
                        for item in output_items:
                            item_content = getattr(item, "content", None)
                            if item_content is None and isinstance(item, dict):
                                item_content = item.get("content")

                            if isinstance(item_content, list):
                                for block in item_content:
                                    block_text = getattr(block, "text", None)
                                    if block_text is None and isinstance(block, dict):
                                        block_text = block.get("text")
                                    if block_text:
                                        text_parts.append(str(block_text))

                        if text_parts:
                            response_text = "".join(text_parts).strip()
                except Exception as output_parse_error:  # noqa: BLE001
                    logger.warning(
                        "Failed parsing response.output fallback",
                        error=str(output_parse_error),
                    )

            if response_text:
                logger.info("Response generated", log_prefix=log_prefix, response=response_text)
                response_id = getattr(response, "id", None) or getattr(response, "response_id", None)
                _append_debug_record(
                    "openai_model_responses",
                    {
                        "kind": "response",
                        "log_prefix": log_prefix,
                        "model": model,
                        "response_id": response_id,
                        "response_text": response_text,
                        "response": response_dump,
                    },
                )
                return response_text, response_id, response

            _append_debug_record(
                "openai_model_responses",
                {
                    "kind": "response_missing_text",
                    "log_prefix": log_prefix,
                    "model": model,
                    "response": response_dump,
                },
            )
            msg = "Nudge response missing output_text"
            error = getattr(response, "error", None)
            if error:
                error_msg = getattr(error, "message", None)
                if error_msg:
                    msg = error_msg
            raise ValueError(msg)
        except Exception as exc:  # noqa: BLE001
            logger.error("Exception in non-streaming response", error=str(exc))
            import traceback

            traceback.print_exc()
            raise

    _patched_async_response.__openrag_override__ = True  # type: ignore[attr-defined]
    agent_module.async_response = _patched_async_response
    LOG.info("OpenRAG override: patched agent.async_response for no-tools summary prompts")


_patch_providers_config()
_patch_flows_service()
_patch_agent_async_response()
