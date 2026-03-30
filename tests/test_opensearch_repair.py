from __future__ import annotations

import pytest

from fastpdf_openrag_native.opensearch import OpenSearchInspector


def test_extract_last_json_line_ignores_log_noise() -> None:
    payload = OpenSearchInspector._extract_last_json_line(
        "\n".join(
            [
                "2026-03-29 19:22:54 [debug] Loaded configuration",
                '{"embedding_provider": "openai", "embedding_model": "text-embedding-3-large"}',
            ]
        )
    )

    assert payload["embedding_provider"] == "openai"
    assert payload["embedding_model"] == "text-embedding-3-large"


def test_extract_last_json_line_raises_when_missing_json() -> None:
    with pytest.raises(ValueError):
        OpenSearchInspector._extract_last_json_line("debug only\nstill no json")
