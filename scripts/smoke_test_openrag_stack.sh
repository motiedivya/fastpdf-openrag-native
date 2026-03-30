#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENRAG_URL="${OPENRAG_URL:-$(grep -E '^OPENRAG_URL=' "$REPO_ROOT/.env" 2>/dev/null | cut -d= -f2- || true)}"
OPENRAG_API_KEY="${OPENRAG_API_KEY:-$(grep -E '^OPENRAG_API_KEY=' "$REPO_ROOT/.env" 2>/dev/null | cut -d= -f2- || true)}"
WITH_CHAT=0

if [[ "${1:-}" == "--with-chat" ]]; then
  WITH_CHAT=1
  shift || true
fi

SEARCH_QUERY="${SEARCH_QUERY:-${1:-patient}}"
SEARCH_LIMIT="${SEARCH_LIMIT:-5}"

if [[ -z "$OPENRAG_URL" || -z "$OPENRAG_API_KEY" ]]; then
  printf '[error] OPENRAG_URL and OPENRAG_API_KEY must be set or present in %s/.env\n' "$REPO_ROOT" >&2
  exit 1
fi

export OPENRAG_URL
export OPENRAG_API_KEY
export SEARCH_QUERY
export SEARCH_LIMIT

(
  cd "$REPO_ROOT"
  uv run fastpdf-openrag-native health
  uv run fastpdf-openrag-native diagnose-stack
)

(
  cd "$REPO_ROOT"
  uv run python - <<'PY'
import json
import os

import httpx

base_url = os.environ["OPENRAG_URL"].rstrip("/")
api_key = os.environ["OPENRAG_API_KEY"]
query = os.environ["SEARCH_QUERY"]
limit = int(os.environ["SEARCH_LIMIT"])

response = httpx.post(
    f"{base_url}/api/v1/search",
    headers={
        "Content-Type": "application/json",
        "X-API-Key": api_key,
    },
    json={"query": query, "limit": limit},
    timeout=60.0,
)
payload = response.json()
print(json.dumps(payload, indent=2))

results = payload.get("results", [])
if results:
    first = results[0]
    print(f"top_filename={first.get('filename')}")
    print(f"top_score={first.get('score')}")
    print(f"top_base_score={first.get('base_score')}")
    print(f"top_rerank_score={first.get('rerank_score')}")
    print(f"top_retrieval_rank={first.get('retrieval_rank')}")
else:
    print("top_filename=")
PY
)

if [[ "$WITH_CHAT" -eq 1 ]]; then
  (
    cd "$REPO_ROOT"
    uv run python - <<'PY'
import json
import os

import httpx

base_url = os.environ["OPENRAG_URL"].rstrip("/")
api_key = os.environ["OPENRAG_API_KEY"]
message = os.environ.get(
    "CHAT_MESSAGE",
    "What is the patient name and reason for visit in the indexed document?",
)

response = httpx.post(
    f"{base_url}/api/v1/chat",
    headers={
        "Content-Type": "application/json",
        "X-API-Key": api_key,
    },
    json={"message": message, "stream": False, "limit": 5},
    timeout=180.0,
)
payload = response.json()
print(json.dumps(payload, indent=2))
PY
  )
fi
