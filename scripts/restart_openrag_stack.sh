#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENRAG_COMPOSE_FILE="${OPENRAG_COMPOSE_FILE:-$HOME/.openrag/tui/docker-compose.yml}"
OPENRAG_ENV_FILE="${OPENRAG_ENV_FILE:-$HOME/.openrag/tui/.env}"
OPENRAG_OVERRIDE_FILE="${OPENRAG_OVERRIDE_FILE:-$REPO_ROOT/docker/openrag-override.yml}"
DOCLING_BIN="${DOCLING_BIN:-$HOME/.openrag/docling-venv/bin/docling-serve}"
DOCLING_LOG="${DOCLING_LOG:-$HOME/.openrag/docling-serve.log}"
DOCLING_PID_FILE="${DOCLING_PID_FILE:-$HOME/.openrag/tui/.docling.pid}"

export FASTPDF_OPENRAG_NATIVE_ROOT="${FASTPDF_OPENRAG_NATIVE_ROOT:-$REPO_ROOT}"
export FASTPDF_OPENRAG_BACKEND_RERANK_ENABLED="${FASTPDF_OPENRAG_BACKEND_RERANK_ENABLED:-true}"
export FASTPDF_OPENRAG_BACKEND_RERANK_PROVIDER="${FASTPDF_OPENRAG_BACKEND_RERANK_PROVIDER:-cross_encoder}"
export FASTPDF_OPENRAG_BACKEND_RERANK_MODEL="${FASTPDF_OPENRAG_BACKEND_RERANK_MODEL:-cross-encoder/ms-marco-MiniLM-L-6-v2}"
export FASTPDF_OPENRAG_BACKEND_RERANK_TOP_N="${FASTPDF_OPENRAG_BACKEND_RERANK_TOP_N:-8}"
export FASTPDF_OPENRAG_BACKEND_SEARCH_RERANK_ENABLED="${FASTPDF_OPENRAG_BACKEND_SEARCH_RERANK_ENABLED:-true}"
export FASTPDF_OPENRAG_BACKEND_SEARCH_RERANK_CANDIDATE_LIMIT="${FASTPDF_OPENRAG_BACKEND_SEARCH_RERANK_CANDIDATE_LIMIT:-24}"

if [[ ! -f "$OPENRAG_COMPOSE_FILE" ]]; then
  printf '[error] OpenRAG compose file not found: %s\n' "$OPENRAG_COMPOSE_FILE" >&2
  exit 1
fi

if [[ ! -f "$OPENRAG_OVERRIDE_FILE" ]]; then
  printf '[error] OpenRAG override file not found: %s\n' "$OPENRAG_OVERRIDE_FILE" >&2
  exit 1
fi

if [[ ! -f "$OPENRAG_ENV_FILE" ]]; then
  printf '[error] OpenRAG env file not found: %s\n' "$OPENRAG_ENV_FILE" >&2
  exit 1
fi

compose() {
  docker compose \
    -f "$OPENRAG_COMPOSE_FILE" \
    -f "$OPENRAG_OVERRIDE_FILE" \
    --env-file "$OPENRAG_ENV_FILE" \
    "$@"
}

wait_for_http() {
  local url="$1"
  local label="$2"
  local attempts="${3:-90}"
  local sleep_seconds="${4:-2}"
  local i

  for ((i = 1; i <= attempts; i += 1)); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      printf '[ok] %s: %s\n' "$label" "$url"
      return 0
    fi
    sleep "$sleep_seconds"
  done

  printf '[error] %s did not become healthy: %s\n' "$label" "$url" >&2
  return 1
}

restart_docling() {
  if [[ ! -x "$DOCLING_BIN" ]]; then
    printf '[warn] Docling binary not found, skipping restart: %s\n' "$DOCLING_BIN"
    return 0
  fi

  pkill -f 'docling-serve run --host 127.0.0.1 --port 5001' >/dev/null 2>&1 || true
  nohup "$DOCLING_BIN" run --host 127.0.0.1 --port 5001 >>"$DOCLING_LOG" 2>&1 &
  echo $! >"$DOCLING_PID_FILE"
  printf '[ok] restarted docling: pid=%s log=%s\n' "$(cat "$DOCLING_PID_FILE")" "$DOCLING_LOG"
}

apply_recommended_settings() {
  (
    cd "$REPO_ROOT"
    uv run python - <<'PY'
import asyncio
from fastpdf_openrag_native.openrag import OpenRAGGateway
from fastpdf_openrag_native.settings import get_settings

async def main() -> None:
    gateway = OpenRAGGateway(get_settings())
    await gateway.apply_recommended_settings()
    print("applied_recommended_settings=true")

asyncio.run(main())
PY
  )
}

printf '[info] repo=%s\n' "$REPO_ROOT"
printf '[info] compose=%s\n' "$OPENRAG_COMPOSE_FILE"
printf '[info] override=%s\n' "$OPENRAG_OVERRIDE_FILE"

restart_docling

compose up -d --no-build --force-recreate opensearch dashboards langflow openrag-backend openrag-frontend

wait_for_http "http://127.0.0.1:5001/health" "Docling"
wait_for_http "http://127.0.0.1:3000/api/settings" "OpenRAG"

(
  cd "$REPO_ROOT"
  uv run fastpdf-openrag-native upgrade-openrag-flows
)

apply_recommended_settings

(
  cd "$REPO_ROOT"
  uv run fastpdf-openrag-native diagnose-stack
)
