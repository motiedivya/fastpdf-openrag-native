#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PACKAGE_ROOT/../.." && pwd)"
ENV_FILE="${OPENRAG_ENV_FILE:-$PACKAGE_ROOT/.env}"
COMPOSE_FILE="${OPENRAG_COMPOSE_FILE:-$PACKAGE_ROOT/docker-compose.yml}"

if [[ ! -f "$ENV_FILE" ]]; then
  printf '[error] env file not found: %s
' "$ENV_FILE" >&2
  printf '[hint] copy %s to %s and fill in the required values first
' "$PACKAGE_ROOT/.env.example" "$ENV_FILE" >&2
  exit 1
fi

mkdir -p "$PACKAGE_ROOT/logs"

set -a
source "$ENV_FILE"
set +a

DOCLING_MANAGED="${DOCLING_MANAGED:-false}"
DOCLING_PORT="${DOCLING_PORT:-5001}"
DOCLING_BIN="${DOCLING_BIN:-$HOME/.openrag/docling-venv/bin/docling-serve}"
DOCLING_HOST="${DOCLING_HOST:-0.0.0.0}"
DOCLING_BIND_PORT="${DOCLING_BIND_PORT:-5001}"
DOCLING_HEALTH_URL="${DOCLING_HEALTH_URL:-http://127.0.0.1:5001/health}"
DOCLING_PID_FILE="${DOCLING_PID_FILE:-$PACKAGE_ROOT/logs/docling.pid}"
DOCLING_LOG_FILE="${DOCLING_LOG_FILE:-$PACKAGE_ROOT/logs/docling.log}"

export FASTPDF_OPENRAG_NATIVE_ROOT="${FASTPDF_OPENRAG_NATIVE_ROOT:-$REPO_ROOT}"
export FASTPDF_OPENRAG_BACKEND_RERANK_ENABLED="${FASTPDF_OPENRAG_BACKEND_RERANK_ENABLED:-true}"
export FASTPDF_OPENRAG_BACKEND_RERANK_PROVIDER="${FASTPDF_OPENRAG_BACKEND_RERANK_PROVIDER:-cross_encoder}"
export FASTPDF_OPENRAG_BACKEND_RERANK_MODEL="${FASTPDF_OPENRAG_BACKEND_RERANK_MODEL:-cross-encoder/ms-marco-MiniLM-L-6-v2}"
export FASTPDF_OPENRAG_BACKEND_RERANK_TOP_N="${FASTPDF_OPENRAG_BACKEND_RERANK_TOP_N:-8}"
export FASTPDF_OPENRAG_BACKEND_SEARCH_RERANK_ENABLED="${FASTPDF_OPENRAG_BACKEND_SEARCH_RERANK_ENABLED:-true}"
export FASTPDF_OPENRAG_BACKEND_SEARCH_RERANK_CANDIDATE_LIMIT="${FASTPDF_OPENRAG_BACKEND_SEARCH_RERANK_CANDIDATE_LIMIT:-24}"

compose() {
  docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" "$@"
}

wait_for_http() {
  local url="$1"
  local label="$2"
  local attempts="${3:-90}"
  local sleep_seconds="${4:-2}"
  local i
  for ((i = 1; i <= attempts; i += 1)); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      printf '[ok] %s: %s
' "$label" "$url"
      return 0
    fi
    sleep "$sleep_seconds"
  done
  printf '[error] %s did not become healthy: %s
' "$label" "$url" >&2
  return 1
}

validate_native_root() {
  local native_root="$FASTPDF_OPENRAG_NATIVE_ROOT"
  local missing=0
  local required=(
    "$native_root/openrag-hotfixes/api_v1_chat.py"
    "$native_root/openrag-hotfixes/api_v1_search.py"
    "$native_root/openrag-hotfixes/session_manager.py"
    "$native_root/scripts/restart_openrag_stack.sh"
    "$native_root/pyproject.toml"
  )

  if [[ ! -d "$native_root" ]]; then
    printf '[error] FASTPDF_OPENRAG_NATIVE_ROOT does not exist: %s
' "$native_root" >&2
    printf '[hint] It must point to the fastpdf-openrag-native repo root, not fastpdf/openrag_native_bridge.
' >&2
    return 1
  fi

  for path in "${required[@]}"; do
    if [[ ! -f "$path" ]]; then
      printf '[error] missing required native file: %s
' "$path" >&2
      missing=1
    fi
  done

  if [[ "$native_root" == *"/openrag_native_bridge"* ]]; then
    printf '[error] FASTPDF_OPENRAG_NATIVE_ROOT is pointing at openrag_native_bridge: %s
' "$native_root" >&2
    printf '[hint] Use the fastpdf-openrag-native checkout path instead.
' >&2
    missing=1
  fi

  if (( missing != 0 )); then
    printf '[hint] Example: FASTPDF_OPENRAG_NATIVE_ROOT=/srv/fastpdf-openrag-native bash ./scripts/restart_stack.sh
' >&2
    return 1
  fi
}

restart_docling() {
  local managed
  managed="$(printf '%s' "$DOCLING_MANAGED" | tr '[:upper:]' '[:lower:]')"
  if [[ "$managed" == "0" || "$managed" == "false" || "$managed" == "no" || "$managed" == "off" ]]; then
    printf '[info] skipping managed docling restart
'
    return 0
  fi

  if [[ ! -x "$DOCLING_BIN" ]]; then
    printf '[error] docling binary not found: %s
' "$DOCLING_BIN" >&2
    return 1
  fi

  if [[ -f "$DOCLING_PID_FILE" ]]; then
    kill "$(cat "$DOCLING_PID_FILE")" >/dev/null 2>&1 || true
  fi
  pkill -f "docling-serve run --host .* --port ${DOCLING_BIND_PORT}" >/dev/null 2>&1 || true
  nohup "$DOCLING_BIN" run --host "$DOCLING_HOST" --port "$DOCLING_BIND_PORT" >>"$DOCLING_LOG_FILE" 2>&1 &
  echo $! > "$DOCLING_PID_FILE"
  wait_for_http "$DOCLING_HEALTH_URL" "Docling" 60 1
}

printf '[info] repo=%s
' "$REPO_ROOT"
printf '[info] compose=%s
' "$COMPOSE_FILE"
printf '[info] env=%s
' "$ENV_FILE"
printf '[info] FASTPDF_OPENRAG_NATIVE_ROOT=%s
' "$FASTPDF_OPENRAG_NATIVE_ROOT"

validate_native_root
restart_docling
compose up -d --remove-orphans docling opensearch dashboards langflow openrag-backend openrag-frontend
wait_for_http "http://127.0.0.1:${DOCLING_PORT:-5001}/health" "Docling" 90 2
wait_for_http "http://127.0.0.1:${LANGFLOW_PORT:-7860}/health" "Langflow" 90 2
wait_for_http "http://127.0.0.1:${FRONTEND_PORT:-3000}/" "OpenRAG Frontend" 90 2

(
  cd "$REPO_ROOT"
  uv run fastpdf-openrag-native upgrade-openrag-flows
  uv run python - <<'PY'
import asyncio
from fastpdf_openrag_native.openrag import OpenRAGGateway
from fastpdf_openrag_native.settings import get_settings

async def main() -> None:
    gateway = OpenRAGGateway(get_settings())
    await gateway.apply_recommended_settings()
    print('applied_recommended_settings=true')

asyncio.run(main())
PY
  uv run fastpdf-openrag-native diagnose-stack
)

printf '[done] stack restarted successfully
'
