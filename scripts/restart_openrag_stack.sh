#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENRAG_COMPOSE_FILE="${OPENRAG_COMPOSE_FILE:-$HOME/.openrag/tui/docker-compose.yml}"
OPENRAG_ENV_FILE="${OPENRAG_ENV_FILE:-$HOME/.openrag/tui/.env}"
OPENRAG_OVERRIDE_FILE="${OPENRAG_OVERRIDE_FILE:-$REPO_ROOT/docker/openrag-override.yml}"
DOCLING_BIN="${DOCLING_BIN:-$HOME/.openrag/docling-venv/bin/docling-serve}"
DOCLING_LOG="${DOCLING_LOG:-$HOME/.openrag/docling-serve.log}"
DOCLING_PID_FILE="${DOCLING_PID_FILE:-$HOME/.openrag/tui/.docling.pid}"
DOCLING_HOST="${DOCLING_HOST:-0.0.0.0}"
DOCLING_PORT="${DOCLING_PORT:-5001}"
DOCLING_HEALTH_URL="${DOCLING_HEALTH_URL:-http://127.0.0.1:${DOCLING_PORT}/health}"

export FASTPDF_OPENRAG_NATIVE_ROOT="${FASTPDF_OPENRAG_NATIVE_ROOT:-$REPO_ROOT}"
export FASTPDF_OPENRAG_BACKEND_RERANK_ENABLED="${FASTPDF_OPENRAG_BACKEND_RERANK_ENABLED:-true}"
export FASTPDF_OPENRAG_BACKEND_RERANK_PROVIDER="${FASTPDF_OPENRAG_BACKEND_RERANK_PROVIDER:-cross_encoder}"
export FASTPDF_OPENRAG_BACKEND_RERANK_MODEL="${FASTPDF_OPENRAG_BACKEND_RERANK_MODEL:-cross-encoder/ms-marco-MiniLM-L-6-v2}"
export FASTPDF_OPENRAG_BACKEND_RERANK_TOP_N="${FASTPDF_OPENRAG_BACKEND_RERANK_TOP_N:-8}"
export FASTPDF_OPENRAG_BACKEND_SEARCH_RERANK_ENABLED="${FASTPDF_OPENRAG_BACKEND_SEARCH_RERANK_ENABLED:-true}"
export FASTPDF_OPENRAG_BACKEND_SEARCH_RERANK_CANDIDATE_LIMIT="${FASTPDF_OPENRAG_BACKEND_SEARCH_RERANK_CANDIDATE_LIMIT:-24}"
export FASTPDF_OPENRAG_RESTART_UI="${FASTPDF_OPENRAG_RESTART_UI:-true}"
export FASTPDF_OPENRAG_UI_HOST="${FASTPDF_OPENRAG_UI_HOST:-127.0.0.1}"
export FASTPDF_OPENRAG_UI_PORT="${FASTPDF_OPENRAG_UI_PORT:-8077}"
FASTPDF_UI_LOG="${FASTPDF_UI_LOG:-$REPO_ROOT/outputs/serve-ui.log}"
FASTPDF_UI_PID_FILE="${FASTPDF_UI_PID_FILE:-$REPO_ROOT/outputs/serve-ui.pid}"

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

  if [[ -f "$DOCLING_PID_FILE" ]]; then
    kill "$(cat "$DOCLING_PID_FILE")" >/dev/null 2>&1 || true
  fi
  pkill -f "docling-serve run --host .* --port ${DOCLING_PORT}" >/dev/null 2>&1 || true
  nohup "$DOCLING_BIN" run --host "$DOCLING_HOST" --port "$DOCLING_PORT" >>"$DOCLING_LOG" 2>&1 &
  echo $! >"$DOCLING_PID_FILE"
  printf '[ok] restarted docling: pid=%s host=%s port=%s log=%s\n' \
    "$(cat "$DOCLING_PID_FILE")" "$DOCLING_HOST" "$DOCLING_PORT" "$DOCLING_LOG"
}

restart_fastpdf_ui() {
  local restart_ui
  restart_ui="$(printf '%s' "$FASTPDF_OPENRAG_RESTART_UI" | tr '[:upper:]' '[:lower:]')"
  if [[ "$restart_ui" == "0" || "$restart_ui" == "false" || "$restart_ui" == "no" || "$restart_ui" == "off" ]]; then
    printf '[info] skipping fastpdf ui restart\n'
    return 0
  fi

  mkdir -p "$(dirname "$FASTPDF_UI_LOG")"
  pkill -f 'fastpdf-openrag-native serve-ui' >/dev/null 2>&1 || true

  (
    cd "$REPO_ROOT"
    nohup uv run fastpdf-openrag-native serve-ui \
      --host "$FASTPDF_OPENRAG_UI_HOST" \
      --port "$FASTPDF_OPENRAG_UI_PORT" >>"$FASTPDF_UI_LOG" 2>&1 &
    echo $! >"$FASTPDF_UI_PID_FILE"
  )

  printf '[ok] restarted fastpdf ui: pid=%s log=%s\n' "$(cat "$FASTPDF_UI_PID_FILE")" "$FASTPDF_UI_LOG"
  wait_for_http "http://$FASTPDF_OPENRAG_UI_HOST:$FASTPDF_OPENRAG_UI_PORT/" "FastPDF UI" 60 1
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

wait_for_http "$DOCLING_HEALTH_URL" "Docling"
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

restart_fastpdf_ui
