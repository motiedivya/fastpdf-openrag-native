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

export SELECTED_EMBEDDING_MODEL="${SELECTED_EMBEDDING_MODEL:-text-embedding-3-large}"
OPENRAG_CONFIG_FILE="$(resolve_path "${OPENRAG_CONFIG_PATH:-./state/config}")/config.yaml"


export SELECTED_EMBEDDING_MODEL="${SELECTED_EMBEDDING_MODEL:-text-embedding-3-large}"

sync_openrag_runtime_config() {
  local config_file="$1"
  if [[ ! -f "$config_file" ]]; then
    printf '[warn] OpenRAG config file not found, skipping provider sync: %s\n' "$config_file" >&2
    return 0
  fi

  python3 - "$OPENRAG_ENV_FILE" "$config_file" <<'PY_SYNC'
from pathlib import Path
import os
import sys


def parse_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding='utf-8').splitlines():
        line = raw.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


env_path = Path(sys.argv[1])
config_path = Path(sys.argv[2])
env_values = parse_env(env_path)
openai_key = (os.environ.get('OPENAI_API_KEY') or env_values.get('OPENAI_API_KEY') or '').strip()
embedding_model = (os.environ.get('SELECTED_EMBEDDING_MODEL') or env_values.get('SELECTED_EMBEDDING_MODEL') or 'text-embedding-3-large').strip()

lines = config_path.read_text(encoding='utf-8').splitlines()
inside_knowledge = False
inside_openai = False
knowledge_updated = False
openai_key_updated = False
openai_configured_updated = False

for idx, line in enumerate(lines):
    if line.startswith('knowledge:'):
        inside_knowledge = True
        inside_openai = False
        continue
    if line.startswith('providers:'):
        inside_openai = False
        inside_knowledge = False
        continue
    if line.startswith('  openai:'):
        inside_openai = True
        inside_knowledge = False
        continue
    if line and not line.startswith(' '):
        inside_knowledge = False
        inside_openai = False
        continue
    if inside_knowledge and line.startswith('  embedding_model:'):
        lines[idx] = f'  embedding_model: {embedding_model}'
        knowledge_updated = True
        continue
    if inside_openai and line.startswith('    api_key:'):
        key_value = openai_key if openai_key else "''"
        lines[idx] = f'    api_key: {key_value}'
        openai_key_updated = True
        continue
    if inside_openai and line.startswith('    configured:'):
        lines[idx] = f"    configured: {'true' if openai_key else 'false'}"
        openai_configured_updated = True
        continue
    if inside_openai and line.startswith('  ') and not line.startswith('    '):
        inside_openai = False

config_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(
    f'synced_openrag_runtime_config=true config={config_path} '
    f'embedding_model={embedding_model} openai_key_present={bool(openai_key)} '
    f'knowledge_updated={knowledge_updated} openai_key_updated={openai_key_updated} '
    f'openai_configured_updated={openai_configured_updated}'
)
PY_SYNC
}


resolve_path() {
  local path="$1"
  if [[ -z "$path" ]]; then
    return 1
  fi
  if [[ "$path" = /* ]]; then
    printf '%s\n' "$path"
    return 0
  fi
  path="${path#./}"
  printf '%s\n' "$PACKAGE_ROOT/$path"
}

read_flow_id() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    return 1
  fi
  python3 - "$path" <<'PY2'
import json
import sys

path = sys.argv[1]
with open(path, 'r', encoding='utf-8') as handle:
    data = json.load(handle)
value = data.get('id')
if isinstance(value, str) and value.strip():
    print(value.strip())
PY2
}

set_env_if_unset() {
  local name="$1"
  local value="$2"
  if [[ -n "${!name:-}" || -z "$value" ]]; then
    return 0
  fi
  export "$name=$value"
  printf '[info] auto-detected %s=%s\n' "$name" "$value"
}

auto_detect_flow_ids() {
  local flows_root raw_root
  raw_root="${OPENRAG_FLOWS_PATH:-./state/flows}"
  flows_root="$(resolve_path "$raw_root")"
  if [[ ! -d "$flows_root" ]]; then
    printf '[warn] flow directory not found, skipping flow ID auto-detection: %s\n' "$flows_root" >&2
    return 0
  fi
  set_env_if_unset LANGFLOW_CHAT_FLOW_ID "$(read_flow_id "$flows_root/openrag_agent.json" || true)"
  set_env_if_unset LANGFLOW_INGEST_FLOW_ID "$(read_flow_id "$flows_root/ingestion_flow.json" || true)"
  set_env_if_unset LANGFLOW_URL_INGEST_FLOW_ID "$(read_flow_id "$flows_root/openrag_url_mcp.json" || true)"
  set_env_if_unset NUDGES_FLOW_ID "$(read_flow_id "$flows_root/openrag_nudges.json" || true)"
  export FASTPDF_OPENRAG_LANGFLOW_FLOWS_ROOT="${FASTPDF_OPENRAG_LANGFLOW_FLOWS_ROOT:-$flows_root}"
}

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
auto_detect_flow_ids
sync_openrag_runtime_config "$OPENRAG_CONFIG_FILE"
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
