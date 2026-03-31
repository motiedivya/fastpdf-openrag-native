#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="${OPENRAG_ENV_FILE:-$PACKAGE_ROOT/.env}"
COMPOSE_FILE="${OPENRAG_COMPOSE_FILE:-$PACKAGE_ROOT/docker-compose.yml}"

if [[ ! -f "$ENV_FILE" ]]; then
  printf '[error] env file not found: %s
' "$ENV_FILE" >&2
  exit 1
fi

set -a
source "$ENV_FILE"
set +a

compose() {
  docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" "$@"
}

printf '[info] docker compose ps
'
compose ps
printf '
'

check() {
  local url="$1"
  local label="$2"
  local code
  code="$(curl -sS -o /dev/null -w '%{http_code}' "$url" || true)"
  printf '[check] %-20s %s -> %s
' "$label" "$url" "$code"
}

check "http://127.0.0.1:${FRONTEND_PORT:-3000}/" "OpenRAG Frontend"
check "http://127.0.0.1:${LANGFLOW_PORT:-7860}/health" "Langflow"
check "http://127.0.0.1:${DOCLING_PORT:-5001}/health" "Docling"
check "https://127.0.0.1:${OPENSEARCH_PORT:-9200}" "OpenSearch"
