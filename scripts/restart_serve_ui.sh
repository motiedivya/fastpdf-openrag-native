#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UI_HOST="${FASTPDF_OPENRAG_UI_HOST:-127.0.0.1}"
UI_PORT="${FASTPDF_OPENRAG_UI_PORT:-8077}"
UI_LOG="${FASTPDF_UI_LOG:-$REPO_ROOT/outputs/serve-ui.log}"
UI_PID_FILE="${FASTPDF_UI_PID_FILE:-$REPO_ROOT/outputs/serve-ui.pid}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

wait_for_http() {
  local url="$1"
  local attempts="${2:-30}"
  local sleep_seconds="${3:-1}"
  local i

  for ((i = 1; i <= attempts; i += 1)); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep "$sleep_seconds"
  done

  return 1
}

kill_existing_ui() {
  if [[ -f "$UI_PID_FILE" ]]; then
    kill "$(cat "$UI_PID_FILE")" >/dev/null 2>&1 || true
  fi

  if command -v lsof >/dev/null 2>&1; then
    lsof -ti tcp:"$UI_PORT" | xargs -r kill >/dev/null 2>&1 || true
    sleep 2
    lsof -ti tcp:"$UI_PORT" | xargs -r kill -9 >/dev/null 2>&1 || true
  elif command -v fuser >/dev/null 2>&1; then
    fuser -k "${UI_PORT}/tcp" >/dev/null 2>&1 || true
    sleep 2
    fuser -k -9 "${UI_PORT}/tcp" >/dev/null 2>&1 || true
  fi

  pkill -f 'fastpdf-openrag-native serve-ui' >/dev/null 2>&1 || true
}

mkdir -p "$(dirname "$UI_LOG")"
mkdir -p "$(dirname "$UI_PID_FILE")"

printf '[info] repo=%s
' "$REPO_ROOT"
printf '[info] ui=%s:%s
' "$UI_HOST" "$UI_PORT"
printf '[info] log=%s
' "$UI_LOG"

kill_existing_ui

(
  cd "$REPO_ROOT"
  if command -v setsid >/dev/null 2>&1; then
    setsid env UV_CACHE_DIR="$UV_CACHE_DIR" uv run fastpdf-openrag-native serve-ui \
      --host "$UI_HOST" \
      --port "$UI_PORT" >"$UI_LOG" 2>&1 < /dev/null &
  else
    nohup env UV_CACHE_DIR="$UV_CACHE_DIR" uv run fastpdf-openrag-native serve-ui \
      --host "$UI_HOST" \
      --port "$UI_PORT" >"$UI_LOG" 2>&1 < /dev/null &
  fi
  echo $! >"$UI_PID_FILE"
)

printf '[ok] started ui pid=%s
' "$(cat "$UI_PID_FILE")"

if wait_for_http "http://$UI_HOST:$UI_PORT/" 45 1; then
  printf '[ok] ui is up: http://%s:%s/
' "$UI_HOST" "$UI_PORT"
else
  printf '[error] ui did not become healthy: http://%s:%s/
' "$UI_HOST" "$UI_PORT" >&2
  tail -n 40 "$UI_LOG" >&2 || true
  exit 1
fi

tail -n 20 "$UI_LOG"
