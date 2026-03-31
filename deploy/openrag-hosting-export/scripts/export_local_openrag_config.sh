#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
STAMP="$(date +%Y%m%dT%H%M%S)"
EXPORT_ROOT="${EXPORT_ROOT:-$PACKAGE_ROOT/exports/local-$STAMP}"
SOURCE_TUI_ROOT="${SOURCE_TUI_ROOT:-$HOME/.openrag/tui}"
SOURCE_OPENRAG_ROOT="${SOURCE_OPENRAG_ROOT:-$HOME/.openrag}"
INCLUDE_KEYS="${INCLUDE_KEYS:-false}"

mkdir -p "$EXPORT_ROOT"

copy_if_exists() {
  local src="$1"
  local dst="$2"
  if [[ -e "$src" ]]; then
    mkdir -p "$(dirname "$dst")"
    cp -R "$src" "$dst"
    printf '[ok] exported %s -> %s
' "$src" "$dst"
  else
    printf '[warn] missing source, skipped: %s
' "$src"
  fi
}

copy_if_exists "$SOURCE_TUI_ROOT/docker-compose.yml" "$EXPORT_ROOT/tui/docker-compose.yml"
copy_if_exists "$SOURCE_TUI_ROOT/.env" "$EXPORT_ROOT/tui/.env"
copy_if_exists "$SOURCE_OPENRAG_ROOT/flows" "$EXPORT_ROOT/openrag/flows"
copy_if_exists "$SOURCE_OPENRAG_ROOT/config" "$EXPORT_ROOT/openrag/config"
copy_if_exists "$SOURCE_OPENRAG_ROOT/data" "$EXPORT_ROOT/openrag/data"
copy_if_exists "$SOURCE_OPENRAG_ROOT/documents" "$EXPORT_ROOT/openrag/documents"

if [[ "$INCLUDE_KEYS" == "true" ]]; then
  copy_if_exists "$SOURCE_OPENRAG_ROOT/keys" "$EXPORT_ROOT/openrag/keys"
else
  printf '[info] skipping keys export (set INCLUDE_KEYS=true if you really want them)
'
fi

cat > "$EXPORT_ROOT/README.txt" <<EOF
This directory is a point-in-time export of the current local OpenRAG setup.
It was created without editing the live config.

Sensitive files may be present here, especially tui/.env and optional keys.
Treat this export as private infrastructure state.
EOF

printf '[done] export written to %s
' "$EXPORT_ROOT"
