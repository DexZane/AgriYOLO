#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CALL_DIR="$(pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

resolve_path() {
  local value="$1"
  if [[ "$value" = /* ]]; then
    printf '%s\n' "$value"
    return
  fi
  if [[ -e "$ROOT_DIR/$value" ]]; then
    printf '%s\n' "$ROOT_DIR/$value"
    return
  fi
  if [[ -e "$CALL_DIR/$value" ]]; then
    printf '%s\n' "$CALL_DIR/$value"
    return
  fi
  printf '%s\n' "$value"
}

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "$path" ]]; then
    echo "$label not found: $path" >&2
    exit 1
  fi
}

parse_bool() {
  local value="${1,,}"
  case "$value" in
    1|true|yes|y|on) printf 'True\n' ;;
    0|false|no|n|off) printf 'False\n' ;;
    *)
      echo "Invalid boolean value: $1" >&2
      exit 1
      ;;
  esac
}
