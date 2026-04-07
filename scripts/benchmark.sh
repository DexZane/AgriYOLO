#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

DEVICE="${DEVICE:-0}"
IMGSZ="${IMGSZ:-640}"
WARMUP="${WARMUP:-10}"
ITERATIONS="${ITERATIONS:-50}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/benchmark.sh [options]

Options:
  --device STR        Device string, e.g. 0, 1, cpu
  --imgsz INT         Input image size
  --warmup INT        Warmup iterations
  --iterations INT    Measured iterations
  --help              Show this message

Environment variables with the same names are also supported.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device) DEVICE="$2"; shift 2 ;;
    --imgsz) IMGSZ="$2"; shift 2 ;;
    --warmup) WARMUP="$2"; shift 2 ;;
    --iterations) ITERATIONS="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

cd "$ROOT_DIR"

"$PYTHON_BIN" experiments/speed_benchmark.py \
  --device "$DEVICE" \
  --imgsz "$IMGSZ" \
  --warmup "$WARMUP" \
  --iterations "$ITERATIONS"
