#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DEVICE="${DEVICE:-0}"
IMGSZ="${IMGSZ:-640}"
WARMUP="${WARMUP:-10}"
ITERATIONS="${ITERATIONS:-50}"

python experiments/speed_benchmark.py \
  --device "$DEVICE" \
  --imgsz "$IMGSZ" \
  --warmup "$WARMUP" \
  --iterations "$ITERATIONS"
