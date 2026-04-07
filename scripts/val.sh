#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

MODEL="${MODEL:-weights/best.pt}"
DATA="${DATA:-data/Crop/data.yaml}"
SPLIT="${SPLIT:-test}"
IMGSZ="${IMGSZ:-640}"
DEVICE="${DEVICE:-0}"
SAVE_JSON="${SAVE_JSON:-true}"
PLOTS="${PLOTS:-true}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/val.sh [options]

Options:
  --model PATH        Checkpoint path
  --data PATH         Dataset YAML path
  --split STR         Dataset split, e.g. val or test
  --imgsz INT         Validation image size
  --device STR        Device string, e.g. 0, 1, cpu
  --save-json BOOL    Save COCO-style predictions JSON
  --plots BOOL        Save validation plots
  --help              Show this message

Environment variables with the same names are also supported.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --data) DATA="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --imgsz) IMGSZ="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --save-json) SAVE_JSON="$2"; shift 2 ;;
    --plots) PLOTS="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

MODEL="$(resolve_path "$MODEL")"
DATA="$(resolve_path "$DATA")"
SAVE_JSON="$(parse_bool "$SAVE_JSON")"
PLOTS="$(parse_bool "$PLOTS")"

require_file "$MODEL" "Checkpoint"
require_file "$DATA" "Dataset config"

cd "$ROOT_DIR"

"$PYTHON_BIN" - "$MODEL" "$DATA" "$SPLIT" "$IMGSZ" "$DEVICE" "$SAVE_JSON" "$PLOTS" <<'PY'
import sys
from ultralytics import YOLOv10

model_path, data, split, imgsz, device, save_json, plots = sys.argv[1:]
model = YOLOv10(model_path)
model.val(
    data=data,
    split=split,
    imgsz=int(imgsz),
    device=device,
    save_json=save_json == "True",
    plots=plots == "True",
)
PY
