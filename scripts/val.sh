#!/usr/bin/env bash
set -euo pipefail

CALL_DIR="$(pwd)"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

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

MODEL="$(resolve_path "${MODEL:-weights/best.pt}")"
DATA="$(resolve_path "${DATA:-data/Crop/data.yaml}")"
SPLIT="${SPLIT:-test}"
IMGSZ="${IMGSZ:-640}"
DEVICE="${DEVICE:-0}"
SAVE_JSON="${SAVE_JSON:-true}"

if [[ ! -f "$MODEL" ]]; then
  echo "Checkpoint not found: $MODEL" >&2
  exit 1
fi

if [[ ! -f "$DATA" ]]; then
  echo "Dataset config not found: $DATA" >&2
  exit 1
fi

cd "$ROOT_DIR"

python - "$MODEL" "$DATA" "$SPLIT" "$IMGSZ" "$DEVICE" "$SAVE_JSON" <<'PY'
import sys
from ultralytics import YOLOv10

model_path, data, split, imgsz, device, save_json = sys.argv[1:]
model = YOLOv10(model_path)
model.val(
    data=data,
    split=split,
    imgsz=int(imgsz),
    device=device,
    save_json=save_json.lower() == "true",
)
PY
