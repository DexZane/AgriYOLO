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

MODEL="$(resolve_path "${MODEL:-ultralytics/cfg/models/v10/yolov10s_TAL_FFN.yaml}")"
DATA="$(resolve_path "${DATA:-data/Crop/data.yaml}")"
EPOCHS="${EPOCHS:-150}"
IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:-16}"
DEVICE="${DEVICE:-0}"

if [[ ! -f "$MODEL" ]]; then
  echo "Model config not found: $MODEL" >&2
  exit 1
fi

if [[ ! -f "$DATA" ]]; then
  echo "Dataset config not found: $DATA" >&2
  exit 1
fi

cd "$ROOT_DIR"

python - "$MODEL" "$DATA" "$EPOCHS" "$IMGSZ" "$BATCH" "$DEVICE" <<'PY'
import sys
from ultralytics import YOLOv10

model_path, data, epochs, imgsz, batch, device = sys.argv[1:]
model = YOLOv10(model_path)
model.train(
    data=data,
    epochs=int(epochs),
    imgsz=int(imgsz),
    batch=int(batch),
    device=device,
)
PY
