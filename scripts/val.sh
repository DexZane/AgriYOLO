#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL="${MODEL:-weights/best.pt}"
DATA="${DATA:-data/Crop/data.yaml}"
SPLIT="${SPLIT:-test}"
IMGSZ="${IMGSZ:-640}"
DEVICE="${DEVICE:-0}"
SAVE_JSON="${SAVE_JSON:-true}"

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
