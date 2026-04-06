#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL="${MODEL:-ultralytics/cfg/models/v10/yolov10s_TAL_FFN.yaml}"
DATA="${DATA:-data/Crop/data.yaml}"
EPOCHS="${EPOCHS:-150}"
IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:-16}"
DEVICE="${DEVICE:-0}"

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
