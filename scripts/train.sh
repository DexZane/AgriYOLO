#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

MODEL="${MODEL:-ultralytics/cfg/models/v10/yolov10s_TAL_FFN.yaml}"
DATA="${DATA:-data/Crop/data.yaml}"
EPOCHS="${EPOCHS:-150}"
IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:-16}"
DEVICE="${DEVICE:-0}"
PROJECT="${PROJECT:-runs/train}"
NAME="${NAME:-agriyolo}"
WORKERS="${WORKERS:-8}"
PRETRAINED="${PRETRAINED:-true}"
RESUME="${RESUME:-false}"
AMP="${AMP:-true}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/train.sh [options]

Options:
  --model PATH        Model YAML or checkpoint path
  --data PATH         Dataset YAML path
  --epochs INT        Number of training epochs
  --imgsz INT         Training image size
  --batch INT         Batch size
  --device STR        Device string, e.g. 0, 1, 0,1, cpu
  --project PATH      Output project directory
  --name STR          Run name
  --workers INT       Data loader workers
  --pretrained BOOL   Whether to use pretrained weights
  --resume BOOL       Resume from the last checkpoint
  --amp BOOL          Enable automatic mixed precision
  --help              Show this message

Environment variables with the same names are also supported.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --data) DATA="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --imgsz) IMGSZ="$2"; shift 2 ;;
    --batch) BATCH="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --project) PROJECT="$2"; shift 2 ;;
    --name) NAME="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --pretrained) PRETRAINED="$2"; shift 2 ;;
    --resume) RESUME="$2"; shift 2 ;;
    --amp) AMP="$2"; shift 2 ;;
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
PROJECT="$(resolve_path "$PROJECT")"
PRETRAINED="$(parse_bool "$PRETRAINED")"
RESUME="$(parse_bool "$RESUME")"
AMP="$(parse_bool "$AMP")"

require_file "$MODEL" "Model config"
require_file "$DATA" "Dataset config"
mkdir -p "$PROJECT"

cd "$ROOT_DIR"

"$PYTHON_BIN" - "$MODEL" "$DATA" "$EPOCHS" "$IMGSZ" "$BATCH" "$DEVICE" "$PROJECT" "$NAME" "$WORKERS" "$PRETRAINED" "$RESUME" "$AMP" <<'PY'
import sys
from ultralytics import YOLOv10

(
    model_path,
    data,
    epochs,
    imgsz,
    batch,
    device,
    project,
    name,
    workers,
    pretrained,
    resume,
    amp,
) = sys.argv[1:]

model = YOLOv10(model_path)
model.train(
    data=data,
    epochs=int(epochs),
    imgsz=int(imgsz),
    batch=int(batch),
    device=device,
    project=project,
    name=name,
    workers=int(workers),
    pretrained=pretrained == "True",
    resume=resume == "True",
    amp=amp == "True",
)
PY
