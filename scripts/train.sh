#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/Common.sh"

MODEL="${MODEL:-ultralytics/cfg/models/v10/YoloV10sTalFFN.yaml}"
DATA="${DATA:-datasets/Crop/data.yaml}"
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

# 多数据集独立训练模式（默认）
# 格式: "name:data_yaml_path:epochs:batch[,name2:...]"
# - 每个数据集完全独立训练，每次都从 MODEL 重新初始化架构
# - 不传递上一个数据集的权重（非迁移学习，非微调）
# Crop  : 单目标近景，27类作物病害
# PDT/LH: 多目标远景，1类，LH子集（Low-density High-resolution）
# PDT/LL: 多目标远景，1类，LL子集（Low-resolution Large-scale）
MULTI_DATASETS="${MULTI_DATASETS:-Crop:datasets/Crop/data.yaml:150:16,PDT_LH:datasets/PDT/LH/PDT.yaml:150:16,PDT_LL:datasets/PDT/LL/PDT.yaml:150:16}"

usage() {
  cat <<'EOF'
Usage:
  bash Scripts/Train.sh [options]

默认行为（无参数）：
  依次独立训练三个数据集：Crop、PDT/LH、PDT/LL
  每个数据集从 MODEL 重新初始化，不传递前一数据集的权重。

Options:
  --model PATH          Model YAML 或 checkpoint 路径
  --data PATH           Dataset YAML 路径（仅单数据集模式，需配合 --multi-datasets ""）
  --epochs INT          训练轮数（单数据集模式）
  --imgsz INT           训练图像尺寸
  --batch INT           Batch size（单数据集模式）
  --device STR          设备，如 0、1、0,1、cpu
  --project PATH        输出目录
  --name STR            运行名称（单数据集模式）
  --workers INT         DataLoader 线程数
  --pretrained BOOL     是否使用预训练权重（针对 ImageNet backbone）
  --resume BOOL         是否从上次断点恢复（仅单数据集模式）
  --amp BOOL            是否开启自动混合精度
  --multi-datasets STR  多数据集独立训练配置
                        格式: "name1:path1:epochs1:batch1,name2:path2:epochs2:batch2"
                        设为 "" 则切换为单数据集模式
  --help                显示此帮助

Environment variables with the same names (MODEL, DATA, EPOCHS, ...) are also supported.

Examples:
  # 默认：依次独立训练 Crop、PDT_LH、PDT_LL（推荐）
  bash Scripts/Train.sh

  # 仅训练单个数据集
  bash Scripts/Train.sh --multi-datasets "" --data datasets/Crop/data.yaml --epochs 150

  # 自定义多数据集
  bash Scripts/Train.sh --multi-datasets "Crop:datasets/Crop/data.yaml:150:16,PDT_LH:datasets/PDT/LH/PDT.yaml:100:8"

  # 指定不同设备
  bash Scripts/Train.sh --device 1
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
    --multi-datasets) MULTI_DATASETS="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

MODEL="$(resolve_path "$MODEL")"
PRETRAINED="$(parse_bool "$PRETRAINED")"
RESUME="$(parse_bool "$RESUME")"
AMP="$(parse_bool "$AMP")"

require_file "$MODEL" "Model config"

# ============================================================
# 多数据集独立训练模式（默认）
# 每个数据集均从 MODEL 重新初始化，彼此完全独立，
# 不进行迁移学习（Transfer Learning），不进行微调（Fine-tuning）。
# ============================================================
if [ -n "$MULTI_DATASETS" ]; then
  echo ""
  echo "======================================================================"
  echo "🚀 多数据集独立训练模式"
  echo "   说明: 每个数据集从 MODEL 架构独立初始化，互不传递权重"
  echo "   非迁移学习 / 非微调 — 数据集之间完全隔离"
  echo "======================================================================"
  echo "   模型架构: $MODEL"
  echo "======================================================================"
  echo ""

  IFS=',' read -ra DATASET_CONFIGS <<< "$MULTI_DATASETS"
  TOTAL=${#DATASET_CONFIGS[@]}
  SUCCESS_COUNT=0
  FAILED_DATASETS=""

  for i in "${!DATASET_CONFIGS[@]}"; do
    IFS=':' read -r DS_NAME DS_PATH DS_EPOCHS DS_BATCH <<< "${DATASET_CONFIGS[$i]}"

    CURRENT=$((i + 1))

    echo ""
    echo "======================================================================"
    echo "📊 训练进度 [$CURRENT/$TOTAL]  →  $DS_NAME"
    echo "======================================================================"
    echo "   数据集路径 : $DS_PATH"
    echo "   Epochs    : $DS_EPOCHS"
    echo "   Batch     : $DS_BATCH"
    echo "   ImgSz     : $IMGSZ"
    echo "   Device    : $DEVICE"
    echo "   输出目录   : $PROJECT/$DS_NAME"
    echo "   ⚠️  从 MODEL 重新初始化（独立训练，不继承前序权重）"
    echo "======================================================================"
    echo ""

    # 检查数据集配置文件
    DS_PATH_RESOLVED="$(resolve_path "$DS_PATH")"
    if [ ! -f "$DS_PATH_RESOLVED" ]; then
      echo "❌ 错误: 数据集配置文件不存在: $DS_PATH_RESOLVED"
      echo "   跳过 $DS_NAME"
      FAILED_DATASETS="${FAILED_DATASETS}${DS_NAME}(文件不存在) "
      continue
    fi

    TRAIN_PROJECT="$(resolve_path "$PROJECT")"
    mkdir -p "$TRAIN_PROJECT"
    cd "$ROOT_DIR"

    # --------------------------------------------------------
    # 独立训练：每次都从 model_path 重新构造 YOLOv10 实例。
    # pretrained=True 表示使用 ImageNet 预训练的 backbone 权重，
    # 这与「迁移学习/微调」不同——这里没有任何来自上一数据集的权重传递。
    # --------------------------------------------------------
    if "$PYTHON_BIN" - "$MODEL" "$DS_PATH_RESOLVED" "$DS_EPOCHS" "$IMGSZ" "$DS_BATCH" \
          "$DEVICE" "$TRAIN_PROJECT" "$DS_NAME" "$WORKERS" "$PRETRAINED" "$AMP" <<'PY'
import sys
from ultralytics import YOLOv10

(model_path, data, epochs, imgsz, batch,
 device, project, name, workers, pretrained, amp) = sys.argv[1:]

# 每次独立初始化，不传递任何前序数据集的权重
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
    resume=False,           # 独立训练模式下禁止 resume，保证每次从头训练
    amp=amp == "True",
)
PY
    then
      echo ""
      echo "✅ [$CURRENT/$TOTAL] $DS_NAME 训练完成！"
      echo "   权重输出: $TRAIN_PROJECT/$DS_NAME/weights/best.pt"
      SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
      echo ""
      echo "❌ [$CURRENT/$TOTAL] $DS_NAME 训练失败！请检查数据集配置或训练日志。"
      FAILED_DATASETS="${FAILED_DATASETS}${DS_NAME}(训练失败) "
    fi
  done

  # 训练汇总
  echo ""
  echo "======================================================================"
  echo "🎉 全部数据集训练完成 — 汇总"
  echo "======================================================================"
  echo "   成功: $SUCCESS_COUNT / $TOTAL 个数据集"
  if [ -n "$FAILED_DATASETS" ]; then
    echo "   失败: $FAILED_DATASETS"
  fi
  echo "   输出根目录: $(resolve_path "$PROJECT")"
  echo ""
  echo "   各数据集权重位置："
  IFS=',' read -ra DATASET_CONFIGS <<< "$MULTI_DATASETS"
  for cfg in "${DATASET_CONFIGS[@]}"; do
    IFS=':' read -r DS_NAME DS_PATH _EPOCHS _BATCH <<< "$cfg"
    echo "     $DS_NAME  →  $(resolve_path "$PROJECT")/$DS_NAME/weights/best.pt"
  done
  echo "======================================================================"
  echo ""

  exit 0
fi

# ============================================================
# 单数据集训练模式（--multi-datasets "" 时激活）
# ============================================================
DATA="$(resolve_path "$DATA")"
PROJECT="$(resolve_path "$PROJECT")"

require_file "$DATA" "Dataset config"
mkdir -p "$PROJECT"

cd "$ROOT_DIR"

echo ""
echo "======================================================================"
echo "🚀 单数据集训练模式"
echo "======================================================================"
echo "   模型  : $MODEL"
echo "   数据集: $DATA"
echo "   Epochs: $EPOCHS  Batch: $BATCH  ImgSz: $IMGSZ"
echo "   输出  : $PROJECT/$NAME"
echo "======================================================================"
echo ""

"$PYTHON_BIN" - "$MODEL" "$DATA" "$EPOCHS" "$IMGSZ" "$BATCH" \
      "$DEVICE" "$PROJECT" "$NAME" "$WORKERS" "$PRETRAINED" "$RESUME" "$AMP" <<'PY'
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
