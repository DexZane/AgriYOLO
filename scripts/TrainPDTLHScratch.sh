#!/bin/bash
set -e

cd "$(dirname "$0")/.."

echo "============================================================"
echo "🚀 使用本地 AgriYOLO/ultralytics 启动 PDT/LH 从头训练对照试验"
echo "============================================================"

export PYTHONPATH="$(pwd):$PYTHONPATH"

python3 Scripts/TrainPDTLHScratchRunner.py
