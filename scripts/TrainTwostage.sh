#!/bin/bash
set -e

cd "$(dirname "$0")/.."

echo "============================================================"
echo "🚀 使用本地 AgriYOLO/ultralytics 启动两阶段训练"
echo "============================================================"

export PYTHONPATH="$(pwd):$PYTHONPATH"

python3 Scripts/TrainTwostageRunner.py
