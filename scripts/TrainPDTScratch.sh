#!/bin/bash
set -e

cd "$(dirname "$0")/.."

export PYTHONPATH="$(pwd):$PYTHONPATH"

python3 - << 'PY'
import os
import sys
import torch

ROOT = os.getcwd()
sys.path.insert(0, ROOT)

from ultralytics import YOLO

model_cfg = "ultralytics/cfg/models/v10/Yolov10sTalFFN.yaml"
data = "datasets/PDT/LL/PDT.yaml"

if torch.cuda.is_available():
    device = 0
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("使用设备:", device)
print("开始 PDT 从头训练对照实验")

model = YOLO(model_cfg)
model.train(
    data=data,
    epochs=150,
    imgsz=640,
    device=device,
    project="runs/TrainTwostage",
    name="PDT_scratch_baseline",
    exist_ok=True,
    resume=False,
    batch=16,
    workers=8,
    cache=True,
)
PY
