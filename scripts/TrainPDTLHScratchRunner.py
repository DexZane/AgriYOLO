import os
import sys
import torch

# 强制优先使用当前 AgriYOLO 项目里的 ultralytics
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from ultralytics import YOLO


def main():
    model_cfg = "ultralytics/cfg/models/v10/Yolov10sTalFFN.yaml"
    if torch.cuda.is_available():
        device = 0
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    root_runs = os.path.join(ROOT, "runs", "TrainTwostage")
    pdt_lh_data = os.path.join(ROOT, "datasets", "PDT", "LH", "PDT.yaml")
    run_name = "PDT_LH_scratch_baseline"

    print("=" * 80)
    print("PDT/LH 从头训练对照试验启动 (Scratch)")
    print("=" * 80)
    print("初始模型架构:", model_cfg)
    print("运行设备:", device)
    print("数据集 YAML:", pdt_lh_data)
    print("输出目录:", os.path.join(root_runs, run_name))
    print("=" * 80)

    model = YOLO(model_cfg)  # 从 YAML 架构从头初始化，不加载任何预训练权重

    model.train(
        data=pdt_lh_data,
        epochs=150,
        imgsz=640,
        device=device,
        project=root_runs,
        name=run_name,
        exist_ok=True,
        resume=False,
        batch=16,
        workers=8,
        cache=True,
    )

    final_weights = os.path.join(root_runs, run_name, "weights", "best.pt")
    print("=" * 80)
    print("PDT/LH 从头训练完成")
    print("最佳权重保存至:", final_weights)
    print("=" * 80)


if __name__ == "__main__":
    main()
