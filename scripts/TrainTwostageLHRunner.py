import os
import sys
import torch

# 强制优先使用当前 AgriYOLO 项目里的 ultralytics，而不是 conda site-packages
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import ultralytics
print("当前使用的 ultralytics 路径:", ultralytics.__file__)

from ultralytics import YOLO


def main():
    model_cfg = "ultralytics/cfg/models/v10/Yolov10sTalFFN.yaml"
    if torch.cuda.is_available():
        device = 0
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    root_runs = "runs/TrainTwostageLH"

    crop_data = "datasets/Crop/data.yaml"
    pdt_lh_data = "datasets/PDT/LH/PDT.yaml"

    crop_name = "Crop_stage1"
    pdt_lh_name = "PDT_LH_stage2"

    print("=" * 80)
    print("两阶段迁移/微调训练启动 (Crop -> PDT/LH)")
    print("=" * 80)
    print("初始模型架构:", model_cfg)
    print("运行设备:", device)
    print("输出根目录:", root_runs)

    stage1_weights = os.path.join(root_runs, crop_name, "weights", "best.pt")

    if os.path.exists(stage1_weights):
        print("=" * 80)
        print("检测到 Stage 1 权重已存在，跳过 Crop 训练")
        print("权重文件:", stage1_weights)
        print("=" * 80)
    else:
        print("=" * 80)
        print("[STAGE 1/2] 开始训练近景作物数据集 Crop")
        print("数据集 YAML:", crop_data)
        print("=" * 80)

        model = YOLO(model_cfg)

        model.train(
            data=crop_data,
            epochs=150,
            imgsz=640,
            device=device,
            project=root_runs,
            name=crop_name,
            exist_ok=True,
            resume=False,
            batch=16,
            workers=8,
            cache=True,
        )

    if not os.path.exists(stage1_weights):
        raise FileNotFoundError(f"❌ Stage 1 未生成权重文件: {stage1_weights}")

    print("=" * 80)
    print("Stage 1 完成")
    print("权重文件:", stage1_weights)
    print("=" * 80)

    print("=" * 80)
    print("[STAGE 2/2] 开始迁移训练远景 PDT/LH 数据集")
    print("数据集 YAML:", pdt_lh_data)
    print("初始化权重:", stage1_weights)
    print("=" * 80)

    # 用 Stage 1 权重作为初始化
    model = YOLO(stage1_weights)

    model.train(
        data=pdt_lh_data,
        epochs=150,
        imgsz=640,
        device=device,
        project=root_runs,
        name=pdt_lh_name,
        exist_ok=True,
        resume=False,
        batch=16,
        workers=8,
        cache=True,
        lr0=0.001,
    )

    final_weights = os.path.join(root_runs, pdt_lh_name, "weights", "best.pt")

    print("=" * 80)
    print("两阶段训练 (Crop -> PDT/LH) 完成")
    print("Stage 1 近景权重:", stage1_weights)
    print("Stage 2 迁移后远景权重:", final_weights)
    print("=" * 80)


if __name__ == "__main__":
    main()
