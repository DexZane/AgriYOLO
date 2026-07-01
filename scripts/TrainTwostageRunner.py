import os
import sys
import torch

# 强制优先使用当前 AgriYOLO 项目里的 ultralytics，而不是 conda site-packages
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

    plantdoc_data = os.path.join(ROOT, "datasets", "PlantDoc", "data.yaml")
    pdt_data = os.path.join(ROOT, "datasets", "PDT", "LL", "PDT.yaml")

    plantdoc_name = "PlantDoc_stage1"
    pdt_name = "PDT_stage2_finetune"

    print("=" * 80)
    print("两阶段迁移/微调训练启动")
    print("=" * 80)
    print("初始模型架构:", model_cfg)
    print("运行设备:", device)
    print("输出根目录:", root_runs)

    stage1_weights = os.path.join(root_runs, plantdoc_name, "weights", "best.pt")

    if os.path.exists(stage1_weights):
        print("=" * 80)
        print("检测到 Stage 1 权重已存在，跳过 PlantDoc 训练")
        print("权重文件:", stage1_weights)
        print("=" * 80)
    else:
        print("=" * 80)
        print("[STAGE 1/2] 开始训练 PlantDoc 检测预训练数据集")
        print("数据集 YAML:", plantdoc_data)
        print("=" * 80)

        model = YOLO(model_cfg)

        model.train(
            data=plantdoc_data,
            epochs=150,
            imgsz=640,
            device=device,
            project=root_runs,
            name=plantdoc_name,
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
    print("[STAGE 2/2] 开始迁移训练远景 PDT 数据集")
    print("数据集 YAML:", pdt_data)
    print("初始化权重:", stage1_weights)
    print("=" * 80)

    # 关键点：
    # 用 Stage 1 权重作为初始化。
    # PDT 的 data.yaml 是 1 类，训练时会按 PDT 的 nc 进行检测头适配。
    model = YOLO(stage1_weights)

    model.train(
        data=pdt_data,
        epochs=150,
        imgsz=640,
        device=device,
        project=root_runs,
        name=pdt_name,
        exist_ok=True,
        resume=False,
        batch=16,
        workers=8,
        cache=True,
        lr0=0.001,
    )

    final_weights = os.path.join(root_runs, pdt_name, "weights", "best.pt")

    print("=" * 80)
    print("两阶段训练完成")
    print("Stage 1 PlantDoc 权重:", stage1_weights)
    print("Stage 2 迁移后远景权重:", final_weights)
    print("=" * 80)


if __name__ == "__main__":
    main()
