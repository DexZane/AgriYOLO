import csv
import json
import os
import sys
import torch

# 强制优先使用当前 AgriYOLO 项目里的 ultralytics，而不是 conda site-packages
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from ultralytics import YOLO


def ensureDir(path):
    os.makedirs(path, exist_ok=True)


def getBestEpochAndMetrics(resultsCsvPath):
    if not os.path.exists(resultsCsvPath):
        return None, None

    with open(resultsCsvPath, newline='') as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    if not rows:
        return None, None

    metricKey = 'metrics/mAP50-95(B)'
    if metricKey not in rows[0]:
        return None, None

    bestRow = max(rows, key=lambda row: float(row.get(metricKey, float('-inf'))))
    bestEpoch = int(float(bestRow['epoch']))
    bestMetrics = {
        key: float(value)
        for key, value in bestRow.items()
        if key and key != 'epoch' and value not in (None, '')
    }
    return bestEpoch, bestMetrics


def saveBestInfo(stageDir, stageName, bestWeightPath):
    resultsCsvPath = os.path.join(stageDir, 'results.csv')
    bestEpoch, bestMetrics = getBestEpochAndMetrics(resultsCsvPath)
    info = {
        'stageName': stageName,
        'bestEpoch': bestEpoch,
        'bestWeightPath': bestWeightPath,
        'resultsCsvPath': resultsCsvPath,
        'bestMetrics': bestMetrics,
    }
    infoPath = os.path.join(stageDir, 'bestInfo.json')
    ensureDir(stageDir)
    with open(infoPath, 'w', encoding='utf-8') as file:
        json.dump(info, file, ensure_ascii=False, indent=2)
    print('最佳信息已保存:', infoPath)
    if bestEpoch is not None:
        print('最佳 epoch:', bestEpoch)
        print('最佳指标:', bestMetrics)
    else:
        print('未能从 results.csv 解析最佳 epoch / 指标')


def loadBestInfo(stageDir):
    infoPath = os.path.join(stageDir, 'bestInfo.json')
    if not os.path.exists(infoPath):
        return None
    with open(infoPath, 'r', encoding='utf-8') as file:
        return json.load(file)


def resolveBestWeightPath(stageDir, fallbackWeightPath):
    info = loadBestInfo(stageDir)
    if info:
        bestWeightPath = info.get('bestWeightPath')
        if bestWeightPath and os.path.exists(bestWeightPath):
            print('读取到最佳权重:', bestWeightPath)
            return bestWeightPath
        print('bestInfo.json 中的权重不存在，回退到:', fallbackWeightPath)
    return fallbackWeightPath


def main():
    model_cfg = "ultralytics/cfg/models/v10/Yolov10sTalFFN.yaml"
    if torch.cuda.is_available():
        device = 0
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    root_runs = os.path.join(ROOT, "runs", "TrainThreestage")

    plantdoc_data = os.path.join(ROOT, "datasets", "PlantDoc", "data.yaml")
    pdt_ll_data = os.path.join(ROOT, "datasets", "PDT", "LL", "PDT.yaml")
    pdt_lh_data = os.path.join(ROOT, "datasets", "PDT", "LH", "PDT.yaml")

    plantdoc_name = "PlantDoc_stage1"
    pdt_ll_name = "PDT_LL_stage2"
    pdt_lh_name = "PDT_LH_stage3"
    project_runs = os.path.join(ROOT, "runs", "TrainThreestage")

    print("=" * 80)
    print("三阶段连续迁移/微调训练启动")
    print("=" * 80)
    print("初始模型架构:", model_cfg)
    print("运行设备:", device)
    print("输出根目录:", project_runs)
    print("Stage 1: PlantDoc 检测预训练, 适配 12GB 显存")
    print("Stage 2: PDT/LL 单类检测微调")
    print("Stage 3: PDT/LH 单类检测继续微调")

    stage1_weights = os.path.join(project_runs, plantdoc_name, "weights", "best.pt")
    stage2_weights = os.path.join(project_runs, pdt_ll_name, "weights", "best.pt")
    stage3_weights = os.path.join(project_runs, pdt_lh_name, "weights", "best.pt")

    plantdoc_epochs = 80
    plantdoc_imgsz = 640
    plantdoc_batch = 8
    plantdoc_workers = 4
    plantdoc_patience = 25

    pdt_epochs = 120
    pdt_imgsz = 640
    pdt_batch = 8
    pdt_workers = 4
    pdt_patience = 30
    pdt_lr0 = 5e-4

    # ==================== STAGE 1: PlantDoc ====================
    stage1WeightsToUse = resolveBestWeightPath(os.path.join(project_runs, plantdoc_name), stage1_weights)

    if os.path.exists(stage1_weights):
        print("=" * 80)
        print("检测到 Stage 1 权重已存在，跳过 PlantDoc 训练")
        print("权重文件:", stage1_weights)
        print("用于下一阶段的权重:", stage1WeightsToUse)
        print("=" * 80)
    else:
        print("=" * 80)
        print("[STAGE 1/3] 开始训练 PlantDoc 检测预训练数据集")
        print("数据集 YAML:", plantdoc_data)
        print("=" * 80)

        model = YOLO(model_cfg)

        model.train(
            data=plantdoc_data,
            epochs=plantdoc_epochs,
            imgsz=plantdoc_imgsz,
            device=device,
            project=project_runs,
            name=plantdoc_name,
            exist_ok=True,
            resume=False,
            batch=plantdoc_batch,
            workers=plantdoc_workers,
            cache=True,
            patience=plantdoc_patience,
        )

    if not os.path.exists(stage1_weights):
        raise FileNotFoundError(f"❌ Stage 1 未生成权重文件: {stage1_weights}")

    stage1Dir = os.path.join(project_runs, plantdoc_name)
    saveBestInfo(stage1Dir, 'Stage 1 PlantDoc', stage1_weights)
    stage1WeightsToUse = resolveBestWeightPath(stage1Dir, stage1_weights)

    print("=" * 80)
    print("Stage 1 完成")
    print("权重文件:", stage1_weights)
    print("=" * 80)

    # ==================== STAGE 2: PDT/LL ====================
    stage2WeightsToUse = resolveBestWeightPath(os.path.join(project_runs, pdt_ll_name), stage2_weights)

    if os.path.exists(stage2_weights):
        print("=" * 80)
        print("检测到 Stage 2 权重已存在，跳过 PDT_LL 训练")
        print("权重文件:", stage2_weights)
        print("用于下一阶段的权重:", stage2WeightsToUse)
        print("=" * 80)
    else:
        print("=" * 80)
        print("[STAGE 2/3] 开始迁移训练远景 PDT/LL 数据集")
        print("数据集 YAML:", pdt_ll_data)
        print("初始化权重:", stage1WeightsToUse)
        print("=" * 80)

        model = YOLO(stage1WeightsToUse)

        model.train(
            data=pdt_ll_data,
            epochs=pdt_epochs,
            imgsz=pdt_imgsz,
            device=device,
            project=project_runs,
            name=pdt_ll_name,
            exist_ok=True,
            resume=False,
            batch=pdt_batch,
            workers=pdt_workers,
            cache=True,
            lr0=pdt_lr0,
            patience=pdt_patience,
        )

    if not os.path.exists(stage2_weights):
        raise FileNotFoundError(f"❌ Stage 2 未生成权重文件: {stage2_weights}")

    stage2Dir = os.path.join(root_runs, pdt_ll_name)
    saveBestInfo(stage2Dir, 'Stage 2 PDT/LL', stage2_weights)
    stage2WeightsToUse = resolveBestWeightPath(stage2Dir, stage2_weights)

    print("=" * 80)
    print("Stage 2 完成")
    print("权重文件:", stage2_weights)
    print("=" * 80)

    # ==================== STAGE 3: PDT/LH ====================
    if os.path.exists(stage3_weights):
        print("=" * 80)
        print("检测到 Stage 3 权重已存在，跳过 PDT_LH 训练")
        print("权重文件:", stage3_weights)
        print("=" * 80)
    else:
        print("=" * 80)
        print("[STAGE 3/3] 开始迁移训练远景 PDT/LH 数据集")
        print("数据集 YAML:", pdt_lh_data)
        print("初始化权重:", stage2WeightsToUse)
        print("=" * 80)

        model = YOLO(stage2WeightsToUse)

        model.train(
            data=pdt_lh_data,
            epochs=pdt_epochs,
            imgsz=pdt_imgsz,
            device=device,
            project=root_runs,
            name=pdt_lh_name,
            exist_ok=True,
            resume=False,
            batch=pdt_batch,
            workers=pdt_workers,
            cache=True,
            lr0=pdt_lr0,
            patience=pdt_patience,
        )

    stage3Dir = os.path.join(root_runs, pdt_lh_name)
    saveBestInfo(stage3Dir, 'Stage 3 PDT/LH', stage3_weights)
    stage3WeightsToUse = resolveBestWeightPath(stage3Dir, stage3_weights)

    print("=" * 80)
    print("三阶段训练全部完成")
    print("Stage 1 PlantDoc 权重:", stage1_weights)
    print("Stage 1 实际传递权重:", stage1WeightsToUse)
    print("Stage 2 PDT/LL 权重:", stage2_weights)
    print("Stage 2 实际传递权重:", stage2WeightsToUse)
    print("Stage 3 PDT/LH 权重:", stage3_weights)
    print("Stage 3 实际传递权重:", stage3WeightsToUse)
    print("Stage 1 参数:", {"epochs": plantdoc_epochs, "imgsz": plantdoc_imgsz, "batch": plantdoc_batch, "patience": plantdoc_patience})
    print("Stage 2/3 参数:", {"epochs": pdt_epochs, "imgsz": pdt_imgsz, "batch": pdt_batch, "lr0": pdt_lr0, "patience": pdt_patience})
    print("=" * 80)


if __name__ == "__main__":
    main()
