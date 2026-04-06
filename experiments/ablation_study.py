"""
TAL-FFN Ablation Study Orchestrator
------------------------------------
Purpose:
    This script automates the training process for 5 different model configurations
    to validate the effectiveness of TAL-FFN (Task-driven Asymmetric Lightweight
    Feature Fusion Network) components:

    Stage 1: Baseline (YOLOv10s with PANet)
    Stage 2: Standard BiFPN (P2 detection head + BiFPN)
    Stage 3: +ADSA (Asymmetric Depth Strategy Allocation)
    Stage 4: +CADFM (Context-Aware Dynamic Fusion Mechanism)
    Stage 5: TAL-FFN Full (+ Depthwise Separable Convolution)

Usage:
    python ablation_study.py

Output:
    Models and training logs are saved in 'TAL_FFN_Ablation/' directory.
    Results summary is saved in 'results/ablation_summary.csv'.
"""
import os
import sys
import yaml
import gc
import torch

# Ensure local 'ultralytics' is used (Must be before any ultralytics imports!)
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from ultralytics import YOLO

# === 实验配置简洁定义 ===
# 配置文件统一根目录
CFG_DIR = "ultralytics/cfg/models/v10"

# 格式: (实验名称, 配置文件名, 使用MPDIoU, 预训练权重)
# 所有的消融实验均从 0 开始训练 (pretrained=False)
EXPERIMENTS = [
    # Stage 1: Baseline - 基线模型 (YOLOv10s + PANet)
    ("Stage1_Baseline",      "yolov10s_baseline.yaml",       False, False),

    # Stage 2: Standard BiFPN - 标准BiFPN (P2检测头 + 均匀深度分配)
    ("Stage2_Standard_BiFPN", "yolov10s_P2_BiFPN.yaml",         False, False),

    # Stage 3: +ADSA - 添加非对称深度分配策略 (Asymmetric Depth Strategy Allocation)
    ("Stage3_ADSA",           "yolov10s_P2_ADSA.yaml",          False, False),

    # Stage 4: +CADFM - 添加上下文感知动态融合机制 (Context-Aware Dynamic Fusion Mechanism)
    ("Stage4_CADFM",          "yolov10s_P2_CADFM.yaml",         False, False),

    # Stage 5: TAL-FFN Full - 完整版 (添加深度可分离卷积 DSConv)
    ("Stage5_TAL_FFN_Full",   "yolov10s_TAL_FFN.yaml",          False, False),
]

# 训练通用配置
# 注意：从零训练 (Scratch) 通常需要更长的 Epoch 才能收敛
TRAIN_CFG = {
    "data": "data/Crop/data.yaml",
    "epochs": 150,  # 增加轮数以适应从零训练
    "patience": 50, # 增加耐心值
    "imgsz": 640,
    "batch": 16,
    "project": "TAL_FFN_Ablation",  # 更新项目名称为TAL-FFN
    "optimizer": "AdamW",
    "lr0": 0.002,
    "warmup_bias_lr": 0.0,
    # 新增：保存详细结果用于可视化
    "save": True,
    "save_period": -1,  # 只保存最佳模型
    "plots": True,      # 保存训练图表
    "verbose": True,
}

def check_datasets(data_yaml_path):
    """自检数据集和标签是否存在"""
    if not os.path.exists(data_yaml_path):
        print(f"❌ 找不到 data.yaml: {data_yaml_path}")
        return False
    
    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    base_path = data.get('path', '')
    train_img = os.path.join(base_path, data.get('train', ''))
    # YOLO 默认逻辑：将路径中的 images 替换为 labels 来找标签
    train_label = train_img.replace('images', 'labels')
    
    print(f"正在检查训练集标签: {train_label}")
    if not os.path.exists(train_label):
        print(f"❌ 错误: 找不到标签文件夹 {train_label}")
        return False
    
    num_labels = len([f for f in os.listdir(train_label) if f.endswith('.txt')])
    if num_labels == 0:
        print(f"❌ 错误: {train_label} 文件夹内没有 .txt 标签文件！")
        return False
    
    print(f"✅ 自检通过: 找到 {num_labels} 个训练标签。")
    return True

def run_ablation():
    print("开始 AgriYOLO 消融实验流程...")
    
    if not check_datasets(TRAIN_CFG["data"]):
        print("程序终止: 请先修复数据集路径和标签问题。")
        return

    for name, yaml_file, use_mpdiou, is_pretrained in EXPERIMENTS:
        print(f"\n>>> 正在运行实验: {name}")
        
        # 显存清理
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        # 1. 检查 YAML 文件是否存在
        model_yaml = os.path.join(CFG_DIR, yaml_file)
        if not os.path.exists(model_yaml):
            print(f"错误: 配置文件 {model_yaml} 不存在，跳过。")
            continue
            
        # 2. 加载模型
        model = YOLO(model_yaml)
        
        # 3. 准备参数
        train_args = TRAIN_CFG.copy()
        train_args.update({
            "name": name,
            "use_mpdiou": use_mpdiou,
            "pretrained": is_pretrained
        })
        
        # 4. 启动训练
        try:
            model.train(**train_args)
            print(f"实验 {name} 训练完成。")
            
            # 5. 测试集评估
            print(f"\n>>> 正在对 {name} 进行 TEST 集最终评估...")
            model.val(split='test', project=train_args["project"], name=f"{name}_TEST")
            
        except Exception as e:
            print(f"实验 {name} 失败: {e}")

if __name__ == "__main__":
    run_ablation()
