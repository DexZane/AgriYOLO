# AgriYOLO 实验手册（最终版）

> **模型**: `ultralytics/cfg/models/v10/Yolov10sTalFFN.yaml`  
> **执行方式**: 所有脚本在项目根目录执行 (`bash Scripts/*.sh`)  
> **设备**: 自动检测 (CUDA → MPS → CPU)

---

## 一、脚本总览

| 脚本 | 用途 | 迁移链路 | 输出目录 |
|------|------|----------|----------|
| `Scripts/Train.sh` | 三数据集**独立训练**（非迁移） | 无 | `runs/train/` |
| `Scripts/TrainTwostage.sh` | **两阶段迁移**: Crop → PDT/LL | Crop → PDT/LL | `runs/TrainTwostage/` |
| `Scripts/TrainTwostageLH.sh` | **两阶段迁移**: Crop → PDT/LH（新增） | Crop → PDT/LH | `runs/TrainTwostageLH/` |
| `Scripts/TrainThreestage.sh` | **三阶段连续迁移** | Crop → PDT/LL → PDT/LH | `runs/TrainThreestage/` |
| `Scripts/TrainPDTScratch.sh` | PDT/LL **从头训练**（对照组） | 无 | `runs/TrainTwostage/PDT_scratch_baseline/` |
| `Scripts/TrainPDTLHScratch.sh` | PDT/LH **从头训练**（对照组，新增） | 无 | `runs/TrainTwostage/PDT_LH_scratch_baseline/` |
| `Scripts/Val.sh` | 验证/测试 | - | `runs/val/` |
| `Scripts/Benchmark.sh` | 推理速度基准测试 | - | 终端输出 |

---

## 二、主实验

### 2.1 三阶段连续迁移（推荐主实验）

```bash
bash Scripts/TrainThreestage.sh
```

**迁移链路**: **Crop（27类近景）→ PDT/LL（1类远景低分辨率）→ PDT/LH（1类远景高分辨率）**

| 阶段 | 数据集 | 初始化权重 | 输出权重 |
|------|--------|------------|----------|
| Stage 1 | `datasets/Crop/data.yaml` | YAML 架构初始化 | `runs/TrainThreestage/Crop_stage1/weights/best.pt` |
| Stage 2 | `datasets/PDT/LL/PDT.yaml` | Stage 1 `best.pt` | `runs/TrainThreestage/PDT_LL_stage2/weights/best.pt` |
| Stage 3 | `datasets/PDT/LH/PDT.yaml` | Stage 2 `best.pt` | `runs/TrainThreestage/PDT_LH_stage3/weights/best.pt` |

> ✅ **支持中断续跑**: 检测到对应 `best.pt` 已存在则自动跳过该阶段。

### 2.2 两阶段迁移（Crop → PDT/LL）

```bash
bash Scripts/TrainTwostage.sh
```

**迁移链路**: **Crop（27类近景）→ PDT/LL（1类远景低分辨率）**

**训练参数**: `epochs=150, imgsz=640, batch=16, lr0=0.001`

**输出权重**:
- `runs/TrainTwostage/Crop_stage1/weights/best.pt`
- `runs/TrainTwostage/PDT_stage2_finetune/weights/best.pt`

### 2.3 两阶段迁移（Crop → PDT/LH，新增）

```bash
bash Scripts/TrainTwostageLH.sh
```

**迁移链路**: **Crop（27类近景）→ PDT/LH（1类远景高分辨率）**

**训练参数**: `epochs=150, imgsz=640, batch=16, lr0=0.001`

**输出权重**:
- `runs/TrainTwostageLH/Crop_stage1/weights/best.pt`
- `runs/TrainTwostageLH/PDT_LH_stage2/weights/best.pt`

### 2.4 三数据集独立训练（非迁移）

```bash
bash Scripts/Train.sh
```

依次独立训练 Crop、PDT/LH、PDT/LL，**各数据集互不传递权重**。

**自定义单数据集**:
```bash
# 仅训练 Crop
bash Scripts/Train.sh --multi-datasets "Crop:datasets/Crop/data.yaml:150:16"

# 仅训练 PDT/LH
bash Scripts/Train.sh --multi-datasets "PDT_LH:datasets/PDT/LH/PDT.yaml:150:16"

# 仅训练 PDT/LL
bash Scripts/Train.sh --multi-datasets "PDT_LL:datasets/PDT/LL/PDT.yaml:150:16"
```

---

## 三、对照实验

### 3.1 PDT/LL 从头训练（迁移效果对照）

```bash
bash Scripts/TrainPDTScratch.sh
```

**输出**: `runs/TrainTwostage/PDT_scratch_baseline/weights/best.pt`

> 📊 **本地验证结果**（10 epochs 采样数据）: 两阶段迁移相比从头训练 **mAP50 +135%，Recall +79%**。

### 3.2 PDT/LH 从头训练（迁移效果对照，新增）

```bash
bash Scripts/TrainPDTLHScratch.sh
```

**输出**: `runs/TrainTwostage/PDT_LH_scratch_baseline/weights/best.pt`

> 📊 **设计目的**: 评估 “Crop → PDT/LH” 两阶段迁移以及 “Crop → PDT/LL → PDT/LH” 三阶段连续迁移在 PDT/LH 高分辨率数据集上的性能增益（迁移学习 vs 从头训练）。

---

## 四、验证与测试

```bash
# 默认（Crop 测试集）
bash Scripts/Val.sh --model runs/TrainThreestage/PDT_LH_stage3/weights/best.pt \
                    --data datasets/PDT/LH/PDT.yaml

# 指定 split
bash Scripts/Val.sh --model <权重路径> --data <yaml路径> --split test

# 可选参数
# --device 0       指定 GPU
# --save-json true 保存 COCO 格式预测结果
# --plots true     保存可视化图表
```

---

## 五、性能基准测试

```bash
bash Scripts/Benchmark.sh

# 可选参数
bash Scripts/Benchmark.sh --device 0 --imgsz 640 --warmup 10 --iterations 50
```

---

## 六、结果对比

由于之前有写好的对比程序，可以用于输出多实验指标对比。

```bash
# 对比三阶段迁移 vs 两阶段迁移 (LL) vs 两阶段迁移 (LH) vs LL从头训练 vs LH从头训练
python3 Scripts/compare_results.py \
    runs/TrainThreestage/PDT_LH_stage3 \
    runs/TrainTwostage/PDT_stage2_finetune \
    runs/TrainTwostageLH/PDT_LH_stage2 \
    runs/TrainTwostage/PDT_scratch_baseline \
    runs/TrainTwostage/PDT_LH_scratch_baseline

# 也可直接传 results.csv 路径
python3 Scripts/compare_results.py runs/TrainTwostage/PDT_scratch_baseline/results.csv
```

**输出列**: `Epoch | Precision | Recall | mAP50 | mAP50-95 | Val Cls Loss`

---

## 七、训练参数速查

| 参数 | 正式训练（服务器） |
|------|--------------------|
| `epochs` | 150 |
| `imgsz` | 640 |
| `batch` | 16 |
| `workers` | 8 |
| `cache` | True |
| `lr0`（迁移阶段） | 0.001 |
| `device` | 自动检测（CUDA → MPS → CPU） |

---

## 八、推荐实验顺序（服务器执行）

```bash
# Step 1：三阶段连续迁移（主实验）
bash Scripts/TrainThreestage.sh

# Step 2：两阶段迁移（Crop → PDT/LL）
bash Scripts/TrainTwostage.sh

# Step 3：两阶段迁移（Crop → PDT/LH，新增）
bash Scripts/TrainTwostageLH.sh

# Step 4a：PDT/LL 从头训练对照
bash Scripts/TrainPDTScratch.sh

# Step 4b：PDT/LH 从头训练对照（新增）
bash Scripts/TrainPDTLHScratch.sh

# Step 5：独立训练（消融）
bash Scripts/Train.sh

# Step 6：汇总对比
python3 Scripts/compare_results.py \
    runs/TrainThreestage/PDT_LH_stage3 \
    runs/TrainTwostage/PDT_stage2_finetune \
    runs/TrainTwostageLH/PDT_LH_stage2 \
    runs/TrainTwostage/PDT_scratch_baseline \
    runs/TrainTwostage/PDT_LH_scratch_baseline
```

---

**文档版本**: v2.1  
**最后更新**: 2026-06-30  
**维护者**: AgriYOLO Team
