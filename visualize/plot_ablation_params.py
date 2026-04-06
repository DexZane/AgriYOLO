"""
消融实验模型参数量对比折线图
----------------------------
数据来源: README.md 消融实验表格 + ablation_study.py 配置
对应 YAML 配置文件:
  Stage 1: yolov10s_baseline.yaml    (Baseline)
  Stage 2: yolov10s_P2_ADSA.yaml     (+ ADSA)
  Stage 3: yolov10s_P2_CADFM.yaml    (+ TAL-FFN)
  Stage 4: yolov10s_TAL_FFN.yaml     (+ SimAM = AgriYOLO)
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 中文字体设置
plt.rcParams['font.family'] = 'STFangsong'
plt.rcParams['axes.unicode_minus'] = False

# ========== 消融实验数据 ==========
stages = [
    'Baseline\n(YOLOv10s+PANet)',
    '+ ADSA\n(P2 检测头)',
    '+ TAL-FFN\n(CADFM+DSConv)',
    '+ SimAM\n(AgriYOLO)',
]

params = [8.09, 6.50, 6.42, 6.24]      # 参数量 (M)
mAP50  = [98.70, 98.80, 98.80, 98.90]  # mAP50 (%)


def main():
    fig, ax1 = plt.subplots(figsize=(10, 6))
    x = np.arange(len(stages))

    # ---------- 参数量折线 (主 Y 轴) ----------
    color_params = '#E74C3C'
    line1 = ax1.plot(
        x, params,
        color=color_params, marker='o', markersize=10,
        linewidth=2.5, label='参数量 (M)', zorder=5,
    )
    for i, (xi, yi) in enumerate(zip(x, params)):
        ax1.annotate(
            f'{yi:.2f}M',
            (xi, yi),
            textcoords='offset points',
            xytext=(0, 14 if i == 0 else -18),
            ha='center', fontsize=11, fontweight='bold',
            color=color_params,
        )

    ax1.set_xlabel('消融实验阶段', fontsize=14, fontweight='bold', labelpad=10)
    ax1.set_ylabel('参数量 (M)', fontsize=13, fontweight='bold', color=color_params)
    ax1.tick_params(axis='y', labelcolor=color_params, labelsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(stages, fontsize=11)
    ax1.set_ylim(5.5, 9.0)

    # ---------- mAP50 折线 (副 Y 轴) ----------
    ax2 = ax1.twinx()
    color_map = '#3498DB'
    line2 = ax2.plot(
        x, mAP50,
        color=color_map, marker='s', markersize=10,
        linewidth=2.5, linestyle='--', label='mAP50 (%)', zorder=5,
    )
    for i, (xi, yi) in enumerate(zip(x, mAP50)):
        ax2.annotate(
            f'{yi:.2f}%',
            (xi, yi),
            textcoords='offset points',
            xytext=(0, 14),
            ha='center', fontsize=11, fontweight='bold',
            color=color_map,
        )

    ax2.set_ylabel('mAP50 (%)', fontsize=13, fontweight='bold', color=color_map)
    ax2.tick_params(axis='y', labelcolor=color_map, labelsize=11)
    ax2.set_ylim(98.5, 99.1)

    # ---------- 参数量下降标注 ----------
    drop_pct = (params[0] - params[-1]) / params[0] * 100
    ax1.annotate(
        f'参数量减少 {drop_pct:.1f}%',
        xy=(1.5, (params[0] + params[-1]) / 2),
        fontsize=12, fontstyle='italic', color='#555555',
        ha='center',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF9C4', edgecolor='#F9A825', alpha=0.9),
    )

    # ---------- 图例 & 网格 ----------
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', fontsize=12,
               framealpha=0.9, edgecolor='gray')

    ax1.set_title('消融实验 — 模型参数量与 mAP50 对比', fontsize=16, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, linestyle='--')

    # ---------- 高亮最终模型 ----------
    ax1.axvspan(x[-1] - 0.3, x[-1] + 0.3, alpha=0.08, color='#E74C3C')

    # ---------- 保存 ----------
    save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "picture")
    os.makedirs(save_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "ablation_params_comparison.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, "ablation_params_comparison.pdf"), bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, "ablation_params_comparison.svg"), bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    main()
