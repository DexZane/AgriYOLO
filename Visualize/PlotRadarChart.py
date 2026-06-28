"""
Radar Chart - Fixed Normalization (No Extreme Compression)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False

COLORS = {
    'AgriYOLO': '#E74C3C',
    'YOLOv10s': '#3498DB',
    'YOLOv8s': '#27AE60',
    'YOLOv9c': '#9B59B6',
    'YOLOv5s': '#F39C12',
    'RT_DETR_l': '#95A5A6'
}

def main():
    df1 = pd.read_csv("logs/sota_comparison_final_v2.csv")
    df2 = pd.read_csv("logs/model_complexity.csv")
    df3 = pd.read_csv("logs/speed_benchmark.csv")
    
    df = df1.merge(df2, on="Model").merge(df3, on="Model")
    df = df[df['mAP50-95'] > 0.1]
    
    # Better normalization: Map to 0.3-1.0 range (not 0-1)
    # This prevents any model from appearing as a tiny dot
    def norm_minmax(col, invert=False):
        if invert:
            col = 1 / (col + 0.001)
        min_val, max_val = col.min(), col.max()
        if max_val == min_val:
            return pd.Series([0.7] * len(col))
        # Map to 0.3-1.0 range so minimum is still visible
        return 0.3 + 0.7 * (col - min_val) / (max_val - min_val)
    
    df['v_map'] = norm_minmax(df['mAP50-95'])
    df['v_param'] = norm_minmax(df['Parameters (M)'], invert=True)
    df['v_flops'] = norm_minmax(df['GFLOPs'], invert=True)
    df['v_speed'] = norm_minmax(df['Latency (ms)'], invert=True)
    
    labels = ['mAP', 'Params↓', 'FLOPs↓', 'Speed↑']
    n = len(labels)
    angles = [i / n * 2 * np.pi for i in range(n)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    for _, row in df.iterrows():
        model = row['Model']
        vals = [row['v_map'], row['v_param'], row['v_flops'], row['v_speed']]
        vals += vals[:1]
        
        color = COLORS.get(model, '#7FDBFF')
        is_ours = 'Agri' in model
        lw = 2.5 if is_ours else 1.2
        ls = '-' if is_ours else '--'
        fill_alpha = 0.2 if is_ours else 0.05
        
        ax.plot(angles, vals, color=color, linewidth=lw, linestyle=ls, label=model)
        ax.fill(angles, vals, color=color, alpha=fill_alpha)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', pad=25)
    
    for i, label in enumerate(ax.get_xticklabels()):
        if 'Param' in label.get_text():
            label.set_verticalalignment('top')
    
    ax.set_yticklabels([])
    ax.set_ylim(0, 1.1)
    
    ax.set_title("Performance Comparison", fontsize=16, fontweight='bold', y=1.12)
    ax.legend(loc='upper left', bbox_to_anchor=(1.1, 1.0), fontsize=11, frameon=True)
    
    os.makedirs("picture", exist_ok=True)
    plt.tight_layout()
    plt.savefig("picture/performance_radar.png", dpi=300, bbox_inches='tight')
    plt.savefig("picture/performance_radar.pdf", bbox_inches='tight')
    plt.savefig("picture/performance_radar.svg", bbox_inches='tight')
    print("✅ Radar chart saved")

if __name__ == "__main__":
    main()
