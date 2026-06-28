"""
Bubble Chart - Final Version with Better Fonts & Separated Legend
"""
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Set professional fonts
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
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for _, row in df.iterrows():
        model = row['Model']
        x = row['Latency (ms)']
        y = row['mAP50-95']
        size = row['GFLOPs'] * 10
        
        color = COLORS.get(model, '#7FDBFF')
        is_ours = 'Agri' in model
        alpha = 0.8 if is_ours else 0.6
        edge = 'black' if is_ours else 'gray'
        edge_width = 2 if is_ours else 1
        
        ax.scatter(x, y, s=size, c=color, alpha=alpha,
                   edgecolors=edge, linewidths=edge_width,
                   label=model, zorder=10 if is_ours else 5)
        
        # Text annotation above bubbles
        ax.annotate(model, (x, y), xytext=(0, 12), textcoords='offset points',
                    ha='center', fontsize=10, fontweight='bold' if is_ours else 'normal')
    
    ax.set_xlabel("Latency (ms) → Lower is Better", fontsize=13, fontweight='bold')
    ax.set_ylabel("mAP 50-95 ↑ Higher is Better", fontsize=13, fontweight='bold')
    ax.set_title("Speed-Accuracy Trade-off", fontsize=15, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # SEPARATED SIZE LEGEND - using text instead of overlapping bubbles
    # Create a separate legend showing size scale
    ax.annotate("Bubble Size = GFLOPs", xy=(0.98, 0.02), xycoords='axes fraction',
                ha='right', va='bottom', fontsize=10, fontstyle='italic',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax.set_xlim(left=0)
    y_min = df['mAP50-95'].min() - 0.015
    y_max = df['mAP50-95'].max() + 0.015
    ax.set_ylim(y_min, y_max)
    
    os.makedirs("picture", exist_ok=True)
    plt.tight_layout()
    plt.savefig("picture/speed_accuracy_bubble.png", dpi=300, bbox_inches='tight')
    plt.savefig("picture/speed_accuracy_bubble.pdf", bbox_inches='tight')
    plt.savefig("picture/speed_accuracy_bubble.svg", bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
