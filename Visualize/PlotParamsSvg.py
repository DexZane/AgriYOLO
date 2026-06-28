"""
Model Parameter Trend Plotter
-----------------------------
Purpose:
    Generates the "Ablation Stage vs Parameters" trend chart (Line plot) in PDF/SVG/PNG format.

Usage:
    python plot_params_svg.py
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_params_svg():
    # 1. Style Setup (Science/Nature Theme)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    sns.set_context("paper", font_scale=1.5)
    
    # 2. Load Data
    log_file = "logs/model_complexity.csv"
    if not os.path.exists(log_file):
        print(f"❌ Log file not found: {log_file}")
        print("💡 Please run 'python experiments/flops.py' first.")
        return

    df = pd.read_csv(log_file)
    
    # 3. Create High-Res Figure
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    # 4. Standard Professional Line Plot
    # Highlighting AgriYOLO with a specific color and marker
    color_main = '#2980B9'
    sns.lineplot(
        data=df, x='Model', y='Parameters (M)', 
        marker='o', markersize=12, linewidth=3,
        color=color_main, ax=ax, label='Model Complexity'
    )
    
    # Fill under the curve for a more "Modern Science" look
    ax.fill_between(range(len(df)), df['Parameters (M)'], alpha=0.1, color=color_main)
    
    # 5. Data Annotations
    for i, point in enumerate(df['Parameters (M)']):
        ax.annotate(f"{point:.1f}M", 
                    (i, point), 
                    textcoords="offset points", 
                    xytext=(0, 15), 
                    ha='center', 
                    fontsize=12, weight='bold', color='#2C3E50')

    # Formatting Labels and Ticks
    ax.set_title("Architecture Efficiency: Parameter Count Analysis", fontsize=18, weight='bold', pad=25)
    ax.set_xlabel("State-of-the-Art Models", fontsize=14, weight='semibold')
    ax.set_ylabel("Total Parameters (Millions)", fontsize=14, weight='semibold')
    
    # SCI Tick markers
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.xticks(rotation=15)
    
    # Adjust Y limit for headroom
    plt.ylim(min(df['Parameters (M)']) - 2, max(df['Parameters (M)']) + 3)
    
    # Remove top/right spines and set grid
    sns.despine(offset=10, trim=True)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend().remove() # We only have one line, title is enough
    
    # 6. Save
    save_dir = "picture"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "model_params_trend.pdf"), format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, "model_params_trend.svg"), format='svg', bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, "model_params_trend.png"), dpi=300, bbox_inches='tight')
    
    print(f"✅ Created Parameter Trend Charts in {save_dir}")

if __name__ == "__main__":
    plot_params_svg()
