"""
Final Accuracy Comparison Plotter
---------------------------------
Purpose:
    Generates high-quality publication-ready graphs (PDF/SVG/PNG) comparing the 
    Baseline and AgriYOLO models' final mAP results.

Usage:
    python plot_mAP_svg.py
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def plot_accuracy_svg():
    # 1. Load Data from Comparison logs
    log_file = "logs/sota_comparison_final_v2.csv"
    if not os.path.exists(log_file):
        print(f"❌ Log file not found: {log_file}")
        print("💡 Please run 'python experiments/run_sota_comparison.py' first.")
        return

    df = pd.read_csv(log_file)
    
    # We focus on Baseline vs AgriYOLO for this specific bar chart
    # Filtering for clarity
    plot_df = df[df['Model'].isin(['YOLOv10s', 'AgriYOLO'])]
    if plot_df.empty:
        plot_df = df # Fallback to all models if specific ones not found
    
    # Convert mapping names for display
    plot_df['DisplayModel'] = plot_df['Model'].apply(lambda x: x + "\n(Ours)" if "Agri" in x else x + "\n(Baseline)")
    plot_df['mAP %'] = plot_df['mAP50-95'] * 100

    # 2. Design System Setup
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    sns.set_context("paper", font_scale=1.5)
    
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    # 3. Bar Chart with Nature Style Colors
    # Using a professional "Nature" palette: Muted grey vs Vibrant green
    nature_colors = ['#BDC3C7', '#27AE60'] 
    
    bars = sns.barplot(
        x='DisplayModel', 
        y='mAP %', 
        data=plot_df, 
        palette=nature_colors,
        edgecolor='black',
        linewidth=1.2,
        ax=ax
    )
    
    # 4. Precision Labels on top of bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}%', 
                    (p.get_x() + p.get_width() / 2., p.get_height() + 0.5), 
                    ha='center', va='center', 
                    fontsize=14, color='black', weight='bold',
                    xytext=(0, 9), 
                    textcoords='offset points')
        
    # 5. Improvement Highlighting
    if len(plot_df) >= 2:
        vals = plot_df['mAP %'].values
        diff = abs(vals[1] - vals[0])
        
        # Draw bracket
        x0 = ax.patches[0].get_x() + ax.patches[0].get_width()/2
        x1 = ax.patches[1].get_x() + ax.patches[1].get_width()/2
        y_pos = max(vals) + 8
        
        ax.plot([x0, x0, x1, x1], [y_pos-2, y_pos, y_pos, y_pos-2], color='black', lw=1.5)
        ax.text((x0+x1)/2, y_pos+1, f"+{diff:.1f}% mAP Improvement", 
                ha='center', va='bottom', fontsize=12, weight='bold', color='#C0392B')

    # Formatting Spines and Labels
    ax.set_title("Robustness Evaluation: mAP Comparison", fontsize=18, weight='bold', pad=30)
    ax.set_ylabel("Mean Average Precision (mAP@.5:.95) [%]", fontsize=14, weight='semibold')
    ax.set_xlabel("") 
    ax.set_ylim(0, 115) # Room for bracket
    
    # SCI Ticks
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%d%%'))
    
    sns.despine(ax=ax, offset=10, trim=True)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # 6. Save
    save_dir = "picture"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "accuracy_comparison.pdf"), format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, "accuracy_comparison.svg"), format='svg', bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, "accuracy_comparison.png"), dpi=300, bbox_inches='tight')
    
    print(f"✅ Created Accuracy Charts in {save_dir}")

if __name__ == "__main__":
    plot_accuracy_svg()
