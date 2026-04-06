"""
Learning Curve and Bar Chart Generator
--------------------------------------
Purpose:
    Reads 'results.csv' from all ablation folders and plots comparative learning curves 
    (mAP vs Epoch) and final performance bar charts. Supports PDF/PNG output.

Usage:
    python plot_comparison.py
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob

# 设置绘图风格
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

def plot_all_models(root_dir="AgriYOLO_Ablation"):
    print(f"🔍 Searching for results in {root_dir}...")
    
    # 查找所有实验文件夹下的 results.csv
    # 递归查找，匹配 pattern: root_dir/*/**/results.csv
    csv_paths = glob.glob(os.path.join(root_dir, "*", "**", "results.csv"), recursive=True) + \
                glob.glob(os.path.join(root_dir, "*", "results.csv"), recursive=False)
    
    # 去重
    csv_paths = sorted(list(set(csv_paths)))
    
    if not csv_paths:
        print("❌ No results.csv files found!")
        return

    print(f"✅ Found {len(csv_paths)} result files.")

    all_data = []
    summary_metrics = []

    plt.figure(figsize=(12, 8))
    
    # 颜色调色板
    palette = sns.color_palette("husl", len(csv_paths))

    for idx, csv_file in enumerate(csv_paths):
        try:
            # 提取模型名称 (取上一级文件夹名)
            # 例如: .../AgriYOLO_Full/results.csv -> AgriYOLO_Full
            model_name = os.path.basename(os.path.dirname(csv_file))
            
            # 排除带有 TEST 的验证结果文件夹，只看训练日志
            if "_TEST" in model_name:
                continue

            # 读取 CSV (YOLO 输出的 CSV 列名通常带有空格，需要处理)
            df = pd.read_csv(csv_file)
            df.columns = [c.strip() for c in df.columns] # 去除列名空格

            # 确定 mAP 列名 (不同版本可能略有差异)
            # 优先找 mAP50-95, 其次找 mAP50
            map_col = None
            for col in df.columns:
                if "mAP50-95" in col:
                    map_col = col
                    break
            
            if map_col is None:
                print(f"⚠️ mAP column not found in {model_name}, skipping.")
                continue

            # 1. 绘制折线图 (Learning Curve)
            # 对 epoch 进行平滑处理以便观察趋势
            sns.lineplot(x=df["epoch"], y=df[map_col], label=model_name, linewidth=2, color=palette[idx])
            
            # 收集最佳指标用于柱状图
            best_map = df[map_col].max()
            best_map50 = df[[c for c in df.columns if "mAP50" in c and "95" not in c][0]].max()
            
            summary_metrics.append({
                "Model": model_name,
                "mAP50-95": best_map,
                "mAP50": best_map50
            })
            
            print(f"   Processed {model_name}: Best mAP50-95 = {best_map:.4f}")

        except Exception as e:
            print(f"❌ Error processing {csv_file}: {e}")

    # --- 图表 1: 训练过程对比折线图 ---
    # --- 图表 1: 训练过程对比折线图 (SCI Professional) ---
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    sns.set_context("paper", font_scale=1.5)
    
    plt.title("Ablation Study: Training Convergence Analysis", fontsize=18, weight='bold', pad=20)
    plt.xlabel("Training Epochs", fontsize=14, weight='semibold')
    plt.ylabel("Precision (mAP@.5:.95) [%]", fontsize=14, weight='semibold')
    
    # Legend improvements
    plt.legend(title="Model Components", frameon=True, fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.3)
    sns.despine(offset=10)
    
    save_dir = "picture"
    os.makedirs(save_dir, exist_ok=True)
    
    plt.savefig(os.path.join(save_dir, "ablation_learning_curve.pdf"), format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, "ablation_learning_curve.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, "ablation_learning_curve.svg"), format='svg', bbox_inches='tight')
    print(f"📈 Saved peak SCI learning curve plots to {save_dir}")
    plt.close()

    # --- 图表 2: 最终精度对比柱状图 (Nature Style) ---
    if summary_metrics:
        df_summary = pd.DataFrame(summary_metrics)
        df_summary = df_summary.sort_values("mAP50-95", ascending=True)

        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        
        # Using a professional gradient or distinct Nature palette
        nature_palette = sns.color_palette("YlGnBu", n_colors=len(df_summary))
        barplot = sns.barplot(data=df_summary, x="Model", y="mAP50-95", palette=nature_palette, ax=ax, edgecolor='black')
        
        # Value annotations
        for p in barplot.patches:
            barplot.annotate(f'{p.get_height():.3f}', 
                             (p.get_x() + p.get_width() / 2., p.get_height() + 0.01), 
                             ha = 'center', va = 'center', 
                             fontsize=12, weight='bold', color='#2C3E50',
                             xytext = (0, 9), 
                             textcoords = 'offset points')

        plt.title("Experimental Synergy: Best Performance Highlights", fontsize=18, weight='bold', pad=25)
        plt.ylabel("Mean Average Precision (mAP50-95)", fontsize=14, weight='semibold')
        plt.xlabel("Model Configurations", fontsize=14, weight='semibold')
        plt.ylim(0, max(df_summary["mAP50-95"]) * 1.2)  # Dynamically adjust for clarity
        plt.xticks(rotation=25)
        
        sns.despine(offset=10, trim=True)
        plt.tight_layout()
        
        plt.savefig(os.path.join(save_dir, "ablation_bar_chart.pdf"), format='pdf', bbox_inches='tight')
        plt.savefig(os.path.join(save_dir, "ablation_bar_chart.png"), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(save_dir, "ablation_bar_chart.svg"), format='svg', bbox_inches='tight')
        print(f"📊 Saved peak SCI bar charts to {save_dir}")
        
        # Save summary CSV
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        summary_csv = os.path.join(log_dir, "ablation_models_summary.csv")
        df_summary.to_csv(summary_csv, index=False)
        print(f"💾 Saved summary data to: {summary_csv}")

if __name__ == "__main__":
    plot_all_models()
