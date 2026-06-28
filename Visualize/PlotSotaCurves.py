
import os
import sys
import glob
import pandas as pd
import matplotlib
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def plot_sota_curves_only(output_root="SOTA_Comparisons"):
    """
    Standalone function to plot training curves from existing logs.
    It looks for 'results.csv' in SOTA_Comparisons/*/results.csv
    """
    print(f"🔍 Searching for training logs in {output_root}...")
    
    # More robust glob pattern to find results.csv in subfolders
    csv_paths = glob.glob(os.path.join(output_root, "*", "results.csv"))
    
    if not csv_paths:
        print(f"❌ No 'results.csv' files found in {os.path.abspath(output_root)}")
        print("   Structure should be: SOTA_Comparisons/<ModelName>/results.csv")
        return

    print(f"✅ Found {len(csv_paths)} logs: {[os.path.basename(os.path.dirname(p)) for p in csv_paths]}")

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    # Align fonts with bubble chart
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12
    plt.figure(figsize=(10, 6))
    
    plotted_count = 0
    for csv_file in csv_paths:
        try:
            model_name = os.path.basename(os.path.dirname(csv_file))
            df = pd.read_csv(csv_file)
            # Clean column names (remove spaces)
            df.columns = [c.strip() for c in df.columns]
            
            # Find the accuracy column (usually metrics/mAP50-95(B))
            # Different YOLO versions might have slightly different headers
            map_col = next((c for c in df.columns if "mAP50-95" in c), None)
            
            if map_col:
                sns.lineplot(x=df.index, y=df[map_col], label=model_name, linewidth=2)
                plotted_count += 1
            else:
                print(f"⚠️ mAP50-95 column not found in {model_name}")
                
        except Exception as e:
            print(f"⚠️ Error processing {csv_file}: {e}")

    if plotted_count > 0:
        plt.title("Different Models Training Progress (mAP50-95)", fontsize=15, weight='bold')
        plt.xlabel("Epoch")
        plt.ylabel("mAP50-95")
        plt.legend(title="Models", loc='lower right')
        plt.tight_layout()
        
        save_dir = "picture"
        os.makedirs(save_dir, exist_ok=True)
        
        plt.savefig(os.path.join(save_dir, "sota_comparison_curve.pdf"), format='pdf', bbox_inches='tight')
        plt.savefig(os.path.join(save_dir, "sota_comparison_curve.png"), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(save_dir, "sota_comparison_curve.svg"), format='svg', bbox_inches='tight')
        print(f"\n✅ Curves generated successfully! Saved to {save_dir}/sota_comparison_curve.png")
    else:
        print("❌ No valid data plotted.")

if __name__ == "__main__":
    plot_sota_curves_only()
