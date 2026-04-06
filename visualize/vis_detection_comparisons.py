import sys
import os
# Force usage of local ultralytics for YOLOv10 compatibility
sys.path.insert(0, os.getcwd())

import random
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO, RTDETR, YOLOv10
import glob

# Configuration
DATASET_ROOT = r"E:\Desktop\datasets\Crop\test\images"
OUTPUT_DIR = "picture/visual_comparisons"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Select random images (Fixed seed for consistency)
random.seed(42)
all_images = glob.glob(os.path.join(DATASET_ROOT, "*.jpg")) + glob.glob(os.path.join(DATASET_ROOT, "*.png"))
SELECTED_IMAGES = random.sample(all_images, 4) # Pick 4 representative images as requested

print(f"Selected {len(SELECTED_IMAGES)} images for visualization.")

# 1. Ablation Study Models
ABLATION_MODELS = {
    "Baseline (YOLOv10s)": r"e:\Desktop\yolov10-main\AgriYOLO_Ablation\AgriYOLO_Ablation\Baseline\weights\best.pt",
    "+ P2 Head": r"e:\Desktop\yolov10-main\AgriYOLO_Ablation\AgriYOLO_Ablation\Exp_P2\weights\best.pt",
    "+ ADBiFPN": r"e:\Desktop\yolov10-main\AgriYOLO_Ablation\AgriYOLO_Ablation\Exp_P2_BiFPN\weights\best.pt",
    "AgriYOLO (Final)": r"e:\Desktop\yolov10-main\AgriYOLO_Ablation\AgriYOLO_Ablation\Exp_P2_BiFPN_SimAM\weights\best.pt"
}

# 2. SOTA Comparison Models
SOTA_MODELS = {
    "YOLOv5s": r"e:\Desktop\yolov10-main\SOTA_Comparisons\YOLOv5s\weights\best.pt",
    "YOLOv8s": r"e:\Desktop\yolov10-main\SOTA_Comparisons\YOLOv8s\weights\best.pt",
    "YOLOv10s": r"e:\Desktop\yolov10-main\SOTA_Comparisons\YOLOv10s\weights\best.pt",
    "AgriYOLO": r"e:\Desktop\yolov10-main\AgriYOLO_Ablation\AgriYOLO_Ablation\Exp_P2_BiFPN_SimAM\weights\best.pt"
}

# Helper for manual drawing to bypass library bugs
def custom_draw(img_path, result):
    # Load original image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Get boxes
    boxes = result.boxes
    if boxes is None:
        return img
        
    for box in boxes:
        # Coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        
        # Filter weak detections (already done in predict, but extra safety)
        if conf < 0.25:
            continue
            
        # Draw Box
        color = (0, 255, 0) # Green for all
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Label (Optional, keeping it clean for paper)
        # label = f"{conf:.2f}"
        # cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
    return img

import traceback

# ... (rest of imports)

def run_visualization(model_dict, title_prefix, filename):
    num_models = len(model_dict)
    num_images = len(SELECTED_IMAGES)
    
    fig, axes = plt.subplots(num_images, num_models, figsize=(4 * num_models, 4 * num_images))
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    
    # Iterate over models (Columns)
    for col_idx, (model_name, model_path) in enumerate(model_dict.items()):
        print(f"Processing {model_name}...")
        
        # Load Model
        try:
            if "RT-DETR" in model_name:
                model = RTDETR(model_path)
            elif "Agri" in model_name or "v10" in model_name or "Baseline" in model_name or "BiFPN" in model_name or "P2" in model_name:
                model = YOLOv10(model_path)
            else:
                model = YOLO(model_path)
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            continue

        # Prepare arguments (YOLOv10 is NMS-free)
        predict_args = {"conf": 0.25, "verbose": False}
        # If using YOLOv10 class, we usually rely on its internal post-processing.
        # But for v5/v8 using YOLO class, passing IOU is standard.
        if isinstance(model, YOLO) and not isinstance(model, YOLOv10):
             predict_args["iou"] = 0.45
             
        # Predict
        for row_idx, img_path in enumerate(SELECTED_IMAGES):
            try:
                # Run inference
                results = model.predict(img_path, **predict_args)
                
                # Manual Plot
                if len(results) > 0:
                    res_plot = custom_draw(img_path, results[0])
                    
                    ax = axes[row_idx, col_idx] if num_images > 1 else axes[col_idx]
                    ax.imshow(res_plot)
                    ax.axis('off')
                
            except Exception as e:
                print(f"❌ FAILED {model_name} img {row_idx}: {e}")
                traceback.print_exc()
                ax = axes[row_idx, col_idx] if num_images > 1 else axes[col_idx]
                ax.text(0.5, 0.5, "Error", ha='center')
                ax.axis('off')

            if row_idx == 0:
                ax.set_title(model_name, fontsize=14, fontweight='bold', pad=10)
    
    # Save
    out_path = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved {title_prefix} visualization to {out_path}")

print("\n--- Generating Ablation Visuals ---")
run_visualization(ABLATION_MODELS, "Ablation Study", "ablation_vis_comparison.png")

print("\n--- Generating SOTA Visuals ---")
run_visualization(SOTA_MODELS, "SOTA Comparison", "sota_vis_comparison.png")
