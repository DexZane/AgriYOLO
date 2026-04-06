"""
Model Complexity Calculator (GFLOPs / Parameters)
-------------------------------------------------
Purpose:
    Calculates technical metrics for the model, including GFLOPs, 
    Total Parameters, and Layers. This is essential for the "Model Complexity" 
    section of the SCI paper.

Usage:
    python experiments/flops.py
"""
import os
import sys
import pandas as pd

# Fix for Windows MKL/Fortran Runtime Error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Ensure local 'ultralytics' is used (Must be before any ultralytics imports!)
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from ultralytics import YOLO, RTDETR, YOLOv10

def calculate_complexity():
    # Define models to analyze (Synced with run_sota_comparison.py)
    models_to_check = [
        {"name": "AgriYOLO",  "cfg": "ultralytics/cfg/models/v10/yolov10s_TAL_FFN.yaml", "type": "v10"},
        {"name": "YOLOv10s",  "cfg": "ultralytics/cfg/models/v10/yolov10s_baseline.yaml", "type": "v10"},
        {"name": "YOLOv8s",   "cfg": "ultralytics/cfg/models/v8/yolov8.yaml",    "type": "v8"},
        {"name": "YOLOv9c",   "cfg": "ultralytics/cfg/models/v9/yolov9c.yaml",   "type": "v9"},
        {"name": "YOLOv5s",   "cfg": "ultralytics/cfg/models/v5/yolov5.yaml",    "type": "v5"},
        {"name": "RT_DETR_l", "cfg": "ultralytics/cfg/models/rt-detr/rtdetr-l.yaml", "type": "rtdetr"}
    ]
    
    results = []
    
    for m in models_to_check:
        if not os.path.exists(m["cfg"]):
            print(f"⚠️ Configuration not found for {m['name']}: {m['cfg']}")
            continue

        print(f"🚀 Calculating complexity for: {m['name']}")
        
        # Load model architecture only (no weights needed for flops/params)
        if m["type"] == "rtdetr": model = RTDETR(m["cfg"])
        elif m["type"] == "v10": model = YOLOv10(m["cfg"])
        else: model = YOLO(m["cfg"])
        
        # Calculate params
        params = sum(p.numel() for p in model.model.parameters()) / 1e6
        
        # Calculate GFLOPs accurately using a dummy input
        try:
            import torch
            dummy_input = torch.randn(1, 3, 640, 640).to(next(model.model.parameters()).device)
            
            # Use thop if available, otherwise estimate from info()
            try:
                from thop import profile
                flops, _ = profile(model.model, inputs=(dummy_input,), verbose=False)
                gflops = flops / 1e9
            except ImportError:
                # Fallback: parse from model.info() output
                info = model.info(verbose=False)
                gflops = info[3] if isinstance(info, tuple) and len(info) > 3 else 0
        except Exception as e:
            print(f"  ⚠️ GFLOPs calculation failed: {e}")
            gflops = 0
        
        results.append({
            "Model": m["name"],
            "Parameters (M)": round(params, 2),
            "GFLOPs": round(gflops, 2)
        })

    # Save to CSV
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(log_dir, "model_complexity.csv"), index=False)
    
    print("\n✅ Model complexity logs saved to logs/model_complexity.csv")
    print(df.to_markdown(index=False))

if __name__ == "__main__":
    calculate_complexity()
