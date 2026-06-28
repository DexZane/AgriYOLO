"""
Qualitative Detection Contrast Generator
----------------------------------------
Purpose:
    Runs inference on random test images using both Baseline and AgriYOLO models 
    and stitches the results side-by-side for visual comparison. Outputs PDF/JPG.

Usage:
    python visualize_comparison.py
"""
import sys
import os
sys.path.append(os.getcwd()) # 确保能导入本地 ultralytics

import random
import glob
import cv2
import numpy as np
from ultralytics import YOLOv10
from pathlib import Path
from PIL import Image # For PDF saving

def draw_comparison(img_path, model_baseline, model_agriyolo, save_dir):
    """
    对同一张图片进行预测，并拼接成左右对比图
    """
    filename = Path(img_path).name
    
    # 1. 读取图片
    img = cv2.imread(img_path)
    if img is None:
        return
    
    # 2. Baseline 预测
    try:
        # predict 返回 Results 对象列表
        res_base = model_baseline.predict(img_path, conf=0.25, verbose=False)[0]
        # plot() 返回 BGR numpy array
        img_base = res_base.plot()
    except Exception as e:
        print(f"Error predicting Baseline on {filename}: {e}")
        img_base = img.copy()

    # 3. AgriYOLO 预测
    try:
        res_agri = model_agriyolo.predict(img_path, conf=0.25, verbose=False)[0]
        img_agri = res_agri.plot()
    except Exception as e:
        print(f"Error predicting AgriYOLO on {filename}: {e}")
        img_agri = img.copy()

    # 4. 拼接图片 (Horizontal Stack)
    # 添加标题栏
    h, w, c = img_base.shape
    header_h = 60
    canvas = np.zeros((h + header_h, w * 2, c), dtype=np.uint8)
    
    # 填充白色背景
    canvas[:, :] = (255, 255, 255)
    
    # 贴图
    canvas[header_h:, :w] = img_base
    canvas[header_h:, w:] = img_agri
    
    # 写字 (Baseline vs AgriYOLO)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "Baseline (Scratch)", (50, 40), font, 1.2, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "AgriYOLO (Scratch)", (w + 50, 40), font, 1.2, (0, 128, 0), 2, cv2.LINE_AA)
    
    # 保存
    out_path = os.path.join(save_dir, f"compare_{filename}")
    cv2.imwrite(out_path, canvas)
    
    # 同时也保存为 PDF (SCI 要求高分辨率格式)
    try:
        # OpenCV 是 BGR, PIL 需要 RGB
        canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        pdf_path = out_path.rsplit('.', 1)[0] + ".pdf"
        img_pil = Image.fromarray(canvas_rgb)
        img_pil.save(pdf_path, "PDF", resolution=300.0)
        print(f"   🖼️ Saved comparison: {out_path} and {pdf_path}")
    except Exception as e:
        print(f"⚠️ Failed to save PDF: {e}")
        print(f"   🖼️ Saved comparison: {out_path}")


def main():
    # === 配置区域 ===
    # 最好在云端运行，因为数据在那里
    # === 配置区域 ===
    # ⚠️ 脚本将尝试在以下路径自动寻找图片（自动纠错）：
    potential_paths = [
        r"E:/Desktop/datasets/Crop/test/images",
    ]
    
    test_imgs_dir = None
    for p in potential_paths:
        if os.path.exists(p) and len(glob.glob(os.path.join(p, "*.*"))) > 0:
            test_imgs_dir = p
            print(f"✅ Found images directory: {test_imgs_dir}")
            break
            
    weight_baseline = "AgriYOLO_Ablation/Baseline/weights/best.pt"
    weight_agriyolo = "AgriYOLO_Ablation/AgriYOLO_Full/weights/best.pt"
    output_dir = "results_output/qualitative_comparison"
    
    num_samples = 5 
    # ================

    if test_imgs_dir is None:
        print("❌ Could not find any images in the following paths:")
        for p in potential_paths:
            print(f"   - {p}")
        print("   👉 Please manually edit this script (line 68) with your correct local image folder path.")
        return

    os.makedirs(output_dir, exist_ok=True)

    print("🚀 Loading Models...")
    try:
        model_b = YOLOv10(weight_baseline)
        model_a = YOLOv10(weight_agriyolo)
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        print("   Make sure you are running this in the Project Root folder.")
        return

    # 获取所有图片列表
    all_imgs = glob.glob(os.path.join(test_imgs_dir, "*.jpg")) + \
               glob.glob(os.path.join(test_imgs_dir, "*.png"))
    
    if not all_imgs:
        print("❌ No images found in directory.")
        return

    # 随机采样
    selected_imgs = random.sample(all_imgs, min(len(all_imgs), num_samples))
    
    print(f"🎨 Generating {len(selected_imgs)} comparison images...")
    
    for img_path in selected_imgs:
        draw_comparison(img_path, model_b, model_a, output_dir)
        
    print(f"\n✅ Done! Images saved to {output_dir}")
    print("   Please download them for your thesis!")

if __name__ == "__main__":
    main()
