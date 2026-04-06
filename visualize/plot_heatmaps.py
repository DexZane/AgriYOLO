
import cv2
import numpy as np
import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(os.getcwd()) 
import torch
import torch.nn.functional as F
from ultralytics import YOLOv10
import matplotlib.pyplot as plt

class YOLOv10GradCAM:
    def __init__(self, model_path, target_layer_name):
        self.model = YOLOv10(model_path).model
        self.model.eval()
        self.target_layer = self.find_layer(target_layer_name)
        
        self.gradients = None
        self.activations = None
        
        # Hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def find_layer(self, layer_name):
        # search layer by name
        for name, module in self.model.named_modules():
            if name == layer_name:
                return module
        raise ValueError(f"Layer {layer_name} not found.")

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor):
        output = self.model(input_tensor)
        # Average Feature Map
        cam = torch.mean(self.activations, dim=1).squeeze()
        cam = F.relu(cam)
        cam = cam.detach().cpu().numpy()
        cam = cv2.resize(cam, (640, 640))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

def plot_heatmaps():
    # Model Paths
    baseline_path = "weights/yolov10s.pt"
    agriyolo_path = "AgriYOLO_Ablation/AgriYOLO_Ablation/Exp_P2_BiFPN_SimAM/weights/best.pt"
    
    # Target Layers
    target_layer_baseline = "model.21" 
    target_layer_agriyolo = "model.23" # Adjusted layer index 
    
    # Test Image
    img_path = "AgriYOLO_Ablation/AgriYOLO_Ablation/Exp_P2_BiFPN_SimAM/val_batch0_labels.jpg"
    
    if not os.path.exists(img_path):
        print(f"❌ Test image not found at {img_path}")
        return
    if not os.path.exists(baseline_path):
        print(f"❌ Baseline weights not found at {baseline_path}")
        return
    if not os.path.exists(agriyolo_path):
        print(f"❌ AgriYOLO weights not found at {agriyolo_path}")
        return

    # Prepare Image
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_input = cv2.resize(img_rgb, (640, 640))
    img_tensor = torch.from_numpy(img_input).permute(2,0,1).float().unsqueeze(0) / 255.0
    
    print(f"🌡️ Generating COMPARATIVE heatmaps for {img_path}...")
    
    try:
        # 1. Baseline CAM
        print("   - Processing Baseline...")
        try:
             cam_baseline = YOLOv10GradCAM(baseline_path, target_layer_baseline)
             heatmap1 = cam_baseline.generate_heatmap(img_tensor)
             heatmap1_color = cv2.applyColorMap(np.uint8(255 * heatmap1), cv2.COLORMAP_JET)
             heatmap1_color = cv2.cvtColor(heatmap1_color, cv2.COLOR_BGR2RGB)
             result1 = cv2.addWeighted(img_input, 0.5, heatmap1_color, 0.5, 0)
        except Exception as e:
             print(f"Baseline generation failed: {e}")
             result1 = img_input # Fallback
        
        # 2. AgriYOLO CAM
        print("   - Processing AgriYOLO...")
        try:
            cam_agriyolo = YOLOv10GradCAM(agriyolo_path, target_layer_agriyolo)
            heatmap2 = cam_agriyolo.generate_heatmap(img_tensor)
            heatmap2_color = cv2.applyColorMap(np.uint8(255 * heatmap2), cv2.COLORMAP_JET)
            heatmap2_color = cv2.cvtColor(heatmap2_color, cv2.COLOR_BGR2RGB)
            result2 = cv2.addWeighted(img_input, 0.5, heatmap2_color, 0.5, 0)
        except Exception as e:
            print(f"AgriYOLO generation failed: {e}")
            result2 = img_input

        # Plot comparison
        plt.figure(figsize=(18, 6))
        
        plt.subplot(1, 3, 1)
        plt.title("Original Image", fontsize=16)
        plt.imshow(img_input)
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.title("Baseline (YOLOv10s)", fontsize=16)
        plt.imshow(result1)
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.title("AgriYOLO (Ours)", fontsize=16, fontweight='bold')
        plt.imshow(result2)
        plt.axis('off')
        
        save_dir = "picture/interpretability"
        os.makedirs(save_dir, exist_ok=True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "grad_cam_comparison.pdf"), format='pdf', bbox_inches='tight')
        plt.savefig(os.path.join(save_dir, "grad_cam_comparison.png"), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(save_dir, "grad_cam_comparison.svg"), format='svg', bbox_inches='tight')
        print(f"✅ Comparison saved to {save_dir}/grad_cam_comparison.png")
        
    except Exception as e:
        print(f"❌ Comparison generation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    plot_heatmaps()
