"""
Comprehensive SOTA model comparison with full-split COCO-style evaluation.

This script trains a set of detector variants and evaluates them on the full
YOLO test split. Ground-truth JSON is generated from every test image, using
the original image sizes, so missed detections remain part of the evaluation.
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml
from PIL import Image


ROOT_PATH = Path(__file__).resolve().parents[1]
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

from ultralytics import RTDETR, YOLO, YOLOv10


IMAGE_SUFFIXES = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}


def xywhn_to_xywh(x, y, w, h, width, height):
    return x * width, y * height, w * width, h * height


def xywh_center_to_topleft(x, y, w, h):
    return x - w / 2, y - h / 2, w, h


def resolve_path(data_yaml_path, path_value, base_path=None):
    path = Path(path_value)
    if path.is_absolute():
        return path
    root = Path(base_path) if base_path else Path(data_yaml_path).resolve().parent
    return (root / path).resolve()


def list_split_images(data_yaml_path, split="test"):
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base_path = resolve_path(data_yaml_path, cfg.get("path", "."), None) if cfg.get("path") else Path(data_yaml_path).resolve().parent
    split_entry = cfg.get(split) or cfg.get("val")
    if split_entry is None:
        raise KeyError(f"Split '{split}' not found in {data_yaml_path}")
    if isinstance(split_entry, (list, tuple)):
        raise TypeError(f"Split '{split}' must resolve to a single path, got {type(split_entry).__name__}")

    split_path = resolve_path(data_yaml_path, split_entry, base_path)
    if split_path.is_dir():
        image_files = sorted(path for path in split_path.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    elif split_path.suffix.lower() == ".txt":
        image_files = []
        for line in split_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            image_path = Path(line)
            if not image_path.is_absolute():
                image_path = (split_path.parent / image_path).resolve()
            image_files.append(image_path)
    else:
        image_files = [split_path]
    return cfg, image_files


def image_id_from_path(image_path):
    stem = Path(image_path).stem
    return int(stem) if stem.isnumeric() else stem


def label_path_from_image(image_path):
    image_path = Path(image_path)
    parts = list(image_path.parts)
    if "images" in parts:
        index = len(parts) - 1 - parts[::-1].index("images")
        parts[index] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def generate_gt_json(data_yaml_path, split="test", output_json="gt.json"):
    """Generate a COCO-style GT file from the full YOLO split."""
    cfg, image_files = list_split_images(data_yaml_path, split=split)
    coco_data = {"images": [], "annotations": [], "categories": []}

    names = cfg.get("names", {})
    if isinstance(names, list):
        names = {i: name for i, name in enumerate(names)}
    for key, value in names.items():
        coco_data["categories"].append({"id": int(key), "name": value})

    ann_id = 1
    for image_path in image_files:
        image_path = Path(image_path)
        image_id = image_id_from_path(image_path)
        with Image.open(image_path) as image:
            width, height = image.size

        coco_data["images"].append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "height": height,
                "width": width,
            }
        )

        label_path = label_path_from_image(image_path)
        if not label_path.exists():
            continue

        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                x_c, y_c, w, h = xywhn_to_xywh(
                    float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]), width, height
                )
                x1, y1, w, h = xywh_center_to_topleft(x_c, y_c, w, h)
                coco_data["annotations"].append(
                    {
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": cls_id,
                        "bbox": [x1, y1, w, h],
                        "area": w * h,
                        "iscrowd": 0,
                    }
                )
                ann_id += 1

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(coco_data, f)
    return output_json


def evaluate_small_objects(weight_path, model_name, data_yaml_path):
    """Evaluate a model on the full test split using pycocotools."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    print(f"Running fine-grained evaluation for {model_name}...")

    if "rtdetr" in model_name.lower():
        model = RTDETR(weight_path)
    elif "v10" in model_name.lower() or "agriyolo" in model_name.lower():
        model = YOLOv10(weight_path)
    else:
        model = YOLO(weight_path)

    project_run = Path("runs") / "sota_eval" / model_name
    model.val(data=data_yaml_path, split="test", save_json=True, plots=True, verbose=False, project=str(project_run), name="val")

    pred_json = project_run / "val" / "predictions.json"
    if not pred_json.exists():
        return {"Model": model_name, "mAP50": 0, "mAP50-95": 0, "mAP_small": 0}

    with open(pred_json, "r", encoding="utf-8") as f:
        preds = json.load(f)
    if not preds:
        return {"Model": model_name, "mAP50": 0, "mAP50-95": 0, "mAP_small": 0}

    gt_json = project_run / "gt_temp.json"
    generate_gt_json(data_yaml_path, split="test", output_json=str(gt_json))

    try:
        coco_gt = COCO(str(gt_json))
        coco_dt = coco_gt.loadRes(str(pred_json))
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.params.imgIds = [img["id"] for img in coco_gt.dataset["images"]]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return {
            "Model": model_name,
            "mAP50": coco_eval.stats[1],
            "mAP50-95": coco_eval.stats[0],
            "mAP_small": coco_eval.stats[3],
        }
    except Exception as exc:
        print(f"Evaluation failed for {model_name}: {exc}")
        return {"Model": model_name, "mAP50": 0, "mAP50-95": 0, "mAP_small": 0}


def plot_sota_curves(output_root):
    """Generate comparative training curves for all SOTA models."""
    print("\nGenerating SOTA comparison charts...")
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    csv_paths = glob.glob(os.path.join(output_root, "*", "results.csv"))
    if not csv_paths:
        print("No results.csv found for plotting.")
        return

    plt.figure(figsize=(10, 6))
    for csv_file in csv_paths:
        model_name = os.path.basename(os.path.dirname(csv_file))
        if "rtdetr" in model_name.lower():
            continue

        df = pd.read_csv(csv_file)
        df.columns = [column.strip() for column in df.columns]
        map_col = next((column for column in df.columns if "mAP50-95" in column), None)
        if map_col:
            sns.lineplot(x=df["epoch"], y=df[map_col], label=model_name, linewidth=2)

    plt.title("SOTA Models Training Progress (mAP50-95)", fontsize=15)
    plt.xlabel("Epoch")
    plt.ylabel("mAP50-95")
    plt.legend(title="Models", loc="lower right")
    plt.tight_layout()

    save_dir = Path("picture")
    save_dir.mkdir(exist_ok=True)
    plt.savefig(save_dir / "sota_comparison_curve.pdf", format="pdf", bbox_inches="tight")
    plt.savefig(save_dir / "sota_comparison_curve.svg", format="svg", bbox_inches="tight")
    plt.savefig(save_dir / "sota_comparison_curve.png", dpi=300, bbox_inches="tight")
    print(f"SOTA comparison curves saved to {save_dir}: .pdf, .svg, .png")
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate SOTA detection models.")
    parser.add_argument("--data", default="data/Crop/data.yaml", help="Path to the YOLO dataset yaml.")
    parser.add_argument("--epochs", type=int, default=150, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--device", default="0", help="Training device, e.g. 0 or cpu.")
    parser.add_argument("--output-root", default="SOTA_Comparisons", help="Directory used to store training runs.")
    return parser.parse_args()


def main(args):
    competitors = [
        {
            "name": "AgriYOLO",
            "cfg": "ultralytics/cfg/models/v10/yolov10s_TAL_FFN.yaml",
            "weights": "yolov10s.pt",
            "type": "v10",
        },
        {
            "name": "YOLOv10s",
            "cfg": "ultralytics/cfg/models/v10/yolov10s_baseline.yaml",
            "weights": "yolov10s.pt",
            "type": "v10",
        },
        {
            "name": "YOLOv8s",
            "cfg": "ultralytics/cfg/models/v8/yolov8.yaml",
            "weights": "yolov8s.pt",
            "type": "v8",
        },
        {
            "name": "YOLOv9c",
            "cfg": "ultralytics/cfg/models/v9/yolov9c.yaml",
            "weights": "yolov9c.pt",
            "type": "v9",
        },
        {
            "name": "YOLOv5s",
            "cfg": "ultralytics/cfg/models/v5/yolov5.yaml",
            "weights": "yolov5su.pt",
            "type": "v5",
        },
    ]

    summary_list = []
    for competitor in competitors:
        print("\n" + "=" * 50 + f"\nProcessing {competitor['name']} (transfer learning)\n" + "=" * 50)
        try:
            if competitor["name"] == "AgriYOLO":
                print(f"Loading custom architecture: {competitor['cfg']} with weights {competitor['weights']}")
                model = YOLOv10(competitor["cfg"]) if competitor["type"] == "v10" else YOLO(competitor["cfg"])
                model.load(competitor["weights"])
            else:
                print(f"Loading pretrained model: {competitor['weights']}")
                if competitor["type"] == "rtdetr":
                    model = RTDETR(competitor["weights"])
                elif competitor["type"] == "v10":
                    model = YOLOv10(competitor["weights"])
                else:
                    model = YOLO(competitor["weights"])
        except Exception as exc:
            print(f"Failed to load specific weights, falling back to config: {exc}")
            if competitor["type"] == "rtdetr":
                model = RTDETR(competitor["cfg"])
            elif competitor["type"] == "v10":
                model = YOLOv10(competitor["cfg"])
            else:
                model = YOLO(competitor["cfg"])

        best_weight = Path(args.output_root) / competitor["name"] / "weights" / "best.pt"
        if best_weight.exists():
            print(f"Found existing weights for {competitor['name']}, skipping training...")
        else:
            model.train(
                data=args.data,
                epochs=args.epochs,
                imgsz=args.imgsz,
                device=args.device,
                project=args.output_root,
                name=competitor["name"],
                pretrained=True,
            )

        if best_weight.exists():
            summary_list.append(evaluate_small_objects(str(best_weight), competitor["name"], args.data))

    df = pd.DataFrame(summary_list)
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    output_csv = log_dir / "sota_comparison_final_v2.csv"
    df.to_csv(output_csv, index=False)

    plot_sota_curves(args.output_root)

    print(f"\nSOTA comparison complete. Summary saved to {output_csv}")
    if not df.empty:
        print(df.to_markdown(index=False))


if __name__ == "__main__":
    parsed_args = parse_args()
    if os.path.exists(parsed_args.data):
        main(parsed_args)
    else:
        print(f"'{parsed_args.data}' not found. Please check data config path.")
