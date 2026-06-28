"""
Inference speed benchmark for the trained model checkpoints.

The benchmark uses raw PyTorch forward passes under `torch.inference_mode()`
to avoid mixing model latency with predictor-side preprocessing or postprocess
overhead.
"""

import argparse
import os
import sys
import time

import pandas as pd
import torch


ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_PATH not in sys.path:
    sys.path.insert(0, ROOT_PATH)

from ultralytics import RTDETR, YOLO, YOLOv10


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark raw model latency.")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu", help="Benchmark device.")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations.")
    parser.add_argument("--iterations", type=int, default=50, help="Measured iterations.")
    return parser.parse_args()


def get_model_wrapper(model_type, weight_path):
    if model_type == "rtdetr":
        return RTDETR(weight_path)
    if model_type == "v10":
        return YOLOv10(weight_path)
    return YOLO(weight_path)


def benchmark_speed(args):
    device = torch.device(f"cuda:{args.device}") if str(args.device).isdigit() and torch.cuda.is_available() else torch.device(args.device)
    models_to_test = [
        {"name": "AgriYOLO", "path": "SOTA_Comparisons/AgriYOLO/weights/best.pt", "type": "v10"},
        {"name": "YOLOv10s", "path": "SOTA_Comparisons/YOLOv10s/weights/best.pt", "type": "v10"},
        {"name": "YOLOv8s", "path": "SOTA_Comparisons/YOLOv8s/weights/best.pt", "type": "v8"},
        {"name": "YOLOv9c", "path": "SOTA_Comparisons/YOLOv9c/weights/best.pt", "type": "v9"},
        {"name": "YOLOv5s", "path": "SOTA_Comparisons/YOLOv5s/weights/best.pt", "type": "v5"},
        {"name": "RT_DETR_l", "path": "SOTA_Comparisons/RT_DETR_l/weights/best.pt", "type": "rtdetr"},
    ]

    results = []
    print(f"Starting speed benchmark on {device}...")

    for model_spec in models_to_test:
        print(f"Testing {model_spec['name']}...")
        if not os.path.exists(model_spec["path"]):
            print(f"  Warning: weights not found at {model_spec['path']}. Skipping...")
            continue

        try:
            wrapper = get_model_wrapper(model_spec["type"], model_spec["path"])
            backend = wrapper.model.to(device).eval()
            dummy_input = torch.randn(1, 3, args.imgsz, args.imgsz, device=device)

            with torch.inference_mode():
                for _ in range(args.warmup):
                    _ = backend(dummy_input)

                if device.type == "cuda":
                    torch.cuda.synchronize()
                start_time = time.time()
                for _ in range(args.iterations):
                    _ = backend(dummy_input)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                end_time = time.time()

            avg_latency = ((end_time - start_time) / args.iterations) * 1000
            fps = 1000 / avg_latency
            results.append({"Model": model_spec["name"], "Latency (ms)": round(avg_latency, 2), "FPS": round(fps, 1)})
            print(f"  Done: {avg_latency:.2f} ms | {fps:.1f} FPS")
        except Exception as exc:
            print(f"  Failed to test {model_spec['name']}: {exc}")

    if results:
        df = pd.DataFrame(results)
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        output_csv = os.path.join(log_dir, "speed_benchmark.csv")
        df.to_csv(output_csv, index=False)
        print(f"\nBenchmark results saved to {output_csv}")
        print(df.to_markdown(index=False))


if __name__ == "__main__":
    benchmark_speed(parse_args())
