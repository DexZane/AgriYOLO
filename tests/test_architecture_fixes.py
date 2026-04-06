import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_sota_comparison import generate_gt_json
from ultralytics.nn.modules.head import Detect
from ultralytics.nn.modules.simam import SimAM
from ultralytics.nn.tasks import YOLOv10DetectionModel


class ArchitectureFixTests(unittest.TestCase):
    def test_ablation_variants_are_structurally_distinct(self):
        model_specs = {
            "stage2": (ROOT / "ultralytics/cfg/models/v10/yolov10s_P2_BiFPN.yaml", (False, False, False)),
            "stage3": (ROOT / "ultralytics/cfg/models/v10/yolov10s_P2_ADSA.yaml", (False, True, False)),
            "stage4": (ROOT / "ultralytics/cfg/models/v10/yolov10s_P2_CADFM.yaml", (True, True, False)),
            "stage5": (ROOT / "ultralytics/cfg/models/v10/yolov10s_TAL_FFN.yaml", (True, True, True)),
        }

        param_counts = []
        for path, expected_flags in model_specs.values():
            model = YOLOv10DetectionModel(str(path), nc=27, verbose=False)
            neck = model.model[20]
            self.assertEqual((neck.use_cawn, neck.use_adsa, neck.use_dsconv), expected_flags)
            param_counts.append(sum(parameter.numel() for parameter in model.parameters()))

        self.assertEqual(len(set(param_counts)), len(param_counts))

    def test_simam_is_not_identity_on_random_input(self):
        module = SimAM()
        x = torch.randn(2, 8, 16, 16)
        y = module(x)
        self.assertGreater((y - x).abs().max().item(), 1e-4)

    def test_v10_eval_does_not_call_one2many_branch(self):
        model = YOLOv10DetectionModel(str(ROOT / "ultralytics/cfg/models/v10/yolov10s_TAL_FFN.yaml"), nc=27, verbose=False)
        model.eval()
        dummy_input = torch.randn(1, 3, 640, 640)

        with mock.patch.object(Detect, "forward", side_effect=AssertionError("one2many branch should not run in eval")):
            with torch.inference_mode():
                output = model(dummy_input)

        self.assertIsInstance(output, tuple)
        self.assertFalse(isinstance(output, dict))
        self.assertEqual(output[0].shape[1], 4 + 27)

    def test_generate_gt_json_uses_full_test_split_and_real_image_sizes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            dataset_root = tmp_path / "dataset"
            image_dir = dataset_root / "test" / "images"
            label_dir = dataset_root / "test" / "labels"
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)

            Image.new("RGB", (32, 24), color="red").save(image_dir / "1.jpg")
            Image.new("RGB", (48, 40), color="blue").save(image_dir / "2.jpg")
            (label_dir / "1.txt").write_text("0 0.5 0.5 0.25 0.5\n", encoding="utf-8")

            data_yaml = tmp_path / "data.yaml"
            data_yaml.write_text(
                "path: dataset\n"
                "test: test/images\n"
                "names:\n"
                "  0: lesion\n",
                encoding="utf-8",
            )

            output_json = tmp_path / "gt.json"
            generate_gt_json(str(data_yaml), split="test", output_json=str(output_json))

            with open(output_json, "r", encoding="utf-8") as f:
                coco_data = json.load(f)

            image_sizes = {item["id"]: (item["width"], item["height"]) for item in coco_data["images"]}
            self.assertEqual(len(coco_data["images"]), 2)
            self.assertEqual(image_sizes[1], (32, 24))
            self.assertEqual(image_sizes[2], (48, 40))
            self.assertEqual(len(coco_data["annotations"]), 1)


if __name__ == "__main__":
    unittest.main()
