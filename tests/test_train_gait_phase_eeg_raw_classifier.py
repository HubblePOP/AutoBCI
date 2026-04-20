from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.train_gait_phase_eeg_raw_classifier as raw_cls


class TrainGaitPhaseEEGRawClassifierTests(unittest.TestCase):
    def test_parse_args_accepts_deepconvnet(self) -> None:
        args = raw_cls.parse_args(
            [
                "--package-dir",
                "/tmp/pkg",
                "--algorithm-family",
                "deepconvnet",
                "--output-json",
                "/tmp/out.json",
            ]
        )
        self.assertEqual(args.algorithm_family, "deepconvnet")
        self.assertEqual(Path(args.package_dir), Path("/tmp/pkg"))

    def test_build_models_forward_shape(self) -> None:
        x = torch.zeros((2, 32, 250), dtype=torch.float32)
        deep = raw_cls.build_model("deepconvnet", in_channels=32, n_times=250, n_classes=2, hidden_size=64, dropout=0.1)
        tmsa = raw_cls.build_model("tmsanet", in_channels=32, n_times=250, n_classes=2, hidden_size=64, dropout=0.1)
        self.assertEqual(tuple(deep(x).shape), (2, 2))
        self.assertEqual(tuple(tmsa(x).shape), (2, 2))

    def test_main_writes_metrics_json(self) -> None:
        rng = np.random.default_rng(7)
        with tempfile.TemporaryDirectory() as tmpdir:
            package_dir = Path(tmpdir) / "pkg"
            package_dir.mkdir(parents=True, exist_ok=True)

            def _make_split(n: int) -> tuple[np.ndarray, np.ndarray]:
                x = rng.normal(size=(n, 32, 250)).astype(np.float32)
                y = np.zeros((n,), dtype=np.int64)
                y[n // 2 :] = 1
                x[y == 1, 0, :25] += 1.5
                return x, y

            for split_name, n in (("train", 32), ("val", 12), ("test", 12)):
                x, y = _make_split(n)
                np.save(package_dir / f"X_{split_name}.npy", x)
                np.save(package_dir / f"y_{split_name}.npy", y)

            metadata = {
                "package_mode": "historical_073_raw32_500hz",
                "window_seconds": 0.5,
                "export_fs_hz": 500.0,
                "class_definition": {"0": "support", "1": "swing"},
            }
            (package_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

            output_json = Path(tmpdir) / "result.json"
            raw_cls.main(
                [
                    "--package-dir",
                    str(package_dir),
                    "--algorithm-family",
                    "deepconvnet",
                    "--output-json",
                    str(output_json),
                    "--epochs",
                    "1",
                    "--batch-size",
                    "8",
                    "--hidden-size",
                    "32",
                    "--seed",
                    "7",
                    "--device",
                    "cpu",
                ]
            )

            saved = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(saved["algorithm_family"], "deepconvnet")
            self.assertEqual(saved["input_mode"], "raw_ecog")
            self.assertEqual(saved["train_shape"], [32, 32, 250])
            self.assertEqual(saved["primary_metric_name"], "balanced_accuracy")
            self.assertIn("val_primary_metric", saved)
            self.assertIn("test_primary_metric", saved)
            self.assertIn("balanced_accuracy", saved["test_metrics"])
            self.assertIn("macro_f1", saved["test_metrics"])
            self.assertIn("support", saved["test_metrics"]["per_class_recall"])
            self.assertIn("swing", saved["test_metrics"]["per_class_recall"])


if __name__ == "__main__":
    unittest.main()
