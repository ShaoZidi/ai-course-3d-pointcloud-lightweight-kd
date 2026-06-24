import math
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class CoreBehaviorTests(unittest.TestCase):
    def test_synthetic_dataset_is_deterministic_and_has_expected_shapes(self):
        from src.data import make_dataset

        a = make_dataset(n_samples=12, seed=7, grid_shape=(8, 12, 12))
        b = make_dataset(n_samples=12, seed=7, grid_shape=(8, 12, 12))

        self.assertEqual(a["x"].shape, (12, 8 * 12 * 12))
        self.assertEqual(a["y_class"].shape, (12,))
        self.assertEqual(a["y_box"].shape, (12, 6))
        self.assertTrue((a["x"] == b["x"]).all())
        self.assertTrue((a["y_class"] == b["y_class"]).all())
        self.assertTrue((a["y_box"] == b["y_box"]).all())

    def test_model_statistics_show_lightweight_model_is_smaller(self):
        from src.models import BaselineMLP, LightweightMLP

        baseline = BaselineMLP(input_dim=8 * 12 * 12, hidden_dim=96, feature_dim=48, seed=1)
        lightweight = LightweightMLP(input_dim=8 * 12 * 12, bottleneck_dim=24, feature_dim=24, groups=4, seed=1)

        base_stats = baseline.count_parameters_and_ops()
        light_stats = lightweight.count_parameters_and_ops()

        self.assertGreater(base_stats["parameters"], light_stats["parameters"])
        self.assertGreater(base_stats["ops_per_sample"], light_stats["ops_per_sample"])

    def test_distillation_losses_are_finite_and_non_negative(self):
        import numpy as np
        from src.losses import distillation_loss_parts

        rng = np.random.default_rng(3)
        student_logits = rng.normal(size=(5, 3))
        teacher_logits = rng.normal(size=(5, 3))
        student_feat = rng.normal(size=(5, 6))
        teacher_feat = rng.normal(size=(5, 6))
        student_box = rng.normal(size=(5, 6))
        teacher_box = rng.normal(size=(5, 6))

        parts = distillation_loss_parts(
            student_logits,
            teacher_logits,
            student_feat,
            teacher_feat,
            student_box,
            teacher_box,
            temperature=2.0,
        )

        for value in parts.values():
            self.assertTrue(math.isfinite(float(value)))
            self.assertGreaterEqual(float(value), 0.0)

    def test_experiment_script_can_be_launched_directly(self):
        completed = subprocess.run(
            [sys.executable, str(ROOT / "src" / "run_experiment.py"), "--dry-run"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=20,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("dry run ok", completed.stdout.lower())


if __name__ == "__main__":
    unittest.main()
