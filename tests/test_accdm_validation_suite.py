from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from campaigns.accdm.validation_suite import SUITE_CAMPAIGNS, build_suite


class AccDMValidationSuiteTests(unittest.TestCase):
    def test_suite_builds_expected_exact_hierarchy_campaigns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runs_root = Path(directory) / "runs"
            run_dirs = build_suite(runs_root)

            self.assertEqual(len(run_dirs), 4)
            self.assertEqual(
                [item.name for item in SUITE_CAMPAIGNS],
                [path.name for path in run_dirs],
            )

            expected = {
                "accdm_validation_mass11_v1": (4, 1001),
                "accdm_qconvergence_f0p3_q1001_v1": (1, 1001),
                "accdm_qconvergence_joint_q2501_v1": (2, 2501),
                "accdm_qconvergence_joint_q5001_v1": (2, 5001),
            }
            for run_dir in run_dirs:
                config = json.loads((run_dir / "campaign.json").read_text())
                point_count, momentum_bins = expected[run_dir.name]
                self.assertEqual(config["point_count"], point_count)
                self.assertEqual(
                    config["model_settings"]["accdm_momentum_bins"], momentum_bins
                )
                self.assertEqual(
                    config["model_settings"]["ncdm_fluid_approximation"], 3
                )

    def test_suite_coordinates_and_resolution_are_recorded_in_models(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runs_root = Path(directory) / "runs"
            build_suite(runs_root)

            mass11 = runs_root / "accdm_validation_mass11_v1"
            with (mass11 / "manifest.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual({float(row["log10m_acc"]) for row in rows}, {11.0})
            self.assertEqual(
                {float(row["log10f_acc"]) for row in rows},
                {-4.0, -2.0, -1.0, 0.0},
            )

            q5001 = runs_root / "accdm_qconvergence_joint_q5001_v1"
            with (q5001 / "manifest.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            for row in rows:
                model = json.loads((q5001 / row["model_json"]).read_text())
                params = model["class_params"]
                self.assertEqual(params["ncdm_N_momentum_bins"], "15, 5001")
                self.assertEqual(params["ncdm_fluid_approximation"], 3)
                self.assertEqual(params["N_ur"], 2.0308)


if __name__ == "__main__":
    unittest.main()
