from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from campaigns.accdm.adaptive_scan import (
    SCAN_CAMPAIGNS,
    build_scan,
    selected_campaigns,
)


class AccDMAdaptiveScanTests(unittest.TestCase):
    def test_campaign_partition_and_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runs_root = Path(directory) / "runs"
            run_dirs = build_scan(runs_root)

            self.assertEqual(len(run_dirs), 3)
            expected = {
                "accdm_scan_lowf_q1001_v2": (850, 1001),
                "accdm_qconvergence_f0p3_q10001_v1": (1, 10001),
                "accdm_scan_highf_q5001_v1": (600, 5001),
            }
            for run_dir in run_dirs:
                config = json.loads((run_dir / "campaign.json").read_text())
                point_count, momentum_bins = expected[run_dir.name]
                self.assertEqual(config["point_count"], point_count)
                self.assertEqual(config["active_point_count"], point_count)
                self.assertEqual(
                    config["model_settings"]["accdm_momentum_bins"], momentum_bins
                )
                self.assertEqual(
                    config["model_settings"]["ncdm_fluid_approximation"], 3
                )

    def test_grid_is_dense_above_fraction_point_one_without_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runs_root = Path(directory) / "runs"
            build_scan(runs_root)

            rows = {}
            for name in ("accdm_scan_lowf_q1001_v2", "accdm_scan_highf_q5001_v1"):
                with (runs_root / name / "manifest.csv").open(
                    newline="", encoding="utf-8"
                ) as handle:
                    rows[name] = list(csv.DictReader(handle))

            low = rows["accdm_scan_lowf_q1001_v2"]
            high = rows["accdm_scan_highf_q5001_v1"]
            low_coordinates = {
                (float(row["log10m_acc"]), float(row["log10f_acc"])) for row in low
            }
            high_coordinates = {
                (float(row["log10m_acc"]), float(row["log10f_acc"])) for row in high
            }
            self.assertTrue(low_coordinates.isdisjoint(high_coordinates))
            masses = sorted({float(row["log10m_acc"]) for row in low})
            self.assertEqual(len(masses), 50)
            self.assertAlmostEqual(masses[0], 11.0)
            self.assertAlmostEqual(masses[-1], 18.0)
            for left, right in zip(masses[:-1], masses[1:]):
                self.assertAlmostEqual(right - left, 1.0 / 7.0, places=12)
            low_fractions = sorted({float(row["log10f_acc"]) for row in low})
            high_fractions = sorted({float(row["log10f_acc"]) for row in high})
            self.assertEqual(len(low_fractions), 17)
            self.assertEqual(len(high_fractions), 12)
            self.assertEqual(max(float(row["log10f_acc"]) for row in low), -1.0)
            self.assertEqual(min(float(row["log10f_acc"]) for row in high), -0.925)
            self.assertEqual(max(float(row["log10f_acc"]) for row in high), 0.0)
            dense = [value for value in low_fractions + high_fractions if -1.6 <= value <= -0.4]
            dense.sort()
            self.assertEqual(len(dense), 17)
            for left, right in zip(dense[:-1], dense[1:]):
                self.assertAlmostEqual(right - left, 0.075, places=12)
            self.assertEqual(len(low_coordinates | high_coordinates), 1450)

    def test_initial_group_cannot_submit_the_high_fraction_grid(self) -> None:
        self.assertEqual(
            [item.group for item in selected_campaigns("initial")],
            ["low", "pilot"],
        )
        self.assertEqual(
            [item.group for item in selected_campaigns("high")],
            ["high"],
        )
        self.assertEqual(len(selected_campaigns("all")), len(SCAN_CAMPAIGNS))


if __name__ == "__main__":
    unittest.main()
