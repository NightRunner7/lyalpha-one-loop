from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

from campaigns.dcdm.build_grid import (
    GAMMA_TAU_CONVERSION,
    build_campaign,
    gamma_from_tau_gyr,
)
from lyalpha_pt.models import ModelSpec


ROOT = Path(__file__).resolve().parents[1]
DCDM = ROOT / "campaigns" / "dcdm"


class DCDMCampaignGridTests(unittest.TestCase):
    def test_gamma_lifetime_conversion(self) -> None:
        self.assertTrue(
            math.isclose(GAMMA_TAU_CONVERSION, 977.7922216807891, rel_tol=2e-15)
        )
        self.assertTrue(
            math.isclose(gamma_from_tau_gyr(20.0), 48.88961108403946, rel_tol=2e-15)
        )

    def test_validation_campaign_is_self_contained_and_stable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            config = build_campaign(
                DCDM / "base_model.json",
                DCDM / "grid_specs" / "validation_grid.json",
                run_dir,
            )
            self.assertEqual(config["point_count"], 10)
            self.assertEqual(config["theory"]["quality"], "production")
            self.assertEqual(config["grid_axes"], ["tau_gyr", "epsilon_dcdm"])

            with (run_dir / "manifest.csv").open(newline="", encoding="utf-8") as handle:
                rows_before = list(csv.DictReader(handle))
            self.assertEqual([int(row["index"]) for row in rows_before], list(range(10)))
            best_fit = next(
                row for row in rows_before if row["point_id"] == "dcdm_tau40_eps0p006"
            )
            model_payload = json.loads(
                (run_dir / best_fit["model_json"]).read_text(encoding="utf-8")
            )
            model = ModelSpec.from_dict(model_payload)
            self.assertEqual(model.loop_source, "total")
            self.assertEqual(model.loop_weight, 1.0)
            self.assertEqual(model.class_params["omega_cdm"], 1.0e-5)
            self.assertEqual(model.class_params["omega_ini_dcdm2"], 0.1200)
            self.assertEqual(model.class_params["N_ncdm"], 2)
            self.assertTrue(
                math.isclose(
                    model.class_params["Gamma_dcdm"],
                    gamma_from_tau_gyr(40.0),
                    rel_tol=1e-15,
                )
            )
            self.assertEqual(model.class_params["epsilon_dcdm"], 0.006)
            self.assertNotIn("P_k_max_h/Mpc", model.class_params)

            build_campaign(
                DCDM / "base_model.json",
                DCDM / "grid_specs" / "validation_grid.json",
                run_dir,
            )
            with (run_dir / "manifest.csv").open(newline="", encoding="utf-8") as handle:
                rows_after = list(csv.DictReader(handle))
            self.assertEqual(rows_before, rows_after)

    def test_paper_grid_has_published_shape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = build_campaign(
                DCDM / "base_model.json",
                DCDM / "grid_specs" / "paper_grid.json",
                Path(directory) / "run",
            )
            self.assertEqual(config["point_count"], 23 * 28)
            self.assertEqual(config["fit"]["mode"], "one_loop")

    def test_existing_campaign_identity_cannot_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            build_campaign(
                DCDM / "base_model.json",
                DCDM / "grid_specs" / "validation_grid.json",
                run_dir,
            )
            with self.assertRaisesRegex(ValueError, "campaign_id"):
                build_campaign(
                    DCDM / "base_model.json",
                    DCDM / "grid_specs" / "precision_anchors.json",
                    run_dir,
                )


if __name__ == "__main__":
    unittest.main()
