from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

from campaigns.accdm.build_grid import (
    build_campaign,
    fixed_early_omega_cdm,
    point_id,
)
from campaigns.accdm.prepare_highf_recovery import prepare_recovery_campaigns
from lyalpha_pt.models import ModelSpec


ROOT = Path(__file__).resolve().parents[1]
ACCDM = ROOT / "campaigns" / "accdm"


class AccDMCampaignGridTests(unittest.TestCase):
    def test_parameter_mapping_and_fixed_early_density(self) -> None:
        omega = fixed_early_omega_cdm(
            0.1201,
            0.01,
            kappa_acc=12.1,
            a_t_acc=0.133,
            a_recombination=1.0 / 1091.0,
        )
        self.assertTrue(math.isclose(omega, 0.1201 / 1.01, rel_tol=2e-15))
        self.assertEqual(point_id(12.0, -2.0), "accdm_logm12_logfm2")

    def test_validation_campaign_is_exact_and_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            config = build_campaign(
                ACCDM / "base_model.json",
                ACCDM / "grid_specs" / "validation_grid.json",
                run_dir,
            )
            self.assertEqual(config["point_count"], 16)
            self.assertEqual(config["grid_axes"], ["log10m_acc", "log10f_acc"])
            self.assertEqual(config["model_settings"]["ncdm_fluid_approximation"], 3)
            self.assertEqual(
                config["model_settings"]["neutrino_convention"],
                "two_massless_plus_one_massive_0p06eV",
            )

            with (run_dir / "manifest.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                rows_before = list(csv.DictReader(handle))
            self.assertEqual([int(row["index"]) for row in rows_before], list(range(16)))

            selected = next(
                row
                for row in rows_before
                if row["point_id"] == "accdm_logm12_logfm2"
            )
            payload = json.loads(
                (run_dir / selected["model_json"]).read_text(encoding="utf-8")
            )
            model = ModelSpec.from_dict(payload)
            params = model.class_params
            self.assertEqual(model.loop_source, "total")
            self.assertEqual(model.loop_weight, 1.0)
            self.assertEqual(params["N_ur"], 2.0308)
            self.assertEqual(params["N_ncdm"], 2)
            self.assertEqual(params["m_ncdm"], "0.06, 0.1")
            self.assertEqual(params["deg_ncdm"], "1, 1")
            self.assertEqual(params["ncdm_fluid_approximation"], 3)
            self.assertEqual(params["m_acc_in_GeV"], 1.0e12)
            self.assertEqual(params["m_cdm_in_GeV"], 1.0e12)
            self.assertTrue(math.isclose(params["eta_acc"], 0.1, rel_tol=1e-15))
            self.assertTrue(math.isclose(params["f_acc"], 0.01, rel_tol=1e-15))
            self.assertTrue(
                math.isclose(params["omega_cdm"], 0.1201 / 1.01, rel_tol=2e-15)
            )
            self.assertNotIn("P_k_max_h/Mpc", params)
            self.assertNotIn("output", params)

            build_campaign(
                ACCDM / "base_model.json",
                ACCDM / "grid_specs" / "validation_grid.json",
                run_dir,
            )
            with (run_dir / "manifest.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                rows_after = list(csv.DictReader(handle))
            self.assertEqual(rows_before, rows_after)

    def test_base_model_cannot_enable_the_fluid_approximation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = json.loads((ACCDM / "base_model.json").read_text(encoding="utf-8"))
            base["class_params"]["ncdm_fluid_approximation"] = 2
            bad_base = root / "base.json"
            bad_base.write_text(json.dumps(base), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exact hierarchy"):
                build_campaign(
                    bad_base,
                    ACCDM / "grid_specs" / "validation_grid.json",
                    root / "run",
                )

    def test_high_fraction_q5001_campaign_resources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = build_campaign(
                ACCDM / "base_models" / "base_model_q5001.json",
                ACCDM / "grid_specs" / "scan_highf_boundary_q5001.json",
                Path(directory) / "run",
            )

        self.assertEqual(config["campaign_id"], "accdm_scan_highf_q5001_v1")
        self.assertEqual(config["point_count"], 600)
        self.assertEqual(config["model_settings"]["accdm_momentum_bins"], 5001)
        self.assertEqual(config["cluster"]["theory_ncpus"], 2)
        self.assertEqual(config["cluster"]["theory_threads"], 2)
        self.assertEqual(config["cluster"]["theory_mem"], "4gb")
        self.assertEqual(config["cluster"]["theory_max_active"], 600)
        self.assertEqual(config["cluster"]["max_user_active"], 650)

    def test_high_fraction_q1001_scout_matches_production_coordinates(self) -> None:
        scout_spec = json.loads(
            (
                ACCDM / "grid_specs" / "scan_highf_scout_q1001.json"
            ).read_text(encoding="utf-8")
        )
        production_spec = json.loads(
            (
                ACCDM / "grid_specs" / "scan_highf_boundary_q5001.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(scout_spec["axes"], production_spec["axes"])

        with tempfile.TemporaryDirectory() as directory:
            config = build_campaign(
                ACCDM / "base_model.json",
                ACCDM / "grid_specs" / "scan_highf_scout_q1001.json",
                Path(directory) / "run",
            )

        self.assertEqual(config["campaign_id"], "accdm_scan_highf_q1001_scout_v1")
        self.assertEqual(config["point_count"], 600)
        self.assertEqual(config["model_settings"]["accdm_momentum_bins"], 1001)
        self.assertEqual(config["theory"]["max_attempts"], 1)
        self.assertEqual(config["cluster"]["theory_ncpus"], 10)
        self.assertEqual(config["cluster"]["theory_threads"], 10)
        self.assertEqual(config["cluster"]["theory_mem"], "4gb")
        self.assertEqual(config["cluster"]["theory_max_active"], 120)
        self.assertEqual(config["cluster"]["fit_max_active"], 500)

    def test_q10001_minimum_validation_has_three_explicit_anchors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            config = build_campaign(
                ACCDM / "base_models" / "base_model_q10001.json",
                ACCDM / "grid_specs" / "validate_minima_q10001.json",
                run_dir,
            )

            with (run_dir / "manifest.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(config["campaign_id"], "accdm_validate_minima_q10001_v1")
        self.assertEqual(config["point_count"], 3)
        self.assertEqual(config["model_settings"]["accdm_momentum_bins"], 10001)
        self.assertEqual(config["cluster"]["theory_ncpus"], 20)
        self.assertEqual(config["cluster"]["theory_threads"], 20)
        self.assertEqual(config["cluster"]["theory_mem"], "16gb")
        self.assertEqual(
            {(float(row["log10m_acc"]), float(row["log10f_acc"])) for row in rows},
            {
                (11.4285714285714, -0.55),
                (11.4285714285714, -0.775),
                (15.7142857142857, -1.3),
            },
        )

    def test_highf_recovery_freezes_missing_and_stiff_points(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_spec = {
                "campaign_id": "test_source_q5001",
                "points": [
                    {"log10m_acc": 11, "log10f_acc": value}
                    for value in (-0.925, -0.85, -0.775, -0.7)
                ],
                "theory": {"quality": "production", "max_attempts": 1},
                "fit": {"mode": "one_loop", "max_attempts": 1},
                "cluster": {
                    "theory_ncpus": 1,
                    "theory_threads": 1,
                    "theory_max_active": 4,
                    "fit_max_active": 4,
                },
            }
            spec_path = root / "source_spec.json"
            spec_path.write_text(json.dumps(source_spec), encoding="utf-8")
            source = root / "source"
            build_campaign(
                ACCDM / "base_models" / "base_model_q5001.json",
                spec_path,
                source,
            )

            complete_id = point_id(11, -0.925)
            (source / "theory" / f"{complete_id}.npz").write_bytes(b"complete")
            (source / "status" / "theory" / f"{complete_id}.done.json").write_text(
                "{}\n", encoding="utf-8"
            )
            for value in (-0.775, -0.7):
                identifier = point_id(11, value)
                (source / "logs" / "theory" / f"{identifier}.err").write_text(
                    "generic_integrator: Step size too small", encoding="utf-8"
                )

            outputs = prepare_recovery_campaigns(source, root / "recoveries")
            main = _campaign_payload(outputs["main"])
            pilot = _campaign_payload(outputs["step-pilot"])
            stiff = _campaign_payload(outputs["step-recovery"])
            pilot_model = json.loads(
                (outputs["step-pilot"] / pilot["rows"][0]["model_json"]).read_text(
                    encoding="utf-8"
                )
            )
            ordinary_id = point_id(11, -0.85)
            (source / "theory" / f"{ordinary_id}.npz").write_bytes(b"late")
            (source / "status" / "theory" / f"{ordinary_id}.done.json").write_text(
                "{}\n", encoding="utf-8"
            )
            repeated = prepare_recovery_campaigns(source, root / "recoveries")
            repeated_main = _campaign_payload(repeated["main"])

        self.assertEqual(main["config"]["point_count"], 1)
        self.assertEqual(repeated_main["config"]["point_count"], 1)
        self.assertEqual(main["rows"][0]["point_id"], point_id(11, -0.85))
        self.assertEqual(main["config"]["cluster"]["theory_ncpus"], 10)
        self.assertEqual(main["config"]["cluster"]["theory_threads"], 10)
        self.assertEqual(main["config"]["cluster"]["theory_mem"], "8gb")
        self.assertEqual(main["config"]["cluster"]["theory_walltime"], "48:00:00")
        self.assertEqual(pilot["config"]["point_count"], 1)
        self.assertEqual(stiff["config"]["point_count"], 1)
        self.assertNotIn("smallest_allowed_variation", pilot_model["class_params"])
        self.assertEqual(
            pilot_model["class_params"]["ncdm_N_momentum_bins"], "15, 5001"
        )
        self.assertEqual(pilot["config"]["cluster"]["theory_ncpus"], 20)
        self.assertEqual(pilot["config"]["cluster"]["theory_threads"], 20)


def _campaign_payload(run_dir: Path) -> dict[str, object]:
    config = json.loads((run_dir / "campaign.json").read_text(encoding="utf-8"))
    with (run_dir / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {"config": config, "rows": rows}


if __name__ == "__main__":
    unittest.main()
