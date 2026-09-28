from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from campaigns.dcdm.build_grid import build_campaign
from cluster.campaign_manager import (
    Campaign,
    Point,
    _qsub_command,
    _qsub_variables,
    point_state,
    prepare_neighbor_refit,
)


ROOT = Path(__file__).resolve().parents[1]
DCDM = ROOT / "campaigns" / "dcdm"


class CampaignManagerStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.temporary.name) / "run"
        build_campaign(
            DCDM / "base_model.json",
            DCDM / "grid_specs" / "validation_grid.json",
            self.run_dir,
        )
        self.campaign = Campaign.load(self.run_dir, ROOT)
        self.point: Point = self.campaign.points[0]

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_fit_waits_for_theory(self) -> None:
        self.assertEqual(point_state(self.campaign, self.point, "theory", None), "pending")
        self.assertEqual(
            point_state(self.campaign, self.point, "fit", None), "waiting_theory"
        )

    def test_completed_theory_releases_fit(self) -> None:
        output = self.campaign.output_path(self.point, "theory")
        done = self.campaign.status_path(self.point, "theory", "done")
        output.write_bytes(b"test bundle placeholder")
        done.write_text("{}\n", encoding="utf-8")
        self.assertEqual(
            point_state(self.campaign, self.point, "theory", None), "complete"
        )
        self.assertEqual(point_state(self.campaign, self.point, "fit", None), "pending")

    def test_offline_submission_record_is_not_assumed_finished(self) -> None:
        record = self.campaign.job_record_path(self.point, "theory")
        record.write_text(
            json.dumps({"job_id": "123.server", "attempt": 1}) + "\n",
            encoding="utf-8",
        )
        self.assertEqual(
            point_state(self.campaign, self.point, "theory", None),
            "submitted_unknown",
        )

    def test_neighbour_refit_sources_are_passed_to_pbs(self) -> None:
        config_path = self.run_dir / "campaign.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["fit"]["strategy"] = "neighbor_refit"
        config["fit"]["source_fit_dirs"] = ["fits_direct", "fits_continuation"]
        config_path.write_text(json.dumps(config), encoding="utf-8")
        campaign = Campaign.load(self.run_dir, ROOT)

        variables = _qsub_variables(campaign, campaign.points[0], "fit")

        self.assertEqual(variables["FIT_STRATEGY"], "neighbor_refit")
        expected = ":".join(
            str((self.run_dir / name).resolve())
            for name in ("fits_direct", "fits_continuation")
        )
        self.assertEqual(variables["FIT_SOURCE_DIRS"], expected)

    def test_theory_resources_and_threads_are_campaign_specific(self) -> None:
        config_path = self.run_dir / "campaign.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["cluster"].update(
            {
                "theory_ncpus": 2,
                "theory_threads": 2,
                "theory_mem": "4gb",
                "theory_walltime": "48:00:00",
            }
        )
        config_path.write_text(json.dumps(config), encoding="utf-8")
        campaign = Campaign.load(self.run_dir, ROOT)

        variables = _qsub_variables(campaign, campaign.points[0], "theory")
        command = _qsub_command(campaign, campaign.points[0], "theory")

        self.assertEqual(variables["THEORY_THREADS"], "2")
        resource_index = command.index("-l") + 1
        self.assertEqual(command[resource_index], "select=1:ncpus=2:mem=4gb")
        self.assertIn("walltime=48:00:00", command)

    def test_invalid_campaign_walltime_is_rejected(self) -> None:
        config_path = self.run_dir / "campaign.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["cluster"]["theory_walltime"] = "48 hours"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        campaign = Campaign.load(self.run_dir, ROOT)

        with self.assertRaisesRegex(ValueError, "theory_walltime"):
            _qsub_command(campaign, campaign.points[0], "theory")

    def test_prepare_neighbor_refit_archives_completed_pass(self) -> None:
        for point in self.campaign.points:
            self.campaign.output_path(point, "fit").write_text("{}\n", encoding="utf-8")
            self.campaign.status_path(point, "fit", "done").write_text(
                "{}\n", encoding="utf-8"
            )
        extra = self.run_dir / "fits_continuation"
        extra.mkdir()

        sources = prepare_neighbor_refit(
            self.campaign,
            archive_label="direct_k20",
            additional_source_fit_dirs=("fits_continuation",),
        )

        self.assertEqual(sources[0], (self.run_dir / "fits_direct_k20").resolve())
        self.assertTrue((self.run_dir / "fits_direct_k20").is_dir())
        self.assertTrue((self.run_dir / "fits").is_dir())
        self.assertFalse(any((self.run_dir / "fits").iterdir()))
        config = json.loads((self.run_dir / "campaign.json").read_text(encoding="utf-8"))
        self.assertEqual(config["fit"]["strategy"], "neighbor_refit")
        self.assertEqual(
            config["fit"]["source_fit_dirs"],
            ["fits_direct_k20", "fits_continuation"],
        )


if __name__ == "__main__":
    unittest.main()
