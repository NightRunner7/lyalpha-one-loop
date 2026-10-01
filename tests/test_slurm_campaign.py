"""Scheduling regressions: avoid duplicate work, wrong resources and lost failures."""
from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from campaigns.accdm.build_grid import build_campaign
from cluster.campaign_manager import (
    Campaign, QueueSnapshot, _sbatch_command, _submit_one, point_state, query_queue,
    submit_stage, build_parser,
)
from cluster.prepare_eagle import prepare
from lyalpha_pt.class_runtime import class_runtime

ROOT = Path(__file__).resolve().parents[1]


class SlurmCampaignTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="lya slurm,")
        self.root = Path(self.tmp.name)
        self.run = self.root / "run"
        spec = {
            "campaign_id": "eagle_test",
            "points": [{"log10m_acc": 15, "log10f_acc": -4}],
            "cluster": {
                "scheduler": "slurm", "account": "pl0503-01", "partition": "fast",
                "theory_ncpus": 2, "theory_threads": 2, "theory_mem": "8gb",
                "theory_walltime": "01:00:00", "theory_python": "/a path/env/bin/python",
                "fit_ncpus": 1, "fit_mem": "4gb", "fit_walltime": "00:30:00",
                "theory_max_active": 4, "max_user_active": 10, "submit_delay_seconds": 0,
            },
        }
        path = self.root / "spec.json"
        path.write_text(json.dumps(spec))
        build_campaign(ROOT / "campaigns/accdm/base_models/base_model_refactor_birth.json", path, self.run)
        self.campaign = Campaign.load(self.run, ROOT)
        self.point = self.campaign.points[0]

    def tearDown(self):
        self.tmp.cleanup()

    def test_builder_retains_slurm_settings_and_resource_units(self):
        cmd = _sbatch_command(self.campaign, self.point, "theory")
        for flag in ("--parsable", "--cpus-per-task=2", "--ntasks=1", "--mem=8G",
                     "--account=pl0503-01", "--partition=fast", "--time=01:00:00"):
            self.assertIn(flag, cmd)
        self.assertTrue(cmd[-1].endswith("theory_point.slurm"))
        self.assertIn(".%j.out", next(arg for arg in cmd if arg.startswith("--output=")))

    @patch("cluster.campaign_manager.subprocess.run")
    def test_held_and_array_jobs_count_but_finished_jobs_do_not(self, run):
        run.return_value = subprocess.CompletedProcess([], 0, "10|RUNNING\n11|PENDING\n12|SUSPENDED\n13|COMPLETING\n14|COMPLETED\n15_0|PENDING\n15_1|RUNNING\n16|REQUEUE_HOLD\n", "")
        snapshot = query_queue(self.campaign)
        self.assertEqual(snapshot.total_active, 7)
        self.assertNotIn("14", snapshot.active_job_ids)
        self.assertIn("--array", run.call_args.args[0])

    @patch("cluster.campaign_manager.subprocess.run")
    def test_squeue_failure_never_becomes_an_empty_queue(self, run):
        run.return_value = subprocess.CompletedProcess([], 1, "", "controller unavailable")
        with self.assertRaisesRegex(RuntimeError, "squeue failed"):
            submit_stage(self.campaign, "theory")
        self.assertEqual(run.call_count, 1)
        self.assertFalse(self.campaign.job_record_path(self.point, "theory").exists())

    @patch("cluster.campaign_manager.subprocess.run")
    def test_unparseable_queue_fails_closed(self, run):
        run.return_value = subprocess.CompletedProcess([], 0, "unexpected banner\n", "")
        with self.assertRaisesRegex(RuntimeError, "Cannot parse"):
            query_queue(self.campaign)

    @patch("cluster.campaign_manager.subprocess.run")
    def test_submission_env_preserves_paths_and_records_numeric_id(self, run):
        run.return_value = subprocess.CompletedProcess([], 0, "12345;eagle\n", "")
        self.assertEqual(_submit_one(self.campaign, self.point, "theory", False), "12345")
        env = run.call_args.kwargs["env"]
        self.assertEqual(env["CAMPAIGN_DIR"], str(self.run))
        self.assertEqual(env["PYTHON_BIN"], "/a path/env/bin/python")
        record = json.loads(self.campaign.job_record_path(self.point, "theory").read_text())
        self.assertEqual(record["job_id"], "12345")
        self.assertEqual(record["scheduler"], "slurm")

    @patch("cluster.campaign_manager.subprocess.run")
    def test_active_job_is_not_submitted_twice(self, run):
        self.campaign.job_record_path(self.point, "theory").write_text(json.dumps({"job_id":"12345", "attempt":1}))
        run.return_value = subprocess.CompletedProcess([], 0, "12345|PENDING\n", "")
        self.assertEqual(submit_stage(self.campaign, "theory"), 0)
        self.assertEqual(run.call_count, 1)

    @patch("cluster.campaign_manager.subprocess.run")
    def test_failed_submit_writes_no_job_record(self, run):
        run.return_value = subprocess.CompletedProcess([], 1, "", "Invalid account")
        with self.assertRaisesRegex(RuntimeError, "Invalid account"):
            _submit_one(self.campaign, self.point, "theory", False)
        self.assertFalse(self.campaign.job_record_path(self.point, "theory").exists())

    @patch("cluster.campaign_manager.subprocess.run")
    def test_dry_run_never_calls_scheduler(self, run):
        self.assertEqual(submit_stage(self.campaign, "theory", offline=True, dry_run=True), 1)
        run.assert_not_called()
        self.assertFalse(self.campaign.job_record_path(self.point, "theory").exists())

    @patch("cluster.campaign_manager.subprocess.run")
    def test_disappeared_job_can_resume_and_fit_waits_for_theory(self, run):
        self.campaign.job_record_path(self.point, "theory").write_text(json.dumps({"job_id":"12345", "attempt":1}))
        run.return_value = subprocess.CompletedProcess([], 0, "", "")
        snapshot = query_queue(self.campaign)
        self.assertEqual(point_state(self.campaign, self.point, "theory", snapshot), "pending")
        self.assertEqual(point_state(self.campaign, self.point, "fit", snapshot), "waiting_theory")

    @patch("cluster.campaign_manager.query_queue", return_value=QueueSnapshot(frozenset(), 0))
    def test_failed_smoke_watch_exits_instead_of_waiting_forever(self, queue):
        self.campaign.status_path(self.point, "theory", "failed").write_text("{}")
        args = build_parser().parse_args(["watch", "--campaign", str(self.run), "--exit-when-complete"])
        self.assertEqual(args.handler(args), 2)

    @patch("cluster.prepare_eagle.class_runtime")
    def test_prepare_keeps_class_precision_in_smoke_and_refuses_quality_reuse(self, runtime):
        runtime.return_value = {"classy_sha256":"abc", "declared_source":{"directory":"/class", "commit":"123"}}
        args = SimpleNamespace(
            run_dir=self.root / "prepared", grid_spec=None, base_model=None, quality=None,
            partition=None, walltime=None, cpus=1, max_active=4, max_user_active=100,
            class_source=Path("/class"), account="pl0503-01", memory="8gb",
        )
        config = prepare(args)
        self.assertIsNone(config["model_settings"]["accdm_momentum_bins"])
        self.assertEqual(config["model_settings"]["accdm_quadrature_strategy"], 5)
        self.assertEqual(config["model_settings"]["class_precision_inputs"]["background_Nloga"], 40000)
        self.assertEqual(config["model_settings"]["ncdm_fluid_approximation"], 3)
        self.assertEqual(config["cluster"]["scheduler"], "slurm")
        args.quality = "production"
        with self.assertRaisesRegex(ValueError, "quality changed"):
            prepare(args)
        args.run_dir = self.root / "production"
        production = prepare(args)
        self.assertEqual(config["base_model_hash"], production["base_model_hash"])
        self.assertEqual(config["model_settings"], production["model_settings"])

    @patch("lyalpha_pt.class_runtime.importlib.util.find_spec")
    def test_changed_wrapper_is_rejected_before_computation(self, find):
        wrapper = self.root / "classy.so"
        wrapper.write_bytes(b"different compiled binary")
        find.return_value = SimpleNamespace(origin=str(wrapper))
        with self.assertRaisesRegex(RuntimeError, "Installed classy changed"):
            class_runtime(expected_wrapper="wrong-sha256")


if __name__ == "__main__":
    unittest.main()
