"""Regressions for new CLASS inputs, automatic q sampling and legacy campaigns."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from campaigns.accdm.build_grid import build_campaign, _validate_base_model
from campaigns.accdm.prepare_highf_recovery import prepare_recovery_campaigns
from cluster.campaign_manager import Campaign, _scheduler
from lyalpha_pt.models import ModelSpec
from lyalpha_pt.theory import _sum_massive_neutrino_mass

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "campaigns/accdm/base_models/base_model_refactor_birth.json"
LEGACY = ROOT / "campaigns/accdm/base_model.json"


class AccDMRefactorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.spec = self.root / "grid.json"
        self.spec.write_text(json.dumps({
            "campaign_id": "refactor", "points": [{"log10m_acc": 11, "log10f_acc": -1}],
        }))
        self.base = json.loads(BASE.read_text())

    def tearDown(self):
        self.tmp.cleanup()

    def build(self, base, name="run"):
        return build_campaign(base, self.spec, self.root / name)

    def test_birth_grid_reaches_class_without_a_hidden_fixed_bin_count(self):
        config = self.build(BASE)
        settings = config["model_settings"]
        self.assertIsNone(settings["accdm_momentum_bins"])
        self.assertEqual(settings["accdm_momentum_sampling"], "birth_grid_density")
        payload = json.loads(next((self.root / "run/models").glob("*.json")).read_text())
        params = ModelSpec.from_dict(payload).class_input(z_max_pk=4.4, pk_max_hmpc=42.0)
        for key in ("ncdm_N_momentum_bins", "Number of momentum bins", "m_ncdm"):
            self.assertNotIn(key, params)
        self.assertEqual(params["ncdm_quadrature_strategy"], "0, 5")
        self.assertEqual(params["m_nu"], 0.06)
        self.assertEqual(params["eta_acc"], 1.0)
        self.assertEqual(params["m_acc_in_GeV"], params["m_cdm_in_GeV"])
        for key, expected in {
            "background_Nloga": 40000, "accdm_q_number_tol": 1e-6,
            "accdm_q_bins_per_decade": 50.0, "accdm_smooth_births": 0,
            "acc_de_sink": "no", "ncdm_fluid_approximation": 3,
        }.items():
            self.assertEqual(params[key], expected)
        self.assertEqual(_scheduler(Campaign.load(self.root / "run", ROOT)), "pbs")

    def test_legacy_fixed_q_and_neutrino_list_still_build(self):
        config = self.build(LEGACY)
        self.assertEqual(config["model_settings"]["accdm_momentum_bins"], 1001)
        self.assertEqual(config["model_settings"]["accdm_momentum_sampling"], "explicit")
        self.assertEqual(_scheduler(Campaign.load(self.root / "run", ROOT)), "pbs")

    def test_profile_change_cannot_reuse_legacy_campaign(self):
        self.build(LEGACY)
        before = (self.root / "run/campaign.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "change the base model"):
            self.build(BASE)
        self.assertEqual((self.root / "run/campaign.json").read_bytes(), before)

    def test_precision_change_requires_new_campaign(self):
        self.build(BASE)
        self.base["class_params"]["accdm_q_bins_per_decade"] = 100.0
        altered = self.root / "altered.json"
        altered.write_text(json.dumps(self.base))
        with self.assertRaisesRegex(ValueError, "change the base model"):
            self.build(altered)

    def test_new_class_input_conflicts_are_rejected_before_writing_models(self):
        cases = (
            ({"m_ncdm": "0.06, 0.1"}, "only one"),
            ({"deg_ncdm": "1, 2"}, "daughter"),
            ({"ncdm_fluid_approximation": 2}, "exact hierarchy"),
            ({"E_acc_in_GeV": 1e11}, "builder- or generator-owned"),
            ({"ncdm_quadrature_strategy": "0, 0"}, "strategy 4 or 5"),
            ({"accdm_q_number_tol": 0}, "lie in"),
            ({"accdm_q_bins_per_decade": -1}, "positive"),
        )
        for overrides, message in cases:
            with self.subTest(overrides=overrides):
                base = copy.deepcopy(self.base)
                base["class_params"].update(overrides)
                with self.assertRaisesRegex(ValueError, message):
                    _validate_base_model(base)

    def test_legacy_recovery_rejects_automatic_grid_with_a_clear_error(self):
        self.build(BASE)
        with self.assertRaisesRegex(ValueError, "5001 accDM momentum bins"):
            prepare_recovery_campaigns(self.root / "run", self.root / "recovery")

    def test_neutrino_weight_excludes_daughter_for_both_mass_inputs(self):
        common = {"N_ncdm": 2, "deg_ncdm": "1, 1", "m_acc_in_GeV": 1e11}
        self.assertEqual(_sum_massive_neutrino_mass(dict(common, m_nu=0.06)), 0.06)
        self.assertEqual(_sum_massive_neutrino_mass(dict(common, m_ncdm="0.06, 1e20")), 0.06)
        self.assertEqual(_sum_massive_neutrino_mass({"N_ncdm": 1, "m_ncdm": 0.06}), 0.06)
        self.assertEqual(_sum_massive_neutrino_mass({"N_ncdm": 1, "m_nu": 0.02, "deg_ncdm": 3}), 0.06)


if __name__ == "__main__":
    unittest.main()
