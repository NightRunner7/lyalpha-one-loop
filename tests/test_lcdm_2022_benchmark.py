from pathlib import Path
import unittest

import numpy as np

from lyalpha_pt.data import load_dr12
from lyalpha_pt.fit import EffectiveModelFit
from lyalpha_pt.theory import load_theory


ROOT = Path(__file__).resolve().parents[1]


class LCDM2022Benchmark(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_dr12(ROOT / "data")
        cls.theory = load_theory(
            ROOT / "results" / "lcdm_2022_planck_precision_v3_3.npz"
        )

    def test_table_1_cosmology_and_neutrino_prescription(self):
        model = self.theory.metadata["model"]
        params = model["class_params"]
        expected = {
            "omega_b": 0.02237,
            "omega_cdm": 0.1200,
            "100*theta_s": 1.04110,
            "ln10^{10}A_s": 3.044,
            "n_s": 0.9649,
            "tau_reio": 0.0544,
            "N_ur": 2.0328,
            "N_ncdm": 1,
            "m_ncdm": 0.06,
        }
        for name, value in expected.items():
            self.assertAlmostEqual(float(params[name]), value, places=8)
        self.assertEqual(model["loop_source"], "cb")
        self.assertEqual(model["loop_weight"], "one_minus_fnu_squared")
        derived = self.theory.metadata["derived_cosmology"]
        self.assertAlmostEqual(derived["sigma8"], 0.810, delta=5e-4)
        self.assertAlmostEqual(derived["S8"], 0.833, delta=5e-4)

    def test_linear_and_one_loop_reference_fit(self):
        linear = EffectiveModelFit(
            self.data,
            self.theory,
            mode="linear",
            k_uv_cut=20.0,
            covariance_mode="paper_diag",
        )
        one_loop = EffectiveModelFit(
            self.data,
            self.theory,
            mode="one_loop",
            k_uv_cut=20.0,
            covariance_mode="paper_diag",
        )
        theta_linear = np.array(
            [-4.3422300273, 3.4193209794, 0.8183054540, 3.8779482464, -0.07834343, 49.073975]
        )
        theta_loop = np.array(
            [-4.62950070, 5.161708, -3.269654, 1.436682, -0.0464256, 39.778935]
        )
        chi2_linear = linear.chi2(theta_linear)
        chi2_loop = one_loop.chi2(theta_loop)
        self.assertAlmostEqual(chi2_linear, 206.248063, places=4)
        self.assertAlmostEqual(chi2_loop, 193.196611, places=4)
        self.assertAlmostEqual(chi2_loop - chi2_linear, -13.051453, places=4)


if __name__ == "__main__":
    unittest.main()
