from pathlib import Path
import unittest

import numpy as np

from lyalpha_pt.data import load_dr12
from lyalpha_pt.fit import EffectiveModelFit
from lyalpha_pt.theory import load_theory


ROOT = Path(__file__).resolve().parents[1]


class OneLoopPaperBenchmark(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_dr12(ROOT / "data")
        cls.theory = load_theory(
            ROOT / "results" / "lcdm_2021_massless_precision_v3_1.npz"
        )

    def test_2021_one_loop_chi2_at_k20(self):
        fit = EffectiveModelFit(
            self.data,
            self.theory,
            mode="one_loop",
            k_uv_cut=20.0,
            covariance_mode="paper_diag",
        )
        best_fit = np.array(
            [
                -4.656906008661715,
                5.171409922339488,
                -3.257756148155558,
                1.4507821023806358,
                -0.047644434197182804,
                38.38822405801375,
            ]
        )
        self.assertAlmostEqual(fit.chi2(best_fit), 192.89658753233607, places=5)

    def test_cutoff_continuation_recovers_k20_basin(self):
        fit = EffectiveModelFit(
            self.data,
            self.theory,
            mode="one_loop",
            k_uv_cut=20.0,
            covariance_mode="paper_diag",
        )
        k15_fit = np.array(
            [
                -4.666893528425126,
                5.157475119673827,
                -3.5368811160695692,
                1.2292760344899973,
                -0.03243299172565021,
                238.26058465355084,
            ]
        )
        result = fit.continue_staged(
            k15_fit,
            expand_counterterm=True,
            profile_amplitudes=True,
        )
        self.assertLessEqual(fit.chi2(result.x), 192.897)
        self.assertFalse(
            any(row["near_bound"] for row in result.boundary_diagnostics)
        )


if __name__ == "__main__":
    unittest.main()
