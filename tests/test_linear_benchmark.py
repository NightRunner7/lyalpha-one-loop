from pathlib import Path
import unittest

import numpy as np

from lyalpha_pt.data import load_dr12
from lyalpha_pt.fit import EffectiveModelFit
from lyalpha_pt.theory import load_theory


ROOT = Path(__file__).resolve().parents[1]


class LinearPaperBenchmark(unittest.TestCase):
    def test_2021_chi2(self):
        data = load_dr12(ROOT / "data")
        theory = load_theory(ROOT / "results" / "lcdm_2021_massless_smoke.npz")
        fit = EffectiveModelFit(
            data,
            theory,
            mode="linear",
            k_uv_cut=20.0,
            covariance_mode="paper_diag",
        )
        best_fit = np.array(
            [
                -4.371847879260232,
                3.3959128410176076,
                0.8373324533704364,
                3.8589626967820374,
                -0.07934612809066335,
                48.84124250027791,
            ]
        )
        self.assertAlmostEqual(fit.chi2(best_fit), 206.47043831383982, places=5)
        self.assertAlmostEqual(fit.chi2(best_fit), 206.5, delta=0.05)


if __name__ == "__main__":
    unittest.main()
