from pathlib import Path
import unittest

import numpy as np

from lyalpha_pt.data import load_dr12


ROOT = Path(__file__).resolve().parents[1]


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_dr12(ROOT / "data")

    def test_fiducial_selection(self):
        self.assertEqual(self.data.n_data, 245)
        np.testing.assert_allclose(
            self.data.z_unique, [3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2]
        )

    def test_covariance_modes(self):
        for mode in (
            "paper_diag",
            "stat_diag",
            "stat_corr",
            "total_corr",
            "outer_systematics",
        ):
            covariance = self.data.covariance(mode)
            self.assertEqual(covariance.shape, (245, 245))
            self.assertTrue(np.all(np.diag(covariance) > 0))
            np.testing.assert_allclose(covariance, covariance.T, atol=1e-14)


if __name__ == "__main__":
    unittest.main()
