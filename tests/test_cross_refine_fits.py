from __future__ import annotations

import unittest

import numpy as np

from lyalpha_pt.fit import PARAMETER_NAMES
from scripts.cross_refine_fits import (
    theta_from_record,
    transfer_counterterm_amplitude,
)


class CrossRefineFitsTests(unittest.TestCase):
    def test_theta_uses_canonical_parameter_order(self) -> None:
        parameters = {name: float(index + 1) for index, name in enumerate(reversed(PARAMETER_NAMES))}
        theta = theta_from_record({"parameters": parameters})
        np.testing.assert_allclose(theta, [parameters[name] for name in PARAMETER_NAMES])

    def test_counterterm_transfer_preserves_dimensional_amplitude(self) -> None:
        theta = np.array([-4.0, 5.0, -3.0, 1.2, -0.04, 80.0])
        transferred = transfer_counterterm_amplitude(theta, 8.0, 10.0)
        alpha_ct_index = PARAMETER_NAMES.index("alpha_ct")
        self.assertAlmostEqual(
            8.0 * theta[alpha_ct_index],
            10.0 * transferred[alpha_ct_index],
        )
        unchanged = [index for index in range(6) if index != alpha_ct_index]
        np.testing.assert_allclose(transferred[unchanged], theta[unchanged])

    def test_counterterm_transfer_rejects_zero_target_scale(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be zero"):
            transfer_counterterm_amplitude(np.ones(6), 1.0, 0.0)


if __name__ == "__main__":
    unittest.main()
