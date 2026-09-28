from __future__ import annotations

import argparse
import unittest

import numpy as np

from scripts.compare_model_responses import (
    fixed_nuisance_theta,
    parse_labeled_path,
    percent_response,
    response_statistics,
)


class _FakeFitter:
    def __init__(self, i0_scale: float):
        self.i0_scale = i0_scale


class CompareModelResponsesTests(unittest.TestCase):
    def test_labeled_path_preserves_label_and_path(self) -> None:
        label, path = parse_labeled_path("m=1e15 GeV, accDM best=runs/point.npz")
        self.assertEqual(label, "m=1e15 GeV, accDM best")
        self.assertEqual(str(path), "runs/point.npz")

        with self.assertRaises(argparse.ArgumentTypeError):
            parse_labeled_path("missing-separator")

    def test_fixed_nuisance_transfer_preserves_counterterm_amplitude(self) -> None:
        theta = np.array([-4.0, 5.0, -3.0, 1.0, -0.04, 20.0])
        reference = _FakeFitter(8.0)
        target = _FakeFitter(10.0)
        transferred = fixed_nuisance_theta(theta, reference, target)

        np.testing.assert_allclose(transferred[[0, 1, 2, 3, 5]], theta[[0, 1, 2, 3, 5]])
        self.assertAlmostEqual(
            reference.i0_scale * theta[4],
            target.i0_scale * transferred[4],
        )

    def test_percent_response_and_statistics(self) -> None:
        response = percent_response(np.array([1.1, 1.8]), np.array([1.0, 2.0]))
        np.testing.assert_allclose(response, [10.0, -10.0])
        stats = response_statistics(response)
        self.assertAlmostEqual(stats["min_pct"], -10.0)
        self.assertAlmostEqual(stats["max_pct"], 10.0)
        self.assertAlmostEqual(stats["max_abs_pct"], 10.0)
        self.assertAlmostEqual(stats["rms_pct"], 10.0)


if __name__ == "__main__":
    unittest.main()
