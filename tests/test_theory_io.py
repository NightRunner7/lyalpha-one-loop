from pathlib import Path
import tempfile
import unittest

import numpy as np

from lyalpha_pt.theory import (
    INVERSE_MPC_TO_GEV,
    REDUCED_PLANCK_MASS_GEV,
    TheoryBundle,
    accdm_energy_budget_diagnostic,
    load_theory,
    save_theory,
)


def fake_bundle() -> TheoryBundle:
    z = np.array([3.0, 3.2])
    k_input = np.geomspace(1e-3, 30, 100)
    k_loop = np.geomspace(1e-3, 20, 30)
    p_input = np.array([(1 + zz) ** -2 * k_input**-1 for zz in z])
    p_tree = np.array([(1 + zz) ** -2 * k_loop**-1 for zz in z])
    zeros = np.zeros((2, 30, 3))
    channels = np.repeat(p_tree[:, :, None], 3, axis=2)
    bundle = TheoryBundle(
        metadata={"model": {"name": "fake"}},
        z=z,
        k_input=k_input,
        p_total_input=p_input,
        p_loop_input=p_input.copy(),
        k_loop=k_loop,
        p_tree=p_tree,
        p22=zeros.copy(),
        p13=zeros.copy(),
        p13_error=zeros.copy(),
        channels_one_loop=channels,
        hubble_km_s_mpc=np.array([300.0, 320.0]),
        velocity_to_hmpc=np.array([100.0, 102.0]),
        h=0.67,
        omega_m=0.31,
        loop_weight=1.0,
    )
    bundle.metadata["bundle_digest"] = bundle.bundle_digest()
    return bundle


class TheoryIOTests(unittest.TestCase):
    def test_accdm_energy_budget_is_a_background_diagnostic(self) -> None:
        params = {
            "eta_acc": 0.1,
            "m_acc_in_GeV": 1.0e12,
            "a_t_acc": 0.133,
        }
        diagnostic = accdm_energy_budget_diagnostic(
            params, hubble_inverse_mpc=2.5e-3
        )
        expected = (
            params["eta_acc"]
            * params["m_acc_in_GeV"]
            * 2.5e-3
            * INVERSE_MPC_TO_GEV
            / (3.0 * REDUCED_PLANCK_MASS_GEV**2)
        )
        self.assertAlmostEqual(diagnostic["ratio"], expected)
        self.assertAlmostEqual(diagnostic["z_acc"], 1.0 / 0.133 - 1.0)
        self.assertTrue(diagnostic["below_conservative_upper_bound"])

    def test_round_trip_and_digest(self):
        bundle = fake_bundle()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "theory.npz"
            save_theory(bundle, path)
            loaded = load_theory(path)
        self.assertEqual(
            loaded.metadata["bundle_digest"], bundle.metadata["bundle_digest"]
        )
        np.testing.assert_array_equal(loaded.channels_one_loop, bundle.channels_one_loop)

    def test_digest_detects_changed_spectrum(self):
        bundle = fake_bundle()
        bundle.p_tree[0, 0] *= 1.01
        with self.assertRaises(ValueError):
            bundle.validate()


if __name__ == "__main__":
    unittest.main()
