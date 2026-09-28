import unittest

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import simpson

from lyalpha_pt.spt import (
    f2_eds,
    g2_eds,
    p13_channels_analytic,
    p13_channels_richardson,
    p13_dd_analytic,
    p13_dtheta_analytic,
    p13_thetatheta_analytic,
    p22_channels,
    p22_channels_full_qp_reference,
)


class SPTTests(unittest.TestCase):
    def test_second_order_kernels(self):
        q1 = np.array([0.7, 1.3, 2.1])
        q2 = np.array([1.1, 0.8, 1.7])
        mu = np.array([-0.4, 0.2, 0.9])
        np.testing.assert_allclose(
            f2_eds(q1, q2, mu),
            5 / 7 + 0.5 * mu * (q1 / q2 + q2 / q1) + 2 / 7 * mu**2,
        )
        np.testing.assert_allclose(
            g2_eds(q1, q2, mu),
            3 / 7 + 0.5 * mu * (q1 / q2 + q2 / q1) + 4 / 7 * mu**2,
        )

    def test_p13_density_against_closed_form(self):
        power = lambda q: np.asarray(q) * np.exp(-(np.asarray(q) / 1.2) ** 2)
        k = 0.5
        direct, error = p13_channels_richardson(
            k,
            power,
            q_min=0.005,
            q_max=5.0,
            n_q=100,
            n_mu=20,
            epsilon_relative=1e-3,
        )
        analytic = p13_dd_analytic(
            k, power, q_min=0.005, q_max=5.0, n_q=10001
        )
        self.assertLess(abs(direct[0] / analytic - 1), 5e-4)
        self.assertLess(error[0], abs(direct[0]) * 1e-4)

    def test_p13_velocity_channels_against_closed_form(self):
        power = lambda q: np.asarray(q) * np.exp(-(np.asarray(q) / 1.2) ** 2)
        k = 0.5
        direct, error = p13_channels_richardson(
            k,
            power,
            q_min=0.005,
            q_max=5.0,
            n_q=100,
            n_mu=20,
            epsilon_relative=1e-3,
        )
        analytic_dtheta = p13_dtheta_analytic(
            k, power, q_min=0.005, q_max=5.0, n_q=10001
        )
        analytic_thetatheta = p13_thetatheta_analytic(
            k, power, q_min=0.005, q_max=5.0, n_q=10001
        )
        self.assertLess(abs(direct[1] / analytic_dtheta - 1), 5e-4)
        self.assertLess(abs(direct[2] / analytic_thetatheta - 1), 5e-4)
        self.assertLess(error[1], abs(direct[1]) * 1e-4)
        self.assertLess(error[2], abs(direct[2]) * 1e-4)

    def test_closed_p13_all_channels_and_resolution_error(self):
        power = lambda q: np.asarray(q) * np.exp(-(np.asarray(q) / 1.2) ** 2)
        values, error = p13_channels_analytic(
            0.5,
            power,
            q_min=0.005,
            q_max=5.0,
            n_q=501,
        )
        reference = np.array(
            [
                p13_dd_analytic(
                    0.5, power, q_min=0.005, q_max=5.0, n_q=10001
                ),
                p13_dtheta_analytic(
                    0.5, power, q_min=0.005, q_max=5.0, n_q=10001
                ),
                p13_thetatheta_analytic(
                    0.5, power, q_min=0.005, q_max=5.0, n_q=10001
                ),
            ]
        )
        np.testing.assert_allclose(values, reference, rtol=3e-7, atol=0)
        self.assertEqual(values[1], 0.5 * (values[0] + values[2]))
        self.assertTrue(np.all(error >= 0))
        self.assertLess(np.max(error / np.abs(values)), 2e-5)

    def test_p22_qp_variables_against_qmu_integration(self):
        power = lambda q: np.asarray(q) * np.exp(-(np.asarray(q) / 1.2) ** 2)
        k = 0.5
        q_min = 0.005
        q_max = 5.0
        qp = p22_channels(
            k,
            power,
            q_min=q_min,
            q_max=q_max,
            n_q_low=180,
            n_q_mid=360,
            n_q_high=180,
            n_p=100,
        )

        q = np.geomspace(q_min, q_max, 1600)
        mu, weights = leggauss(180)
        qq = q[:, None]
        pp = np.sqrt(k**2 + qq**2 - 2 * k * qq * mu[None, :])
        valid = pp >= q_min
        p_eval = np.maximum(pp, q_min)
        mu_qp = (k * mu[None, :] - qq) / np.maximum(pp, 1e-300)
        f2 = f2_eds(qq, p_eval, mu_qp)
        g2 = g2_eds(qq, p_eval, mu_qp)
        common = power(qq) * power(p_eval) * valid
        qmu = np.array(
            [
                simpson(
                    q**2 * np.sum(common * kernel * weights, axis=1), x=q
                )
                / (2 * np.pi**2)
                for kernel in (f2**2, f2 * g2, g2**2)
            ]
        )
        np.testing.assert_allclose(qp, qmu, rtol=8e-5, atol=0)

    def test_p22_half_domain_against_full_qp_reference(self):
        power = lambda q: np.asarray(q) * np.exp(-(np.asarray(q) / 1.2) ** 2)
        common = dict(q_min=0.005, q_max=5.0)
        half_domain = p22_channels(
            0.5,
            power,
            n_q_low=180,
            n_q_mid=360,
            n_q_high=180,
            n_p=100,
            **common,
        )
        full_domain = p22_channels_full_qp_reference(
            0.5,
            power,
            n_q_per_region=801,
            n_p=140,
            **common,
        )
        np.testing.assert_allclose(half_domain, full_domain, rtol=3e-6, atol=0)

    def test_p22_symmetry_reduction_is_grid_stable(self):
        power = lambda q: np.asarray(q) * np.exp(-(np.asarray(q) / 1.2) ** 2)
        common = dict(q_min=0.005, q_max=5.0, middle_width=0.25)
        coarse = p22_channels(
            0.5,
            power,
            n_q_low=90,
            n_q_mid=180,
            n_q_high=90,
            n_p=60,
            **common,
        )
        fine = p22_channels(
            0.5,
            power,
            n_q_low=180,
            n_q_mid=360,
            n_q_high=180,
            n_p=100,
            **common,
        )
        np.testing.assert_allclose(coarse, fine, rtol=2e-6, atol=0)


if __name__ == "__main__":
    unittest.main()
