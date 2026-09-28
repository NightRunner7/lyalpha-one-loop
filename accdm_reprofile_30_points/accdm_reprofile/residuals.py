"""Independent full-six-parameter residuals and analytic Jacobian.

This module uses the production model directly and never calls an amplitude
profiler.  The parameter domain is unchanged.  In particular signed alpha_ct
is passed in native units, without subtracting a box midpoint.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular


PARAMETER_NAMES = (
    "log_alpha_F", "beta_F", "alpha_bias", "beta_bias", "alpha_ct", "beta_ct"
)


def _column_norms(matrix):
    """Euclidean column norms without squaring very large raw entries."""
    matrix = np.asarray(matrix, dtype=float)
    largest = np.max(np.abs(matrix), axis=0)
    safe = np.where(largest > 0, largest, 1.0)
    return largest * np.sqrt(np.sum((matrix / safe) ** 2, axis=0))


class FullResidual:
    """Production residual plus an independently derived analytic Jacobian.

    ``ctx`` is the dictionary returned by inputs.load_context and must contain
    ``fitter`` and the exact saved ``bounds``.  Use with ``least_squares`` in the
    original six parameters, ``loss='linear'``, ``jac=full.jac`` and
    ``x_scale=full.scales(start)``.  Keep finite evaluation budgets, record
    termination status, and compare independently re-evaluated candidate chi2.
    In native units ``optimality`` can remain enormous for an otherwise tiny
    residual because dP/dalpha_ct is enormous.  Conversely an ``xtol`` exit
    alone does not establish convergence; use the independent-method checks.
    """

    def __init__(self, ctx):
        self.fitter = ctx["fitter"]
        self.bounds = np.asarray(ctx["bounds"], dtype=float)
        if self.bounds.shape != (6, 2) or not np.all(np.isfinite(self.bounds)):
            raise ValueError("FullResidual requires six finite saved bounds.")
        if np.any(self.bounds[:, 0] >= self.bounds[:, 1]):
            raise ValueError("Every saved lower bound must be below its upper bound.")
        self.ratio = ((1.0 + np.asarray(self.fitter.dataset.z, dtype=float)) /
                      (1.0 + self.fitter.config.z_pivot))
        self.log_ratio = np.log(self.ratio)
        self.common = np.asarray(self.fitter._siiii() * self.fitter._thermal())
        self.target = np.asarray(self.fitter.dataset.p1d, dtype=float)
        # Use the actual covariance, rather than trusting an approximate
        # diagonal-detection flag in a caller or an inverse covariance matrix.
        covariance = np.asarray(self.fitter.covariance, dtype=float)
        if covariance.shape != (len(self.target), len(self.target)):
            raise ValueError("Covariance shape does not match the data vector.")
        if np.array_equal(covariance, np.diag(np.diag(covariance))):
            diagonal = np.diag(covariance)
            if np.any(diagonal <= 0):
                raise ValueError("Covariance diagonal must be positive.")
            self._sigma = np.sqrt(diagonal)
            self._lower_cholesky = None
        else:
            self._sigma = None
            self._lower_cholesky = np.linalg.cholesky(covariance)
        self.residual_evaluations = 0
        self.jacobian_evaluations = 0

    @staticmethod
    def _theta(theta):
        value = np.asarray(theta, dtype=float)
        if value.shape != (6,) or not np.all(np.isfinite(value)):
            raise ValueError("Expected six finite nuisance parameters.")
        return value

    def _whiten(self, values):
        values = np.asarray(values, dtype=float)
        if self._sigma is not None:
            return values / (self._sigma if values.ndim == 1 else self._sigma[:, None])
        return solve_triangular(self._lower_cholesky, values, lower=True,
                                check_finite=True)

    def residual(self, theta):
        theta = self._theta(theta)
        self.residual_evaluations += 1
        return self._whiten(np.asarray(self.fitter.model(theta)) - self.target)

    def jac(self, theta):
        """Whitened derivatives in native coordinates, including tiny CT."""
        theta = self._theta(theta)
        self.jacobian_evaluations += 1
        log_af, beta_f, alpha_bias, beta_bias, alpha_ct, beta_ct = theta
        ratio = self.ratio
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            amplitude = np.exp(log_af) * ratio ** beta_f
            beta_shape = ratio ** beta_bias
            beta = alpha_bias * beta_shape
            ct_shape = self.fitter.i0_scale * ratio ** (-beta_ct)
            ct = alpha_ct * ct_shape
            prefactor = amplitude * self.common
            model = prefactor * (self.fitter.i0 + ct + 2.0 * beta * self.fitter.i2
                                 + beta ** 2 * self.fitter.i4)
            bias_derivative = 2.0 * prefactor * (self.fitter.i2 + beta * self.fitter.i4)
            raw = np.column_stack((
                model,
                model * self.log_ratio,
                bias_derivative * beta_shape,
                bias_derivative * beta * self.log_ratio,
                prefactor * ct_shape,
                -prefactor * ct * self.log_ratio,
            ))
        return self._whiten(raw)

    def scales(self, theta):
        """Static TRF scales accounting for its distance-to-bound scaling.

        In bounded TRF the Jacobian is multiplied by
        ``sqrt(distance_to_bound * x_scale)``.  Merely setting x_scale=1/||J||
        therefore leaves extreme CT columns severely ill-conditioned when
        alpha_ct~1e-87 but its physical bounds are +/-1000.  The conservative
        choice below makes that effective column norm at most about one at
        the start, without modifying either the coordinates or the bounds.
        Recompute scales for each independent start, not on every iteration.
        """
        theta = self._theta(theta)
        norms = _column_norms(self.jac(theta))
        defaults = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 10.0])
        distance = np.maximum(np.abs(theta - self.bounds[:, 0]),
                              np.abs(self.bounds[:, 1] - theta))
        distance = np.maximum(distance, np.finfo(float).tiny)
        nonzero = norms > 0
        log_scale = np.log(defaults)
        log_scale[nonzero] = np.minimum(
            log_scale[nonzero],
            -2.0 * np.log(norms[nonzero]) - np.log(distance[nonzero]))
        # Keep reciprocal scales representable. Values outside this range
        # require a different numerical representation and must not be clipped
        # silently into an apparently successful optimizer run.
        if np.any(log_scale < np.log(1e-280)) or not np.all(np.isfinite(log_scale)):
            raise FloatingPointError("Full residual requires unrepresentable TRF scaling.")
        return np.exp(log_scale)

    def validate(self, theta, *, derivative_step=1e-5, derivative_rtol=5e-5,
                 chi2_atol=1e-7, chi2_rtol=1e-11):
        """Check canonical chi2 and analytic derivatives at one exact vector.

        Finite differences use sensitivity-sized physical steps, not the much
        smaller TRF x_scale values.  Tests remain inside the original domain.
        A returned ``passed`` flag is a numerical check, not a fit convergence
        or global-minimum claim.
        """
        theta = self._theta(theta)
        if np.any(theta < self.bounds[:, 0]) or np.any(theta > self.bounds[:, 1]):
            raise ValueError("Validation theta is outside the saved bounds.")
        residual = self.residual(theta)
        analytic = self.jac(theta)
        chi2 = float(residual @ residual)
        canonical = float(self.fitter.chi2(theta))
        chi2_ok = bool(np.isfinite(chi2) and np.isfinite(canonical) and
                       np.isclose(chi2, canonical, rtol=chi2_rtol, atol=chi2_atol))
        norms = _column_norms(analytic)
        typical = np.array([1., 1., 1., 1., 1., 10.])
        positive = norms > 0
        typical[positive] = np.minimum(typical[positive], 1. / norms[positive])
        checks = []
        for j, name in enumerate(PARAMETER_NAMES):
            wanted_step = derivative_step * typical[j]
            plus = theta.copy()
            minus = theta.copy()
            plus[j] = min(self.bounds[j, 1], theta[j] + wanted_step)
            minus[j] = max(self.bounds[j, 0], theta[j] - wanted_step)
            if plus[j] == minus[j]:
                checks.append({"parameter": name, "passed": False,
                               "reason": "finite_difference_step_unrepresentable"})
                continue
            rplus = residual if plus[j] == theta[j] else self.residual(plus)
            rminus = residual if minus[j] == theta[j] else self.residual(minus)
            finite_scaled = ((rplus - rminus) /
                             ((plus[j] - minus[j]) / typical[j]))
            analytic_scaled = analytic[:, j] * typical[j]
            absolute = float(np.max(np.abs(finite_scaled - analytic_scaled)))
            denominator = max(1.0, float(np.max(np.abs(analytic_scaled))))
            error = absolute / denominator
            checks.append({"parameter": name, "passed": bool(np.isfinite(error)
                            and error <= derivative_rtol),
                           "scaled_max_error": error,
                           "physical_difference_scale": float(typical[j])})
        return {"passed": bool(chi2_ok and all(c["passed"] for c in checks)),
                "chi2_matches": chi2_ok, "chi2_residual": chi2,
                "chi2_canonical": canonical,
                "chi2_abs_difference": abs(chi2 - canonical),
                "analytic_jacobian_checks": checks,
                "trf_x_scale": self.scales(theta).tolist(),
                "validation_kind": "full_six_parameter_residual_and_jacobian"}
