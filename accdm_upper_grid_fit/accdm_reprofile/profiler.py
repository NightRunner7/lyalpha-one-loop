"""Stable bounded variable projection for the verified lyalpha_pt effective model.

The two physical coefficients are a=exp(log_alpha_F) and c=a*alpha_ct.
Their feasible set is a trapezoid, not an independently bounded rectangle.
No CLASS calculation or production-source modification is performed here.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.linalg import cholesky, solve_triangular

PARAMETER_NAMES = ('log_alpha_F', 'beta_F', 'alpha_bias', 'beta_bias', 'alpha_ct', 'beta_ct')
NONLINEAR_INDICES = (1, 2, 3, 5)


class ProfileNumericsError(RuntimeError):
    """A numerically reconstructed candidate does not match the canonical model."""


def _norm(vector):
    """Euclidean norm without squaring large unscaled numbers."""
    scale = float(np.max(np.abs(vector)))
    return 0.0 if scale == 0 else scale * float(np.linalg.norm(vector / scale))


def _signed_exp(sign, logarithm):
    if sign == 0:
        return 0.0
    if logarithm > np.log(np.finfo(float).max):
        return math.copysign(np.inf, sign)
    # np.exp preserves representable subnormal amplitudes.
    return math.copysign(float(np.exp(logarithm)), sign)


class StableProfiler:
    """Constrained amplitude profiler using the original saved six bounds.

    ctx must contain fitter, theta and bounds. If it contains validated=False,
    construction fails; contexts without this optional marker are allowed for
    explicit unit tests and independently validated application callers.
    """

    def __init__(self, ctx):
        if ctx.get('validated', True) is not True:
            raise ValueError('Likelihood context has not passed reproduction checks.')
        self.ctx = ctx
        self.fitter = ctx['fitter']
        self.bounds6 = np.asarray(ctx['bounds'], dtype=float).copy()
        if self.bounds6.shape != (6, 2) or not np.isfinite(self.bounds6).all():
            raise ValueError('Six finite saved bounds are required.')
        if np.any(self.bounds6[:, 0] >= self.bounds6[:, 1]):
            raise ValueError('Each lower bound must be smaller than its upper bound.')
        self.bounds4 = self.bounds6[list(NONLINEAR_INDICES)].copy()
        self.baseline_theta = np.asarray(ctx['theta'], dtype=float).copy()
        self.dataset = self.fitter.dataset
        self.target = np.asarray(self.dataset.p1d, dtype=float)
        covariance = np.asarray(self.fitter.covariance, dtype=float)
        if covariance.shape != (len(self.target), len(self.target)):
            raise ValueError('Covariance and data sizes disagree.')
        self._chol = cholesky(covariance, lower=True, check_finite=True)
        # Only use an exact diagonal matrix for the direct whitening shortcut.
        self._sigma = (np.sqrt(np.diag(covariance))
                       if np.array_equal(covariance, np.diag(np.diag(covariance))) else None)
        self.y = self.whiten(self.target)
        self.log_ratio = np.log((1 + self.dataset.z) / (1 + self.fitter.config.z_pivot))
        self.common = np.asarray(self.fitter._siiii() * self.fitter._thermal(), dtype=float)
        self.i0, self.i2, self.i4 = (np.asarray(getattr(self.fitter, n), dtype=float)
                                     for n in ('i0', 'i2', 'i4'))
        if np.any(self.common <= 0) or not np.isfinite(self.common).all():
            raise ValueError('Expected positive finite SiIII/thermal factors.')
        if not np.isfinite(self.fitter.i0_scale) or self.fitter.i0_scale <= 0:
            raise ValueError('Counterterm I0 normalization must be positive.')
        self.a_bounds = np.exp(self.bounds6[0])
        if not np.isfinite(self.a_bounds).all() or np.any(self.a_bounds <= 0):
            raise ValueError('Amplitude bounds are not representable as positive floats.')
        self.ct_bounds = self.bounds6[4].copy()

    @staticmethod
    def q_from_theta(theta):
        return np.asarray(theta, dtype=float)[list(NONLINEAR_INDICES)].copy()

    def whiten(self, values):
        values = np.asarray(values, dtype=float)
        if self._sigma is not None:
            return values / (self._sigma if values.ndim == 1 else self._sigma[:, None])
        return solve_triangular(self._chol, values, lower=True, check_finite=False)

    def residual(self, theta):
        """Whitened full-model residual; ||r||² equals canonical chi2."""
        residual = self.whiten(np.asarray(self.fitter.model(theta)) - self.target)
        if not np.isfinite(residual).all():
            raise ProfileNumericsError('Canonical residual is not finite.')
        return residual

    def _basis(self, q):
        beta_f, alpha_bias, beta_bias, beta_ct = q
        with np.errstate(over='raise', invalid='raise'):
            bias = alpha_bias * np.exp(beta_bias * self.log_ratio)
            base = self.common * np.exp(beta_f * self.log_ratio) * (
                self.i0 + 2 * bias * self.i2 + bias**2 * self.i4)
            ct_log = (np.log(self.common) + np.log(self.fitter.i0_scale)
                      + (beta_f - beta_ct) * self.log_ratio)
            shift = float(np.max(ct_log))
            ct_scaled = np.exp(ct_log - shift)
        wbase, wct = self.whiten(base), self.whiten(ct_scaled)
        nbase, nct = _norm(wbase), _norm(wct)
        if not np.isfinite(nbase) or not np.isfinite(nct) or nct == 0:
            raise ProfileNumericsError('Nonfinite or zero counterterm basis norm.')
        # A null base is allowed: edges still solve the rank-deficient problem.
        if nbase == 0:
            nbase = 1.0
        u, v = wbase / nbase, wct / nct
        return u, v, nbase, shift + np.log(nct), base, ct_scaled, shift

    def _theta(self, q, a, alpha_ct):
        if not np.isfinite(a) or a <= 0 or not np.isfinite(alpha_ct):
            return None
        loga = float(np.log(a))
        # Remove only roundoff at an explicitly active boundary, never an
        # unconstrained solution materially outside the saved feasible set.
        if loga < self.bounds6[0, 0] - 2e-13 or loga > self.bounds6[0, 1] + 2e-13:
            return None
        if alpha_ct < self.ct_bounds[0] or alpha_ct > self.ct_bounds[1]:
            return None
        loga = float(np.clip(loga, *self.bounds6[0]))
        return np.array([loga, q[0], q[1], q[2], alpha_ct, q[3]])

    def _normalized_coefficients(self, theta, nbase, log_nct):
        a = float(np.exp(theta[0]))
        alpha = float(theta[4])
        ct = (0.0 if alpha == 0 else
              _signed_exp(alpha, theta[0] + np.log(abs(alpha)) + log_nct))
        return a * nbase, ct

    def profile(self, q):
        """Return (canonical chi2, six theta, diagnostics) at fixed nonlinear q.

        The convex two-amplitude optimum is among the unconstrained SVD
        solution and the four one-dimensional feasible edges. All reported
        values are evaluated again on the exact returned six-vector.
        """
        q = np.asarray(q, dtype=float)
        if q.shape != (4,) or not np.isfinite(q).all():
            raise ValueError('Four finite nonlinear parameters are required.')
        if np.any(q < self.bounds4[:, 0]) or np.any(q > self.bounds4[:, 1]):
            raise ValueError('Nonlinear parameters are outside the saved bounds.')
        u, v, nbase, log_nct, base, ct_scaled, shift = self._basis(q)
        design = np.column_stack((u, v))
        coeff, _, rank, singular = np.linalg.lstsq(design, self.y, rcond=None)
        candidates = []

        def add(a, alpha, origin):
            theta = self._theta(q, a, alpha)
            if theta is None:
                return
            A, C = self._normalized_coefficients(theta, nbase, log_nct)
            with np.errstate(over='ignore', invalid='ignore'):
                rr = A * u + C * v - self.y
                chi = float(rr @ rr)
            if np.isfinite(chi):
                candidates.append((chi, theta, origin, rr))

        a = float(coeff[0] / nbase)
        if a > 0 and np.isfinite(a):
            alpha = (0.0 if coeff[1] == 0 else
                     _signed_exp(coeff[1], np.log(abs(coeff[1])) - np.log(a) - log_nct))
            add(a, alpha, 'interior_svd')

        # Fixed a: analytically minimize over the remaining linear coefficient.
        for a, label in zip(self.a_bounds, ('a_lower', 'a_upper')):
            C = float(v @ (self.y - a * nbase * u))
            limits = [0.0 if t == 0 else _signed_exp(t, np.log(a) + np.log(abs(t)) + log_nct)
                      for t in self.ct_bounds]
            C = float(np.clip(C, limits[0], limits[1]))
            alpha = (0.0 if C == 0 else
                     _signed_exp(C, np.log(abs(C)) - np.log(a) - log_nct))
            # A decoded endpoint may differ by one ulp; enforce the exact edge.
            alpha = float(np.clip(alpha, *self.ct_bounds))
            add(float(a), alpha, label)

        # Fixed alpha_ct: model = a*(base+alpha_ct*counterterm_shape).
        for alpha, label in zip(self.ct_bounds, ('alpha_ct_lower', 'alpha_ct_upper')):
            log_t = -np.inf if alpha == 0 else np.log(abs(alpha)) + log_nct
            scale = max(np.log(nbase), log_t)
            w = np.exp(np.log(nbase) - scale) * u
            if alpha != 0:
                w = w + np.sign(alpha) * np.exp(log_t - scale) * v
            n = _norm(w)
            if n == 0:
                a = float(self.a_bounds[0])
            else:
                numerator = float((w / n) @ self.y)
                a = (0.0 if numerator <= 0 else
                     _signed_exp(1, np.log(numerator) - np.log(n) - scale))
                a = float(np.clip(a, *self.a_bounds))
            add(a, float(alpha), label)
        if not candidates:
            raise ProfileNumericsError('No finite feasible amplitude candidate.')
        design_chi2, theta, origin, profiled_residual = min(candidates, key=lambda item: item[0])
        canonical_residual = self.residual(theta)
        canonical_chi2 = float(self.fitter.chi2(theta))
        whitened_chi2 = float(canonical_residual @ canonical_residual)
        if not np.isfinite(canonical_chi2) or canonical_chi2 >= 1e99:
            raise ProfileNumericsError('Canonical likelihood rejected the profiled candidate.')
        discrepancy = float(np.max(np.abs(canonical_residual - profiled_residual)))
        residual_scale = max(1.0, float(np.max(np.abs(canonical_residual))), float(np.max(np.abs(self.y))))
        if discrepancy > 2e-8 * residual_scale:
            raise ProfileNumericsError(f'Stabilized basis differs from canonical model: {discrepancy:g}.')
        if not np.isclose(canonical_chi2, whitened_chi2, atol=1e-7, rtol=1e-9):
            raise ProfileNumericsError('Whitened residual does not reproduce canonical chi2.')
        condition = (float(singular[0] / singular[-1]) if singular[-1] > 0 else np.inf)
        diagnostic = dict(amplitude_candidate_origin=origin, amplitude_candidates=len(candidates),
                          design_chi2=design_chi2, canonical_chi2=canonical_chi2,
                          whitened_chi2=whitened_chi2, design_residual_max_error=discrepancy,
                          normalized_design_rank=int(rank), normalized_design_condition=condition,
                          counterterm_log_norm=float(log_nct), counterterm_log_shift=float(shift))
        return canonical_chi2, theta, diagnostic

    def validate(self):
        """Check model identity and amplitude improvement at the saved q."""
        theta = self.baseline_theta
        saved_chi2 = float(self.fitter.chi2(theta))
        residual_chi2 = float(self.residual(theta) @ self.residual(theta))
        profiled_chi2, optimum, diag = self.profile(self.q_from_theta(theta))
        tolerance = 1e-6 + 1e-8 * abs(saved_chi2)
        if abs(residual_chi2 - saved_chi2) > tolerance:
            raise ProfileNumericsError('Saved point residual/chi2 identity failed.')
        if profiled_chi2 > saved_chi2 + tolerance:
            raise ProfileNumericsError('Constrained profile is worse than feasible saved amplitudes.')
        return dict(validated=True, saved_chi2=saved_chi2, residual_chi2=residual_chi2,
                    fixed_q_profile_chi2=profiled_chi2, fixed_q_gain=saved_chi2-profiled_chi2,
                    fixed_q_theta=optimum.tolist(), **diag)
