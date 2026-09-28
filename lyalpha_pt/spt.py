"""Direct EdS standard-perturbation-theory kernels and one-loop integrals.

Production P13 values use the closed one-dimensional EdS expressions.  The
regulated F3/G3 recursion is retained as an independent regression check, not
as the default scientific calculation.
"""

from __future__ import annotations

import itertools
from typing import Callable

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import simpson


ArrayFunction = Callable[[np.ndarray], np.ndarray]


def f2_eds(q1, q2, mu12):
    q1 = np.asarray(q1, dtype=float)
    q2 = np.asarray(q2, dtype=float)
    mu12 = np.asarray(mu12, dtype=float)
    return 5.0 / 7.0 + 0.5 * mu12 * (q1 / q2 + q2 / q1) + 2.0 / 7.0 * mu12**2


def g2_eds(q1, q2, mu12):
    q1 = np.asarray(q1, dtype=float)
    q2 = np.asarray(q2, dtype=float)
    mu12 = np.asarray(mu12, dtype=float)
    return 3.0 / 7.0 + 0.5 * mu12 * (q1 / q2 + q2 / q1) + 4.0 / 7.0 * mu12**2


def _norm2(vector: np.ndarray) -> float:
    return float(np.dot(vector, vector))


def _alpha(k1: np.ndarray, k2: np.ndarray, *, zero_tol: float = 1e-28) -> float:
    denominator = _norm2(k1)
    if denominator <= zero_tol:
        raise FloatingPointError("alpha received a vanishing first momentum")
    return float(np.dot(k1 + k2, k1) / denominator)


def _beta(k1: np.ndarray, k2: np.ndarray, *, zero_tol: float = 1e-28) -> float:
    k1_sq = _norm2(k1)
    k2_sq = _norm2(k2)
    if k1_sq <= zero_tol or k2_sq <= zero_tol:
        raise FloatingPointError("beta received a vanishing momentum")
    return float(_norm2(k1 + k2) * np.dot(k1, k2) / (2 * k1_sq * k2_sq))


def _fg_ordered(vectors: tuple[np.ndarray, ...]) -> tuple[float, float]:
    n = len(vectors)
    if n == 1:
        return 1.0, 1.0
    denominator = (2 * n + 3) * (n - 1)
    f_total = 0.0
    g_total = 0.0
    for split in range(1, n):
        left = vectors[:split]
        right = vectors[split:]
        f_left, g_left = _fg_ordered(left)
        f_right, g_right = _fg_ordered(right)
        k_left = np.sum(left, axis=0)
        k_right = np.sum(right, axis=0)
        alpha = _alpha(k_left, k_right)
        beta = _beta(k_left, k_right)
        f_total += g_left * (
            (2 * n + 1) * alpha * f_right + 2 * beta * g_right
        ) / denominator
        g_total += g_left * (
            3 * alpha * f_right + 2 * n * beta * g_right
        ) / denominator
    return float(f_total), float(g_total)


_PERMUTATIONS_3 = tuple(itertools.permutations(range(3)))


def fg3_symmetrized(vectors: tuple[np.ndarray, np.ndarray, np.ndarray]) -> tuple[float, float]:
    values = np.array(
        [_fg_ordered(tuple(vectors[i] for i in order)) for order in _PERMUTATIONS_3]
    )
    return float(np.mean(values[:, 0])), float(np.mean(values[:, 1]))


def _regulated_k_q_minus_q(
    k: float, q: float, mu: float, epsilon_relative: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sin_angle = np.sqrt(max(0.0, 1.0 - float(mu) ** 2))
    k_vector = np.array([0.0, 0.0, float(k)])
    q_vector = float(q) * np.array([sin_angle, 0.0, float(mu)])
    epsilon = epsilon_relative * max(float(k), float(q))
    epsilon_vector = np.array([0.0, epsilon, 0.0])
    return k_vector - epsilon_vector, q_vector, -q_vector + epsilon_vector


def p22_channels(
    k: float,
    p_linear: ArrayFunction,
    *,
    q_min: float,
    q_max: float,
    n_q_low: int,
    n_q_mid: int,
    n_q_high: int,
    n_p: int,
    middle_width: float = 0.25,
    p_min: float | None = None,
) -> np.ndarray:
    """Return P22 for dd, dtheta and thetatheta.

    The change of variables from ``mu`` to ``p=|k-q|`` removes the narrow
    angular feature.  The integrand is symmetric under ``q <-> p``.  We use
    that identity to integrate only the half-domain ``p >= q`` and multiply
    by two.  This maps the otherwise non-analytic ``p -> 0`` region at
    ``q=k`` onto the single smooth ``q -> 0`` boundary and removes a severe
    odd/even grid effect in the small P22+P13 residual.

    ``middle_width`` is retained for API/checkpoint compatibility with older
    configurations; the symmetry-reduced quadrature does not use it.
    """

    k = float(k)
    p_min = float(q_min if p_min is None else p_min)
    x_p, w_p = leggauss(int(n_p))

    def inner(q_values: np.ndarray) -> tuple[np.ndarray, ...]:
        q_values = np.asarray(q_values, dtype=float)
        p_lower = np.maximum(np.abs(k - q_values), q_values)
        p_upper = k + q_values
        p_mid = 0.5 * (p_upper + p_lower)
        p_half = 0.5 * (p_upper - p_lower)
        p = p_mid[:, None] + p_half[:, None] * x_p[None, :]
        p_eval = np.maximum(p, p_min)
        qq = q_values[:, None]
        mu_qp = (k * k - qq * qq - p * p) / (2 * qq * np.maximum(p, 1e-300))
        mu_qp = np.clip(mu_qp, -1.0, 1.0)
        f2 = f2_eds(qq, p_eval, mu_qp)
        g2 = g2_eds(qq, p_eval, mu_qp)
        common = p * p_linear(p_eval)
        return tuple(
            p_half * np.sum(common * kernel * w_p[None, :], axis=1)
            for kernel in (f2**2, f2 * g2, g2**2)
        )

    total = np.zeros(3)

    def integrate_region(q_values: np.ndarray) -> np.ndarray:
        values = inner(q_values)
        common_q = q_values * p_linear(q_values)
        return np.array([simpson(common_q * value, x=q_values) for value in values])

    # The lower p boundary changes from k-q to q at q=k/2.  Splitting there
    # makes each radial region smooth.  The former middle/high point budgets
    # are combined for the long q>=k/2 interval.
    q_split = min(max(0.5 * k, q_min), q_max)
    if q_split > q_min:
        total += integrate_region(np.geomspace(q_min, q_split, int(n_q_low)))
    if q_max > q_split:
        n_q_upper = int(n_q_mid) + int(n_q_high)
        total += integrate_region(np.geomspace(q_split, q_max, n_q_upper))
    # Twice the half-domain restores the full symmetric P22 integral.
    return 2 * total / (2 * np.pi**2 * k)


def p22_channels_full_qp_reference(
    k: float,
    p_linear: ArrayFunction,
    *,
    q_min: float,
    q_max: float,
    n_q_per_region: int = 1001,
    n_p: int = 240,
    p_min: float | None = None,
) -> np.ndarray:
    """Independent full-domain ``(q,p)`` reference for all P22 channels.

    Unlike :func:`p22_channels`, this routine does not use the
    ``q <-> p`` half-domain reduction.  It integrates the full radial domain
    and is intentionally slower.  The lower ``p`` cutoff is imposed on the
    integration domain itself; replacing only ``P(p)`` by ``P(p_min)`` would
    be inconsistent with the kernels and creates a false high-k discrepancy.
    """

    k = float(k)
    q_min = float(q_min)
    q_max = float(q_max)
    p_min = float(q_min if p_min is None else p_min)
    n_q_per_region = int(n_q_per_region)
    if n_q_per_region < 41 or n_p < 16:
        raise ValueError("Reference P22 quadrature is too small.")

    breakpoints = sorted(
        {
            value
            for value in (q_min, 0.5 * k, k, 2.0 * k, q_max)
            if q_min <= value <= q_max
        }
    )
    if breakpoints[0] != q_min:
        breakpoints.insert(0, q_min)
    if breakpoints[-1] != q_max:
        breakpoints.append(q_max)
    q_regions = []
    for lower, upper in zip(breakpoints[:-1], breakpoints[1:]):
        if upper <= lower:
            continue
        values = np.geomspace(lower, upper, n_q_per_region)
        if q_regions:
            values = values[1:]
        q_regions.append(values)
    q = np.concatenate(q_regions)

    x_p, w_p = leggauss(int(n_p))
    p_lower = np.maximum(np.abs(k - q), p_min)
    p_upper = k + q
    valid = p_upper > p_lower
    p_mid = 0.5 * (p_upper + p_lower)
    p_half = 0.5 * (p_upper - p_lower)
    p = p_mid[:, None] + p_half[:, None] * x_p[None, :]
    qq = q[:, None]
    mu_qp = (k * k - qq * qq - p * p) / (2 * qq * p)
    mu_qp = np.clip(mu_qp, -1.0, 1.0)
    f2 = f2_eds(qq, p, mu_qp)
    g2 = g2_eds(qq, p, mu_qp)
    common_p = p * p_linear(p)

    result = []
    for kernel in (f2**2, f2 * g2, g2**2):
        inner = p_half * np.sum(common_p * kernel * w_p[None, :], axis=1)
        inner[~valid] = 0.0
        result.append(
            simpson(q * p_linear(q) * inner, x=q) / (2 * np.pi**2 * k)
        )
    return np.asarray(result)


def _p13_at_regulator(
    k: float,
    p_linear: ArrayFunction,
    *,
    q_min: float,
    q_max: float,
    n_q: int,
    n_mu: int,
    epsilon_relative: float,
) -> np.ndarray:
    k = float(k)
    log_q = np.linspace(np.log(q_min), np.log(q_max), int(n_q))
    q_values = np.exp(log_q)
    mu_nodes, mu_weights = leggauss(int(n_mu))
    angular_f3 = np.zeros_like(q_values)
    angular_g3 = np.zeros_like(q_values)

    for iq, q in enumerate(q_values):
        f_values = np.empty_like(mu_nodes)
        g_values = np.empty_like(mu_nodes)
        for imu, mu in enumerate(mu_nodes):
            f_values[imu], g_values[imu] = fg3_symmetrized(
                _regulated_k_q_minus_q(k, q, float(mu), epsilon_relative)
            )
        angular_f3[iq] = np.dot(mu_weights, f_values)
        angular_g3[iq] = np.dot(mu_weights, g_values)

    radial = q_values**3 * p_linear(q_values)
    integral_f3 = simpson(radial * angular_f3, x=log_q)
    integral_g3 = simpson(radial * angular_g3, x=log_q)
    p_k = float(np.asarray(p_linear(np.array([k])))[0])
    return np.array(
        [
            3 * p_k * integral_f3 / (2 * np.pi**2),
            3 * p_k * (integral_f3 + integral_g3) / (4 * np.pi**2),
            3 * p_k * integral_g3 / (2 * np.pi**2),
        ]
    )


def p13_channels_richardson(
    k: float,
    p_linear: ArrayFunction,
    *,
    q_min: float,
    q_max: float,
    n_q: int,
    n_mu: int,
    epsilon_relative: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Return regulator-free P13 and a conservative extrapolation error.

    The symmetric transverse regulator has an O(epsilon^2) leading error.
    Values at epsilon and epsilon/2 therefore give
    ``P13(0) = (4*P13(eps/2)-P13(eps))/3``.
    """

    coarse = _p13_at_regulator(
        k,
        p_linear,
        q_min=q_min,
        q_max=q_max,
        n_q=n_q,
        n_mu=n_mu,
        epsilon_relative=epsilon_relative,
    )
    fine = _p13_at_regulator(
        k,
        p_linear,
        q_min=q_min,
        q_max=q_max,
        n_q=n_q,
        n_mu=n_mu,
        epsilon_relative=epsilon_relative / 2,
    )
    extrapolated = (4 * fine - coarse) / 3
    error = np.abs(extrapolated - fine)
    return extrapolated, error


def _p13_channels_analytic_at_resolution(
    k: float,
    p_linear: ArrayFunction,
    *,
    q_min: float,
    q_max: float,
    n_q: int,
) -> np.ndarray:
    """Evaluate all three closed EdS P13 channels on one radial grid."""

    k = float(k)
    q = np.geomspace(q_min, q_max, int(n_q))
    r = q / k
    density_bracket, velocity_bracket = _p13_closed_brackets(r)
    power = p_linear(q)
    density_integral = simpson(power * density_bracket, x=r)
    velocity_integral = simpson(power * velocity_bracket, x=r)
    p_k = float(np.asarray(p_linear(np.array([k])))[0])
    density = k**3 * p_k * density_integral / (252 * (2 * np.pi) ** 2)
    velocity = k**3 * p_k * velocity_integral / (84 * (2 * np.pi) ** 2)
    return np.array([density, 0.5 * (density + velocity), velocity])


def p13_channels_analytic(
    k: float,
    p_linear: ArrayFunction,
    *,
    q_min: float,
    q_max: float,
    n_q: int = 501,
) -> tuple[np.ndarray, np.ndarray]:
    """Return closed EdS P13 channels and a radial-resolution error estimate.

    The returned order is ``(dd, dtheta, thetatheta)``.  The cross channel is
    fixed by the one-loop identity
    ``P13_dtheta = (P13_dd + P13_thetatheta)/2``.  ``n_q`` must be odd so that
    both the requested grid and its nested coarse grid are suitable for
    Simpson integration.  The absolute fine-minus-coarse difference is kept
    as a deliberately conservative quadrature diagnostic.
    """

    n_q = int(n_q)
    if n_q < 41 or n_q % 2 == 0:
        raise ValueError("Analytic P13 requires an odd n_q >= 41.")
    n_q_coarse = (n_q + 1) // 2
    if n_q_coarse % 2 == 0:
        n_q_coarse += 1
    fine = _p13_channels_analytic_at_resolution(
        k,
        p_linear,
        q_min=q_min,
        q_max=q_max,
        n_q=n_q,
    )
    coarse = _p13_channels_analytic_at_resolution(
        k,
        p_linear,
        q_min=q_min,
        q_max=q_max,
        n_q=n_q_coarse,
    )
    return fine, np.abs(fine - coarse)


def p13_dd_analytic(
    k: float,
    p_linear: ArrayFunction,
    *,
    q_min: float,
    q_max: float,
    n_q: int = 5001,
) -> float:
    """Known one-dimensional EdS expression for the full density P13.

    This routine is primarily an independent regression test for the F3
    recursion.  Series expansions are used for very small and very large
    ``r=q/k`` so the expression remains stable over the full loop range.
    """

    return float(
        _p13_channels_analytic_at_resolution(
            k, p_linear, q_min=q_min, q_max=q_max, n_q=n_q
        )[0]
    )


def p13_thetatheta_analytic(
    k: float,
    p_linear: ArrayFunction,
    *,
    q_min: float,
    q_max: float,
    n_q: int = 5001,
) -> float:
    """Known one-dimensional EdS expression for the full velocity P13.

    The velocity-divergence convention is the one used throughout this
    project, ``theta = -div(v)/(a H f)``, so the linear density and theta
    spectra are identical.  Stable small- and large-r series are used as in
    :func:`p13_dd_analytic`.
    """

    return float(
        _p13_channels_analytic_at_resolution(
            k, p_linear, q_min=q_min, q_max=q_max, n_q=n_q
        )[2]
    )


def p13_dtheta_analytic(
    k: float,
    p_linear: ArrayFunction,
    *,
    q_min: float,
    q_max: float,
    n_q: int = 5001,
) -> float:
    """Return the closed one-dimensional EdS density--theta P13.

    At one loop the cross-channel contribution is the symmetric average of
    the density and velocity P13 terms.
    """

    return float(
        _p13_channels_analytic_at_resolution(
            k, p_linear, q_min=q_min, q_max=q_max, n_q=n_q
        )[1]
    )


def _p13_closed_brackets(r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the density and velocity P13 brackets without cancellation."""

    r = np.asarray(r, dtype=float)
    density = np.empty_like(r)
    velocity = np.empty_like(r)
    small = r < 0.05
    large = r > 20.0
    middle = ~(small | large)

    # Increasing-power coefficients in x=r^2, obtained by expanding the
    # logarithm around r=0 through O(r^14).
    density_small = np.array(
        [
            -168,
            928 / 5,
            -4512 / 35,
            416 / 21,
            2656 / 1155,
            3232 / 5005,
            544 / 2145,
            4384 / 36465,
        ]
    )
    velocity_small = np.array(
        [
            -56,
            -32 / 5,
            -96 / 7,
            352 / 105,
            544 / 1155,
            736 / 5005,
            928 / 15015,
            224 / 7293,
        ]
    )
    density[small] = np.polynomial.polynomial.polyval(
        r[small] ** 2, density_small
    )
    velocity[small] = np.polynomial.polynomial.polyval(
        r[small] ** 2, velocity_small
    )

    # Increasing-power coefficients in x=1/r^2, obtained by expanding around
    # r=infinity through O(r^-14).
    density_large = np.array(
        [
            -488 / 5,
            96 / 5,
            -160 / 21,
            -1376 / 1155,
            -1952 / 5005,
            -2528 / 15015,
            -3104 / 36465,
            -2208 / 46189,
        ]
    )
    velocity_large = np.array(
        [
            -504 / 5,
            1248 / 35,
            -608 / 105,
            -160 / 231,
            -992 / 5005,
            -1184 / 15015,
            -1376 / 36465,
            -4704 / 230945,
        ]
    )
    density[large] = np.polynomial.polynomial.polyval(
        1 / r[large] ** 2, density_large
    )
    velocity[large] = np.polynomial.polynomial.polyval(
        1 / r[large] ** 2, velocity_large
    )

    if np.any(middle):
        values = r[middle]
        log_term = np.empty_like(values)
        at_one = np.isclose(values, 1.0, rtol=0.0, atol=1e-13)
        log_term[~at_one] = np.log(
            np.abs((1 + values[~at_one]) / (1 - values[~at_one]))
        )
        log_term[at_one] = 0.0
        density[middle] = (
            12 / values**2
            - 158
            + 100 * values**2
            - 42 * values**4
            + 3
            / values**3
            * (values**2 - 1) ** 3
            * (7 * values**2 + 2)
            * log_term
        )
        velocity[middle] = (
            12 / values**2
            - 82
            + 4 * values**2
            - 6 * values**4
            + 3
            / values**3
            * (values**2 - 1) ** 3
            * (values**2 + 2)
            * log_term
        )
        middle_indices = np.flatnonzero(middle)
        density[middle_indices[at_one]] = -88.0
        velocity[middle_indices[at_one]] = -72.0
    return density, velocity
