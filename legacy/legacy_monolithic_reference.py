"""
Ly-alpha P1D effective-model analysis with an in-house direct one-loop SPT
implementation.

Supported theory modes:

    mode="linear"
    mode="one_loop"

For ``one_loop`` the three real-space channels P_dd, P_dtheta and P_thetatheta
are evaluated directly from the standard EdS SPT kernels.  No FOLPS,
velocileptors, FFTLog reconstruction, taper, or replacement by the linear
spectrum is used.  P22 is integrated in the stable (q,p=|k-q|) variables and
P13 is obtained from fully symmetrized F3/G3 recursions.  The direct spectra
are cached on disk because their construction for all redshifts is expensive.

The P1D integrals are evaluated from cumulative trapezoidal integrals on one
fixed k grid.  Cutoff tests therefore change only the upper P1D integration
limit and remain additive.

The nuisance vector is always

    theta = [log_alpha_F, beta_F, alpha_bias, beta_bias, alpha_ct, beta_ct]

and every fit includes the redshift-dependent additive counterterm.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid, simpson
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.linalg import inv
from scipy.optimize import differential_evolution, minimize
from scipy.stats import chi2 as chi2_dist


# ============================================================================
# Configuration
# ============================================================================


@dataclass(frozen=True)
class DataConfig:
    data_dir: str | Path = "."

    pk_filename: str = "Pk1D_data.dat"
    cor_filename: str = "Pk1D_cor.dat"
    syst_filename: str = "Pk1D_syst.dat"
    icov_filename: str = "pk_1d_DR12_13bins_invCov.out"

    covariance_mode: str = "from_release_files"
    covariance_syst_mode: str = "paper_diag"

    z_min: float = 3.0
    z_max: float = 4.2
    delta_divide_by_pi: bool = True
    fallback_dir: str | Path = "/mnt/data"


@dataclass(frozen=True)
class CosmoConfig:
    omega_b: float = 0.02237
    omega_cdm: float = 0.1200

    theta_s_100: Optional[float] = 1.04110
    H0: Optional[float] = None

    ln10_10_As: float = 3.044
    n_s: float = 0.9649
    tau_reio: float = 0.0544

    N_ur: float = 2.0328
    N_ncdm: int = 1
    m_ncdm: Any = 0.06
    deg_ncdm: Any = 1.0
    T_ncdm: Any = 0.71611

    # Baseline CLASS limit. build_class() raises it automatically when the
    # auxiliary velocileptors input grid requires a larger value.
    P_k_max_hmpc: float = 50.0
    extra_class_params: Optional[Dict[str, Any]] = None

    def to_class_params(
        self,
        z_max_pk: float,
        required_pk_max_hmpc: Optional[float] = None,
    ) -> dict:
        pk_max_hmpc = float(self.P_k_max_hmpc)
        if required_pk_max_hmpc is not None:
            # Small safety margin avoids evaluating CLASS exactly at its boundary.
            pk_max_hmpc = max(pk_max_hmpc, 1.05 * float(required_pk_max_hmpc))

        params = {
            "omega_b": self.omega_b,
            "omega_cdm": self.omega_cdm,
            "ln10^{10}A_s": self.ln10_10_As,
            "n_s": self.n_s,
            "tau_reio": self.tau_reio,
            "N_ur": self.N_ur,
            "N_ncdm": self.N_ncdm,
            "output": "mPk",
            "P_k_max_h/Mpc": pk_max_hmpc,
            "z_max_pk": float(z_max_pk),
            "input_verbose": 0,
            "background_verbose": 0,
            "perturbations_verbose": 0,
        }

        if int(self.N_ncdm) > 0:
            params.update(
                {
                    "m_ncdm": self.m_ncdm,
                    "deg_ncdm": self.deg_ncdm,
                    "T_ncdm": self.T_ncdm,
                }
            )

        if self.H0 is not None:
            params["H0"] = self.H0
        elif self.theta_s_100 is not None:
            params["100*theta_s"] = self.theta_s_100
        else:
            raise ValueError("Provide either H0 or theta_s_100.")

        if self.extra_class_params:
            params.update(self.extra_class_params)
            if "H0" in self.extra_class_params:
                params.pop("100*theta_s", None)

        return params


@dataclass(frozen=True)
class EffectiveConfig:
    z_pivot: float = 3.0

    # Velocity-space thermal smoothing [s/km].
    k_s_fixed: float = 0.11

    # Three-dimensional pressure smoothing [h/Mpc].
    k_F_fixed: float = 18.0

    f_SiIII: float = 6.0e-3
    DeltaV_SiIII: float = 2.0 * np.pi / 0.0028  # km/s

    use_kF_filter: bool = True
    use_SiIII: bool = True
    use_thermal: bool = True


@dataclass(frozen=True)
class TheoryConfig:
    """Numerical settings for the linear or direct one-loop SPT model."""

    mode: str = "linear"  # "linear" or "one_loop"

    # Common final grid used by channel splines and P1D cumulative integrals.
    k_min: float = 1.0e-3
    k_max: float = 10.0
    n_k: int = 2000

    # Direct-SPT channels are computed on a smaller logarithmic grid and then
    # interpolated once onto the common n_k grid.
    direct_spt_n_k: int = 90
    direct_spt_q_min: float = 1.0e-5
    direct_spt_q_max: float = 20.0

    # P22 quadrature in (q,p=|k-q|).
    direct_spt_n_q_low: int = 500
    direct_spt_n_q_mid: int = 1000
    direct_spt_n_q_high: int = 500
    direct_spt_n_p: int = 200
    direct_spt_middle_width: float = 0.25

    # P13 quadrature from symmetrized F3/G3 recursions.
    direct_spt_p13_n_q: int = 180
    direct_spt_p13_n_mu: int = 40
    direct_spt_epsilon_relative: float = 1.0e-4

    # Linear CLASS input used inside loop integrals.
    direct_spt_input_n_k: int = 6000
    direct_spt_loop_source: str = "total"  # "total" or "cb"

    # Expensive direct channels are cached as compressed npz files.
    direct_spt_cache_dir: str | Path = "direct_spt_cache"
    direct_spt_use_cache: bool = True
    direct_spt_overwrite_cache: bool = False
    direct_spt_verbose: bool = True

    @property
    def linear_input_kmin(self) -> float:
        return min(self.direct_spt_q_min, self.k_min)

    @property
    def linear_input_kmax(self) -> float:
        # P22 needs P(|k-q|) up to approximately k_max + q_max.
        return self.k_max + self.direct_spt_q_max

    def validate(self) -> None:
        if self.mode not in {"linear", "one_loop"}:
            raise ValueError("TheoryConfig.mode must be 'linear' or 'one_loop'.")
        if not (0.0 < self.k_min < self.k_max):
            raise ValueError("Require 0 < k_min < k_max.")
        if self.n_k < 100:
            raise ValueError("n_k is too small for stable P1D integration.")
        if self.direct_spt_n_k < 20:
            raise ValueError("direct_spt_n_k must be at least 20.")
        if not (0.0 < self.direct_spt_q_min < self.direct_spt_q_max):
            raise ValueError("Require 0 < direct_spt_q_min < direct_spt_q_max.")
        if self.direct_spt_n_p < 20:
            raise ValueError("direct_spt_n_p must be at least 20.")
        if self.direct_spt_p13_n_q < 20 or self.direct_spt_p13_n_mu < 8:
            raise ValueError("Direct P13 quadrature is too coarse.")
        if not (0.0 < self.direct_spt_middle_width < 1.0):
            raise ValueError("direct_spt_middle_width must lie in (0,1).")
        if self.direct_spt_epsilon_relative <= 0.0:
            raise ValueError("direct_spt_epsilon_relative must be positive.")
        if self.direct_spt_loop_source not in {"total", "cb"}:
            raise ValueError("direct_spt_loop_source must be 'total' or 'cb'.")


# ============================================================================
# Small helpers
# ============================================================================


def make_spline(k: np.ndarray, values: np.ndarray) -> InterpolatedUnivariateSpline:
    """Cubic spline in log(k), without forcing the spectrum to be positive."""
    k = np.asarray(k, dtype=float)
    values = np.asarray(values, dtype=float)

    valid = np.isfinite(k) & np.isfinite(values) & (k > 0.0)
    if np.count_nonzero(valid) < 8:
        raise ValueError("Too few finite points to construct a spectrum spline.")

    return InterpolatedUnivariateSpline(np.log(k[valid]), values[valid], k=3)


def evaluate_spline(spline: InterpolatedUnivariateSpline, k: np.ndarray) -> np.ndarray:
    return np.asarray(spline(np.log(np.asarray(k, dtype=float))), dtype=float)



# ============================================================================
# Direct standard-PT kernels and loop integrals
# ============================================================================


def F2_EdS(q1, q2, mu12):
    q1 = np.asarray(q1, dtype=float)
    q2 = np.asarray(q2, dtype=float)
    mu12 = np.asarray(mu12, dtype=float)
    return 5.0 / 7.0 + 0.5 * mu12 * (q1 / q2 + q2 / q1) + 2.0 / 7.0 * mu12**2


def G2_EdS(q1, q2, mu12):
    q1 = np.asarray(q1, dtype=float)
    q2 = np.asarray(q2, dtype=float)
    mu12 = np.asarray(mu12, dtype=float)
    return 3.0 / 7.0 + 0.5 * mu12 * (q1 / q2 + q2 / q1) + 4.0 / 7.0 * mu12**2


def _norm2(v):
    v = np.asarray(v, dtype=float)
    return float(np.dot(v, v))


def _alpha_kernel(k1, k2, zero_tol=1.0e-28):
    k1 = np.asarray(k1, dtype=float)
    k2 = np.asarray(k2, dtype=float)
    k1_sq = _norm2(k1)
    if k1_sq <= zero_tol:
        raise FloatingPointError("alpha received a vanishing first momentum.")
    return float(np.dot(k1 + k2, k1) / k1_sq)


def _beta_kernel(k1, k2, zero_tol=1.0e-28):
    k1 = np.asarray(k1, dtype=float)
    k2 = np.asarray(k2, dtype=float)
    k1_sq = _norm2(k1)
    k2_sq = _norm2(k2)
    if k1_sq <= zero_tol or k2_sq <= zero_tol:
        raise FloatingPointError("beta received a vanishing momentum.")
    return float(_norm2(k1 + k2) * np.dot(k1, k2) / (2.0 * k1_sq * k2_sq))


def _vector_sum(vectors):
    return np.sum(np.asarray(vectors, dtype=float), axis=0)


def _FG_ordered(vectors):
    vectors = tuple(np.asarray(v, dtype=float) for v in vectors)
    n = len(vectors)
    if n == 1:
        return 1.0, 1.0
    denominator = (2.0 * n + 3.0) * (n - 1.0)
    F_total = 0.0
    G_total = 0.0
    for m in range(1, n):
        left, right = vectors[:m], vectors[m:]
        F_left, G_left = _FG_ordered(left)
        F_right, G_right = _FG_ordered(right)
        k_left, k_right = _vector_sum(left), _vector_sum(right)
        alpha = _alpha_kernel(k_left, k_right)
        beta = _beta_kernel(k_left, k_right)
        F_total += G_left * ((2.0*n+1.0)*alpha*F_right + 2.0*beta*G_right) / denominator
        G_total += G_left * (3.0*alpha*F_right + 2.0*n*beta*G_right) / denominator
    return float(F_total), float(G_total)


def _FG_symmetrized(vectors):
    import itertools
    vectors = tuple(np.asarray(v, dtype=float) for v in vectors)
    values = []
    for permutation in itertools.permutations(range(len(vectors))):
        values.append(_FG_ordered(tuple(vectors[i] for i in permutation)))
    values = np.asarray(values, dtype=float)
    return float(np.mean(values[:, 0])), float(np.mean(values[:, 1]))


def _regulated_k_q_minus_q(k, q, mu, epsilon_relative):
    sin_angle = np.sqrt(max(0.0, 1.0 - float(mu)**2))
    k_vector = np.array([0.0, 0.0, float(k)])
    q_vector = float(q) * np.array([sin_angle, 0.0, float(mu)])
    eps = epsilon_relative * max(float(k), float(q))
    epsilon_vector = np.array([0.0, eps, 0.0])
    return k_vector - epsilon_vector, q_vector, -q_vector + epsilon_vector


def P22_channels_direct_qp(
    k, P_lin, q_min, q_max, n_q_low, n_q_mid, n_q_high, n_p,
    middle_width=0.25, p_min=1.0e-5,
):
    from numpy.polynomial.legendre import leggauss
    k = float(k)
    x_p, w_p = leggauss(int(n_p))

    def inner(q):
        q = np.asarray(q, dtype=float)
        p_lower, p_upper = np.abs(k-q), k+q
        p_mid, p_half = 0.5*(p_upper+p_lower), 0.5*(p_upper-p_lower)
        p = p_mid[:, None] + p_half[:, None]*x_p[None, :]
        p_eval = np.maximum(p, p_min)
        qq = q[:, None]
        mu_qp = (k*k - qq*qq - p*p) / (2.0*qq*np.maximum(p, 1.0e-30))
        mu_qp = np.clip(mu_qp, -1.0, 1.0)
        F2 = F2_EdS(qq, p_eval, mu_qp)
        G2 = G2_EdS(qq, p_eval, mu_qp)
        common = p * P_lin(p_eval)
        return tuple(
            p_half*np.sum(common*kernel*w_p[None, :], axis=1)
            for kernel in (F2**2, F2*G2, G2**2)
        )

    q_left = max(q_min, k*(1.0-middle_width))
    q_right = min(q_max, k*(1.0+middle_width))
    totals = np.zeros(3)

    def integrate_region(q):
        vals = inner(q)
        common_q = q*P_lin(q)
        return np.array([simpson(common_q*v, x=q) for v in vals])

    if q_left > q_min:
        totals += integrate_region(np.geomspace(q_min, q_left, int(n_q_low)))
    if q_right > q_left:
        totals += integrate_region(np.linspace(q_left, q_right, int(n_q_mid)))
    if q_max > q_right:
        totals += integrate_region(np.geomspace(q_right, q_max, int(n_q_high)))
    return tuple(totals/(2.0*np.pi**2*k))


def P13_channels_from_F3G3(
    k, P_lin, q_min, q_max, n_q, n_mu, epsilon_relative=1.0e-4,
):
    from numpy.polynomial.legendre import leggauss
    k = float(k)
    xq = np.linspace(np.log(q_min), np.log(q_max), int(n_q))
    q_values = np.exp(xq)
    mu_nodes, mu_weights = leggauss(int(n_mu))
    angular_F3 = np.zeros_like(q_values)
    angular_G3 = np.zeros_like(q_values)
    for iq, q in enumerate(q_values):
        fvals = np.empty(len(mu_nodes))
        gvals = np.empty(len(mu_nodes))
        for im, mu in enumerate(mu_nodes):
            fvals[im], gvals[im] = _FG_symmetrized(
                _regulated_k_q_minus_q(k, q, mu, epsilon_relative)
            )
        angular_F3[iq] = np.dot(mu_weights, fvals)
        angular_G3[iq] = np.dot(mu_weights, gvals)
    radial = q_values**3 * P_lin(q_values)
    int_F3 = simpson(radial*angular_F3, x=xq)
    int_G3 = simpson(radial*angular_G3, x=xq)
    Pk = float(np.asarray(P_lin(k)))
    p13_dd = 3.0*Pk*int_F3/(2.0*np.pi**2)
    p13_dt = 3.0*Pk*(int_F3+int_G3)/(4.0*np.pi**2)
    p13_tt = 3.0*Pk*int_G3/(2.0*np.pi**2)
    return p13_dd, p13_dt, p13_tt


# ============================================================================
# Main analysis
# ============================================================================


class LyAlphaP1DAnalysis:
    def __init__(
        self,
        data_config: DataConfig = DataConfig(),
        cosmo_config: CosmoConfig = CosmoConfig(),
        theory_config: TheoryConfig = TheoryConfig(),
        effective_config: EffectiveConfig = EffectiveConfig(),
    ):
        theory_config.validate()

        self.data_config = data_config
        self.cosmo_config = cosmo_config
        self.theory_config = theory_config
        self.effective_config = effective_config

        self.data_ready = False
        self.cosmo_ready = False
        self.theory_ready = False

    @classmethod
    def paper_default(
        cls,
        data_dir: str | Path = ".",
        theory_mode: str = "linear",
        **theory_kwargs,
    ) -> "LyAlphaP1DAnalysis":
        return cls(
            data_config=DataConfig(data_dir=data_dir),
            theory_config=TheoryConfig(mode=theory_mode, **theory_kwargs),
        )

    def with_theory(self, **kwargs) -> "LyAlphaP1DAnalysis":
        return LyAlphaP1DAnalysis(
            data_config=self.data_config,
            cosmo_config=self.cosmo_config,
            theory_config=replace(self.theory_config, **kwargs),
            effective_config=self.effective_config,
        )

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def _resolve_path(self, filename: str) -> Path:
        primary = Path(self.data_config.data_dir) / filename
        if primary.exists():
            return primary

        fallback = Path(self.data_config.fallback_dir) / filename
        if fallback.exists():
            return fallback

        raise FileNotFoundError(
            f"Could not find {filename!r} in {primary.parent} or {fallback.parent}."
        )

    def load_data(self) -> "LyAlphaP1DAnalysis":
        if self.data_config.covariance_mode == "from_release_files":
            self._load_data_from_release_files()
        elif self.data_config.covariance_mode == "from_inverse_covariance":
            self._load_data_from_inverse_covariance()
        else:
            raise ValueError(
                "covariance_mode must be 'from_release_files' or "
                "'from_inverse_covariance'."
            )

        self.data_ready = True
        return self

    def _load_data_from_release_files(self) -> None:
        pk_path = self._resolve_path(self.data_config.pk_filename)
        cor_path = self._resolve_path(self.data_config.cor_filename)
        syst_path = self._resolve_path(self.data_config.syst_filename)

        raw = np.loadtxt(pk_path)
        cor_raw = np.loadtxt(cor_path)
        syst = np.loadtxt(syst_path)

        if raw.ndim != 2 or raw.shape[1] < 4:
            raise ValueError("P1D data file must contain at least z, k, P1D, stat.")
        if syst.shape[0] != raw.shape[0]:
            raise ValueError("Systematic-error and P1D files have different lengths.")

        z_all, k_all, p1d_all, stat_all = raw[:, :4].T
        syst_quad_all = np.sqrt(np.sum(syst**2, axis=1))
        sigma_total_all = np.sqrt(stat_all**2 + syst_quad_all**2)

        z_unique_all = np.sort(np.unique(z_all))
        counts = [np.count_nonzero(np.isclose(z_all, z)) for z in z_unique_all]
        if len(set(counts)) != 1:
            raise ValueError(f"Different k-bin counts in redshift blocks: {counts}")

        n_z = len(z_unique_all)
        n_k = counts[0]
        if cor_raw.shape != (n_z * n_k, n_k):
            raise ValueError(
                f"Unexpected correlation matrix shape {cor_raw.shape}; "
                f"expected {(n_z * n_k, n_k)}."
            )

        covariance = np.zeros((len(raw), len(raw)), dtype=float)
        mode = self.data_config.covariance_syst_mode

        for iz in range(n_z):
            block = slice(iz * n_k, (iz + 1) * n_k)
            corr = 0.5 * (cor_raw[block] + cor_raw[block].T)
            stat = stat_all[block]
            syst_block = syst[block]
            syst_var = np.sum(syst_block**2, axis=1)

            if mode == "paper_diag":
                cov_block = np.diag(stat**2 + syst_var)
            elif mode == "diag":
                cov_block = corr * np.outer(stat, stat) + np.diag(syst_var)
            elif mode == "total_corr":
                sigma = sigma_total_all[block]
                cov_block = corr * np.outer(sigma, sigma)
            elif mode == "outer":
                cov_block = corr * np.outer(stat, stat)
                for column in syst_block.T:
                    cov_block += np.outer(column, column)
            elif mode == "stat_only":
                cov_block = corr * np.outer(stat, stat)
            else:
                raise ValueError(
                    "covariance_syst_mode must be paper_diag, diag, total_corr, "
                    "outer, or stat_only."
                )

            covariance[block, block] = cov_block

        fit_mask = (
            (z_all >= self.data_config.z_min - 1.0e-9)
            & (z_all <= self.data_config.z_max + 1.0e-9)
        )

        self._store_selected_data(
            z_all=z_all,
            k_all=k_all,
            p1d_all=p1d_all,
            sigma_all=sigma_total_all,
            stat_all=stat_all,
            syst_all=syst_quad_all,
            fit_mask=fit_mask,
        )

        self.C_fit = covariance[np.ix_(fit_mask, fit_mask)]
        self.C_fit = 0.5 * (self.C_fit + self.C_fit.T)
        jitter = 1.0e-12 * np.nanmedian(np.diag(self.C_fit))
        self.C_fit += jitter * np.eye(len(self.C_fit))
        self.icov = inv(self.C_fit)

        self.data_paths = {"pk": pk_path, "cor": cor_path, "syst": syst_path}

    def _load_data_from_inverse_covariance(self) -> None:
        pk_path = self._resolve_path(self.data_config.pk_filename)
        icov_path = self._resolve_path(self.data_config.icov_filename)

        raw = np.loadtxt(pk_path)
        icov_full = np.loadtxt(icov_path)

        z_all, k_all, p1d_all, sigma_all = raw[:, :4].T
        if icov_full.shape != (len(raw), len(raw)):
            raise ValueError("Inverse covariance has an incompatible shape.")

        fit_mask = (
            (z_all >= self.data_config.z_min - 1.0e-9)
            & (z_all <= self.data_config.z_max + 1.0e-9)
        )

        self._store_selected_data(
            z_all=z_all,
            k_all=k_all,
            p1d_all=p1d_all,
            sigma_all=sigma_all,
            stat_all=sigma_all,
            syst_all=np.zeros_like(sigma_all),
            fit_mask=fit_mask,
        )

        if np.all(fit_mask):
            self.icov = icov_full
            self.C_fit = inv(icov_full)
        else:
            covariance_full = inv(icov_full)
            self.C_fit = covariance_full[np.ix_(fit_mask, fit_mask)]
            self.C_fit = 0.5 * (self.C_fit + self.C_fit.T)
            self.icov = inv(self.C_fit)

        self.data_paths = {"pk": pk_path, "icov": icov_path}

    def _store_selected_data(
        self,
        z_all: np.ndarray,
        k_all: np.ndarray,
        p1d_all: np.ndarray,
        sigma_all: np.ndarray,
        stat_all: np.ndarray,
        syst_all: np.ndarray,
        fit_mask: np.ndarray,
    ) -> None:
        self.z_data = z_all[fit_mask]
        self.k_data = k_all[fit_mask]
        self.P1D_data = p1d_all[fit_mask]
        self.sigma_P1D_plot = sigma_all[fit_mask]
        self.stat_error = stat_all[fit_mask]
        self.syst_error = syst_all[fit_mask]
        self.z_unique = np.sort(np.unique(self.z_data))

        factor = np.pi if self.data_config.delta_divide_by_pi else 1.0
        self.delta_data = self.k_data * self.P1D_data / factor
        self.sigma_delta = self.k_data * self.sigma_P1D_plot / factor

    # ------------------------------------------------------------------
    # CLASS and common k grid
    # ------------------------------------------------------------------

    def build_class(self) -> "LyAlphaP1DAnalysis":
        if not self.data_ready:
            self.load_data()

        from classy import Class

        params = self.cosmo_config.to_class_params(
            z_max_pk=float(np.max(self.z_unique) + 0.2),
            required_pk_max_hmpc=self.theory_config.linear_input_kmax,
        )
        self.class_params = dict(params)
        self.class_pk_max_hmpc = float(params["P_k_max_h/Mpc"])

        self.cosmo = Class()
        self.cosmo.set(params)
        self.cosmo.compute()

        self.h = self.cosmo.h()
        self.Omega_m = self.cosmo.Omega_m()
        self.cosmo_ready = True
        return self

    def H_z_km_s_Mpc(self, z: float) -> float:
        return self.cosmo.Hubble(float(z)) * 299792.458

    def Aconv_velocity_to_hmpc(self, z: float) -> float:
        return self.H_z_km_s_Mpc(z) / ((1.0 + z) * self.h)

    def get_pk_lin_h_units(self, k_hmpc: np.ndarray, z: float) -> np.ndarray:
        k_hmpc = np.asarray(k_hmpc, dtype=float)
        k_mpc = k_hmpc * self.h
        return np.array(
            [self.cosmo.pk_lin(float(k), float(z)) for k in k_mpc]
        ) * self.h**3

    def get_pk_cb_lin_h_units(self, k_hmpc: np.ndarray, z: float) -> np.ndarray:
        """Linear cb spectrum in (Mpc/h)^3, when exposed by CLASS."""
        if not hasattr(self.cosmo, "pk_cb_lin"):
            raise AttributeError(
                "This CLASS build does not expose pk_cb_lin; use "
                "direct_spt_loop_source='total'."
            )
        k_hmpc = np.asarray(k_hmpc, dtype=float)
        k_mpc = k_hmpc * self.h
        return np.array(
            [self.cosmo.pk_cb_lin(float(k), float(z)) for k in k_mpc]
        ) * self.h**3

    def prepare(self) -> "LyAlphaP1DAnalysis":
        self.load_data()
        self.build_class()

        theory = self.theory_config
        self.k_grid = np.geomspace(theory.k_min, theory.k_max, theory.n_k)

        self._validate_data_scales()
        self._build_linear_spectra()

        if theory.mode == "one_loop":
            self._build_one_loop_spectra()
        else:
            self.channel_splines = {
                z: {
                    "Pdd": self.linear_spectra[z]["spline"],
                    "Pdt": self.linear_spectra[z]["spline"],
                    "Ptt": self.linear_spectra[z]["spline"],
                }
                for z in self.z_unique
            }

        self._build_integral_cache()
        self.theory_ready = True
        return self

    def _validate_data_scales(self) -> None:
        required = []
        for z in self.z_unique:
            mask = np.isclose(self.z_data, z)
            required.extend(self.Aconv_velocity_to_hmpc(z) * self.k_data[mask])

        required = np.asarray(required)
        self.k_parallel_min = float(np.min(required))
        self.k_parallel_max = float(np.max(required))

        if self.k_parallel_max >= self.theory_config.k_max:
            raise ValueError(
                f"Largest data k_parallel={self.k_parallel_max:.4g} h/Mpc "
                f"reaches/exceeds k_max={self.theory_config.k_max:.4g} h/Mpc."
            )

    def _build_linear_spectra(self) -> None:
        self.linear_spectra = {}
        for z in self.z_unique:
            power = self.get_pk_lin_h_units(self.k_grid, z)
            self.linear_spectra[z] = {
                "k": self.k_grid,
                "P_lin": power,
                "spline": make_spline(self.k_grid, power),
            }

    def growth_rate_f(self, z: float) -> float:
        z = float(z)
        if hasattr(self.cosmo, "scale_independent_growth_factor_f"):
            return float(self.cosmo.scale_independent_growth_factor_f(z))

        if not hasattr(self.cosmo, "scale_independent_growth_factor"):
            raise AttributeError("CLASS does not expose a growth factor.")

        eps = 1.0e-3
        a = 1.0 / (1.0 + z)
        z_lo = 1.0 / (a * np.exp(-eps)) - 1.0
        z_hi = 1.0 / (a * np.exp(+eps)) - 1.0
        D_lo = float(self.cosmo.scale_independent_growth_factor(z_lo))
        D_hi = float(self.cosmo.scale_independent_growth_factor(z_hi))
        return (np.log(D_hi) - np.log(D_lo)) / (2.0 * eps)

    # ------------------------------------------------------------------
    # Velocileptors one-loop spectra on the complete integration grid
    # ------------------------------------------------------------------

    def _direct_spt_cache_path(self, z: float) -> Path:
        t = self.theory_config
        cache_dir = Path(t.direct_spt_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        source = t.direct_spt_loop_source
        name = (
            f"direct_spt_z{float(z):.3f}_{source}_"
            f"k{t.k_min:.3g}-{t.k_max:.3g}_nk{t.direct_spt_n_k}_"
            f"q{t.direct_spt_q_max:.3g}_"
            f"p{t.direct_spt_n_p}_r{t.direct_spt_p13_n_q}_"
            f"mu{t.direct_spt_p13_n_mu}.npz"
        )
        return cache_dir / name

    def _build_loop_linear_callable(self, z: float):
        t = self.theory_config
        k_input = np.geomspace(
            t.linear_input_kmin,
            t.linear_input_kmax,
            t.direct_spt_input_n_k,
        )
        if t.direct_spt_loop_source == "cb":
            p_input = self.get_pk_cb_lin_h_units(k_input, z)
        else:
            p_input = self.get_pk_lin_h_units(k_input, z)
        if np.any(~np.isfinite(p_input)) or np.any(p_input <= 0.0):
            raise RuntimeError(f"Invalid loop linear spectrum at z={z}.")
        spline = InterpolatedUnivariateSpline(
            np.log(k_input), np.log(p_input), k=3
        )
        def P_loop_linear(k):
            k = np.asarray(k, dtype=float)

            k_lo = float(k_input[0])
            k_hi = float(k_input[-1])

            rtol = 1.0e-12
            atol = 1.0e-14

            below = k < (
                k_lo - atol - rtol * abs(k_lo)
            )

            above = k > (
                k_hi + atol + rtol * abs(k_hi)
            )

            if np.any(below) or np.any(above):
                requested_min = float(
                    np.min(k)
                )

                requested_max = float(
                    np.max(k)
                )

                raise ValueError(
                    "Loop requested k outside "
                    f"[{k_lo}, {k_hi}] h/Mpc: "
                    f"requested range "
                    f"[{requested_min}, {requested_max}] h/Mpc."
                )

            k_eval = np.clip(
                k,
                k_lo,
                k_hi,
            )

            return np.exp(
                spline(np.log(k_eval))
            )

        # def P_loop_linear(k):
        #     k = np.asarray(k, dtype=float)
        #     if np.any(k < k_input[0]) or np.any(k > k_input[-1]):
        #         raise ValueError(
        #             f"Loop requested k outside [{k_input[0]}, {k_input[-1]}] h/Mpc."
        #         )
        #     return np.exp(spline(np.log(k)))

        return P_loop_linear

    def _build_one_loop_spectra(self) -> None:
        """Build direct Pdd/Pdt/Ptt channels for every fitted redshift."""
        t = self.theory_config
        self.one_loop_spectra = {}
        self.channel_splines = {}

        for z_raw in self.z_unique:
            z = float(z_raw)
            path = self._direct_spt_cache_path(z)
            loaded = False

            if t.direct_spt_use_cache and path.exists() and not t.direct_spt_overwrite_cache:
                data = np.load(path)
                k_spt = np.asarray(data["k"], dtype=float)
                pdd_spt = np.asarray(data["Pdd"], dtype=float)
                pdt_spt = np.asarray(data["Pdt"], dtype=float)
                ptt_spt = np.asarray(data["Ptt"], dtype=float)
                loaded = True
                if t.direct_spt_verbose:
                    print(f"Loaded direct SPT cache: {path}")

            if not loaded:
                P_loop_linear = self._build_loop_linear_callable(z)
                k_spt = np.geomspace(t.k_min, t.k_max, t.direct_spt_n_k)
                P22 = np.empty((len(k_spt), 3), dtype=float)
                P13 = np.empty((len(k_spt), 3), dtype=float)

                for i, k in enumerate(k_spt):
                    if t.direct_spt_verbose:
                        print(
                            f"z={z:.1f} direct SPT {i+1:3d}/{len(k_spt)} "
                            f"k={k:.6g}", end="\r", flush=True
                        )
                    P22[i] = P22_channels_direct_qp(
                        k, P_loop_linear,
                        q_min=t.direct_spt_q_min,
                        q_max=t.direct_spt_q_max,
                        n_q_low=t.direct_spt_n_q_low,
                        n_q_mid=t.direct_spt_n_q_mid,
                        n_q_high=t.direct_spt_n_q_high,
                        n_p=t.direct_spt_n_p,
                        middle_width=t.direct_spt_middle_width,
                    )
                    P13[i] = P13_channels_from_F3G3(
                        k, P_loop_linear,
                        q_min=t.direct_spt_q_min,
                        q_max=t.direct_spt_q_max,
                        n_q=t.direct_spt_p13_n_q,
                        n_mu=t.direct_spt_p13_n_mu,
                        epsilon_relative=t.direct_spt_epsilon_relative,
                    )
                if t.direct_spt_verbose:
                    print()

                # The linear tree-level term is always the total matter spectrum.
                p_tree = self.get_pk_lin_h_units(k_spt, z)
                pdd_spt = p_tree + P22[:, 0] + P13[:, 0]
                pdt_spt = p_tree + P22[:, 1] + P13[:, 1]
                ptt_spt = p_tree + P22[:, 2] + P13[:, 2]

                if t.direct_spt_use_cache:
                    np.savez_compressed(
                        path, z=z, k=k_spt, Ptree=p_tree,
                        P22=P22, P13=P13,
                        Pdd=pdd_spt, Pdt=pdt_spt, Ptt=ptt_spt,
                    )
                    if t.direct_spt_verbose:
                        print(f"Saved direct SPT cache: {path}")

            if not all(np.all(np.isfinite(x)) for x in (pdd_spt, pdt_spt, ptt_spt)):
                raise RuntimeError(f"Non-finite direct SPT spectrum at z={z}.")

            # Interpolate once from the expensive SPT grid to the common grid.
            pdd_common = evaluate_spline(make_spline(k_spt, pdd_spt), self.k_grid)
            pdt_common = evaluate_spline(make_spline(k_spt, pdt_spt), self.k_grid)
            ptt_common = evaluate_spline(make_spline(k_spt, ptt_spt), self.k_grid)

            self.one_loop_spectra[z] = {
                "k": self.k_grid,
                "Pdd": pdd_common,
                "Pdt": pdt_common,
                "Ptt": ptt_common,
                "direct_k": k_spt,
                "cache_path": str(path),
                "diagnostics": {"source": "direct_spt", "loaded_from_cache": loaded},
            }
            self.channel_splines[z] = {
                "Pdd": make_spline(self.k_grid, pdd_common),
                "Pdt": make_spline(self.k_grid, pdt_common),
                "Ptt": make_spline(self.k_grid, ptt_common),
            }

    def theory_grid_summary(self) -> dict:
        t = self.theory_config
        return {
            "mode": t.mode,
            "integration_k_min_hmpc": t.k_min,
            "integration_k_max_hmpc": t.k_max,
            "integration_n_k": t.n_k,
            "direct_spt_n_k": t.direct_spt_n_k,
            "direct_spt_q_min_hmpc": t.direct_spt_q_min,
            "direct_spt_q_max_hmpc": t.direct_spt_q_max,
            "direct_spt_loop_source": t.direct_spt_loop_source,
            "class_P_k_max_hmpc": getattr(self, "class_pk_max_hmpc", None),
            "k_parallel_data_min_hmpc": getattr(self, "k_parallel_min", None),
            "k_parallel_data_max_hmpc": getattr(self, "k_parallel_max", None),
        }

    def direct_spt_diagnostics(self, z: Optional[float] = None) -> dict:
        if self.theory_config.mode != "one_loop" or not hasattr(self, "one_loop_spectra"):
            raise RuntimeError("Prepare an analysis with mode='one_loop' first.")
        if z is None:
            return {float(zz): entry["diagnostics"] for zz, entry in self.one_loop_spectra.items()}
        z_match = self.z_unique[np.argmin(np.abs(self.z_unique - float(z)))]
        return self.one_loop_spectra[float(z_match)]["diagnostics"]

    # ------------------------------------------------------------------
    # Effective model
    # ------------------------------------------------------------------

    def Fbar_of_z(self, z: np.ndarray | float) -> np.ndarray:
        return np.exp(-0.0025 * (1.0 + np.asarray(z)) ** 3.7)

    def k_SiIII_factor(self, kv: np.ndarray, z: float) -> np.ndarray:
        if not self.effective_config.use_SiIII:
            return np.ones_like(np.asarray(kv), dtype=float)

        r = self.effective_config.f_SiIII / (1.0 - self.Fbar_of_z(z))
        return (
            1.0
            + 2.0
            * r
            * np.cos(self.effective_config.DeltaV_SiIII * np.asarray(kv))
            + r**2
        )

    def W_F_filter(self, k_hmpc: np.ndarray) -> np.ndarray:
        if not self.effective_config.use_kF_filter:
            return np.ones_like(np.asarray(k_hmpc), dtype=float)
        return np.exp(-(np.asarray(k_hmpc) / self.effective_config.k_F_fixed) ** 2)

    def thermal_factor(self, kv: np.ndarray) -> np.ndarray:
        if not self.effective_config.use_thermal:
            return np.ones_like(np.asarray(kv), dtype=float)
        return np.exp(-(np.asarray(kv) / self.effective_config.k_s_fixed) ** 2)

    @staticmethod
    def unpack_theta(theta: np.ndarray) -> Tuple[float, ...]:
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (6,):
            raise ValueError("theta must contain six nuisance parameters.")
        log_alpha_F, beta_F, alpha_bias, beta_bias, alpha_ct, beta_ct = theta
        return (
            float(np.exp(log_alpha_F)),
            float(beta_F),
            float(alpha_bias),
            float(beta_bias),
            float(alpha_ct),
            float(beta_ct),
        )

    def A_of_z(self, z: float, alpha_F: float, beta_F: float) -> float:
        zp = self.effective_config.z_pivot
        return float(alpha_F * ((1.0 + z) / (1.0 + zp)) ** beta_F)

    def beta_of_z(self, z: float, alpha_bias: float, beta_bias: float) -> float:
        zp = self.effective_config.z_pivot
        return float(alpha_bias * ((1.0 + z) / (1.0 + zp)) ** beta_bias)

    def Ict_of_z(self, z: float, alpha_ct: float, beta_ct: float) -> float:
        zp = self.effective_config.z_pivot
        return float(
            self.I0_scale * alpha_ct * ((1.0 + zp) / (1.0 + z)) ** beta_ct
        )

    # ------------------------------------------------------------------
    # P1D integrals
    # ------------------------------------------------------------------

    def _build_cumulative_integral_splines(
        self,
        z: float,
        pdd_spline: InterpolatedUnivariateSpline,
        pdt_spline: InterpolatedUnivariateSpline,
        ptt_spline: InterpolatedUnivariateSpline,
        k_cut: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Build cumulative integrals on one fixed k grid.

        For a chosen upper integration limit k_cut, define

            F0(k) = integral_k^k_cut dq q W_F(q) Pdd(q)
            F2(k) = integral_k^k_cut dq W_F(q) Pdt(q) / q
            F4(k) = integral_k^k_cut dq W_F(q) Ptt(q) / q^3

        Then

            I0 = Aconv * F0(k_parallel)
            I2 = Aconv * k_parallel^2 * F2(k_parallel)
            I4 = Aconv * k_parallel^4 * F4(k_parallel)

        All observed k_parallel values are evaluated from the same cumulative
        integrals. This avoids non-additive Simpson-grid effects.
        """
        del z

        if k_cut is None:
            k_cut = self.theory_config.k_max

        k_cut = float(k_cut)
        k_min = float(self.theory_config.k_min)
        k_max = float(self.theory_config.k_max)

        if not (k_min < k_cut <= k_max):
            raise ValueError(
                f"k_cut must satisfy {k_min} < k_cut <= {k_max}; "
                f"received {k_cut}."
            )

        interior = self.k_grid[
            (self.k_grid >= k_min)
            & (self.k_grid < k_cut)
        ]
        kk = np.unique(
            np.concatenate(
                [np.asarray(interior, dtype=float), [k_cut]]
            )
        )

        if len(kk) < 8:
            raise RuntimeError(
                f"Only {len(kk)} grid points are available below "
                f"k_cut={k_cut}. Increase TheoryConfig.n_k."
            )

        pdd = evaluate_spline(pdd_spline, kk)
        pdt = evaluate_spline(pdt_spline, kk)
        ptt = evaluate_spline(ptt_spline, kk)
        pressure = self.W_F_filter(kk)

        integrand_0 = kk * pressure * pdd
        integrand_2 = pressure * pdt / kk
        integrand_4 = pressure * ptt / kk**3

        def reverse_cumulative(integrand: np.ndarray) -> np.ndarray:
            cumulative_reversed = cumulative_trapezoid(
                integrand[::-1],
                kk[::-1],
                initial=0.0,
            )
            return -np.asarray(cumulative_reversed[::-1], dtype=float)

        F0 = reverse_cumulative(integrand_0)
        F2 = reverse_cumulative(integrand_2)
        F4 = reverse_cumulative(integrand_4)

        return {
            "k": kk,
            "k_cut": k_cut,
            "F0_values": F0,
            "F2_values": F2,
            "F4_values": F4,
            "F0": InterpolatedUnivariateSpline(np.log(kk), F0, k=3),
            "F2": InterpolatedUnivariateSpline(np.log(kk), F2, k=3),
            "F4": InterpolatedUnivariateSpline(np.log(kk), F4, k=3),
        }

    def _evaluate_cumulative_I024(
        self,
        kv_values: np.ndarray,
        z: float,
        cumulative: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate I0, I2 and I4 from cumulative-integral splines."""
        kv_values = np.asarray(kv_values, dtype=float)
        conversion = self.Aconv_velocity_to_hmpc(z)
        k_parallel = conversion * kv_values

        k_min = float(cumulative["k"][0])
        k_cut = float(cumulative["k_cut"])

        if np.any(k_parallel < k_min):
            raise ValueError(
                "Some k_parallel values lie below the cumulative grid "
                f"minimum {k_min:.6g} h/Mpc."
            )

        if np.any(k_parallel >= k_cut):
            raise ValueError(
                "Some k_parallel values reach or exceed the integration "
                f"cutoff {k_cut:.6g} h/Mpc."
            )

        log_k_parallel = np.log(k_parallel)

        F0 = np.asarray(cumulative["F0"](log_k_parallel), dtype=float)
        F2 = np.asarray(cumulative["F2"](log_k_parallel), dtype=float)
        F4 = np.asarray(cumulative["F4"](log_k_parallel), dtype=float)

        I0 = conversion * F0
        I2 = conversion * k_parallel**2 * F2
        I4 = conversion * k_parallel**4 * F4

        return I0, I2, I4

    def _compute_I024_for_kv_array(
        self,
        kv_values: np.ndarray,
        z: float,
        pdd_spline: InterpolatedUnivariateSpline,
        pdt_spline: InterpolatedUnivariateSpline,
        ptt_spline: InterpolatedUnivariateSpline,
        k_cut: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute I0, I2 and I4 on the fixed theory grid.

        Changing k_cut changes only the upper integration limit. CLASS,
        FFTLog, velocileptors and the channel splines are not regenerated.
        """
        cumulative = self._build_cumulative_integral_splines(
            z=z,
            pdd_spline=pdd_spline,
            pdt_spline=pdt_spline,
            ptt_spline=ptt_spline,
            k_cut=k_cut,
        )
        return self._evaluate_cumulative_I024(
            kv_values=kv_values,
            z=z,
            cumulative=cumulative,
        )

    def build_integral_cache_for_cutoff(
        self,
        k_cut: float,
    ) -> Dict[float, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Build a cutoff-specific I-cache while keeping the prepared theory fixed.

        The returned cache does not replace self.I_cache.
        """
        if not hasattr(self, "channel_splines"):
            raise RuntimeError(
                "Prepare the theory before building a cutoff-specific cache."
            )

        cache: Dict[
            float,
            Tuple[np.ndarray, np.ndarray, np.ndarray],
        ] = {}

        for z in self.z_unique:
            mask = np.isclose(self.z_data, z)
            channels = self.channel_splines[z]

            cumulative = self._build_cumulative_integral_splines(
                z=z,
                pdd_spline=channels["Pdd"],
                pdt_spline=channels["Pdt"],
                ptt_spline=channels["Ptt"],
                k_cut=k_cut,
            )

            cache[float(z)] = self._evaluate_cumulative_I024(
                kv_values=self.k_data[mask],
                z=z,
                cumulative=cumulative,
            )

        return cache

    def _build_integral_cache(self) -> None:
        self.I_cache = {}
        self.cumulative_integral_splines = {}

        for z in self.z_unique:
            mask = np.isclose(self.z_data, z)
            channels = self.channel_splines[z]

            cumulative = self._build_cumulative_integral_splines(
                z=z,
                pdd_spline=channels["Pdd"],
                pdt_spline=channels["Pdt"],
                ptt_spline=channels["Ptt"],
                k_cut=self.theory_config.k_max,
            )

            self.cumulative_integral_splines[float(z)] = cumulative
            self.I_cache[float(z)] = self._evaluate_cumulative_I024(
                kv_values=self.k_data[mask],
                z=z,
                cumulative=cumulative,
            )

        zp = self.effective_config.z_pivot
        z_scale = self.z_unique[np.argmin(np.abs(self.z_unique - zp))]
        self.I0_scale = float(
            np.median(self.I_cache[float(z_scale)][0])
        )

    def integration_tail_diagnostic(
        self,
        theta: Optional[np.ndarray] = None,
        tail_fraction: float = 0.2,
    ) -> pd.DataFrame:
        """Estimate the contribution from the final fraction of the k range.

        This is a numerical convergence diagnostic, not a physical k cut. It
        reports the absolute tail/full ratio separately for I0, I2 and I4 at
        the largest observed k_parallel in every redshift bin.
        """
        if not self.theory_ready:
            raise RuntimeError("Call prepare() first.")
        if not (0.0 < tail_fraction < 1.0):
            raise ValueError("tail_fraction must lie between zero and one.")

        tail_start = self.theory_config.k_max * (1.0 - tail_fraction)
        rows = []

        for z in self.z_unique:
            mask_z = np.isclose(self.z_data, z)
            kv = float(np.max(self.k_data[mask_z]))
            kpar = self.Aconv_velocity_to_hmpc(z) * kv
            mask = self.k_grid >= kpar
            kk = self.k_grid[mask]
            tail = kk >= max(tail_start, kpar)
            channels = self.channel_splines[z]

            if np.count_nonzero(tail) < 2:
                raise RuntimeError(
                    "Too few points in the integration tail. Increase n_k or "
                    "tail_fraction."
                )

            pdd = evaluate_spline(channels["Pdd"], kk)
            pdt = evaluate_spline(channels["Pdt"], kk)
            ptt = evaluate_spline(channels["Ptt"], kk)
            pressure = self.W_F_filter(kk)

            integrands = {
                "I0": kk * pressure * pdd,
                "I2": (kpar**2 / kk) * pressure * pdt,
                "I4": (kpar**4 / kk**3) * pressure * ptt,
            }

            row = {"z": float(z), "kv_max": kv, "k_parallel": float(kpar)}
            for name, integrand in integrands.items():
                full = simpson(integrand, x=kk)
                tail_value = simpson(integrand[tail], x=kk[tail])
                row[f"{name}_tail_fraction"] = (
                    abs(tail_value / full) if full != 0.0 else np.nan
                )
            rows.append(row)

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Model and chi2
    # ------------------------------------------------------------------

    def model_p1d(self, theta: np.ndarray) -> np.ndarray:
        if not self.theory_ready:
            raise RuntimeError("Call prepare() before model_p1d().")

        alpha_F, beta_F, alpha_bias, beta_bias, alpha_ct, beta_ct = (
            self.unpack_theta(theta)
        )
        output = np.empty_like(self.P1D_data, dtype=float)

        for z in self.z_unique:
            mask = np.isclose(self.z_data, z)
            kv = self.k_data[mask]
            I0, I2, I4 = self.I_cache[z]

            A = self.A_of_z(z, alpha_F, beta_F)
            beta = self.beta_of_z(z, alpha_bias, beta_bias)
            counterterm = self.Ict_of_z(z, alpha_ct, beta_ct)

            output[mask] = (
                A
                * self.k_SiIII_factor(kv, z)
                * self.thermal_factor(kv)
                * (I0 + counterterm + 2.0 * beta * I2 + beta**2 * I4)
            )

        return output

    def model_delta(self, theta: np.ndarray) -> np.ndarray:
        factor = np.pi if self.data_config.delta_divide_by_pi else 1.0
        return self.k_data * self.model_p1d(theta) / factor

    def chi2(self, theta: np.ndarray) -> float:
        residual = self.model_p1d(theta) - self.P1D_data
        if not np.all(np.isfinite(residual)):
            return 1.0e100
        value = float(residual @ self.icov @ residual)
        return value if np.isfinite(value) else 1.0e100

    # ------------------------------------------------------------------
    # The only fit: six parameters, always with the counterterm
    # ------------------------------------------------------------------

    @staticmethod
    def bounds_physical() -> List[Tuple[float, float]]:
        return [
            (np.log(1.0e-3), np.log(0.1)),
            (0.0, 8.0),
            (0.0, 4.0),
            (-4.0, 6.0),
            (-1.0, 1.0),
            (-6.0, 6.0),
        ]

    @staticmethod
    def bounds_broad() -> List[Tuple[float, float]]:
        return [
            (np.log(1.0e-4), np.log(1.0)),
            (-5.0, 12.0),
            (-4.0, 4.0),
            (-8.0, 10.0),
            (-50.0, 50.0),
            (-50.0, 50.0),
        ]

    @staticmethod
    def default_theta0() -> np.ndarray:
        return np.array([np.log(0.012), 4.0, 1.0, 2.0, 0.0, 0.0])

    def fit(
        self,
        bounds: Optional[List[Tuple[float, float]]] = None,
        theta0: Optional[np.ndarray] = None,
        run_global_first: bool = True,
        seed: int = 12345,
        de_kwargs: Optional[dict] = None,
        local_method: str = "Powell",
        local_kwargs: Optional[dict] = None,
    ):
        if not self.theory_ready:
            raise RuntimeError("Call prepare() before fit().")

        bounds = self.bounds_physical() if bounds is None else list(bounds)
        theta0 = self.default_theta0() if theta0 is None else np.asarray(theta0)

        global_result = None
        start = theta0

        if run_global_first:
            options = {
                "maxiter": 300,
                "popsize": 20,
                "tol": 1.0e-7,
                "atol": 1.0e-7,
                "polish": False,
                "seed": seed,
                "workers": 1,
                "updating": "immediate",
            }
            if de_kwargs:
                options.update(de_kwargs)
            global_result = differential_evolution(self.chi2, bounds, **options)
            start = global_result.x

        if local_method == "Powell":
            options = {"maxiter": 20000, "xtol": 1.0e-9, "ftol": 1.0e-10}
        elif local_method == "L-BFGS-B":
            options = {
                "maxiter": 20000,
                "ftol": 1.0e-13,
                "gtol": 1.0e-9,
                "maxls": 100,
            }
        else:
            raise ValueError("local_method must be 'Powell' or 'L-BFGS-B'.")

        minimize_arguments = {
            "method": local_method,
            "bounds": bounds,
            "options": options,
        }
        if local_kwargs:
            local_kwargs = dict(local_kwargs)
            user_options = local_kwargs.pop("options", None)
            minimize_arguments.update(local_kwargs)
            if user_options:
                minimize_arguments["options"].update(user_options)

        result = minimize(self.chi2, start, **minimize_arguments)
        result.global_result = global_result
        result.theta6 = np.asarray(result.x, dtype=float)
        result.chi2_value = self.chi2(result.theta6)
        result.npar = 6
        return result

    # Backward-compatible name for existing notebooks.
    fit_full_counterterm = fit

    # ------------------------------------------------------------------
    # Summary and compact diagnostics
    # ------------------------------------------------------------------

    def summary_dict(
        self, theta: np.ndarray, label: str = "", npar: int = 6
    ) -> dict:
        theta = np.asarray(theta, dtype=float)
        chi2_value = self.chi2(theta)
        ndof = len(self.P1D_data) - npar
        alpha_F, beta_F, alpha_bias, beta_bias, alpha_ct, beta_ct = (
            self.unpack_theta(theta)
        )

        return {
            "label": label,
            "mode": self.theory_config.mode,
            "k_min": self.theory_config.k_min,
            "k_max": self.theory_config.k_max,
            "n_k": self.theory_config.n_k,
            "N_data": len(self.P1D_data),
            "N_par": npar,
            "N_dof": ndof,
            "chi2": chi2_value,
            "chi2/dof": chi2_value / ndof,
            "p-value": chi2_dist.sf(chi2_value, ndof),
            "log_alpha_F": theta[0],
            "alpha_F": alpha_F,
            "beta_F": beta_F,
            "alpha_bias": alpha_bias,
            "beta_bias": beta_bias,
            "alpha_ct": alpha_ct,
            "beta_ct": beta_ct,
        }

    def print_summary(self, theta: np.ndarray, label: str = "", npar: int = 6) -> None:
        row = self.summary_dict(theta, label=label, npar=npar)

        print("Fit summary")
        print("-----------")
        if label:
            print("label    =", label)
        print("mode     =", row["mode"])
        print(
            f"k grid   = [{row['k_min']:.3g}, {row['k_max']:.3g}] h/Mpc "
            f"with {row['n_k']} points"
        )
        print("N_data   =", row["N_data"])
        print("N_par    =", row["N_par"])
        print("N_dof    =", row["N_dof"])
        print(f"chi2     = {row['chi2']:.6f}")
        print(f"chi2/dof = {row['chi2/dof']:.6f}")
        print(f"p-value  = {row['p-value']:.6e}")
        print()
        for name in (
            "log_alpha_F",
            "alpha_F",
            "beta_F",
            "alpha_bias",
            "beta_bias",
            "alpha_ct",
            "beta_ct",
        ):
            print(f"{name:12s} = {row[name]:.8g}")

    def nuisance_evolution(self, theta: np.ndarray) -> pd.DataFrame:
        alpha_F, beta_F, alpha_bias, beta_bias, alpha_ct, beta_ct = (
            self.unpack_theta(theta)
        )
        return pd.DataFrame(
            [
                {
                    "z": float(z),
                    "A": self.A_of_z(z, alpha_F, beta_F),
                    "beta": self.beta_of_z(z, alpha_bias, beta_bias),
                    "Ict": self.Ict_of_z(z, alpha_ct, beta_ct),
                }
                for z in self.z_unique
            ]
        )

    def chi2_by_redshift(self, theta: np.ndarray) -> pd.DataFrame:
        model = self.model_p1d(theta)
        rows = []
        for z in self.z_unique:
            mask = np.isclose(self.z_data, z)
            residual = model[mask] - self.P1D_data[mask]
            icov_z = self.icov[np.ix_(mask, mask)]
            value = float(residual @ icov_z @ residual)
            rows.append(
                {
                    "z": float(z),
                    "N_data": int(np.count_nonzero(mask)),
                    "chi2": value,
                    "chi2_per_point": value / np.count_nonzero(mask),
                }
            )
        return pd.DataFrame(rows)

    def model_components_by_z(self, theta: np.ndarray, z: float) -> dict:
        z_match = self.z_unique[np.argmin(np.abs(self.z_unique - float(z)))]
        mask = np.isclose(self.z_data, z_match)
        alpha_F, beta_F, alpha_bias, beta_bias, alpha_ct, beta_ct = (
            self.unpack_theta(theta)
        )
        I0, I2, I4 = self.I_cache[z_match]

        return {
            "z": float(z_match),
            "mask": mask,
            "kv": self.k_data[mask],
            "I0": I0,
            "I2": I2,
            "I4": I4,
            "A": self.A_of_z(z_match, alpha_F, beta_F),
            "beta": self.beta_of_z(z_match, alpha_bias, beta_bias),
            "Ict": self.Ict_of_z(z_match, alpha_ct, beta_ct),
            "P_model": self.model_p1d(theta)[mask],
            "P_data": self.P1D_data[mask],
        }
