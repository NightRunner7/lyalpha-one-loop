"""CLASS-free P1D projection, six-parameter effective model and robust fit."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import OptimizeResult, differential_evolution, minimize
from scipy.stats import chi2 as chi2_distribution

from .data import DR12Dataset
from .theory import TheoryBundle


PARAMETER_NAMES = (
    "log_alpha_F",
    "beta_F",
    "alpha_bias",
    "beta_bias",
    "alpha_ct",
    "beta_ct",
)


@dataclass(frozen=True)
class EffectiveConfig:
    z_pivot: float = 3.0
    k_s_velocity: float = 0.11
    k_filter_hmpc: float = 18.0
    f_siiii: float = 6e-3
    delta_v_siiii: float = 2 * np.pi / 0.0028
    use_siiii: bool = True
    use_thermal: bool = True
    use_filter: bool = True


class EffectiveModelFit:
    """P1D fit that consumes only a saved theory bundle and public data."""

    def __init__(
        self,
        dataset: DR12Dataset,
        theory: TheoryBundle,
        *,
        mode: str,
        k_uv_cut: float = 20.0,
        covariance_mode: str = "paper_diag",
        effective_config: EffectiveConfig = EffectiveConfig(),
        n_projection_k: int = 4000,
    ):
        if mode not in {"linear", "one_loop"}:
            raise ValueError("mode must be linear or one_loop")
        theory.validate()
        self.dataset = dataset
        self.theory = theory
        self.mode = mode
        self.k_uv_cut = float(k_uv_cut)
        self.covariance_mode = covariance_mode
        self.config = effective_config
        self.n_projection_k = int(n_projection_k)

        if not np.allclose(dataset.z_unique, theory.z, rtol=0, atol=1e-9):
            raise ValueError(
                f"Data redshifts {dataset.z_unique} and theory redshifts {theory.z} differ."
            )
        k_available_max = (
            float(theory.k_input[-1]) if mode == "linear" else float(theory.k_loop[-1])
        )
        if not (0 < self.k_uv_cut <= k_available_max):
            raise ValueError(
                f"k_uv_cut={self.k_uv_cut} exceeds available {mode} theory "
                f"range ending at {k_available_max} h/Mpc."
            )

        covariance = dataset.covariance(covariance_mode)
        self._cho = cho_factor(covariance, lower=True, check_finite=True)
        self.covariance = covariance
        self.inverse_covariance = cho_solve(self._cho, np.eye(dataset.n_data))
        diagonal = np.diag(np.diag(covariance))
        self._diagonal_weights = (
            1.0 / np.diag(covariance)
            if np.allclose(covariance, diagonal, rtol=0, atol=1e-15)
            else None
        )
        self._build_integrals()

    @staticmethod
    def bounds_broad() -> list[tuple[float, float]]:
        """Wide numerical safety box; it is not a scientific prior."""

        return [
            (np.log(1e-5), np.log(2.0)),
            (-10.0, 20.0),
            (-8.0, 8.0),
            (-15.0, 20.0),
            (-100.0, 100.0),
            (-100.0, 100.0),
        ]

    @staticmethod
    def bounds_expanded_counterterm() -> list[tuple[float, float]]:
        bounds = EffectiveModelFit.bounds_broad()
        bounds[-2] = (-1000.0, 1000.0)
        bounds[-1] = (-1000.0, 1000.0)
        return bounds

    @staticmethod
    def default_theta() -> np.ndarray:
        return np.array([np.log(0.012), 4.0, 1.0, 2.0, 0.0, 0.0])

    def _source_channels(self, iz: int) -> tuple[np.ndarray, np.ndarray]:
        if self.mode == "linear":
            values = self.theory.p_total_input[iz]
            channels = np.repeat(values[:, None], 3, axis=1)
            return self.theory.k_input, channels
        return self.theory.k_loop, self.theory.channels_one_loop[iz]

    def _filter(self, k_hmpc: np.ndarray) -> np.ndarray:
        if not self.config.use_filter:
            return np.ones_like(k_hmpc)
        return np.exp(-(k_hmpc / self.config.k_filter_hmpc) ** 2)

    def _build_integrals(self) -> None:
        n_data = self.dataset.n_data
        self.i0 = np.empty(n_data)
        self.i2 = np.empty(n_data)
        self.i4 = np.empty(n_data)
        self.k_parallel = np.empty(n_data)

        for iz, redshift in enumerate(self.theory.z):
            mask = np.isclose(self.dataset.z, redshift)
            conversion = float(self.theory.velocity_to_hmpc[iz])
            k_parallel = conversion * self.dataset.k_velocity[mask]
            self.k_parallel[mask] = k_parallel
            if np.max(k_parallel) >= self.k_uv_cut:
                raise ValueError(
                    f"At z={redshift}, data reach k_parallel={np.max(k_parallel):.5g} "
                    f">= k_uv_cut={self.k_uv_cut}."
                )

            source_k, source_channels = self._source_channels(iz)
            source_mask = (source_k >= source_k[0]) & (source_k <= self.k_uv_cut)
            source_k = source_k[source_mask]
            source_channels = source_channels[source_mask]
            if source_k[0] > np.min(k_parallel):
                raise ValueError("Theory grid does not extend below the observed k_parallel.")

            splines = [
                InterpolatedUnivariateSpline(np.log(source_k), source_channels[:, i], k=3)
                for i in range(3)
            ]
            integration_k = np.geomspace(
                max(source_k[0], min(np.min(k_parallel) * 0.95, source_k[0])),
                self.k_uv_cut,
                self.n_projection_k,
            )
            integration_k = np.unique(
                np.concatenate([integration_k, k_parallel, [self.k_uv_cut]])
            )
            channels = np.column_stack(
                [spline(np.log(integration_k)) for spline in splines]
            )
            pressure = self._filter(integration_k)
            integrands = np.column_stack(
                [
                    integration_k * pressure * channels[:, 0],
                    pressure * channels[:, 1] / integration_k,
                    pressure * channels[:, 2] / integration_k**3,
                ]
            )

            cumulative = np.empty_like(integrands)
            for column in range(3):
                reversed_integral = cumulative_trapezoid(
                    integrands[::-1, column],
                    integration_k[::-1],
                    initial=0.0,
                )
                cumulative[:, column] = -reversed_integral[::-1]
            cumulative_splines = [
                InterpolatedUnivariateSpline(np.log(integration_k), cumulative[:, i], k=3)
                for i in range(3)
            ]
            evaluated = np.column_stack(
                [spline(np.log(k_parallel)) for spline in cumulative_splines]
            )
            self.i0[mask] = conversion * evaluated[:, 0]
            self.i2[mask] = conversion * k_parallel**2 * evaluated[:, 1]
            self.i4[mask] = conversion * k_parallel**4 * evaluated[:, 2]

        pivot_mask = np.isclose(self.dataset.z, self.config.z_pivot)
        if not np.any(pivot_mask):
            raise ValueError("The pivot redshift is absent from the data.")
        self.i0_scale = float(np.median(np.abs(self.i0[pivot_mask])))
        if not np.isfinite(self.i0_scale) or self.i0_scale == 0:
            raise RuntimeError("Invalid I0 scale used to parameterize the counterterm.")

    def _siiii(self) -> np.ndarray:
        if not self.config.use_siiii:
            return np.ones(self.dataset.n_data)
        mean_flux = np.exp(-0.0025 * (1 + self.dataset.z) ** 3.7)
        ratio = self.config.f_siiii / (1 - mean_flux)
        return (
            1
            + 2 * ratio * np.cos(self.config.delta_v_siiii * self.dataset.k_velocity)
            + ratio**2
        )

    def _thermal(self) -> np.ndarray:
        if not self.config.use_thermal:
            return np.ones(self.dataset.n_data)
        return np.exp(-(self.dataset.k_velocity / self.config.k_s_velocity) ** 2)

    def model(self, theta: Sequence[float]) -> np.ndarray:
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (6,):
            raise ValueError("theta must contain six parameters")
        log_alpha_f, beta_f, alpha_bias, beta_bias, alpha_ct, beta_ct = theta
        redshift_ratio = (1 + self.dataset.z) / (1 + self.config.z_pivot)
        amplitude = np.exp(log_alpha_f) * redshift_ratio**beta_f
        beta = alpha_bias * redshift_ratio**beta_bias
        counterterm = self.i0_scale * alpha_ct * redshift_ratio ** (-beta_ct)
        return (
            amplitude
            * self._siiii()
            * self._thermal()
            * (self.i0 + counterterm + 2 * beta * self.i2 + beta**2 * self.i4)
        )

    def chi2(self, theta: Sequence[float]) -> float:
        try:
            residual = self.model(theta) - self.dataset.p1d
        except (FloatingPointError, OverflowError, ValueError):
            return 1e100
        if not np.all(np.isfinite(residual)):
            return 1e100
        whitened = cho_solve(self._cho, residual)
        value = float(residual @ whitened)
        return value if np.isfinite(value) else 1e100

    def _profile_amplitudes(
        self,
        nonlinear: Sequence[float],
        bounds: Sequence[tuple[float, float]],
    ) -> tuple[float, np.ndarray]:
        """Profile the two exactly linear amplitude combinations.

        For fixed ``(beta_F, alpha_bias, beta_bias, beta_ct)`` the model is
        ``a * base + c * counterterm_shape``, where ``a=alpha_F`` and
        ``c=alpha_F*alpha_ct``.  Solving the 2x2 weighted normal equations
        eliminates the stiff amplitude/counterterm degeneracy from the global
        optimizer without imposing a prior.
        """

        beta_f, alpha_bias, beta_bias, beta_ct = np.asarray(nonlinear, dtype=float)
        ratio = (1 + self.dataset.z) / (1 + self.config.z_pivot)
        beta = alpha_bias * ratio**beta_bias
        common = self._siiii() * self._thermal()
        base = common * ratio**beta_f * (
            self.i0 + 2 * beta * self.i2 + beta**2 * self.i4
        )
        counterterm_shape = (
            common * self.i0_scale * ratio ** (beta_f - beta_ct)
        )
        design = np.column_stack([base, counterterm_shape])
        target = self.dataset.p1d
        try:
            if self._diagonal_weights is not None:
                weighted_design = self._diagonal_weights[:, None] * design
                normal = design.T @ weighted_design
                rhs = design.T @ (self._diagonal_weights * target)
            else:
                inverse_design = cho_solve(self._cho, design)
                normal = design.T @ inverse_design
                rhs = design.T @ cho_solve(self._cho, target)
            coefficients = np.linalg.solve(normal, rhs)
        except (FloatingPointError, np.linalg.LinAlgError, ValueError):
            return 1e100, np.full(6, np.nan)

        alpha_f, combined_counterterm = coefficients
        if not np.isfinite(alpha_f) or alpha_f <= 0:
            return 1e100, np.full(6, np.nan)
        log_alpha_f = float(np.log(alpha_f))
        alpha_ct = float(combined_counterterm / alpha_f)
        theta = np.array(
            [log_alpha_f, beta_f, alpha_bias, beta_bias, alpha_ct, beta_ct]
        )
        if any(value < lower or value > upper for value, (lower, upper) in zip(theta, bounds)):
            return 1e100, theta
        residual = design @ coefficients - target
        if self._diagonal_weights is not None:
            value = float(np.sum(self._diagonal_weights * residual**2))
        else:
            value = float(residual @ cho_solve(self._cho, residual))
        return (value if np.isfinite(value) else 1e100), theta

    def fit_profiled_once(
        self,
        *,
        bounds: Sequence[tuple[float, float]] | None = None,
        seed: int = 12345,
        de_maxiter: int = 300,
        de_popsize: int = 20,
        search_counterterm_faces: bool = True,
    ) -> OptimizeResult:
        """Globally fit four nonlinear parameters, profiling two amplitudes.

        The profiled likelihood contains narrow valleys that can terminate on
        a ``beta_ct`` boundary.  A four-dimensional differential-evolution
        population can miss such a face even for many seeds.  We therefore
        search the interior and, independently, the two ``beta_ct`` faces.
        Every global candidate is refined with bounded local methods, but a
        local result is never accepted when it is worse than its starting
        point.  This last guard is important because Powell can jump between
        disconnected profiled valleys.
        """

        bounds = list(self.bounds_broad() if bounds is None else bounds)
        if len(bounds) != 6:
            raise ValueError("Exactly six bounds are required.")
        reduced_bounds = [bounds[index] for index in (1, 2, 3, 5)]

        def objective(values):
            return self._profile_amplitudes(values, bounds)[0]

        global_result = differential_evolution(
            objective,
            reduced_bounds,
            seed=int(seed),
            maxiter=int(de_maxiter),
            popsize=int(de_popsize),
            tol=1e-9,
            atol=1e-9,
            polish=False,
            workers=1,
            updating="immediate",
        )

        candidates: list[tuple[str, np.ndarray, float, object | None]] = [
            ("interior_de", np.asarray(global_result.x), float(global_result.fun), global_result)
        ]

        if search_counterterm_faces:
            face_bounds = reduced_bounds[:3]
            for face_name, beta_ct in (
                ("beta_ct_lower", reduced_bounds[3][0]),
                ("beta_ct_upper", reduced_bounds[3][1]),
            ):
                def face_objective(values, fixed_beta_ct=beta_ct):
                    return objective([values[0], values[1], values[2], fixed_beta_ct])

                face_global = differential_evolution(
                    face_objective,
                    face_bounds,
                    seed=int(seed),
                    maxiter=int(de_maxiter),
                    popsize=int(de_popsize),
                    tol=1e-9,
                    atol=1e-9,
                    polish=False,
                    workers=1,
                    updating="immediate",
                )
                face_values = np.r_[face_global.x, beta_ct]
                candidates.append(
                    (face_name + "_de", face_values, float(face_global.fun), face_global)
                )

        refined: list[tuple[str, np.ndarray, float, object | None]] = []
        for source, start, start_value, source_result in candidates:
            refined.append((source, start, start_value, source_result))
            for method, options in (
                (
                    "L-BFGS-B",
                    {"maxiter": 30000, "ftol": 1e-15, "gtol": 1e-10},
                ),
                (
                    "Nelder-Mead",
                    {"maxiter": 30000, "xatol": 1e-9, "fatol": 1e-10},
                ),
                (
                    "Powell",
                    {"maxiter": 30000, "xtol": 1e-10, "ftol": 1e-11},
                ),
            ):
                local = minimize(
                    objective,
                    start,
                    method=method,
                    bounds=reduced_bounds,
                    options=options,
                )
                refined.append(
                    (
                        source + "_" + method.lower(),
                        np.asarray(local.x),
                        float(local.fun),
                        local,
                    )
                )

        best_source, best_values, _, local = min(refined, key=lambda item: item[2])
        value, theta = self._profile_amplitudes(best_values, bounds)
        result = OptimizeResult(
            x=theta,
            fun=value,
            success=bool(getattr(local, "success", True)),
            message=f"profiled amplitudes; selected {best_source}",
        )
        result.global_result = global_result
        result.profiled_result = local
        result.search_candidates = [
            {"source": source, "chi2": float(candidate_value)}
            for source, _, candidate_value, _ in refined
        ]
        result.boundary_diagnostics = self.boundary_diagnostics(theta, bounds)
        return result

    def refine_from_theta(
        self,
        theta: Sequence[float],
        *,
        bounds: Sequence[tuple[float, float]],
        profile_amplitudes: bool = True,
    ) -> OptimizeResult:
        """Continue an existing minimum inside a new numerical box."""

        theta = np.asarray(theta, dtype=float)
        if theta.shape != (6,):
            raise ValueError("theta must contain six parameters")
        bounds = list(bounds)
        if profile_amplitudes:
            reduced_bounds = [bounds[index] for index in (1, 2, 3, 5)]
            start = np.clip(
                theta[[1, 2, 3, 5]],
                [lower for lower, _ in reduced_bounds],
                [upper for _, upper in reduced_bounds],
            )

            def objective(values):
                return self._profile_amplitudes(values, bounds)[0]

            candidates = [("continuation_start", start, objective(start), None)]
            for method, options in (
                (
                    "L-BFGS-B",
                    {"maxiter": 30000, "ftol": 1e-15, "gtol": 1e-10},
                ),
                (
                    "Nelder-Mead",
                    {"maxiter": 30000, "xatol": 1e-9, "fatol": 1e-10},
                ),
                (
                    "Powell",
                    {"maxiter": 30000, "xtol": 1e-10, "ftol": 1e-11},
                ),
            ):
                local = minimize(
                    objective,
                    start,
                    method=method,
                    bounds=reduced_bounds,
                    options=options,
                )
                candidates.append(
                    (method.lower(), np.asarray(local.x), float(local.fun), local)
                )
            source, values, _, local = min(candidates, key=lambda item: item[2])
            value, fitted_theta = self._profile_amplitudes(values, bounds)
            result = OptimizeResult(
                x=fitted_theta,
                fun=value,
                success=bool(getattr(local, "success", True)),
                message=f"profiled continuation; selected {source}",
            )
            result.search_candidates = [
                {"source": name, "chi2": float(candidate_value)}
                for name, _, candidate_value, _ in candidates
            ]
        else:
            result = self.fit_once(
                bounds=bounds,
                global_first=False,
                theta0=np.clip(
                    theta,
                    [lower for lower, _ in bounds],
                    [upper for _, upper in bounds],
                ),
            )
        result.boundary_diagnostics = self.boundary_diagnostics(result.x, bounds)
        return result

    def fit_once(
        self,
        *,
        bounds: Sequence[tuple[float, float]] | None = None,
        seed: int = 12345,
        global_first: bool = True,
        theta0: Sequence[float] | None = None,
        de_maxiter: int = 300,
        de_popsize: int = 20,
    ):
        bounds = list(self.bounds_broad() if bounds is None else bounds)
        if len(bounds) != 6:
            raise ValueError("Exactly six bounds are required.")
        start = self.default_theta() if theta0 is None else np.asarray(theta0, dtype=float)
        global_result = None
        if global_first:
            global_result = differential_evolution(
                self.chi2,
                bounds,
                seed=int(seed),
                maxiter=int(de_maxiter),
                popsize=int(de_popsize),
                tol=1e-8,
                atol=1e-8,
                polish=False,
                workers=1,
                updating="immediate",
            )
            start = global_result.x
        local = minimize(
            self.chi2,
            start,
            method="Powell",
            bounds=bounds,
            options={"maxiter": 30000, "xtol": 1e-10, "ftol": 1e-11},
        )
        local.global_result = global_result
        local.boundary_diagnostics = self.boundary_diagnostics(local.x, bounds)
        return local

    def fit_multistart(
        self,
        *,
        seeds: Iterable[int] = (12345, 23456, 34567),
        bounds: Sequence[tuple[float, float]] | None = None,
        de_maxiter: int = 300,
        de_popsize: int = 20,
        profile_amplitudes: bool = True,
        initial_thetas: Iterable[Sequence[float]] = (),
    ):
        fit_method = self.fit_profiled_once if profile_amplitudes else self.fit_once
        results = [
            fit_method(
                bounds=bounds,
                seed=seed,
                de_maxiter=de_maxiter,
                de_popsize=de_popsize,
            )
            for seed in seeds
        ]
        results.extend(
            self.refine_from_theta(
                theta,
                bounds=(self.bounds_broad() if bounds is None else bounds),
                profile_amplitudes=profile_amplitudes,
            )
            for theta in initial_thetas
        )
        best = min(results, key=lambda result: self.chi2(result.x))
        best.multistart_results = results
        return best

    def fit_staged(
        self,
        *,
        seeds: Iterable[int] = (12345, 23456, 34567),
        expand_counterterm: bool = False,
        de_maxiter: int = 300,
        de_popsize: int = 20,
        profile_amplitudes: bool = True,
        initial_thetas: Iterable[Sequence[float]] = (),
    ) -> OptimizeResult:
        """Fit the base box first and expand it only by continuation.

        Expanding ``beta_ct`` before the first global search makes a narrow
        likelihood valley much harder to find.  This staged procedure always
        retains the best base-box solution and therefore cannot return a
        worse fit merely because the user requested a wider safety box.
        """

        seeds = tuple(int(seed) for seed in seeds)
        base = self.fit_multistart(
            seeds=seeds,
            bounds=self.bounds_broad(),
            de_maxiter=de_maxiter,
            de_popsize=de_popsize,
            profile_amplitudes=profile_amplitudes,
            initial_thetas=initial_thetas,
        )
        if not expand_counterterm:
            base.fit_stages = [
                {"stage": "base", "chi2": float(self.chi2(base.x))}
            ]
            return base

        # Do not launch a new global population in the much wider box.  It
        # dilutes the sampling density and was the source of missed narrow
        # minima in the legacy optimizer.  Continue every base-box candidate
        # locally instead, while retaining all original candidates.
        expanded_results = [
            self.refine_from_theta(
                item.x,
                bounds=self.bounds_expanded_counterterm(),
                profile_amplitudes=profile_amplitudes,
            )
            for item in base.multistart_results
        ]
        expanded = min(expanded_results, key=lambda result: self.chi2(result.x))
        all_results = [*base.multistart_results, *expanded_results]
        best = min(all_results, key=lambda result: self.chi2(result.x))
        best.multistart_results = all_results
        best.fit_stages = [
            {"stage": "base", "chi2": float(self.chi2(base.x))},
            {"stage": "expanded", "chi2": float(self.chi2(expanded.x))},
        ]
        return best

    def continue_staged(
        self,
        theta: Sequence[float],
        *,
        expand_counterterm: bool = False,
        profile_amplitudes: bool = True,
    ) -> OptimizeResult:
        """Continue a trusted solution to a neighboring projection cutoff."""

        base = self.refine_from_theta(
            theta,
            bounds=self.bounds_broad(),
            profile_amplitudes=profile_amplitudes,
        )
        candidates = [base]
        stages = [{"stage": "base_continuation", "chi2": float(self.chi2(base.x))}]
        if expand_counterterm:
            expanded_from_original = self.refine_from_theta(
                theta,
                bounds=self.bounds_expanded_counterterm(),
                profile_amplitudes=profile_amplitudes,
            )
            expanded_from_base = self.refine_from_theta(
                base.x,
                bounds=self.bounds_expanded_counterterm(),
                profile_amplitudes=profile_amplitudes,
            )
            candidates.extend([expanded_from_original, expanded_from_base])
            stages.append(
                {
                    "stage": "expanded_continuation",
                    "chi2": float(
                        min(
                            self.chi2(expanded_from_original.x),
                            self.chi2(expanded_from_base.x),
                        )
                    ),
                }
            )
        best = min(candidates, key=lambda result: self.chi2(result.x))
        best.multistart_results = candidates
        best.fit_stages = stages
        return best

    @staticmethod
    def boundary_diagnostics(
        theta: Sequence[float], bounds: Sequence[tuple[float, float]]
    ) -> list[dict[str, float | str | bool]]:
        rows = []
        for name, value, (lower, upper) in zip(PARAMETER_NAMES, theta, bounds):
            width = upper - lower
            distance = min(value - lower, upper - value) / width
            rows.append(
                {
                    "parameter": name,
                    "value": float(value),
                    "lower": float(lower),
                    "upper": float(upper),
                    "relative_distance_to_nearest_bound": float(distance),
                    "near_bound": bool(distance < 0.01),
                }
            )
        return rows

    def summary(self, theta: Sequence[float]) -> dict:
        theta = np.asarray(theta, dtype=float)
        chi2_value = self.chi2(theta)
        dof = self.dataset.n_data - 6
        return {
            "mode": self.mode,
            "covariance_mode": self.covariance_mode,
            "k_uv_cut_hmpc": self.k_uv_cut,
            "theory_model": self.theory.metadata["model"]["name"],
            "theory_digest": self.theory.metadata["bundle_digest"],
            "n_data": self.dataset.n_data,
            "n_parameters": 6,
            "dof": dof,
            "chi2": chi2_value,
            "chi2_per_dof": chi2_value / dof,
            "p_value": float(chi2_distribution.sf(chi2_value, dof)),
            "parameters": {name: float(value) for name, value in zip(PARAMETER_NAMES, theta)},
            "alpha_F": float(np.exp(theta[0])),
        }

    def chi2_by_redshift(self, theta: Sequence[float]) -> list[dict[str, float | int]]:
        prediction = self.model(theta)
        rows = []
        for redshift in self.dataset.z_unique:
            mask = np.isclose(self.dataset.z, redshift)
            indices = np.flatnonzero(mask)
            residual = prediction[mask] - self.dataset.p1d[mask]
            inverse_block = self.inverse_covariance[np.ix_(indices, indices)]
            value = float(residual @ inverse_block @ residual)
            rows.append(
                {
                    "z": float(redshift),
                    "n_data": int(np.count_nonzero(mask)),
                    "chi2": value,
                }
            )
        return rows

    def result_record(self, result) -> dict:
        record = self.summary(result.x)
        record.update(
            {
                "optimizer_success": bool(result.success),
                "optimizer_message": str(result.message),
                "boundary_diagnostics": result.boundary_diagnostics,
                "chi2_by_redshift": self.chi2_by_redshift(result.x),
            }
        )
        if hasattr(result, "multistart_results"):
            record["multistart_chi2"] = [
                self.chi2(item.x) for item in result.multistart_results
            ]
        if hasattr(result, "fit_stages"):
            record["fit_stages"] = result.fit_stages
        if hasattr(result, "search_candidates"):
            record["search_candidates"] = result.search_candidates
        return record


def write_fit_result(path: str | Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
