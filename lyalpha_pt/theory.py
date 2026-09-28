"""CLASS adapter, one-loop generator, checkpointing and auditable theory files."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

from .models import ModelSpec
from .spt import p13_channels_analytic, p13_channels_richardson, p22_channels


GENERATOR_VERSION = "3.2.0"
CHANNELS = ("dd", "dtheta", "thetatheta")
INVERSE_MPC_TO_GEV = 6.394_931_914_307_479e-39
REDUCED_PLANCK_MASS_GEV = 2.435e18


def _canonical_json(value: Any) -> str:
    def default(obj: Any):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Cannot JSON-encode {type(obj).__name__}")

    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=default)


def _update_array_hash(digest, array: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(array)
    digest.update(str(contiguous.dtype).encode())
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes())


@dataclass(frozen=True)
class LoopNumerics:
    k_output_min: float = 1e-3
    k_output_max: float = 20.0
    n_output_k: int = 110
    q_min: float = 1e-5
    q_max: float = 50.0
    n_q_low: int = 240
    n_q_mid: int = 480
    n_q_high: int = 240
    n_p: int = 120
    middle_width: float = 0.25
    p13_method: str = "analytic_closed"
    p13_n_q: int = 501
    p13_n_mu: int = 24
    p13_epsilon_relative: float = 1e-4
    n_linear_input: int = 5000
    k_trust: float = 2.0

    def validate(self) -> None:
        if not (0 < self.k_output_min < self.k_trust <= self.k_output_max):
            raise ValueError("Require 0 < k_output_min < k_trust <= k_output_max.")
        if not (0 < self.q_min < self.q_max):
            raise ValueError("Require 0 < q_min < q_max.")
        if self.n_output_k < 20 or self.n_linear_input < 100:
            raise ValueError("Theory grids are too small.")
        if min(self.n_q_low, self.n_q_mid, self.n_q_high, self.n_p) < 8:
            raise ValueError("P22 quadrature is too small.")
        if self.p13_method not in {"analytic_closed", "recursive_richardson"}:
            raise ValueError(
                "p13_method must be analytic_closed or recursive_richardson."
            )
        if self.p13_method == "analytic_closed":
            if self.p13_n_q < 41 or self.p13_n_q % 2 == 0:
                raise ValueError("Closed P13 requires an odd p13_n_q >= 41.")
        else:
            if self.p13_n_q < 20 or self.p13_n_mu < 8:
                raise ValueError("Recursive P13 quadrature is too small.")
            if self.p13_epsilon_relative <= 0:
                raise ValueError("P13 regulator must be positive.")

    @property
    def input_k_min(self) -> float:
        return min(self.q_min, self.k_output_min)

    @property
    def input_k_max(self) -> float:
        return self.q_max + self.k_output_max

    @classmethod
    def for_quality(cls, quality: str) -> "LoopNumerics":
        if quality == "smoke":
            return cls(
                k_output_max=20.0,
                n_output_k=28,
                q_max=20.0,
                n_q_low=50,
                n_q_mid=100,
                n_q_high=50,
                n_p=32,
                p13_n_q=121,
                p13_n_mu=8,
                n_linear_input=1000,
            )
        if quality == "production":
            return cls()
        if quality == "precision":
            return cls(
                n_output_k=160,
                q_max=100.0,
                n_q_low=500,
                n_q_mid=1000,
                n_q_high=500,
                n_p=200,
                p13_n_q=1001,
                p13_n_mu=40,
                n_linear_input=6000,
            )
        raise ValueError("quality must be smoke, production, or precision")


@dataclass(frozen=True)
class TheoryBundle:
    metadata: dict[str, Any]
    z: np.ndarray
    k_input: np.ndarray
    p_total_input: np.ndarray
    p_loop_input: np.ndarray
    k_loop: np.ndarray
    p_tree: np.ndarray
    p22: np.ndarray
    p13: np.ndarray
    p13_error: np.ndarray
    channels_one_loop: np.ndarray
    hubble_km_s_mpc: np.ndarray
    velocity_to_hmpc: np.ndarray
    h: float
    omega_m: float
    loop_weight: float

    def validate(self, *, verify_digest: bool = True) -> None:
        nz = len(self.z)
        expected = {
            "p_total_input": (nz, len(self.k_input)),
            "p_loop_input": (nz, len(self.k_input)),
            "p_tree": (nz, len(self.k_loop)),
            "p22": (nz, len(self.k_loop), 3),
            "p13": (nz, len(self.k_loop), 3),
            "p13_error": (nz, len(self.k_loop), 3),
            "channels_one_loop": (nz, len(self.k_loop), 3),
            "hubble_km_s_mpc": (nz,),
            "velocity_to_hmpc": (nz,),
        }
        for name, shape in expected.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape:
                raise ValueError(f"{name} has shape {value.shape}, expected {shape}.")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} contains non-finite values.")
        if not np.all(np.diff(self.z) > 0):
            raise ValueError("Redshift grid is not strictly increasing.")
        if not np.all(np.diff(self.k_input) > 0) or not np.all(np.diff(self.k_loop) > 0):
            raise ValueError("Theory k grids are not strictly increasing.")
        if verify_digest:
            expected_digest = self.metadata.get("bundle_digest")
            actual_digest = self.bundle_digest()
            if not expected_digest or expected_digest != actual_digest:
                raise ValueError(
                    "Theory-bundle digest mismatch: the file is stale or corrupted."
                )

    def bundle_digest(self) -> str:
        digest = sha256()
        for array in (
            self.z,
            self.k_input,
            self.p_total_input,
            self.p_loop_input,
            self.k_loop,
            self.p_tree,
            self.p22,
            self.p13,
            self.p13_error,
            self.channels_one_loop,
            self.hubble_km_s_mpc,
            self.velocity_to_hmpc,
        ):
            _update_array_hash(digest, np.asarray(array))
        for scalar in (self.h, self.omega_m, self.loop_weight):
            digest.update(np.float64(scalar).tobytes())
        return digest.hexdigest()


def _log_spectrum_callable(k_grid: np.ndarray, values: np.ndarray):
    if np.any(values <= 0) or np.any(~np.isfinite(values)):
        raise ValueError("Linear input spectrum must be positive and finite.")
    spline = InterpolatedUnivariateSpline(np.log(k_grid), np.log(values), k=3)
    k_min = float(k_grid[0])
    k_max = float(k_grid[-1])

    def evaluate(k):
        k = np.asarray(k, dtype=float)
        tolerance = 2e-12
        if np.any(k < k_min * (1 - tolerance)) or np.any(k > k_max * (1 + tolerance)):
            raise ValueError(
                f"Loop requested k outside [{k_min}, {k_max}] h/Mpc: "
                f"[{np.min(k)}, {np.max(k)}]."
            )
        return np.exp(spline(np.log(np.clip(k, k_min, k_max))))

    return evaluate


def _class_pk(cosmo, k_hmpc: np.ndarray, z: float, *, source: str, h: float) -> np.ndarray:
    if source == "total":
        method = cosmo.pk_lin
    elif source == "cb":
        if not hasattr(cosmo, "pk_cb_lin"):
            raise AttributeError(
                "This CLASS build does not expose pk_cb_lin, required by the cb prescription."
            )
        method = cosmo.pk_cb_lin
    else:
        raise ValueError(f"Unknown spectrum source {source!r}.")
    return np.array([method(float(k * h), float(z)) for k in k_hmpc]) * h**3


def _sum_massive_neutrino_mass(params: dict[str, Any]) -> float:
    masses = params.get("m_ncdm", 0.0)
    degeneracies = params.get("deg_ncdm", 1.0)

    def numbers(value):
        if isinstance(value, str):
            return [float(item.strip()) for item in value.split(",")]
        if np.isscalar(value):
            return [float(value)]
        return [float(item) for item in value]

    mass_values = numbers(masses)
    deg_values = numbers(degeneracies)
    if len(deg_values) == 1 and len(mass_values) > 1:
        deg_values *= len(mass_values)
    if len(mass_values) != len(deg_values):
        raise ValueError("m_ncdm and deg_ncdm have incompatible lengths.")
    return float(np.dot(mass_values, deg_values))


def _resolve_loop_weight(
    model: ModelSpec, *, h: float, omega_m: float, class_params: dict[str, Any]
) -> float:
    if isinstance(model.loop_weight, (int, float)):
        return float(model.loop_weight)
    if model.loop_weight == "one_minus_fnu_squared":
        omega_nu_physical = _sum_massive_neutrino_mass(class_params) / 93.14
        f_nu = omega_nu_physical / (omega_m * h**2)
        if not (0 <= f_nu < 1):
            raise ValueError(f"Unphysical neutrino fraction f_nu={f_nu}.")
        return float((1 - f_nu) ** 2)
    raise ValueError(f"Unknown loop-weight prescription {model.loop_weight!r}.")


def accdm_energy_budget_diagnostic(
    class_params: dict[str, Any], *, hubble_inverse_mpc: float
) -> dict[str, Any] | None:
    """Evaluate the conservative accDM energy-per-Hubble-volume bound.

    The condition ``eta*m_acc << rho_crit(a_acc)*H_acc^-3`` is a background
    consistency diagnostic; it does not depend on the perturbation hierarchy.
    We evaluate it at the transition centre, ``a_acc = a_t_acc``, and use
    ``rho_crit = 3 Mpl_reduced^2 H^2`` in natural units.
    """

    required = {"eta_acc", "m_acc_in_GeV", "a_t_acc"}
    if not required.issubset(class_params):
        return None
    eta = float(class_params["eta_acc"])
    mass_gev = float(class_params["m_acc_in_GeV"])
    a_acc = float(class_params["a_t_acc"])
    hubble_mpc = float(hubble_inverse_mpc)
    if eta < 0.0 or mass_gev <= 0.0 or not 0.0 < a_acc < 1.0:
        raise ValueError("Invalid accDM inputs for the energy-budget diagnostic.")
    if not np.isfinite(hubble_mpc) or hubble_mpc <= 0.0:
        raise ValueError("The transition Hubble rate must be positive and finite.")

    kinetic_energy_gev = eta * mass_gev
    hubble_gev = hubble_mpc * INVERSE_MPC_TO_GEV
    critical_energy_in_hubble_volume_gev = (
        3.0 * REDUCED_PLANCK_MASS_GEV**2 / hubble_gev
    )
    ratio = kinetic_energy_gev / critical_energy_in_hubble_volume_gev
    return {
        "condition": "eta*m_acc << rho_crit(a_acc)*H_acc^-3",
        "evaluation": "a_acc = a_t_acc (transition centre)",
        "a_acc": a_acc,
        "z_acc": 1.0 / a_acc - 1.0,
        "eta_acc": eta,
        "m_acc_GeV": mass_gev,
        "eta_m_acc_GeV": kinetic_energy_gev,
        "H_acc_inverse_Mpc": hubble_mpc,
        "H_acc_GeV": hubble_gev,
        "rho_crit_H_minus3_GeV": critical_energy_in_hubble_volume_gev,
        "ratio": ratio,
        "log10_ratio": float(np.log10(ratio)) if ratio > 0.0 else -np.inf,
        "below_conservative_upper_bound": bool(ratio < 1.0),
        "reduced_planck_mass_GeV": REDUCED_PLANCK_MASS_GEV,
    }


def _checkpoint_key(
    *,
    model: ModelSpec,
    numerics: LoopNumerics,
    z: float,
    k_input: np.ndarray,
    p_total_input: np.ndarray,
    p_loop_input: np.ndarray,
    class_params: dict[str, Any],
) -> str:
    digest = sha256()
    digest.update(GENERATOR_VERSION.encode())
    digest.update(_canonical_json(model.as_dict()).encode())
    digest.update(_canonical_json(asdict(numerics)).encode())
    digest.update(_canonical_json(class_params).encode())
    digest.update(np.float64(z).tobytes())
    for array in (k_input, p_total_input, p_loop_input):
        _update_array_hash(digest, array)
    return digest.hexdigest()


def _load_checkpoint(path: Path, key: str) -> dict[str, np.ndarray] | None:
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as data:
        stored_key = str(data["checkpoint_key"].item())
        if stored_key != key:
            return None
        return {name: np.asarray(data[name]) for name in ("p_tree", "p22", "p13", "p13_error")}


def _save_checkpoint(path: Path, key: str, values: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, checkpoint_key=np.array(key), **values)
    temporary.replace(path)


def generate_theory(
    model: ModelSpec,
    output_path: str | Path,
    *,
    z_values: np.ndarray | list[float] = (3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2),
    numerics: LoopNumerics = LoopNumerics(),
    checkpoint_dir: str | Path | None = None,
    overwrite_checkpoints: bool = False,
    verbose: bool = True,
) -> TheoryBundle:
    """Run CLASS and direct one-loop SPT, then write one self-validating bundle."""

    model.validate()
    numerics.validate()
    z = np.sort(np.asarray(z_values, dtype=float))
    if len(z) == 0 or len(np.unique(z)) != len(z):
        raise ValueError("z_values must be non-empty and unique.")

    try:
        from classy import Class
    except ImportError as exc:
        raise RuntimeError(
            "Generating theory requires the Python CLASS wrapper 'classy'. "
            "The local fitting step does not require CLASS."
        ) from exc

    class_params = model.class_input(
        z_max_pk=float(z[-1] + 0.2),
        pk_max_hmpc=1.05 * numerics.input_k_max,
    )
    cosmo = Class()
    cosmo.set(class_params)
    cosmo.compute()
    h = float(cosmo.h())
    omega_m = float(cosmo.Omega_m())
    sigma8 = float(cosmo.sigma8())
    s8 = float(sigma8 * np.sqrt(omega_m / 0.3))
    loop_weight = _resolve_loop_weight(
        model, h=h, omega_m=omega_m, class_params=class_params
    )
    energy_budget = None
    if {"eta_acc", "m_acc_in_GeV", "a_t_acc"}.issubset(class_params):
        z_acc = 1.0 / float(class_params["a_t_acc"]) - 1.0
        energy_budget = accdm_energy_budget_diagnostic(
            class_params,
            hubble_inverse_mpc=float(cosmo.Hubble(z_acc)),
        )

    k_input = np.geomspace(
        numerics.input_k_min, numerics.input_k_max, numerics.n_linear_input
    )
    k_loop = np.geomspace(
        numerics.k_output_min, numerics.k_output_max, numerics.n_output_k
    )

    nz = len(z)
    p_total_input = np.empty((nz, len(k_input)))
    p_loop_input = np.empty_like(p_total_input)
    p_tree = np.empty((nz, len(k_loop)))
    p22 = np.empty((nz, len(k_loop), 3))
    p13 = np.empty_like(p22)
    p13_error = np.empty_like(p22)
    hubble = np.empty(nz)
    conversion = np.empty(nz)

    output_path = Path(output_path)
    if checkpoint_dir is None:
        checkpoint_dir = output_path.parent / "checkpoints"
    checkpoint_dir = Path(checkpoint_dir)

    for iz, redshift in enumerate(z):
        p_total_input[iz] = _class_pk(
            cosmo, k_input, redshift, source="total", h=h
        )
        p_loop_input[iz] = _class_pk(
            cosmo, k_input, redshift, source=model.loop_source, h=h
        )
        hubble[iz] = float(cosmo.Hubble(float(redshift)) * 299792.458)
        conversion[iz] = hubble[iz] / ((1 + redshift) * h)

        key = _checkpoint_key(
            model=model,
            numerics=numerics,
            z=redshift,
            k_input=k_input,
            p_total_input=p_total_input[iz],
            p_loop_input=p_loop_input[iz],
            class_params=class_params,
        )
        checkpoint_path = checkpoint_dir / (
            f"{model.name}_z{redshift:.1f}_{key[:16]}.npz"
        )
        checkpoint = None if overwrite_checkpoints else _load_checkpoint(checkpoint_path, key)

        if checkpoint is not None:
            p_tree[iz] = checkpoint["p_tree"]
            p22[iz] = checkpoint["p22"]
            p13[iz] = checkpoint["p13"]
            p13_error[iz] = checkpoint["p13_error"]
            if verbose:
                print(f"z={redshift:.1f}: loaded verified checkpoint {checkpoint_path.name}")
            continue

        p_tree[iz] = _class_pk(cosmo, k_loop, redshift, source="total", h=h)
        p_linear = _log_spectrum_callable(k_input, p_loop_input[iz])
        for ik, kval in enumerate(k_loop):
            if verbose:
                print(
                    f"z={redshift:.1f}  k {ik + 1:3d}/{len(k_loop)}  "
                    f"{kval:.6g} h/Mpc",
                    end="\r",
                    flush=True,
                )
            p22[iz, ik] = p22_channels(
                kval,
                p_linear,
                q_min=numerics.q_min,
                q_max=numerics.q_max,
                n_q_low=numerics.n_q_low,
                n_q_mid=numerics.n_q_mid,
                n_q_high=numerics.n_q_high,
                n_p=numerics.n_p,
                middle_width=numerics.middle_width,
            )
            if numerics.p13_method == "analytic_closed":
                p13[iz, ik], p13_error[iz, ik] = p13_channels_analytic(
                    kval,
                    p_linear,
                    q_min=numerics.q_min,
                    q_max=numerics.q_max,
                    n_q=numerics.p13_n_q,
                )
            else:
                p13[iz, ik], p13_error[iz, ik] = p13_channels_richardson(
                    kval,
                    p_linear,
                    q_min=numerics.q_min,
                    q_max=numerics.q_max,
                    n_q=numerics.p13_n_q,
                    n_mu=numerics.p13_n_mu,
                    epsilon_relative=numerics.p13_epsilon_relative,
                )
        if verbose:
            print()
        _save_checkpoint(
            checkpoint_path,
            key,
            {
                "p_tree": p_tree[iz],
                "p22": p22[iz],
                "p13": p13[iz],
                "p13_error": p13_error[iz],
            },
        )
        if verbose:
            print(f"z={redshift:.1f}: saved checkpoint {checkpoint_path.name}")

    channels = p_tree[:, :, None] + loop_weight * (p22 + p13)
    metadata: dict[str, Any] = {
        "format": "lyalpha_pt_theory_bundle",
        "format_version": 1,
        "generator_version": GENERATOR_VERSION,
        "model": model.as_dict(),
        "class_params": class_params,
        "numerics": asdict(numerics),
        "channels": list(CHANNELS),
        "tree_source": "total",
        "loop_source": model.loop_source,
        "loop_weight": loop_weight,
        "loop_formula": "P_XY = P_total_linear + loop_weight*(P22_XY+P13_XY)",
        "derived_cosmology": {
            "h": h,
            "Omega_m": omega_m,
            "sigma8": sigma8,
            "S8": s8,
        },
        "p22_method": "q-p variables, symmetric half-domain p>=q",
        "p13_method": numerics.p13_method,
        "p13_error_definition": (
            "absolute fine-minus-coarse radial quadrature difference"
            if numerics.p13_method == "analytic_closed"
            else "absolute Richardson-extrapolated minus epsilon/2 value"
        ),
    }
    if energy_budget is not None:
        metadata["accdm_energy_budget"] = energy_budget
    bundle = TheoryBundle(
        metadata=metadata,
        z=z,
        k_input=k_input,
        p_total_input=p_total_input,
        p_loop_input=p_loop_input,
        k_loop=k_loop,
        p_tree=p_tree,
        p22=p22,
        p13=p13,
        p13_error=p13_error,
        channels_one_loop=channels,
        hubble_km_s_mpc=hubble,
        velocity_to_hmpc=conversion,
        h=h,
        omega_m=omega_m,
        loop_weight=loop_weight,
    )
    metadata["bundle_digest"] = bundle.bundle_digest()
    bundle.validate()
    save_theory(bundle, output_path)

    if hasattr(cosmo, "struct_cleanup"):
        cosmo.struct_cleanup()
    if hasattr(cosmo, "empty"):
        cosmo.empty()
    return bundle


def save_theory(bundle: TheoryBundle, output_path: str | Path) -> None:
    bundle.validate()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            metadata_json=np.array(_canonical_json(bundle.metadata)),
            z=bundle.z,
            k_input=bundle.k_input,
            p_total_input=bundle.p_total_input,
            p_loop_input=bundle.p_loop_input,
            k_loop=bundle.k_loop,
            p_tree=bundle.p_tree,
            p22=bundle.p22,
            p13=bundle.p13,
            p13_error=bundle.p13_error,
            channels_one_loop=bundle.channels_one_loop,
            hubble_km_s_mpc=bundle.hubble_km_s_mpc,
            velocity_to_hmpc=bundle.velocity_to_hmpc,
            h=np.array(bundle.h),
            omega_m=np.array(bundle.omega_m),
            loop_weight=np.array(bundle.loop_weight),
        )
    temporary.replace(path)


def refine_p13_analytic(
    bundle: TheoryBundle,
    *,
    n_q: int = 1001,
    q_min: float | None = None,
    q_max: float | None = None,
    verbose: bool = True,
) -> TheoryBundle:
    """Replace P13 in an existing bundle with closed one-dimensional values.

    This CLASS-free path is useful for isolating the numerical effect of P13
    while preserving the original tree and P22 arrays exactly.  It always
    returns a new bundle and records the parent digest in its metadata.
    """

    bundle.validate()
    numerics = bundle.metadata.get("numerics", {})
    q_min = float(numerics.get("q_min", bundle.k_input[0]) if q_min is None else q_min)
    q_max = float(numerics.get("q_max", bundle.k_input[-1]) if q_max is None else q_max)
    if q_min < bundle.k_input[0] or q_max > bundle.k_input[-1] or q_min >= q_max:
        raise ValueError(
            "Requested P13 q range must lie inside the stored linear-spectrum grid."
        )

    p13 = np.empty_like(bundle.p13)
    p13_error = np.empty_like(bundle.p13_error)
    for iz, redshift in enumerate(bundle.z):
        p_linear = _log_spectrum_callable(bundle.k_input, bundle.p_loop_input[iz])
        for ik, kval in enumerate(bundle.k_loop):
            p13[iz, ik], p13_error[iz, ik] = p13_channels_analytic(
                float(kval),
                p_linear,
                q_min=q_min,
                q_max=q_max,
                n_q=n_q,
            )
        if verbose:
            print(f"z={redshift:.1f}: replaced P13 with closed EdS integrals")

    metadata = json.loads(_canonical_json(bundle.metadata))
    parent_digest = metadata.pop("bundle_digest")
    history = metadata.pop("refinement_history", [])
    previous_refinement = metadata.pop("refinement", None)
    if previous_refinement is not None:
        history.append(previous_refinement)
    history.append(
        {
            "operation": "replace_p13_analytic_closed",
            "parent_bundle_digest": parent_digest,
            "p13_n_q": int(n_q),
            "q_min": q_min,
            "q_max": q_max,
            "p22_reused_exactly": True,
        }
    )
    metadata.update(
        {
            "generator_version": GENERATOR_VERSION,
            "p13_method": "analytic_closed",
            "p13_error_definition": (
                "absolute fine-minus-coarse radial quadrature difference"
            ),
            "refinement_history": history,
        }
    )
    channels = bundle.p_tree[:, :, None] + bundle.loop_weight * (bundle.p22 + p13)
    refined = TheoryBundle(
        metadata=metadata,
        z=bundle.z.copy(),
        k_input=bundle.k_input.copy(),
        p_total_input=bundle.p_total_input.copy(),
        p_loop_input=bundle.p_loop_input.copy(),
        k_loop=bundle.k_loop.copy(),
        p_tree=bundle.p_tree.copy(),
        p22=bundle.p22.copy(),
        p13=p13,
        p13_error=p13_error,
        channels_one_loop=channels,
        hubble_km_s_mpc=bundle.hubble_km_s_mpc.copy(),
        velocity_to_hmpc=bundle.velocity_to_hmpc.copy(),
        h=bundle.h,
        omega_m=bundle.omega_m,
        loop_weight=bundle.loop_weight,
    )
    metadata["bundle_digest"] = refined.bundle_digest()
    refined.validate()
    return refined


def refine_p22_symmetric(
    bundle: TheoryBundle,
    *,
    q_min: float | None = None,
    q_max: float | None = None,
    n_q_low: int | None = None,
    n_q_mid: int | None = None,
    n_q_high: int | None = None,
    n_p: int | None = None,
    verbose: bool = True,
) -> TheoryBundle:
    """Recompute P22 with the symmetry-reduced q-p quadrature, without CLASS."""

    bundle.validate()
    numerics = bundle.metadata.get("numerics", {})
    q_min = float(numerics.get("q_min", bundle.k_input[0]) if q_min is None else q_min)
    q_max = float(numerics.get("q_max", bundle.k_input[-1]) if q_max is None else q_max)
    n_q_low = int(numerics.get("n_q_low", 240) if n_q_low is None else n_q_low)
    n_q_mid = int(numerics.get("n_q_mid", 480) if n_q_mid is None else n_q_mid)
    n_q_high = int(numerics.get("n_q_high", 240) if n_q_high is None else n_q_high)
    n_p = int(numerics.get("n_p", 120) if n_p is None else n_p)
    if q_min < bundle.k_input[0] or q_max + bundle.k_loop[-1] > bundle.k_input[-1] * (1 + 2e-12):
        raise ValueError(
            "Requested P22 domain requires q and p values outside the stored linear grid."
        )

    p22 = np.empty_like(bundle.p22)
    for iz, redshift in enumerate(bundle.z):
        p_linear = _log_spectrum_callable(bundle.k_input, bundle.p_loop_input[iz])
        for ik, kval in enumerate(bundle.k_loop):
            p22[iz, ik] = p22_channels(
                float(kval),
                p_linear,
                q_min=q_min,
                q_max=q_max,
                n_q_low=n_q_low,
                n_q_mid=n_q_mid,
                n_q_high=n_q_high,
                n_p=n_p,
            )
        if verbose:
            print(f"z={redshift:.1f}: recomputed P22 on symmetric half-domain")

    metadata = json.loads(_canonical_json(bundle.metadata))
    parent_digest = metadata.pop("bundle_digest")
    history = metadata.pop("refinement_history", [])
    previous_refinement = metadata.pop("refinement", None)
    if previous_refinement is not None:
        history.append(previous_refinement)
    history.append(
        {
            "operation": "replace_p22_symmetric_half_domain",
            "parent_bundle_digest": parent_digest,
            "q_min": q_min,
            "q_max": q_max,
            "n_q_low": n_q_low,
            "n_q_mid": n_q_mid,
            "n_q_high": n_q_high,
            "n_p": n_p,
            "p13_reused_exactly": True,
            "max_absolute_change_by_channel": np.max(
                np.abs(p22 - bundle.p22), axis=(0, 1)
            ).tolist(),
        }
    )
    metadata.update(
        {
            "generator_version": GENERATOR_VERSION,
            "p22_method": "q-p variables, symmetric half-domain p>=q",
            "refinement_history": history,
        }
    )
    channels = bundle.p_tree[:, :, None] + bundle.loop_weight * (p22 + bundle.p13)
    refined = TheoryBundle(
        metadata=metadata,
        z=bundle.z.copy(),
        k_input=bundle.k_input.copy(),
        p_total_input=bundle.p_total_input.copy(),
        p_loop_input=bundle.p_loop_input.copy(),
        k_loop=bundle.k_loop.copy(),
        p_tree=bundle.p_tree.copy(),
        p22=p22,
        p13=bundle.p13.copy(),
        p13_error=bundle.p13_error.copy(),
        channels_one_loop=channels,
        hubble_km_s_mpc=bundle.hubble_km_s_mpc.copy(),
        velocity_to_hmpc=bundle.velocity_to_hmpc.copy(),
        h=bundle.h,
        omega_m=bundle.omega_m,
        loop_weight=bundle.loop_weight,
    )
    metadata["bundle_digest"] = refined.bundle_digest()
    refined.validate()
    return refined


def load_theory(path: str | Path, *, verify_digest: bool = True) -> TheoryBundle:
    with np.load(Path(path), allow_pickle=False) as data:
        bundle = TheoryBundle(
            metadata=json.loads(str(data["metadata_json"].item())),
            z=np.asarray(data["z"]),
            k_input=np.asarray(data["k_input"]),
            p_total_input=np.asarray(data["p_total_input"]),
            p_loop_input=np.asarray(data["p_loop_input"]),
            k_loop=np.asarray(data["k_loop"]),
            p_tree=np.asarray(data["p_tree"]),
            p22=np.asarray(data["p22"]),
            p13=np.asarray(data["p13"]),
            p13_error=np.asarray(data["p13_error"]),
            channels_one_loop=np.asarray(data["channels_one_loop"]),
            hubble_km_s_mpc=np.asarray(data["hubble_km_s_mpc"]),
            velocity_to_hmpc=np.asarray(data["velocity_to_hmpc"]),
            h=float(data["h"].item()),
            omega_m=float(data["omega_m"].item()),
            loop_weight=float(data["loop_weight"].item()),
        )
    bundle.validate(verify_digest=verify_digest)
    return bundle
