"""The only model-dependent part of the Ly-alpha calculation.

Add DCDM or accelerated-DM models by defining one :class:`ModelSpec`.  The
generator, theory file and fitter do not need to be changed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class ModelSpec:
    """Complete CLASS input and the prescription used for the loop term.

    ``loop_source='cb'`` implements Eq. (16) of arXiv:2011.03050.  In that
    case ``loop_weight='one_minus_fnu_squared'`` multiplies P22+P13 by
    ``(1-f_nu)**2`` while the tree-level term remains the total-matter power.
    """

    name: str
    description: str
    class_params: Mapping[str, Any]
    loop_source: str = "total"
    loop_weight: str | float = 1.0
    tags: tuple[str, ...] = field(default_factory=tuple)

    def validate(self) -> None:
        if self.loop_source not in {"total", "cb"}:
            raise ValueError("loop_source must be 'total' or 'cb'.")
        if not (
            isinstance(self.loop_weight, (int, float))
            or self.loop_weight == "one_minus_fnu_squared"
        ):
            raise ValueError(
                "loop_weight must be numeric or 'one_minus_fnu_squared'."
            )
        if not self.name.strip():
            raise ValueError("Model name cannot be empty.")

    def class_input(self, *, z_max_pk: float, pk_max_hmpc: float) -> dict[str, Any]:
        self.validate()
        params = dict(self.class_params)
        params.update(
            {
                "output": "mPk",
                "P_k_max_h/Mpc": float(pk_max_hmpc),
                "z_max_pk": float(z_max_pk),
                "input_verbose": 0,
                "background_verbose": 0,
                "perturbations_verbose": 0,
            }
        )
        return params

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["class_params"] = dict(self.class_params)
        out["tags"] = list(self.tags)
        return out

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ModelSpec":
        data = dict(value)
        data["tags"] = tuple(data.get("tags", ()))
        spec = cls(**data)
        spec.validate()
        return spec


_H_2021 = 0.678


PRESETS: dict[str, ModelSpec] = {
    "lcdm_2021_massless": ModelSpec(
        name="lcdm_2021_massless",
        description=(
            "Massless-neutrino LCDM benchmark of arXiv:2011.03050, "
            "Sec. 3.1."
        ),
        class_params={
            "H0": 100.0 * _H_2021,
            "omega_b": 0.0482 * _H_2021**2,
            "omega_cdm": 0.260 * _H_2021**2,
            "A_s": 2.12e-9,
            "n_s": 0.961,
            "tau_reio": 0.0952,
            "N_ur": 3.046,
            "N_ncdm": 0,
        },
        loop_source="total",
        loop_weight=1.0,
        tags=("paper-2021", "benchmark", "massless-neutrinos"),
    ),
    "lcdm_2022_planck": ModelSpec(
        name="lcdm_2022_planck",
        description=(
            "Planck LCDM reference from Table 1 of arXiv:2210.06117, with "
            "one 0.06 eV massive and two massless neutrino species."
        ),
        class_params={
            "omega_b": 0.02237,
            "omega_cdm": 0.1200,
            "100*theta_s": 1.04110,
            "ln10^{10}A_s": 3.044,
            "n_s": 0.9649,
            "tau_reio": 0.0544,
            "N_ur": 2.0328,
            "N_ncdm": 1,
            "m_ncdm": 0.06,
            "deg_ncdm": 1.0,
            "T_ncdm": 0.71611,
        },
        loop_source="cb",
        loop_weight="one_minus_fnu_squared",
        tags=("paper-2022", "benchmark", "planck-2018"),
    ),
}


def available_presets() -> tuple[str, ...]:
    return tuple(sorted(PRESETS))


def get_preset(name: str) -> ModelSpec:
    try:
        return PRESETS[name]
    except KeyError as exc:
        choices = ", ".join(available_presets())
        raise KeyError(f"Unknown model preset {name!r}. Available: {choices}") from exc
