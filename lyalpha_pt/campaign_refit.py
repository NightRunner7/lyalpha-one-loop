"""Neighbour-seeded nuisance refits for completed grid campaigns."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import itertools
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import OptimizeResult

from .data import load_dr12
from .fit import EffectiveModelFit, PARAMETER_NAMES, write_fit_result
from .theory import load_theory


@dataclass(frozen=True)
class GridPoint:
    """One active point from a campaign manifest."""

    point_id: str
    tau_gyr: float | None = None
    epsilon_dcdm: float | None = None
    coordinate_names: tuple[str, ...] = ()
    coordinates: tuple[float, ...] = ()

    def coordinate_tuple(self) -> tuple[float, ...]:
        """Return this point's ordered model-independent grid coordinates."""

        if self.coordinates:
            if len(self.coordinates) != len(self.coordinate_names):
                raise ValueError(
                    f"Grid point {self.point_id!r} has inconsistent coordinate metadata."
                )
            return self.coordinates
        if self.tau_gyr is not None and self.epsilon_dcdm is not None:
            return (float(self.tau_gyr), float(self.epsilon_dcdm))
        raise ValueError(f"Grid point {self.point_id!r} has no coordinates.")


@dataclass(frozen=True)
class StartCandidate:
    """A nuisance solution imported from one campaign fit file."""

    source_directory: str
    source_point_id: str
    theta: np.ndarray


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _grid_axes_from_campaign(manifest_path: Path) -> tuple[str, ...]:
    config_path = manifest_path.parent / "campaign.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        raw_axes = config.get("grid_axes")
        if raw_axes is not None:
            if not isinstance(raw_axes, list) or not all(
                isinstance(value, str) and value.strip() for value in raw_axes
            ):
                raise ValueError("campaign.json grid_axes must be a list of names.")
            axes = tuple(raw_axes)
            if len(axes) != len(set(axes)):
                raise ValueError("campaign.json grid_axes contains duplicate names.")
            if not axes:
                raise ValueError("campaign.json grid_axes cannot be empty.")
            return axes
    # Backward compatibility for campaigns created before grid_axes was stored.
    return ("tau_gyr", "epsilon_dcdm")


def load_active_grid(
    manifest_path: str | Path,
    coordinate_names: Sequence[str] | None = None,
) -> tuple[GridPoint, ...]:
    """Load active coordinates while preserving manifest order.

    New campaigns declare their coordinate columns through ``grid_axes`` in
    ``campaign.json``.  Old DCDM campaigns remain readable through the
    historical ``tau_gyr`` and ``epsilon_dcdm`` fallback.
    """

    manifest_path = Path(manifest_path)
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    axes = (
        tuple(str(value) for value in coordinate_names)
        if coordinate_names is not None
        else _grid_axes_from_campaign(manifest_path)
    )
    if not axes or len(axes) != len(set(axes)):
        raise ValueError("Grid coordinate names must be non-empty and unique.")
    required = {"point_id", "active", *axes}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(
            f"Manifest {manifest_path} must contain {sorted(required)}."
        )
    points = tuple(
        GridPoint(
            point_id=str(row["point_id"]),
            tau_gyr=(
                float(row["tau_gyr"]) if "tau_gyr" in axes else None
            ),
            epsilon_dcdm=(
                float(row["epsilon_dcdm"])
                if "epsilon_dcdm" in axes
                else None
            ),
            coordinate_names=axes,
            coordinates=tuple(float(row[name]) for name in axes),
        )
        for row in rows
        if _truthy(row["active"])
    )
    if not points:
        raise ValueError(f"Manifest has no active points: {manifest_path}")
    if len({point.point_id for point in points}) != len(points):
        raise ValueError("Manifest contains duplicate active point IDs.")
    return points


def grid_neighbour_ids(
    points: Sequence[GridPoint], target_point_id: str
) -> tuple[str, ...]:
    """Return existing points in the surrounding coordinate stencil.

    The target itself is excluded.  Missing coordinate combinations are simply
    skipped, which also makes the routine useful for refinement grids.  In two
    dimensions this is the familiar surrounding 3x3 stencil with at most eight
    neighbours.
    """

    by_id = {point.point_id: point for point in points}
    if target_point_id not in by_id:
        raise KeyError(f"Unknown active campaign point: {target_point_id}")
    target = by_id[target_point_id]
    coordinate_names = target.coordinate_names
    if not coordinate_names:
        coordinate_names = ("tau_gyr", "epsilon_dcdm")
    for point in points:
        names = point.coordinate_names or ("tau_gyr", "epsilon_dcdm")
        if names != coordinate_names:
            raise ValueError("All grid points must use the same coordinate order.")

    coordinate_rows = [point.coordinate_tuple() for point in points]
    dimensions = len(target.coordinate_tuple())
    if dimensions < 1:
        raise ValueError("A campaign grid needs at least one coordinate axis.")
    axes = [
        sorted({coordinate[index] for coordinate in coordinate_rows})
        for index in range(dimensions)
    ]
    target_coordinate = target.coordinate_tuple()
    target_indices = [
        axis.index(target_coordinate[index]) for index, axis in enumerate(axes)
    ]
    by_coordinate = {
        point.coordinate_tuple(): point.point_id for point in points
    }

    neighbours: list[str] = []
    for offsets in itertools.product((-1, 0, 1), repeat=dimensions):
        if all(offset == 0 for offset in offsets):
            continue
        neighbour_indices = [
            target_index + offset
            for target_index, offset in zip(target_indices, offsets)
        ]
        if any(
            index < 0 or index >= len(axis)
            for index, axis in zip(neighbour_indices, axes)
        ):
            continue
        coordinate = tuple(
            axis[index] for axis, index in zip(axes, neighbour_indices)
        )
        point_id = by_coordinate.get(coordinate)
        if point_id is not None:
            neighbours.append(point_id)
    return tuple(neighbours)


def parameter_vector(fit_payload: Mapping[str, object], mode: str) -> np.ndarray:
    """Extract the canonical six-parameter vector from one fit payload."""

    try:
        fits = fit_payload["fits"]
        record = fits[mode]  # type: ignore[index]
        parameters = record["parameters"]  # type: ignore[index]
        theta = np.array([parameters[name] for name in PARAMETER_NAMES], dtype=float)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Fit payload has no valid {mode!r} parameter vector.") from error
    if theta.shape != (6,) or not np.all(np.isfinite(theta)):
        raise ValueError("Fit parameter vector must contain six finite values.")
    return theta


def collect_start_candidates(
    *,
    source_directories: Iterable[str | Path],
    target_point_id: str,
    neighbour_point_ids: Sequence[str],
    mode: str,
) -> tuple[StartCandidate, ...]:
    """Load target and neighbour solutions from all available source folders."""

    candidates: list[StartCandidate] = []
    point_ids = (target_point_id, *neighbour_point_ids)
    for source in source_directories:
        source_path = Path(source).resolve()
        for point_id in point_ids:
            path = source_path / f"{point_id}.json"
            if not path.is_file():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            candidates.append(
                StartCandidate(
                    source_directory=str(source_path),
                    source_point_id=point_id,
                    theta=parameter_vector(payload, mode),
                )
            )
    if not any(candidate.source_point_id == target_point_id for candidate in candidates):
        raise FileNotFoundError(
            f"No source fit for target point {target_point_id!r} in the configured folders."
        )
    return tuple(candidates)


def _raw_result(
    fitter: EffectiveModelFit,
    theta: Sequence[float],
    *,
    message: str,
    bounds: Sequence[tuple[float, float]],
) -> OptimizeResult:
    theta_array = np.asarray(theta, dtype=float)
    inside_box = all(
        lower <= value <= upper
        for value, (lower, upper) in zip(theta_array, bounds)
    )
    result = OptimizeResult(
        x=theta_array,
        fun=float(fitter.chi2(theta_array)) if inside_box else float("inf"),
        success=inside_box,
        message=message if inside_box else f"{message}; outside final bounds",
    )
    result.boundary_diagnostics = fitter.boundary_diagnostics(result.x, bounds)
    return result


def refit_campaign_point(
    *,
    campaign_dir: str | Path,
    point_id: str,
    theory_path: str | Path,
    data_dir: str | Path,
    output_path: str | Path,
    source_fit_directories: Sequence[str | Path],
    modes: Sequence[str] = ("one_loop",),
    k_uv_cut: float = 20.0,
    covariance: str = "paper_diag",
    expanded_counterterm: bool = True,
    full_six_dimensional: bool = False,
) -> dict[str, object]:
    """Refit one target from its own and neighbouring nuisance solutions.

    Every start is evaluated against the target theory before local refinement.
    The saved result is therefore guaranteed not to be worse than any imported
    target solution that uses the same objective and numerical box.
    """

    campaign_dir = Path(campaign_dir).resolve()
    theory_path = Path(theory_path).resolve()
    data_dir = Path(data_dir).resolve()
    output_path = Path(output_path).resolve()
    points = load_active_grid(campaign_dir / "manifest.csv")
    neighbours = grid_neighbour_ids(points, point_id)
    dataset = load_dr12(data_dir)
    theory = load_theory(theory_path)
    final_bounds = (
        EffectiveModelFit.bounds_expanded_counterterm()
        if expanded_counterterm
        else EffectiveModelFit.bounds_broad()
    )
    profile_amplitudes = not full_six_dimensional

    records: dict[str, object] = {}
    audits: dict[str, object] = {}
    for mode in modes:
        starts = collect_start_candidates(
            source_directories=source_fit_directories,
            target_point_id=point_id,
            neighbour_point_ids=neighbours,
            mode=mode,
        )
        fitter = EffectiveModelFit(
            dataset,
            theory,
            mode=mode,
            k_uv_cut=k_uv_cut,
            covariance_mode=covariance,
        )

        trials: list[tuple[float, OptimizeResult, dict[str, object]]] = []
        for start in starts:
            label = f"{Path(start.source_directory).name}:{start.source_point_id}"
            raw = _raw_result(
                fitter,
                start.theta,
                message=f"imported start {label}",
                bounds=final_bounds,
            )
            refined = fitter.continue_staged(
                start.theta,
                expand_counterterm=expanded_counterterm,
                profile_amplitudes=profile_amplitudes,
            )
            # Report every solution against the final box rather than the
            # narrower intermediate box used by the staged search.
            refined.boundary_diagnostics = fitter.boundary_diagnostics(
                refined.x, final_bounds
            )
            raw_value = float(raw.fun)
            refined_value = float(fitter.chi2(refined.x))
            if np.isfinite(raw_value):
                trials.append(
                    (
                        raw_value,
                        raw,
                        {
                            "source_directory": start.source_directory,
                            "source_point_id": start.source_point_id,
                            "candidate_kind": "imported",
                            "chi2": raw_value,
                        },
                    )
                )
            trials.append(
                (
                    refined_value,
                    refined,
                    {
                        "source_directory": start.source_directory,
                        "source_point_id": start.source_point_id,
                        "candidate_kind": "locally_refined",
                        "chi2": refined_value,
                    },
                )
            )

        best_value, best_result, selected = min(trials, key=lambda item: item[0])
        best_result.message = (
            "grid-neighbour refit; selected "
            f"{Path(str(selected['source_directory'])).name}:"
            f"{selected['source_point_id']}:{selected['candidate_kind']}"
        )
        best_result.boundary_diagnostics = fitter.boundary_diagnostics(
            best_result.x, final_bounds
        )
        record = fitter.result_record(best_result)
        record["neighbour_refit_selected_source"] = dict(selected)
        records[mode] = record
        audits[mode] = {
            "selected": dict(selected),
            "best_chi2": best_value,
            "candidate_count": len(trials),
            "candidates": [metadata for _, _, metadata in trials],
        }

    output: dict[str, object] = {
        "theory_file": str(theory_path),
        "data_dir": str(data_dir),
        "optimizer_parameterization": (
            "full_six_dimensional"
            if full_six_dimensional
            else "four_dimensional_with_two_profiled_amplitudes"
        ),
        "cutoff_continuation_enabled": False,
        "refit_strategy": "target_and_adjacent_grid_neighbours",
        "target_point_id": point_id,
        "neighbour_point_ids": list(neighbours),
        "source_fit_directories": [
            str(Path(path).resolve()) for path in source_fit_directories
        ],
        "fits": records,
        "neighbour_refit_audit": audits,
    }
    if set(records) == {"linear", "one_loop"}:
        linear = records["linear"]  # type: ignore[assignment]
        one_loop = records["one_loop"]  # type: ignore[assignment]
        output["delta_chi2_one_loop_minus_linear"] = (
            one_loop["chi2"] - linear["chi2"]  # type: ignore[index]
        )
    write_fit_result(output_path, output)
    return output
