from __future__ import annotations

import json
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from lyalpha_pt.campaign_refit import (
    GridPoint,
    collect_start_candidates,
    grid_neighbour_ids,
    load_active_grid,
    parameter_vector,
)
from lyalpha_pt.fit import PARAMETER_NAMES


def _fit_payload(offset: float = 0.0) -> dict:
    parameters = {
        name: float(index + offset) for index, name in enumerate(PARAMETER_NAMES)
    }
    return {"fits": {"one_loop": {"parameters": parameters}}}


class GridNeighbourTests(unittest.TestCase):
    def setUp(self) -> None:
        self.points = tuple(
            GridPoint(
                point_id=f"p_{tau}_{epsilon}",
                tau_gyr=float(tau),
                epsilon_dcdm=float(epsilon),
            )
            for tau in (1, 2, 4)
            for epsilon in (0.001, 0.002, 0.004)
        )

    def test_central_point_has_eight_neighbours(self) -> None:
        neighbours = grid_neighbour_ids(self.points, "p_2_0.002")
        self.assertEqual(len(neighbours), 8)
        self.assertNotIn("p_2_0.002", neighbours)

    def test_corner_point_has_three_neighbours(self) -> None:
        neighbours = set(grid_neighbour_ids(self.points, "p_1_0.001"))
        self.assertEqual(
            neighbours,
            {"p_1_0.002", "p_2_0.001", "p_2_0.002"},
        )

    def test_missing_coordinate_is_skipped(self) -> None:
        points = tuple(
            point for point in self.points if point.point_id != "p_4_0.004"
        )
        neighbours = grid_neighbour_ids(points, "p_2_0.002")
        self.assertEqual(len(neighbours), 7)

    def test_model_independent_accdm_axes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "campaign.json").write_text(
                json.dumps({"grid_axes": ["log10m_acc", "log10f_acc"]}),
                encoding="utf-8",
            )
            with (root / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=(
                        "point_id",
                        "active",
                        "log10m_acc",
                        "log10f_acc",
                    ),
                )
                writer.writeheader()
                for log_mass in (9, 12, 15):
                    for log_fraction in (-4, -2, 0):
                        writer.writerow(
                            {
                                "point_id": f"p_{log_mass}_{log_fraction}",
                                "active": 1,
                                "log10m_acc": log_mass,
                                "log10f_acc": log_fraction,
                            }
                        )
            points = load_active_grid(root / "manifest.csv")
            neighbours = grid_neighbour_ids(points, "p_12_-2")
            self.assertEqual(len(neighbours), 8)
            self.assertEqual(points[0].coordinate_names, ("log10m_acc", "log10f_acc"))


class StartCandidateTests(unittest.TestCase):
    def test_parameter_vector_uses_canonical_order(self) -> None:
        theta = parameter_vector(_fit_payload(0.5), "one_loop")
        np.testing.assert_allclose(theta, np.arange(6, dtype=float) + 0.5)

    def test_collects_target_and_available_neighbours_from_all_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            (first / "target.json").write_text(
                json.dumps(_fit_payload(0.0)), encoding="utf-8"
            )
            (first / "left.json").write_text(
                json.dumps(_fit_payload(1.0)), encoding="utf-8"
            )
            (second / "target.json").write_text(
                json.dumps(_fit_payload(2.0)), encoding="utf-8"
            )
            (second / "right.json").write_text(
                json.dumps(_fit_payload(3.0)), encoding="utf-8"
            )

            candidates = collect_start_candidates(
                source_directories=(first, second),
                target_point_id="target",
                neighbour_point_ids=("left", "right", "missing"),
                mode="one_loop",
            )

            self.assertEqual(len(candidates), 4)
            self.assertEqual(
                [(Path(item.source_directory).name, item.source_point_id) for item in candidates],
                [
                    ("first", "target"),
                    ("first", "left"),
                    ("second", "target"),
                    ("second", "right"),
                ],
            )

    def test_target_fit_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            (source / "left.json").write_text(
                json.dumps(_fit_payload()), encoding="utf-8"
            )
            with self.assertRaises(FileNotFoundError):
                collect_start_candidates(
                    source_directories=(source,),
                    target_point_id="target",
                    neighbour_point_ids=("left",),
                    mode="one_loop",
                )


if __name__ == "__main__":
    unittest.main()
