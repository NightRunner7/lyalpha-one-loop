"""Loading and strict validation of the public BOSS DR12/DR14 P1D files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class DR12Dataset:
    z: np.ndarray
    k_velocity: np.ndarray
    p1d: np.ndarray
    stat: np.ndarray
    systematic_components: np.ndarray
    correlation_blocks: np.ndarray
    source_paths: dict[str, str]

    @property
    def z_unique(self) -> np.ndarray:
        return np.unique(self.z)

    @property
    def n_data(self) -> int:
        return len(self.z)

    @property
    def systematic_quadrature(self) -> np.ndarray:
        return np.sqrt(np.sum(self.systematic_components**2, axis=1))

    @property
    def sigma_total(self) -> np.ndarray:
        return np.sqrt(self.stat**2 + self.systematic_quadrature**2)

    def covariance(self, mode: str = "paper_diag") -> np.ndarray:
        """Return a covariance matrix for the selected redshift range.

        ``paper_diag`` is the fiducial prescription in arXiv:2011.03050:
        statistical and eight systematic errors are added in quadrature and
        off-diagonal correlations are discarded.
        """

        if mode == "paper_diag":
            return np.diag(self.sigma_total**2)
        if mode == "stat_diag":
            return np.diag(self.stat**2)

        cov = np.zeros((self.n_data, self.n_data), dtype=float)
        z_values = self.z_unique
        n_k = self.correlation_blocks.shape[1]
        for iz, z in enumerate(z_values):
            mask = np.isclose(self.z, z)
            indices = np.flatnonzero(mask)
            corr = self.correlation_blocks[iz]
            stat = self.stat[mask]
            syst = self.systematic_components[mask]

            if len(indices) != n_k:
                raise RuntimeError("Correlation block and selected data disagree.")
            if mode == "stat_corr":
                block = corr * np.outer(stat, stat)
            elif mode == "total_corr":
                sigma = np.sqrt(stat**2 + np.sum(syst**2, axis=1))
                block = corr * np.outer(sigma, sigma)
            elif mode == "outer_systematics":
                block = corr * np.outer(stat, stat)
                for column in syst.T:
                    block += np.outer(column, column)
            else:
                raise ValueError(
                    "covariance mode must be paper_diag, stat_diag, stat_corr, "
                    "total_corr, or outer_systematics"
                )
            cov[np.ix_(indices, indices)] = 0.5 * (block + block.T)
        return cov


def _load_release_blocks(correlation_path: Path, n_z: int, n_k: int) -> np.ndarray:
    raw = np.loadtxt(correlation_path)
    expected = (n_z * n_k, n_k)
    if raw.shape != expected:
        raise ValueError(
            f"Unexpected correlation-file shape {raw.shape}; expected {expected}."
        )
    blocks = raw.reshape(n_z, n_k, n_k)
    blocks = 0.5 * (blocks + np.swapaxes(blocks, 1, 2))
    if not np.allclose(np.diagonal(blocks, axis1=1, axis2=2), 1.0, atol=1e-8):
        raise ValueError("Correlation blocks do not have a unit diagonal.")
    return blocks


def load_dr12(
    data_dir: str | Path,
    *,
    z_min: float = 3.0,
    z_max: float = 4.2,
) -> DR12Dataset:
    data_dir = Path(data_dir)
    paths = {
        "data": data_dir / "Pk1D_data.dat",
        "systematics": data_dir / "Pk1D_syst.dat",
        "correlation": data_dir / "Pk1D_cor.dat",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing DR12 release files: " + ", ".join(missing))

    raw = np.loadtxt(paths["data"])
    systematics = np.loadtxt(paths["systematics"])
    if raw.shape != (455, 6):
        raise ValueError(f"Expected Pk1D_data shape (455, 6), found {raw.shape}.")
    if systematics.shape != (455, 8):
        raise ValueError(
            f"Expected Pk1D_syst shape (455, 8), found {systematics.shape}."
        )

    z_all, k_all, p1d_all, stat_all = raw[:, :4].T
    z_release = np.unique(z_all)
    counts = np.array([np.count_nonzero(np.isclose(z_all, z)) for z in z_release])
    if len(z_release) != 13 or not np.all(counts == 35):
        raise ValueError(
            "Expected 13 redshift blocks with 35 k bins in every block; "
            f"found redshifts={len(z_release)}, counts={counts.tolist()}."
        )

    release_blocks = _load_release_blocks(paths["correlation"], 13, 35)
    selected_z_mask = (z_release >= z_min - 1e-10) & (z_release <= z_max + 1e-10)
    selected_point_mask = (z_all >= z_min - 1e-10) & (z_all <= z_max + 1e-10)

    result = DR12Dataset(
        z=z_all[selected_point_mask],
        k_velocity=k_all[selected_point_mask],
        p1d=p1d_all[selected_point_mask],
        stat=stat_all[selected_point_mask],
        systematic_components=systematics[selected_point_mask],
        correlation_blocks=release_blocks[selected_z_mask],
        source_paths={key: str(path.resolve()) for key, path in paths.items()},
    )
    if result.n_data != 245 or len(result.z_unique) != 7:
        raise ValueError(
            f"The fiducial selection must contain 245 points in 7 z bins; "
            f"found {result.n_data} points in {len(result.z_unique)} bins."
        )
    return result
