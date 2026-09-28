# Version 3.7.0: neighbour-seeded campaign refits

This update adds a traceable second-pass nuisance refit for completed model
grids.  It is intended for isolated missed minima such as the discontinuous
lower-slope points in the DCDM paper-grid reconstruction.

## What changed

- `lyalpha_pt/campaign_refit.py` selects up to eight adjacent grid points,
  imports their saved nuisance vectors, evaluates them on the target theory,
  and locally refines every candidate.
- `scripts/refit_p1d_neighbors.py` exposes the procedure for one point.
- `cluster/campaign_manager.py` and `cluster/fit_point.pbs` support the
  `neighbor_refit` fit strategy.
- `prepare-neighbor-refit` safely archives a completed fit pass and opens a
  clean fit stage without touching theory bundles or checkpoints.
- Every final JSON records the selected source and the complete candidate
  audit.

## Update an existing installation

Copy the update files into the project root, then reinstall the editable
package and run the tests:

```bash
python -m pip install -e '.[validation]'
python -m unittest discover -s tests -v
```

## Refit the completed DCDM paper grid

Verify that the current direct pass has 644 completed fits:

```bash
python cluster/campaign_manager.py status \
  --campaign runs/dcdm/dcdm_paper_grid_v1
```

Archive that pass.  Include the second option only if the named directory
already exists:

```bash
python cluster/campaign_manager.py prepare-neighbor-refit \
  --campaign runs/dcdm/dcdm_paper_grid_v1 \
  --archive-label direct_k20 \
  --additional-source-fit-dir fits_continuation_old
```

Without a preserved continuation pass, use:

```bash
python cluster/campaign_manager.py prepare-neighbor-refit \
  --campaign runs/dcdm/dcdm_paper_grid_v1 \
  --archive-label direct_k20
```

Preview two PBS submissions:

```bash
python cluster/campaign_manager.py submit-fits \
  --campaign runs/dcdm/dcdm_paper_grid_v1 \
  --dry-run --offline --max-points 2
```

Then run the complete refit in `tmux`:

```bash
tmux new -s dcdm_neighbor_refit
python -u cluster/campaign_manager.py watch \
  --campaign runs/dcdm/dcdm_paper_grid_v1 \
  --stages fit \
  --exit-when-complete
```

Detach with `Ctrl-b`, then `d`.  Reattach with:

```bash
tmux attach -t dcdm_neighbor_refit
```

After completion, rerun
`campaigns/dcdm/notebooks/01_inspect_campaign.ipynb`.  The notebook reads the
new active `fits/` directory, while `fits_direct_k20/` remains available for
point-by-point regression comparisons.
