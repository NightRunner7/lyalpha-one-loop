# Model-independent grid campaigns

This layer schedules the validated shared engine; it does not contain another
SPT implementation or another nuisance fitter.

One campaign point produces exactly two scientific files:

- `theory/<point_id>.npz`, containing the linear and one-loop channels;
- `fits/<point_id>.json`, produced by `scripts/fit_p1d.py`.

The DCDM, accelerated-DM or any future model layer is responsible only for
creating a full `ModelSpec` JSON for every grid coordinate.

## Directory layout

```text
campaigns/dcdm/
  base_model.json
  build_grid.py
  grid_specs/
cluster/
  campaign_manager.py
  theory_point.pbs
  fit_point.pbs
runs/dcdm/<campaign_id>/
  campaign.json
  manifest.csv
  models/
  theory/
  fits/
  checkpoints/
  status/{theory,fit}/
  jobs/{theory,fit}/
  logs/{theory,fit}/
  controller/
```

`runs/` contains generated products and should not be committed as source.

## 1. Build the small DCDM validation campaign

Run this from the project root:

```bash
python campaigns/dcdm/build_grid.py \
  --grid-spec campaigns/dcdm/grid_specs/validation_grid.json \
  --run-dir runs/dcdm/dcdm_validation_v1
```

The command can be repeated safely. Existing point indices and model files are
preserved. New coordinates in the spec are appended; an attempt to silently
change an existing model is rejected.

Inspect the generated plan without PBS:

```bash
python cluster/campaign_manager.py status \
  --campaign runs/dcdm/dcdm_validation_v1 \
  --offline

python cluster/campaign_manager.py submit-theory \
  --campaign runs/dcdm/dcdm_validation_v1 \
  --dry-run --offline --max-points 2
```

## 2. Configure the cluster runtime

Before submitting jobs, edit the `cluster` block in the grid spec and rebuild
the run directory. Choose one of these runtime routes:

1. Set `theory_env` to the conda environment containing the tested custom
   `classy` wrapper. Keep `theory_python` empty.
2. Set `theory_python` to the absolute interpreter path. This bypasses conda
   activation inside the PBS job.

The fit can use a lighter environment, but using the same tested environment
for the first validation is simpler. Adjust `theory_max_active`,
`fit_max_active` and `max_user_active` to the actual PBS limits. Each theory
job uses one CPU because the direct quadrature is not parallelized by this
workflow.

## 3. Submit and continuously refill both queues

First check live state:

```bash
python cluster/campaign_manager.py status \
  --campaign runs/dcdm/dcdm_validation_v1
```

Run the persistent controller in `tmux` or another login-node session allowed
by the cluster:

```bash
python -u cluster/campaign_manager.py watch \
  --campaign runs/dcdm/dcdm_validation_v1 \
  --stages theory fit
```

The controller reloads `manifest.csv` every cycle. Therefore, a repeated
`build_grid.py` call can append refinement points while the campaign is
running. Fits are submitted only after the matching self-validating theory
bundle has completed.

To stop cleanly from another shell:

```bash
python cluster/campaign_manager.py stop \
  --campaign runs/dcdm/dcdm_validation_v1
```

Clear the stop request before restarting:

```bash
python cluster/campaign_manager.py resume \
  --campaign runs/dcdm/dcdm_validation_v1
```

Use `--exit-when-complete` on `watch` if no live refinement will be added.

## 4. Failure and interruption semantics

The manager stores the PBS job ID for every attempt and reconciles it with
`qstat`. A historical submission record never blocks a point by itself. If a
job disappears from `qstat` without producing `done` or `failed`, the point is
eligible for another attempt until `max_attempts` is reached.

Explicitly reset failed or exhausted points with:

```bash
python cluster/campaign_manager.py retry-failed \
  --campaign runs/dcdm/dcdm_validation_v1 \
  --stage both
```

Restrict the reset by repeating `--point-id POINT_ID`. The command does not
delete valid theory or fit products. PBS wrappers validate existing products
and restore a missing completion marker without recomputation when possible.

## 5. Escalation from validation to the paper grid

Use this order:

1. `validation_grid.json` with `production` quality;
2. `precision_anchors.json` for the best-fit and LCDM-like anchor points;
3. only after production/precision agreement, `paper_grid.json` for all
   23 x 28 coordinates listed in arXiv:2210.06117.

Build the precision anchors in a separate run directory:

```bash
python campaigns/dcdm/build_grid.py \
  --grid-spec campaigns/dcdm/grid_specs/precision_anchors.json \
  --run-dir runs/dcdm/dcdm_precision_anchors_v1
```

After both campaigns finish, compare the raw best-fit anchor before looking at
the nuisance fit:

```bash
python scripts/compare_theories.py \
  runs/dcdm/dcdm_validation_v1/theory/dcdm_tau40_eps0p006.npz \
  runs/dcdm/dcdm_precision_anchors_v1/theory/dcdm_tau40_eps0p006.npz \
  --k-max 2
```

The validated reference values are 189.906473 (`production`) and 189.887294
(`precision`) for the one-loop chi2 at this point. They are documented for
comparison and are not enforced as an artificial optimizer target.

Build the full paper grid only after accepting those checks:

```bash
python campaigns/dcdm/build_grid.py \
  --grid-spec campaigns/dcdm/grid_specs/paper_grid.json \
  --run-dir runs/dcdm/dcdm_paper_grid_v1
```

The full paper grid has 644 points. Its fit mode is `one_loop`, because a
separate linear fit for every cosmological point is not required for the DCDM
likelihood surface. The small validation and precision campaigns keep
`mode=both` as an end-to-end diagnostic.

## 6. Robust second-pass fit of a completed grid

A smooth physical likelihood surface can still contain isolated optimizer
failures.  Do not smooth or interpolate those values away.  Instead, run a
second fit pass in which every target point is initialized from:

- its own saved solution in every configured source directory;
- the saved solutions of up to eight adjacent grid points;
- a local refinement of every imported solution on the **target's own**
  theory bundle.

The target and neighbour coordinates are used only to select starting
vectors.  The likelihood, data and theory are never borrowed from another
point.  The raw imported target solution is retained as a candidate, so the
second pass cannot be worse than the preserved first pass within the same
numerical box.

First make sure the direct `k_uv=20` pass is complete:

```bash
python cluster/campaign_manager.py status \
  --campaign runs/dcdm/dcdm_paper_grid_v1
```

Then archive it and prepare a clean neighbour-refit stage:

```bash
python cluster/campaign_manager.py prepare-neighbor-refit \
  --campaign runs/dcdm/dcdm_paper_grid_v1 \
  --archive-label direct_k20
```

If an earlier continuation pass was preserved, include it as another source:

```bash
python cluster/campaign_manager.py prepare-neighbor-refit \
  --campaign runs/dcdm/dcdm_paper_grid_v1 \
  --archive-label direct_k20 \
  --additional-source-fit-dir fits_continuation_old
```

The preparation command refuses to run unless all active fits are complete
and refuses to overwrite an existing archive.  It does not modify theory
bundles or checkpoints.  It moves the current fit products to directories
whose names end in the archive label, opens an empty `fits/` stage, and writes
the following fit configuration to `campaign.json`:

```json
{
  "strategy": "neighbor_refit",
  "source_fit_dirs": ["fits_direct_k20", "fits_continuation_old"]
}
```

Run only the new fit stage; no CLASS or one-loop calculation is repeated:

```bash
python -u cluster/campaign_manager.py watch \
  --campaign runs/dcdm/dcdm_paper_grid_v1 \
  --stages fit \
  --exit-when-complete
```

Each output JSON remains compatible with the campaign notebook and also
contains `neighbour_refit_audit`, `neighbour_point_ids`, and
`neighbour_refit_selected_source`.  These fields record every tested start
and the exact source of the selected minimum.  After completion, rerun
`campaigns/dcdm/notebooks/01_inspect_campaign.ipynb`; it automatically reads
the new active `fits/` directory.

## 7. Adding another model family

Do not edit `lyalpha_pt`, `generate_theory.py`, `fit_p1d.py`, the PBS wrappers
or the manager. Add a new directory under `campaigns/` with:

- one base model configuration;
- one grid builder that emits complete `ModelSpec` JSON files;
- one or more grid specs.

The generated run directory must follow the same `campaign.json` and
`manifest.csv` contract. This keeps SPT, projection, covariance and nuisance
optimization model-independent.
