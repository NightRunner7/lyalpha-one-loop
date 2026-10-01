# Accelerating-DM campaign definition

This directory is the accDM-specific layer of the Ly-alpha grid workflow.

For **CLASS `accDM_refactor`**, use `base_models/base_model_refactor_birth.json`
and follow [the Eagle / refactor guide](../../docs/EAGLE_PL.md). The same model
can be passed explicitly to `build_grid` for PBS. The new profile uses `m_nu`,
strategy `0, 5`, automatic daughter sampling, and the requested CLASS precision.
`cluster.prepare_eagle` selects it for every SPT quality, including `smoke`.

The fixed-q profiles and historical campaign recipes below are retained for
existing calculations. `build_grid` alone still defaults to the legacy
`base_model.json`; pass `--base-model` explicitly for a refactor campaign.
Historical adaptive-scan/recovery helpers are not birth-grid convergence tests.

## Physical convention

Each point scans

```text
log10m_acc = log10(m_acc / GeV)
log10f_acc = log10(f_acc)
```

and the builder maps these coordinates to CLASS through

```text
m_acc_in_GeV = 10**log10m_acc
m_cdm_in_GeV = m_acc_in_GeV
eta_acc      = 1e11 / m_acc_in_GeV
f_acc        = 10**log10f_acc
```

The transition is fixed at `kappa_acc=12.1` and `a_t_acc=0.133`.  The exact
Boltzmann hierarchy is enforced with `ncdm_fluid_approximation=3`; no fluid
switch is used.  The NCDM entries are one standard 0.06 eV neutrino and the
accDM daughter, while `N_ur=2.0308` represents the two effectively massless
neutrino species.

For every point the stable present-day CDM input is recalculated from

```text
omega_cdm = omega_cdm_early_total /
            (1 + f_acc * (1-a_rec**kappa) /
                         (1+(a_rec/a_t)**kappa))
```

so the total CDM density before acceleration remains `0.12010`.  This
convention is recorded in every model tag and in `campaign.json`.

The accDM one-loop term is generated directly from the total linear matter
spectrum with unit weight.  This is a physical model choice, not a neutrino
`cb` reweighting prescription.

Every newly generated accDM theory bundle also records the conservative
energy-budget diagnostic

```text
eta*m_acc / [rho_crit(a_t)*H(a_t)^-3]
```

in `metadata["accdm_energy_budget"]`.  CLASS supplies the background Hubble
rate at the transition centre, but the diagnostic does not use perturbations.
The same ratio and its base-ten logarithm are printed near the end of the PBS
`.out` log.  Since this scan fixes `eta*m_acc=1e11 GeV`, the numerator is
mass-independent; only the small point-to-point change in the transition
background can change the ratio.

## First validation campaign

Before building a large grid, edit the cluster environment names in
`grid_specs/validation_grid.json` if necessary and create a new run directory:

```bash
python campaigns/accdm/build_grid.py \
  --grid-spec campaigns/accdm/grid_specs/validation_grid.json \
  --run-dir runs/accdm/accdm_validation_v1
```

Inspect the immutable models and planned PBS commands:

```bash
python cluster/campaign_manager.py status \
  --campaign runs/accdm/accdm_validation_v1 --offline

python cluster/campaign_manager.py submit-theory \
  --campaign runs/accdm/accdm_validation_v1 \
  --dry-run --offline --max-points 2
```

Run theory and fits under a detachable terminal session:

```bash
tmux new -s accdm_validation

python -u cluster/campaign_manager.py watch \
  --campaign runs/accdm/accdm_validation_v1 \
  --stages theory fit \
  --exit-when-complete
```

Detach with `Ctrl-b`, then `d`.  Reattach with
`tmux attach -t accdm_validation`.

The validation grid intentionally includes difficult corners at
`log10m_acc=9`, `log10f_acc=0`.  A CLASS failure at an extreme point should be
diagnosed rather than silently removed.  Do not launch the final 1000--1500
point campaign until all intended corners have either completed or have a
documented physical/numerical exclusion.

## Required checks before a large scan

1. Confirm the `f_acc -> 0` edge approaches the existing LCDM reference.
2. Compare selected central and extreme points with 1001 and 2001 accDM
   momentum bins in separate immutable campaigns.
3. Generate production and precision theory for at least three representative
   points and compare raw channels below `2 h/Mpc`.
4. Check the direct `k_uv=20` nuisance fits and the five-percent numerical-bound
   audit in `notebooks/01_inspect_campaign.ipynb`.
5. Only after these checks, define the broad grid in a new grid-spec file and
   a new campaign directory.

Never modify generated model JSON files inside an existing run.  Change the
base model or grid spec and build a new campaign ID instead.

## Parallel validation and momentum-resolution suite

The first 16-point run identified the sampled minimum at
`log10m_acc=18`, `log10f_acc=-1`, with `chi2_one_loop=192.566262`, and no
nuisance parameter within five percent of its numerical bound.  It also found
RK failures at two extreme `1e9 GeV` points and at `1e18 GeV, f_acc=1`.

Before a large scan, build the follow-up suite:

```bash
python campaigns/accdm/validation_suite.py build
python campaigns/accdm/validation_suite.py status
python campaigns/accdm/validation_suite.py dry-run
```

The suite contains nine new theory jobs in four immutable campaigns:

| campaign | points | daughter momentum bins |
| --- | ---: | ---: |
| `accdm_validation_mass11_v1` | 4 | 1001 |
| `accdm_qconvergence_f0p3_q1001_v1` | 1 | 1001 |
| `accdm_qconvergence_joint_q2501_v1` | 2 | 2501 |
| `accdm_qconvergence_joint_q5001_v1` | 2 | 5001 |

The 2501- and 5001-bin campaigns both contain the sampled best-fit anchor
`(log10m_acc, log10f_acc)=(18,-1)` and the high-fraction convergence anchor
`(12, log10(0.3))`.  The existing 16-point campaign supplies the 1001-bin
best-fit bundle, so it is not recomputed.

Launch all four controllers concurrently inside one detachable session:

```bash
tmux new -s accdm_validation_suite

python -u campaigns/accdm/validation_suite.py watch \
  --stages theory fit
```

Detach with `Ctrl-b`, then `d`.  Check every campaign with:

```bash
python campaigns/accdm/validation_suite.py status
```

Each controller writes its own log to
`runs/accdm/<campaign>/controller/watch.log`; PBS theory and fit logs remain in
the standard campaign subdirectories.  A clean stop request for all four
controllers is:

```bash
python campaigns/accdm/validation_suite.py stop
```

Stopping controllers does not cancel already submitted PBS jobs.  Clear suite
stop markers before restarting the controllers:

```bash
python campaigns/accdm/validation_suite.py resume
```

## Resolution-adaptive production scan

The momentum-grid tests give two distinct numerical regimes.  At
`f_acc=0.1`, the 1001-, 2501-, and 5001-bin bundles agree at sub-per-mille
level below `2 h/Mpc`, and the fitted chi2 changes by only `0.016` between
2501 and 5001 bins.  At `f_acc=0.3`, changing 2501 to 5001 bins changes raw
channels by up to a few percent and changes the fitted chi2 by `-1.729`.
Cross-starting the two nuisance minima reproduces both target-theory minima,
so this difference is not an optimizer-basin artifact.

The production scan is therefore split into immutable pieces:

| campaign | log10 fraction range | points | daughter bins |
| --- | ---: | ---: | ---: |
| `accdm_scan_lowf_q1001_v2` | `-4.00 ... -1.00`, boundary-refined | 850 | 1001 |
| `accdm_qconvergence_f0p3_q10001_v1` | exact `log10(0.3)` pilot | 1 | 10001 |
| `accdm_scan_highf_q10001_v2` | `-0.925 ... 0.00`, boundary-refined | 600 | 10001 |

All grids use 50 masses from `log10m_acc=11` to `18` in steps of `1/7`, or
about `0.143` dex.  The fraction spacing is `0.075` dex from `-1.6` to
`-0.4`, around the expected constraint at `f_acc=0.1`, and is coarser in the
tails.  The low- and high-fraction coordinates are disjoint and combine into
1450 points.  This matches the logarithmic resolution of the successful DCDM
scan while assigning most expensive evaluations to the expected boundary.
The 10001-bin pilot is an additional resolution check at the exact fraction
used in the 1001/2501/5001 comparison.  The accepted production compromise is
5001 daughter momentum bins for the 600-point high-fraction grid, followed by
targeted 10001-bin validation near the fitted minimum and confidence contours.

Build every immutable directory and inspect the planned initial jobs:

```bash
python campaigns/accdm/adaptive_scan.py build
python campaigns/accdm/adaptive_scan.py status --group all
python campaigns/accdm/adaptive_scan.py dry-run --group initial
```

The default `initial` group contains the complete low-fraction scan and the
single 10001-bin pilot.  The initial group does not contain the 600-point
high-fraction scan:

```bash
tmux new -s accdm_adaptive_initial

python -u campaigns/accdm/adaptive_scan.py watch \
  --group initial \
  --stages theory fit
```

Detach with `Ctrl-b`, then `d`.  Monitor all three components from another
terminal:

```bash
python campaigns/accdm/adaptive_scan.py status --group all
```

The high-fraction campaign requests two cores and 4 GB per theory point.  Its
600 simultaneous theory jobs therefore use at most 1200 cores.  Fit jobs keep
one core each.  Run the two stages separately so fit jobs do not compete with
theory for the same user-level queue slots:

```bash
python campaigns/accdm/adaptive_scan.py dry-run --group high

tmux new -s accdm_adaptive_high

python -u campaigns/accdm/adaptive_scan.py watch \
  --group high \
  --stages theory
```

After all theory points are complete, run:

```bash
python -u campaigns/accdm/adaptive_scan.py watch \
  --group high \
  --stages fit
```

PBS resources and CLASS thread counts are read from the immutable campaign
configuration; editing the shared `cluster/theory_point.pbs` template is not
required.  To stop or resume controller submission without cancelling
existing PBS jobs, use:

```bash
python campaigns/accdm/adaptive_scan.py stop --group all
python campaigns/accdm/adaptive_scan.py resume --group all
```

## Exploratory high-fraction q=1001 scout

If the 5001-bin high-fraction campaign is too expensive to complete, build a
separate exploratory campaign on exactly the same 600 coordinates:

```bash
python campaigns/accdm/build_grid.py \
  --base-model campaigns/accdm/base_model.json \
  --grid-spec campaigns/accdm/grid_specs/scan_highf_scout_q1001.json \
  --run-dir runs/accdm/accdm_scan_highf_q1001_scout_v1
```

The scout uses 1001 daughter momentum bins, ten requested cores and ten CLASS
threads per theory point.  At most 120 theory jobs run concurrently, so the
campaign requests at most 1200 cores.  The theory stage permits only one
attempt per point: deterministic CLASS failures are recorded for targeted
recovery instead of being automatically repeated with identical settings.

Verify the resource request before submission:

```bash
python cluster/campaign_manager.py submit-theory \
  --campaign runs/accdm/accdm_scan_highf_q1001_scout_v1 \
  --dry-run --offline --max-points 2
```

Run theory alone in a detachable session:

```bash
tmux new -s accdm_highf_q1001_scout

python -u cluster/campaign_manager.py watch \
  --campaign runs/accdm/accdm_scan_highf_q1001_scout_v1 \
  --stages theory \
  --exit-when-complete
```

Only after the theory controller exits, run the fits:

```bash
python -u cluster/campaign_manager.py watch \
  --campaign runs/accdm/accdm_scan_highf_q1001_scout_v1 \
  --stages fit \
  --exit-when-complete
```

This campaign is a likelihood-topology scout, not a final numerical result.
When the maps are merged, completed 5001-bin points override matching 1001-bin
coordinates.  New 5001- or 10001-bin calculations should then target the
sampled minimum and the one- and two-sigma contour neighborhoods.

## Cross-fit and three-point minimum validation

Before interpreting a minimum found by the 1001-bin scout, separate momentum-
resolution effects from nuisance-optimizer effects.  The cross-refinement
script evaluates and locally refines both saved nuisance minima on both saved
theories.  It also includes a start that preserves the dimensional
counterterm amplitude when the theory normalization changes:

```bash
POINT=accdm_logm11p4285714285714_logfm0p55
Q1001=runs/accdm/accdm_scan_highf_q1001_scout_v1
Q5001=runs/accdm/accdm_scan_highf_q5001_v1

python scripts/cross_refine_fits.py \
  --left-label q1001 \
  --left-theory "$Q1001/theory/$POINT.npz" \
  --left-fit "$Q1001/fits/$POINT.json" \
  --right-label q5001 \
  --right-theory "$Q5001/theory/$POINT.npz" \
  --right-fit "$Q5001/fits/$POINT.json" \
  --data-dir data \
  --output "results/cross_refine_${POINT}_q1001_q5001.json"
```

If the best refined target chi2 is nearly equal to the stored target chi2,
the discrepancy is not a missed nuisance minimum.  A drop of order unity or
larger means that the target fit must be repaired before comparing momentum
resolutions.

The final anchor test contains exactly three 10001-bin points: the q=1001
scout minimum, the q=5001 overlap minimum at the same mass, and the previous
low-fraction minimum.  It requests 20 cores and 16 GB for each of the three
theory jobs:

```bash
python campaigns/accdm/build_grid.py \
  --base-model campaigns/accdm/base_models/base_model_q10001.json \
  --grid-spec campaigns/accdm/grid_specs/validate_minima_q10001.json \
  --run-dir runs/accdm/accdm_validate_minima_q10001_v1

python cluster/campaign_manager.py submit-theory \
  --campaign runs/accdm/accdm_validate_minima_q10001_v1 \
  --dry-run --offline --max-points 3

python -u cluster/campaign_manager.py watch \
  --campaign runs/accdm/accdm_validate_minima_q10001_v1 \
  --stages theory \
  --max-active 3
```

After all three theories are complete, stop the theory watcher with `Ctrl-C`
and run the fit stage:

```bash
python -u cluster/campaign_manager.py watch \
  --campaign runs/accdm/accdm_validate_minima_q10001_v1 \
  --stages fit \
  --max-active 3 \
  --exit-when-complete
```

## Spectrum-response diagnostic

The profile-likelihood map alone does not show whether a weak constraint is
caused by a small cosmological response or by nuisance-parameter degeneracy.
`scripts/compare_model_responses.py` separates these effects using already
saved theory bundles and fits.  It does not run CLASS or SPT.

For example, compare an accDM mass sequence at fixed fraction with LCDM:

```bash
python scripts/compare_model_responses.py \
  --reference-theory results/lcdm_2022_planck_precision_v3_3.npz \
  --reference-fit results/lcdm_2022_planck_precision_v3_3_fit.json \
  --data-dir data \
  --point 'm=1e11 GeV, f=0.05=runs/accdm/accdm_scan_lowf_q1001_v2/theory/accdm_logm11_logfm1p3.npz' \
  --point 'm=1e13 GeV, f=0.05=runs/accdm/accdm_scan_lowf_q1001_v2/theory/accdm_logm13_logfm1p3.npz' \
  --point 'low-f best=runs/accdm/accdm_scan_lowf_q1001_v2/theory/accdm_logm15p7142857142857_logfm1p3.npz' \
  --point 'm=1e18 GeV, f=0.05=runs/accdm/accdm_scan_lowf_q1001_v2/theory/accdm_logm18_logfm1p3.npz' \
  --output-dir runs/accdm/accdm_response_mass_scan_f0p05
```

Campaign fit JSON paths are inferred automatically from each theory path.  A
non-campaign theory file can be paired explicitly with repeated
`--point-fit 'LABEL=FIT_JSON'` arguments.  The output contains linear-spectrum
ratios, trusted one-loop channel ratios, P1D ratios with fixed nuisance
parameters, P1D ratios after independent profiling, and a machine-readable
CSV summary.  In the fixed-nuisance comparison, `alpha_ct` is rescaled to keep
the physical counterterm amplitude `i0_scale * alpha_ct` unchanged.

## Recovering the incomplete high-fraction q=5001 campaign

The recovery builder reads the final filesystem state of
`accdm_scan_highf_q5001_v1` and freezes three disjoint selections:

| role | CLASS settings | PBS request |
| --- | --- | --- |
| ordinary missing points | unchanged q=5001 exact hierarchy | 10 threads, 8 GB, 48 h; at most 120 jobs |
| one stiff pilot | unchanged q=5001 exact hierarchy | 20 threads, 16 GB, 48 h; one job |
| remaining stiff points | same as the successful pilot | 20 threads, 16 GB, 48 h |

A stiff point is selected only when its source `.err` log contains
`Step size too small`.  The special branch first tests whether the unchanged
model completes with a larger OpenMP allocation; it does not loosen a CLASS
tolerance or enable the fluid approximation.  Other incomplete points retain
the same physical and numerical model.  Stop or let every old source job
finish before freezing the selections, then run:

```bash
python campaigns/accdm/prepare_highf_recovery.py
```

The command prints the exact counts and creates only non-empty campaigns.
It freezes each selection in `frozen_recovery_grid.json`; rerunning the command
reuses that file rather than silently changing an existing campaign.

Inspect the main resource request and launch its theory stage:

```bash
MAIN=runs/accdm/accdm_scan_highf_q5001_recovery_10core_v1

python cluster/campaign_manager.py submit-theory \
  --campaign "$MAIN" \
  --dry-run --offline --max-points 2

tmux new -s accdm_highf_q5001_recovery
python -u cluster/campaign_manager.py watch \
  --campaign "$MAIN" \
  --stages theory \
  --exit-when-complete
```

The ordinary recovery may run concurrently with the stiff pilot.  Launch only
the pilot first from the special branch:

```bash
PILOT=runs/accdm/accdm_highf_q5001_20core_pilot_v2

python cluster/campaign_manager.py submit-theory \
  --campaign "$PILOT" \
  --dry-run --offline --max-points 1

python -u cluster/campaign_manager.py watch \
  --campaign "$PILOT" \
  --stages theory \
  --exit-when-complete
```

Only if this pilot completes with an empty `.err` log, submit the remaining
special points:

```bash
STEP=runs/accdm/accdm_highf_q5001_20core_recovery_v2

python -u cluster/campaign_manager.py watch \
  --campaign "$STEP" \
  --stages theory \
  --exit-when-complete
```

If the builder reports no remaining stiff points, the `STEP` directory is
intentionally absent.  Run fits separately after each theory campaign is
complete, for example:

```bash
python -u cluster/campaign_manager.py watch \
  --campaign "$MAIN" \
  --stages fit \
  --exit-when-complete
```

The command-line `walltime` request is campaign-specific and overrides the
24-hour default in the shared PBS template.  If the batch queue rejects a
48-hour request, use a queue that permits it; shortening the request would
recreate the original failure mode.
