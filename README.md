# BOSS Ly-alpha one-loop reconstruction

**Eagle / Slurm:** see [the Polish migration guide](docs/EAGLE_PL.md)
for the `accDM_refactor` birth-grid profile (strategy 5, automatic bin count,
`m_nu`, background_Nloga=40000), environment-aware campaign preparation and a
one-point smoke test. Existing PBS campaigns remain supported.

This project separates the calculation into three independent stages:

1. `lyalpha_pt/models.py` contains the only model-dependent CLASS block.
2. `scripts/generate_theory.py` runs CLASS and direct EdS one-loop SPT on the
   cluster and writes one self-validating `.npz` theory bundle.
3. `scripts/fit_p1d.py` reads that bundle and performs the P1D projection and
   six-parameter fit locally, without importing or running CLASS.

Model-specific grid and validation workflows live under `campaigns/`.  The
accDM follow-up suite can build and monitor its lower-mass and momentum-grid
checks concurrently through `campaigns/accdm/validation_suite.py`.
The resolution-adaptive production grid is controlled separately through
`campaigns/accdm/adaptive_scan.py`; its default launch group deliberately
excludes the expensive high-fraction bulk grid until the 10001-bin pilot has
finished.

The public `Pk1D_data.dat`, `Pk1D_syst.dat`, and `Pk1D_cor.dat` files supplied
with this project are validated as 13 redshift blocks times 35 k bins.  The
fiducial fit selects 245 points at z=3.0,...,4.2.

## Status of the reconstruction

The complete linear end-to-end benchmark gives

```text
lcdm_2021_massless, paper_diag, k_uv=20 h/Mpc
chi2_linear = 206.470438
```

The published value is 206.5.  This agreement checks the data selection,
units, 3D-to-1D projection, fixed SiIII and smoothing factors, covariance,
counterterm, redshift parameterization, and optimizer.

The committed `smoke` one-loop bundle has only 28 k points over
1e-3--20 h/Mpc.  It is an end-to-end software check and **must not** be used
as a scientific one-loop result.  The publication targets are:

| preset | linear target | one-loop target |
| --- | ---: | ---: |
| `lcdm_2021_massless` | 206.5 | 193.4 |
| `lcdm_2022_planck` | about 206.3 from Delta chi2=-13.4 | 192.89 |

The v3.1 precision calculation (160 output k values, internal q_max=100
h/Mpc) has now been completed with the closed P13 and symmetry-reduced P22.
The raw loop sum at z=3 agrees with the independent FFTLog implementation in
FAST-PT 4.0.0 to better than 6.6e-4 relative to the linear spectrum for
k<=2 h/Mpc in all three channels.  Independent full-domain P22 and recursive
F3/G3 point checks give maximum differences of 1.2e-4 and 6.8e-4,
respectively.

The corrected staged optimizer gives

```text
lcdm_2021_massless, paper_diag, k_uv=20 h/Mpc
chi2_linear   = 206.470438
chi2_one_loop = 192.896588
Delta chi2    = -13.573851
```

The formerly reported one-loop values above 214--233 were missed-minimum
artifacts: the expanded counterterm interval had been searched globally from
the start and a later Powell step was accepted even when it made chi2 worse.
The fitter now searches the base box first, retains every better candidate,
and reaches a requested cutoff through continuation from 10 to 15 to 20
h/Mpc.  The remaining cutoff results are 199.0021, 193.2937 and 192.8966 at
10, 15 and 20 h/Mpc.  Thus 15--20 is stable at Delta chi2=0.40, while the
10 h/Mpc endpoint remains a deliberately visible sensitivity test.

## LCDM 2022 paper benchmark

The preset `lcdm_2022_planck` reproduces Table 1 of arXiv:2210.06117:

```text
h       = 0.671095
Omega_m = 0.317549
sigma8  = 0.809964
S8      = 0.833317
```

It uses one 0.06 eV massive neutrino, two massless species, and the massive
neutrino loop prescription
`P_total_linear + (1-f_nu)^2 (P22+P13)[P_cb]`.  For the precision v3.3 bundle,
`paper_diag` covariance, and `k_uv=20 h/Mpc`, the reconstructed values are

```text
                         reconstruction     paper
chi2_linear                 206.248          about 206.29
chi2_one_loop               193.197          192.89
Delta chi2                  -13.051          -13.4
```

The residual difference is 0.307 in the absolute one-loop chi2 and 0.349 in
Delta chi2.  It is kept as an explicit numerical reconstruction budget rather
than removed by tuning nuisance parameters.  Independent FAST-PT checks of
the raw loop corrections agree within `6.1e-4` of the tree spectrum for
`k<=2 h/Mpc`.

The complete local audit, tables, and publication-style figures are in
`notebooks/03_validate_lcdm_2022_planck.ipynb`.  Install its optional tools
with

```bash
python -m pip install -e '.[notebook]'
jupyter lab notebooks/03_validate_lcdm_2022_planck.ipynb
```

The notebook normally reads the committed fit result.  Set
`RUN_NEW_FITS=True` in its configuration cell to repeat the multi-start fit.

For a model-independent workflow, use the Polish guide
`docs/INSTRUKCJA_UZYTKOWNIKA_PL.md` and
`notebooks/04_analyze_any_model.ipynb`.  The notebook has no hard-coded paper
target: changing its theory and fit paths is sufficient for LCDM, DCDM or
accelerated-DM bundles.  A compact consistency report is available with

```bash
python scripts/inspect_results.py \
  --theory results/my_model_precision.npz \
  --fit results/my_model_fit.json \
  --cutoff-scan results/my_model_cutoff_scan.json
```

## Corrections relative to the legacy file

- A cache/checkpoint key contains the full model, CLASS input, numerical
  configuration, redshift, k grid, and the actual total and loop-source
  linear spectra.  Every final bundle also has a SHA-256 digest checked on
  load.
- Production P13 uses stable closed one-dimensional EdS expressions for all
  three channels.  The regulated F3/G3 recursion with Richardson
  extrapolation is retained only as an independent regression test.  The
  bundle stores a conservative fine-minus-coarse radial quadrature error.
- P22 uses the exact q--p exchange symmetry and integrates only the
  half-domain p>=q.  This removes the non-analytic p->0 point at q=k that
  caused a large odd/even-grid error after the P22/P13 cancellation.
- The massive-neutrino prescription is
  `P_XY = P_total_linear + (1-f_nu)^2(P22_XY+P13_XY)` with the loop generated
  from `P_cb`, as in Eq. (16) of arXiv:2011.03050.
- `k_trust` (2 h/Mpc), the loop integration range, and the technical P1D UV
  cutoff (tested at 10--20 h/Mpc) are distinct settings.
- Tree level, P22, P13, their extrapolation error, H(z), conversion factors,
  and both total and loop-source linear spectra are stored separately.
- Local fitting never starts CLASS.  It reports normalized distance to every
  numerical bound and supports several seeds and cutoff scans.

The fitter profiles the two exactly linear amplitude combinations by default,
leaving a four-dimensional global search.  It separately searches both
counterterm faces and never replaces a candidate by a worse local result.
Use `--full-six-dimensional` as an independent optimizer cross-check.

`legacy/legacy_monolithic_reference.py` is the untouched input implementation
for regression comparisons.

## Installation

The local fit needs only NumPy and SciPy:

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

The independent raw-loop validation additionally needs FAST-PT and
Matplotlib:

```bash
python -m pip install -e '.[validation]'
```

Theory generation additionally needs the CLASS wrapper used for the chosen
cosmological model.  On a cluster, activate the already working CLASS/DCDM/
accDM environment.  For stock CLASS, a source build can be installed with:

```bash
CC=gcc CXX=g++ python -m pip install 'classy==3.3.4.0'
```

Do not replace a custom DCDM or accDM CLASS wrapper with stock `classy`.

## Resumable model grids

The validated generator and fitter are also exposed through one reusable PBS
campaign layer. Model-specific code emits complete `ModelSpec` JSON files;
the shared cluster jobs still call only `scripts/generate_theory.py` and
`scripts/fit_p1d.py`.

The DCDM campaign includes three prepared grid specifications:

| specification | points | purpose |
| --- | ---: | --- |
| `validation_grid.json` | 10 | production patch around tau=40 Gyr, epsilon=0.006 |
| `precision_anchors.json` | 2 | precision best-fit and LCDM-like anchors |
| `paper_grid.json` | 644 | full 23 x 28 grid listed in arXiv:2210.06117 |

Start with the small campaign:

```bash
python campaigns/dcdm/build_grid.py \
  --grid-spec campaigns/dcdm/grid_specs/validation_grid.json \
  --run-dir runs/dcdm/dcdm_validation_v1

python cluster/campaign_manager.py submit-theory \
  --campaign runs/dcdm/dcdm_validation_v1 \
  --dry-run --offline --max-points 2
```

The complete directory layout, PBS configuration, live queue refill,
retry behavior and refinement workflow are documented in
`cluster/README.md`.

An accelerating-DM campaign layer is available under `campaigns/accdm/`.
It fixes `kappa_acc=12.1` and `a_t_acc=0.133`, scans mass and fraction, uses
one 0.06 eV massive plus two massless neutrinos, and enforces the exact NCDM
Boltzmann hierarchy.  Start with its 16-point validation grid; the physical
mapping, fixed-early-density convention, cluster commands, and acceptance
sequence are documented in `campaigns/accdm/README.md`.

Completed two-dimensional grids support a final neighbour-seeded optimizer
audit.  It tests the target's own saved nuisance minimum and those of up to
eight adjacent grid points against the target theory, locally refines all of
them, and keeps the lowest actual target chi2.  This is a minimization check,
not interpolation or smoothing.  Prepare it with
`campaign_manager.py prepare-neighbor-refit`; the full procedure is in
`cluster/README.md`.

## Step 1: generate theory on the cluster

Quick software check:

```bash
python scripts/generate_theory.py \
  --preset lcdm_2021_massless \
  --quality smoke \
  --output results/lcdm_2021_massless_smoke.npz
```

Publication benchmark:

```bash
python scripts/generate_theory.py \
  --preset lcdm_2021_massless \
  --quality precision \
  --output results/lcdm_2021_massless_precision.npz \
  --checkpoint-dir results/checkpoints_lcdm_2021
```

If a job stops, submit the identical command again.  Completed redshift
checkpoints are reused only after their complete keys match.  The PBS template
is `scripts/run_theory_pbs.sh`.

Available numerical profiles:

- `smoke`: execution and I/O test only;
- `production`: first converged physics run;
- `precision`: reference run for the paper benchmark.

The one-loop result is accepted only after `production` and `precision` agree
within the chosen numerical tolerance on k<=2 h/Mpc and yield stable fitted
chi2 values.

Compare their raw channels before fitting:

```bash
python scripts/compare_theories.py \
  results/lcdm_2021_massless_production.npz \
  results/lcdm_2021_massless_precision.npz \
  --k-max 2
```

Run the independent raw-channel validation before interpreting a fit:

```bash
python scripts/validate_raw_spt.py \
  --theory results/lcdm_2021_massless_precision_v3_1.npz \
  --compare-theory results/lcdm_2021_massless_production.npz \
  --redshift 3 \
  --output-dir results/raw_spt_validation
```

This performs four different checks without CLASS or likelihood fitting:

- exact bundle identities and positivity diagnostics;
- direct full-domain `(q,p)` P22 quadrature versus the production
  symmetry-reduced integral;
- regulated recursive F3/G3 P13 versus the closed one-dimensional P13;
- all three physical loop sums versus independent FAST-PT/FFTLog.

FAST-PT rearranges P22 and P13 by IR regularization, so only its physical
sum is compared.  The script writes JSON/CSV data and three diagnostic plots;
it does not impose a pass/fail threshold.

Targeted density--theta diagnostic (before rerunning full precision):

```bash
python scripts/validate_dtheta.py \
  --theory results/lcdm_2021_massless_production.npz \
  --redshift 3 \
  --q-maxes 50 75 100 150 \
  --jobs 4 \
  --output-dir results/dtheta_validation_z3
```

This recomputes only selected values of
`P22_dtheta`, `P13_dtheta`, and their sum.  It scans the loop cutoff and the
P22/P13 quadratures, varies the P13 regulator, and compares the regulated F3/G3
recursion with independent closed one-dimensional P13 expressions.  If the
requested q range exceeds the stored linear grid, the default `auto` mode uses
the CLASS input embedded in the bundle to extend only the linear spectrum.
No acceptance threshold is imposed by the script; it writes raw CSV tables,
summary statistics, and diagnostic figures for inspection.

To isolate the P13 correction without rerunning CLASS or changing the stored
P22 values, refine an existing bundle into a new file:

```bash
python scripts/refine_p13.py \
  --theory results/lcdm_2021_massless_production.npz \
  --n-q 1001 \
  --output results/lcdm_2021_massless_production_p13_closed.npz
```

The original file is never overwritten, and the refined metadata records its
parent digest.  A full fresh `production` or `precision` generation uses the
same closed P13 method by default.

Legacy bundles can likewise be corrected without CLASS by recomputing P22:

```bash
python scripts/refine_p22.py \
  --theory results/lcdm_2021_massless_production_p13_closed.npz \
  --output results/lcdm_2021_massless_production_corrected.npz
```

The default point counts and loop limits are read from the parent bundle, but
the integration domain is replaced by the symmetry-reduced one.

## Step 2: fit locally

Copy only the generated `.npz` file to the local computer.  CLASS and the
checkpoint directory are not needed.

```bash
python scripts/fit_p1d.py \
  --theory results/lcdm_2021_massless_precision_v3_1.npz \
  --data-dir data \
  --mode both \
  --k-uv-cut 20 \
  --seeds 12345 23456 34567 \
  --expanded-counterterm \
  --output results/lcdm_2021_massless_fit.json
```

`--expanded-counterterm` is safe here: it no longer starts a global search in
the enormous box.  The base box is fitted first and then continued into the
wider box.  For a requested cutoff above 10 h/Mpc the script also follows the
default ladder 10 -> 15 -> requested cutoff, which is needed to find the
narrow k_uv=20 basin reproducibly.  Use `--no-cutoff-continuation` only as an
optimizer diagnostic.

Cutoff test:

```bash
python scripts/scan_cutoffs.py \
  --theory results/lcdm_2021_massless_precision_v3_1.npz \
  --data-dir data \
  --cutoffs 10 15 20 \
  --expanded-counterterm \
  --output results/lcdm_2021_cutoff_scan.json
```

The nuisance parameters, especially the counterterm, may move with the cutoff.
The important stability criterion is the minimum chi2 and cosmological profile.

## Changing LCDM to DCDM or accelerated DM

There are two equivalent routes.

1. Add one `ModelSpec` to the `PRESETS` dictionary in
   `lyalpha_pt/models.py`.
2. Copy `examples/custom_lcdm_model.json`, change its `class_params`, and run
   `generate_theory.py --model-json your_model.json ...`.

For DCDM, replace `omega_cdm` with the initial DCDM density parameter expected
by that CLASS branch; do not leave both parameters in the dictionary.  For
accelerated DM, add its CLASS parameters in the same JSON block.  No SPT,
projection, covariance, or fit code should be edited.

Choose the loop prescription explicitly:

- `"loop_source": "total", "loop_weight": 1.0` for a direct EdS loop of the
  model's total linear matter spectrum;
- `"loop_source": "cb", "loop_weight": "one_minus_fnu_squared"` for the
  massive-neutrino prescription used in the 2021 paper.

The exact choice for a new model is part of its physical definition and must be
recorded in the `ModelSpec`; it is also embedded in the output metadata.

## Acceptance sequence

1. All unit tests pass.
2. Linear `lcdm_2021_massless` reproduces chi2=206.5 (already passed).
3. Precision one-loop `lcdm_2021_massless` reproduces chi2 about 193.4 and
   Delta chi2 about -13.1 (passed: 192.8966 and -13.5739).
4. The raw channels agree with direct quadrature and independent FAST-PT
   (passed for k<=2 h/Mpc).
5. Understand or profile the residual k_uv=10 sensitivity; k_uv=15--20 is
   already stable within Delta chi2=0.40.
6. `lcdm_2022_planck` reproduces the paper benchmark (passed: 193.1966 versus
   192.89, with Delta chi2=-13.0515 versus -13.4).
7. Only then generate DCDM and accelerated-DM grids.
