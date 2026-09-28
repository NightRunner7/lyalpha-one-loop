# Update v3.8.0: exact-hierarchy accDM campaigns

This update adds the first accelerating-dark-matter campaign layer without
changing the validated CLASS/SPT generator, P1D projection, covariance, or
nuisance objective.

## Physical settings

- Scan axes: `log10m_acc` and `log10f_acc`.
- Mapping: `m_acc_in_GeV=10**log10m_acc`, `f_acc=10**log10f_acc`, and
  `eta_acc=1e11/m_acc_in_GeV`.
- Fixed transition: `kappa_acc=12.1`, `a_t_acc=0.133`.
- Exact NCDM hierarchy: `ncdm_fluid_approximation=3`.
- Neutrinos: two massless plus one 0.06 eV massive species.
- The second NCDM entry is the accDM daughter.
- The stable `omega_cdm` input is recalculated point by point so that the
  pre-acceleration total CDM density remains `0.12010`.
- One-loop source: total matter with unit loop weight.

## New files

- `campaigns/accdm/__init__.py`
- `campaigns/accdm/base_model.json`
- `campaigns/accdm/build_grid.py`
- `campaigns/accdm/grid_specs/validation_grid.json`
- `campaigns/accdm/README.md`
- `campaigns/accdm/notebooks/README.md`
- `campaigns/accdm/notebooks/01_inspect_campaign.ipynb`
- `tests/test_accdm_campaign_grid.py`

## Updated shared files

- `lyalpha_pt/campaign_refit.py`: reads arbitrary campaign axes from
  `campaign.json`; existing DCDM manifests remain backward compatible.
- `campaigns/dcdm/build_grid.py`: records its coordinate names explicitly.
- `tests/test_campaign_refit.py`: covers model-independent accDM neighbours.
- `tests/test_campaign_grid.py`: covers DCDM axis metadata.
- `pyproject.toml`: registers the accDM package and
  `lya-build-accdm-grid` entry point.
- `README.md`: links the accDM workflow.

## Validation

All 35 unit and benchmark tests pass.  A complete dry run builds 16 immutable
models, reports 16 pending theory jobs and 16 fits waiting for theory, and
constructs valid PBS submissions using the configured `accdm_class`
environment.

No scientific CLASS calculation was performed in the packaging environment;
the custom `accDM_refactor` CLASS wrapper remains a cluster-side dependency.
