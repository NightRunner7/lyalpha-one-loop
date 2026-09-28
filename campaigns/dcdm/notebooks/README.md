# DCDM notebooks

Keep only model-specific exploration and final plots here.  Theory generation,
P1D projection and nuisance fitting must continue to use the shared
`lyalpha_pt/` and `scripts/` implementation.

`01_inspect_campaign.ipynb` summarizes status, ranks completed one-loop fits,
compares them with the LCDM reference, plots the sampled likelihood surface,
and exposes nuisance/boundary diagnostics at the sampled minimum.  When the
active fits were produced by the neighbour-refit campaign stage, it also
reports which target or adjacent point supplied the selected starting vector
and the improvement obtained by local refinement.

The recommended analysis input is a completed run directory under
`runs/dcdm/<campaign_id>/`, especially `manifest.csv`, `theory/*.npz` and
`fits/*.json`.  Do not place large generated grids in this source directory.
