# DCDM campaign definition

This directory is the only DCDM-specific layer of the grid workflow.

`base_model.json` reproduces the tested two-body-decay setup for the
`merging_with_master` branch of `class_decays`. The stable CDM component is
kept at `omega_cdm=1e-5`, while the decaying initial density is specified by
`omega_ini_dcdm2=0.12`. A standard 0.06 eV neutrino and the massive daughter
are represented as two NCDM species.

For every grid point, `build_grid.py` changes only

```text
Gamma_dcdm = 977.7922216807891 / tau_Gyr  [km/s/Mpc]
epsilon_dcdm
```

and writes a complete, immutable `ModelSpec` JSON. DCDM loops use the total
linear matter spectrum with unit loop weight, matching the reconstruction of
arXiv:2210.06117.

The validated reference point is

```text
tau = 40 Gyr, epsilon = 0.006
production chi2_one_loop = 189.906473
precision  chi2_one_loop = 189.887294
published best fit       = 189.77
```

The production-to-precision change is 0.0192 in chi2. Relative raw-channel
RMS differences below 2 h/Mpc are approximately 5e-4 to 6e-4. These values
are reference diagnostics, not hard-coded acceptance gates.

For the full paper grid, use a direct `k_uv=20` fit as the first pass and the
shared neighbour-seeded campaign refit as the final optimizer audit.  The
second pass imports nuisance vectors from the target and up to eight adjacent
grid points, but evaluates and refines every vector on the target theory.
This removes isolated missed minima without smoothing the physical likelihood
surface.  See `cluster/README.md`, section "Robust second-pass fit of a
completed grid", for the exact archival and submission commands.
