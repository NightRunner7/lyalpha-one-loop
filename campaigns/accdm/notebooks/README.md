# accDM notebooks

Keep only accDM-specific inspection, validation, and final plots here.  CLASS,
SPT, the P1D projection, and nuisance fitting must continue to use the shared
project implementation.

`01_inspect_campaign.ipynb` reads an existing run directory and summarizes
completion, the sampled profile likelihood, linear-to-one-loop improvement,
and nuisance-bound diagnostics.  It does not submit jobs or recompute theory.
