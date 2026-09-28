# accDM high-fraction recovery update 3.9.3

This update adds a frozen recovery workflow for the incomplete q=5001
high-fraction scan.  Completed source points are not recomputed.  Ordinary
missing points use ten CLASS threads, 8 GB, and a campaign-level 48-hour PBS
walltime with at most 120 simultaneous jobs.

Points whose source log contains `Step size too small` are isolated from the
main recovery.  One pilot and the remaining stiff points use the unchanged
q=5001 exact hierarchy plus `smallest_allowed_variation=1e-14`, twenty threads,
16 GB, and 48 hours.  The remaining special campaign is launched only after
the pilot succeeds.

The shared campaign manager now supports validated `theory_walltime` and
`fit_walltime` fields without changing existing campaigns that omit them.

New accDM theory bundles record the background-only consistency ratio
`eta*m_acc/[rho_crit(a_t)*H(a_t)^-3]` in their metadata and print it to the PBS
output.  Evaluating this condition does not require or modify perturbations.
