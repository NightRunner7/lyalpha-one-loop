# accDM adaptive production-scan update 3.9.0

This update converts the validated momentum-resolution result into a gated,
resolution-adaptive production scan.

The final coordinate set contains 1476 disjoint points:

- 576 points at `f_acc<=0.1`, using the validated 1001-bin exact hierarchy;
- 900 denser points at `0.1<f_acc<=1`, configured for 10001 bins;
- one additional 10001-bin pilot at `log10m_acc=12`, `f_acc=0.3`.

The pilot and the entire low-fraction scan can run concurrently.  The
high-fraction campaign is built at the same time but is excluded from the
default launch group.  This prevents 900 expensive submissions before the
pilot confirms that the exact hierarchy completes within the PBS memory and
24-hour wall-time limits.

No fluid approximation is enabled.  All physical conventions from v3.8.2 are
unchanged: `N_ur=2.0308`, one 0.06 eV massive neutrino, fixed early CDM
density, `kappa_acc=12.1`, and `a_t_acc=0.133`.
