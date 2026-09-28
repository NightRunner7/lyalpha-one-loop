# accDM DCDM-resolution production-grid update 3.9.1

This update replaces the provisional uniform production axes with a grid that
matches the logarithmic resolution of the successful DCDM reconstruction and
concentrates fraction samples around the expected `f_acc=0.1` boundary.

The production coordinate set contains 1450 disjoint points:

- 50 masses over `11<=log10m_acc<=18`, spaced by `1/7` dex;
- 17 fractions at `f_acc<=0.1`, giving 850 points at 1001 momentum bins;
- 12 fractions at `f_acc>0.1`, giving 600 points at 10001 momentum bins;
- fraction spacing `0.075` dex over `-1.6<=log10f_acc<=-0.4` and coarser tails.

One additional 10001-bin pilot remains at `log10m_acc=12`, `f_acc=0.3`.
The default initial group launches the complete 850-point fast scan together
with this pilot, capped at 200 simultaneous theory jobs.  The 600-point
high-fraction campaign remains behind an explicit launch gate.

Campaign IDs use `v2` for both production pieces so an already-built v3.9.0
run directory is never overwritten.  Physical conventions and the exact
Boltzmann hierarchy are unchanged.
