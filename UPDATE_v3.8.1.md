# Update v3.8.1: accDM convention correction

This patch keeps the accDM radiation convention used by the working custom
CLASS implementation:

```text
N_ur = 2.0308
```

The accDM validation test now protects this exact value.  The LCDM and DCDM
benchmark presets are unchanged.  The default cluster environment name in the
accDM validation specification is also corrected to `class_accdm`.

All other v3.8.0 accDM settings remain unchanged: one 0.06 eV massive
neutrino, two effectively massless species, the accDM daughter as the second
NCDM entry, and `ncdm_fluid_approximation=3` for the exact hierarchy.
