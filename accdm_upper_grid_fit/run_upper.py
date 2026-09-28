#!/usr/bin/env python3
"""Run Stage A profiling; set numerical threads before importing NumPy/SciPy."""
import os
import sys

sys.dont_write_bytecode = True
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'

if __name__ == '__main__':
    from accdm_upper_grid.runner import main
    raise SystemExit(main())
