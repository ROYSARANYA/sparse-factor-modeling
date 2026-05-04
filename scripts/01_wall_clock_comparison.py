"""
Fix 2: Wall-clock normalized algorithm comparison
Generates: outputs/benchmark_timing.png
"""
import sys, numpy as np, time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '.')
from src.data_loader import load_all_data
from src.solvers import (LassoProximal, FISTALasso, FISTARestart,
                         BBLasso, CoordinateDescent)
from src.cvxpy_solvers import lasso_cvxpy

X, Y, factor_names, _ = load_all_data()
X_vals   = X.values
X_scaled = (X_vals - X_vals.mean(0)) / X_vals.std(0)
alpha    = 0.003

methods = [
    ('Proximal GD',       LassoProximal,  {}),
    ('FISTA vanilla',     FISTALasso,     {}),
    ('FISTA+fn restart',  FISTARestart,   {'restart': 'function'}),
    ('BB LASSO (alt)',    BBLasso,        {}),
    ('Coord Descent',     CoordinateDescent, {}),
]

print(f'{"Method":<26} {"Mean Iters":>11} {"Wall-Clock ms":>14} {"Coef Error":>12} {"vs PGD"}')
print('='*72)

pgd_ms  = None
results = []
for name, cls, kwargs in methods:
    iters_all, ms_all, err_all = [], [], []
    for j in range(25):
        yj    = Y.iloc[:, j].values
        ref_j = lasso_cvxpy(X_scaled, yj, alpha)
        runs  = []
        for _ in range(20):
            t0 = time.perf_counter()
            m  = cls(alpha=alpha, max_iter=500, **kwargs).fit(X_scaled, yj)
            runs.append((time.perf_counter()-t0)*1000)
        iters_all.append(m.n_iter_ if m.n_iter_ > 0 else 500)
        ms_all.append(np.mean(runs))
        if ref_j is not None:
            err_all.append(np.max(np.abs(m.coef_ - ref_j)))
    mi = np.mean(iters_all); mt = np.mean(ms_all); md = np.mean(err_all)
    results.append((name, mi, mt, md))
    if name == 'Proximal GD': pgd_ms = mt
    ratio = f'{pgd_ms/mt:.2f}x' if pgd_ms else '—'
    print(f'{name:<26} {mi:>11.1f} {mt:>14.3f} {md:>12.2e}  {ratio}')

cd_ms = [r[2] for r in results if 'Coord' in r[0]][0]
bb_ms = [r[2] for r in results if 'BB' in r[0]][0]
print(f'\nCD vs BB wall-clock ratio: {bb_ms/cd_ms:.2f}x (near-identical at p=6)')
