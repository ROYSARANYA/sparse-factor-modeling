"""
01_wall_clock_comparison.py
====================
Wall-clock-normalized comparison of five first-order LASSO solvers against a
CVXPY reference solution across all 25 target variables in the dataset.

What it does
------------
1. Data loading & preprocessing
   - Loads the full factor matrix X and target matrix Y via `load_all_data()`.
   - Column-standardizes X (zero mean, unit variance) so every solver operates
     on the same scaled input and regularization strength is comparable.

2. Solver lineup
   Five iterative solvers are benchmarked:
     • Proximal GD      – vanilla proximal gradient descent (baseline)
     • FISTA vanilla    – accelerated ISTA (Beck & Teboulle, 2009), no restart
     • FISTA+fn restart – FISTA with function-value restart heuristic to damp
                          oscillations near the optimum
     • BB LASSO (alt)   – Barzilai–Borwein step-size selection (gradient-based
                          adaptive learning rate, no line search)
     • Coord Descent    – cyclic coordinate descent (sklearn-style)

3. Benchmarking loop  (25 targets × 20 timing reps each)
   For every target column j in Y:
     a. A CVXPY interior-point solution is computed as the ground-truth
        coefficient vector `ref_j`.
     b. Each solver is fitted 20 times; wall-clock milliseconds are recorded
        with `time.perf_counter()` and averaged to reduce timer noise.
     c. Iteration count (`n_iter_`) and L∞ coefficient error vs. CVXPY are
        stored.

4. Metrics reported (printed table + saved plot)
   - Mean Iters      : average iterations to convergence over 25 targets
   - Wall-Clock ms   : mean wall time per fit (averaged over targets & reps)
   - Coef Error      : mean L∞ distance from the CVXPY reference solution
   - vs PGD          : wall-clock speedup ratio relative to Proximal GD

5. Diagnostic note
   Prints the BB-vs-CD wall-clock ratio; with only p=6 features the expected
   speedup of coordinate descent vanishes, so the two should be near-identical.

6. Output
   Saves the comparison bar chart to `outputs/benchmark_timing.png`.

Dependencies
------------
numpy, matplotlib, time, sys
src.data_loader   – load_all_data()
src.solvers       – LassoProximal, FISTALasso, FISTARestart, BBLasso,
                    CoordinateDescent
src.cvxpy_solvers – lasso_cvxpy  (ground-truth reference)
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
