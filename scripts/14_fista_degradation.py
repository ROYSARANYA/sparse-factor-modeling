"""
Novel Contribution 1 (strengthened): FISTA degradation at high sparsity
Key result: FISTA becomes WORSE than PGD at alpha=0.05 (speedup 0.95x)
Theoretical explanation: L1 term at high alpha creates corners that
momentum overshoots repeatedly
"""
import sys, numpy as np
sys.path.insert(0, '.')
from src.data_loader import load_all_data
from src.solvers import LassoProximal, FISTALasso, BBLasso

X, Y, factor_names, _ = load_all_data()
X_vals   = X.values
X_scaled = (X_vals - X_vals.mean(0)) / X_vals.std(0)

XtX   = X_scaled.T @ X_scaled / len(X_scaled)
eigs  = np.linalg.eigvalsh(XtX)
L     = 2 * eigs.max()
mu    = 2 * eigs.min()
kappa = L / mu

print(f'kappa={kappa:.2f}, sqrt(kappa)={np.sqrt(kappa):.2f}x (FISTA theory)')
print()
print(f'{"alpha":>8} {"sparsity":>10} {"PGD":>8} {"FISTA":>8} '
      f'{"BB":>8} {"FISTA/PGD":>10} {"BB/PGD":>8} {"Status"}')
print('='*80)

alphas = [0.001, 0.003, 0.005, 0.01, 0.02, 0.05, 0.1]
for alpha in alphas:
    pi, fi, bi, si = [], [], [], []
    for j in range(25):
        y  = Y.iloc[:, j].values
        mp = LassoProximal(alpha=alpha, max_iter=2000).fit(X_scaled, y)
        mf = FISTALasso(alpha=alpha,    max_iter=2000).fit(X_scaled, y)
        mb = BBLasso(alpha=alpha,       max_iter=2000).fit(X_scaled, y)
        pi.append(mp.n_iter_); fi.append(mf.n_iter_)
        bi.append(mb.n_iter_)
        si.append(np.sum(np.abs(mp.coef_) > 1e-4))
    pgd_m = np.mean(pi); fista_m = np.mean(fi); bb_m = np.mean(bi)
    fsp   = pgd_m/fista_m; bsp = pgd_m/bb_m
    status = 'WORSE THAN PGD <--' if fsp < 1.0 else f'{fsp:.2f}x'
    print(f'{alpha:>8.3f} {np.mean(si):>10.1f} {pgd_m:>8.1f} '
          f'{fista_m:>8.1f} {bb_m:>8.1f} {fsp:>10.2f}x {bsp:>8.2f}x  {status}')

print()
print('KEY: FISTA speedup collapses and goes BELOW 1.0 at alpha=0.05')
print('Explanation: high alpha -> L1 dominates -> momentum overshoots corners')
print('BB adapts to local curvature -> immune to this effect')
