"""
Novel Contribution 1a: Momentum-BB algorithm
Generates: outputs/novelty_momentum_bb.png and outputs/momentum_bb_convergence.png
"""
import sys, numpy as np, time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '.')
from src.data_loader import load_all_data
from src.solvers import (LassoProximal, FISTALasso, FISTARestart,
                         BBLasso, MomentumBBLasso)
from src.cvxpy_solvers import lasso_cvxpy

X, Y, factor_names, _ = load_all_data()
X_vals   = X.values
X_scaled = (X_vals - X_vals.mean(0)) / X_vals.std(0)
alpha    = 0.003

methods = [
    ('Proximal GD',        LassoProximal,   {}, '#E74C3C'),
    ('FISTA vanilla',      FISTALasso,      {}, '#2E74B5'),
    ('FISTA+fn restart',   FISTARestart,    {'restart':'function'}, '#27AE60'),
    ('BB LASSO (alt)',     BBLasso,         {}, '#8E44AD'),
    ('Momentum-BB (novel)',MomentumBBLasso, {}, '#FF6B00'),
]

print(f'{"Method":<26} {"Mean Iters":>11} {"ms":>8} {"vs PGD"}')
print('='*55)
pgd_ms = None; results = []
for name, cls, kwargs, color in methods:
    iters_all, ms_all = [], []
    for j in range(25):
        yj   = Y.iloc[:, j].values
        runs = []
        for _ in range(10):
            t0 = time.perf_counter()
            m  = cls(alpha=alpha, max_iter=500, **kwargs).fit(X_scaled, yj)
            runs.append((time.perf_counter()-t0)*1000)
        iters_all.append(m.n_iter_ if m.n_iter_ > 0 else 500)
        ms_all.append(np.mean(runs))
    mi = np.mean(iters_all); mt = np.mean(ms_all)
    results.append({'name':name,'iters':mi,'ms':mt,'color':color,'cls':cls,'kwargs':kwargs})
    if name == 'Proximal GD': pgd_ms = mt
    ratio = f'{pgd_ms/mt:.2f}x' if pgd_ms else '—'
    marker = '  ← NEW' if 'novel' in name else ''
    print(f'{name:<26} {mi:>11.1f} {mt:>8.3f} {ratio}{marker}')

# Convergence plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
COLORS_MAP = {r['name']: r['color'] for r in results}
for ax_idx, port_idx in enumerate([0, 7, 18]):
    y_p = Y.iloc[:, port_idx].values
    hists = {}; opt = np.inf
    for name, cls, kwargs, _ in methods:
        m = cls(alpha=alpha, max_iter=300, **kwargs).fit(X_scaled, y_p)
        hists[name] = m.loss_history_; opt = min(opt, min(m.loss_history_))
    ax = axes[ax_idx]
    for name, h in hists.items():
        gap = np.array(h) - opt + 1e-12
        lw  = 3.0 if 'novel' in name else 1.8
        ax.semilogy(gap, color=COLORS_MAP[name], lw=lw, label=f'{name} ({len(h)})',
                    zorder=5 if 'novel' in name else 3)
    ax.set_title(f'Portfolio P{port_idx+1}', fontweight='bold')
    ax.set_xlabel('Iteration'); ax.set_ylabel('Objective Gap (log)')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
plt.suptitle('Novel: Momentum-BB LASSO vs All Methods', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/momentum_bb_convergence.png', dpi=150, bbox_inches='tight')
print('Saved: outputs/momentum_bb_convergence.png')
