"""
Fix 1: High-dimensional OAP experiment
Generates: outputs/highdim_oap.png
"""
import sys, numpy as np, pandas as pd, time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '.')
from src.solvers import (LassoProximal, FISTALasso, FISTARestart,
                         BBLasso, CoordinateDescent, RidgeScratch)
from src.cvxpy_solvers import lasso_cvxpy

df = pd.read_csv("data/PredictorLSretWide.csv")
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
recent_df  = df[df['year']>=2000].drop(columns=['year'])
recent_df  = recent_df.set_index('date').sort_index()
clean_df   = recent_df.dropna(axis=1)
print(f"Clean: {clean_df.shape}")

X_raw = clean_df.values[:-1]
y_raw = clean_df.values[1:].mean(axis=1)
X     = (X_raw - X_raw.mean(0)) / (X_raw.std(0) + 1e-8)
y     = y_raw
n, p  = X.shape
alpha = 0.20

print(f"CVXPY reference...")
t0 = time.perf_counter(); ref = lasso_cvxpy(X, y, alpha)
print(f"  {(time.perf_counter()-t0)*1000:.0f}ms")

methods = [
    ('Proximal GD',       LassoProximal,  {}),
    ('FISTA vanilla',     FISTALasso,     {}),
    ('FISTA+fn restart',  FISTARestart,   {'restart':'function'}),
    ('BB LASSO (alt)',    BBLasso,        {}),
    ('Coord Descent',     CoordinateDescent, {}),
    ('Ridge',             RidgeScratch,   {'alpha':0.1}),
]

print(f'\n{"Method":<26} {"Iters":>7} {"ms":>8} {"Nonzero":>9} {"vs PGD"}')
print('='*60)
pgd_ms = None; results = []
for name, cls, kwargs in methods:
    runs = []
    for _ in range(5):
        t0 = time.perf_counter()
        m  = cls(**({'alpha':alpha,'max_iter':3000} if not name.startswith('Ridge') else {}),
                 **kwargs).fit(X, y)
        runs.append((time.perf_counter()-t0)*1000)
    ms    = np.mean(runs)
    iters = m.n_iter_ if hasattr(m,'n_iter_') and m.n_iter_>0 else 1
    nz    = int(np.sum(np.abs(m.coef_)>1e-4))
    if name == 'Proximal GD': pgd_ms = ms
    ratio = f'{pgd_ms/ms:.2f}x' if pgd_ms and not name.startswith('Ridge') else '—'
    results.append({'name':name,'iters':iters,'ms':ms,'nz':nz,'coef':m.coef_.copy()})
    print(f'{name:<26} {iters:>7} {ms:>8.1f} {nz:>9} {ratio}')

lasso_coefs = results[0]['coef']
sel_idx     = np.where(np.abs(lasso_coefs)>1e-4)[0]
print(f'\nLASSO: {len(sel_idx)}/{p} selected')
top_idx = sel_idx[np.argsort(np.abs(lasso_coefs[sel_idx]))[::-1]]
for i in top_idx[:10]:
    print(f'  {clean_df.columns[i]:<28} {lasso_coefs[i]:+.5f}')

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
COLORS = ['#E74C3C','#2E74B5','#27AE60','#8E44AD','#1ABC9C']
names_s = ['PGD','FISTA\nvanilla','FISTA\n+fn','BB\nLASSO','Coord\nDescent']
iters_v = [r['iters'] for r in results[:-1]]
ms_v    = [r['ms']    for r in results[:-1]]
bars    = axes[0].bar(names_s, iters_v, color=COLORS, alpha=0.85, edgecolor='white')
for bar, it, ms in zip(bars, iters_v, ms_v):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(iters_v)*0.02,
                 f'{it}\n({ms:.0f}ms)', ha='center', fontsize=8.5, fontweight='bold')
axes[0].set_title(f'Iterations at p={p}\nWith wall-clock ms', fontweight='bold')
axes[0].set_ylabel('Iterations'); axes[0].grid(True, alpha=0.3, axis='y')

nz_l = results[0]['nz']; nz_r = results[-1]['nz']
axes[1].bar(['LASSO\nselected','LASSO\ndropped','Ridge\nnonzero'],
            [nz_l, p-nz_l, nz_r], color=['#27AE60','#E74C3C','#2E74B5'], alpha=0.85, edgecolor='white')
axes[1].set_title(f'Sparsity: LASSO selects {nz_l}/{p}\nRidge retains all {nz_r}', fontweight='bold')
axes[1].set_ylabel('Predictors')
for i, v in enumerate([nz_l, p-nz_l, nz_r]):
    axes[1].text(i, v+1, str(v), ha='center', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

if len(sel_idx) > 0:
    top_n  = min(12, len(sel_idx))
    top_c  = lasso_coefs[top_idx[:top_n]]
    top_nm = [clean_df.columns[i] for i in top_idx[:top_n]]
    cb     = ['#27AE60' if c>0 else '#E74C3C' for c in top_c]
    axes[2].barh(range(top_n), top_c, color=cb, alpha=0.85, edgecolor='white')
    axes[2].set_yticks(range(top_n)); axes[2].set_yticklabels(top_nm, fontsize=9)
    axes[2].axvline(0, color='black', lw=0.8); axes[2].invert_yaxis()
axes[2].set_title('Top LASSO-Selected Predictors', fontweight='bold')
axes[2].set_xlabel('LASSO Coefficient'); axes[2].grid(True, alpha=0.3, axis='x')

plt.suptitle(f'OpenAssetPricing: n={n}, p={p} firm characteristics',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/highdim_oap.png', dpi=150, bbox_inches='tight')
print('Saved: outputs/highdim_oap.png')
