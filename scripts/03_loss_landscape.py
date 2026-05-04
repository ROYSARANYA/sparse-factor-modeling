"""
Fix 5: Loss landscape 2D contour Ridge vs LASSO
Generates: outputs/loss_landscape.png
"""
import sys, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '.')
from src.data_loader import load_all_data
from src.solvers import LassoProximal, RidgeScratch

X, Y, factor_names, _ = load_all_data()
X_vals   = X.values
X_scaled = (X_vals - X_vals.mean(0)) / X_vals.std(0)
y_vals   = Y.iloc[:, 0].values
alpha    = 0.003
HML_idx, CMA_idx = 2, 4

lasso_opt = LassoProximal(alpha=alpha).fit(X_scaled, y_vals).coef_
ridge_opt = RidgeScratch(alpha=0.1).fit(X_scaled, y_vals).coef_

print(f'LASSO: beta_HML={lasso_opt[HML_idx]:.5f}, beta_CMA={lasso_opt[CMA_idx]:.5f}')
print(f'Ridge: beta_HML={ridge_opt[HML_idx]:.5f}, beta_CMA={ridge_opt[CMA_idx]:.5f}')

b_fixed = lasso_opt.copy()
grid    = 100
hml_g   = np.linspace(-0.04, 0.015, grid)
cma_g   = np.linspace(-0.025, 0.01,  grid)
HH, CC  = np.meshgrid(hml_g, cma_g)

ridge_obj = np.zeros((grid, grid))
lasso_obj = np.zeros((grid, grid))
for i in range(grid):
    for j in range(grid):
        b = b_fixed.copy()
        b[HML_idx] = HH[i,j]; b[CMA_idx] = CC[i,j]
        mse = np.mean((y_vals - X_scaled@b)**2)
        ridge_obj[i,j] = mse + 0.1*np.sum(b**2)
        lasso_obj[i,j] = mse + alpha*np.sum(np.abs(b))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
c1 = axes[0].contourf(HH, CC, ridge_obj, levels=30, cmap='Blues')
axes[0].contour(HH, CC, ridge_obj, levels=15, colors='navy', alpha=0.4, lw=0.8)
plt.colorbar(c1, ax=axes[0])
axes[0].plot(ridge_opt[HML_idx], ridge_opt[CMA_idx], 'r*', ms=15, zorder=5, label='Ridge optimum')
axes[0].axhline(0, color='k', lw=0.8, ls='--', alpha=0.5)
axes[0].axvline(0, color='k', lw=0.8, ls='--', alpha=0.5)
axes[0].set_xlabel('β_HML'); axes[0].set_ylabel('β_CMA')
axes[0].set_title('Ridge: Elliptical contours\nInterior solution — both nonzero', fontweight='bold')
axes[0].legend()

c2 = axes[1].contourf(HH, CC, lasso_obj, levels=30, cmap='Reds')
axes[1].contour(HH, CC, lasso_obj, levels=15, colors='darkred', alpha=0.4, lw=0.8)
plt.colorbar(c2, ax=axes[1])
axes[1].plot(lasso_opt[HML_idx], lasso_opt[CMA_idx], 'b*', ms=15, zorder=5,
             label=f'LASSO: β_CMA={lasso_opt[CMA_idx]:.4f}')
axes[1].axhline(0, color='k', lw=1.5, alpha=0.7)
axes[1].axvline(0, color='k', lw=1.5, alpha=0.7)
r = alpha
axes[1].plot([r,0,-r,0,r],[0,r,0,-r,0],'b-',lw=2,alpha=0.5,label='L1 ball')
axes[1].set_xlabel('β_HML'); axes[1].set_ylabel('β_CMA')
axes[1].set_title('LASSO: L1 corner forces β_CMA=0\nExplains 14/25 CMA dropout', fontweight='bold')
axes[1].legend()

plt.suptitle('Loss Landscape: Ridge vs LASSO over (β_HML, β_CMA)\nHML-CMA correlation=0.632', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/loss_landscape.png', dpi=150, bbox_inches='tight')
print('Saved: outputs/loss_landscape.png')
