"""
Novel Contribution 2 (improved): KKT-based factor importance ranking
Spearman rank correlation between C^{-1}c_k and regularization path order
Result: Mean rho=0.906, 22/25 portfolios rho>0.75
"""
import sys, numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, '.')
from src.data_loader import load_all_data
from src.solvers import LassoProximal

X, Y, factor_names, _ = load_all_data()
X_vals   = X.values
X_scaled = (X_vals - X_vals.mean(0)) / X_vals.std(0)
n, p     = X_scaled.shape
alpha    = 0.003

C    = np.corrcoef(X_scaled.T)
Cinv = np.linalg.inv(C)

rho_scores = []
for k in range(25):
    y_k = Y.iloc[:, k].values
    c_k = np.array([np.corrcoef(X_scaled[:,j], y_k)[0,1] for j in range(p)])
    pc        = np.abs(Cinv @ c_k)
    pred_rank = np.argsort(pc)[::-1]
    alphas_path  = np.logspace(-3, 0, 60)
    dropout_alpha = {}
    for a in alphas_path:
        m = LassoProximal(alpha=a, max_iter=1000).fit(X_scaled, y_k)
        for j in range(p):
            if j not in dropout_alpha and abs(m.coef_[j]) <= 1e-4:
                dropout_alpha[j] = a
    for j in range(p):
        if j not in dropout_alpha:
            dropout_alpha[j] = 0.0
    actual_rank = sorted(range(p), key=lambda j: -dropout_alpha[j])
    pred_pos    = [list(pred_rank).index(j)   for j in range(p)]
    actual_pos  = [list(actual_rank).index(j) for j in range(p)]
    rho, pval   = spearmanr(pred_pos, actual_pos)
    rho_scores.append(rho)
    print(f'P{k+1:<3}: rho={rho:+.3f}  p={pval:.4f}')

print(f'\nMean rho:   {np.mean(rho_scores):.4f}')
print(f'Median rho: {np.median(rho_scores):.4f}')
print(f'rho>0.75:   {sum(r>0.75 for r in rho_scores)}/25 portfolios')
print(f'rho>0.50:   {sum(r>0.50 for r in rho_scores)}/25 portfolios')
