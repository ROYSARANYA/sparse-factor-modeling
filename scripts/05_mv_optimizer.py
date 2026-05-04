"""
Fix 8: Mean-variance portfolio optimizer
Generates: outputs/mv_portfolio.png
"""
import sys, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cvxpy as cp
sys.path.insert(0, '.')
from src.data_loader import load_all_data
from src.solvers import LassoProximal

X, Y, factor_names, _ = load_all_data()
X_vals   = X.values
X_scaled = (X_vals - X_vals.mean(0)) / X_vals.std(0)
Y_vals   = Y.values
T, n_ports  = len(X), 25
train_window = 120

def sharpe(r): return np.array(r).mean()/np.array(r).std()*np.sqrt(12)

naive_rets, mv_rets, dates = [], [], []
for t in range(train_window, T):
    X_tr = X_scaled[t-train_window:t]; Y_tr = Y_vals[t-train_window:t]
    Y_te = Y_vals[t]
    preds = [LassoProximal(alpha=0.003).fit(X_tr, Y_tr[:,j]).predict(X_scaled[t:t+1])[0]
             for j in range(n_ports)]
    preds = np.array(preds)
    ranks = np.argsort(preds)
    w_n   = np.zeros(n_ports); w_n[ranks[-3:]] = 1/3; w_n[ranks[:3]] = -1/3
    naive_rets.append(w_n @ Y_te)
    mu_hat = preds / (np.std(preds) + 1e-8)
    Sigma  = np.cov(Y_tr.T) + 1e-4*np.eye(n_ports)
    w = cp.Variable(n_ports)
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, Sigma) - 2.0*mu_hat@w),
                      [cp.sum(w)==0, cp.norm(w,1)<=2, w>=-0.15, w<=0.15])
    prob.solve(solver=cp.CLARABEL, verbose=False)
    mv_rets.append((w.value if w.value is not None else w_n) @ Y_te)
    dates.append(Y.index[t])

naive = np.array(naive_rets); mv = np.array(mv_rets)
print(f'Naive Sharpe:        {sharpe(naive):.4f}')
print(f'MV Optimizer Sharpe: {sharpe(mv):.4f}')
print(f'Difference:          {(sharpe(mv)-sharpe(naive))/abs(sharpe(naive))*100:+.1f}%')

fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(dates, np.cumsum(naive)*100, color='#E74C3C', lw=2, label=f'Naive (Sharpe={sharpe(naive):.3f})')
ax.plot(dates, np.cumsum(mv)*100,   color='#2E74B5', lw=2, label=f'MV Optimizer (Sharpe={sharpe(mv):.3f})')
ax.set_title('Naive vs Mean-Variance Portfolio\nQP: min wΣw - λμ̂w  s.t. dollar-neutral', fontweight='bold')
ax.set_xlabel('Date'); ax.set_ylabel('Cumulative Return (%)')
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/mv_portfolio.png', dpi=150, bbox_inches='tight')
print('Saved: outputs/mv_portfolio.png')
