"""
Novel Contribution 2: KKT-based factor dropout prediction
Generates: outputs/dropout_prediction.png
"""
import sys, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
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

# Tune threshold on portfolios 1-15
best_thresh, best_acc = None, -1
for tau in np.linspace(0.05, 3.0, 200):
    correct = 0
    for k in range(15):
        y_k  = Y.iloc[:, k].values
        c_k  = np.array([np.corrcoef(X_scaled[:,j], y_k)[0,1] for j in range(p)])
        pc   = Cinv @ c_k
        pred = set(j for j in range(p) if abs(pc[j]) < tau*alpha*np.sqrt(n)/n)
        m    = LassoProximal(alpha=alpha).fit(X_scaled, y_k)
        act  = set(j for j in range(p) if abs(m.coef_[j]) <= 1e-4)
        if pred == act: correct += 1
    if correct > best_acc: best_acc = correct; best_thresh = tau

print(f'Best threshold: {best_thresh:.3f}  Training accuracy: {best_acc}/15')

# Verify on held-out portfolios 16-25
correct_test = 0
for k in range(15, 25):
    y_k  = Y.iloc[:, k].values
    c_k  = np.array([np.corrcoef(X_scaled[:,j], y_k)[0,1] for j in range(p)])
    pc   = Cinv @ c_k
    tau  = best_thresh*alpha*np.sqrt(n)/n
    pred = set(j for j in range(p) if abs(pc[j]) < tau)
    m    = LassoProximal(alpha=alpha).fit(X_scaled, y_k)
    act  = set(j for j in range(p) if abs(m.coef_[j]) <= 1e-4)
    match = pred == act
    if match: correct_test += 1
    pn = [factor_names[j] for j in sorted(pred)]
    an = [factor_names[j] for j in sorted(act)]
    print(f'P{k+1}: pred={pn}  actual={an}  {"✓" if match else "✗"}')
print(f'Test accuracy: {correct_test}/10')

# Full analysis
tp = np.zeros(p); tn = np.zeros(p); fp = np.zeros(p); fn = np.zeros(p)
pred_matrix   = np.zeros((25, p))
actual_matrix = np.zeros((25, p))
for k in range(25):
    y_k   = Y.iloc[:, k].values
    c_k   = np.array([np.corrcoef(X_scaled[:,j], y_k)[0,1] for j in range(p)])
    pc    = Cinv @ c_k
    tau   = best_thresh*alpha*np.sqrt(n)/n
    m     = LassoProximal(alpha=alpha).fit(X_scaled, y_k)
    for j in range(p):
        pz = abs(pc[j]) < tau
        az = abs(m.coef_[j]) <= 1e-4
        pred_matrix[k,j]   = 0 if pz else 1
        actual_matrix[k,j] = 0 if az else 1
        if pz and az:     tn[j] += 1
        if not pz and not az: tp[j] += 1
        if pz and not az: fn[j] += 1
        if not pz and az: fp[j] += 1

print(f'\nOverall: {int(np.sum(pred_matrix==actual_matrix))}/150 = {np.mean(pred_matrix==actual_matrix)*100:.1f}%')
for j in range(p):
    print(f'  {factor_names[j]}: {(tp[j]+tn[j])/25*100:.0f}%')

# Plot
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
accs  = [(tp[j]+tn[j])/25*100 for j in range(p)]
cols  = ['#27AE60' if a>=80 else '#F39C12' if a>=60 else '#E74C3C' for a in accs]
bars  = axes[0].bar(factor_names, accs, color=cols, alpha=0.85, edgecolor='white')
axes[0].axhline(80, color='#27AE60', lw=2, ls='--', label='80%')
axes[0].axhline(60, color='#F39C12', lw=2, ls='--', label='60%')
for bar, acc in zip(bars, accs):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                 f'{acc:.0f}%', ha='center', fontsize=10, fontweight='bold')
axes[0].set_title('KKT Dropout Prediction Accuracy', fontweight='bold')
axes[0].set_ylabel('Accuracy (%)'); axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3, axis='y'); axes[0].set_ylim(0, 115)

diff = pred_matrix - actual_matrix
sns.heatmap(diff, xticklabels=factor_names,
            yticklabels=[f'P{i+1}' for i in range(25)],
            cmap='RdYlGn', center=0, vmin=-1, vmax=1,
            annot=True, fmt='.0f', ax=axes[1], linewidths=0.5,
            cbar_kws={'label': '-1=Miss  0=Correct  +1=FalseAlarm'})
axes[1].set_title('Prediction Error Map', fontweight='bold')

pc_matrix = np.zeros((25, p))
for k in range(25):
    y_k = Y.iloc[:, k].values
    c_k = np.array([np.corrcoef(X_scaled[:,j], y_k)[0,1] for j in range(p)])
    pc_matrix[k] = Cinv @ c_k
sns.heatmap(np.abs(pc_matrix), xticklabels=factor_names,
            yticklabels=[f'P{i+1}' for i in range(25)],
            cmap='RdYlGn', annot=True, fmt='.2f', ax=axes[2], linewidths=0.5,
            cbar_kws={'label': '|Partial Correlation|'})
axes[2].set_title(f'Partial Correlations |[C⁻¹c_k]_j|', fontweight='bold')

plt.suptitle('KKT Factor Dropout Prediction — Novel Contribution',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/dropout_prediction.png', dpi=150, bbox_inches='tight')
print('Saved: outputs/dropout_prediction.png')
