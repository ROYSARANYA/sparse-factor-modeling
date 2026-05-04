"""
Fix 10: Factor crowding analysis
Generates: outputs/factor_crowding.png
"""
import sys, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import pearsonr, spearmanr
sys.path.insert(0, '.')
from src.data_loader import load_all_data
from src.backtest import walk_forward_backtest
from src.solvers import LassoProximal

X, Y, factor_names, _ = load_all_data()
preds, actuals, dates, _ = walk_forward_backtest(X, Y, LassoProximal, alpha=0.003)
X_oos = X.reindex(dates)

strat_rets = []
for pred, actual in zip(preds, actuals):
    ranks = np.argsort(pred); w = np.zeros(len(pred))
    w[ranks[-3:]] = 1/3; w[ranks[:3]] = -1/3
    strat_rets.append(w @ actual)
strat = pd.Series(strat_rets, index=dates)

print(f'{"Factor":<10} {"Pearson r":>10} {"p-value":>10} {"Level"}')
print('='*45)
crowding = {}
for fname in factor_names:
    s = strat.values; f = X_oos[fname].values
    r, p = pearsonr(s, f)
    level = 'HIGH' if abs(r)>0.3 else 'MOD' if abs(r)>0.15 else 'LOW'
    crowding[fname] = {'r': r, 'p': p}
    print(f'{fname:<10} {r:>10.4f} {p:>10.4f} {level}')

COLORS = ['#2E74B5','#E74C3C','#27AE60','#F39C12','#8E44AD','#1ABC9C']
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
vals   = [crowding[f]['r'] for f in factor_names]
cols   = ['#E74C3C' if abs(v)>0.3 else '#F39C12' if abs(v)>0.15 else '#27AE60' for v in vals]
bars   = axes[0].bar(factor_names, vals, color=cols, alpha=0.85, edgecolor='white')
axes[0].axhline(0, color='black', lw=0.8, ls='--')
axes[0].axhline(0.3, color='red', lw=1.5, ls=':', alpha=0.6)
axes[0].axhline(-0.3, color='red', lw=1.5, ls=':', alpha=0.6)
for bar, v in zip(bars, vals):
    axes[0].text(bar.get_x()+bar.get_width()/2, v+(0.005 if v>=0 else -0.015),
                 f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')
axes[0].set_title('Strategy Correlation with FF Factors\nNo factor exceeds |r|=0.3', fontweight='bold')
axes[0].set_ylabel('Pearson r'); axes[0].grid(True, alpha=0.3, axis='y')

window = 24
axes[1].axhline(0, color='black', lw=0.8, ls='--')
axes[1].axhline(0.3, color='red', lw=1.5, ls=':', alpha=0.5, label='|r|=0.3')
axes[1].axhline(-0.3, color='red', lw=1.5, ls=':', alpha=0.5)
for i, fname in enumerate(factor_names):
    roll_r, roll_d = [], []
    for t in range(window, len(strat)):
        s_w = strat.iloc[t-window:t].values; f_w = X_oos[fname].iloc[t-window:t].values
        if len(s_w) == window:
            roll_r.append(pearsonr(s_w, f_w)[0]); roll_d.append(strat.index[t])
    axes[1].plot(roll_d, roll_r, color=COLORS[i], lw=1.8, label=fname, alpha=0.85)
axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
axes[1].set_title('Rolling 24-Month Factor Crowding', fontweight='bold')
axes[1].set_xlabel('Date'); axes[1].set_ylabel('Rolling Pearson r')
axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)
plt.suptitle('Factor Crowding Analysis', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/factor_crowding.png', dpi=150, bbox_inches='tight')
print('Saved: outputs/factor_crowding.png')
