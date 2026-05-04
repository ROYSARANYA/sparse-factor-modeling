"""
Fix 9: Capacity analysis
Generates: outputs/capacity_analysis.png
"""
import sys, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '.')
from src.data_loader import load_all_data
from src.backtest import walk_forward_backtest, compute_metrics_with_costs
from src.solvers import LassoProximal

X, Y, factor_names, _ = load_all_data()
preds, actuals, dates, _ = walk_forward_backtest(X, Y, LassoProximal, alpha=0.003)
m = compute_metrics_with_costs(preds, actuals, bps_cost=0)

base_ann_ret = m['Gross_Ann_Ret'] / 100
avg_turnover = m['Avg_Turnover']
ann_vol      = base_ann_ret / m['Gross_Sharpe']
ADV, gamma   = 5e9, 0.1

aum_range   = np.logspace(6, 9.5, 300)
net_sharpes = [(base_ann_ret - gamma*(a*avg_turnover*12)/ADV)/ann_vol for a in aum_range]
net_sharpes = np.array(net_sharpes)
be_idx      = np.argmin(np.abs(net_sharpes - 1.0))
be_aum      = aum_range[be_idx]

print(f'Gross Sharpe: {m["Gross_Sharpe"]:.4f}')
print(f'Monthly turnover: {avg_turnover:.4f}')
for aum_tgt in [1e7, 1e8, 5e8, 1e9]:
    idx = np.argmin(np.abs(aum_range - aum_tgt))
    print(f'Net Sharpe at ${aum_tgt/1e6:.0f}M: {net_sharpes[idx]:.3f}')
print(f'Breakeven AUM: ${be_aum/1e6:.0f}M')

fig, ax = plt.subplots(figsize=(10, 6))
ax.semilogx(aum_range/1e6, net_sharpes, color='#2E74B5', lw=2.5)
ax.axhline(1.0, color='#E74C3C', lw=2, ls='--', label='Sharpe=1.0')
ax.axhline(m['Gross_Sharpe'], color='gray', lw=1.5, ls=':', label=f'Gross Sharpe={m["Gross_Sharpe"]:.2f}')
ax.axvline(be_aum/1e6, color='#E74C3C', lw=2, ls=':', label=f'Breakeven=${be_aum/1e6:.0f}M')
ax.fill_between(aum_range/1e6, net_sharpes, 1.0, where=net_sharpes<1.0, alpha=0.2, color='red')
ax.fill_between(aum_range/1e6, np.minimum(net_sharpes, m['Gross_Sharpe']+0.5), 1.0,
                where=net_sharpes>=1.0, alpha=0.1, color='green')
for aum_tgt, label in [(10,'$10M'),(100,'$100M'),(500,'$500M')]:
    idx  = np.argmin(np.abs(aum_range/1e6 - aum_tgt))
    ax.annotate(f'{net_sharpes[idx]:.2f}', (aum_tgt, net_sharpes[idx]),
                textcoords='offset points', xytext=(5,6), fontsize=9, color='#2E74B5')
ax.set_xlabel('AUM ($ millions, log scale)', fontsize=12)
ax.set_ylabel('Net Sharpe Ratio', fontsize=12)
ax.set_title(f'Capacity Analysis: Breakeven=${be_aum/1e6:.0f}M\nLinear impact: γ=0.1, ADV=$5B, turnover={avg_turnover:.2f}×/month',
             fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_ylim(0, m['Gross_Sharpe']+0.5)
plt.tight_layout()
plt.savefig('outputs/capacity_analysis.png', dpi=150, bbox_inches='tight')
print('Saved: outputs/capacity_analysis.png')
