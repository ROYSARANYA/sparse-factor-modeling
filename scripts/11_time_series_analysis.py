"""
Professor feedback fix: Time series structure analysis
Generates: outputs/time_series_structure.png
"""
import sys, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
sys.path.insert(0, '.')
from src.data_loader import load_all_data

X, Y, factor_names, _ = load_all_data()
X_vals   = X.values
X_scaled = (X_vals - X_vals.mean(0)) / X_vals.std(0)
n        = len(X)

print('Stationarity (ADF test):')
for fname in factor_names:
    r      = X[fname].values
    result = adfuller(r, maxlag=12, autolag='AIC')
    print(f'  {fname}: ADF={result[0]:.3f}  p={result[1]:.4f}  {"STATIONARY" if result[1]<0.05 else "NON-STAT"}')

print('\nSerial correlation in OLS residuals:')
X_const = add_constant(X_scaled)
for j in [0, 4, 6, 12, 18, 24]:
    y     = Y.iloc[:, j].values
    resid = OLS(y, X_const).fit().resid
    lb    = acorr_ljungbox(resid, lags=[6], return_df=True)
    dw    = 2*(1-np.corrcoef(resid[:-1], resid[1:])[0,1])
    print(f'  P{j+1}: LB(6)p={lb["lb_pvalue"].iloc[0]:.3f}  DW={dw:.3f}')

# Rolling betas
y_p1 = Y.iloc[:, 0].values
window = 60
roll_betas = []
roll_dates = []
for t in range(window, n):
    b = OLS(y_p1[t-window:t], add_constant(X_scaled[t-window:t])).fit().params[1:]
    roll_betas.append(b); roll_dates.append(X.index[t])
roll_betas = np.array(roll_betas)

print('\nRolling betas for P1:')
for i, fname in enumerate(factor_names):
    b = roll_betas[:,i]
    print(f'  {fname}: [{b.min():.3f}, {b.max():.3f}]  std={b.std():.4f}')

# Figure
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
COLORS = ['#2E74B5','#E74C3C','#27AE60','#F39C12','#8E44AD','#1ABC9C']

means_grid = np.array([Y.iloc[:,j].mean()*12*100 for j in range(25)]).reshape(5,5)
im1 = axes[0].imshow(means_grid, cmap='RdYlGn', aspect='auto')
plt.colorbar(im1, ax=axes[0], label='Annual Return (%)')
axes[0].set_xticks(range(5)); axes[0].set_xticklabels(['Growth','2','3','4','Value'], fontsize=9)
axes[0].set_yticks(range(5)); axes[0].set_yticklabels(['Small','2','3','4','Large'], fontsize=9)
for i in range(5):
    for j in range(5):
        axes[0].text(j, i, f'{means_grid[i,j]:.1f}', ha='center', va='center',
                     fontsize=8.5, fontweight='bold',
                     color='white' if means_grid[i,j]>14 or means_grid[i,j]<5 else 'black')
axes[0].set_title('25 Portfolio Annual Returns (%)\nSize × Book-to-Market Sorts (2000-2023)', fontweight='bold')
axes[0].set_xlabel('Book-to-Market (Value →)'); axes[0].set_ylabel('← Size (Small top)')

ci = 1.96/np.sqrt(n)
for i, fname in enumerate(factor_names):
    r     = X[fname].values
    acf_v = [np.corrcoef(r[:-lag], r[lag:])[0,1] for lag in range(1,13)]
    axes[1].plot(range(1,13), acf_v, 'o-', color=COLORS[i], lw=2, ms=5, label=fname)
axes[1].axhline(0, color='black', lw=0.8, ls='--')
axes[1].axhline(ci, color='red', lw=1.5, ls=':', alpha=0.7, label='95% CI')
axes[1].axhline(-ci, color='red', lw=1.5, ls=':', alpha=0.7)
axes[1].set_xlabel('Lag (months)'); axes[1].set_ylabel('Autocorrelation')
axes[1].set_title('Factor Return Autocorrelations\nMost near-IID; RMW mild persistence', fontweight='bold')
axes[1].legend(fontsize=8, ncol=2); axes[1].grid(True, alpha=0.3)

for i, fname in enumerate(factor_names):
    axes[2].plot(roll_dates, roll_betas[:,i], color=COLORS[i], lw=1.8, label=fname, alpha=0.85)
axes[2].axhline(0, color='black', lw=0.8, ls='--')
crisis_s = pd.Timestamp('2008-01-01'); crisis_e = pd.Timestamp('2009-12-31')
axes[2].axvspan(crisis_s, crisis_e, alpha=0.15, color='red', label='2008 crisis')
axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
axes[2].set_xlabel('Date'); axes[2].set_ylabel('Rolling 60M Beta')
axes[2].set_title('Rolling Factor Loadings — Portfolio P1\nTime-varying structure motivates online learning', fontweight='bold')
axes[2].legend(fontsize=8, ncol=2); axes[2].grid(True, alpha=0.3)

plt.suptitle('25 Portfolio Dataset Description and Time Series Structure', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/time_series_structure.png', dpi=150, bbox_inches='tight')
print('Saved: outputs/time_series_structure.png')
