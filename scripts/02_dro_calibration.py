"""
Fix 4: DRO with data-driven epsilon=0.091
Generates: Table 6 numbers in report
"""
import sys
sys.path.insert(0, '.')
from src.data_loader import load_all_data
from src.backtest import walk_forward_backtest, compute_metrics
from src.dro_solver import DROLasso
from src.solvers import LassoProximal

X, Y, factor_names, _ = load_all_data()

print('DRO with data-driven epsilon = 0.091')
print(f'{"epsilon":>10} {"OOS R2":>10} {"Sharpe":>10} {"ICIR":>10}')
print('='*45)

for eps in [0.0, 0.01, 0.05, 0.091, 0.15]:
    if eps == 0.0:
        preds, actuals, dates, _ = walk_forward_backtest(
            X, Y, LassoProximal, alpha=0.003)
    else:
        preds, actuals, dates, _ = walk_forward_backtest(
            X, Y, DROLasso, alpha=0.003, epsilon=eps)
    m    = compute_metrics(preds, actuals)
    note = '<-- data-driven' if abs(eps-0.091) < 0.001 else ''
    print(f'{eps:>10.3f} {m["OOS_R2"]:>10} {m["Sharpe"]:>10} {m["ICIR"]:>10} {note}')
