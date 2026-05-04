"""
Fix 7: Dollar-neutral Sharpe breakdown
"""
import sys, numpy as np
sys.path.insert(0, '.')
from src.data_loader import load_all_data
from src.backtest import walk_forward_backtest
from src.solvers import LassoProximal

X, Y, factor_names, _ = load_all_data()
preds, actuals, dates, _ = walk_forward_backtest(X, Y, LassoProximal, alpha=0.003)

def sharpe(rets):
    r = np.array(rets)
    return r.mean()/r.std()*np.sqrt(12)

gross, lo, dn = [], [], []
for pred, actual in zip(preds, actuals):
    ranks = np.argsort(pred); n = len(pred)
    w_g = np.zeros(n); w_g[ranks[-3:]] = 1/3; w_g[ranks[:3]] = -1/3
    w_l = np.zeros(n); w_l[ranks[-5:]] = 1/5
    w_d = np.zeros(n); w_d[ranks[-5:]] = 1/5; w_d[ranks[:5]] = -1/5
    w_d -= w_d.mean()
    gross.append(w_g @ actual); lo.append(w_l @ actual); dn.append(w_d @ actual)

print(f'{"Strategy":<30} {"Sharpe":>10} {"Ann Ret%":>10} {"Ann Vol%":>10}')
print('='*62)
for name, rets in [('Gross top3/bot3', gross),('Long-only top5', lo),('Dollar-neutral', dn)]:
    r = np.array(rets)
    print(f'{name:<30} {sharpe(r):>10.4f} {r.mean()*12*100:>10.2f} {r.std()*np.sqrt(12)*100:>10.2f}')
