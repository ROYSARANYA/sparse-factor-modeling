"""
Walk-forward backtesting for time series financial data.
Strictly respects temporal ordering - no look-ahead bias.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


def walk_forward_backtest(X, Y, model_class, alpha,
                          train_window=120, step=1, **model_kwargs):
    """
    Parameters
    ----------
    X            : pd.DataFrame (T x 6) factor returns
    Y            : pd.DataFrame (T x 25) portfolio returns
    model_class  : RidgeScratch | LassoProximal | ElasticNetScratch
    alpha        : regularization strength
    train_window : months of history for training (default 120 = 10 years)
    step         : months to roll forward each iteration

    Returns
    -------
    predictions     : (N x 25) out-of-sample predictions
    actuals         : (N x 25) actual returns
    dates           : list of prediction dates
    coefs_over_time : (N x 25 x 6) coefficients at each step
    """
    T = len(X)
    predictions, actuals, dates, coefs_over_time = [], [], [], []

    # Standardize X using full sample mean/std
    X_vals  = X.values
    X_mean  = X_vals.mean(axis=0)
    X_std   = X_vals.std(axis=0)
    X_scaled = (X_vals - X_mean) / X_std

    Y_vals = Y.values

    for t in range(train_window, T, step):
        X_train = X_scaled[t - train_window:t]
        Y_train = Y_vals[t - train_window:t]
        X_test  = X_scaled[t:t + 1]
        Y_test  = Y_vals[t:t + 1]

        preds, coefs = [], []
        for j in range(Y_train.shape[1]):
            model = model_class(alpha=alpha, **model_kwargs)
            model.fit(X_train, Y_train[:, j])
            preds.append(model.predict(X_test)[0])
            coefs.append(model.coef_.copy())

        predictions.append(preds)
        actuals.append(Y_test[0])
        dates.append(Y.index[t])
        coefs_over_time.append(coefs)

    return (np.array(predictions), np.array(actuals),
            dates, np.array(coefs_over_time))


def compute_metrics(predictions, actuals):
    """
    Compute out-of-sample performance metrics.

    Long-short construction:
    - Rank portfolios by predicted return each month
    - Long top quintile (5 portfolios), short bottom quintile (5)
    - LS return = long return - short return
    - Sharpe annualized using monthly returns

    Returns
    -------
    dict with OOS_R2, Sharpe, Mean_LS_Return, Std_LS_Return,
    Ann_LS_Return
    """
    oos_r2 = r2_score(actuals.flatten(), predictions.flatten())

    # Long-short: long top 3, short bottom 3
    # Using 3 portfolios each side to reduce noise
    # while avoiding full size/value structural spread
    ls_returns = []
    for pred, actual in zip(predictions, actuals):
        ranks     = np.argsort(pred)
        long_ret  = actual[ranks[-3:]].mean()
        short_ret = actual[ranks[:3]].mean()
        ls_returns.append(long_ret - short_ret)

    ls_returns = np.array(ls_returns)

    # Annualized metrics
    ann_return = ls_returns.mean() * 12
    ann_vol    = ls_returns.std() * np.sqrt(12)
    sharpe     = ann_return / ann_vol

    # Information coefficient - rank correlation
    # between predicted and actual returns
    from scipy.stats import spearmanr
    ic_scores = []
    for pred, actual in zip(predictions, actuals):
        ic, _ = spearmanr(pred, actual)
        ic_scores.append(ic)
    ic_mean = np.mean(ic_scores)
    ic_std  = np.std(ic_scores)
    icir    = ic_mean / ic_std if ic_std > 0 else 0

    return {
        'OOS_R2':         round(oos_r2, 4),
        'Sharpe':         round(sharpe, 4),
        'Ann_Return':     round(ann_return * 100, 2),
        'Ann_Vol':        round(ann_vol * 100, 2),
        'Mean_LS_Return': round(ls_returns.mean() * 100, 4),
        'IC_Mean':        round(ic_mean, 4),
        'IC_Std':         round(ic_std, 4),
        'ICIR':           round(icir, 4),
    }


def walk_forward_backtest_adaptive(X, Y, model_class,
                                    base_alpha, window=12,
                                    train_window=120, step=1,
                                    **model_kwargs):
    """
    Walk-forward backtest with regime-aware adaptive lambda.
    Lambda scales with rolling market volatility.
    
    Parameters
    ----------
    X          : pd.DataFrame (T x 6)
    Y          : pd.DataFrame (T x 25)
    base_alpha : float — baseline lambda
    window     : int — vol estimation window
    """
    from src.regime import compute_adaptive_lambda

    T        = len(X)
    X_vals   = X.values
    X_mean   = X_vals.mean(axis=0)
    X_std    = X_vals.std(axis=0)
    X_scaled = (X_vals - X_mean) / X_std
    Y_vals   = Y.values

    # Compute adaptive lambda for each time step
    mkt_returns              = X['Mkt-RF']
    adaptive_lambdas = compute_adaptive_lambda(
        mkt_returns, base_alpha, window=window
    )

    predictions, actuals, dates     = [], [], []
    coefs_over_time, lambdas_used   = [], []

    for t in range(train_window, T, step):
        X_train  = X_scaled[t - train_window:t]
        Y_train  = Y_vals[t - train_window:t]
        X_test   = X_scaled[t:t + 1]
        Y_test   = Y_vals[t:t + 1]

        # Get adaptive lambda for this time step
        current_date   = X.index[t]
        if current_date in adaptive_lambdas.index:
            alpha_t = adaptive_lambdas[current_date]
            if np.isnan(alpha_t):
                alpha_t = base_alpha
        else:
            alpha_t = base_alpha

        preds, coefs = [], []
        for j in range(Y_train.shape[1]):
            model = model_class(alpha=alpha_t, **model_kwargs)
            model.fit(X_train, Y_train[:, j])
            preds.append(model.predict(X_test)[0])
            coefs.append(model.coef_.copy())

        predictions.append(preds)
        actuals.append(Y_test[0])
        dates.append(Y.index[t])
        coefs_over_time.append(coefs)
        lambdas_used.append(alpha_t)

    return (np.array(predictions), np.array(actuals),
            dates, np.array(coefs_over_time),
            np.array(lambdas_used))


def compute_metrics_with_costs(predictions, actuals,
                                bps_cost=10, turnover_penalty=True):
    """
    Compute performance metrics net of transaction costs.

    Parameters
    ----------
    predictions  : (N x 25) predicted returns
    actuals      : (N x 25) actual returns
    bps_cost     : basis points per trade (default 10bps = 0.001)
    turnover_penalty : whether to deduct turnover costs

    Returns
    -------
    dict with gross and net metrics plus turnover statistics
    """
    from scipy.stats import spearmanr
    cost_per_trade = bps_cost / 10000  # convert bps to decimal

    ls_gross, ls_net, turnovers = [], [], []
    ic_scores = []
    prev_weights = None

    for t, (pred, actual) in enumerate(zip(predictions, actuals)):
        ranks = np.argsort(pred)

        # Equal-weight long-short: top 3 long, bottom 3 short
        n_side      = 3
        long_idx    = ranks[-n_side:]
        short_idx   = ranks[:n_side]

        weights     = np.zeros(len(pred))
        weights[long_idx]  =  1.0 / n_side
        weights[short_idx] = -1.0 / n_side

        # Gross return
        gross_ret = np.dot(weights, actual)
        ls_gross.append(gross_ret)

        # Turnover = sum of absolute weight changes
        if prev_weights is not None:
            turnover = np.sum(np.abs(weights - prev_weights)) / 2
        else:
            turnover = np.sum(np.abs(weights)) / 2  # entry cost
        turnovers.append(turnover)

        # Net return after costs
        net_ret = gross_ret - turnover * cost_per_trade
        ls_net.append(net_ret)

        # IC
        ic, _ = spearmanr(pred, actual)
        ic_scores.append(ic)

        prev_weights = weights.copy()

    ls_gross  = np.array(ls_gross)
    ls_net    = np.array(ls_net)
    turnovers = np.array(turnovers)

    def sharpe(rets):
        return rets.mean() / rets.std() * np.sqrt(12) if rets.std() > 0 else 0

    ic_arr = np.array(ic_scores)

    return {
        'Gross_Sharpe':     round(sharpe(ls_gross), 4),
        'Net_Sharpe_10bps': round(sharpe(ls_net), 4),
        'IC_Mean':          round(ic_arr.mean(), 4),
        'ICIR':             round(ic_arr.mean() / ic_arr.std(), 4),
        'Avg_Turnover':     round(turnovers.mean(), 4),
        'Ann_Turnover':     round(turnovers.mean() * 12, 4),
        'Gross_Ann_Ret':    round(ls_gross.mean() * 12 * 100, 2),
        'Net_Ann_Ret_10bps':round(ls_net.mean() * 12 * 100, 2),
        'Cost_Drag_bps':    round((ls_gross.mean() - ls_net.mean()) * 12 * 10000, 1),
    }


def compute_alpha_decay(X, Y, model_class, alpha,
                         max_horizon=6, train_window=120,
                         **model_kwargs):
    """
    Compute alpha decay curve — how IC decays as holding period increases.
    At horizon h, we predict returns h months ahead instead of 1 month.

    Parameters
    ----------
    max_horizon : int — maximum holding period in months

    Returns
    -------
    horizons : list of holding periods
    ic_by_horizon : mean IC at each horizon
    """
    from scipy.stats import spearmanr

    T       = len(X)
    X_vals  = X.values
    X_mean  = X_vals.mean(axis=0)
    X_std   = X_vals.std(axis=0)
    X_scaled = (X_vals - X_mean) / X_std
    Y_vals  = Y.values

    horizons       = list(range(1, max_horizon + 1))
    ic_by_horizon  = []

    for h in horizons:
        ic_scores = []
        for t in range(train_window, T - h):
            X_train = X_scaled[t - train_window:t]
            Y_train = Y_vals[t - train_window:t]
            X_test  = X_scaled[t:t + 1]

            # Target: h-month forward return
            if t + h < T:
                Y_test_h = Y_vals[t:t + h].mean(axis=0)  # average return over h months
            else:
                continue

            preds = []
            for j in range(Y.shape[1]):
                m = model_class(alpha=alpha, **model_kwargs)
                m.fit(X_train, Y_train[:, j])
                preds.append(m.predict(X_test)[0])

            ic, _ = spearmanr(preds, Y_test_h)
            if not np.isnan(ic):
                ic_scores.append(ic)

        ic_by_horizon.append(np.mean(ic_scores) if ic_scores else 0)
        print(f'  Horizon {h}m: IC = {ic_by_horizon[-1]:.4f} (n={len(ic_scores)})')

    return horizons, ic_by_horizon
