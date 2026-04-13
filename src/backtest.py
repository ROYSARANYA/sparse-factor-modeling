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

    Returns
    -------
    dict with OOS_R2, Sharpe, Mean_LS_Return, Std_LS_Return
    """
    oos_r2 = r2_score(actuals.flatten(), predictions.flatten())

    # Long-short: long top 5 predicted portfolios, short bottom 5
    ls_returns = []
    for pred, actual in zip(predictions, actuals):
        ranks     = np.argsort(pred)
        long_ret  = actual[ranks[-5:]].mean()
        short_ret = actual[ranks[:5]].mean()
        ls_returns.append(long_ret - short_ret)

    ls_returns = np.array(ls_returns)
    sharpe     = ls_returns.mean() / ls_returns.std() * np.sqrt(12)

    return {
        'OOS_R2':         round(oos_r2, 4),
        'Sharpe':         round(sharpe, 4),
        'Mean_LS_Return': round(ls_returns.mean(), 4),
        'Std_LS_Return':  round(ls_returns.std(), 4),
    }
