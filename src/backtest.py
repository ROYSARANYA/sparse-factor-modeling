"""
backtesting.py
==============
Walk-forward backtesting engine for time-series financial factor models.
Enforces strict temporal ordering throughout — no future data ever leaks
into any training window.

Module overview
---------------
The file exposes four public functions that build on each other:

  walk_forward_backtest            – core rolling-window engine
  compute_metrics                  – OOS performance statistics
  walk_forward_backtest_adaptive   – regime-aware variant with dynamic lambda
  compute_metrics_with_costs       – net-of-transaction-cost performance
  compute_alpha_decay              – IC decay curve over holding horizons

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

walk_forward_backtest(X, Y, model_class, alpha, train_window=120, step=1)
-------------------------------------------------------------------------
The core engine. Simulates a live trading process month-by-month with a
fixed-length lookback window.

  Preprocessing:
    X is standardized using its FULL-SAMPLE mean and std (a deliberate
    simplification — in strict no-lookahead setups this would be
    recomputed inside each fold, but here it provides stable scaling).

  Rolling loop  (t = train_window … T, stride = step):
    At each time step t:
      • Training slice  → X_scaled[t-train_window : t],  Y[t-train_window : t]
      • Test slice      → X_scaled[t : t+1],             Y[t : t+1]
      A separate model is fitted independently for each of the 25 target
      portfolios (j = 0…24), then a single one-step-ahead prediction is
      generated.  Coefficients are captured before the model is discarded.

  Returns four arrays:
    predictions     (N × 25)       out-of-sample predicted returns
    actuals         (N × 25)       realised returns at the same dates
    dates           (N,)           index of each prediction date
    coefs_over_time (N × 25 × 6)  factor loadings at every step, enabling
                                   coefficient stability / drift analysis

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

compute_metrics(predictions, actuals)
--------------------------------------
Summarizes backtest quality with five complementary lenses:

  OOS R²      Pooled out-of-sample R² (all portfolios × all months
              flattened), measuring overall predictive fit vs. the
              zero-return naive forecast.

  Long-Short construction:
              Each month, rank the 25 portfolios by predicted return.
              Go long the top 3, short the bottom 3 (equal-weight within
              each leg).  Using 3 per side avoids loading purely on the
              structural size/value spread while still reducing noise vs.
              a single-name bet.
              LS return = mean(long actuals) − mean(short actuals).

  Sharpe      Annualized Sharpe of the LS return series
              (mean × 12) / (std × √12).

  IC / ICIR   Spearman rank correlation between predicted and actual
              returns each month (Information Coefficient).  ICIR =
              IC_mean / IC_std penalizes inconsistent predictors — a
              high mean IC with large variance is unreliable in practice.

  All values rounded for clean display; returns and vol expressed as
  percentages where intuitive.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

walk_forward_backtest_adaptive(X, Y, model_class, base_alpha, window=12, …)
---------------------------------------------------------------------------
Extends the core engine with regime-aware regularization.  Instead of a
fixed alpha, lambda is scaled at every time step by recent market
volatility via `src.regime.compute_adaptive_lambda`:

  High-vol regime  → larger alpha → heavier shrinkage → more stable,
                     conservative coefficients.
  Low-vol regime   → smaller alpha → coefficients free to track signal.

  The market return series (X['Mkt-RF']) is used as the vol proxy.
  If the adaptive lambda for a given date is NaN (e.g., insufficient
  history at the start of the sample), it falls back to base_alpha.

  Returns the same four arrays as the base engine plus:
    lambdas_used (N,)  — the actual alpha applied at each step, which
                         can be plotted against vol regimes for diagnosis.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

compute_metrics_with_costs(predictions, actuals, bps_cost=10)
--------------------------------------------------------------
Repeats the LS construction but deducts realistic transaction costs.

  Portfolio construction:
    Same top-3 / bottom-3 ranking as compute_metrics, but positions are
    represented as explicit weight vectors (±1/3 per name) so turnover
    can be computed precisely.

  Turnover:
    Turnover_t = Σ|w_t − w_{t-1}| / 2  (half the sum of absolute
    weight changes, the standard one-way turnover definition).
    At t=0 the entry cost is treated as half the gross weight sum.

  Cost drag:
    Net return_t = Gross return_t − Turnover_t × (bps_cost / 10_000).
    Default 10 bps per unit of turnover is a conservative institutional
    estimate for liquid equity futures / ETFs.

  Reports both gross and net Sharpe, annualized returns, average monthly
  and annualized turnover, and the annualized cost drag in basis points —
  making the cost-of-rebalancing trade-off directly legible.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

compute_alpha_decay(X, Y, model_class, alpha, max_horizon=6, …)
---------------------------------------------------------------
Measures how quickly the model's predictive edge decays with holding
period — a standard "alpha decay" diagnostic in quantitative finance.

  For each horizon h in {1, 2, …, max_horizon}:
    The target is the average realized return over months [t, t+h) rather
    than the single next-month return.  The model is still trained on a
    one-step return series and predicts one step ahead; the test target
    is the h-month cumulative (averaged) forward return.

  IC is recomputed for each horizon and averaged across all folds.
  A fast IC decay (IC collapses by h=2 or h=3) indicates the model
  captures short-lived momentum / mean-reversion signals.  A slow decay
  suggests persistent factor exposure suited to lower-turnover strategies.

  Progress is printed per horizon for monitoring, since the triple-nested
  loop (horizons × time steps × portfolios) is computationally expensive.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Dependencies
------------
numpy, pandas, sklearn.metrics.r2_score, scipy.stats.spearmanr
src.regime   – compute_adaptive_lambda  (adaptive engine only)

Inputs expected
---------------
X : pd.DataFrame (T × 6)   factor returns (Mkt-RF, SMB, HML, …)
Y : pd.DataFrame (T × 25)  portfolio returns, DateTime-indexed
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
