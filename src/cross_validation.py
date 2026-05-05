"""
cross_validation.py
===================
Time-series-safe cross-validation for selecting the regularization
hyperparameter lambda (alpha) without introducing look-ahead bias.

All splits use an expanding window — training always strictly precedes
validation, and no random shuffling is ever applied.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

time_series_cv(X, y, model_class, alphas, n_splits=5)
------------------------------------------------------
Core expanding-window CV loop for a single target series y.

  Fold construction:
    The T observations are divided into (n_splits + 1) equal-length
    blocks of size fold_size = T // (n_splits + 1).

    Fold k  (k = 1 … n_splits):
      Training set  → X[0 : fold_size × k]          (expands each fold)
      Validation set→ X[fold_size×k : fold_size×(k+1)]  (next block)

    This guarantees that every validation month is strictly in the
    future relative to every training month — the same discipline
    enforced in walk_forward_backtest.py.  Any fold whose validation
    end-index would exceed T is silently skipped.

  Scoring:
    For each (alpha, fold) pair the model is freshly instantiated,
    fitted on the training slice, and scored on the validation slice
    using R².  Negative R² values are kept — they carry real signal
    about poor alpha choices and should not be clipped.

  cv_results dict layout (keyed by alpha):
    mean_r2  – mean validation R² across all completed folds
               (primary selection criterion)
    std_r2   – fold-to-fold standard deviation, useful for flagging
               high-variance alphas that happen to score well on one
               fold but are unreliable overall
    scores   – raw per-fold R² list for further inspection or plotting

  Selection rule:
    best_alpha = argmax mean_r2.  No 1-SE rule is applied; the caller
    can implement one using std_r2 if a more conservative choice is
    desired.

  Returns:
    best_alpha   – scalar, the alpha with the highest mean validation R²
    cv_results   – full dict for every alpha (for diagnostic plots)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

find_best_alphas(X, Y, model_class, alphas=None, n_splits=5)
-------------------------------------------------------------
Runs time_series_cv independently for each of the 25 portfolio columns
in Y and aggregates the results into a single recommended alpha.

  Preprocessing:
    X is column-standardized (zero mean, unit variance) using its full-
    sample statistics before being passed to any fold — consistent with
    the scaling convention used in backtesting.py.

  Default alpha grid:
    [0.001, 0.003, 0.005, 0.007, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1]
    Ten log-spaced candidates spanning weak-to-moderate regularization,
    calibrated for the scale of monthly factor return data.  A custom
    grid can be passed in to widen or narrow the search.

  Per-portfolio selection:
    Each of the 25 targets gets its own best_alpha via time_series_cv.
    The heterogeneity across portfolios is preserved in best_alphas (a
    list of length 25) and can be used in portfolio-specific model
    configurations.

  Aggregation — median not mean:
    The single recommended alpha is the median of the 25 per-portfolio
    optima rather than the mean.  The median is more robust to the
    handful of portfolios where CV may pick an extreme alpha due to a
    noisy or near-zero R² surface.  This single value is convenient for
    global-alpha experiments (e.g., benchmarking solvers) where fitting
    25 different lambdas would obscure comparisons.

  Returns:
    best_alphas  – list (length 25) of per-portfolio optimal alphas
    median_alpha – scalar, recommended global alpha for the full dataset

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Design notes
------------
  Expanding vs. rolling window:
    An expanding window is used (training always starts at t=0) rather
    than a fixed-width rolling window.  This is appropriate when the
    dataset is short (≈200 months) and each additional observation
    meaningfully improves coefficient estimates.  For longer datasets
    a rolling variant would better capture non-stationarity.

  Why not sklearn TimeSeriesSplit:
    sklearn's TimeSeriesSplit does the same thing but returns indices
    only and requires wrapping for multi-alpha loops.  The explicit
    implementation here keeps the alpha-sweep logic self-contained and
    makes the fold boundaries transparent.

  Computational cost:
    O(n_splits × |alphas| × 25) model fits are performed inside
    find_best_alphas.  With the defaults that is 5 × 10 × 25 = 1,250
    fits — fast for the solvers in this codebase but worth noting if
    larger grids are substituted.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Dependencies
------------
numpy, sklearn.metrics.r2_score

Inputs expected
---------------
X : pd.DataFrame (T × 6)   factor returns, DateTime-indexed
Y : pd.DataFrame (T × 25)  portfolio returns, DateTime-indexed
"""
import numpy as np
from sklearn.metrics import r2_score


def time_series_cv(X, y, model_class, alphas,
                   n_splits=5, **model_kwargs):
    """
    Expanding window cross-validation for time series.
    
    For each alpha, splits data into n_splits folds
    where training always precedes validation.
    
    Returns best alpha and full CV results.
    """
    n = len(X)
    fold_size = n // (n_splits + 1)
    
    cv_results = {}

    for alpha in alphas:
        fold_scores = []
        for k in range(1, n_splits + 1):
            train_end  = fold_size * k
            val_end    = fold_size * (k + 1)
            if val_end > n:
                break
            X_train = X[:train_end]
            y_train = y[:train_end]
            X_val   = X[train_end:val_end]
            y_val   = y[train_end:val_end]

            model = model_class(alpha=alpha, **model_kwargs)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            score = r2_score(y_val, preds)
            fold_scores.append(score)

        cv_results[alpha] = {
            'mean_r2': np.mean(fold_scores),
            'std_r2':  np.std(fold_scores),
            'scores':  fold_scores
        }

    best_alpha = max(cv_results, key=lambda a: cv_results[a]['mean_r2'])
    return best_alpha, cv_results


def find_best_alphas(X, Y, model_class,
                     alphas=None, n_splits=5, **model_kwargs):
    """
    Find best alpha for each of the 25 portfolios.
    Returns array of best alphas and mean best alpha.
    """
    if alphas is None:
        alphas = [0.001, 0.003, 0.005, 0.007, 0.01,
                  0.02, 0.03, 0.05, 0.07, 0.1]

    X_vals   = X.values
    X_scaled = (X_vals - X_vals.mean(axis=0)) / X_vals.std(axis=0)

    best_alphas = []
    for j in range(Y.shape[1]):
        y_vals = Y.iloc[:, j].values
        best_alpha, _ = time_series_cv(
            X_scaled, y_vals, model_class,
            alphas, n_splits, **model_kwargs
        )
        best_alphas.append(best_alpha)

    return best_alphas, np.median(best_alphas)
