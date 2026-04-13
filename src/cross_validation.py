"""
Time-series cross-validation for lambda selection.
Uses expanding window — no random splits, no look-ahead bias.
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
