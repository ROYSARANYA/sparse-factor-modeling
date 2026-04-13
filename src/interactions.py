"""
Sparse Nonlinear Factor Model via Convex Relaxation.
Extends linear factor model with pairwise interaction terms.

Original features: 6 factors
Interaction features: C(6,2) = 15 pairwise products
Total feature set: 6 + 15 = 21 features

LASSO on expanded feature set discovers which factor
interactions jointly predict returns — a research question
not answered by standard linear factor models.
"""
import numpy as np
import pandas as pd
from itertools import combinations


FACTOR_NAMES = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Mom']


def build_interaction_features(X, factor_names=FACTOR_NAMES):
    """
    Constructs pairwise interaction terms and appends to X.

    Parameters
    ----------
    X : np.ndarray (T x 6) — standardized factor returns

    Returns
    -------
    X_expanded    : np.ndarray (T x 21)
    feature_names : list of 21 feature names
    interaction_names : list of 15 interaction names only
    """
    n, p         = X.shape
    interactions = []
    inter_names  = []

    for i, j in combinations(range(p), 2):
        interactions.append(X[:, i] * X[:, j])
        inter_names.append(f'{factor_names[i]}x{factor_names[j]}')

    X_inter    = np.column_stack(interactions)
    X_expanded = np.hstack([X, X_inter])

    feature_names = factor_names + inter_names
    return X_expanded, feature_names, inter_names


def analyze_interaction_selection(X, Y,
                                   model_class,
                                   alpha,
                                   factor_names=FACTOR_NAMES,
                                   **model_kwargs):
    """
    Run LASSO with interactions across all 25 portfolios.
    Returns selection matrix and coefficient matrix.
    """
    X_expanded, feature_names, inter_names = \
        build_interaction_features(X, factor_names)

    # Standardize expanded features
    X_mean = X_expanded.mean(axis=0)
    X_std  = X_expanded.std(axis=0)
    X_std[X_std == 0] = 1  # avoid division by zero
    X_scaled = (X_expanded - X_mean) / X_std

    n_features   = X_expanded.shape[1]
    n_portfolios = Y.shape[1]

    selection   = np.zeros((n_portfolios, n_features))
    coef_matrix = np.zeros((n_portfolios, n_features))

    for j in range(n_portfolios):
        model = model_class(alpha=alpha, **model_kwargs)
        model.fit(X_scaled, Y.iloc[:, j].values)
        coef_matrix[j] = model.coef_
        selection[j]   = (np.abs(model.coef_) > 1e-4).astype(int)

    return selection, coef_matrix, feature_names, inter_names
