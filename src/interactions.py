"""
interactions.py
===============
Extends the linear six-factor model into a sparse nonlinear factor model
by augmenting the feature space with all pairwise factor interactions and
applying LASSO to the expanded design matrix.  The convex relaxation
(L1 penalty) performs automatic feature selection over the 21-dimensional
space, identifying which factor interactions jointly predict portfolio
returns — a question the standard linear model cannot answer.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Motivation
----------
Linear factor models assume returns decompose additively across factors:

    r = β₁·Mkt + β₂·SMB + β₃·HML + … + ε

This misses multiplicative regime effects — e.g., the HML premium may
amplify or invert during high-volatility (large |Mkt-RF|) months, or
the size premium (SMB) may co-move with momentum (Mom) in a way that
linearly contributes to returns.  Pairwise products capture these
second-order effects while keeping the optimization problem convex.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Feature space construction
--------------------------
Starting from p = 6 base factors, all C(6, 2) = 15 ordered pairs (i, j)
with i < j are enumerated via itertools.combinations.  For each pair the
element-wise product X[:, i] * X[:, j] is computed across all T months,
yielding a (T × 15) interaction matrix.

The six pairs that produce interactions and their financial interpretation:

  Mkt-RF × SMB    market excess return scaled by size spread —
                  captures small-cap sensitivity to broad market moves
  Mkt-RF × HML    value premium interaction with market direction —
                  value stocks may behave differently in up vs down markets
  Mkt-RF × RMW    profitability × market — profitable firms may be
                  more defensively priced in downturns
  Mkt-RF × CMA    investment × market — conservative-investment firms
                  tend to be less cyclically exposed
  Mkt-RF × Mom    momentum × market — momentum strategies are known to
                  crash in sharp market reversals
  SMB × HML       size × value joint loading — the classic Fama-French
                  corner portfolios sit at the extremes of this product
  SMB × RMW       small-cap profitability interaction
  SMB × CMA       small-cap investment-style interaction
  SMB × Mom       small-cap momentum — small stocks tend to exhibit
                  stronger momentum effects
  HML × RMW       value × profitability — the "quality value" interaction
  HML × CMA       value × investment — conservative firms are often cheap
  HML × Mom       value-momentum interaction — historically low correlation
                  between these two factors makes their product informative
  RMW × CMA       profitability × investment — the "quality" composite
  RMW × Mom       profitable-momentum stocks
  CMA × Mom       conservative-investment × momentum

The 15 interaction columns are horizontally stacked after the original 6
columns, producing a (T × 21) expanded design matrix X_expanded.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

build_interaction_features(X, factor_names)
--------------------------------------------
Pure feature engineering step; no model fitting occurs here.

  Input:
    X (T × 6) — standardized base factor returns (pre-scaled by caller).

  Standardization note:
    Interaction terms are products of already-standardized factors, so
    each product has mean ≈ 0 (since each factor has mean ≈ 0) but
    variance that varies across pairs depending on their correlation.
    A separate re-standardization step in analyze_interaction_selection
    ensures all 21 features are on a common scale before LASSO is applied.

  Returns:
    X_expanded      (T × 21)   base + interaction columns
    feature_names   list[21]   e.g. ['Mkt-RF', …, 'Mkt-RFxSMB', …]
    interaction_names list[15] interaction-only names for subsetting
                               coefficient or selection matrices

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

analyze_interaction_selection(X, Y, model_class, alpha, …)
-----------------------------------------------------------
Fits a LASSO model on the full 21-feature expanded space for each of the
25 portfolio target series and records which features survive shrinkage.

  Preprocessing:
    X_expanded is re-standardized column-wise using its own mean and std
    (not the base X statistics) so interaction features, which have
    different scales from the base factors, receive proportionate
    regularization.  A zero-std guard (X_std[X_std == 0] = 1) prevents
    division-by-zero for any degenerate constant interaction column.

  Per-portfolio LASSO fits:
    A fresh model instance is created for each of the 25 portfolios.
    LASSO's sparsity means most of the 21 coefficients will be exactly
    zero for any given portfolio — only the interactions genuinely
    predictive for that specific portfolio survive.

  Selection matrix (25 × 21), binary:
    Entry (j, k) = 1 if |coef_jk| > 1e-4 (i.e. LASSO did not zero it
    out), 0 otherwise.  The 1e-4 threshold is a numerical tolerance —
    interior-point solvers occasionally return values like 3e-7 that
    are effectively zero but not exactly so.  Column sums of the
    selection matrix give the "selection frequency" of each feature
    across all 25 portfolios, a natural measure of cross-portfolio
    importance.

  Coefficient matrix (25 × 21), real-valued:
    Full coefficient magnitudes retained for signed analysis — e.g.
    to distinguish portfolios where Mkt×Mom loads positively vs.
    negatively.

  Returns:
    selection     (25 × 21)  binary selection indicators
    coef_matrix   (25 × 21)  LASSO coefficient values
    feature_names list[21]   column labels for both matrices
    inter_names   list[15]   interaction-only column labels for
                             subsetting to the nonlinear part

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Interpretation guidance
-----------------------
  Base factor selection frequency:
    If a base factor (columns 0–5) is selected for fewer portfolios in
    the expanded model than in the plain LASSO, the interaction terms
    are absorbing its explanatory power — evidence of genuine nonlinearity.

  Interaction selection frequency:
    High selection frequency across many portfolios (e.g. Mkt×Mom
    selected in 18/25) suggests a pervasive second-order effect.  Low
    or zero selection means the interaction adds no predictive content
    beyond the base factors at the chosen alpha.

  Alpha sensitivity:
    Interaction coefficients are generally smaller in magnitude than base
    factor coefficients and will be zeroed first as alpha increases.
    Using a smaller alpha than the plain LASSO (e.g. the median CV alpha
    divided by 2) is often appropriate when the goal is discovery rather
    than conservative shrinkage.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Dependencies
------------
numpy, pandas, itertools.combinations
Any model_class from solvers.py or cvxpy_solvers.py with a
fit(X, y) / coef_ interface.

Inputs expected
---------------
X : np.ndarray (T × 6)    pre-standardized base factor matrix
Y : pd.DataFrame (T × 25) portfolio return matrix, column-accessible
                           via .iloc[:, j]
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
