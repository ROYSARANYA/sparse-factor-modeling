"""
dro_solver.py
==============
Distributionally Robust Optimization (DRO) formulation of the LASSO
factor model, grounded in the Wasserstein DRO theory of Esfahani & Kuhn
(2018).  Provides both a CVXPY reference solver and an sklearn-compatible
wrapper for drop-in use in the backtesting and benchmarking pipelines.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Theoretical motivation
----------------------
Standard ERM (empirical risk minimization) minimizes loss on the observed
training distribution P̂ₙ.  If P̂ₙ is a poor approximation of the true
data-generating process — due to limited history, regime shifts, or
estimation noise — the resulting model can perform poorly out-of-sample.

Wasserstein DRO instead minimizes the worst-case expected loss over all
distributions P within a Wasserstein ball of radius ε around P̂ₙ:

    minimize  sup_{P : W(P, P̂ₙ) ≤ ε}  𝔼_P [ ℓ(X, y, β) ]

Larger ε forces the model to be robust to larger distributional shifts
from the training sample — a natural fit for financial time series where
the return-generating process is non-stationary.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Tractable dual reformulation
-----------------------------
The primal Wasserstein minimax problem is infinite-dimensional (it
optimizes over a space of probability measures).  Esfahani & Kuhn (2018)
show that under squared loss and a Lipschitz transport cost, the dual
collapses to a finite-dimensional convex program.

The key structural result: for squared loss with L2 transport cost, the
worst-case perturbation term contributes an L2 norm penalty on β scaled
by ε.  The full tractable objective becomes:

    minimize  (1/n) ‖y − Xβ‖₂²  +  ε ‖β‖₂  +  α ‖β‖₁

  Term 1  (1/n) ‖y − Xβ‖₂²   Standard mean-squared prediction loss.

  Term 2  ε ‖β‖₂              Wasserstein robustness penalty.  Penalizes
                               large coefficients in the L2 sense, making
                               the model conservative under distribution
                               shift.  Note this is the L2 *norm* (not
                               squared norm) — it is convex but
                               non-smooth at β = 0, unlike Ridge.

  Term 3  α ‖β‖₁              LASSO sparsity penalty, unchanged from the
                               standard formulation.

  Combined effect: the L2 norm term provides ridge-like shrinkage toward
  zero while preserving the L1 term's ability to zero out irrelevant
  factors entirely.  The two penalties are complementary — L1 selects
  factors, L2 robustifies the non-zero loadings against shift.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

dro_lasso_cvxpy(X, y, alpha, epsilon=0.1)
-----------------------------------------
CVXPY implementation of the tractable dual objective above.

  Solver:
    Explicitly selects CLARABEL (the default modern CVXPY interior-point
    solver), with verbose=False to suppress per-iteration output in
    backtesting loops.

  Failure handling:
    Returns np.zeros(p) rather than None if the solver fails to find a
    feasible point, so downstream code that computes X @ coef_ never
    encounters a NaN propagation.  This is an improvement over the bare
    beta.value return pattern in cvxpy_solvers.py.

  Epsilon interpretation:
    epsilon=0   → reduces exactly to standard LASSO (L2 term vanishes).
    epsilon>0   → adds distributional robustness; typical useful range
                  for monthly return data is 0.01–0.2.  Values above ~0.5
                  tend to over-shrink all coefficients toward zero,
                  effectively producing a near-null model.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DROLasso  (sklearn-style wrapper)
----------------------------------
Wraps dro_lasso_cvxpy with a fit / predict interface compatible with
walk_forward_backtest and find_best_alphas.

  Parameters:
    alpha    (float, default 0.003)   L1 regularization strength.
    epsilon  (float, default 0.05)    Wasserstein ball radius.

  Attributes set after fit():
    coef_    np.ndarray (p,)   optimal β; guaranteed non-None (zeros on
                               solver failure).

  Limiting cases:
    DROLasso(alpha=a, epsilon=0)  ≡  LassoProximal(alpha=a)  in expectation
                                     (same objective, different solver).
    DROLasso(alpha=0, epsilon=e)  → pure L2-norm regularization (rarely
                                     useful; included for completeness).

  Note: n_iter_ is not set, so this class cannot be passed to the
  benchmark_timing.py loop which reads m.n_iter_.  It is intended for
  the backtesting and comparison pipelines only.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Hyperparameter guidance
-----------------------
  alpha and epsilon interact:
    Both penalties shrink β, but through different geometry.  When tuning
    jointly, it helps to fix one and grid-search the other, or to use the
    time_series_cv function in cross_validation.py with a 2D grid.

  Regime sensitivity:
    ε can be thought of as encoding how much the out-of-sample
    distribution is expected to differ from the training window.  During
    calm periods a small ε (0.01–0.05) is appropriate; in high-vol
    regimes a larger ε (0.1–0.3) is more conservative.  This pairs
    naturally with the adaptive-lambda logic in
    walk_forward_backtest_adaptive — ε could be scaled by the same
    volatility signal as alpha.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Dependencies
------------
numpy, cvxpy (CLARABEL backend)

Reference
---------
Esfahani, P. M. & Kuhn, D. (2018). Data-driven distributionally robust
optimization using the Wasserstein metric: Performance guarantees and
tractable reformulations. Mathematical Programming, 171(1), 115–166.
"""
import numpy as np
import cvxpy as cp


def dro_lasso_cvxpy(X, y, alpha, epsilon=0.1):
    """
    Wasserstein DRO LASSO — correct dual formulation.

    minimize  (1/n) sum_i t_i + epsilon * lambda_w + alpha*||b||_1
    subject to  ||b|| <= lambda_w  (dual Lipschitz constraint)
                t_i >= (y_i - x_i'b)^2 - lambda_w * r_i
                for all perturbation radii r_i

    Simplified tractable form:
    minimize  (1/n)||y - Xb||^2 + epsilon * ||b||_2 + alpha * ||b||_1
    
    This is the key insight: Wasserstein DRO with squared loss
    adds an L2 norm penalty on beta scaled by epsilon,
    ON TOP of the L1 regularization.
    This is still convex and directly interpretable.
    """
    n, p = X.shape
    beta = cp.Variable(p)

    # Wasserstein DRO = standard loss + epsilon * ||beta||_2 + alpha * ||beta||_1
    # The L2 term provides robustness to distribution shift
    # The L1 term provides sparsity
    objective = cp.Minimize(
        cp.sum_squares(y - X @ beta) / n
        + epsilon * cp.norm2(beta)      # robustness term
        + alpha * cp.norm1(beta)        # sparsity term
    )

    prob = cp.Problem(objective)
    prob.solve(solver=cp.CLARABEL, verbose=False)

    return beta.value if beta.value is not None else np.zeros(p)


class DROLasso:
    """
    Sklearn-style wrapper for DRO LASSO.
    Compatible with walk_forward_backtest.

    epsilon=0   → standard LASSO
    epsilon>0   → distributionally robust LASSO
    larger eps  → stronger robustness, more shrinkage
    """
    def __init__(self, alpha=0.003, epsilon=0.05):
        self.alpha   = alpha
        self.epsilon = epsilon
        self.coef_   = None

    def fit(self, X, y):
        self.coef_ = dro_lasso_cvxpy(
            X, y, self.alpha, self.epsilon
        )
        if self.coef_ is None:
            self.coef_ = np.zeros(X.shape[1])
        return self

    def predict(self, X):
        return X @ self.coef_
