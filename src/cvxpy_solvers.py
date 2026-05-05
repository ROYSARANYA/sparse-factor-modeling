"""
cvxpy_solvers.py
================
Canonical convex program formulations of three regularized regression
models using CVXPY.  These are reference implementations — solved by an
interior-point method to high numerical precision — and serve as
ground-truth coefficient vectors against which the iterative first-order
solvers in solvers.py are benchmarked.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ridge_cvxpy(X, y, alpha)
------------------------
Solves:   minimize  (1/n) ‖y − Xβ‖₂²  +  α ‖β‖₂²

  The L2 penalty shrinks all coefficients proportionally toward zero but
  never to exactly zero.  The (1/n) scaling on the loss term makes alpha
  comparable across datasets of different lengths — doubling T does not
  require halving alpha to maintain the same effective regularization.

  Convex structure: sum of two convex quadratics → strictly convex,
  unique global minimum, closed-form solution exists but CVXPY is used
  here for consistency and to provide an exact numeric reference.

lasso_cvxpy(X, y, alpha)
------------------------
Solves:   minimize  (1/n) ‖y − Xβ‖₂²  +  α ‖β‖₁

  The L1 penalty induces sparsity — coefficients of irrelevant factors
  are driven to exactly zero, performing implicit variable selection.
  This is the primary reference target for the proximal / FISTA / BB /
  coordinate-descent solvers in solvers.py; their coefficient errors
  (L∞ norm vs. beta.value) measure convergence quality.

  Convex structure: smooth quadratic loss + non-smooth convex penalty.
  Not strictly convex when p > n, so multiple optima can exist, but
  CVXPY's interior-point solver finds a high-precision solution that
  serves as a stable benchmark.

elasticnet_cvxpy(X, y, alpha, l1_ratio=0.5)
--------------------------------------------
Solves:   minimize  (1/n) ‖y − Xβ‖₂²  +  λ₁ ‖β‖₁  +  λ₂ ‖β‖₂²

  where  λ₁ = alpha × l1_ratio
         λ₂ = alpha × (1 − l1_ratio)

  Elastic Net blends L1 sparsity with L2 stability.  At l1_ratio=1 it
  reduces to LASSO; at l1_ratio=0 it reduces to Ridge.  The default
  l1_ratio=0.5 gives equal weight to both penalties.

  The L2 term breaks the degeneracy of LASSO when predictors are
  correlated — grouped correlated features tend to receive similar
  (non-zero) coefficients rather than one arbitrary member being
  selected and the rest zeroed out.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Shared design conventions
--------------------------
  All three functions:
    • Accept a pre-standardized X (caller's responsibility).
    • Return beta.value — a plain numpy array of shape (p,), or None if
      the solver fails to converge (CVXPY will return None for infeasible
      / numerical failure).  Callers should guard against None before
      computing error metrics.
    • Use CVXPY's default solver selection (typically CLARABEL or ECOS),
      which applies an interior-point method suitable for small-to-medium
      dense problems (T ≈ 200, p = 6).
    • Do not set warm-start or solver tolerances — default tolerances
      (≈1e-8) are tight enough to serve as ground truth for benchmarking
      first-order methods that typically converge to 1e-4 or 1e-5.

Performance note
----------------
  Interior-point solvers scale as O(p³) per iteration and are called
  once per (target, benchmark run) pair.  At p=6 this is negligible, but
  these functions are not suitable as production solvers for p >> 100 or
  in tight loops — use the iterative solvers in solvers.py instead.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Dependencies
------------
numpy, cvxpy

Inputs expected
---------------
X     : np.ndarray (n × p)  pre-standardized design matrix
y     : np.ndarray (n,)     target return vector
alpha : float               overall regularization strength (> 0)
"""
import numpy as np
import cvxpy as cp


def ridge_cvxpy(X, y, alpha):
    """
    Ridge as convex program:
    minimize (1/n)||y - Xb||_2^2 + alpha*||b||_2^2
    """
    n, p = X.shape
    beta = cp.Variable(p)
    objective = cp.Minimize(
        cp.sum_squares(y - X @ beta) / n
        + alpha * cp.sum_squares(beta)
    )
    cp.Problem(objective).solve()
    return beta.value


def lasso_cvxpy(X, y, alpha):
    """
    LASSO as convex program:
    minimize (1/n)||y - Xb||_2^2 + alpha*||b||_1
    """
    n, p = X.shape
    beta = cp.Variable(p)
    objective = cp.Minimize(
        cp.sum_squares(y - X @ beta) / n
        + alpha * cp.norm1(beta)
    )
    cp.Problem(objective).solve()
    return beta.value


def elasticnet_cvxpy(X, y, alpha, l1_ratio=0.5):
    """
    Elastic Net as convex program:
    minimize (1/n)||y-Xb||_2^2 + l1*||b||_1 + l2*||b||_2^2
    """
    n, p = X.shape
    beta = cp.Variable(p)
    l1 = alpha * l1_ratio
    l2 = alpha * (1 - l1_ratio)
    objective = cp.Minimize(
        cp.sum_squares(y - X @ beta) / n
        + l1 * cp.norm1(beta)
        + l2 * cp.sum_squares(beta)
    )
    cp.Problem(objective).solve()
    return beta.value
