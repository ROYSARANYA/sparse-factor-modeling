"""
Distributionally Robust Optimization (DRO) for Factor Models.

Uses the correct Wasserstein DRO formulation from
Esfahani & Kuhn (2018) "Data-driven distributionally robust
optimization using the Wasserstein metric".

The key insight: DRO adds a penalty on the worst-case
perturbation of each training sample, controlled by epsilon.
Larger epsilon = model must be robust to larger distribution shifts.

Tractable dual formulation:
minimize  (1/n) sum_i max_delta [ loss(x_i + delta, y_i, beta)
          - lambda * ||delta||] + lambda * epsilon + alpha * ||beta||_1
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
