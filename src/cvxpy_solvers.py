"""
Explicit convex program formulations using CVXPY.
These are the rigorous convex optimization formulations
required by the course.
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
