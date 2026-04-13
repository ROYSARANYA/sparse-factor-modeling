"""
From-scratch implementations of Ridge, LASSO, and Elastic Net
as convex optimization programs using proximal gradient descent.
"""
import numpy as np


class RidgeScratch:
    """
    Ridge Regression via closed-form solution.
    Solves: minimize (1/n)||y - Xb||^2 + alpha * ||b||^2
    """
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.coef_ = None

    def fit(self, X, y):
        n, p = X.shape
        I = np.eye(p)
        self.coef_ = np.linalg.solve(X.T @ X + self.alpha * I, X.T @ y)
        return self

    def predict(self, X):
        return X @ self.coef_


class LassoProximal:
    """
    LASSO via Proximal Gradient Descent.
    Solves: minimize (1/n)||y - Xb||^2 + alpha * ||b||_1
    Uses soft-thresholding as the proximal operator of the L1 norm.
    """
    def __init__(self, alpha=1.0, lr=None, max_iter=1000, tol=1e-6):
        self.alpha = alpha
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.loss_history_ = []
        self.n_iter_ = 0

    @staticmethod
    def _soft_threshold(x, threshold):
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

    def fit(self, X, y):
        n, p = X.shape
        beta = np.zeros(p)
        lr = self.lr or 1.0 / (2 * np.linalg.norm(X.T @ X, ord=2) / n)

        self.loss_history_ = []
        for i in range(self.max_iter):
            residual = y - X @ beta
            grad = -2.0 / n * X.T @ residual
            beta_half = beta - lr * grad
            beta_new = self._soft_threshold(beta_half, self.alpha * lr)
            loss = (np.mean(residual ** 2)
                    + self.alpha * np.sum(np.abs(beta_new)))
            self.loss_history_.append(loss)
            if np.linalg.norm(beta_new - beta) < self.tol:
                self.n_iter_ = i + 1
                break
            beta = beta_new

        self.coef_ = beta
        return self

    def predict(self, X):
        return X @ self.coef_


class ElasticNetScratch:
    """
    Elastic Net via Proximal Gradient Descent.
    Solves: minimize (1/n)||y-Xb||^2 + l1*||b||_1 + l2*||b||^2
    where l1 = alpha*l1_ratio, l2 = alpha*(1-l1_ratio)
    """
    def __init__(self, alpha=1.0, l1_ratio=0.5, lr=None,
                 max_iter=1000, tol=1e-6):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.loss_history_ = []
        self.n_iter_ = 0

    def fit(self, X, y):
        n, p = X.shape
        beta = np.zeros(p)
        l1 = self.alpha * self.l1_ratio
        l2 = self.alpha * (1 - self.l1_ratio)
        lr = self.lr or 1.0 / (2 * np.linalg.norm(X.T @ X, ord=2) / n + 2 * l2)

        self.loss_history_ = []
        for i in range(self.max_iter):
            residual = y - X @ beta
            grad = -2.0 / n * X.T @ residual + 2 * l2 * beta
            beta_half = beta - lr * grad
            beta_new = (np.sign(beta_half)
                        * np.maximum(np.abs(beta_half) - lr * l1, 0)
                        / (1 + 2 * lr * l2))
            loss = (np.mean(residual ** 2)
                    + l1 * np.sum(np.abs(beta_new))
                    + l2 * np.sum(beta_new ** 2))
            self.loss_history_.append(loss)
            if np.linalg.norm(beta_new - beta) < self.tol:
                self.n_iter_ = i + 1
                break
            beta = beta_new

        self.coef_ = beta
        return self

    def predict(self, X):
        return X @ self.coef_
