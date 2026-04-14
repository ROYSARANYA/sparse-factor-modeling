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


class FISTALasso:
    """
    FISTA (Fast Iterative Shrinkage-Thresholding Algorithm).
    Achieves O(1/t^2) convergence vs O(1/t) for proximal GD.
    Beck & Teboulle (2009).
    """
    def __init__(self, alpha=1.0, lr=None, max_iter=1000, tol=1e-6):
        self.alpha    = alpha
        self.lr       = lr
        self.max_iter = max_iter
        self.tol      = tol
        self.coef_    = None
        self.loss_history_     = []
        self.duality_gap_hist_ = []
        self.n_iter_  = 0

    @staticmethod
    def _soft_threshold(x, threshold):
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

    def _duality_gap(self, X, y, beta, n):
        residual = y - X @ beta
        primal   = np.mean(residual**2) + self.alpha * np.sum(np.abs(beta))
        Xtr      = X.T @ residual / n
        scale    = max(1.0, np.max(np.abs(Xtr)) / self.alpha)
        nu       = residual / scale
        dual     = -np.mean(nu**2) + np.mean(y * nu)
        return primal - dual

    def fit(self, X, y):
        n, p      = X.shape
        beta      = np.zeros(p)
        beta_prev = np.zeros(p)
        lr        = self.lr or 1.0 / (2 * np.linalg.norm(X.T @ X, ord=2) / n)
        t         = 1.0

        self.loss_history_     = []
        self.duality_gap_hist_ = []

        for i in range(self.max_iter):
            momentum = (t - 1) / (t + 2)
            y_mom    = beta + momentum * (beta - beta_prev)
            residual = y - X @ y_mom
            grad     = -2.0 / n * X.T @ residual
            beta_new = self._soft_threshold(y_mom - lr * grad, self.alpha * lr)

            loss = np.mean((y - X @ beta_new)**2) + self.alpha * np.sum(np.abs(beta_new))
            self.loss_history_.append(loss)

            if i % 10 == 0:
                gap = self._duality_gap(X, y, beta_new, n)
                self.duality_gap_hist_.append((i, gap))
                if gap < self.tol:
                    beta_prev = beta.copy()
                    beta      = beta_new
                    t        += 1
                    self.n_iter_ = i + 1
                    break

            if np.linalg.norm(beta_new - beta) < self.tol:
                self.n_iter_ = i + 1
                break

            beta_prev = beta.copy()
            beta      = beta_new
            t        += 1

        self.coef_ = beta
        return self

    def predict(self, X):
        return X @ self.coef_


class WarmStartLasso:
    """
    LASSO with warm starting across the regularization path.
    Uses the previous lambda solution as the starting point
    for the next lambda — much faster than cold starting.
    """
    def __init__(self, alphas=None, max_iter=1000, tol=1e-6):
        self.alphas       = alphas
        self.max_iter     = max_iter
        self.tol          = tol
        self.coef_path_   = []
        self.alpha_path_  = []
        self.iter_counts_ = []

    @staticmethod
    def _soft_threshold(x, threshold):
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

    def fit_path(self, X, y):
        n, p   = X.shape
        alphas = self.alphas if self.alphas is not None else np.logspace(-3, 0, 50)[::-1]
        beta   = np.zeros(p)
        lr     = 1.0 / (2 * np.linalg.norm(X.T @ X, ord=2) / n)

        self.coef_path_   = []
        self.alpha_path_  = list(alphas)
        self.iter_counts_ = []

        for alpha in alphas:
            for i in range(self.max_iter):
                residual = y - X @ beta
                grad     = -2.0 / n * X.T @ residual
                beta_new = self._soft_threshold(beta - lr * grad, alpha * lr)
                if np.linalg.norm(beta_new - beta) < self.tol:
                    self.iter_counts_.append(i + 1)
                    break
                beta = beta_new
            else:
                self.iter_counts_.append(self.max_iter)

            beta = beta_new
            self.coef_path_.append(beta.copy())

        return self

    def get_path_array(self):
        return np.array(self.coef_path_)
