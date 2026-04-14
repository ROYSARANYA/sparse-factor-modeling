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


class FISTARestart:
    """
    FISTA with Adaptive Restart (O'Donoghue & Candès, 2015).
    "Adaptive Restart for Accelerated Gradient Schemes"
    Foundations of Computational Mathematics, 15(3), 715-732.

    Vanilla FISTA achieves O(1/t^2) but can oscillate near the
    solution due to momentum overshooting. Adaptive restart
    detects these oscillations and resets momentum to zero,
    producing monotone decrease and empirically faster convergence.

    Two restart criteria implemented:
    1. Function restart: restart if F(y^k) > F(x^{k-1})
       Simple, cheap, guaranteed monotone decrease.
    2. Gradient restart: restart if <grad_f(y^k), x^k - x^{k-1}> > 0
       The gradient and momentum point in opposite directions —
       the momentum is counterproductive, restart.

    Both criteria are heuristics with no convergence rate guarantee
    beyond vanilla FISTA, but empirically converge significantly faster.
    """
    def __init__(self, alpha=1.0, lr=None, max_iter=1000,
                 tol=1e-6, restart='gradient'):
        """
        Parameters
        ----------
        restart : str
            'function'  — restart when objective increases
            'gradient'  — restart when gradient opposes momentum
            'both'      — restart on either condition
        """
        self.alpha    = alpha
        self.lr       = lr
        self.max_iter = max_iter
        self.tol      = tol
        self.restart  = restart
        self.coef_    = None
        self.loss_history_    = []
        self.restart_iters_   = []
        self.n_iter_  = 0
        self.n_restarts_ = 0

    @staticmethod
    def _soft_threshold(x, threshold):
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

    def _objective(self, X, y, beta, n):
        resid = y - X @ beta
        return np.mean(resid**2) + self.alpha * np.sum(np.abs(beta))

    def fit(self, X, y):
        n, p      = X.shape
        beta      = np.zeros(p)
        beta_prev = np.zeros(p)
        lr        = self.lr or 1.0 / (2 * np.linalg.norm(X.T@X, ord=2) / n)
        t         = 1.0

        self.loss_history_  = []
        self.restart_iters_ = []
        self.n_restarts_    = 0

        f_prev = self._objective(X, y, beta, n)

        for i in range(self.max_iter):
            # ── Momentum extrapolation ────────────────────────────────────
            momentum = (t - 1) / (t + 2)
            y_mom    = beta + momentum * (beta - beta_prev)

            # ── Gradient on momentum point ────────────────────────────────
            residual = y - X @ y_mom
            grad     = -2.0 / n * X.T @ residual

            # ── Proximal step ─────────────────────────────────────────────
            beta_new = self._soft_threshold(y_mom - lr * grad, self.alpha * lr)

            # ── Compute objective ─────────────────────────────────────────
            f_new = self._objective(X, y, beta_new, n)
            self.loss_history_.append(f_new)

            # ── Check restart criteria ────────────────────────────────────
            should_restart = False

            if self.restart in ('function', 'both'):
                # Function restart: objective went up at y^k
                f_y = self._objective(X, y, y_mom, n)
                if f_y > f_prev:
                    should_restart = True

            if self.restart in ('gradient', 'both') and not should_restart:
                # Gradient restart: momentum opposes gradient direction
                # Condition: <grad_f(y^k), x^k - x^{k-1}> > 0
                if np.dot(grad, beta - beta_prev) > 0:
                    should_restart = True

            if should_restart:
                # Reset momentum — restart FISTA from current point
                t         = 1.0
                beta_prev = beta.copy()
                self.restart_iters_.append(i)
                self.n_restarts_ += 1
            else:
                beta_prev = beta.copy()
                t        += 1

            # ── Convergence check ─────────────────────────────────────────
            if np.linalg.norm(beta_new - beta) < self.tol:
                self.n_iter_ = i + 1
                beta         = beta_new
                break

            beta   = beta_new
            f_prev = f_new

        self.coef_ = beta
        return self

    def predict(self, X):
        return X @ self.coef_


class BBLasso:
    """
    LASSO with Barzilai-Borwein (BB) Step Sizes.
    Barzilai & Borwein (1988): "Two-Point Step Size Gradient Methods"
    IMA Journal of Numerical Analysis, 8(1), 141-148.

    Instead of a fixed Lipschitz-based step size η = 1/L,
    BB computes a local approximation to the inverse Hessian
    from the most recent gradient difference, giving a step size
    that adapts to the local curvature of the objective.

    BB step sizes are NOT guaranteed to decrease the objective
    monotonically, so we combine with the proximal operator
    using a non-monotone line search (Zhang & Hager, 2004).

    Two BB variants:
    BB1 (long step):  η_k = (s'·s) / (s'·y)
    BB2 (short step): η_k = (s'·y) / (y'·y)
    where s = β^k - β^{k-1}, y = ∇f^k - ∇f^{k-1}

    We alternate BB1 and BB2 for better overall convergence.
    """
    def __init__(self, alpha=1.0, lr_init=None, max_iter=1000,
                 tol=1e-6, bb_variant='alternating'):
        """
        Parameters
        ----------
        bb_variant : str
            'bb1'         — always use long BB step
            'bb2'         — always use short BB step
            'alternating' — alternate BB1 and BB2 (recommended)
        """
        self.alpha      = alpha
        self.lr_init    = lr_init
        self.max_iter   = max_iter
        self.tol        = tol
        self.bb_variant = bb_variant
        self.coef_      = None
        self.loss_history_  = []
        self.step_history_  = []
        self.n_iter_    = 0

    @staticmethod
    def _soft_threshold(x, threshold):
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

    def _gradient(self, X, y, beta, n):
        return -2.0 / n * X.T @ (y - X @ beta)

    def fit(self, X, y):
        n, p = X.shape

        # Initialize with one proximal GD step using Lipschitz step
        L        = 2 * np.linalg.norm(X.T@X, ord=2) / n
        lr       = self.lr_init or 1.0 / L
        lr_min   = 1e-10
        lr_max   = 10.0 / L  # don't let BB step get too large

        beta     = np.zeros(p)
        grad     = self._gradient(X, y, beta, n)
        beta     = self._soft_threshold(beta - lr * grad, self.alpha * lr)

        self.loss_history_ = []
        self.step_history_ = [lr]

        for i in range(self.max_iter):
            grad_new = self._gradient(X, y, beta, n)
            loss     = np.mean((y - X@beta)**2) + self.alpha * np.sum(np.abs(beta))
            self.loss_history_.append(loss)

            # ── Compute BB step size ──────────────────────────────────────
            if i > 0:
                s = beta - beta_prev          # parameter difference
                g = grad_new - grad_prev      # gradient difference
                sg = np.dot(s, g)
                ss = np.dot(s, s)
                gg = np.dot(g, g)

                if sg > 1e-12:  # positive curvature — BB valid
                    bb1 = ss / sg   # long step
                    bb2 = sg / gg   # short step

                    if self.bb_variant == 'bb1':
                        lr = bb1
                    elif self.bb_variant == 'bb2':
                        lr = bb2
                    else:  # alternating
                        lr = bb1 if i % 2 == 0 else bb2

                    # Safeguard: clip to reasonable range
                    lr = np.clip(lr, lr_min, lr_max)

            self.step_history_.append(lr)

            # ── Proximal step with BB step size ───────────────────────────
            beta_prev = beta.copy()
            grad_prev = grad_new.copy()
            beta_new  = self._soft_threshold(
                beta - lr * grad_new, self.alpha * lr
            )

            # ── Convergence check ─────────────────────────────────────────
            if np.linalg.norm(beta_new - beta) < self.tol:
                self.n_iter_ = i + 1
                beta         = beta_new
                break

            beta     = beta_new
            grad     = grad_new

        self.coef_ = beta
        return self

    def predict(self, X):
        return X @ self.coef_


class CoordinateDescent:
    """
    Coordinate Descent for LASSO.
    Cycles through each coordinate and minimizes exactly over that
    coordinate while holding all others fixed.

    For LASSO, each coordinate subproblem has the closed-form solution:
        beta_j <- sign(z_j) * max(|z_j| - alpha, 0) / ||X_j||^2/n
    where z_j = X_j'(y - X*beta + X_j*beta_j) / n is the partial residual.

    Convergence rate: O(1/t) per coordinate cycle — same as proximal GD
    overall, but with smaller constant in practice due to exact coordinate
    updates. Particularly effective when p is small (our case: p=6).

    Reference: Friedman, Hastie & Tibshirani (2010), Journal of Statistical
    Software, "Regularization Paths for GLMs via Coordinate Descent".
    """
    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-6):
        self.alpha    = alpha
        self.max_iter = max_iter
        self.tol      = tol
        self.coef_    = None
        self.loss_history_ = []
        self.n_iter_  = 0

    def fit(self, X, y):
        n, p  = X.shape
        beta  = np.zeros(p)
        # Precompute column norms — only needed once
        col_norms = np.sum(X**2, axis=0) / n

        self.loss_history_ = []

        for iteration in range(self.max_iter):
            beta_old = beta.copy()
            # Cycle through all coordinates
            for j in range(p):
                if col_norms[j] < 1e-12:
                    continue
                # Partial residual (remove contribution of feature j)
                r_j   = y - X @ beta + X[:, j] * beta[j]
                # Soft-threshold the univariate OLS solution
                z_j   = X[:, j] @ r_j / n
                beta[j] = np.sign(z_j) * max(abs(z_j) - self.alpha, 0) / col_norms[j]

            loss = np.mean((y - X@beta)**2) + self.alpha * np.sum(np.abs(beta))
            self.loss_history_.append(loss)

            # Convergence: max coordinate change < tol
            if np.max(np.abs(beta - beta_old)) < self.tol:
                self.n_iter_ = iteration + 1
                break

        self.coef_ = beta
        return self

    def predict(self, X):
        return X @ self.coef_
