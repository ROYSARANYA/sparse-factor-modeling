"""
Online Proximal Gradient Descent for LASSO.
Instead of retraining from scratch each period,
updates the model incrementally as new data arrives.

Key difference from batch:
- Batch: retrain on full window every month O(n*p) per step
- Online: update using only the new observation O(p) per step

This is how production trading systems actually work.
"""
import numpy as np
import time


class OnlineLasso:
    """
    Online LASSO via incremental proximal gradient updates.
    Processes one observation at a time using a decaying
    learning rate schedule.

    Solves: minimize sum_t (y_t - x_t'b)^2 + alpha * ||b||_1
    """
    def __init__(self, alpha=0.003, lr_init=0.01,
                 decay=0.01, max_iter=10):
        self.alpha    = alpha
        self.lr_init  = lr_init
        self.decay    = decay
        self.max_iter = max_iter
        self.coef_    = None
        self.t_       = 0
        self.loss_history_ = []

    @staticmethod
    def _soft_threshold(x, threshold):
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

    def initialize(self, p):
        """Initialize coefficients to zero."""
        self.coef_ = np.zeros(p)
        self.t_    = 0
        return self

    def update(self, x, y):
        """
        Process one new observation (x, y).
        Updates coefficients using proximal gradient step.
        """
        if self.coef_ is None:
            self.initialize(len(x))

        self.t_ += 1
        # Decaying learning rate
        lr       = self.lr_init / (1 + self.decay * self.t_)

        for _ in range(self.max_iter):
            # Gradient on single observation
            residual  = y - np.dot(x, self.coef_)
            grad      = -2 * residual * x
            beta_half = self.coef_ - lr * grad
            # Proximal step
            self.coef_ = self._soft_threshold(
                beta_half, self.alpha * lr
            )

        loss = (y - np.dot(x, self.coef_))**2 + \
               self.alpha * np.sum(np.abs(self.coef_))
        self.loss_history_.append(loss)
        return self

    def predict(self, x):
        return np.dot(x, self.coef_)


def walk_forward_online(X, Y, alpha=0.003,
                         train_window=120, step=1):
    """
    Online walk-forward backtest.
    Warms up on first train_window observations,
    then updates incrementally.

    Returns same format as walk_forward_backtest
    for direct comparison.
    """
    T        = len(X)
    X_vals   = X.values
    X_mean   = X_vals.mean(axis=0)
    X_std    = X_vals.std(axis=0)
    X_scaled = (X_vals - X_mean) / X_std
    Y_vals   = Y.values

    predictions, actuals, dates = [], [], []
    coefs_over_time              = []

    # One online model per portfolio
    models = [OnlineLasso(alpha=alpha)
              for _ in range(Y.shape[1])]

    # Warm up on first train_window observations
    for t in range(train_window):
        for j in range(Y.shape[1]):
            models[j].update(X_scaled[t], Y_vals[t, j])

    # Online updates
    for t in range(train_window, T, step):
        X_test = X_scaled[t:t + 1]
        Y_test = Y_vals[t:t + 1]

        preds, coefs = [], []
        for j in range(Y.shape[1]):
            pred = models[j].predict(X_scaled[t])
            preds.append(pred)
            coefs.append(models[j].coef_.copy())
            # Update with new observation
            models[j].update(X_scaled[t], Y_vals[t, j])

        predictions.append(preds)
        actuals.append(Y_test[0])
        dates.append(Y.index[t])
        coefs_over_time.append(coefs)

    return (np.array(predictions), np.array(actuals),
            dates, np.array(coefs_over_time))
