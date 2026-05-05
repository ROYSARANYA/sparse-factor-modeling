"""
Fix 4: DRO with data-driven epsilon=0.091
Generates: Table 6 numbers in report

This script benchmarks Distributionally Robust Optimization (DRO) with Lasso
regularization across a range of epsilon (uncertainty ball radius) values,
comparing against a standard Lasso baseline. The data-driven epsilon of 0.091
is highlighted in the output, as it corresponds to the optimal uncertainty
radius estimated from empirical data and is the value reported in Table 6.
"""

import sys
sys.path.insert(0, '.')

from src.data_loader import load_all_data
from src.backtest import walk_forward_backtest, compute_metrics
from src.dro_solver import DROLasso
from src.solvers import LassoProximal


def run_dro_epsilon_sweep(
    X,
    Y,
    epsilons: list[float],
    alpha: float = 0.003,
) -> list[dict]:
    """
    Run a walk-forward backtest for each epsilon value and collect metrics.

    For epsilon=0.0, a standard LassoProximal solver is used as the baseline.
    For all other epsilon values, DROLasso is used with the given uncertainty
    ball radius.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Factor/feature matrix aligned with Y.
    Y : array-like of shape (n_samples,)
        Target return series to predict.
    epsilons : list of float
        Uncertainty ball radii to sweep over. Use 0.0 to include the plain
        Lasso baseline (no distributional robustness).
    alpha : float, optional
        Lasso regularization strength applied to all solvers. Default is 0.003.

    Returns
    -------
    list of dict
        One entry per epsilon, each containing:
        - 'epsilon'  : float  — the epsilon value used
        - 'metrics'  : dict   — output of compute_metrics (OOS_R2, Sharpe, ICIR, …)
        - 'preds'    : array  — out-of-sample predictions
        - 'actuals'  : array  — realized returns
        - 'dates'    : array  — corresponding dates
    """
    results = []
    for eps in epsilons:
        if eps == 0.0:
            preds, actuals, dates, _ = walk_forward_backtest(
                X, Y, LassoProximal, alpha=alpha
            )
        else:
            preds, actuals, dates, _ = walk_forward_backtest(
                X, Y, DROLasso, alpha=alpha, epsilon=eps
            )
        metrics = compute_metrics(preds, actuals)
        results.append(
            {"epsilon": eps, "metrics": metrics, "preds": preds,
             "actuals": actuals, "dates": dates}
        )
    return results


def print_results_table(results: list[dict], data_driven_eps: float = 0.091) -> None:
    """
    Print a formatted results table to stdout (mirrors Table 6 in the report).

    Rows corresponding to the data-driven epsilon are annotated with
    '<-- data-driven' to make the recommended operating point explicit.

    Parameters
    ----------
    results : list of dict
        Output of ``run_dro_epsilon_sweep``.
    data_driven_eps : float, optional
        The epsilon value to flag as data-driven. Default is 0.091.
    """
    print(f'{"epsilon":>10} {"OOS R2":>10} {"Sharpe":>10} {"ICIR":>10}')
    print('=' * 45)
    for row in results:
        eps = row["epsilon"]
        m = row["metrics"]
        note = '<-- data-driven' if abs(eps - data_driven_eps) < 1e-9 else ''
        print(
            f'{eps:>10.3f} {m["OOS_R2"]:>10} {m["Sharpe"]:>10}'
            f' {m["ICIR"]:>10} {note}'
        )


def main() -> None:
    """
    Entry point: load data, sweep epsilon values, and print Table 6.

    Loads the full factor/return dataset, runs walk-forward backtests for
    epsilon in {0.0, 0.01, 0.05, 0.091, 0.15}, and prints out-of-sample R²,
    annualized Sharpe ratio, and IC information ratio for each setting.
    """
    X, Y, factor_names, _ = load_all_data()

    epsilons = [0.0, 0.01, 0.05, 0.091, 0.15]
    print('DRO with data-driven epsilon = 0.091')

    results = run_dro_epsilon_sweep(X, Y, epsilons, alpha=0.003)
    print_results_table(results, data_driven_eps=0.091)


if __name__ == '__main__':
    main()
