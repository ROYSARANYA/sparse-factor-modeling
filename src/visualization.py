"""
Visualization utilities for the project.
All plots used in the report are generated here.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

COLORS       = ['#2E74B5', '#E74C3C', '#27AE60', '#F39C12', '#8E44AD', '#1ABC9C']
FACTOR_NAMES = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Mom']


def plot_correlation_matrix(X, factor_names=FACTOR_NAMES, save_path=None):
    import seaborn as sns
    corr = X.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, ax=ax, square=True,
                xticklabels=factor_names,
                yticklabels=factor_names)
    ax.set_title('Factor Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_regularization_path(X_vals, y_vals, model_class, lambdas,
                              title='Regularization Path',
                              factor_names=FACTOR_NAMES,
                              save_path=None, **model_kwargs):
    coef_paths = []
    for lam in lambdas:
        m = model_class(alpha=lam, **model_kwargs).fit(X_vals, y_vals)
        coef_paths.append(m.coef_.copy())
    coef_paths = np.array(coef_paths)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, name in enumerate(factor_names):
        ax.plot(np.log10(lambdas), coef_paths[:, i],
                label=name, color=COLORS[i], linewidth=2)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel('log10(lambda)', fontsize=12)
    ax.set_ylabel('Coefficient Value', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_convergence(loss_histories, title='Convergence of Proximal Gradient Descent',
                     save_path=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (label, history) in enumerate(loss_histories.items()):
        ax.plot(history, label=label, color=COLORS[i], linewidth=2)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Objective Value', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_sparsity_vs_performance(sparsity, r2_scores,
                                  title='Sparsity vs Out-of-Sample R2',
                                  save_path=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(sparsity, r2_scores, color=COLORS[0], alpha=0.7, s=40)
    ax.set_xlabel('Number of Nonzero Coefficients', fontsize=12)
    ax.set_ylabel('Out-of-Sample R2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_cumulative_ls_returns(dates, ls_returns_dict,
                                title='Cumulative Long-Short Portfolio Returns',
                                save_path=None):
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (label, returns) in enumerate(ls_returns_dict.items()):
        cumulative = np.cumprod(1 + np.array(returns)) - 1
        ax.plot(dates, cumulative, label=label,
                color=COLORS[i], linewidth=2)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative Return', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_coefs_over_time(dates, coefs_over_time, portfolio_idx=0,
                          factor_names=FACTOR_NAMES,
                          title='Factor Coefficients Over Time',
                          save_path=None):
    # coefs_over_time: (N x 25 x 6)
    coefs = coefs_over_time[:, portfolio_idx, :]
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, name in enumerate(factor_names):
        ax.plot(dates, coefs[:, i], label=name,
                color=COLORS[i], linewidth=1.5)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Coefficient Value', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig
