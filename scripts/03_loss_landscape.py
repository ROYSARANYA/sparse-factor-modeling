"""
Fix 5: Loss landscape 2D contour Ridge vs LASSO
Generates: outputs/loss_landscape.png

Visualizes and contrasts the objective function landscapes of Ridge and LASSO
regression over a 2D slice of the coefficient space (β_HML, β_CMA), holding
all other coefficients fixed at their LASSO-optimal values.

Ridge produces smooth elliptical contours whose minimum lies in the interior,
so both coefficients remain nonzero. LASSO's diamond-shaped L1 constraint
causes its contours to touch a corner of the constraint set, driving sparse
solutions — this directly explains why 14 of 25 CMA factor loadings are zeroed
out across rolling windows.

The output figure (outputs/loss_landscape.png) reproduces the analysis shown
in the report's loss-landscape section.
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '.')
from src.data_loader import load_all_data
from src.solvers import LassoProximal, RidgeScratch


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def load_and_scale(
    hml_idx: int = 2,
    cma_idx: int = 4,
) -> tuple:
    """
    Load the full factor/return dataset and standardize the feature matrix.

    Standardization (zero mean, unit variance per column) is applied so that
    regularization penalties are comparable across factors with different
    scales.

    Parameters
    ----------
    hml_idx : int, optional
        Column index of the HML (High-Minus-Low) factor in X. Default is 2.
    cma_idx : int, optional
        Column index of the CMA (Conservative-Minus-Aggressive) factor in X.
        Default is 4.

    Returns
    -------
    X_scaled : ndarray of shape (n_samples, n_features)
        Standardized feature matrix.
    y_vals : ndarray of shape (n_samples,)
        Target return series (first column of Y).
    factor_names : list of str
        Human-readable factor labels corresponding to X columns.
    """
    X, Y, factor_names, _ = load_all_data()
    X_vals = X.values
    X_scaled = (X_vals - X_vals.mean(0)) / X_vals.std(0)
    y_vals = Y.iloc[:, 0].values
    return X_scaled, y_vals, factor_names


# ---------------------------------------------------------------------------
# Solver fitting
# ---------------------------------------------------------------------------

def fit_optimal_coefficients(
    X_scaled: np.ndarray,
    y_vals: np.ndarray,
    lasso_alpha: float = 0.003,
    ridge_alpha: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit LASSO and Ridge models and return their optimal coefficient vectors.

    Parameters
    ----------
    X_scaled : ndarray of shape (n_samples, n_features)
        Standardized feature matrix.
    y_vals : ndarray of shape (n_samples,)
        Target return series.
    lasso_alpha : float, optional
        L1 regularization strength for LassoProximal. Default is 0.003.
    ridge_alpha : float, optional
        L2 regularization strength for RidgeScratch. Default is 0.1.

    Returns
    -------
    lasso_coef : ndarray of shape (n_features,)
        Fitted LASSO coefficient vector.
    ridge_coef : ndarray of shape (n_features,)
        Fitted Ridge coefficient vector.
    """
    lasso_coef = LassoProximal(alpha=lasso_alpha).fit(X_scaled, y_vals).coef_
    ridge_coef = RidgeScratch(alpha=ridge_alpha).fit(X_scaled, y_vals).coef_
    return lasso_coef, ridge_coef


# ---------------------------------------------------------------------------
# Objective surface computation
# ---------------------------------------------------------------------------

def compute_objective_grids(
    X_scaled: np.ndarray,
    y_vals: np.ndarray,
    b_fixed: np.ndarray,
    hml_idx: int,
    cma_idx: int,
    hml_range: tuple[float, float] = (-0.04, 0.015),
    cma_range: tuple[float, float] = (-0.025, 0.01),
    grid: int = 100,
    lasso_alpha: float = 0.003,
    ridge_alpha: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate Ridge and LASSO objectives over a 2D grid of (β_HML, β_CMA).

    All coefficients other than β_HML and β_CMA are held fixed at the values
    in ``b_fixed``. For each grid point, the mean squared error plus the
    appropriate regularization penalty is computed.

    Parameters
    ----------
    X_scaled : ndarray of shape (n_samples, n_features)
        Standardized feature matrix.
    y_vals : ndarray of shape (n_samples,)
        Target return series.
    b_fixed : ndarray of shape (n_features,)
        Base coefficient vector; β_HML and β_CMA are overwritten at each
        grid point while all other entries stay constant.
    hml_idx : int
        Column index of HML in the coefficient vector.
    cma_idx : int
        Column index of CMA in the coefficient vector.
    hml_range : tuple of (float, float), optional
        (min, max) range for the β_HML axis. Default is (-0.04, 0.015).
    cma_range : tuple of (float, float), optional
        (min, max) range for the β_CMA axis. Default is (-0.025, 0.01).
    grid : int, optional
        Number of points along each axis. Default is 100 (100×100 grid).
    lasso_alpha : float, optional
        L1 penalty weight used for the LASSO surface. Default is 0.003.
    ridge_alpha : float, optional
        L2 penalty weight used for the Ridge surface. Default is 0.1.

    Returns
    -------
    HH : ndarray of shape (grid, grid)
        Meshgrid of β_HML values.
    CC : ndarray of shape (grid, grid)
        Meshgrid of β_CMA values.
    ridge_obj : ndarray of shape (grid, grid)
        Ridge objective (MSE + L2 penalty) evaluated at each grid point.
    lasso_obj : ndarray of shape (grid, grid)
        LASSO objective (MSE + L1 penalty) evaluated at each grid point.
    """
    hml_g = np.linspace(*hml_range, grid)
    cma_g = np.linspace(*cma_range, grid)
    HH, CC = np.meshgrid(hml_g, cma_g)

    ridge_obj = np.zeros((grid, grid))
    lasso_obj = np.zeros((grid, grid))

    for i in range(grid):
        for j in range(grid):
            b = b_fixed.copy()
            b[hml_idx] = HH[i, j]
            b[cma_idx] = CC[i, j]
            mse = np.mean((y_vals - X_scaled @ b) ** 2)
            ridge_obj[i, j] = mse + ridge_alpha * np.sum(b ** 2)
            lasso_obj[i, j] = mse + lasso_alpha * np.sum(np.abs(b))

    return HH, CC, ridge_obj, lasso_obj


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_loss_landscape(
    HH: np.ndarray,
    CC: np.ndarray,
    ridge_obj: np.ndarray,
    lasso_obj: np.ndarray,
    ridge_coef: np.ndarray,
    lasso_coef: np.ndarray,
    hml_idx: int,
    cma_idx: int,
    lasso_alpha: float = 0.003,
    output_path: str = 'outputs/loss_landscape.png',
) -> None:
    """
    Render and save the side-by-side Ridge / LASSO loss landscape figure.

    The left panel shows Ridge contours (elliptical, interior minimum) with
    the Ridge optimum marked. The right panel shows LASSO contours with the
    L1 ball boundary overlaid, illustrating why the optimum is pushed onto
    an axis corner (β_CMA → 0).

    Parameters
    ----------
    HH : ndarray of shape (grid, grid)
        Meshgrid of β_HML values from ``compute_objective_grids``.
    CC : ndarray of shape (grid, grid)
        Meshgrid of β_CMA values from ``compute_objective_grids``.
    ridge_obj : ndarray of shape (grid, grid)
        Ridge objective surface.
    lasso_obj : ndarray of shape (grid, grid)
        LASSO objective surface.
    ridge_coef : ndarray of shape (n_features,)
        Ridge optimal coefficients (used to mark the optimum on the plot).
    lasso_coef : ndarray of shape (n_features,)
        LASSO optimal coefficients (used to mark the optimum and label).
    hml_idx : int
        Column index of HML, used to extract the plotted coefficient value.
    cma_idx : int
        Column index of CMA, used to extract the plotted coefficient value.
    lasso_alpha : float, optional
        L1 penalty weight; controls the size of the L1 ball drawn on the
        LASSO panel. Default is 0.003.
    output_path : str, optional
        File path where the figure is saved. Default is
        'outputs/loss_landscape.png'.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Ridge panel ---
    c1 = axes[0].contourf(HH, CC, ridge_obj, levels=30, cmap='Blues')
    axes[0].contour(HH, CC, ridge_obj, levels=15, colors='navy', alpha=0.4, lw=0.8)
    plt.colorbar(c1, ax=axes[0])
    axes[0].plot(
        ridge_coef[hml_idx], ridge_coef[cma_idx],
        'r*', ms=15, zorder=5, label='Ridge optimum',
    )
    axes[0].axhline(0, color='k', lw=0.8, ls='--', alpha=0.5)
    axes[0].axvline(0, color='k', lw=0.8, ls='--', alpha=0.5)
    axes[0].set_xlabel('β_HML')
    axes[0].set_ylabel('β_CMA')
    axes[0].set_title(
        'Ridge: Elliptical contours\nInterior solution — both nonzero',
        fontweight='bold',
    )
    axes[0].legend()

    # --- LASSO panel ---
    c2 = axes[1].contourf(HH, CC, lasso_obj, levels=30, cmap='Reds')
    axes[1].contour(HH, CC, lasso_obj, levels=15, colors='darkred', alpha=0.4, lw=0.8)
    plt.colorbar(c2, ax=axes[1])
    axes[1].plot(
        lasso_coef[hml_idx], lasso_coef[cma_idx],
        'b*', ms=15, zorder=5,
        label=f'LASSO: β_CMA={lasso_coef[cma_idx]:.4f}',
    )
    axes[1].axhline(0, color='k', lw=1.5, alpha=0.7)
    axes[1].axvline(0, color='k', lw=1.5, alpha=0.7)
    r = lasso_alpha
    axes[1].plot(
        [r, 0, -r, 0, r], [0, r, 0, -r, 0],
        'b-', lw=2, alpha=0.5, label='L1 ball',
    )
    axes[1].set_xlabel('β_HML')
    axes[1].set_ylabel('β_CMA')
    axes[1].set_title(
        'LASSO: L1 corner forces β_CMA=0\nExplains 14/25 CMA dropout',
        fontweight='bold',
    )
    axes[1].legend()

    plt.suptitle(
        'Loss Landscape: Ridge vs LASSO over (β_HML, β_CMA)\n'
        'HML-CMA correlation=0.632',
        fontsize=13, fontweight='bold',
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {output_path}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Orchestrate data loading, model fitting, grid evaluation, and plotting.

    Workflow
    --------
    1. Load and standardize the factor dataset.
    2. Fit LASSO (α=0.003) and Ridge (α=0.1) on the full sample to obtain
       optimal coefficient vectors.
    3. Sweep a 100×100 grid over (β_HML, β_CMA), holding other coefficients
       at their LASSO-optimal values, and evaluate both objective surfaces.
    4. Render the side-by-side contour figure and save to
       outputs/loss_landscape.png.
    """
    HML_IDX, CMA_IDX = 2, 4
    LASSO_ALPHA, RIDGE_ALPHA = 0.003, 0.1

    X_scaled, y_vals, factor_names = load_and_scale(HML_IDX, CMA_IDX)

    lasso_coef, ridge_coef = fit_optimal_coefficients(
        X_scaled, y_vals, lasso_alpha=LASSO_ALPHA, ridge_alpha=RIDGE_ALPHA
    )
    print(f'LASSO: beta_HML={lasso_coef[HML_IDX]:.5f}, beta_CMA={lasso_coef[CMA_IDX]:.5f}')
    print(f'Ridge: beta_HML={ridge_coef[HML_IDX]:.5f}, beta_CMA={ridge_coef[CMA_IDX]:.5f}')

    HH, CC, ridge_obj, lasso_obj = compute_objective_grids(
        X_scaled, y_vals,
        b_fixed=lasso_coef.copy(),
        hml_idx=HML_IDX, cma_idx=CMA_IDX,
        lasso_alpha=LASSO_ALPHA, ridge_alpha=RIDGE_ALPHA,
    )

    plot_loss_landscape(
        HH, CC, ridge_obj, lasso_obj,
        ridge_coef, lasso_coef,
        hml_idx=HML_IDX, cma_idx=CMA_IDX,
        lasso_alpha=LASSO_ALPHA,
    )


if __name__ == '__main__':
    main()
      
