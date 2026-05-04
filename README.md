# Sparse Factor Modeling for Equity Return Prediction

**Authors:** Renuka Oladri · Saranya Roy

---

## Problem Statement

Every month, 25 groups of US stocks each produce a return. The objective is to predict
which groups will outperform using six Fama-French economic factors as predictors.

The core challenge is multicollinearity — HML (value) and CMA (investment) have a
0.632 correlation, making ordinary least squares unstable. Small changes in the data
cause large swings in coefficient estimates.

The solution is to reformulate regression as a **convex optimization problem** with an
L1 penalty (LASSO). This simultaneously stabilizes estimation and performs automatic
variable selection — setting irrelevant factor coefficients to exactly zero.

```
Factors (X)          Sparse Solver          Portfolios (Y)
                                            
  Mkt-RF  ─────────►                ──────► P1  (Small-Growth)
  SMB     ─────────►   min ||y-Xβ||²  ───► P5
  HML     ─────────►   + α||β||₁    ──────► P13
  RMW     ─────────►                ──────► P25 (Large-Value)
  CMA  ── ╌ ╌ ╌ ╌ ►   [LASSO drops  ╌ ╌ ╌►
  Mom  ── ╌ ╌ ╌ ╌ ►    CMA & Mom]   ╌ ╌ ╌►

  n = 288 months · p = 6 factors · 25 portfolios · 2000–2023
```

---

## What We Built

Six first-order optimization algorithms implemented from scratch, with formal
convergence guarantees, validated on 25 Fama-French portfolios over 14 years.

```
Algorithm            Iterations    Wall-clock    vs Baseline
─────────────────────────────────────────────────────────────
Proximal GD          42.2          0.444 ms      1.00×  (baseline)
FISTA vanilla        37.3          0.557 ms      1.13×
FISTA + restart      24.0          0.460 ms      1.76×
BB LASSO             15.1          0.259 ms      2.79×  (fastest)
Coordinate Descent    9.2          0.251 ms      4.59×
Online LASSO         streaming     8.1× faster   5.061 Sharpe
```

---

## Key Results

```
Metric                        Value        Method
──────────────────────────────────────────────────────
Best Sharpe Ratio             5.061        Online LASSO
OOS R²                        0.900        Ridge
Information Coefficient       0.614        All methods
ICIR                          2.702        Online LASSO
Annual Return                 59.14%       Ridge
Breakeven AUM                 $1.265B      LASSO
Survives 50bps costs          Sharpe 4.14  LASSO
BB speedup at p=193           6.4×         BB LASSO
```

### Three Novel Contributions

**1. FISTA Degrades at High Sparsity**

```
Alpha     Active Factors    FISTA/PGD    BB/PGD
────────────────────────────────────────────────
0.001          5.4           1.13×        2.98×
0.003          4.7           1.13×        2.79×
0.020          2.8           1.02×        2.59×
0.050          1.8           0.95×  ←     2.85×
              FISTA SLOWER THAN BASELINE
```

Theory predicts sqrt(κ) = 2.84×. At α=0.05 FISTA achieves 0.95× — slower than
plain gradient descent. The L1 term dominates the objective at high sparsity,
making momentum extrapolation counterproductive.

**2. KKT Factor Importance Prediction**

Factor importance order is analytically predictable from the correlation matrix
alone — no optimization required:

```
|[C⁻¹ c_k]_j| < τ  →  factor j excluded for portfolio k

Mean Spearman ρ = 0.906 across 25 portfolios
22/25 portfolios above ρ = 0.75

Factor      Prediction Accuracy
────────────────────────────────
Mkt-RF           100%
SMB               96%
HML               96%
RMW               88%
CMA               44%   ← near threshold
Mom               48%   ← near threshold
```

**3. BB Step Size Bounds (Theorem 4)**

Under positive curvature (s'g > 0), BB step sizes satisfy 1/L ≤ η_k ≤ 1/μ.
Verified empirically: 0 violations across 1,000 measurements.
Maximum observed step: 1.944440 vs bound 1.944441 — machine precision tight.

---

## Repository Structure

```
sparse-factor-modeling/
│
├── src/                          Core implementation
│   ├── solvers.py                9 optimization algorithms from scratch
│   │                               Ridge, LASSO, Elastic Net, FISTA,
│   │                               FISTA+restart, BB LASSO, Coord Descent,
│   │                               Online LASSO, Momentum-BB
│   ├── backtest.py               Walk-forward backtesting engine
│   │                               walk_forward_backtest
│   │                               compute_metrics, compute_metrics_with_costs
│   │                               compute_alpha_decay
│   │                               walk_forward_backtest_adaptive
│   ├── online_solver.py          Online LASSO with regret bound O(sqrt(T))
│   ├── dro_solver.py             Wasserstein distributionally robust LASSO
│   ├── cross_validation.py       Expanding-window time-series CV
│   ├── regime.py                 Volatility regime classification
│   ├── interactions.py           Pairwise factor interaction features
│   ├── signal_discovery.py       Price/volume signal construction
│   ├── data_loader.py            Fama-French data loader
│   └── cvxpy_solvers.py          CVXPY reference implementations
│
├── scripts/                      Reproducibility scripts
│   ├── 01_wall_clock_comparison.py     Table 2: algorithm timing
│   ├── 02_dro_calibration.py           DRO epsilon calibration
│   ├── 03_loss_landscape.py            Ridge vs LASSO geometry
│   ├── 04_dollar_neutral_sharpe.py     Portfolio construction variants
│   ├── 05_mv_optimizer.py              Mean-variance optimizer
│   ├── 06_capacity_analysis.py         AUM breakeven curve
│   ├── 07_factor_crowding.py           Factor crowding analysis
│   ├── 08_kkt_dropout_prediction.py    KKT sparsity prediction
│   ├── 09_novelty_momentum_bb.py       Momentum-BB convergence
│   ├── 10_novelty_algorithm_theory.py  Algorithm selection theory
│   ├── 11_time_series_analysis.py      ADF tests, autocorrelation
│   ├── 12_highdim_oap.py               p=193 OAP validation
│   ├── 13_spearman_rank_correlation.py  Novel: KKT ranking ρ=0.906
│   ├── 14_fista_degradation.py          Novel: FISTA 0.95× at α=0.05
│   └── 15_theorem4_bounded_steps.py     Novel: BB step size proof
│
├── notebooks/                    Step-by-step development with outputs
│   ├── 01_data_preparation.ipynb
│   ├── 02_ridge_implementation.ipynb
│   ├── 03_lasso_implementation.ipynb
│   ├── 04_elasticnet_implementation.ipynb
│   ├── 05_cvxpy_formulation.ipynb
│   ├── 06_experiments.ipynb
│   ├── 07_backtesting.ipynb
│   └── 08_novel_contributions.ipynb    Three novel findings with outputs
│
├── outputs/                      45 generated figures
├── app/app.py                    Streamlit analytics platform (5 pages)
├── .streamlit/config.toml        Dark theme configuration
├── requirements.txt              Python dependencies
└── README.md                     This file
```

---

## Setup

```bash
git clone https://github.com/oladri-renuka/sparse-factor-modeling
cd sparse-factor-modeling
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Dependencies:** numpy, pandas, matplotlib, scipy, cvxpy, statsmodels,
streamlit, plotly, seaborn, scikit-learn, yfinance

Data is downloaded automatically from the Kenneth French Data Library
on first run — no manual download required.

---

## Reproducing Results

Each script in `scripts/` is standalone and reproduces a specific result from the report.

```bash
source venv/bin/activate

# Core algorithm benchmark (Table 2)
python3 scripts/01_wall_clock_comparison.py

# Novel finding 1: FISTA degrades at high sparsity
python3 scripts/14_fista_degradation.py

# Novel finding 2: KKT factor ranking (Spearman ρ=0.906)
python3 scripts/13_spearman_rank_correlation.py

# Novel finding 3: BB step size bounds (0 violations)
python3 scripts/15_theorem4_bounded_steps.py

# Capacity analysis ($1.265B breakeven)
python3 scripts/06_capacity_analysis.py
```

Scripts 02, 05, 08, 12 require 5–15 minutes each (full backtesting loops).
All other scripts complete in under 2 minutes.

---

## Running the App

```bash
source venv/bin/activate
streamlit run app/app.py
```

Opens at `http://localhost:8501`. Five pages:

```
Home               Landing page with key results and system overview
Algorithm Benchmark  Live convergence race — drag alpha to 0.05 to see FISTA degrade
Factor Screening   Drag regularization slider to watch factors drop out in real time
Strategy Performance  14-year backtest results, capacity curve, transaction costs
Technical Contributions  Three novel findings with evidence
```

---

## Data Sources

| Dataset | Source | Period | Dimensions |
|---|---|---|---|
| Fama-French 5 Factors + Momentum | Kenneth French Data Library | 2000–2023 | 288 × 6 |
| 25 Size-Value Portfolios | Kenneth French Data Library | 2000–2023 | 288 × 25 |
| OpenAssetPricing (193 characteristics) | Chen, Pelger & Zhu (2023) | 2000–2023 | 299 × 193 |

Portfolio construction: NYSE/AMEX/NASDAQ stocks sorted independently into
5 size quintiles and 5 book-to-market quintiles. Value-weighted. Rebalanced
annually each June.

---

## Theoretical Results

| Result | Statement | Status |
|---|---|---|
| Theorem 1 | Proximal GD converges at O(1/t) | Proved |
| Theorem 2 | FISTA converges at O(1/t²) | Proved |
| Theorem 3 | Online LASSO regret bound O(√T) | Proved |
| Theorem 4 | BB step sizes bounded: 1/L ≤ η_k ≤ 1/μ | Proved (novel) |
| Proposition 1 | BB convergence rate for composite LASSO | Open problem |
| Conjecture 1 | Algorithm selection rule: BB for κ<10 | Empirically supported |

---

## References

Beck, A., & Teboulle, M. (2009). A fast iterative shrinkage-thresholding algorithm.
*SIAM Journal on Imaging Sciences*, 2(1), 183–202.

Barzilai, J., & Borwein, J. M. (1988). Two-point step size gradient methods.
*IMA Journal of Numerical Analysis*, 8(1), 141–148.

Fama, E. F., & French, K. R. (2015). A five-factor asset pricing model.
*Journal of Financial Economics*, 116(1), 1–22.

Harvey, C. R., Liu, Y., & Zhu, H. (2016). ...and the cross-section of expected returns.
*Review of Financial Studies*, 29(1), 5–68.

O'Donoghue, B., & Candès, E. (2015). Adaptive restart for accelerated gradient schemes.
*Foundations of Computational Mathematics*, 15(3), 715–732.

Raydan, M. (1997). The Barzilai and Borwein gradient method.
*SIAM Journal on Optimization*, 7(1), 26–33.

Full reference list available in the final report.