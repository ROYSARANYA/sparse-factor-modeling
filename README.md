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
├── src/                 Core implementation (10 modules)
├── scripts/             Reproducibility scripts (15 files)
├── notebooks/           Jupyter notebooks with outputs (8 files)
├── outputs/             Generated figures (45 PNG files)
├── app/                 Streamlit analytics platform
├── .streamlit/          Dark theme configuration
├── requirements.txt     Python dependencies
└── README.md            This file
```

---

## File Descriptions

### src/ — Core Implementation

**`src/solvers.py`**
Nine optimization algorithms implemented from scratch. Every class has `fit()`,
`predict()`, `coef_`, `n_iter_`, and `loss_history_` for consistency.

```
RidgeScratch         Closed-form solution: β = (X'X + αnI)⁻¹ X'y
LassoProximal        Proximal gradient descent with soft-thresholding
ElasticNetScratch    Combined L1+L2 penalty via proximal gradient
FISTALasso           Accelerated proximal gradient — O(1/t²) convergence
FISTARestart         FISTA with adaptive restart (O'Donoghue & Candès 2015)
BBLasso              Barzilai-Borwein adaptive step sizes — alternates BB1/BB2
CoordinateDescent    Cyclic coordinate descent — fastest empirically (9 iters)
WarmStartLasso       Full regularization path with warm-start initialization
MomentumBBLasso      Novel: combines BB step sizes with Nesterov momentum
```

**`src/backtest.py`**
Walk-forward backtesting engine. Training window expands month by month —
no look-ahead bias. All five functions follow the same interface.

```
walk_forward_backtest()          Standard walk-forward: 120-month training window,
                                 168 out-of-sample predictions (2010–2023)

compute_metrics()                Computes OOS R², Sharpe, ICIR, mean IC,
                                 annualized return — all from predictions alone

compute_metrics_with_costs()     Adds transaction cost model: gross/net Sharpe
                                 at 10bps and 50bps round-trip cost

compute_alpha_decay()            Measures IC at horizons 1M through 12M —
                                 quantifies how quickly signal decays

walk_forward_backtest_adaptive() Regime-aware backtest: scales lambda by current
                                 rolling volatility relative to historical mean
```

**`src/online_solver.py`**
Online LASSO using stochastic proximal gradient. Updates one observation at a time —
no retraining required. Achieves O(√T) regret bound.

```
OnlineLasso              update(x, y) → incremental fit
                         predict(x) → single prediction
walk_forward_online()    Full streaming backtest returning same format
                         as walk_forward_backtest for direct comparison
```

**`src/dro_solver.py`**
Wasserstein distributionally robust LASSO. Adds an L2 penalty on β scaled by ε,
representing worst-case perturbation under a Wasserstein ball of radius ε.
Tractable dual formulation: min (1/n)||y-Xβ||² + ε||β||₂ + α||β||₁

```
dro_lasso_cvxpy()    CVXPY reference implementation
DROLasso             Sklearn-compatible wrapper (fit, predict, coef_)
                     epsilon=0 reduces to standard LASSO
```

**`src/cross_validation.py`**
Time-series cross-validation with expanding window. Never uses future data —
training always precedes validation.

```
time_series_cv()       n_splits folds, returns best alpha and full CV results
find_best_alphas()     Runs CV for all 25 portfolios, returns median best alpha
```

**`src/regime.py`**
Volatility regime classification based on rolling market return standard deviation.
Four regimes: low, medium, high, crisis.

```
compute_rolling_volatility()    Rolling std of market returns (annualized)
compute_adaptive_lambda()       Scales base_lambda by vol_t / mean_vol,
                                clipped to [0.5×, 3.0×] range
identify_regimes()              Assigns each month to low/medium/high/crisis
                                using 33rd and 67th percentile thresholds
```

**`src/interactions.py`**
Extends the 6-factor model with C(6,2)=15 pairwise interaction terms,
expanding X from shape (288,6) to (288,21).

```
build_interaction_features()        Constructs all 15 pairwise products
analyze_interaction_selection()     Runs LASSO on expanded feature set
                                    across all 25 portfolios, returns
                                    selection matrix (25×21)
```

**`src/signal_discovery.py`**
Constructs 15+ predictive signals from raw price/volume data using yfinance.
Used to test whether fundamental accounting signals (FF factors) can be
replicated from price history alone. Key finding: they cannot (IC 0.614 vs 0.008).

```
download_price_data()           Downloads monthly OHLCV for 50 S&P500 stocks
construct_signals()             Builds momentum, reversal, volatility,
                                volume, and price-ratio signals
build_cross_sectional_dataset() Aligns signals with forward returns
```

**`src/data_loader.py`**
Loads and aligns three Kenneth French Data Library datasets. Downloads
automatically from the web on first run — no manual setup required.

```
load_all_data()     Returns (X, Y, factor_names, portfolio_names)
                    X: (288, 6) factor returns 2000-2023
                    Y: (288, 25) portfolio returns 2000-2023
```

**`src/cvxpy_solvers.py`**
CVXPY reference implementations for Ridge, LASSO, and Elastic Net.
Used to verify that scratch implementations match the convex program solution.
All three are confirmed DCP (disciplined convex programs).

---

### scripts/ — Reproducibility Scripts

Each script is standalone — run from the project root with `python3 scripts/name.py`.
All figures are saved to `outputs/`.

```
01_wall_clock_comparison.py      Benchmarks all 6 algorithms on 25 portfolios
                                 Generates: outputs/benchmark_timing.png

02_dro_calibration.py            Tests DRO at ε ∈ {0, 0.01, 0.05, 0.091, 0.15}
                                 Finds data-driven ε = 0.091 via bootstrap

03_loss_landscape.py             Visualizes Ridge vs LASSO objective geometry
                                 Shows L1 ball corner solutions vs L2 sphere

04_dollar_neutral_sharpe.py      Tests long-short portfolio construction
                                 Dollar-neutral constraint: sum(weights) = 0

05_mv_optimizer.py               Mean-variance QP: min w'Σw - λμ'w
                                 Subject to dollar-neutral, |w|≤2, |wi|≤0.15

06_capacity_analysis.py          Computes AUM breakeven as function of fund size
                                 Market impact model: cost ~ AUM^0.6

07_factor_crowding.py            Measures pairwise factor correlation over time
                                 Rolling 36-month windows, identifies crowding

08_kkt_dropout_prediction.py     Tests KKT condition as sparsity predictor
                                 Computes |[C⁻¹c_k]_j| vs actual dropout

09_novelty_momentum_bb.py        Compares Momentum-BB vs all other algorithms
                                 Shows convergence on 3 representative portfolios

10_novelty_algorithm_theory.py   Algorithm selection theory: κ vs speedup
                                 Generates conjecture 1 supporting evidence

11_time_series_analysis.py       ADF stationarity tests, Ljung-Box autocorrelation
                                 Heteroskedasticity analysis (ARCH effects)

12_highdim_oap.py                Validates all algorithms at p=193 (OAP dataset)
                                 BB LASSO 6.4× faster than PGD at high dimension

13_spearman_rank_correlation.py  NOVEL: KKT factor ranking prediction
                                 Result: mean Spearman ρ = 0.906 across 25 portfolios

14_fista_degradation.py          NOVEL: FISTA speedup vs regularization strength
                                 Result: FISTA 0.95× at α=0.05 (slower than PGD)

15_theorem4_bounded_steps.py     NOVEL: BB step size bound verification
                                 Result: 0 violations of 1/L ≤ η_k ≤ 1/μ
```

### notebooks/ — Step-by-Step Development

All notebooks have embedded outputs — no re-execution needed to view results.

```
01_data_preparation.ipynb        Data loading, summary statistics, correlation
                                 matrix, portfolio return distributions

02_ridge_implementation.ipynb    Ridge from scratch, closed-form derivation,
                                 regularization path, coefficient stability

03_lasso_implementation.ipynb    Proximal gradient derivation, soft-thresholding
                                 proof, convergence plot, sparsity demonstration

04_elasticnet_implementation.ipynb  Grouping effect: HML/CMA both survive EN
                                    but LASSO drops one (correlation = 0.632)

05_cvxpy_formulation.ipynb       All 4 methods as explicit DCP programs
                                 is_dcp()=True verified for all formulations

06_experiments.ipynb             CV lambda selection, factor dropout table,
                                 regime analysis, DRO, interactions, signals

07_backtesting.ipynb             Full walk-forward results, rolling IC chart,
                                 regime breakdown, master results figure

08_novel_contributions.ipynb     Three novel findings with code and outputs:
                                 FISTA degradation, KKT ranking, Theorem 4
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
