import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '.')
from src.data_loader import load_all_data
from src.solvers import (LassoProximal, RidgeScratch, ElasticNetScratch,
                          FISTARestart, BBLasso, CoordinateDescent)
from src.backtest import walk_forward_backtest, compute_metrics, compute_metrics_with_costs
from src.online_solver import walk_forward_online
import time

st.set_page_config(page_title="MSML 604 — Sparse Factor Models",
                   layout="wide", page_icon="📈")

st.title("📈 Sparse Factor Modeling for Equity Return Prediction")
st.caption("MSML 604 — Convex Optimization | Fama-French 5-Factor + Momentum")

# ── Load data ──────────────────────────────────────────────────────────────
@st.cache_data
def load():
    return load_all_data()

X, Y, factor_names, portfolio_names = load()
X_vals   = X.values
X_scaled = (X_vals - X_vals.mean(0)) / X_vals.std(0)

# ── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Controls")

tab = st.sidebar.radio("Section", [
    "1. Algorithm Comparison",
    "2. Regularization Paths",
    "3. Factor Selection",
    "4. Backtest Results",
    "5. Transaction Costs",
])

# ══════════════════════════════════════════════════════════════════════════
if tab == "1. Algorithm Comparison":
    st.header("Algorithm Convergence Comparison")
    st.markdown("Six algorithms implemented from scratch. Select a portfolio and alpha to compare.")

    col1, col2 = st.columns(2)
    port_idx = col1.selectbox("Portfolio", range(25),
                               format_func=lambda i: portfolio_names[i])
    alpha    = col2.select_slider("Alpha (λ)", [0.001,0.003,0.005,0.01,0.02,0.05], value=0.003)

    y_sel = Y.iloc[:, port_idx].values

    methods = [
        ("Proximal GD",       LassoProximal,     {},                     "#E74C3C"),
        ("FISTA vanilla",     lambda **kw: __import__('src.solvers', fromlist=['FISTALasso']).FISTALasso(**kw), {}, "#2E74B5"),
        ("FISTA+fn restart",  FISTARestart,      {"restart":"function"}, "#27AE60"),
        ("BB LASSO",          BBLasso,           {},                     "#8E44AD"),
        ("Coord Descent",     CoordinateDescent, {},                     "#1ABC9C"),
    ]

    from src.solvers import FISTALasso
    methods = [
        ("Proximal GD",      LassoProximal,     {},                     "#E74C3C"),
        ("FISTA vanilla",    FISTALasso,        {},                     "#2E74B5"),
        ("FISTA+fn restart", FISTARestart,      {"restart":"function"}, "#27AE60"),
        ("BB LASSO",         BBLasso,           {},                     "#8E44AD"),
        ("Coord Descent",    CoordinateDescent, {},                     "#1ABC9C"),
    ]

    results = []
    for name, cls, kwargs, color in methods:
        t0 = time.perf_counter()
        m  = cls(alpha=alpha, max_iter=500, **kwargs).fit(X_scaled, y_sel)
        ms = (time.perf_counter()-t0)*1000
        results.append({"Method": name, "Iterations": m.n_iter_,
                         "Time (ms)": round(ms,2),
                         "Nonzero": int(np.sum(np.abs(m.coef_)>1e-4)),
                         "_hist": m.loss_history_, "_color": color})

    # Table
    df_res = pd.DataFrame([{k:v for k,v in r.items() if not k.startswith('_')}
                            for r in results])
    pgd_ms = df_res.loc[0, "Time (ms)"]
    df_res["Speedup vs PGD"] = (pgd_ms / df_res["Time (ms)"]).round(2).astype(str) + "×"
    st.dataframe(df_res, use_container_width=True)

    # Convergence plot
    fig, ax = plt.subplots(figsize=(10, 4))
    opt = min(min(r["_hist"]) for r in results)
    for r in results:
        gap = np.array(r["_hist"]) - opt + 1e-12
        ax.semilogy(gap, color=r["_color"], lw=2,
                    label=f'{r["Method"]} ({r["Iterations"]} iters)')
    ax.set_xlabel("Iteration"); ax.set_ylabel("Objective Gap (log)")
    ax.set_title(f"Convergence on {portfolio_names[port_idx]}  |  α={alpha}")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════
elif tab == "2. Regularization Paths":
    st.header("Regularization Path")
    st.markdown("How coefficients change as lambda varies from large (sparse) to small (dense).")

    col1, col2 = st.columns(2)
    method = col1.selectbox("Method", ["LASSO", "Ridge", "Elastic Net"])
    port_idx = col2.selectbox("Portfolio", range(25),
                               format_func=lambda i: portfolio_names[i])

    y_sel   = Y.iloc[:, port_idx].values
    lambdas = np.logspace(-4, 0, 80)
    paths   = []

    for lam in lambdas:
        if method == "LASSO":
            m = LassoProximal(alpha=lam).fit(X_scaled, y_sel)
        elif method == "Ridge":
            m = RidgeScratch(alpha=lam).fit(X_scaled, y_sel)
        else:
            m = ElasticNetScratch(alpha=lam, l1_ratio=0.5).fit(X_scaled, y_sel)
        paths.append(m.coef_.copy())
    paths = np.array(paths)

    fig, ax = plt.subplots(figsize=(11, 5))
    COLORS = ['#2E74B5','#E74C3C','#27AE60','#F39C12','#8E44AD','#1ABC9C']
    for i, fname in enumerate(factor_names):
        ax.plot(np.log10(lambdas), paths[:, i],
                label=fname, color=COLORS[i], lw=2.5)
    ax.axhline(0, color='black', lw=0.8, ls='--')
    ax.set_xlabel("log10(lambda)"); ax.set_ylabel("Coefficient")
    ax.set_title(f"{method} Regularization Path — {portfolio_names[port_idx]}")
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════
elif tab == "3. Factor Selection":
    st.header("Factor Selection Across 25 Portfolios")
    st.markdown("LASSO drops CMA and Mom in most portfolios. Ridge retains everything.")

    alpha  = st.slider("Alpha", 0.001, 0.05, 0.003, step=0.001)
    method = st.radio("Method", ["LASSO", "Ridge", "Elastic Net"], horizontal=True)

    selection = np.zeros((25, 6))
    for j in range(Y.shape[1]):
        yj = Y.iloc[:, j].values
        if method == "LASSO":
            m = LassoProximal(alpha=alpha).fit(X_scaled, yj)
        elif method == "Ridge":
            m = RidgeScratch(alpha=alpha).fit(X_scaled, yj)
        else:
            m = ElasticNetScratch(alpha=alpha, l1_ratio=0.5).fit(X_scaled, yj)
        selection[j] = (np.abs(m.coef_) > 1e-4).astype(float)

    fig, ax = plt.subplots(figsize=(10, 8))
    import seaborn as sns
    sns.heatmap(selection, xticklabels=factor_names,
                yticklabels=portfolio_names, cmap='RdYlGn',
                vmin=0, vmax=1, linewidths=0.5,
                annot=True, fmt='.0f', ax=ax)
    ax.set_title(f"{method} Factor Selection  |  α={alpha}\n"
                  f"Mkt-RF selected {int(selection[:,0].sum())}/25  |  "
                  f"CMA selected {int(selection[:,4].sum())}/25  |  "
                  f"Mom selected {int(selection[:,5].sum())}/25")
    st.pyplot(fig)
    plt.close()

    col1, col2, col3 = st.columns(3)
    col1.metric("Mkt-RF selected", f"{int(selection[:,0].sum())}/25")
    col2.metric("CMA selected",    f"{int(selection[:,4].sum())}/25")
    col3.metric("Mom selected",    f"{int(selection[:,5].sum())}/25")

# ══════════════════════════════════════════════════════════════════════════
elif tab == "4. Backtest Results":
    st.header("Walk-Forward Backtest Results")
    st.markdown("168 genuine out-of-sample predictions, Jan 2010 – Dec 2023.")

    results_table = {
        "Method":    ["Ridge",  "LASSO",  "Elastic Net", "Online LASSO"],
        "OOS R²":    [0.9009,   0.8983,   0.8960,        0.8999],
        "Sharpe":    [4.6123,   4.9393,   4.9400,        5.0611],
        "ICIR":      [2.5854,   2.5755,   2.5596,        2.7023],
        "Ann Ret %": [59.14,    58.78,    58.72,         58.58],
        "Ann Vol %": [12.82,    11.90,    11.89,         11.58],
    }
    st.dataframe(pd.DataFrame(results_table).set_index("Method"),
                 use_container_width=True)

    st.markdown("---")
    st.markdown("### Key findings")
    col1, col2, col3 = st.columns(3)
    col1.metric("Best Sharpe",  "5.061",  "Online LASSO")
    col2.metric("Best ICIR",    "2.702",  "Online LASSO")
    col3.metric("Speedup",      "8.1×",   "Online vs Batch")

    st.markdown("### Dollar-neutral breakdown (LASSO)")
    dn_table = {
        "Construction":  ["Gross top3/bot3", "Dollar-neutral", "Long-only top5"],
        "Sharpe":        [4.939, 4.720, 2.005],
        "Ann Ret %":     [58.78, 50.56, 38.16],
        "Ann Vol %":     [11.90, 10.71, 19.03],
    }
    st.dataframe(pd.DataFrame(dn_table).set_index("Construction"),
                 use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
elif tab == "5. Transaction Costs":
    st.header("Net-of-Cost Performance & Capacity")

    col1, col2 = st.columns(2)

    # Cost table
    with col1:
        st.markdown("### Net Sharpe by Transaction Cost")
        cost_table = {
            "Method":  ["Ridge","LASSO","Elastic Net","Online LASSO"],
            "0 bps":   [4.61, 4.94, 4.94, 5.06],
            "10 bps":  [4.46, 4.78, 4.78, 4.90],
            "20 bps":  [4.31, 4.62, 4.62, 4.74],
            "50 bps":  [3.85, 4.14, 4.14, 4.25],
        }
        st.dataframe(pd.DataFrame(cost_table).set_index("Method"),
                     use_container_width=True)

    # Capacity curve
    with col2:
        st.markdown("### Capacity Analysis")
        aum_range  = np.logspace(6, 9.5, 200)
        base_ret   = 0.5878
        base_vol   = 0.1190
        base_sh    = base_ret / base_vol
        turnover   = 1.537
        ADV, gamma = 5e9, 0.1
        net_sh     = [(base_ret - gamma*(a*turnover*12)/ADV)/base_vol
                      for a in aum_range]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.semilogx(aum_range/1e6, net_sh, color='#2E74B5', lw=2.5)
        ax.axhline(1.0, color='red', lw=2, ls='--', label='Sharpe=1.0 breakeven')
        ax.axvline(1265, color='red', lw=1.5, ls=':', label='$1,265M breakeven')
        ax.fill_between(aum_range/1e6, net_sh, 1.0,
                        where=np.array(net_sh)<1.0, alpha=0.2, color='red')
        ax.fill_between(aum_range/1e6, net_sh, 1.0,
                        where=np.array(net_sh)>=1.0, alpha=0.1, color='green')
        ax.set_xlabel("AUM ($ millions, log scale)")
        ax.set_ylabel("Net Sharpe Ratio")
        ax.set_title("Capacity: Breakeven = $1,265M")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 5.5)
        st.pyplot(fig)
        plt.close()

    st.markdown("### Alpha Decay")
    decay_table = {
        "Holding Period": ["1M","2M","3M","4M","5M","6M"],
        "LASSO IC":       [0.610, 0.434, 0.351, 0.329, 0.280, 0.271],
        "IC Decay":       ["0%","29%","42%","46%","54%","56%"],
    }
    st.dataframe(pd.DataFrame(decay_table).set_index("Holding Period"),
                 use_container_width=True)
    st.caption("Signal decays 56% from 1-month to 6-month holding period — strategy is a short-horizon predictor.")
