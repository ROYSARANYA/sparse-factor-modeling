"""
Regime-Aware Regularization.
Lambda adapts to market volatility — higher volatility = stronger regularization.
During crises (2008, 2020), factor relationships break down.
Adaptive lambda stabilizes factor selection during stress periods.
"""
import numpy as np
import pandas as pd


def compute_rolling_volatility(market_returns, window=12):
    """
    Compute rolling annualized volatility of market returns.
    
    Parameters
    ----------
    market_returns : pd.Series — monthly Mkt-RF returns
    window         : int — rolling window in months
    
    Returns
    -------
    pd.Series of rolling annualized volatility
    """
    return market_returns.rolling(window).std() * np.sqrt(12)


def compute_adaptive_lambda(market_returns, base_lambda,
                             window=12, clip_min=0.5, clip_max=3.0):
    """
    Scale lambda by current volatility relative to historical mean.
    
    lambda_t = base_lambda * (vol_t / mean_vol)
    
    Intuition: when markets are volatile, factor relationships
    are less stable — use stronger regularization to avoid
    fitting noise.
    
    Parameters
    ----------
    market_returns : pd.Series
    base_lambda    : float — baseline regularization
    window         : int — rolling window for vol estimate
    clip_min/max   : float — bounds on scaling factor
    
    Returns
    -------
    pd.Series of adaptive lambdas
    """
    rolling_vol  = compute_rolling_volatility(market_returns, window)
    mean_vol     = rolling_vol.mean()
    scale        = (rolling_vol / mean_vol).clip(clip_min, clip_max)
    adaptive_lam = base_lambda * scale
    return adaptive_lam, rolling_vol


def identify_regimes(market_returns, window=12,
                     low_pct=33, high_pct=67):
    """
    Label each month as low/medium/high volatility regime.
    
    Returns
    -------
    pd.Series with labels: 'low', 'medium', 'high', 'crisis'
    Crisis defined as monthly return < -10%
    """
    rolling_vol = compute_rolling_volatility(market_returns, window)
    low_thresh  = rolling_vol.quantile(low_pct / 100)
    high_thresh = rolling_vol.quantile(high_pct / 100)

    regimes = pd.Series('medium', index=market_returns.index)
    regimes[rolling_vol <= low_thresh]  = 'low'
    regimes[rolling_vol >= high_thresh] = 'high'
    regimes[market_returns < -0.10]     = 'crisis'

    return regimes, rolling_vol
