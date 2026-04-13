"""
Signal Discovery from Raw Price/Volume Data.
Constructs 50+ candidate signals from OHLCV data,
then uses LASSO to select which signals predict
next-month returns — discovering factors from scratch
rather than using pre-built Fama-French factors.
"""
import numpy as np
import pandas as pd
import yfinance as yf
from itertools import product


# S&P 500 representative tickers across sectors
TICKERS = [
    # Tech
    'AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA',
    # Finance
    'JPM', 'BAC', 'GS', 'MS', 'WFC',
    # Healthcare
    'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK',
    # Energy
    'XOM', 'CVX', 'COP', 'SLB', 'EOG',
    # Consumer
    'AMZN', 'WMT', 'HD', 'MCD', 'NKE',
    # Industrials
    'BA', 'CAT', 'GE', 'MMM', 'HON',
    # Utilities
    'NEE', 'DUK', 'SO', 'AEP', 'EXC',
    # Materials
    'LIN', 'APD', 'ECL', 'DD', 'NEM',
    # Real Estate
    'AMT', 'PLD', 'CCI', 'EQIX', 'PSA',
    # Communication
    'VZ', 'T', 'CMCSA', 'NFLX', 'DIS'
]


def download_price_data(tickers=TICKERS,
                         start='2000-01-01',
                         end='2023-12-31'):
    """Download monthly OHLCV data via yfinance."""
    print(f'Downloading price data for {len(tickers)} stocks...')
    data = yf.download(tickers, start=start, end=end,
                        interval='1mo', auto_adjust=True,
                        progress=False)
    return data


def construct_signals(data):
    """
    Construct 50+ candidate predictive signals from price/volume data.

    Signal categories:
    1. Momentum (1,3,6,12 month)
    2. Short-term reversal (1 week via monthly proxy)
    3. Volatility (realized vol, vol ratio)
    4. Volume signals (turnover, volume trend)
    5. Price ratios (price to moving average)
    6. Trend signals (52-week high ratio)
    """
    close  = data['Close']
    volume = data['Volume'] if 'Volume' in data else None

    signals = {}

    # ── Momentum signals ──────────────────────────────────────
    for months in [1, 3, 6, 12]:
        ret = close.pct_change(months)
        signals[f'mom_{months}m'] = ret

    # ── Skip-month momentum (months 2-12, skip month 1) ──────
    signals['mom_2_12'] = close.pct_change(12) - close.pct_change(1)

    # ── Short-term reversal ───────────────────────────────────
    signals['reversal_1m'] = -close.pct_change(1)

    # ── Long-term reversal (36-month) ─────────────────────────
    signals['reversal_36m'] = -close.pct_change(36)

    # ── Realized volatility ───────────────────────────────────
    monthly_ret = close.pct_change(1)
    for window in [3, 6, 12]:
        signals[f'vol_{window}m'] = monthly_ret.rolling(window).std()

    # ── Volatility ratio (short / long) ───────────────────────
    signals['vol_ratio_3_12'] = (
        monthly_ret.rolling(3).std() /
        (monthly_ret.rolling(12).std() + 1e-8)
    )

    # ── Price to moving average ratio ─────────────────────────
    for window in [3, 6, 12]:
        ma = close.rolling(window).mean()
        signals[f'price_ma_{window}m'] = close / (ma + 1e-8) - 1

    # ── 52-week high ratio ─────────────────────────────────────
    high_12m = close.rolling(12).max()
    signals['high_52w_ratio'] = close / (high_12m + 1e-8)

    # ── Volume signals (if available) ─────────────────────────
    if volume is not None:
        for window in [3, 6]:
            vol_ma = volume.rolling(window).mean()
            signals[f'vol_trend_{window}m'] = (
                volume / (vol_ma + 1e-8) - 1
            )

    # Stack into panel: (date x ticker x signal)
    signal_df = pd.concat(signals, axis=1)
    signal_df.columns = pd.MultiIndex.from_tuples(
        [(sig, ticker)
         for sig in signals.keys()
         for ticker in close.columns],
        names=['signal', 'ticker']
    )

    return signal_df, list(signals.keys())


def build_cross_sectional_dataset(data, signal_names,
                                   forward_months=1):
    """
    Build cross-sectional dataset for return prediction.

    For each month t:
    - X: signal values at month t for all stocks
    - y: forward return at month t+1 for all stocks

    Returns stacked (observations x signals) dataset
    suitable for LASSO.
    """
    close       = data['Close']
    fwd_returns = close.pct_change(forward_months).shift(-forward_months)

    X_list, y_list, dates_list = [], [], []

    for date in close.index[24:-forward_months]:
        x_row, y_row = [], []
        valid        = True

        for ticker in close.columns:
            # Get signals for this stock at this date
            sig_vals = []
            for sig in signal_names:
                try:
                    val = data['Close'][ticker].pct_change(
                        1 if '1m' in sig else
                        3 if '3m' in sig else
                        6 if '6m' in sig else 12
                    ).loc[date]
                    sig_vals.append(val if not np.isnan(val) else 0)
                except Exception:
                    sig_vals.append(0)

            fwd_ret = fwd_returns[ticker].loc[date] \
                      if date in fwd_returns.index else np.nan

            if not np.isnan(fwd_ret):
                x_row.append(sig_vals)
                y_row.append(fwd_ret)

        if len(x_row) >= 20:
            X_list.append(np.array(x_row))
            y_list.append(np.array(y_row))
            dates_list.append(date)

    return X_list, y_list, dates_list
