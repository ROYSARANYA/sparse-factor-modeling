"""
Data loading and preprocessing for Fama-French factor data.
"""
import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

FACTOR_NAMES    = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Mom']
PORTFOLIO_NAMES = [f'P{i+1}' for i in range(25)]


def _read_ff_section(filepath, header_row):
    """
    Reads only the first data section of a Fama-French CSV.
    header_row: 0-indexed line number of the column header.
    Data starts at header_row+1 and stops at the next blank line.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Parse header
    header = [c.strip() for c in lines[header_row].strip().split(',')]

    # Parse data rows — stop at first blank line after header
    rows = []
    for line in lines[header_row + 1:]:
        stripped = line.strip()
        if stripped == '':
            break
        parts = [p.strip() for p in stripped.split(',')]
        if len(parts) == len(header):
            rows.append(parts)

    df = pd.DataFrame(rows, columns=header)
    index_col = df.columns[0]
    df = df.set_index(index_col)
    df = df[df.index.str.match(r'^\d{6}$')]
    df.index = pd.to_datetime(df.index, format='%Y%m')
    df = df.apply(pd.to_numeric, errors='coerce').dropna()
    return df


def _load_ff5() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, 'F-F_Research_Data_5_Factors_2x3.csv')
    return _read_ff_section(path, header_row=4) / 100


def _load_momentum() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, 'F-F_Momentum_Factor.csv')
    return _read_ff_section(path, header_row=13) / 100


def _load_portfolios() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, '25_Portfolios_5x5.csv')
    # Find the header row dynamically
    with open(path, 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if 'SMALL LoBM' in line or 'Lo BM' in line:
            header_row = i
            break
    df = _read_ff_section(path, header_row=header_row)
    df = df.replace(-99.99, np.nan).dropna()
    return df / 100


def load_all_data(start='2000-01-01', end='2023-12-31'):
    """
    Returns
    -------
    X              : pd.DataFrame (T x 6) - factor returns
    Y              : pd.DataFrame (T x 25) - portfolio returns
    factor_names   : list of factor column names
    portfolio_names: list of portfolio column names
    """
    ff5   = _load_ff5()
    mom   = _load_momentum()
    ports = _load_portfolios()

    # Build X: drop RF, join momentum = 6 factors
    ff5_no_rf = ff5.drop(columns=['RF'], errors='ignore')
    X = ff5_no_rf.join(mom, how='inner')
    X.columns = FACTOR_NAMES

    # Build Y: 25 portfolio returns
    Y = ports.copy()
    Y.columns = PORTFOLIO_NAMES[:Y.shape[1]]

    # Align X and Y on dates (axis=0 = rows only, not columns)
    X, Y = X.align(Y, join='inner', axis=0)

    # Apply date range
    X = X.loc[start:end]
    Y = Y.loc[start:end]

    print(f"Data loaded: {len(X)} months | {X.shape[1]} factors | {Y.shape[1]} portfolios")
    print(f"Date range: {X.index[0].strftime('%Y-%m')} to {X.index[-1].strftime('%Y-%m')}")

    return X, Y, FACTOR_NAMES, list(Y.columns)
