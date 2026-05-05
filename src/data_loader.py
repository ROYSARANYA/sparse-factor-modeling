"""
data_loader.py
==============
Ingests and aligns raw Fama-French CSV files into clean, analysis-ready
DataFrames.  All parsing logic is isolated here so the rest of the
codebase receives consistent, date-indexed, decimal-return data with no
further preprocessing required.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Source files  (expected in <repo_root>/data/)
---------------------------------------------
  F-F_Research_Data_5_Factors_2x3.csv   – monthly FF5 factors
  F-F_Momentum_Factor.csv               – monthly momentum factor
  25_Portfolios_5x5.csv                 – monthly 25 size×value portfolio
                                          returns (value-weighted)

  All three files are the standard Kenneth French data library CSVs,
  which embed descriptive header text, multiple data sections (monthly,
  annual, etc.), and percentage-scaled returns.  The raw format is not
  directly pandas-readable — hence the custom parser below.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_read_ff_section(filepath, header_row)
---------------------------------------
Low-level parser for a single data section within a French-library CSV.

  Header detection:
    The caller provides the 0-indexed line number of the column header
    (different for each file — see per-file loaders below).  The header
    line is split on commas and stripped.

  Data extraction:
    Lines are consumed from header_row+1 onward.  Parsing stops at the
    first blank line, which is how French-library files delimit sections
    (monthly data is always the first section; annual data follows after
    a blank line).  This prevents annual rows from contaminating the
    monthly panel.

  Index parsing and validation:
    The first column is used as the index.  Rows whose index does not
    match the regex ^\d{6}$ (i.e. exactly six digits) are dropped —
    this silently removes any stray footnote or label rows that slip
    through the blank-line guard.  Valid indices are parsed as %Y%m
    (e.g. "200301" → 2003-01-01) and converted to a DatetimeIndex with
    monthly frequency.

  Numeric coercion:
    All remaining columns are coerced to float; any row containing a
    non-numeric value (e.g. "999.99" sentinel or partial rows) is
    dropped via dropna().

_load_ff5()
-----------
  Reads the five-factor file.  Header is on line 4 (0-indexed) — the
  first four lines are the French-library copyright / description block.
  Divides all values by 100 to convert from percentage to decimal.

_load_momentum()
----------------
  Reads the momentum factor file.  Header is on line 13 — this file has
  a longer descriptive preamble than the five-factor file.
  Divides by 100 for decimal returns.

_load_portfolios()
------------------
  Reads the 25 size×value portfolio file.  The header row is detected
  dynamically by scanning for the string 'SMALL LoBM' or 'Lo BM', which
  French uses to label the first column of the value-weight monthly
  section — making the loader robust to minor formatting changes across
  file vintages.
  Replaces the -99.99 sentinel (used by French to flag missing portfolio
  observations) with NaN, then drops any month with a missing portfolio.
  Divides by 100 for decimal returns.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

load_all_data(start='2000-01-01', end='2023-12-31')
----------------------------------------------------
Public entry point.  Assembles X and Y from the three source files.

  Factor matrix X  (T × 6):
    FF5 file contains six columns: Mkt-RF, SMB, HML, RMW, CMA, and RF
    (the risk-free rate).  RF is dropped — it is not a tradeable factor
    and is not used as a predictor.  The momentum factor is joined on the
    date index (inner join, so only months present in both files are
    kept).  Columns are renamed to the canonical FACTOR_NAMES list.

  Portfolio matrix Y  (T × 25):
    The 25 portfolio columns are renamed P1…P25 in the order they appear
    in the French file (sorted from small-low-BM in the top-left to
    big-high-BM in the bottom-right of the 5×5 grid).

  Alignment:
    X and Y are inner-joined on their DatetimeIndex (axis=0 only) before
    the date filter is applied.  This means any month missing from any
    one of the three source files is dropped from both matrices,
    guaranteeing X and Y always have identical row counts and identical
    index values — a hard requirement for the regression and backtesting
    code downstream.

  Date filtering:
    After alignment, both matrices are sliced to [start, end].  The
    defaults yield a 24-year, ~288-month panel (2000-01 through 2023-12).

  Diagnostics:
    Prints a one-line summary of the loaded shape and date range on every
    call so data issues (truncated file, unexpected gaps) are immediately
    visible in any script's output.

  Returns:
    X               pd.DataFrame (T × 6)   factor returns, decimal
    Y               pd.DataFrame (T × 25)  portfolio returns, decimal
    factor_names    list[str]  ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Mom']
    portfolio_names list[str]  ['P1', … , 'P25']

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Dependencies
------------
pandas, numpy, os

Data directory
--------------
Resolved relative to this file's location as ../data/, so the module
works regardless of the working directory from which scripts are run.
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
