"""Refresh cache/ung_iv_rank_daily.csv from the PG IV surface (closes the iv_rank staleness gap).

iv_rank (252-day percentile of real ATM IV) is a VALIDATED signal the live champion uses
(iv_rank_z_scale: top-quintile IV → -23% fwd-63d, p=.002). It was previously built ad-hoc and had
NO refresh path, so the CSV froze at 2026-06-12 while the engine reads it with ffill(limit=10) — i.e.
live silently DROPPED the signal (NaN → neutral) ~10 trading days later, diverging from the backtest.

APPEND-ONLY by design: the existing CSV history is preserved verbatim (its original atm_iv method is
not perfectly reproducible from PG — 871/1243 historical dates differ, some by illiquid-contract
outliers). We only compute atm_iv for NEW dates from the fresh PG `ung_iv_surface` (nearest ~30-DTE,
nearest-strike, mean of C+P — matches the CSV exactly on recent liquid dates), append them, and
recompute the trailing-252d percentile over the full series. A self-check confirms the recomputed rank
still matches the stored rank on the preserved history before overwriting. Idempotent; safe to run daily.
Wired into refresh_options_data.py after the surface refresh.

  venv/bin/python backtest/refresh_iv_rank.py
"""
import os
import shutil
import warnings

import pandas as pd

THIS = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(THIS, 'cache', 'ung_iv_rank_daily.csv')
TARGET_DTE = 30
WINDOW = 252
IV_MIN, IV_MAX = 0.05, 2.5          # sanity clip — drop illiquid-contract garbage marks
MONEYNESS_MAX = 0.25                # nearest ATM strike must be within 25% of spot


def _atm_iv_from_pg(since=None):
    """ATM IV per date: for each session pick the tenor nearest TARGET_DTE, then the strike nearest
    spot (within MONEYNESS_MAX), and average the call+put IV. `since` limits to dates strictly after it.
    Returns a date-indexed Series."""
    from backfill_ung_iv_pg import DB_PARAMS
    import psycopg2
    q = ("select date, dte, strike_adj, spot_adj, option_right, iv from ung_iv_surface "
         "where iv is not null and dte between 10 and 60")
    if since is not None:
        q += f" and date > '{pd.Timestamp(since).date()}'"
    conn = psycopg2.connect(**DB_PARAMS)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            df = pd.read_sql(q, conn)
    finally:
        conn.close()
    if df.empty:
        return pd.Series(dtype=float)
    df['strike_adj'] = df['strike_adj'].astype(float)
    df['spot_adj'] = df['spot_adj'].astype(float)
    df['iv'] = df['iv'].astype(float)
    out = {}
    for d, g in df.groupby('date'):
        dte0 = g['dte'].iloc[(g['dte'] - TARGET_DTE).abs().values.argmin()]
        gg = g[g['dte'] == dte0]
        mny = (gg['strike_adj'] - gg['spot_adj']).abs()
        if mny.min() > MONEYNESS_MAX * gg['spot_adj'].iloc[0]:
            continue                                  # no strike near ATM this session — skip
        k0 = gg['strike_adj'].iloc[mny.values.argmin()]
        atm = gg[gg['strike_adj'] == k0]['iv'].mean()
        if IV_MIN <= atm <= IV_MAX:
            out[pd.Timestamp(d)] = atm
    return pd.Series(out).sort_index()


def _rolling_pct_rank(s):
    """Trailing-252d percentile: fraction of the trailing window <= the current value (incl. self)."""
    return s.rolling(WINDOW).apply(lambda w: (w <= w[-1]).mean(), raw=True)


def refresh(verbose=True):
    if not os.path.exists(CSV):
        print(f'[iv_rank] no existing CSV at {CSV} — abort (append-only refresh needs a base)', flush=True)
        return
    old = pd.read_csv(CSV, index_col=0, parse_dates=True)
    last = old.index[-1]
    new_atm = _atm_iv_from_pg(since=last)
    if new_atm.empty:
        if verbose:
            print(f'[iv_rank] already current (last={last.date()}); no new PG dates', flush=True)
        return
    # APPEND-ONLY: preserve historical atm_iv verbatim, add new dates, recompute rank over full series.
    atm = pd.concat([old['atm_iv'], new_atm]).sort_index()
    atm = atm[~atm.index.duplicated(keep='first')]     # keep existing history on any overlap
    rank = _rolling_pct_rank(atm)
    new = pd.DataFrame({'atm_iv': atm, 'iv_rank': rank})
    new.index.name = 'date'
    # SELF-CHECK: recomputed rank must still match the stored rank on preserved history (method guard).
    common = old.index.intersection(new.index)
    chk = common[old.loc[common, 'iv_rank'].notna() & new.loc[common, 'iv_rank'].notna()]
    if len(chk):
        diff = (old.loc[chk, 'iv_rank'] - new.loc[chk, 'iv_rank']).abs().max()
        if diff > 0.02:
            print(f'[iv_rank] ABORT: recompute drifts from stored history (max |Δrank|={diff:.4f}). '
                  'Not overwriting.', flush=True)
            return
        if verbose:
            print(f'[iv_rank] self-check OK on {len(chk)} preserved dates (max |Δrank|={diff:.5f})', flush=True)
    shutil.copy2(CSV, CSV + '.bak')
    new.to_csv(CSV)
    if verbose:
        added = new.index.difference(old.index)
        lastv = new.dropna().iloc[-1]
        print(f'[iv_rank] refreshed → {os.path.basename(CSV)}: +{len(added)} dates '
              f'({last.date()} → {new.index[-1].date()}), last atm_iv={lastv["atm_iv"]:.4f} '
              f'iv_rank={lastv["iv_rank"]:.4f}', flush=True)


if __name__ == '__main__':
    refresh()
