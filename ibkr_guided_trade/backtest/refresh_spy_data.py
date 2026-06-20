"""ONE-COMMAND daily refresh of all SPY options data from ThetaData (authoritative source).

Mirrors refresh_options_data.py (the UNG runner). Order:
  0) update spy_vix_daily.csv (SPY stock EOD + VIX index EOD from ThetaData) → today
  1) spy_options_history  (minute quotes, resume → today)
  2) spy_iv_surface       (from the minute table, resume → today)
  3) spy_options_oi       (open interest)
Each step auto-resumes from its last ingested date and is idempotent/no-op when current.

  venv/bin/python backtest/refresh_spy_data.py            # all steps
  venv/bin/python backtest/refresh_spy_data.py --no-oi    # skip the slow OI pass
"""
import argparse
import os
import subprocess
import sys
from io import StringIO

import pandas as pd
import requests

THIS = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
SPY_CSV = os.path.join(THIS, '..', 'research', 'spy_vol', 'cache', 'spy_vix_daily.csv')
THETA = 'http://127.0.0.1:25503'


def _eod_close(kind, sym, start):
    """ThetaData EOD closes (stock or index) from `start` → today, indexed by date."""
    try:
        r = requests.get(f'{THETA}/v3/{kind}/history/eod',
                         params={'symbol': sym, 'start_date': pd.Timestamp(start).strftime('%Y%m%d'),
                                 'end_date': pd.Timestamp.today().strftime('%Y%m%d')}, timeout=20)
        if r.status_code != 200 or not r.text.strip():
            return None
        df = pd.read_csv(StringIO(r.text))
        if 'created' not in df or 'close' not in df:
            return None
        df['d'] = pd.to_datetime(df['created']).dt.normalize()
        return df.groupby('d')['close'].last()
    except Exception:
        return None


def refresh_underlying():
    """Append new SPY + VIX daily closes from ThetaData (authoritative — not yahoo)."""
    df = pd.read_csv(SPY_CSV, index_col=0, parse_dates=True)
    last = df.index.max()
    spy = _eod_close('stock', 'SPY', last)
    vix = _eod_close('index', 'VIX', last)
    if spy is None:
        print('[underlying] ThetaData SPY EOD unavailable — keeping existing'); return
    add = pd.DataFrame({'SPY': spy, 'VIX': vix}).dropna(subset=['SPY'])
    add = add[add.index > last]
    if add.empty:
        print('[underlying] already current'); return
    out = pd.concat([df, add]).sort_index()
    out = out[~out.index.duplicated(keep='last')]
    out.to_csv(SPY_CSV)
    print(f'[underlying] +{len(add)} sessions → {out.index.max().date()} (SPY {out["SPY"].iloc[-1]:.2f}, VIX {out["VIX"].iloc[-1]:.1f})')


def _run(label, args):
    print(f"\n{'='*66}\n[{label}]\n{'='*66}", flush=True)
    r = subprocess.run([PY, os.path.join(THIS, args[0])] + args[1:])
    print(f"[{label}] exit {r.returncode}", flush=True)
    return r.returncode


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--no-oi', action='store_true')
    a = p.parse_args()
    try:
        refresh_underlying()
    except Exception as e:
        print(f'[underlying] skipped ({e!r})')
    rc = 0
    rc |= _run('SPY minute quotes', ['backfill_spy_intraday.py', '--workers', '6'])
    rc |= _run('SPY IV surface', ['backfill_spy_iv_pg.py'])
    if not a.no_oi:
        rc |= _run('SPY open interest', ['backfill_spy_oi.py'])
    print(f"\n{'='*66}\nDONE (exit {rc})", flush=True)
    sys.exit(rc)


if __name__ == '__main__':
    main()
