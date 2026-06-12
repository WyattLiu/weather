"""EIA storage-release microstructure study — 1-min UNG bars from IBKR.

EIA weekly NG storage prints Thursdays 10:30 ET. This pulls 1-minute
UNG bars for every release day (resumable, IBKR pacing-safe) and
measures announcement drift:

  - r_pre   (10:00 → 10:29)   positioning into the print
  - r_jump  (10:29 → 10:35)   the print reaction
  - r_drift (10:35 → 11:30)   does the jump CONTINUE or revert?
  - r_rest  (11:30 → 15:55)   rest-of-day
  Conditional: drift given jump sign/size (the market reveals the
  surprise; we test if the first reaction underreacts).

Usage:
    venv/bin/python research/microstructure/eia_release_study.py --fetch
    venv/bin/python research/microstructure/eia_release_study.py --study
Requires IB gateway at 192.168.1.127:20009 (same as ng_lne_chains.py).
"""
import os
import sys
import time
import argparse
from datetime import date, timedelta

import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BARS_DIR = os.path.join(THIS_DIR, 'bars')
os.makedirs(BARS_DIR, exist_ok=True)

IBKR_HOST = '192.168.1.127'
IBKR_PORT = 20009
CLIENT_ID = 98          # ng_lne_chains uses 97


def release_days(years=4):
    """Thursdays for the last N years (holiday shifts → bar file will be
    empty for non-release Thursdays; Friday releases get missed — ~3/yr,
    acceptable)."""
    out = []
    d = date.today()
    start = d - timedelta(days=365 * years)
    cur = start + timedelta(days=(3 - start.weekday()) % 7)  # first Thursday
    while cur < d:
        out.append(cur)
        cur += timedelta(days=7)
    return out


def fetch(years=4):
    from ib_insync import IB, Stock
    ib = IB()
    ib.connect(IBKR_HOST, IBKR_PORT, clientId=CLIENT_ID, timeout=30)
    contract = Stock('UNG', 'SMART', 'USD')
    ib.qualifyContracts(contract)
    days = release_days(years)
    done = skip = fail = 0
    for d in days:
        dest = os.path.join(BARS_DIR, f'ung_{d.isoformat()}.csv')
        if os.path.exists(dest):
            skip += 1
            continue
        end_dt = f'{d.strftime("%Y%m%d")} 16:30:00 US/Eastern'
        try:
            bars = ib.reqHistoricalData(
                contract, endDateTime=end_dt, durationStr='1 D',
                barSizeSetting='1 min', whatToShow='TRADES',
                useRTH=True, formatDate=1)
        except Exception as e:
            print(f'  {d}: FAILED ({e})')
            fail += 1
            time.sleep(5)
            continue
        df = pd.DataFrame([{'ts': b.date, 'open': b.open, 'high': b.high,
                            'low': b.low, 'close': b.close,
                            'volume': b.volume} for b in bars])
        df.to_csv(dest, index=False)
        done += 1
        if done % 20 == 0:
            print(f'  progress: {done} fetched, {skip} cached, {fail} failed')
        time.sleep(2.5)   # pacing: stay well under 60 req / 10 min
    ib.disconnect()
    print(f'[fetch] complete: {done} new, {skip} cached, {fail} failed')


def study():
    rows = []
    for f in sorted(os.listdir(BARS_DIR)):
        if not f.endswith('.csv'):
            continue
        df = pd.read_csv(os.path.join(BARS_DIR, f), parse_dates=['ts'])
        if df.empty:
            continue
        df['hm'] = df['ts'].dt.strftime('%H:%M')
        px = df.set_index('hm')['close']

        def at(hm, fallback=None):
            return float(px.get(hm, fallback)) if (hm in px.index or fallback) else None

        p1000, p1029 = at('10:00'), at('10:29')
        p1035, p1130 = at('10:35'), at('11:30')
        p1555 = at('15:55')
        if None in (p1000, p1029, p1035, p1130, p1555):
            continue
        rows.append({
            'date': f[4:14],
            'r_pre': p1029 / p1000 - 1,
            'r_jump': p1035 / p1029 - 1,
            'r_drift': p1130 / p1035 - 1,
            'r_rest': p1555 / p1130 - 1,
        })
    s = pd.DataFrame(rows)
    if s.empty:
        print('no bar data yet — run with --fetch first (gateway must be up)')
        return
    print(f'=== EIA RELEASE MICROSTRUCTURE ({len(s)} Thursdays) ===\n')
    print(s[['r_pre', 'r_jump', 'r_drift', 'r_rest']].describe().T
          [['mean', 'std', '50%']].round(5).to_string())

    # Drift conditional on jump
    from scipy import stats as sstats
    big = s['r_jump'].abs() > s['r_jump'].abs().median()
    same_sign = (s['r_drift'] * s['r_jump'] > 0)
    print(f'\nP(drift continues jump direction): {same_sign.mean():.1%} '
          f'(all) / {same_sign[big].mean():.1%} (big jumps)')
    up, dn = s[s['r_jump'] > 0], s[s['r_jump'] < 0]
    print(f'drift after UP jump:   {up["r_drift"].mean():+.3%} (n={len(up)})')
    print(f'drift after DOWN jump: {dn["r_drift"].mean():+.3%} (n={len(dn)})')
    t, p = sstats.ttest_ind(up['r_drift'], dn['r_drift'], equal_var=False)
    print(f'up-vs-down drift: t={t:.2f} p={p:.3f}')
    s.to_csv(os.path.join(THIS_DIR, 'eia_release_panel.csv'), index=False)
    print(f'\n→ {THIS_DIR}/eia_release_panel.csv')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--fetch', action='store_true')
    ap.add_argument('--study', action='store_true')
    ap.add_argument('--years', type=int, default=4)
    a = ap.parse_args()
    if a.fetch:
        fetch(a.years)
    if a.study or not a.fetch:
        study()
