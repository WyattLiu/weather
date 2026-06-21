"""WHAT TIME OF DAY to enter the long-vega straddle? (SPY, clean minute 2021-2026)

For each low-VIX/consolidated entry day, BUY the ATM straddle at each candidate clock time
(combo-mid fill from the real minute path), hold to a common exit (~21 DTE, EOD mid), and
compare: entry cost (net debit) and resulting return BY ENTRY HOUR. Exit is identical across
hours, so differences isolate the entry-timing/execution effect.

  venv/bin/python research/spy_vol/spy_intraday_timing.py
"""
import os
import sys
from collections import defaultdict
import numpy as np
import pandas as pd

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
from spy_vega_study import _conn, pick_entry
from spy_minute_combo_study import minute_net, fill_combo_mid

SPY_CSV = os.path.join(THIS, 'cache', 'spy_vix_daily.csv')
HOURS = ['09:45', '10:30', '11:30', '12:30', '13:30', '14:30', '15:45']


def eod_mid(cur, exp, K, right, d):
    cur.execute("""SELECT DISTINCT ON (data_type) data_type, close FROM spy_options_history
        WHERE trade_date=%s AND expiration=%s AND strike=%s AND option_right=%s
        ORDER BY data_type, bar_time DESC""", (d, exp, K, right))
    m = {dt: float(c) for dt, c in cur.fetchall()}
    return (m['BID'] + m['ASK']) / 2 if 'BID' in m and 'ASK' in m and m['ASK'] > m['BID'] else None


def main():
    spv = pd.read_csv(SPY_CSV, index_col=0, parse_dates=True); spv.index = spv.index.normalize()
    vix = spv['VIX']; v10 = vix.rolling(10).std()
    ent = spv[(spv.index.weekday == 0) & (vix < 16) & (v10 < 1.5)]   # low-ish & settled, 2021-26
    conn = _conn(); cur = conn.cursor()
    by_hour_ret = defaultdict(list); by_hour_cost = defaultdict(list)
    n = 0
    for d_ts, row in ent.iterrows():
        d = d_ts.date()
        if d_ts.year < 2021:                 # clean minute only 2021+
            continue
        spot = float(row['SPY'])
        pe = pick_entry(cur, d, spot, 38, 52)
        if not pe:
            continue
        exp, K, dte = pe
        net = minute_net(cur, exp, K, d)
        if not net or len(net) < 50:
            continue
        # common exit ~21 DTE later (first date with quotes), EOD mid
        exit_val = None
        for off in range(max(1, dte - 21), dte - 16):
            cand = (pd.Timestamp(d) + pd.Timedelta(days=off)).date()
            c = eod_mid(cur, exp, K, 'C', cand); p = eod_mid(cur, exp, K, 'P', cand)
            if c and p:
                exit_val = c + p; break
        if not exit_val:
            continue
        n += 1
        for h in HOURS:
            entry = fill_combo_mid(net, 'buy', start=h, work=15)
            if entry and entry > 0:
                by_hour_ret[h].append(exit_val / entry - 1)
                by_hour_cost[h].append(entry)
    conn.close()

    print(f"=== SPY entry TIME-OF-DAY (low-VIX entries 2021-2026, n={n}) ===")
    print(f"{'enter @':<9}{'n':>4}{'avg ret':>9}{'win%':>7}{'avg entry$':>11}")
    print('-' * 40)
    base_cost = np.mean(by_hour_cost['12:30']) if by_hour_cost['12:30'] else 1
    for h in HOURS:
        r = np.array(by_hour_ret[h]); c = np.array(by_hour_cost[h])
        if len(r):
            print(f"{h:<9}{len(r):>4}{r.mean():>+9.1%}{(r>0).mean()*100:>6.0f}%{c.mean():>11.2f}")
    # cheapest entry hour (lowest net debit = best fill, controlling for the day)
    print("\n  relative entry cost vs 12:30 (lower = cheaper straddle that hour):")
    for h in HOURS:
        c = by_hour_cost[h]
        if c:
            print(f"    {h}: {(np.mean(c)/base_cost-1)*100:+.2f}%")
    print("\nDONE", flush=True)


if __name__ == '__main__':
    main()
