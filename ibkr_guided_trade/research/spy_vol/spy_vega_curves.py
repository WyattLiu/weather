"""Continuous curves for the SPY long-vega edge — we have the data, so draw it.

Curve 1: edge vs DTE   (CONSOLIDATED-LOW entries, daily) → which tenor maximizes the vega-scrape.
Curve 2: edge vs entry-VIX (all daily entries, 45 DTE)   → the monotonic 'buy cheap vega' surface.
Daily entries (not weekly) for a smoother curve. Saves PNGs + prints the tables.

  venv/bin/python research/spy_vol/spy_vega_curves.py
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
from spy_vega_study import _conn, pick_entry, run_trade

plt.style.use('seaborn-v0_8-whitegrid')
SPY_CSV = os.path.join(THIS, 'cache', 'spy_vix_daily.csv')


def main():
    spv = pd.read_csv(SPY_CSV, index_col=0, parse_dates=True); spv.index = spv.index.normalize()
    vix = spv['VIX']; v10 = vix.rolling(10).std()
    vix_path = {pd.Timestamp(k).date(): v for k, v in zip(spv.index, spv['VIX'])}
    conn = _conn(); cur = conn.cursor()

    def trade_ret(d, spot, vix0, lo, hi, hold):
        pe = pick_entry(cur, d, spot, lo, hi)
        if not pe:
            return None
        exp, K, dte = pe
        tr = run_trade(cur, d, spot, exp, K, hold, 0.30, 0.40, 3.0, vix0, vix_path)
        return tr['ret'] if tr else None

    # ---- Curve 1: edge vs DTE (consolidated-low, daily entries) ----
    cons = spv[(vix < 15) & (v10 < 1.2)]
    DTES = list(range(21, 86, 7))
    c1 = []
    for dt in DTES:
        rs = []
        for d_ts, row in cons.iterrows():
            r = trade_ret(d_ts.date(), float(row['SPY']), float(row['VIX']),
                          dt - 5, dt + 5, min(dt - 7, 35))
            if r is not None:
                rs.append(r)
        if rs:
            c1.append((dt, len(rs), np.mean(rs), (np.array(rs) > 0).mean()))
    print("=== Curve 1: long-vega edge vs DTE (CONSOLIDATED-LOW, daily) ===")
    print(f"{'DTE':>5}{'n':>5}{'avg ret':>9}{'win%':>7}")
    for dt, n, a, w in c1:
        print(f"{dt:>5}{n:>5}{a:>+9.1%}{w*100:>6.0f}%")

    # ---- Curve 2: edge vs entry-VIX (all daily entries, 45 DTE) ----
    alld = spv.iloc[::1]              # every trading day
    pts = []
    for d_ts, row in alld.iterrows():
        r = trade_ret(d_ts.date(), float(row['SPY']), float(row['VIX']), 40, 50, 30)
        if r is not None:
            pts.append((float(row['VIX']), r))
    conn.close()
    pts = np.array(pts)
    # bin by VIX
    edges = [10, 12, 14, 16, 18, 20, 24, 30, 60]
    print("\n=== Curve 2: long-vega edge (45 DTE) vs entry VIX bin ===")
    print(f"{'VIX bin':>10}{'n':>5}{'avg ret':>9}{'win%':>7}")
    c2 = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (pts[:, 0] >= lo) & (pts[:, 0] < hi)
        if m.sum() >= 5:
            a = pts[m, 1].mean(); w = (pts[m, 1] > 0).mean()
            c2.append(((lo + hi) / 2, m.sum(), a, w))
            print(f"{f'{lo}-{hi}':>10}{m.sum():>5}{a:>+9.1%}{w*100:>6.0f}%")

    # ---- plot ----
    fig, ax = plt.subplots(1, 2, figsize=(13, 5), dpi=150)
    if c1:
        x = [r[0] for r in c1]; y = [r[2] * 100 for r in c1]
        ax[0].plot(x, y, 'o-', color='#1f77b4', lw=2)
        ax[0].axhline(0, color='gray', lw=.8); ax[0].axvline(45, color='red', ls='--', alpha=.5, label='45 DTE')
        ax[0].set(xlabel='DTE at entry', ylabel='avg straddle return (%)',
                  title='Long-vega edge vs DTE\n(consolidated-low entries)')
        ax[0].legend()
    if c2:
        x = [r[0] for r in c2]; y = [r[2] * 100 for r in c2]
        ax[1].plot(x, y, 'o-', color='#2ca02c', lw=2)
        ax[1].axhline(0, color='gray', lw=.8); ax[1].axvline(15, color='red', ls='--', alpha=.5, label='VIX 15')
        ax[1].set(xlabel='entry VIX', ylabel='avg straddle return (%)',
                  title='Long-vega edge vs entry VIX\n(45 DTE, all daily entries)')
        ax[1].legend()
    plt.tight_layout()
    out = os.path.join(THIS, 'vega_curves.png')
    plt.savefig(out); print(f"\nsaved {out}")
    print("DONE", flush=True)


if __name__ == '__main__':
    main()
