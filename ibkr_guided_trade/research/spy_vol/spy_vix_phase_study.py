"""VIX PHASE / POST-SPIKE CONSOLIDATION study for vega-scraping timing.

Q: after IV spikes high and comes back DOWN, how long until VIX consolidates at a low, and is
   entering long-vega at CONSOLIDATION better than entering while VIX is still FALLING?

Part A (VIX-only): for each spike (VIX peak ≥ SPIKE), days until VIX first < LOW, and until
   it CONSOLIDATES (VIX<LOW and 10d std < STABLE).
Part B (straddle returns by phase): tag each weekly entry FALLING vs CONSOLIDATED-LOW vs other,
   and compare the long ATM straddle vega-scrape (reuses spy_vega_study pricing).

  venv/bin/python research/spy_vol/spy_vix_phase_study.py
"""
import os
import sys
from collections import defaultdict
import numpy as np
import pandas as pd

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
from spy_vega_study import _conn, pick_entry, run_trade

SPY_CSV = os.path.join(THIS, 'cache', 'spy_vix_daily.csv')
SPIKE, LOW, STABLE = 25.0, 15.0, 1.2


def main():
    spv = pd.read_csv(SPY_CSV, index_col=0, parse_dates=True); spv.index = spv.index.normalize()
    vix = spv['VIX']
    v10std = vix.rolling(10).std()
    vmax20 = vix.rolling(20).max()

    # ---- Part A: spike → comedown → consolidation wait times ----
    above = vix >= SPIKE
    # spike "peaks" = first day of each contiguous ≥SPIKE episode
    starts = vix.index[above & ~above.shift(1, fill_value=False)]
    waits_low, waits_cons = [], []
    for s in starts:
        fut = vix.loc[s:]
        lo = fut[fut < LOW]
        if len(lo):
            waits_low.append((lo.index[0] - s).days)
            consd = fut[(fut < LOW) & (v10std.loc[fut.index] < STABLE)]
            if len(consd):
                waits_cons.append((consd.index[0] - s).days)
    print(f"=== Part A: after a VIX spike (≥{SPIKE}), wait to LOW(<{LOW}) / CONSOLIDATION(<{LOW} & 10d-std<{STABLE}) ===")
    print(f"  spikes: {len(starts)} | reached low: {len(waits_low)} | consolidated: {len(waits_cons)}")
    if waits_low:
        print(f"  days to first VIX<{LOW}:   median {int(np.median(waits_low))}  IQR {int(np.percentile(waits_low,25))}-{int(np.percentile(waits_low,75))}  (range {min(waits_low)}-{max(waits_low)})")
    if waits_cons:
        print(f"  days to CONSOLIDATION:     median {int(np.median(waits_cons))}  IQR {int(np.percentile(waits_cons,25))}-{int(np.percentile(waits_cons,75))}")

    # ---- Part B: straddle returns by vol-PHASE ----
    def phase(d):
        v = vix.loc[d]; vm = vmax20.loc[d]; vs = v10std.loc[d]
        if vs != vs:  # nan early
            return 'other'
        if v < LOW and vs < STABLE:
            return 'CONSOLIDATED-LOW'      # came down + stable → thesis entry
        if v < LOW and vs >= STABLE:
            return 'LOW-unsettled'
        if v >= vm - 4 and v >= SPIKE:
            return 'SPIKING/high'
        if v < vm - 5 and v >= LOW:
            return 'FALLING (still coming down)'
        return 'mid'

    entries = spv[spv.index.weekday == 0]
    conn = _conn(); cur = conn.cursor()
    res = defaultdict(list)
    for d_ts, row in entries.iterrows():
        d = d_ts.date(); spot = float(row['SPY']); vix0 = float(row['VIX'])
        pe = pick_entry(cur, d, spot, 38, 52)
        if not pe:
            continue
        exp, K, dte = pe
        vix_path = {pd.Timestamp(k).date(): v for k, v in zip(spv.index, spv['VIX'])}
        tr = run_trade(cur, d, spot, exp, K, 30, 0.30, 0.40, 3.0, vix0, vix_path)
        if tr:
            res[phase(d_ts)].append(tr['ret'])
    conn.close()

    print("\n=== Part B: long ATM straddle (~45 DTE) vega-scrape by VOL PHASE at entry ===")
    hdr = f"{'phase':<28}{'n':>4}{'win%':>7}{'avg':>9}{'median':>9}{'total':>9}"
    print(hdr); print('-' * len(hdr))
    for name in ['CONSOLIDATED-LOW', 'LOW-unsettled', 'FALLING (still coming down)',
                 'SPIKING/high', 'mid', 'other']:
        r = np.array(res.get(name, []))
        if len(r) == 0:
            print(f"{name:<28}{0:>4}"); continue
        print(f"{name:<28}{len(r):>4}{(r>0).mean()*100:>6.0f}%{r.mean():>+9.1%}{np.median(r):>+9.1%}{r.sum():>+9.1%}")
    print("\nDONE", flush=True)


if __name__ == '__main__':
    main()
