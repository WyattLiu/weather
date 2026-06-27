"""Spike-leg study — tests the insight: on an intraday SPIKE UP, prefer SELLING THE CALL;
on a SPIKE DOWN, prefer SELLING THE PUT. Should you WAIT for the spike to sell the
richening side?

Uses the reconstructed intraday underlying (put-call parity) + the ATM call/put minute
mids. For each day we find the session's dominant intraday excursion (the spike) and ask,
for the spike-ALIGNED leg (call on up-spike / put on down-spike) vs the ANTI-aligned leg:
  • PREMIUM PICKUP from waiting: sell at the spike-extreme minute vs at the open.
  • FILL QUALITY: median spread on each side during/after the spike.
If selling the aligned leg into the spike captures more premium at no worse spread, the
rule 'sell the moving side' is confirmed.
"""
import os
import sys
import argparse
import pandas as pd
import psycopg2

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
import replay_engine as R
from intraday_underlying import atm_frame, DB


def main(start, end, spike_thr):
    u = pd.read_csv(os.path.join(R.CACHE_DIR, 'master_dataset.csv'),
                    index_col=0, parse_dates=True)['UNG'].dropna()
    u.index = u.index.normalize()
    u = u.loc[start:end]
    conn = psycopg2.connect(**DB)
    rows = []
    for d, spot in u.items():
        f = atm_frame(d, float(spot), conn=conn)
        if f is None or len(f) < 30:
            continue
        S = f['S'].values
        o = S[0]
        i_hi, i_lo = int(S.argmax()), int(S.argmin())
        up = (S[i_hi] / o - 1) * 100
        dn = (S[i_lo] / o - 1) * 100
        # dominant excursion = the spike; require it to come AFTER the open (waitable)
        if abs(up) >= abs(dn):
            direction, i_ext, mag = 'up', i_hi, up
        else:
            direction, i_ext, mag = 'down', i_lo, dn
        if abs(mag) < spike_thr or i_ext < 5:
            continue
        # premium pickup from waiting to sell, on each leg (sell → want higher mid)
        call_pickup = (f['call_mid'].iloc[i_ext] / f['call_mid'].iloc[0] - 1) * 100
        put_pickup = (f['put_mid'].iloc[i_ext] / f['put_mid'].iloc[0] - 1) * 100
        rows.append({'date': d, 'dir': direction, 'spike_pct': round(mag, 2),
                     'call_pickup': call_pickup, 'put_pickup': put_pickup,
                     'call_spr_ext': f['call_spr'].iloc[i_ext],
                     'put_spr_ext': f['put_spr'].iloc[i_ext],
                     'aligned_pickup': call_pickup if direction == 'up' else put_pickup,
                     'anti_pickup': put_pickup if direction == 'up' else call_pickup,
                     'aligned_spr': f['call_spr'].iloc[i_ext] if direction == 'up' else f['put_spr'].iloc[i_ext],
                     'anti_spr': f['put_spr'].iloc[i_ext] if direction == 'up' else f['call_spr'].iloc[i_ext]})
    conn.close()
    r = pd.DataFrame(rows)
    if not len(r):
        print("no spike days"); return
    r.to_csv(os.path.join(THIS, 'results', 'spike_study.csv'), index=False)
    nu = (r['dir'] == 'up').sum(); nd = (r['dir'] == 'down').sum()
    print(f"=== SPIKE-LEG STUDY  ({start}→{end}, |spike|≥{spike_thr}%) ===")
    print(f"spike days: {len(r)}  (up {nu}, down {nd})\n")
    print("PREMIUM PICKUP from waiting to sell at the spike extreme (vs open; higher=better):")
    print(f"  ALIGNED leg (call on up / put on down):   mean {r['aligned_pickup'].mean():+6.1f}%  median {r['aligned_pickup'].median():+6.1f}%")
    print(f"  ANTI    leg (put on up / call on down):    mean {r['anti_pickup'].mean():+6.1f}%  median {r['anti_pickup'].median():+6.1f}%")
    edge = r['aligned_pickup'].mean() - r['anti_pickup'].mean()
    print(f"  → selling the ALIGNED (moving) leg into the spike picks up {edge:+.1f}pp more premium\n")
    print("FILL QUALITY at the spike extreme (median spread %, lower=better):")
    print(f"  ALIGNED leg spread: {r['aligned_spr'].median():.1f}%   ANTI leg spread: {r['anti_spr'].median():.1f}%")
    print("\nBY DIRECTION:")
    for dr, sub in r.groupby('dir'):
        leg = 'CALL' if dr == 'up' else 'PUT'
        print(f"  spike {dr:4} (n={len(sub)}): sell {leg} pickup {sub['aligned_pickup'].mean():+.1f}%  "
              f"vs other leg {sub['anti_pickup'].mean():+.1f}%  | {leg} spread {sub['aligned_spr'].median():.1f}%")
    verdict = "CONFIRMED — sell the moving side into the spike" if edge > 3 else \
              ("WEAK — small/no edge" if edge > -3 else "INVERTED — fade, don't chase")
    print(f"\n  VERDICT: {verdict}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='2021-06-17')
    ap.add_argument('--end', default='2026-06-12')
    ap.add_argument('--spike_thr', type=float, default=2.0)
    a = ap.parse_args()
    main(a.start, a.end, a.spike_thr)
