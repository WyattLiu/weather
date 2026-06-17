"""MULTI-HORIZON walk-forward: regime_wheel_boxx_v4 (crash-fallback) vs champion vs the
original boxx, across rolling windows of 6/12/18/24 months. Tests (a) horizon-dependence
of the edge and (b) whether the crash-fallback closes the 2022 blind spot.
"""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd, numpy as np
import replay_engine as R

HORIZONS = {'6mo': 126, '12mo': 252, '18mo': 378, '24mo': 504}
STEP = 63


def met(nav):
    nav = nav.dropna()
    if len(nav) < 50: return None
    r = nav.pct_change().dropna(); y = (nav.index[-1]-nav.index[0]).days/365.25
    return (((nav.iloc[-1]/nav.iloc[0])**(1/y)-1)*100, r.mean()/(r.std()+1e-12)*math.sqrt(252),
            (nav/nav.cummax()-1).min()*100)


def run(df, key, sl):
    h, _ = R.run_strategy_simple(df.loc[sl], R.STRATEGIES[key], 100000, 0)
    h = h.set_index(pd.to_datetime(h['date']))
    return met(h['nav'])


def main():
    df = pd.read_csv(os.path.join(R.CACHE_DIR, 'master_dataset.csv'), parse_dates=[0], index_col=0)
    df = R.precompute_factor_z(df).dropna(subset=['UNG'])
    dates = df.index
    allrows = []
    for hname, WIN in HORIZONS.items():
        i = 252
        while i + WIN < len(dates):
            s, e = dates[i], dates[i+WIN]
            sl = slice(s, e)
            ch = run(df, 'champion_kold15_ivrank_kbh', sl)
            v4 = run(df, 'regime_wheel_boxx_v4', sl)
            b0 = run(df, 'regime_wheel_boxx', sl)
            if ch and v4 and b0:
                allrows.append({'h': hname, 'start': str(s.date()),
                    'ung_ret': (df.loc[sl, 'UNG'].iloc[-1]/df.loc[sl, 'UNG'].iloc[0]-1)*100,
                    'd_ann': v4[0]-ch[0], 'd_sh': v4[1]-ch[1], 'd_mdd': v4[2]-ch[2],
                    'v4_vs_b0': v4[0]-b0[0], 'v4_sh': v4[1], 'b0_sh': b0[1]})
            i += STEP
        print(f"  {hname}: done", flush=True)
    r = pd.DataFrame(allrows)
    r.to_csv(os.path.join(os.path.dirname(__file__), 'results', 'walk_forward_mh.csv'), index=False)
    print(f"\n=== MULTI-HORIZON WALK-FORWARD (v4 crash-fallback vs champion) ===")
    print(f"{'horizon':8}{'#win':>5}{'Sh-win%':>9}{'med Δann':>9}{'med ΔSh':>9}{'med ΔMDD':>9}{'worst Δann':>11}")
    for h in HORIZONS:
        sub = r[r['h'] == h]
        if not len(sub): continue
        print(f"{h:8}{len(sub):5}{(sub['d_sh']>0).mean()*100:8.0f}%{sub['d_ann'].median():+9.1f}"
              f"{sub['d_sh'].median():+9.2f}{sub['d_mdd'].median():+9.1f}{sub['d_ann'].min():+11.1f}")
    print(f"\n=== DID CRASH-FALLBACK CLOSE THE 2022 BLIND SPOT? (v4 vs original boxx) ===")
    crash = r[r['start'].str.startswith('2022')]
    print(f"  2022-start windows: median v4-vs-boxx0 Δann {crash['v4_vs_b0'].median():+.1f}pp "
          f"(min {crash['v4_vs_b0'].min():+.1f}, max {crash['v4_vs_b0'].max():+.1f})")
    print(f"  overall v4-vs-boxx0: median {r['v4_vs_b0'].median():+.1f}pp  | windows v4 better: {(r['v4_vs_b0']>0).mean()*100:.0f}%")
    print(f"  v4 worst window overall: Δann {r['d_ann'].min():+.1f}pp")


if __name__ == '__main__':
    main()
