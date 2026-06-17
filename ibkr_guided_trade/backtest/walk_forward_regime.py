"""Walk-forward / rolling-window robustness for regime_wheel_boxx vs champion.

Params are FIXED (not refit per window), so this is a rolling-window CONSISTENCY test:
does the regime+BOXX edge hold across every sub-period (2021-22 spike, 2023 decline,
2024-26 grind), or is it concentrated in one window? Also logs each window's leading
signals (avg regime_strength, storage-surprise vol, IV-rank) and correlates them with
the relative outperformance — to find EARLY SIGNS of when the alpha shows up.
"""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd, numpy as np
import replay_engine as R

WIN = 252        # 12-month test window
STEP = 63        # step quarterly


def met(nav):
    nav = nav.dropna()
    if len(nav) < 60: return None
    r = nav.pct_change().dropna(); y = (nav.index[-1]-nav.index[0]).days/365.25
    return (((nav.iloc[-1]/nav.iloc[0])**(1/y)-1)*100,
            r.mean()/(r.std()+1e-12)*math.sqrt(252),
            (nav/nav.cummax()-1).min()*100)


def run(df, key, sl):
    h, _ = R.run_strategy_simple(df.loc[sl], R.STRATEGIES[key], 100000, 0)
    h = h.set_index(pd.to_datetime(h['date']))
    return met(h['nav'])


def main():
    df = pd.read_csv(os.path.join(R.CACHE_DIR, 'master_dataset.csv'), parse_dates=[0], index_col=0)
    df = R.precompute_factor_z(df).dropna(subset=['UNG'])
    dates = df.index
    rows = []
    i = 252  # leave 1y lookback for signals
    while i + WIN < len(dates):
        s, e = dates[i], dates[i+WIN]
        sl = slice(s, e)
        ch = run(df, 'champion_kold15_ivrank_kbh', sl)
        rw = run(df, 'regime_wheel_boxx', sl)
        if not ch or not rw:
            i += STEP; continue
        seg = df.loc[sl]
        rows.append({
            'start': str(s.date()), 'end': str(e.date()),
            'ung_ret': (seg['UNG'].iloc[-1]/seg['UNG'].iloc[0]-1)*100,
            'champ_ann': ch[0], 'champ_sh': ch[1], 'champ_mdd': ch[2],
            'boxx_ann': rw[0], 'boxx_sh': rw[1], 'boxx_mdd': rw[2],
            'd_ann': rw[0]-ch[0], 'd_sh': rw[1]-ch[1], 'd_mdd': rw[2]-ch[2],
            'avg_regime': seg['regime_strength'].mean(),
            'ssz_vol': seg['storage_surprise_z'].std(),
            'avg_ivr': seg['iv_rank'].mean() if 'iv_rank' in seg else np.nan,
        })
        i += STEP
    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(os.path.dirname(__file__), 'results', 'walk_forward_regime.csv'), index=False)
    print(f"=== WALK-FORWARD: regime_wheel_boxx vs champion ({len(r)} rolling 12mo windows) ===\n")
    print(f"{'window':24}{'UNG%':>7}{'champ Sh':>9}{'boxx Sh':>9}{'Δann':>7}{'ΔSh':>7}{'ΔMDD':>7}{'regime':>8}")
    for _, x in r.iterrows():
        print(f"{x['start']}→{x['end'][:7]:11}{x['ung_ret']:+7.0f}{x['champ_sh']:9.2f}{x['boxx_sh']:9.2f}"
              f"{x['d_ann']:+7.1f}{x['d_sh']:+7.2f}{x['d_mdd']:+7.1f}{x['avg_regime']:+8.2f}")
    print(f"\n=== ROBUSTNESS ===")
    print(f"  windows boxx beats champ on Sharpe: {(r['d_sh']>0).mean()*100:.0f}%  on return: {(r['d_ann']>0).mean()*100:.0f}%  on MDD: {(r['d_mdd']>0).mean()*100:.0f}%")
    print(f"  median Δann {r['d_ann'].median():+.1f}pp  median ΔSharpe {r['d_sh'].median():+.2f}  median ΔMDD {r['d_mdd'].median():+.1f}pp")
    print(f"  worst window for boxx: Δann {r['d_ann'].min():+.1f}pp at {r.loc[r['d_ann'].idxmin(),'start']}")
    print(f"\n=== EARLY SIGNS OF ALPHA (corr of relative outperformance with leading signals) ===")
    for sig in ['avg_regime', 'ssz_vol', 'avg_ivr', 'ung_ret']:
        if r[sig].notna().sum() > 5:
            c_ann = r['d_ann'].corr(r[sig]); c_sh = r['d_sh'].corr(r[sig])
            print(f"  {sig:12}: corr w/ Δann {c_ann:+.2f}   corr w/ ΔSharpe {c_sh:+.2f}")
    # where does the edge concentrate?
    hi = r[r['avg_regime'] > r['avg_regime'].median()]
    lo = r[r['avg_regime'] <= r['avg_regime'].median()]
    print(f"\n  edge in HIGH-distribute-regime windows: Δann {hi['d_ann'].mean():+.1f}pp / ΔSh {hi['d_sh'].mean():+.2f}")
    print(f"  edge in LOW-distribute-regime  windows: Δann {lo['d_ann'].mean():+.1f}pp / ΔSh {lo['d_sh'].mean():+.2f}")


if __name__ == '__main__':
    main()
