"""Walk-forward of regime_wheel_boxx_greeks (delta-hedge + gamma-cap) vs the champion,
under VERIFIABLE MINUTE FILLS (intraday_exec). Rolling 12mo windows, stepped ~6mo (fewer
windows because verifiable fills are slow per-fill against the 198M-row minute table).
Confirms the greeks-managed layer holds its edge across sub-periods before promotion.
"""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import replay_engine as R

WIN, STEP = 252, 126
MIN = {'intraday_exec': True, 'exec_window': 15, 'avoid_eia_print': True}


def met(nav):
    nav = nav.dropna()
    if len(nav) < 60:
        return None
    r = nav.pct_change().dropna(); y = (nav.index[-1] - nav.index[0]).days / 365.25
    return (((nav.iloc[-1] / nav.iloc[0]) ** (1 / y) - 1) * 100,
            r.mean() / (r.std() + 1e-12) * math.sqrt(252),
            (nav / nav.cummax() - 1).min() * 100)


def run(df, key, sl):
    p = {**R.STRATEGIES[key], **MIN}
    h, _ = R.run_strategy_simple(df.loc[sl], p, 100000, 0)
    h = h.set_index(pd.to_datetime(h['date']))
    return met(h['nav'])


def main():
    df = pd.read_csv(os.path.join(R.CACHE_DIR, 'master_dataset.csv'), parse_dates=[0], index_col=0)
    df = R.precompute_factor_z(df).dropna(subset=['UNG'])
    dates = df.index
    rows, i = [], 252
    while i + WIN < len(dates):
        s, e = dates[i], dates[i + WIN]
        ch = run(df, 'champion_kold15_ivrank_kbh', slice(s, e))
        gk = run(df, 'regime_wheel_boxx_greeks', slice(s, e))
        if ch and gk:
            rows.append({'start': str(s.date()),
                         'ung': (df.loc[s:e, 'UNG'].iloc[-1] / df.loc[s:e, 'UNG'].iloc[0] - 1) * 100,
                         'ch_ann': ch[0], 'ch_sh': ch[1], 'ch_mdd': ch[2],
                         'gk_ann': gk[0], 'gk_sh': gk[1], 'gk_mdd': gk[2],
                         'd_ann': gk[0] - ch[0], 'd_sh': gk[1] - ch[1], 'd_mdd': gk[2] - ch[2]})
            print(f"  {s.date()}→{e.date()}  UNG{rows[-1]['ung']:+5.0f}%  "
                  f"champ {ch[0]:+5.1f}/{ch[1]:.2f}/{ch[2]:.0f}  greeks {gk[0]:+5.1f}/{gk[1]:.2f}/{gk[2]:.0f}  "
                  f"Δsh {gk[1]-ch[1]:+.2f}", flush=True)
        i += STEP
    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(os.path.dirname(__file__), 'results', 'walk_forward_greeks.csv'), index=False)
    print(f"\n=== VERIFIABLE-FILL WALK-FORWARD: regime_wheel_boxx_greeks vs champion ({len(r)} windows) ===")
    print(f"  greeks beats champ on Sharpe: {(r['d_sh']>0).mean()*100:.0f}%  return: {(r['d_ann']>0).mean()*100:.0f}%  MDD: {(r['d_mdd']>0).mean()*100:.0f}%")
    print(f"  median Δann {r['d_ann'].median():+.1f}pp  ΔSharpe {r['d_sh'].median():+.2f}  ΔMDD {r['d_mdd'].median():+.1f}pp")
    print(f"  greeks: median Sharpe {r['gk_sh'].median():.2f} (champ {r['ch_sh'].median():.2f}); worst window Δann {r['d_ann'].min():+.1f}pp")
    print("DONE", flush=True)


if __name__ == '__main__':
    main()
