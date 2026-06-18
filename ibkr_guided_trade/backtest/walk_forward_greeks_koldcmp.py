"""ISOLATED KOLD A/B walk-forward, under VERIFIABLE MINUTE FILLS.

After removing the legacy KOLD book-hedge from the promoted greeks kernel (KOLD was a
silent net-delta blind spot — bearishness is meant to be expressed via long puts, which the
book-greeks engine actually sees), re-validate that the edge HOLDS without it.

Per rolling 12mo window we run THREE kernels on the SAME data + SAME fills:
  • champion        — champion_kold15_ivrank_kbh (reference baseline)
  • greeks_free     — regime_wheel_boxx_greeks AS PROMOTED NOW (kold_book_hedge=False)
  • greeks_kold     — same kernel with kold_book_hedge=True (the PREVIOUSLY-validated config)

The clean question is greeks_free vs greeks_kold: did dropping KOLD help, hurt, or wash?
(Champion stays in for continuity with the earlier walk-forward.)

  venv/bin/python backtest/walk_forward_greeks_koldcmp.py
"""
import os
import sys
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import replay_engine as R

WIN, STEP = 252, 126
MIN = {'intraday_exec': True, 'exec_window': 15, 'avoid_eia_print': True}
# greeks WITH the legacy KOLD book-hedge re-enabled (the pre-fix config)
KOLD_ON = {'kold_book_hedge': True, 'kold_shoulder_hedge': 0.15}


def met(nav):
    nav = nav.dropna()
    if len(nav) < 60:
        return None
    r = nav.pct_change().dropna()
    y = (nav.index[-1] - nav.index[0]).days / 365.25
    return (((nav.iloc[-1] / nav.iloc[0]) ** (1 / y) - 1) * 100,
            r.mean() / (r.std() + 1e-12) * math.sqrt(252),
            (nav / nav.cummax() - 1).min() * 100)


def run(df, params, sl):
    h, _ = R.run_strategy_simple(df.loc[sl], {**params, **MIN}, 100000, 0)
    h = h.set_index(pd.to_datetime(h['date']))
    return met(h['nav'])


def main():
    df = pd.read_csv(os.path.join(R.CACHE_DIR, 'master_dataset.csv'), parse_dates=[0], index_col=0)
    df = R.precompute_factor_z(df).dropna(subset=['UNG'])
    dates = df.index
    GK_FREE = R.STRATEGIES['regime_wheel_boxx_greeks']                 # KOLD off (promoted now)
    GK_KOLD = {**R.STRATEGIES['regime_wheel_boxx_greeks'], **KOLD_ON}  # KOLD on  (pre-fix)
    rows, i = [], 252
    while i + WIN < len(dates):
        s, e = dates[i], dates[i + WIN]
        ch = run(df, R.STRATEGIES['champion_kold15_ivrank_kbh'], slice(s, e))
        gf = run(df, GK_FREE, slice(s, e))
        gk = run(df, GK_KOLD, slice(s, e))
        if ch and gf and gk:
            rows.append({'start': str(s.date()),
                         'ung': (df.loc[s:e, 'UNG'].iloc[-1] / df.loc[s:e, 'UNG'].iloc[0] - 1) * 100,
                         'ch_ann': ch[0], 'ch_sh': ch[1], 'ch_mdd': ch[2],
                         'gf_ann': gf[0], 'gf_sh': gf[1], 'gf_mdd': gf[2],
                         'gk_ann': gk[0], 'gk_sh': gk[1], 'gk_mdd': gk[2],
                         # KOLD effect = free − kold (positive = removing KOLD HELPED)
                         'd_ann': gf[0] - gk[0], 'd_sh': gf[1] - gk[1], 'd_mdd': gf[2] - gk[2]})
            r = rows[-1]
            print(f"  {s.date()}→{e.date()}  UNG{r['ung']:+5.0f}%  "
                  f"champ {ch[0]:+5.1f}/{ch[1]:.2f}/{ch[2]:.0f}  "
                  f"gkFREE {gf[0]:+5.1f}/{gf[1]:.2f}/{gf[2]:.0f}  "
                  f"gkKOLD {gk[0]:+5.1f}/{gk[1]:.2f}/{gk[2]:.0f}  "
                  f"ΔKOLD(free−kold) ann{r['d_ann']:+.1f}/sh{r['d_sh']:+.2f}", flush=True)
        i += STEP
    rr = pd.DataFrame(rows)
    rr.to_csv(os.path.join(os.path.dirname(__file__), 'results', 'walk_forward_greeks_koldcmp.csv'), index=False)
    print(f"\n=== KOLD-FREE A/B WALK-FORWARD (verifiable fills, {len(rr)} windows) ===")
    print(f"  removing KOLD beats KOLD-on on Sharpe: {(rr['d_sh']>0).mean()*100:.0f}%  "
          f"return: {(rr['d_ann']>0).mean()*100:.0f}%  MDD: {(rr['d_mdd']>0).mean()*100:.0f}%")
    print(f"  median ΔKOLD: ann {rr['d_ann'].median():+.1f}pp  Sharpe {rr['d_sh'].median():+.2f}  "
          f"MDD {rr['d_mdd'].median():+.1f}pp   (positive = KOLD-free is better)")
    print(f"  greeks_free: median Sharpe {rr['gf_sh'].median():.2f}  ann {rr['gf_ann'].median():+.1f}  MDD {rr['gf_mdd'].median():.0f}")
    print(f"  greeks_kold: median Sharpe {rr['gk_sh'].median():.2f}  ann {rr['gk_ann'].median():+.1f}  MDD {rr['gk_mdd'].median():.0f}")
    print(f"  champion   : median Sharpe {rr['ch_sh'].median():.2f}  ann {rr['ch_ann'].median():+.1f}  MDD {rr['ch_mdd'].median():.0f}")
    print(f"  greeks_free beats champion on Sharpe: {(rr['gf_sh']>rr['ch_sh']).mean()*100:.0f}%  "
          f"MDD: {(rr['gf_mdd']>rr['ch_mdd']).mean()*100:.0f}%")
    print("DONE", flush=True)


if __name__ == '__main__':
    main()
