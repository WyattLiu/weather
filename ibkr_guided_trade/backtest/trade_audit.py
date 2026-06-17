"""Trade-by-trade AUDIT under verifiable minute fills — understand EVERY action.

Part A — per-TYPE gross economics from the baseline run: count, premium collected,
realized P&L, and the SPREAD $ actually paid (half-spread × qty, from the audit fields).
Shows where money is made and where it leaks to the bid/ask.

Part B — MARGINAL ablation ([[feedback_attribution_counterfactual]]): re-run with each
mechanism turned OFF and measure the OOS delta. A churn action only earns its keep if
removing it HURTS (return or risk) by more than the spread it saves.
"""
import os
import sys
import math
import pandas as pd

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
import replay_engine as R

OOS = '2024-01-02'
BASE = {'intraday_exec': True, 'exec_window': 15, 'avoid_eia_print': True}
ABLATIONS = {
    'baseline':            {},
    'no_PUT_TP':           {'ablate_put_tp': True},   # stop taking profit on puts
    'no_CALL_TP':          {'ablate_call_tp': True},   # stop taking profit on calls
    'no_ROLL_DOWN':        {'roll_down': False},        # let puts assign instead of rolling
    'no_ELEVATOR':         {'elevator_close': False},   # no lock-the-rally close
}


def met(nav):
    nav = nav.dropna(); r = nav.pct_change().dropna()
    y = (nav.index[-1] - nav.index[0]).days / 365.25
    return (round(((nav.iloc[-1] / nav.iloc[0]) ** (1 / y) - 1) * 100, 1),
            round(r.mean() / (r.std() + 1e-12) * math.sqrt(252), 2),
            round((nav / nav.cummax() - 1).min() * 100, 1))


def spread_cost(row):
    """Half-spread $ crossed on this fill (what realistic execution paid vs mid)."""
    b, a, q = row.get('bid'), row.get('ask'), row.get('qty')
    if b is None or a is None or b != b or a != a:
        return 0.0
    return (a - b) / 2.0 * (abs(q) if q == q and q else 1) * 100


def main():
    df = pd.read_csv(os.path.join(R.CACHE_DIR, 'master_dataset.csv'), parse_dates=[0], index_col=0)
    df = R.precompute_factor_z(df).dropna(subset=['UNG']).loc[OOS:]
    out = []
    base_t = None
    for name, ov in ABLATIONS.items():
        params = {**R.STRATEGIES['champion_kold15_ivrank_kbh'], **BASE, **ov}
        h, t = R.run_strategy_simple(df, params, 100000, 0)
        h = h.set_index(pd.to_datetime(h['date']))
        m = met(h['nav'])
        out.append((name, m))
        if name == 'baseline':
            base_t = t
        print(f"{name:14} OOS  {m[0]:+5.1f}% / Sh {m[1]:.2f} / MDD {m[2]:+.1f}%", flush=True)

    # Part A — per-type economics from baseline
    print("\n=== PER-TYPE ECONOMICS (baseline, verifiable fills) ===")
    t = base_t
    t['spr$'] = t.apply(spread_cost, axis=1)
    print(f"{'type':16} {'n':>5} {'credit$':>10} {'pnl$':>10} {'spread$paid':>12} {'avg_spr%':>9}")
    for ty, sub in t.groupby('type'):
        cr = sub['credit'].sum() if 'credit' in sub else 0
        pn = sub['pnl'].sum() if 'pnl' in sub else 0
        sp = sub['spr$'].sum()
        asp = sub['spread_pct'].mean() if 'spread_pct' in sub else float('nan')
        if len(sub) < 2 and abs(cr) + abs(pn) + sp < 1:
            continue
        print(f"{ty:16} {len(sub):5} {cr:10,.0f} {pn:10,.0f} {sp:12,.0f} {asp:9.1f}")

    # Part B — marginal deltas vs baseline
    print("\n=== MARGINAL CONTRIBUTION (Δ vs baseline; negative Δreturn = removing it HURTS = it earns its keep) ===")
    b = dict(out)['baseline']
    for name, m in out:
        if name == 'baseline':
            continue
        print(f"  {name:14} Δret {m[0]-b[0]:+5.1f}pp  ΔSharpe {m[1]-b[1]:+.2f}  ΔMDD {m[2]-b[2]:+5.1f}pp"
              f"   → {'KEEP (removing hurts)' if (m[1] < b[1] - 0.02 or m[2] < b[2] - 1) else 'QUESTION (removing helps/neutral)'}")


if __name__ == '__main__':
    main()
