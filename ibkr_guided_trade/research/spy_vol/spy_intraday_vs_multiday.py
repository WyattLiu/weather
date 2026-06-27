"""SPY vega-scraping: INTRADAY vs MULTI-DAY hold, + which signals boost the frontier.

For dense SPY entries 2018-2026 (Mon/Wed/Fri), buy the ~45 DTE ATM straddle at 14:00 (the cheap
afternoon window) via combo-mid, then measure the mid-to-mid return at a ladder of holding horizons:
  h=0  = INTRADAY (exit same-day close, ~2h)        ← pure intraday gamma/vega scrape
  h=0* = INTRADAY-MAX (best mid after entry, oracle ceiling for an intraday pop)
  h=1,2,3,5,10 days = MULTI-DAY (exit that day's close)
Mid-to-mid isolates the vega/gamma/theta; one round-trip combo spread (~0.5%) is paid once either way.

Then for the best horizon, bucket forward return by candidate SIGNALS to see what lifts return/Sharpe:
  iv_atm−rv20 (cheap vs realized) · skew (flat) · vix_std10 (consolidated) · overnight gap ·
  intraday RV 09:30→14:00 (is today already moving?) · VIX level · day-of-week.

  venv/bin/python research/spy_vol/spy_intraday_vs_multiday.py
"""
import os
import sys
import math
import numpy as np
import pandas as pd

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
from spy_vega_study import _conn, pick_entry
from spy_minute_combo_study import minute_net, fill_combo_mid

SPY_CSV = os.path.join(THIS, 'cache', 'spy_vix_daily.csv')
HOR = [1, 2, 3, 5, 10]
ENTRY_T = '14:00'


def combo_mid_path(net):
    """sorted [(hhmm, mid)] from a minute_net dict {hhmm:(bid,ask)}."""
    return [(t, (b + a) / 2) for t, (b, a) in sorted(net.items())]


def close_mid(cur, exp, K, d):
    net = minute_net(cur, exp, K, d)
    if not net:
        return None
    return combo_mid_path(net)[-1][1]


def leg_mid(cur, exp, K, right, d, at):
    cur.execute("""SELECT bar_time, data_type, close FROM spy_options_history
        WHERE trade_date=%s AND expiration=%s AND strike=%s AND option_right=%s
        AND bar_time::time >= %s ORDER BY bar_time LIMIT 40""", (d, exp, K, right, at))
    bid = ask = None
    for bt, dt, c in cur.fetchall():
        if dt == 'BID' and bid is None:
            bid = float(c)
        elif dt == 'ASK' and ask is None:
            ask = float(c)
        if bid and ask:
            return (bid + ask) / 2
    return None


def atm_iv(straddle, S, T):
    """Invert ATM straddle ≈ 0.7979·S·σ·√T  →  σ."""
    if straddle <= 0 or T <= 0:
        return None
    return straddle / (0.7979 * S * math.sqrt(T))


def wing_skew(cur, exp, d, S, dte, ks):
    """5% put_iv − call_iv via the ATM-straddle-implied σ scaled by price ratio (cheap proxy):
       use raw 5% put mid vs 5% call mid normalized — here approximate skew = (P5%/C5%)-1 in IV terms."""
    Kp = min(ks, key=lambda k: abs(k - S * 0.95))
    Kc = min(ks, key=lambda k: abs(k - S * 1.05))
    pm = leg_mid(cur, exp, Kp, 'P', d, ENTRY_T)
    cm = leg_mid(cur, exp, Kc, 'C', d, ENTRY_T)
    if not pm or not cm:
        return None
    T = dte / 365
    ivp = pm / (0.40 * S * math.sqrt(T)) if T > 0 else None       # rough OTM-IV proxy
    ivc = cm / (0.40 * S * math.sqrt(T)) if T > 0 else None
    return (ivp - ivc) if (ivp and ivc) else None


def intraday_rv(cur, d):
    """Realized vol of SPY minute 09:30→14:00 (annualized) from IBKR etf_spot_minute."""
    cur.execute("""SELECT close FROM etf_spot_minute WHERE symbol='SPY'
        AND bar_time::date=%s AND bar_time::time BETWEEN '09:30' AND '14:00'
        ORDER BY bar_time""", (d,))
    px = [float(r[0]) for r in cur.fetchall()]
    if len(px) < 30:
        return None
    r = np.diff(np.log(px))
    return float(np.std(r) * math.sqrt(252 * 390))


def main():
    spv = pd.read_csv(SPY_CSV, index_col=0, parse_dates=True)
    spv.index = spv.index.normalize()
    spv = spv[(spv.index >= '2018-01-01')]
    vix = spv['VIX']; v10 = vix.rolling(10).std()
    spy = spv['SPY']
    rv20 = (np.log(spy / spy.shift(1)).rolling(20).std() * math.sqrt(252))
    idx = list(spv.index)
    pos = {d: i for i, d in enumerate(idx)}
    entries = [d for d in idx if d.weekday() in (0, 2, 4)]

    cur = _conn().cursor()
    rows = []
    for d_ts in entries:
        d = d_ts.date()
        S = float(spy.loc[d_ts])
        pe = pick_entry(cur, d, S, 38, 52)
        if not pe:
            continue
        exp, K, dte = pe
        net = minute_net(cur, exp, K, d)
        if not net or len(net) < 60:
            continue
        entry = fill_combo_mid(net, 'buy', start=ENTRY_T, work=15)
        if not entry or entry <= 0:
            continue
        path = combo_mid_path(net)
        after = [m for t, m in path if t >= ENTRY_T]
        if not after:
            continue
        r0 = after[-1] / entry - 1                       # intraday → close
        r0max = max(after) / entry - 1                   # intraday oracle pop
        rec = {'date': d, 'r0': r0, 'r0max': r0max}
        for n in HOR:
            i = pos[d_ts] + n
            if i >= len(idx):
                continue
            m = close_mid(cur, exp, K, idx[i].date())
            if m:
                rec[f'r{n}'] = m / entry - 1
        # signals
        T = dte / 365
        iv = atm_iv(entry, S, T)
        rec['iv_rv'] = (iv - float(rv20.loc[d_ts])) if (iv and not math.isnan(rv20.loc[d_ts])) else None
        rec['vix'] = float(vix.loc[d_ts]); rec['vix_std10'] = float(v10.loc[d_ts])
        # overnight gap = today open vs prev close (etf_spot_minute)
        cur.execute("""SELECT close FROM etf_spot_minute WHERE symbol='SPY'
            AND bar_time::date=%s AND bar_time::time>='09:30' ORDER BY bar_time LIMIT 1""", (d,))
        op = cur.fetchone()
        pi = pos[d_ts] - 1
        rec['gap'] = abs(float(op[0]) / float(spy.iloc[pi]) - 1) if (op and pi >= 0) else None
        rec['intra_rv'] = intraday_rv(cur, d)
        cur.execute("SELECT DISTINCT strike FROM spy_options_history WHERE trade_date=%s AND expiration=%s", (d, exp))
        ks = [float(r[0]) for r in cur.fetchall()]
        rec['skew'] = wing_skew(cur, exp, d, S, dte, ks) if ks else None
        rec['dow'] = d_ts.weekday()
        rows.append(rec)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(THIS, 'spy_intraday_vs_multiday.csv'), index=False)
    n = len(df)
    print(f"=== SPY INTRADAY vs MULTI-DAY  (n={n} entries, 14:00 combo-mid, mid-to-mid) ===\n")

    def stat(col):
        x = df[col].dropna().values
        if len(x) == 0:
            return None
        sh = x.mean() / x.std() * math.sqrt(len(x)) if x.std() > 0 else 0
        return len(x), x.mean(), (x > 0).mean(), sh

    print(f"{'horizon':<14}{'n':>5}{'avg ret':>9}{'win%':>7}{'t-stat':>8}")
    print('-' * 44)
    for col, lab in [('r0', 'intraday→close'), ('r0max', 'intraday-MAX*'),
                     ('r1', '+1 day'), ('r2', '+2 day'), ('r3', '+3 day'),
                     ('r5', '+5 day'), ('r10', '+10 day')]:
        s = stat(col)
        if s:
            print(f"{lab:<14}{s[0]:>5}{s[1]:>+9.2%}{s[2]:>6.0%}{s[3]:>8.2f}")
    print("  (*intraday-MAX = exit at the best post-14:00 mid same day — ceiling for an intraday pop)")

    # which signal lifts the +3d frontier?  (3d = representative multi-day)
    print("\n=== SIGNAL LIFT on +3d return (top-tercile vs bottom-tercile) ===")
    print(f"{'signal':<12}{'lowT ret':>10}{'highT ret':>11}{'spread':>9}{'hi-win':>8}")
    print('-' * 50)
    base = df['r3'].dropna()
    for sig, hi_is_good in [('iv_rv', False), ('skew', False), ('vix_std10', False),
                            ('gap', None), ('intra_rv', None), ('vix', None)]:
        sub = df[[sig, 'r3']].dropna()
        if len(sub) < 30:
            continue
        lo, hi = sub[sig].quantile(1/3), sub[sig].quantile(2/3)
        loT = sub[sub[sig] <= lo]['r3']; hiT = sub[sub[sig] >= hi]['r3']
        print(f"{sig:<12}{loT.mean():>+10.2%}{hiT.mean():>+11.2%}{(hiT.mean()-loT.mean()):>+9.2%}{(hiT>0).mean():>7.0%}")
    print(f"\n  baseline +3d: {base.mean():+.2%}  win {(base>0).mean():.0%}  (n={len(base)})")
    print("DONE", flush=True)


if __name__ == '__main__':
    main()
