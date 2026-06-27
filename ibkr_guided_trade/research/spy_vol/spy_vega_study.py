"""VEGA-SCRAPING STUDY — buy SPY ATM straddles when VIX is LOW, harvest the IV expansion.

Thesis: when implied vol is cheap (low VIX), a long ATM straddle is cheap vega; if vol mean-
reverts UP (or realized exceeds implied), you scrape the expansion — exiting on the pop, not
holding to decay. Tests whether that actually makes money, under which VIX entry, DTE, and exit.

Verifiable: entry/exit priced from the REAL minute chain (spy_options_history) — entry crosses
to the ASK (you're a buyer), exit crosses to the BID (you're a seller); daily MTM uses EOD mid.
SPY ATM spreads are pennies, so net≈mid, but we cross anyway to be honest.

  venv/bin/python research/spy_vol/spy_vega_study.py
"""
import os
from collections import defaultdict
import numpy as np
import pandas as pd
import psycopg2

DB = {'host': '192.168.1.172', 'port': 5432, 'database': 'market_scanner',
      'user': 'postgres', 'password': 'shinobi2025'}
THIS = os.path.dirname(os.path.abspath(__file__))
SPY_CSV = os.path.join(THIS, 'cache', 'spy_vix_daily.csv')


def _conn():
    return psycopg2.connect(**DB)


def eod_mid(cur, exp, K, right, d0, d1):
    """date -> EOD (last RTH bar) mid for one contract over [d0,d1]. From the minute table."""
    cur.execute("""
        SELECT DISTINCT ON (trade_date, data_type) trade_date, data_type, close
        FROM spy_options_history
        WHERE expiration=%s AND strike=%s AND option_right=%s AND trade_date BETWEEN %s AND %s
        ORDER BY trade_date, data_type, bar_time DESC
    """, (exp, K, right, d0, d1))
    bid, ask = {}, {}
    for td, dt, close in cur.fetchall():
        (bid if dt == 'BID' else ask)[td] = float(close)
    out = {}
    for td in set(bid) & set(ask):
        if ask[td] > bid[td] > 0:
            out[td] = ((bid[td] + ask[td]) / 2, bid[td], ask[td])
    return out


def pick_entry(cur, d, spot, dte_lo, dte_hi):
    """On date d, pick the monthly expiry in [dte_lo,dte_hi] DTE and the ATM strike that has
    BOTH a call and a put quoted. Returns (exp, K, dte) or None."""
    cur.execute("""SELECT DISTINCT expiration FROM spy_options_history
                   WHERE trade_date=%s AND expiration BETWEEN %s AND %s
                   ORDER BY expiration""",
                (d, d + pd.Timedelta(days=dte_lo), d + pd.Timedelta(days=dte_hi)))
    exps = [r[0] for r in cur.fetchall()]
    if not exps:
        return None
    exp = exps[0]                                  # nearest expiry in window (≥dte_lo)
    cur.execute("""SELECT strike, count(DISTINCT option_right) FROM spy_options_history
                   WHERE trade_date=%s AND expiration=%s GROUP BY strike HAVING count(DISTINCT option_right)=2""",
                (d, exp))
    ks = [float(r[0]) for r in cur.fetchall()]
    if not ks:
        return None
    K = min(ks, key=lambda k: abs(k - spot))
    return exp, K, (exp - d).days


def run_trade(cur, d, spot, exp, K, max_hold_days, pt, stop, volpop, vix0, vix_path):
    """Long ATM straddle. Entry crosses to ask, exit crosses to bid. Exit on first of:
    profit-target pt, stop, vol-pop (VIX≥vix0+volpop), or max_hold_days. Returns dict."""
    d1 = exp
    c = eod_mid(cur, exp, K, 'C', d, d1)
    p = eod_mid(cur, exp, K, 'P', d, d1)
    days = sorted(set(c) & set(p))
    if d not in days:
        return None
    entry = c[d][2] + p[d][2]                       # buy at ASK (call ask + put ask)
    if entry <= 0:
        return None
    hold = [x for x in days if x > d][:max_hold_days]
    for i, t in enumerate(hold):
        val_mid = c[t][0] + p[t][0]                 # MTM at mid
        exit_bid = c[t][1] + p[t][1]                # sell at BID
        ret_mid = val_mid / entry - 1
        vix_t = vix_path.get(t, vix0)
        reason = None
        if ret_mid >= pt:
            reason = 'profit'
        elif ret_mid <= -stop:
            reason = 'stop'
        elif volpop and vix_t >= vix0 + volpop:
            reason = 'volpop'
        elif i == len(hold) - 1:
            reason = 'time'
        if reason:
            pnl = exit_bid - entry
            return {'entry': entry, 'exit': exit_bid, 'pnl': pnl, 'ret': pnl / entry,
                    'held': (t - d).days, 'reason': reason, 'vix0': vix0, 'vix_exit': vix_t}
    return None


def main():
    spv = pd.read_csv(SPY_CSV, index_col=0, parse_dates=True)
    spv.index = spv.index.normalize()
    vix_path = {d.date() if hasattr(d, 'date') else d: float(v) for d, v in spv['VIX'].items()}
    # normalize keys to date objects
    vix_path = {pd.Timestamp(k).date(): v for k, v in zip(spv.index, spv['VIX'])}

    DTE_LO, DTE_HI = 38, 52          # ~45 DTE target (start ≥30)
    MAX_HOLD = 30                    # close within ~30 calendar days (≈ 15 DTE left)
    PT, STOP, VOLPOP = 0.30, 0.40, 3.0
    buckets = {'VIX<14 (cheap)': (0, 14), 'VIX 14-16': (14, 16),
               'VIX 16-20': (16, 20), 'VIX>20 (rich)': (20, 99)}
    # one entry per week (Mondays) to avoid overlapping dupes
    entries = spv[spv.index.weekday == 0]
    conn = _conn(); cur = conn.cursor()
    res = defaultdict(list)
    n_try = 0
    for d_ts, row in entries.iterrows():
        d = d_ts.date(); spot = float(row['SPY']); vix0 = float(row['VIX'])
        pe = pick_entry(cur, d, spot, DTE_LO, DTE_HI)
        if not pe:
            continue
        exp, K, dte = pe
        n_try += 1
        tr = run_trade(cur, d, spot, exp, K, MAX_HOLD, PT, STOP, VOLPOP, vix0, vix_path)
        if not tr:
            continue
        for name, (lo, hi) in buckets.items():
            if lo <= vix0 < hi:
                res[name].append(tr); break
    conn.close()

    print(f"=== SPY VEGA-SCRAPING: long ATM straddle, ~45 DTE, exit first-of "
          f"(+{PT:.0%} / -{STOP:.0%} / VIX+{VOLPOP:.0f} / {MAX_HOLD}d) ===")
    print(f"    entries attempted: {n_try}\n")
    hdr = f"{'entry VIX bucket':<18}{'n':>4}{'win%':>7}{'avg ret':>9}{'med ret':>9}{'tot ret':>9}  exit mix"
    print(hdr); print('-' * len(hdr))
    allr = []
    for name in buckets:
        ts = res[name]
        if not ts:
            print(f"{name:<18}{0:>4}"); continue
        rets = np.array([t['ret'] for t in ts]); allr += list(rets)
        win = (rets > 0).mean() * 100
        mix = defaultdict(int)
        for t in ts:
            mix[t['reason']] += 1
        mixs = ' '.join(f"{k}:{v}" for k, v in sorted(mix.items(), key=lambda x: -x[1]))
        print(f"{name:<18}{len(ts):>4}{win:>6.0f}%{rets.mean():>+9.1%}{np.median(rets):>+9.1%}{rets.sum():>+9.1%}  {mixs}")
    if allr:
        a = np.array(allr)
        print(f"\n  ALL: n={len(a)} win {((a>0).mean()*100):.0f}% avg {a.mean():+.1%} "
              f"median {np.median(a):+.1%} | best {a.max():+.0%} worst {a.min():+.0%}")
    print("\nDONE", flush=True)


if __name__ == '__main__':
    main()
