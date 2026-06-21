"""LIVE vega-scrape opportunity scanner — SPY/QQQ/IWM, latest data.

Checks the conditions the study found predictive and prints a GREEN/RED verdict per name:
  • VIX low-ish + CONSOLIDATED (10d-std small)      — wait for vol to settle, don't buy it falling
  • IV < RV (cheap vs realized; vol-risk-premium)    — the cleanest edge
  • FLAT skew (5% put−call IV)                        — complacency = cheap wings
Recipe when GREEN: long ~45 DTE ATM straddle, work a combo-limit at mid in the AFTERNOON (~14:00),
harvest the vol-pop. Also prints the most recent historical GOOD dates as the track record.

  venv/bin/python research/spy_vol/opportunity_scan.py
"""
import os
import sys
import numpy as np
import pandas as pd
import psycopg2

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS, '..', '..', 'backtest'))
from fetch_thetadata_iv import bs_implied_vol

DB = {'host': '192.168.1.172', 'port': 5432, 'database': 'market_scanner',
      'user': 'postgres', 'password': 'shinobi2025'}
DAILY = os.path.join(THIS, 'cache', 'etf_vix_daily.csv')
R = 0.045


def latest_iv_skew(cur, table, spot):
    """On the table's latest date, ATM IV + 5% skew (put_iv − call_iv) at the ~45 DTE expiry."""
    cur.execute(f"SELECT max(trade_date) FROM {table}")
    d = cur.fetchone()[0]
    if not d:
        return None
    cur.execute(f"""SELECT DISTINCT expiration FROM {table} WHERE trade_date=%s
        AND expiration BETWEEN %s AND %s ORDER BY expiration""",
                (d, d + pd.Timedelta(days=30), d + pd.Timedelta(days=72)))
    exps = [r[0] for r in cur.fetchall()]
    if not exps:
        return None
    exp = exps[0]; dte = (exp - d).days

    def iv(K, right):
        cur.execute(f"""SELECT DISTINCT ON (data_type) data_type, close FROM {table}
            WHERE trade_date=%s AND expiration=%s AND strike=%s AND option_right=%s
            ORDER BY data_type, bar_time DESC""", (d, exp, K, right))
        m = {dt: float(c) for dt, c in cur.fetchall()}
        if 'BID' not in m or 'ASK' not in m or m['ASK'] <= m['BID']:
            return None
        return bs_implied_vol((m['BID'] + m['ASK']) / 2, spot, K, dte / 365, R, right)

    cur.execute(f"SELECT DISTINCT strike FROM {table} WHERE trade_date=%s AND expiration=%s", (d, exp))
    ks = [float(r[0]) for r in cur.fetchall()]
    if not ks:
        return None
    Katm = min(ks, key=lambda k: abs(k - spot))
    Kp = min(ks, key=lambda k: abs(k - spot * 0.95)); Kc = min(ks, key=lambda k: abs(k - spot * 1.05))
    iva = iv(Katm, 'C') or iv(Katm, 'P'); ivp = iv(Kp, 'P'); ivc = iv(Kc, 'C')
    return {'date': d, 'dte': dte, 'iv_atm': iva,
            'skew': (ivp - ivc) if (ivp and ivc) else None}


def main():
    daily = pd.read_csv(DAILY, index_col=0, parse_dates=True)
    vix = daily['VIX']; v10 = vix.rolling(10).std()
    vpct = vix.rolling(252).apply(lambda w: (w.iloc[-1] >= w).mean())
    conn = psycopg2.connect(**DB); cur = conn.cursor()
    print(f"=== VEGA-SCRAPE OPPORTUNITY SCAN  (VIX {vix.iloc[-1]:.1f}, {daily.index[-1].date()}) ===\n")
    for sym in ('SPY', 'QQQ', 'IWM'):
        table = f'{sym.lower()}_options_history'
        cur.execute("SELECT to_regclass(%s)", (table,))
        if cur.fetchone()[0] is None:
            continue
        s = daily[sym].dropna()
        rv20 = float((np.log(s / s.shift(1)).rolling(20).std() * np.sqrt(252)).iloc[-1])
        spot = float(s.iloc[-1])
        info = latest_iv_skew(cur, table, spot)
        vlast = float(vix.iloc[-1]); vstd = float(v10.iloc[-1]); vpc = float(vpct.iloc[-1])
        print(f"  {sym}  (spot {spot:.2f}, options asof {info['date'] if info else 'n/a'})")
        # criteria
        cons = vstd < 1.5
        print(f"    VIX {vlast:.1f} (pctile {vpc:.0%}) | 10d-std {vstd:.2f} → consolidated: {'✓' if cons else '✗'}")
        if info and info['iv_atm']:
            gap = info['iv_atm'] - rv20
            print(f"    ATM IV {info['iv_atm']:.0%} vs RV20 {rv20:.0%} → IV−RV {gap:+.0%}  cheap(IV<RV): {'✓' if gap < 0 else '✗'}")
        else:
            gap = None; print(f"    IV: n/a (no fresh option data)")
        if info and info['skew'] is not None:
            print(f"    5% skew {info['skew']:.3f} → flat(<0.08): {'✓' if info['skew'] < 0.08 else '✗'}")
        flat = bool(info and info.get('skew') is not None and info['skew'] < 0.08)
        greens = int(cons) + int(gap is not None and gap < 0) + int(flat)
        verdict = 'SETUP (long 45d straddle, afternoon combo-mid)' if greens >= 3 else \
                  ('WATCH (partial)' if greens == 2 else 'NO SETUP (vol not cheap/settled enough)')
        print(f"    → {greens}/3 green: {verdict}\n")
    conn.close()
    print("Recipe when GREEN: long ~45 DTE ATM straddle (≥42 DTE), combo-limit at mid ~14:00 ET, "
          "exit on the vol-pop (+30% / VIX+3). Rare: ~1-2/yr/name; trade SPY+QQQ (low corr).")


if __name__ == '__main__':
    main()
