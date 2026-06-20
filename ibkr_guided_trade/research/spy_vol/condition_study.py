"""WHEN does the long-vega scrape work? Run it ALL the time across SPY/QQQ/IWM (2018-2026),
record entry CONDITIONS per trade, and rank which conditions predict positive returns.

Features per entry (45 DTE ATM straddle, exit first-of +30%/-40%/VIX+3/30d):
  vix          — VIX level
  vix_pct      — VIX percentile (trailing 252d)
  vix_chg10    — VIX 10d change (falling vs rising)
  vix_std10    — VIX 10d std (consolidation)
  rv20         — underlying 20d realized vol (annualized)
  iv_atm       — entry ATM implied vol (BS-inverted from the straddle)
  iv_minus_rv  — vol-risk-premium proxy (IV cheap vs realized when negative)
  skew         — 5%-OTM put IV − 5%-OTM call IV (the 'shape')

  venv/bin/python research/spy_vol/condition_study.py
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


def eod(cur, table, exp, K, right, d0, d1):
    cur.execute(f"""SELECT DISTINCT ON (trade_date,data_type) trade_date,data_type,close
        FROM {table} WHERE expiration=%s AND strike=%s AND option_right=%s
        AND trade_date BETWEEN %s AND %s ORDER BY trade_date,data_type,bar_time DESC""",
                (exp, K, right, d0, d1))
    bid, ask = {}, {}
    for td, dt, c in cur.fetchall():
        (bid if dt == 'BID' else ask)[td] = float(c)
    return {td: (bid[td] + ask[td]) / 2 for td in set(bid) & set(ask) if ask[td] > bid[td] > 0}


def entry_chain(cur, table, d, spot, lo, hi):
    cur.execute(f"""SELECT DISTINCT expiration FROM {table} WHERE trade_date=%s
        AND expiration BETWEEN %s AND %s ORDER BY expiration""",
                (d, d + pd.Timedelta(days=lo), d + pd.Timedelta(days=hi)))
    exps = [r[0] for r in cur.fetchall()]
    if not exps:
        return None
    exp = exps[0]
    cur.execute(f"""SELECT strike,count(DISTINCT option_right) FROM {table}
        WHERE trade_date=%s AND expiration=%s GROUP BY strike HAVING count(DISTINCT option_right)=2""", (d, exp))
    ks = [float(r[0]) for r in cur.fetchall()]
    if not ks:
        return None
    return exp, sorted(ks), (exp - d).days


def iv_at_strike(cur, table, d, exp, K, right, spot, dte):
    px = eod(cur, table, exp, K, right, d, d).get(d)
    if not px:
        return None
    return bs_implied_vol(px, spot, K, dte / 365, R, right)


def run(symbol, table, daily, vixser, cur):
    spot_s = daily[symbol]
    rv20 = (np.log(spot_s / spot_s.shift(1)).rolling(20).std() * np.sqrt(252))
    vpct = vixser.rolling(252).apply(lambda w: (w.iloc[-1] >= w).mean(), raw=False)
    vchg = vixser - vixser.shift(10)
    vstd = vixser.rolling(10).std()
    vix_path = {pd.Timestamp(k).date(): v for k, v in zip(vixser.index, vixser.values)}
    rows = []
    mondays = daily.index[daily.index.weekday == 0]
    for d_ts in mondays:
        if d_ts not in spot_s.index or np.isnan(spot_s.loc[d_ts]):
            continue
        d = d_ts.date(); spot = float(spot_s.loc[d_ts])
        ch = entry_chain(cur, table, d, spot, 38, 52)
        if not ch:
            continue
        exp, ks, dte = ch
        Katm = min(ks, key=lambda k: abs(k - spot))
        c = eod(cur, table, exp, Katm, 'C', d, exp); p = eod(cur, table, exp, Katm, 'P', d, exp)
        days = sorted(set(c) & set(p))
        if d not in days:
            continue
        entry = c[d] + p[d]
        if entry <= 0:
            continue
        # exit: first-of +30%/-40%/VIX+3/30d (mid path)
        vix0 = float(vixser.loc[d_ts]); ret = None
        for t in [x for x in days if x > d][:30]:
            v = (c[t] + p[t]) / entry - 1
            vt = vix_path.get(t, vix0)
            if v >= 0.30 or v <= -0.40 or vt >= vix0 + 3 or t == [x for x in days if x > d][:30][-1]:
                ret = v; break
        if ret is None:
            continue
        # features
        iv_atm = iv_at_strike(cur, table, d, exp, Katm, 'C', spot, dte)
        Kp = min(ks, key=lambda k: abs(k - spot * 0.95))
        Kc = min(ks, key=lambda k: abs(k - spot * 1.05))
        ivp = iv_at_strike(cur, table, d, exp, Kp, 'P', spot, dte)
        ivc = iv_at_strike(cur, table, d, exp, Kc, 'C', spot, dte)
        rows.append({
            'sym': symbol, 'date': d, 'ret': ret, 'vix': vix0,
            'vix_pct': float(vpct.loc[d_ts]) if d_ts in vpct.index else np.nan,
            'vix_chg10': float(vchg.loc[d_ts]) if d_ts in vchg.index else np.nan,
            'vix_std10': float(vstd.loc[d_ts]) if d_ts in vstd.index else np.nan,
            'rv20': float(rv20.loc[d_ts]) if d_ts in rv20.index else np.nan,
            'iv_atm': iv_atm,
            'iv_minus_rv': (iv_atm - float(rv20.loc[d_ts])) if (iv_atm and d_ts in rv20.index) else np.nan,
            'skew': (ivp - ivc) if (ivp and ivc) else np.nan,
        })
    return rows


def bucketed(df, col, qs=(0, .25, .5, .75, 1.0)):
    s = df[col].dropna()
    if len(s) < 20:
        return
    edges = s.quantile(list(qs)).values
    print(f"\n  {col}: avg straddle return by quartile")
    for i in range(len(edges) - 1):
        m = (df[col] >= edges[i]) & (df[col] <= edges[i + 1] if i == len(edges) - 2 else df[col] < edges[i + 1])
        r = df.loc[m, 'ret']
        if len(r):
            print(f"    {edges[i]:+.2f}..{edges[i+1]:+.2f}  n={len(r):>3}  avg {r.mean():+.1%}  win {(r>0).mean()*100:.0f}%")


def main():
    daily = pd.read_csv(DAILY, index_col=0, parse_dates=True)
    vixser = daily['VIX']
    conn = psycopg2.connect(**DB); cur = conn.cursor()
    all_rows = []
    for sym in ('SPY', 'QQQ', 'IWM'):
        table = f'{sym.lower()}_options_history'
        cur.execute("SELECT to_regclass(%s)", (table,))
        if cur.fetchone()[0] is None:
            print(f"  {table}: not present yet — skipping"); continue
        r = run(sym, table, daily, vixser, cur)
        print(f"  {sym}: {len(r)} entries")
        all_rows += r
    conn.close()
    df = pd.DataFrame(all_rows)
    if df.empty:
        print("no data"); return
    print(f"\n=== CONDITION STUDY — {len(df)} entries (SPY/QQQ/IWM, {df['date'].min()}→{df['date'].max()}) ===")
    print(f"  baseline (all): avg {df['ret'].mean():+.1%}  win {(df['ret']>0).mean()*100:.0f}%")
    for col in ['vix', 'vix_pct', 'iv_minus_rv', 'rv20', 'vix_chg10', 'vix_std10', 'skew', 'iv_atm']:
        bucketed(df, col)
    df.to_csv(os.path.join(THIS, 'condition_study.csv'), index=False)
    print("\nsaved condition_study.csv\nDONE", flush=True)


if __name__ == '__main__':
    main()
