"""Intraday UNG underlying reconstructed via PUT-CALL PARITY from the option minute path.

ThetaData stock bars are a paid tier; we don't need them. For an ATM strike we already
have minute call & put mids, so:  S(t) = C(t) - P(t) + K · e^(-rT)
gives the implied underlying every minute — same data the fills use, free, full history.
American-premium bias is tiny for short-DTE ATM and cancels in DIRECTION, which is all the
spike study needs. Validated against the daily close.
"""
import os
import math
import psycopg2
import pandas as pd

DB = {'host': '192.168.1.172', 'port': 5432, 'database': 'market_scanner',
      'user': 'postgres', 'password': 'shinobi2025'}
_SPLITS = [('2018-01-05', 4.0), ('2024-01-24', 4.0)]
R = 0.045


def _sf(d):
    sf = 1.0
    for sd, f in _SPLITS:
        if pd.Timestamp(d) < pd.Timestamp(sd):
            sf = f
    return sf


def intraday_underlying(date, spot_adj, conn=None):
    """Minute implied-underlying path (adjusted $) for `date`. spot_adj = daily UNG close
    (adjusted) used to pick the ATM strike + front expiry. Returns DataFrame[bar_time, S]."""
    own = conn is None
    conn = conn or psycopg2.connect(**DB)
    cur = conn.cursor()
    ds = pd.Timestamp(date).date().isoformat()
    sf = _sf(date)
    spot_raw = spot_adj / sf
    # front-ish expiry (~20-45 DTE) that has near-money two-sided quotes
    cur.execute("""SELECT expiration FROM ung_options_history
                   WHERE trade_date=%s GROUP BY expiration
                   HAVING (expiration - %s) BETWEEN 15 AND 50
                   ORDER BY abs((expiration - %s) - 30) LIMIT 1""", (ds, ds, ds))
    r = cur.fetchone()
    if not r:
        if own: conn.close()
        return None
    exp = r[0]
    T = max((exp - pd.Timestamp(date).date()).days, 1) / 365.0
    disc = math.exp(-R * T)
    # ATM strike (raw) with both P and C present
    cur.execute("""SELECT strike, count(DISTINCT option_right) FROM ung_options_history
                   WHERE trade_date=%s AND expiration=%s GROUP BY strike
                   HAVING count(DISTINCT option_right)=2
                   ORDER BY abs(strike-%s) LIMIT 1""", (ds, exp.isoformat(), spot_raw))
    r = cur.fetchone()
    if not r:
        if own: conn.close()
        return None
    K = float(r[0])
    # minute call & put mids → implied S
    cur.execute("""
        WITH q AS (
          SELECT bar_time, option_right,
                 (MAX(CASE WHEN data_type='BID' THEN close END)
                + MAX(CASE WHEN data_type='ASK' THEN close END))/2.0 mid
          FROM ung_options_history
          WHERE trade_date=%s AND expiration=%s AND ABS(strike-%s)<0.001
            AND bar_time::time>='09:30' AND bar_time::time<='16:00'
          GROUP BY bar_time, option_right)
        SELECT bar_time,
               MAX(CASE WHEN option_right='C' THEN mid END) c,
               MAX(CASE WHEN option_right='P' THEN mid END) p
        FROM q GROUP BY bar_time ORDER BY bar_time""", (ds, exp.isoformat(), K))
    rows = cur.fetchall()
    if own: conn.close()
    out = []
    for t, c, p in rows:
        if c is None or p is None:
            continue
        S = (float(c) - float(p) + K * disc) * sf      # back to adjusted $
        out.append((t, S))
    if not out:
        return None
    return pd.DataFrame(out, columns=['bar_time', 'S'])


def atm_frame(date, spot_adj, conn=None):
    """Minute frame for the ATM strike: bar_time, S (implied underlying), call_mid,
    put_mid, call_spr_pct, put_spr_pct — all adjusted $. For the spike-leg study."""
    own = conn is None
    conn = conn or psycopg2.connect(**DB)
    cur = conn.cursor()
    ds = pd.Timestamp(date).date().isoformat()
    sf = _sf(date)
    spot_raw = spot_adj / sf
    cur.execute("""SELECT expiration FROM ung_options_history
                   WHERE trade_date=%s GROUP BY expiration
                   HAVING (expiration - %s) BETWEEN 15 AND 50
                   ORDER BY abs((expiration - %s) - 30) LIMIT 1""", (ds, ds, ds))
    r = cur.fetchone()
    if not r:
        if own: conn.close()
        return None
    exp = r[0]
    T = max((exp - pd.Timestamp(date).date()).days, 1) / 365.0
    disc = math.exp(-R * T)
    cur.execute("""SELECT strike FROM ung_options_history
                   WHERE trade_date=%s AND expiration=%s GROUP BY strike
                   HAVING count(DISTINCT option_right)=2 ORDER BY abs(strike-%s) LIMIT 1""",
                (ds, exp.isoformat(), spot_raw))
    r = cur.fetchone()
    if not r:
        if own: conn.close()
        return None
    K = float(r[0])
    cur.execute("""
        WITH q AS (
          SELECT bar_time, option_right,
                 MAX(CASE WHEN data_type='BID' THEN close END) bid,
                 MAX(CASE WHEN data_type='ASK' THEN close END) ask
          FROM ung_options_history
          WHERE trade_date=%s AND expiration=%s AND ABS(strike-%s)<0.001
            AND bar_time::time>='09:30' AND bar_time::time<='16:00'
          GROUP BY bar_time, option_right)
        SELECT bar_time,
          MAX(CASE WHEN option_right='C' THEN (bid+ask)/2 END) cmid,
          MAX(CASE WHEN option_right='C' THEN (ask-bid)/NULLIF((bid+ask)/2,0)*100 END) cspr,
          MAX(CASE WHEN option_right='P' THEN (bid+ask)/2 END) pmid,
          MAX(CASE WHEN option_right='P' THEN (ask-bid)/NULLIF((bid+ask)/2,0)*100 END) pspr
        FROM q WHERE bid>0 AND ask>bid GROUP BY bar_time ORDER BY bar_time""",
        (ds, exp.isoformat(), K))
    rows = cur.fetchall()
    if own: conn.close()
    out = []
    for t, cm, cs, pm, ps in rows:
        if cm is None or pm is None:
            continue
        out.append((t, (float(cm) - float(pm) + K * disc) * sf, float(cm) * sf,
                    float(pm) * sf, float(cs or 0), float(ps or 0)))
    if not out:
        return None
    return pd.DataFrame(out, columns=['bar_time', 'S', 'call_mid', 'put_mid',
                                      'call_spr', 'put_spr'])


def day_spike(date, spot_adj, conn=None):
    """Intraday spike summary for a day: open→close move, intraday range, and the
    signed move from the session open to its most extreme point (the 'spike')."""
    df = intraday_underlying(date, spot_adj, conn=conn)
    if df is None or len(df) < 10:
        return None
    s = df['S'].values
    o, c = s[0], s[-1]
    hi, lo = s.max(), s.min()
    oc = (c / o - 1) * 100
    up = (hi / o - 1) * 100
    dn = (lo / o - 1) * 100
    spike = up if abs(up) >= abs(dn) else dn          # dominant intraday excursion
    return {'open': round(o, 3), 'close': round(c, 3), 'oc_pct': round(oc, 2),
            'up_pct': round(up, 2), 'dn_pct': round(dn, 2), 'spike_pct': round(spike, 2),
            'range_pct': round((hi - lo) / o * 100, 2), 'n': len(s)}


if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import replay_engine as Rg
    u = pd.read_csv(os.path.join(Rg.CACHE_DIR, 'master_dataset.csv'),
                    index_col=0, parse_dates=True)['UNG'].dropna()
    u.index = u.index.normalize()
    conn = psycopg2.connect(**DB)
    print("VALIDATION — reconstructed EOD vs daily close:")
    for ds in ['2024-06-12', '2025-01-15', '2025-11-20', '2026-06-12', '2023-03-15']:
        d = pd.Timestamp(ds)
        if d not in u.index:
            continue
        df = intraday_underlying(ds, float(u.loc[d]), conn=conn)
        if df is None:
            print(f"  {ds}: no data"); continue
        print(f"  {ds}: recon close ${df['S'].iloc[-1]:.2f}  vs daily ${u.loc[d]:.2f}  "
              f"(open ${df['S'].iloc[0]:.2f}, n={len(df)})")
    conn.close()
