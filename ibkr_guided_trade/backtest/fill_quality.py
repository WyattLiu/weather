"""Empirical MID-FILL model — calibrated from past experience, not a hand formula.

Ground truth comes from the actual minute path: for each historical contract-day we
post a passive limit at the first-window MID and observe whether the real bid/ask
later trades THROUGH it (a buyer lifts your offer / a seller hits your bid). That
binary outcome is the label. We then fit P(mid-fill) as a function of exactly the
liquidity features that drive it:

    • spread in PENNIES (ask-bid in cents) — penny-wide markets fill at mid; dime-wide make you cross
    • OPEN INTEREST (ung_options_oi)        — deep books fill at mid; thin ones don't
    • DTE                                    — front-month is more liquid

Output: a calibration table P(mid | spread_cents bucket × OI bucket) + a lookup the
advisor/live path uses to quote realistic fill odds. The BACKTEST still uses the
verifiable path-replay (intraday_fill.execute_audit); this model is for forward
decisions where we don't yet have the path.
"""
import os
import json
import psycopg2

DB = {'host': '192.168.1.172', 'port': 5432, 'database': 'market_scanner',
      'user': 'postgres', 'password': 'shinobi2025'}
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
OUT = os.path.join(CACHE, 'fill_quality_calibration.json')

# Per contract-day: first-window mid, whether the path traded through it (both sides),
# median spread in cents, median spread %, DTE, joined to daily OI. RTH ex-print window.
QUERY = """
WITH bars AS (
  SELECT trade_date, expiration, strike, option_right, bar_time,
         MAX(CASE WHEN data_type='BID' THEN close END) bid,
         MAX(CASE WHEN data_type='ASK' THEN close END) ask
  FROM ung_options_history
  WHERE bar_time::time >= '11:00' AND bar_time::time <= '16:00'
  GROUP BY 1,2,3,4,5),
two AS (SELECT * FROM bars WHERE bid > 0 AND ask > bid),
firstbar AS (
  SELECT DISTINCT ON (trade_date,expiration,strike,option_right)
         trade_date,expiration,strike,option_right, (bid+ask)/2.0 AS entry_mid
  FROM two ORDER BY trade_date,expiration,strike,option_right, bar_time),
agg AS (
  SELECT t.trade_date,t.expiration,t.strike,t.option_right, f.entry_mid,
         MAX(t.bid) maxbid, MIN(t.ask) minask,
         percentile_cont(0.5) WITHIN GROUP (ORDER BY (t.ask-t.bid)) spread_c,
         percentile_cont(0.5) WITHIN GROUP (ORDER BY (t.ask-t.bid)/((t.ask+t.bid)/2.0)) spread_p
  FROM two t JOIN firstbar f USING (trade_date,expiration,strike,option_right)
  GROUP BY 1,2,3,4,5)
SELECT (a.expiration - a.trade_date) dte,
       a.spread_c, a.spread_p,
       (a.maxbid >= a.entry_mid) OR (a.minask <= a.entry_mid) AS mid_fill,
       o.open_interest
FROM agg a
LEFT JOIN ung_options_oi o USING (trade_date,expiration,strike,option_right)
WHERE (a.expiration - a.trade_date) BETWEEN 0 AND 60
"""

SPREAD_BUCKETS = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 10), (10, 1e9)]   # cents
OI_BUCKETS = [(0, 100), (100, 500), (500, 2000), (2000, 8000), (8000, 1e9)]


def _lbl(buckets, v):
    for lo, hi in buckets:
        if lo <= v < hi:
            return f"{lo}-{'+' if hi > 1e8 else int(hi)}"
    return "na"


def build(refresh=True):
    conn = psycopg2.connect(**DB); cur = conn.cursor()
    print("aggregating minute path × OI (server-side, may take a few min)...", flush=True)
    cur.execute(QUERY)
    rows = cur.fetchall(); conn.close()
    print(f"  {len(rows):,} contract-days", flush=True)
    # cents are in RAW scale; convert raw spread to cents (×100) — raw IS real market $.
    grid = {}
    overall = {'n': 0, 'fills': 0}
    feats = []
    for dte, sc, sp, mid_fill, oi in rows:
        if sc is None:
            continue
        cents = float(sc) * 100.0
        oiv = int(oi) if oi is not None else 0
        sb, ob = _lbl(SPREAD_BUCKETS, cents), _lbl(OI_BUCKETS, oiv)
        k = f"{sb}|{ob}"
        g = grid.setdefault(k, {'n': 0, 'fills': 0, 'spread_c': sb, 'oi': ob})
        g['n'] += 1; g['fills'] += int(bool(mid_fill))
        overall['n'] += 1; overall['fills'] += int(bool(mid_fill))
        feats.append((cents, oiv, float(dte), int(bool(mid_fill))))
    for g in grid.values():
        g['p_mid'] = round(g['fills'] / g['n'], 3) if g['n'] else None
    cal = {'overall_p_mid': round(overall['fills'] / max(1, overall['n']), 3),
           'n': overall['n'], 'grid': grid,
           'spread_buckets_cents': SPREAD_BUCKETS, 'oi_buckets': OI_BUCKETS}
    if refresh:
        json.dump(cal, open(OUT, 'w'), indent=1, default=str)
    return cal, feats


def report(cal):
    print(f"\nMID-FILL CALIBRATION (overall P(mid)={cal['overall_p_mid']}, n={cal['n']:,})")
    print("rows = spread in CENTS, cols = OPEN INTEREST  → P(mid-fill)\n")
    sbs = [f"{lo}-{'+' if hi > 1e8 else int(hi)}" for lo, hi in SPREAD_BUCKETS]
    obs = [f"{lo}-{'+' if hi > 1e8 else int(hi)}" for lo, hi in OI_BUCKETS]
    print(f"{'spread¢ \\ OI':>14} " + " ".join(f"{o:>10}" for o in obs))
    for sb in sbs:
        cells = []
        for ob in obs:
            g = cal['grid'].get(f"{sb}|{ob}")
            cells.append(f"{g['p_mid']:>10.2f}" if g and g['p_mid'] is not None else f"{'·':>10}")
        print(f"{sb:>14} " + " ".join(cells))
    print("\n(penny-wide + deep OI → top-left-ish high P(mid); wide + thin → low)")


if __name__ == '__main__':
    cal, _ = build()
    report(cal)
