"""DoW / time-of-day EXECUTION study (descriptive, read-only, no-leak).

Question: is there a stable intraday / day-of-week structure in UNG option spreads (and premium richness)
we can exploit to fill sell-to-open and buy-to-close better? Framing per operator:
  - The robust axis is the CONTINUOUS intraday curve (minute-of-day), not 5 hardcoded weekday dummies.
  - Thursday is just the EIA print (well-known); model it as an EVENT (minutes-from-10:30 ET), not a
    calendar flag. Then test whether DoW BEYOND the print is just noise (expected: yes).

Instruments = the wheel's actual fills: near-ATM (|K-underlying|/underlying < 10%), short-DTE (0-14).
Metric = relative spread (ask-bid)/mid — the execution cost. Measures only; wires nothing.
"""
import sys

import psycopg2

sys.path.insert(0, '.')
from backfill_ung_iv_pg import DB_PARAMS

# rows pivoted to per-contract-per-minute bid/ask, filtered to the wheel's instruments
_CTE = """
WITH q AS (
  SELECT trade_date, bar_time, strike, option_right,
         MAX(CASE WHEN data_type='BID' THEN close END) AS bid,
         MAX(CASE WHEN data_type='ASK' THEN close END) AS ask,
         MAX(underlying_price) AS ul
  FROM ung_options_history
  WHERE expiration - trade_date BETWEEN 0 AND 14
  GROUP BY trade_date, bar_time, expiration, strike, option_right
),
f AS (
  SELECT trade_date, bar_time,
         (ask-bid)/((ask+bid)/2.0) AS rs,
         EXTRACT(dow  FROM trade_date)::int AS dow,
         EXTRACT(hour FROM bar_time)::int*60 + EXTRACT(minute FROM bar_time)::int AS mod
  FROM q
  WHERE bid > 0 AND ask > bid AND ul > 0 AND abs(strike-ul)/ul < 0.10
)
"""

DOW = {0: 'Sun', 1: 'Mon', 2: 'Tue', 3: 'Wed', 4: 'Thu', 5: 'Fri', 6: 'Sat'}


def _bar(v, lo, hi, width=40):
    n = int(round((v - lo) / (hi - lo) * width)) if hi > lo else 0
    return '█' * max(0, min(width, n))


def main():
    c = psycopg2.connect(**DB_PARAMS, connect_timeout=8)
    cur = c.cursor()

    print("\n=== (1) INTRADAY SPREAD CURVE — 30-min buckets, all days pooled ===")
    cur.execute(_CTE + """
      SELECT (mod/30)*30 AS b30, COUNT(*) n, AVG(rs) avg_rs
      FROM f GROUP BY 1 ORDER BY 1""")
    rows = [(int(b), int(n), float(a)) for b, n, a in cur.fetchall() if 9*60 <= b <= 16*60]
    lo = min(a for _, _, a in rows); hi = max(a for _, _, a in rows)
    for b, n, a in rows:
        hh, mm = divmod(b, 60)
        print(f"  {hh:02d}:{mm:02d}  rs={a*100:5.1f}%  n={n:>9,}  {_bar(a, lo, hi)}")

    print("\n=== (2) DoW spread POOLED (confounded by ToD + EIA) ===")
    cur.execute(_CTE + "SELECT dow, COUNT(*) n, AVG(rs) FROM f GROUP BY 1 ORDER BY 1")
    for d, n, a in cur.fetchall():
        if d in (0, 6):
            continue
        print(f"  {DOW[d]}  rs={float(a)*100:5.1f}%  n={int(n):>9,}")

    print("\n=== (3) DoW spread CONTROLLING ToD (midday only, 12:00-14:00, excludes open/close/print) ===")
    print("    -> if this is flat, DoW-beyond-EIA is NOISE (not a hardcode):")
    cur.execute(_CTE + """
      SELECT dow, COUNT(*) n, AVG(rs)
      FROM f WHERE mod BETWEEN 12*60 AND 14*60 GROUP BY 1 ORDER BY 1""")
    md = [(d, int(n), float(a)) for d, n, a in cur.fetchall() if d not in (0, 6)]
    for d, n, a in md:
        print(f"  {DOW[d]}  rs={a*100:5.1f}%  n={n:>9,}")
    spread_range = max(a for _, _, a in md) - min(a for _, _, a in md)
    print(f"    midday DoW range = {spread_range*100:.2f}pp "
          f"(vs intraday-curve range {(hi-lo)*100:.1f}pp — ToD dominates if this is tiny)")

    print("\n=== (4) EIA PRINT as an EVENT — spread by 15-min bucket, Thursday vs other days ===")
    cur.execute(_CTE + """
      SELECT (mod/15)*15 AS b15, dow=4 AS is_thu, COUNT(*) n, AVG(rs)
      FROM f WHERE mod BETWEEN 9*60+30 AND 12*60 GROUP BY 1,2 ORDER BY 1,2""")
    agg = {}
    for b, thu, n, a in cur.fetchall():
        agg.setdefault(int(b), {})[bool(thu)] = (int(n), float(a))
    print("   time    Thu-spread   other-spread   Thu/other")
    for b in sorted(agg):
        hh, mm = divmod(b, 60)
        t = agg[b].get(True); o = agg[b].get(False)
        if t and o:
            ratio = t[1] / o[1] if o[1] else 0
            flag = '  <-- print' if 10*60 <= b <= 10*60+30 else ''
            print(f"  {hh:02d}:{mm:02d}    {t[1]*100:5.1f}%       {o[1]*100:5.1f}%        {ratio:.2f}x{flag}")

    c.close()
    print("\nDONE", flush=True)


if __name__ == '__main__':
    main()
