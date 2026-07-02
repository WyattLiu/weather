"""HONEST passive-limit capture study (read-only, real tape, adverse-selection-aware).

Question: does resting a passive limit at the mid produce NET price improvement vs just crossing —
AFTER paying the chase cost on the misses (adverse selection)? Naive backtests count only the mid-fills
("free half-spread") and ignore that the unfilled cases are the ones where the market moved against you.

Method — replay the ACTUAL minute bid/ask path of the wheel's instruments (near-ATM, short-DTE):
  For a decision at minute t0, side S:
    baseline  = cross NOW at t0's touch (buy→ask_t0, sell→bid_t0)
    passive   = rest a limit at mid_t0 for `patience` minutes:
                  filled if the tape trades through mid (buy: ask_t<=mid_t0; sell: bid_t>=mid_t0) → mid_t0
                  else CROSS at t_end's touch (the chase — may be worse than t0 if it moved against you)
    improvement = baseline_cost - passive_cost   (>0 = passive better; captures adverse selection honestly)
  Aggregated by time-of-day bucket and side. Improvement reported as % of option price (scale-free) so the
  UNG reverse split is irrelevant. Measures only; wires nothing.
"""
import sys

import psycopg2

sys.path.insert(0, '.')
from backfill_ung_iv_pg import DB_PARAMS

PATIENCE_MIN = 30          # how long we rest the passive limit before crossing
DAY_STRIDE = 2             # sample every Nth trading day (speed)


def _minbars(cur, d, exp, K, right):
    cur.execute("""
      SELECT bar_time,
             MAX(CASE WHEN data_type='BID' THEN close END) bid,
             MAX(CASE WHEN data_type='ASK' THEN close END) ask
      FROM ung_options_history
      WHERE trade_date=%s AND expiration=%s AND strike=%s AND option_right=%s
      GROUP BY bar_time ORDER BY bar_time""", (d, exp, K, right))
    return [(t, float(b), float(a)) for t, b, a in cur.fetchall()
            if b is not None and a is not None and a > b and b > 0]


def _simulate(bars, side):
    """Return list of (tod_bucket, improvement_pct) over 30-min decision starts across the day."""
    out = []
    n = len(bars)
    for i in range(n):
        t0, b0, a0 = bars[i]
        mod0 = t0.hour * 60 + t0.minute
        if mod0 % 30 != 0 or mod0 < 9 * 60 + 30 or mod0 > 15 * 60 + 30:
            continue                                   # decision starts on 30-min grid, RTH
        mid0 = (a0 + b0) / 2.0
        baseline = a0 if side == 'buy' else b0         # cross now at the touch
        # rest at mid0 for PATIENCE_MIN
        filled = None
        j = i
        while j < n and (bars[j][0].hour * 60 + bars[j][0].minute) - mod0 <= PATIENCE_MIN:
            _, bj, aj = bars[j]
            if side == 'buy' and aj <= mid0:
                filled = mid0; break
            if side == 'sell' and bj >= mid0:
                filled = mid0; break
            j += 1
        if filled is None:                             # chase: cross at the window-end touch
            _, be, ae = bars[min(j, n - 1)]
            passive = ae if side == 'buy' else be
        else:
            passive = filled
        # improvement: for a buy lower cost is better; for a sell higher proceeds is better
        if side == 'buy':
            impr = baseline - passive
        else:
            impr = passive - baseline
        out.append((mod0, impr / mid0 * 100.0, filled is not None))
    return out


def main():
    c = psycopg2.connect(**DB_PARAMS, connect_timeout=8)
    cur = c.cursor()
    cur.execute("SELECT DISTINCT trade_date FROM ung_options_history ORDER BY trade_date")
    days = [r[0] for r in cur.fetchall()][::DAY_STRIDE]
    print(f"sampling {len(days)} days (stride {DAY_STRIDE}), patience {PATIENCE_MIN}min\n")

    from collections import defaultdict
    agg = defaultdict(lambda: {'buy': [], 'sell': []})      # bucket -> side -> [impr%]
    fills = defaultdict(lambda: {'buy': [0, 0], 'sell': [0, 0]})  # bucket -> side -> [filled, total]
    for d in days:
        cur.execute("""SELECT MIN(expiration) FROM ung_options_history
                       WHERE trade_date=%s AND expiration>trade_date
                       AND expiration-trade_date BETWEEN 1 AND 14""", (d,))
        r = cur.fetchone()
        if not r or r[0] is None:
            continue
        exp = r[0]
        cur.execute("""SELECT strike, underlying_price FROM ung_options_history
                       WHERE trade_date=%s AND expiration=%s ORDER BY abs(strike-underlying_price) LIMIT 1""",
                    (d, exp))
        rr = cur.fetchone()
        if not rr:
            continue
        K = float(rr[0])
        for right in ('P', 'C'):
            bars = _minbars(cur, d, exp, K, right)
            if len(bars) < 20:
                continue
            for side in ('buy', 'sell'):
                for mod0, impr, ok in _simulate(bars, side):
                    agg[mod0][side].append(impr)
                    fills[mod0][side][1] += 1
                    fills[mod0][side][0] += 1 if ok else 0
    c.close()

    def stat(xs):
        return (sum(xs) / len(xs)) if xs else 0.0

    for side in ('sell', 'buy'):
        print(f"=== SIDE={side.upper()}  (net improvement % of option price, + = passive beats crossing) ===")
        print("  time    net_impr%   pass_fill%   n")
        allimp = []
        for b in sorted(agg):
            xs = agg[b][side]
            if not xs:
                continue
            allimp += xs
            f, t = fills[b][side]
            hh, mm = divmod(b, 60)
            print(f"  {hh:02d}:{mm:02d}   {stat(xs):+6.2f}%     {100*f/max(1,t):5.1f}%   {t:>6,}")
        print(f"  ---- ALL-DAY mean net improvement: {stat(allimp):+.2f}% of option price "
              f"(n={len(allimp):,}) ----\n")
    print("DONE", flush=True)


if __name__ == '__main__':
    main()
