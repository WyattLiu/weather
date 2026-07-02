"""3-WAY execution policy comparison (read-only, real tape). Benchmark = CROSS NOW at t0's touch.

Policies, all measured as IMPROVEMENT vs cross-now (+ = beats crossing immediately), % of arrival mid:
  A cross_now      : pay t0's touch (the benchmark → improvement 0 by definition)
  B naive_rest     : rest a limit at mid0 for `patience`; fill at mid if tape trades through, else CROSS at
                     the WINDOW-END touch (the dumb chase)
  C smart_cross    : don't rest — just CROSS at the TIGHTEST-spread minute in the window (execute_audit core)
  D audit_full     : rest at mid ONLY if arrival spread is WIDE; else cross tightest; on a miss cross tightest
                     (the actual execute_audit policy)

Question: is waiting worth anything, and which policy wins by time-of-day? Same instruments as the
passive study (near-ATM, short-DTE), both sides. Improvement is scale-free (% of mid).
"""
import sys
from collections import defaultdict

import psycopg2

sys.path.insert(0, '.')
from backfill_ung_iv_pg import DB_PARAMS

PATIENCE_MIN = 30
DAY_STRIDE = 2
TIGHT = 0.10          # arrival rel-spread below which D crosses immediately (matches intraday_fill TIGHT_SPREAD idea)


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


def _touch(side, b, a):
    return a if side == 'buy' else b


def _policies(bars, side):
    """Yield (tod_bucket, impr_B, impr_C, impr_D) in % of arrival mid, vs cross-now."""
    n = len(bars)
    for i in range(n):
        t0, b0, a0 = bars[i]
        mod0 = t0.hour * 60 + t0.minute
        if mod0 % 30 != 0 or mod0 < 9 * 60 + 30 or mod0 > 15 * 60 + 30:
            continue
        mid0 = (a0 + b0) / 2.0
        arr_rel = (a0 - b0) / mid0
        cross_now = _touch(side, b0, a0)
        # window
        win = [(t, b, a) for (t, b, a) in bars[i:]
               if (t.hour * 60 + t.minute) - mod0 <= PATIENCE_MIN]
        if not win:
            continue
        # tightest-spread minute in window
        tt, tb, ta = min(win, key=lambda x: (x[2] - x[1]) / ((x[1] + x[2]) / 2))
        tight_cross = _touch(side, tb, ta)
        # passive fill?
        passive_fill = None
        for (t, b, a) in win:
            if side == 'buy' and a <= mid0:
                passive_fill = mid0; break
            if side == 'sell' and b >= mid0:
                passive_fill = mid0; break
        # B naive_rest: mid if filled else cross at window END
        end_cross = _touch(side, win[-1][1], win[-1][2])
        cost_B = passive_fill if passive_fill is not None else end_cross
        # C smart_cross: tightest minute
        cost_C = tight_cross
        # D audit_full: if wide, rest (mid if filled else tightest); if tight, cross tightest
        if arr_rel > TIGHT:
            cost_D = passive_fill if passive_fill is not None else tight_cross
        else:
            cost_D = tight_cross

        def impr(cost):
            # buy: paying less is better; sell: receiving more is better
            return ((cross_now - cost) if side == 'buy' else (cost - cross_now)) / mid0 * 100.0
        yield mod0, impr(cost_B), impr(cost_C), impr(cost_D)


def main():
    c = psycopg2.connect(**DB_PARAMS, connect_timeout=8)
    cur = c.cursor()
    cur.execute("SELECT DISTINCT trade_date FROM ung_options_history ORDER BY trade_date")
    days = [r[0] for r in cur.fetchall()][::DAY_STRIDE]
    print(f"sampling {len(days)} days, patience {PATIENCE_MIN}min, vs benchmark=cross-now\n")

    agg = defaultdict(lambda: defaultdict(lambda: {'B': [], 'C': [], 'D': []}))
    for d in days:
        cur.execute("""SELECT MIN(expiration) FROM ung_options_history
                       WHERE trade_date=%s AND expiration>trade_date
                       AND expiration-trade_date BETWEEN 1 AND 14""", (d,))
        r = cur.fetchone()
        if not r or r[0] is None:
            continue
        exp = r[0]
        cur.execute("""SELECT strike FROM ung_options_history WHERE trade_date=%s AND expiration=%s
                       ORDER BY abs(strike-underlying_price) LIMIT 1""", (d, exp))
        rr = cur.fetchone()
        if not rr:
            continue
        K = float(rr[0])
        for right in ('P', 'C'):
            bars = _minbars(cur, d, exp, K, right)
            if len(bars) < 20:
                continue
            for side in ('buy', 'sell'):
                for mod0, iB, iC, iD in _policies(bars, side):
                    a = agg[side][mod0]
                    a['B'].append(iB); a['C'].append(iC); a['D'].append(iD)
    c.close()

    def m(xs):
        return sum(xs) / len(xs) if xs else 0.0

    for side in ('sell', 'buy'):
        print(f"=== SIDE={side.upper()}  improvement vs CROSS-NOW (%, +=better)  B=naive_rest C=smart_cross D=audit ===")
        print("  time     B_rest   C_cross   D_audit    n")
        tot = {'B': [], 'C': [], 'D': []}
        for b in sorted(agg[side]):
            a = agg[side][b]
            for k in tot:
                tot[k] += a[k]
            hh, mm = divmod(b, 60)
            print(f"  {hh:02d}:{mm:02d}   {m(a['B']):+6.2f}%  {m(a['C']):+6.2f}%  {m(a['D']):+6.2f}%   {len(a['C']):>6,}")
        print(f"  ---- ALL-DAY:  B={m(tot['B']):+.2f}%   C={m(tot['C']):+.2f}%   D={m(tot['D']):+.2f}%   "
              f"(n={len(tot['C']):,}) ----\n")
    print("DONE", flush=True)


if __name__ == '__main__':
    main()
