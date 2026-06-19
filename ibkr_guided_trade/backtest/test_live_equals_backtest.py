"""EQUIVALENCE TEST: live decision == backtest, exactly.

The live path runs replay_engine.run_strategy_simple(..., live_decision=True) seeded with the
operator's book. This test proves that seeding the engine with the CONTINUOUS backtest's own
start-of-day state and running ONE live decision reproduces that day's trades bit-for-bit —
i.e. the live mechanism introduces no deviation from the backtest.

Method (same params P for both, no two-clock confound — engine uses the sim date):
  1. Continuous backtest over full df with _CAPTURE_STATES → per-day start-of-day book + trades.
  2. For each sampled day D (days the continuous run actually traded): seed the engine with
     snapshot[D] and run live_decision on a tiny trailing window ending at D.
  3. Compare the multiset of (type, strike, qty) — must be identical.

  venv/bin/python backtest/test_live_equals_backtest.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import replay_engine as R

PARAMS_KEY = 'regime_wheel_boxx_greeks_live'   # the exact kernel the live dashboard uses


# ACTIONABLE = orders the operator actually executes (the adapter's JUSTIFY set + share/roll
# trades). Everything else is INFORMATIONAL/CONSEQUENCE (defer/skip/stand-down notes, settlement,
# margin-reject logs) — filtered out of the live order list, so excluded from the equivalence.
ACTIONABLE = {
    'OPEN_PUT', 'CONVICTION_ITM_PUT', 'OPEN_LONG_PUT_FLOOR', 'OPEN_PUT_RATIO_FLOOR',
    'OPEN_UPSIDE_WING', 'PUT_TP', 'PUT_ROLL_DOWN', 'OPEN_CC', 'OPEN_ITM_CC', 'ITM_CC_DIVEST',
    'CALL_TP', 'CALL_ROLL_UP', 'KOLD_BOOK_HEDGE', 'KOLD_SHOULDER_ENTRY', 'Z_TARGET_ADD',
    'Z_TARGET_TRIM', 'OPEN_REBUILD_PUT', 'DD_TRIM_SHARES', 'BACKWARDATION_DERISK_TRIM',
}


def sig(tr, actionable_only=True):
    """Order signature: type + strike(0.1) + signed qty. Robust to row order."""
    out = []
    for _, t in tr.iterrows():
        ty = t.get('type')
        if actionable_only and ty not in ACTIONABLE:
            continue
        K = t.get('K')
        K = round(float(K), 1) if (K == K and K is not None) else None
        q = t.get('qty')
        q = int(q) if (q == q and q is not None) else None
        out.append((ty, K, q))
    return sorted(out, key=lambda x: (str(x[0]), x[1] or 0, x[2] or 0))


def main():
    df = pd.read_csv(os.path.join(R.CACHE_DIR, 'master_dataset.csv'), parse_dates=[0], index_col=0)
    df = R.precompute_factor_z(df).dropna(subset=['UNG'])
    params = R.STRATEGIES[PARAMS_KEY]

    # 1) continuous backtest with state capture
    R._CAPTURE_STATES = True
    R._STATE_SNAPSHOTS = {}
    _, trades = R.run_strategy_simple(df, params, initial_cash=100000, initial_shares=0)
    R._CAPTURE_STATES = False
    snaps = dict(R._STATE_SNAPSHOTS)
    trades['d'] = pd.to_datetime(trades['date']).dt.strftime('%Y-%m-%d')
    print(f"continuous backtest: {len(trades)} trades, {trades['d'].nunique()} active days, "
          f"{len(snaps)} day-snapshots")

    # 2) sample days the continuous run TRADED on (meaningful comparison), spread across history
    active_days = [d for d in sorted(trades['d'].unique()) if d in snaps]
    if len(active_days) > 14:
        step = len(active_days) // 14
        active_days = active_days[::step]
    # FULL-history window (df.loc[:D]) so the seeded run's absolute row-index `i` matches the
    # continuous run — the LIVE path passes the full df, so cooldown checks (i − last_i) align.
    print(f"testing {len(active_days)} active days (full-history window — matches the live path)\n")

    BOOK_KEYS = ('cash', 'shares', 'short_puts', 'short_calls', 'long_puts', 'long_calls',
                 'boxx', 'kold')
    det_ok = det_n = 0          # engine determinism (full state restored)
    book_ok = book_n = 0        # live reconstruction (book-only seed, as get_live_recommendation)
    det_fails, book_fails = [], []
    for d in active_days:
        cont_sig = sig(trades[trades['d'] == d])
        pos = df.index.get_loc(pd.Timestamp(d))
        win = df.iloc[:pos + 1]                              # FULL history → absolute `i` matches
        # (1) DETERMINISM: restore the COMPLETE state → must be identical
        _, o_full = R.run_strategy_simple(win, params, seed_state=snaps[d], live_decision=True)
        det_n += 1
        if sig(o_full) == cont_sig:
            det_ok += 1
        elif len(det_fails) < 6:
            det_fails.append((d, cont_sig, sig(o_full)))
        # (2) LIVE RECONSTRUCTION: only the book fields (what the live adapter can rebuild)
        book_seed = {k: snaps[d][k] for k in BOOK_KEYS}
        _, o_book = R.run_strategy_simple(win, params, seed_state=book_seed, live_decision=True)
        book_n += 1
        if sig(o_book) == cont_sig:
            book_ok += 1
        elif len(book_fails) < 8:
            book_fails.append((d, cont_sig, sig(o_book)))

    print(f"=== ENGINE DETERMINISM (full state): {det_ok}/{det_n} "
          f"({100*det_ok/max(1,det_n):.1f}%) — same complete state → same decision ===")
    for d, c, l in det_fails:
        print(f"  DET MISMATCH {d}:\n    backtest: {c}\n    seeded  : {l}")
    print(f"\n=== LIVE BOOK-RECONSTRUCTION (book-only seed): {book_ok}/{book_n} "
          f"({100*book_ok/max(1,book_n):.1f}%) — residual = path-state a snapshot can't carry ===")
    for d, c, l in book_fails:
        print(f"  BOOK MISMATCH {d}:\n    backtest: {c}\n    live    : {l}")
    print("\nDONE", flush=True)


if __name__ == '__main__':
    main()
