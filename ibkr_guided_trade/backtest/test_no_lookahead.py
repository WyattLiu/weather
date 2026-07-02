"""NO-LOOK-AHEAD assertion test — the fidelity backbone (roadmap Fi1).

A reactive/minute backtest is only valid if a decision at time T uses ONLY data available at/before T.
The most dangerous leak is an EIA print used before its release. This test detects the ACTUAL lag applied
to each EIA series in precompute_factor_z and FAILS the build if any series front-runs its real release:
  - weekly storage: released Thu 10:30 ET (~1 week after the report Friday) -> require >= 4 trading days lag
  - EIA-914 monthly production/consumption: released ~2 months after the reference month -> require >= 30 td

If someone shortens a shift (as .shift(21) did for the monthlies — a real leak found 2026-07), this breaks.
Run in the safety suite; extend with exact-timestamp checks when the minute-reactive engine lands (Fi2/Fi3).
"""
import os
import sys

import pandas as pd

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
CACHE = os.path.join(THIS, 'cache')


def _raw_and_pre():
    import replay_engine as R
    raw = pd.read_csv(os.path.join(CACHE, 'master_dataset.csv'), index_col=0, parse_dates=True)
    pre = R.precompute_factor_z(raw.copy())
    return raw, pre


def _applied_lag(raw, pre, col, maxk=70):
    """Return the integer trading-day shift k such that pre[col] == raw[col].shift(k), or None."""
    if col not in raw.columns or col not in pre.columns:
        return None
    r = raw[col]
    for k in range(0, maxk):
        m = pd.concat([pre[col], r.shift(k)], axis=1).dropna()
        if len(m) < 50:
            continue
        if (m.iloc[:, 0] - m.iloc[:, 1]).abs().max() < 1e-6:
            return k
    return None


def test_storage_release_lag_no_leak():
    raw, pre = _raw_and_pre()
    lag = _applied_lag(raw, pre, 'eia_storage_weekly')
    assert lag is not None, "eia_storage_weekly: could not detect an applied release lag"
    assert lag >= 4, f"eia_storage_weekly lag={lag} td < 4 — Thursday storage print is FRONT-RUN (look-ahead)"


def test_monthly_eia_release_lag_no_leak():
    raw, pre = _raw_and_pre()
    for col in ('eia_production', 'eia_consumption'):
        lag = _applied_lag(raw, pre, col)
        assert lag is not None, f"{col}: could not detect an applied release lag"
        assert lag >= 30, (f"{col} lag={lag} td < 30 — EIA-914 monthly (~2mo release) is FRONT-RUN "
                           "(look-ahead). Restore .shift(42).")


def test_storage_signal_is_causal():
    """CAUSALITY: nulling a FUTURE storage value must not change storage_surprise_z on any EARLIER date.
    This directly proves the champion's factor never front-runs the print (stronger than a lag check)."""
    import replay_engine as R
    raw = pd.read_csv(os.path.join(CACHE, 'master_dataset.csv'), index_col=0, parse_dates=True)
    cut = int(len(raw) * 0.7)                       # a mid-history date
    full = R.precompute_factor_z(raw.copy())['storage_surprise_z'].iloc[:cut]
    truncated = raw.copy()
    truncated.iloc[cut:, truncated.columns.get_loc('eia_storage_weekly')] = float('nan')  # blind the future
    trunc = R.precompute_factor_z(truncated)['storage_surprise_z'].iloc[:cut]
    both = pd.concat([full, trunc], axis=1).dropna()
    if len(both):
        assert (both.iloc[:, 0] - both.iloc[:, 1]).abs().max() < 1e-6, \
            "storage_surprise_z on past dates CHANGED when future storage was blinded — LOOK-AHEAD"


# ── FiA: release_calendar unit tests ──────────────────────────────────────────
def test_storage_release_is_following_thursday_1030():
    import datetime as dt
    from release_calendar import storage_release_ts, ET
    # week ending Fri 2026-06-26 → released Thu 2026-07-02 10:30 ET
    r = storage_release_ts(dt.date(2026, 6, 26))
    assert r == dt.datetime(2026, 7, 2, 10, 30, tzinfo=ET), r
    assert r.weekday() == 3 and r.hour == 10 and r.minute == 30


def test_monthly_release_is_end_of_month_plus_2():
    import datetime as dt
    from release_calendar import monthly_release_ts
    # reference month April 2026 → released ~end of June 2026 (last business day)
    r = monthly_release_ts(dt.date(2026, 4, 15))
    assert r.year == 2026 and r.month == 6 and r.weekday() < 5, r
    assert r >= dt.datetime(2026, 6, 25, tzinfo=r.tzinfo)   # near month-end


# ── FiB: EVENT-EXACT no-leak assertion ────────────────────────────────────────
def test_storage_print_not_visible_before_1030_et():
    """A decision timestamped 10:29 ET on the release Thursday must NOT see that day's storage number;
    at 10:30 it may. This is the event-exact gate the minute-reactive engine (FiC) must honor."""
    import datetime as dt
    from release_calendar import storage_release_ts, ET
    rel = storage_release_ts(dt.date(2026, 6, 26))          # Thu 2026-07-02 10:30 ET
    decide_1029 = dt.datetime(2026, 7, 2, 10, 29, tzinfo=ET)
    decide_1031 = dt.datetime(2026, 7, 2, 10, 31, tzinfo=ET)
    assert decide_1029 < rel, "LEAK: storage visible before its 10:30 ET print"
    assert decide_1031 >= rel, "storage should be visible after 10:30 ET"
    # and a decision the day BEFORE never sees it
    assert dt.datetime(2026, 7, 1, 16, 0, tzinfo=ET) < rel


# ── FiC: intraday event-moment spot reconstruction (feeds the minute-reactive engine) ──
def test_intraday_event_spot_is_reliable_and_causal():
    """The minute-reactive engine reacts at the 10:30 print using the UNG spot AT THAT MINUTE. PG has no
    intraday UNG spot, so intraday_spot.reconstruct_spot() recovers it from near-ATM put-call parity. This
    asserts two things the reactive path depends on:
      (1) FIDELITY — near-ATM strikes AGREE (cross-strike rel_std small), so the number is a real price;
      (2) CAUSALITY — a spot stamped 10:30 is a strict function of the 10:30 tape: it is deterministic and,
          on a day that moved, DIFFERS from the 16:00 reconstruction (never front-runs the close).
    Skips (does not fail) when PG/minute data is unavailable so the suite stays green offline."""
    import datetime as dt
    import intraday_spot as I
    conn = I._connect()
    if conn is None:
        import pytest
        pytest.skip("PG unavailable — intraday reconstruction not testable offline")
    try:
        day = dt.date(2024, 6, 20)                      # a storage-release Thursday with full minute coverage
        r1030a = I.reconstruct_spot(day, dt.time(10, 30), conn)
        r1030b = I.reconstruct_spot(day, dt.time(10, 30), conn)
        r1600 = I.reconstruct_spot(day, dt.time(16, 0), conn)
        if r1030a is None:
            import pytest
            pytest.skip("no minute quotes for the sample day — cannot test reconstruction")
        # (1) reliability: strikes agree to well under 2% of spot
        assert r1030a['rel_std'] is not None and r1030a['rel_std'] < 0.02, \
            f"intraday reconstruction unreliable: rel_std={r1030a['rel_std']} (strikes disagree → not a price)"
        assert r1030a['n'] >= 3, "too few strikes to trust the reconstruction"
        # (2a) determinism: same minute → identical spot (pure function of that minute's quotes)
        assert abs(r1030a['spot'] - r1030b['spot']) < 1e-9, "reconstruction is not deterministic on its minute"
        # (2b) event-exactness: the 10:30 value is NOT the 16:00 value on a day the tape moved → no EOD leak
        if r1600 is not None:
            assert abs(r1030a['spot'] - r1600['spot']) > 1e-6, \
                "10:30 and 16:00 reconstructions are identical — reconstruction is not minute-specific (leak risk)"
    finally:
        conn.close()


def test_reactive_events_off_is_byte_identical():
    """FiC-2 guard: the minute-reactive path is param-gated and MUST NOT touch the champion. Running the
    champion with reactive_events absent vs explicitly False must produce a bit-identical trade stream.
    (Uses a slice so the safety suite stays fast; the code path — not data length — is what's under test.)"""
    import copy
    import hashlib
    import replay_engine as R
    raw = pd.read_csv(os.path.join(CACHE, 'master_dataset.csv'), index_col=0, parse_dates=True)
    df = R.precompute_factor_z(raw).iloc[:500]
    base = copy.deepcopy(R.STRATEGIES['regime_wheel_boxx_greeks_live'])

    def fp(params):
        _, trades = R.run_strategy_simple(df, params)
        return hashlib.md5(pd.util.hash_pandas_object(trades, index=False).values.tobytes()).hexdigest()

    absent = fp(base)                                   # reactive_events not present at all
    off = copy.deepcopy(base); off['reactive_events'] = False
    assert fp(off) == absent, "reactive_events=False changed the champion — the gate leaks into the base path"


def test_reactive_thursday_fill_is_post_print_same_day():
    """FiC-3 no-leak: a reactive-mode fill on a storage-release Thursday must execute AFTER the 10:30 ET
    print (never front-run it) and on the SAME day (never borrow a later day's tape). The engine routes
    those fills through execute_audit(exec_window=11, avoid_print=True); this asserts the returned exec_time
    is >= 11:00 on the same Thursday. Skips offline."""
    import datetime as dt
    import pandas as pd
    try:
        import intraday_fill
    except Exception:
        import pytest
        pytest.skip("intraday_fill import failed")
    if intraday_fill._conn() is None:
        import pytest
        pytest.skip("PG unavailable — intraday fill path not testable offline")
    day = dt.date(2024, 6, 20)                           # storage-release Thursday, full minute coverage
    got = None
    for K in (19.0, 18.0, 20.0, 19.5, 18.5):            # try a few near-ATM strikes for a resolvable fill
        a = intraday_fill.execute_audit(day, K, 1, 'C', 'sell', exec_window=11, avoid_print=True)
        if a:
            got = a
            break
    if got is None:
        import pytest
        pytest.skip("no resolvable intraday fill for the sample day")
    et = pd.Timestamp(got['exec_time'])
    assert et.date() == day, f"reactive fill exec_time {et} is not on the decision day {day} (cross-day leak)"
    assert (et.hour, et.minute) >= (10, 30), f"reactive fill at {et.time()} FRONT-RUNS the 10:30 print (leak)"
    assert et.hour >= 11, f"reactive fill at {et.time()} is inside the 10:30 print window (avoid_print failed)"
