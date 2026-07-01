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
