"""COST-MODEL TEST: Wealthsimple commission-free + in-engine spread accounting.

Covers backtest/honest_walkforward.py — the cost constants and measure_period():

  1. COST CONSTANTS: COMMISSION_PER_CONTRACT == 0 (Wealthsimple is commission-free, the
     real cost is the bid/ask SPREAD modeled in-engine via SPREAD_OPTION, not a per-leg fee)
     and SLIPPAGE_PCT_OF_PREMIUM == 0 (spread is in-engine; an extra slippage term would
     double-count the open). Together they mean measure_period's cost_drag is ALWAYS $0.

  2. measure_period(): on a short real slice of master_dataset.csv with a real STRATEGIES
     kernel — asserts the result dict shape (ret/ann/sharpe/mdd/cost_drag/n_trades/yrs),
     cost_drag == 0, and finite/sane ann/sharpe/mdd. Plus the two guards: <50 rows -> None,
     and an exception inside the body -> {'error': ...}.

  3. EARLY-ASSIGNMENT ACCOUNTING: same kernel with model_early_assign True vs False — both
     produce valid measure_period results, and PUT_EARLY_ASSIGN trades appear ONLY when the
     flag is on (run_strategy_simple inspected directly).

Run:    venv/bin/python -m pytest backtest/test_cost_model.py -q
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import math

import pandas as pd
import pytest

# Import via the `backtest` package path so `coverage --source=backtest/honest_walkforward.py`
# matches the imported module's file. (replay_engine resolves via honest_walkforward's own
# sys.path.insert.) Fall back to the flat import if run from inside backtest/.
try:
    import backtest.honest_walkforward as H
    import backtest.replay_engine as R
except ImportError:  # pragma: no cover
    import honest_walkforward as H
    import replay_engine as R

# A real kernel that trades actively enough to exercise opens/TPs/rolls and early-assignment.
STRAT_KEY = 'regime_wheel_boxx_greeks'
# Slice chosen (see scan) so PUT_EARLY_ASSIGN fires with model_early_assign on, zero when off.
WIN_START = 200
WIN_LEN = 220


@pytest.fixture(scope='module')
def df_full():
    """master_dataset.csv -> factor-Z precomputed, UNG present (same prep as honest_walkforward)."""
    df = pd.read_csv(os.path.join(H.THIS_DIR, 'cache', 'master_dataset.csv'),
                     index_col=0, parse_dates=True)
    df = R.precompute_factor_z(df).dropna(subset=['UNG'])
    return df


@pytest.fixture(scope='module')
def df_period(df_full):
    """A short (~220-row) window — keeps the whole test fast."""
    return df_full.iloc[WIN_START:WIN_START + WIN_LEN]


@pytest.fixture(scope='module')
def strat():
    return R.STRATEGIES[STRAT_KEY]


# --------------------------------------------------------------------------- #
# 1. COST CONSTANTS — the Wealthsimple model                                  #
# --------------------------------------------------------------------------- #

def test_commission_is_zero():
    """Wealthsimple is commission-free — no per-contract fee (was IBKR $0.65)."""
    assert H.COMMISSION_PER_CONTRACT == 0.0


def test_slippage_is_zero():
    """No extra slippage term — the bid/ask spread is already modeled in-engine."""
    assert H.SLIPPAGE_PCT_OF_PREMIUM == 0.0


def test_spread_is_modeled_in_engine():
    """The real cost lives in the engine (SPREAD_OPTION half-spread), not in the cost constants."""
    assert R.SPREAD_OPTION > 0  # spread cost is real and in-engine, just not a commission/slippage fee


# --------------------------------------------------------------------------- #
# 2. measure_period — happy path + dict shape + zero cost_drag                 #
# --------------------------------------------------------------------------- #

def test_measure_period_shape_and_sanity(strat, df_period):
    res = H.measure_period(strat, df_period, cash_start=100000)
    assert res is not None
    assert 'error' not in res

    # exact dict shape
    assert set(res.keys()) == {'ret', 'ann', 'sharpe', 'mdd', 'cost_drag', 'n_trades', 'yrs'}

    # Wealthsimple: commission 0 + slippage 0 -> cost_drag is exactly 0
    assert res['cost_drag'] == 0

    # finite + sane
    for k in ('ret', 'ann', 'sharpe', 'mdd'):
        assert math.isfinite(res[k]), f'{k} not finite: {res[k]!r}'
    assert -100.0 <= res['mdd'] <= 0.0          # drawdown is non-positive, bounded by -100%
    assert -100.0 < res['ann'] < 1000.0          # annualized return in a sane band
    assert abs(res['sharpe']) < 100.0            # Sharpe not blown up
    assert res['n_trades'] >= 0
    assert res['yrs'] > 0


def test_measure_period_cost_drag_zero_regardless_of_trades(strat, df_full):
    """cost_drag stays $0 even on a much busier (longer) window — constants drive it, not volume."""
    busy = df_full.iloc[WIN_START:WIN_START + 400]
    res = H.measure_period(strat, busy, cash_start=100000)
    assert res is not None and 'error' not in res
    assert res['cost_drag'] == 0
    assert res['n_trades'] > 0  # it really did trade; drag is still zero


def test_measure_period_short_window_guard(df_full, strat):
    """<50 rows -> None (cannot measure a period this short)."""
    tiny = df_full.iloc[WIN_START:WIN_START + 40]
    assert len(tiny) < 50
    assert H.measure_period(strat, tiny, cash_start=100000) is None


def test_measure_period_exception_branch(df_period):
    """An error inside the body is caught and surfaced as {'error': ...}, not raised."""
    # A non-dict 'strategy' makes the engine's param lookups blow up inside the try-block,
    # which measure_period must catch and return as an error dict (len>=50 so it passes the guard).
    res = H.measure_period(object(), df_period, cash_start=100000)
    assert isinstance(res, dict)
    assert 'error' in res
    assert isinstance(res['error'], str) and res['error']


# --------------------------------------------------------------------------- #
# 3. Early-assignment accounting effect                                       #
# --------------------------------------------------------------------------- #

def _trade_types(trades):
    return trades['type'].astype(str)


def test_early_assign_on_vs_off(strat, df_period):
    """PUT_EARLY_ASSIGN trades appear only with model_early_assign on; both runs stay valid."""
    strat_on = {**strat, 'model_early_assign': True}
    strat_off = {**strat, 'model_early_assign': False}

    _, trades_on = R.run_strategy_simple(df_period, strat_on, 100000, 0)
    _, trades_off = R.run_strategy_simple(df_period, strat_off, 100000, 0)

    ea_on = int((_trade_types(trades_on) == 'PUT_EARLY_ASSIGN').sum())
    ea_off = int((_trade_types(trades_off) == 'PUT_EARLY_ASSIGN').sum())

    # The accounting effect: early-assignment is a modeled event ONLY when the flag is on.
    assert ea_on > 0, 'expected PUT_EARLY_ASSIGN trades with model_early_assign=True on this window'
    assert ea_off == 0, 'no PUT_EARLY_ASSIGN should occur with model_early_assign=False'

    # Both configurations still produce valid measure_period results.
    res_on = H.measure_period(strat_on, df_period, 100000)
    res_off = H.measure_period(strat_off, df_period, 100000)
    for res in (res_on, res_off):
        assert res is not None and 'error' not in res
        assert res['cost_drag'] == 0
        assert math.isfinite(res['ann']) and math.isfinite(res['sharpe']) and math.isfinite(res['mdd'])
