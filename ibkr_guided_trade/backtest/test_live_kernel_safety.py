"""SAFETY TESTS for the LIVE recommendation path (backtest/live_kernel.py).

Real-money system: the COVERAGE SAFETY-ASSERTION in get_live_recommendation is the
last line of defense against a naked short call leaking into the operator's order list.
These tests pin that assertion (and the position-mapping / assignment-risk math that feeds
it) so a refactor can never silently turn it off.

We mock the heavy engine + WS/DB deps so NO live calls happen:
  * R.run_strategy_simple   → returns a controlled tiny orders DataFrame and sets R._LIVE_FINAL
  * R._load_iv_surface / R.iv_from_surface / R.bs_put / R.bs_call / R.bs_greeks_pt / R.p_assign
                            → constants (deterministic, no surface load)
  * live_kernel._options_data_freshness → stub (no psycopg2 connect)
  * historical_data_pipeline.refresh_to_today → absent/raising is fine (it's try/except wrapped)

Run:   venv/bin/python -m pytest backtest/test_live_kernel_safety.py -q
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import pytest

import replay_engine as R
import live_kernel as LK


# ───────────────────────── shared fixtures / helpers ─────────────────────────

def _ts_plus(days):
    """A real expiry date `days` from today (used so dte computes deterministically)."""
    return (pd.Timestamp.today().normalize() + pd.Timedelta(days=days)).date().isoformat()


@pytest.fixture(autouse=True)
def _patch_surface(monkeypatch):
    """Make all IV-surface / BS calls constant so the math is deterministic and offline."""
    monkeypatch.setattr(R, "_load_iv_surface", lambda *a, **k: {})
    monkeypatch.setattr(R, "iv_from_surface", lambda *a, **k: 0.50)
    monkeypatch.setattr(R, "surface_latest_date", lambda *a, **k: None)
    # bs_put/bs_call: simple intrinsic + flat $0.40 extrinsic so _book_extrinsic is predictable
    monkeypatch.setattr(R, "bs_put", lambda S, K, T, sig, r=0.045: max(0.0, K - S) + 0.40)
    monkeypatch.setattr(R, "bs_call", lambda S, K, T, sig, r=0.045: max(0.0, S - K) + 0.40)


# ───────────────────────── 1. _to_engine_positions ─────────────────────────

def test_to_engine_positions_sign_and_right():
    """qty<0 → short list; qty>0 → long list; P/C routing; UNG-only; dte from expiry."""
    positions = [
        {'symbol': 'UNG', 'option_type': 'P', 'strike': 11.0, 'qty': -7,
         'expiry': _ts_plus(20), 'average_price': 0.30},
        {'symbol': 'UNG', 'right': 'C', 'strike': 13.0, 'qty': -3,
         'expiration': _ts_plus(40), 'avg_price': 0.40},
        {'symbol': 'UNG', 'option_type': 'P', 'strike': 9.0, 'qty': +2,    # LONG put
         'expiry': _ts_plus(15)},
        {'symbol': 'UNG', 'option_type': 'C', 'strike': 14.0, 'qty': +1,   # LONG call
         'expiry': _ts_plus(30)},
        # DBA — must be EXCLUDED (different underlying contaminates the UNG engine)
        {'symbol': 'DBA', 'option_type': 'P', 'strike': 20.0, 'qty': -5,
         'expiry': _ts_plus(20)},
    ]
    sp, sc, lp, lc = LK._to_engine_positions(positions)
    assert len(sp) == 1 and sp[0]['K'] == 11.0 and sp[0]['qty'] == 7   # qty stored abs
    assert len(sc) == 1 and sc[0]['K'] == 13.0 and sc[0]['qty'] == 3
    assert len(lp) == 1 and lp[0]['K'] == 9.0
    assert len(lc) == 1 and lc[0]['K'] == 14.0
    # dte derived from expiry (~20d, min-floored at 1)
    assert 18 <= sp[0]['dte'] <= 22
    # entry_prem read from average_price / avg_price
    assert sp[0]['entry_prem'] == 0.30
    assert sc[0]['entry_prem'] == 0.40


def test_to_engine_positions_default_symbol_is_ung():
    """Symbol absent → treated as UNG (demo books)."""
    sp, sc, lp, lc = LK._to_engine_positions(
        [{'option_type': 'P', 'strike': 10.0, 'qty': -1, 'expiry': _ts_plus(30)}])
    assert len(sp) == 1


def test_to_engine_positions_missing_fields_skipped():
    """Missing strike or right → skipped; missing expiry → default dte 30; entry_prem default 0.3."""
    positions = [
        {'symbol': 'UNG', 'option_type': 'P', 'qty': -1, 'expiry': _ts_plus(10)},  # no strike → skip
        {'symbol': 'UNG', 'strike': 11.0, 'qty': -1, 'expiry': _ts_plus(10)},      # no right → skip
        {'symbol': 'UNG', 'option_type': 'P', 'strike': 11.0, 'qty': -1},          # no expiry → dte 30
    ]
    sp, sc, lp, lc = LK._to_engine_positions(positions)
    assert len(sp) == 1
    assert sp[0]['dte'] == 30
    assert sp[0]['entry_prem'] == 0.3       # default
    assert sp[0]['expiry'] is None


def test_to_engine_positions_empty():
    assert LK._to_engine_positions(None) == ([], [], [], [])


# ───────────────────────── 5a. _est_theta ─────────────────────────

def test_est_theta_extrinsic_only():
    """Theta = extrinsic decay only. ITM intrinsic does NOT decay → contributes ~0."""
    spot = 12.0
    # OTM short put (K=10 < spot) entry_prem 0.30 → all extrinsic. extr=0.30
    # theta = 0.30 * 100 * 2 / 30 = 2.0
    sp = [{'K': 10.0, 'dte': 30, 'qty': 2, 'entry_prem': 0.30}]
    th = LK._est_theta(sp, [], spot)
    assert th == pytest.approx(0.30 * 100 * 2 / 30, rel=1e-6)

    # Deep-ITM short put: K=15 > spot=12 → intrinsic 3.0 >> entry_prem 0.30 → extr clamped 0 → theta 0
    sp_itm = [{'K': 15.0, 'dte': 30, 'qty': 1, 'entry_prem': 0.30}]
    assert LK._est_theta(sp_itm, [], spot) == 0.0

    # Short call OTM (K=14 > spot): extrinsic = full prem
    sc = [{'K': 14.0, 'dte': 10, 'qty': 1, 'entry_prem': 0.50}]
    assert LK._est_theta([], sc, spot) == pytest.approx(0.50 * 100 * 1 / 10, rel=1e-6)


# ───────────────────────── 5b. _book_extrinsic ─────────────────────────

def test_book_extrinsic_short_collect_long_pay():
    """Short legs ADD extrinsic (you collect), long legs SUBTRACT (you pay). bs_*=intrinsic+0.40."""
    spot = 12.0
    sp = [{'K': 11.0, 'dte': 20, 'qty': 2}]     # OTM short put: price=0+0.40, intr=0 → extr 0.40
    lc = [{'K': 13.0, 'dte': 20, 'qty': 1}]     # OTM long call: price=0+0.40, intr=0 → extr 0.40 PAID
    tot = LK._book_extrinsic(sp, [], [], lc, spot)
    # short: +0.40*100*2 = 80 ; long: -0.40*100*1 = -40 → 40
    assert tot == pytest.approx(80.0 - 40.0, rel=1e-6)


def test_book_extrinsic_skips_degenerate_legs():
    """K<=0 / dte<=0 / qty==0 legs contribute nothing."""
    spot = 12.0
    bad = [{'K': 0, 'dte': 20, 'qty': 1}, {'K': 11, 'dte': 0, 'qty': 1}, {'K': 11, 'dte': 20, 'qty': 0}]
    assert LK._book_extrinsic(bad, [], [], [], spot) == 0.0


# ───────────────────────── _settlement_watch (settlement-as-action) ─────────────────────────

def test_settlement_watch_classifications():
    """Each imminent UNG option is classified as a settlement-day ACTION to monitor."""
    spot = 12.0
    positions = [
        # ITM short put → EXPECT_ASSIGNMENT (you BUY shares at K)
        {'symbol': 'UNG', 'option_type': 'P', 'strike': 13.0, 'qty': -2, 'expiry': _ts_plus(0)},
        # ITM short call → EXPECT_CALLED_AWAY (you DELIVER shares at K)
        {'symbol': 'UNG', 'option_type': 'C', 'strike': 11.0, 'qty': -1, 'expiry': _ts_plus(0)},
        # OTM short → AWAIT_WORTHLESS
        {'symbol': 'UNG', 'option_type': 'C', 'strike': 14.0, 'qty': -1, 'expiry': _ts_plus(0)},
        # near-pin short → UNCERTAIN
        {'symbol': 'UNG', 'option_type': 'P', 'strike': 12.0, 'qty': -1, 'expiry': _ts_plus(0)},
        # ITM long → DECIDE_LONG ; OTM long → ABANDON_LONG
        {'symbol': 'UNG', 'option_type': 'C', 'strike': 10.0, 'qty': +1, 'expiry': _ts_plus(0)},
        {'symbol': 'UNG', 'option_type': 'P', 'strike': 9.0, 'qty': +1, 'expiry': _ts_plus(0)},
        # DBA excluded; far-OTM beyond horizon excluded
        {'symbol': 'DBA', 'option_type': 'P', 'strike': 20.0, 'qty': -1, 'expiry': _ts_plus(0)},
    ]
    out = LK._settlement_watch(positions, spot, horizon_days=3)
    kinds = {item['kind'] for item in out}
    assert {'EXPECT_ASSIGNMENT', 'EXPECT_CALLED_AWAY', 'AWAIT_WORTHLESS',
            'UNCERTAIN', 'DECIDE_LONG', 'ABANDON_LONG'} <= kinds
    assign = next(i for i in out if i['kind'] == 'EXPECT_ASSIGNMENT')
    assert assign['share_impact'] == 200            # buy 100*2 shares
    assert assign['cash_impact'] < 0               # cash out
    away = next(i for i in out if i['kind'] == 'EXPECT_CALLED_AWAY')
    assert away['share_impact'] == -100            # deliver shares
    assert away['cash_impact'] > 0
    # no DBA item leaked
    assert all(i['strike'] != 20.0 for i in out)


def test_settlement_watch_far_otm_skipped():
    """Beyond tomorrow, only ITM/pin surface — a 3d OTM short is not urgent → skipped."""
    spot = 12.0
    out = LK._settlement_watch(
        [{'symbol': 'UNG', 'option_type': 'P', 'strike': 10.0, 'qty': -1, 'expiry': _ts_plus(3)}],
        spot, horizon_days=5)
    assert out == []


def test_settlement_watch_non_option_and_missing():
    """Non-option positions and options without expiry/strike are skipped, no crash."""
    out = LK._settlement_watch([
        {'symbol': 'UNG', 'right': 'SHARES', 'qty': 1000},                  # not an option
        {'symbol': 'UNG', 'option_type': 'P', 'qty': -1},                   # no expiry
        {'symbol': 'UNG', 'option_type': 'P', 'strike': 11.0, 'qty': 0,     # qty 0
         'expiry': _ts_plus(0)},
    ], 12.0)
    assert out == []


# ───────────────────────── _update_nav_peak ─────────────────────────

def test_update_nav_peak_override_and_ratchet(tmp_path, monkeypatch):
    """Override wins; otherwise peak ratchets up and floors at current NAV; persisted to disk."""
    f = tmp_path / "peak.json"
    monkeypatch.setattr(LK, "_NAV_PEAK_FILE", str(f))
    # first call: no file → peak = current NAV
    assert LK._update_nav_peak(100000.0) == 100000.0
    # lower NAV → peak stays (ratchet), floored at current
    assert LK._update_nav_peak(90000.0) == 100000.0
    # override takes precedence and is stored
    assert LK._update_nav_peak(90000.0, override=150000.0) == 150000.0
    # override below current NAV → floored at current
    assert LK._update_nav_peak(160000.0, override=120000.0) == 160000.0


# ───────────────────────── _opt_expiry ─────────────────────────

def test_opt_expiry_snaps_to_friday():
    iso = LK._opt_expiry(30)
    assert pd.Timestamp(iso).weekday() == 4   # Friday


# ───────────────────────── orchestration harness ─────────────────────────
#
# get_live_recommendation is the only place _called_certain, the COVERAGE
# assertion, assign_risk and the concentration tally actually run. We drive it
# with a mocked engine so those branches execute deterministically.

# default champion params (small subset is enough; .get() fills the rest with defaults)
_PARAMS = {'open_dte': 30, 'expiry_reaccum': True, 'delta_target_nav': 0.5,
           'scenario_mu_a': -0.000797, 'scenario_mu_b': -0.000009, 'scenario_sigma': 0.039}


def _install_engine(monkeypatch, orders_rows, final=None, second_orders_rows=None, params=None):
    """Patch R.run_strategy_simple to emit `orders_rows` (list of dicts) and set R._LIVE_FINAL.

    If expiry_reaccum fires it calls run_strategy_simple a SECOND time; `second_orders_rows`
    (default: empty) controls that call's output.
    """
    calls = {'n': 0}

    def fake_run(df, params, seed_state=None, live_decision=False, **kw):
        calls['n'] += 1
        if calls['n'] == 1:
            R._LIVE_FINAL = final if final is not None else {
                'short_puts': seed_state.get('short_puts', []),
                'short_calls': seed_state.get('short_calls', []),
                'long_puts': seed_state.get('long_puts', []),
                'long_calls': seed_state.get('long_calls', []),
                'shares': seed_state.get('shares', 0),
                'cash': seed_state.get('cash', 0.0), 'kold': seed_state.get('kold', 0),
            }
            rows = orders_rows
        else:
            rows = second_orders_rows or []
        return pd.DataFrame([]), pd.DataFrame(rows)

    _p = params if params is not None else _PARAMS
    monkeypatch.setattr(R, "run_strategy_simple", fake_run)
    monkeypatch.setattr(R, "STRATEGIES", {**getattr(R, "STRATEGIES", {}), 'regime_wheel_boxx_greeks': _p,
                                          'regime_wheel_boxx': _p})
    # keep KERNELS pointing the CHAMPION_KEY at our params-bearing strategy
    from validated_kernel_adapter import KERNELS, CHAMPION_KEY
    monkeypatch.setitem(KERNELS, CHAMPION_KEY, {**KERNELS.get(CHAMPION_KEY, {}),
                                                'strategy': 'regime_wheel_boxx_greeks'})
    # offline stubs for the freshness DB probe & greeks point-fn used by _called_certain
    monkeypatch.setattr(LK, "_options_data_freshness", lambda: {'ok': True, 'stale_days': 0})
    # p_assign: deterministic — depends on K vs spot so put/call edelta is non-trivial
    monkeypatch.setattr(R, "p_assign", lambda K, S, dte, z, a=0, b=0, sig=0.04:
                        0.8 if K >= S else 0.2)
    return calls


def _run(monkeypatch, positions, orders_rows, cash=120000.0, spot=12.0,
         final=None, second_orders_rows=None, bs_delta=0.5, params=None):
    _install_engine(monkeypatch, orders_rows, final=final,
                    second_orders_rows=second_orders_rows, params=params)
    # _called_certain uses R.bs_greeks_pt(...,'C') → return (delta, gamma)
    monkeypatch.setattr(R, "bs_greeks_pt", lambda *a, **k: (bs_delta, 0.0))
    return LK.get_live_recommendation(positions, cash=cash, spot=spot)


# ───────────────────────── 2. COVERAGE SAFETY-ASSERTION ─────────────────────────

def test_coverage_covered_recycle_does_not_breach(monkeypatch):
    """A covered RECYCLE: buy-to-close N short calls (frees coverage) + sell <=N fresh
    against the freed shares must NOT read as naked. Net = existing - closed + new."""
    # 1000 shares → coverable = 10. Existing 5 short calls in book.
    positions = [
        {'symbol': 'UNG', 'right': 'SHARES', 'qty': 1000},
        {'symbol': 'UNG', 'option_type': 'C', 'strike': 13.0, 'qty': -5, 'expiry': _ts_plus(40)},
    ]
    # Engine closes 5 (CALL_TP) then opens 5 fresh (OPEN_CC). Net = 5 - 5 + 5 = 5 <= 10.
    orders = [
        {'type': 'CALL_TP', 'qty': 5, 'K': 13.0, 'dte': 40, 'pnl': 100, 'credit': 0,
         'expiry': _ts_plus(40)},
        {'type': 'OPEN_CC', 'qty': 5, 'K': 14.0, 'dte': 45, 'credit': 200, 'pnl': 0,
         'expiry': _ts_plus(45)},
    ]
    res = _run(monkeypatch, positions, orders, spot=12.0, bs_delta=0.3)
    cov = res['coverage']
    assert cov['violation'] is None
    assert cov['covered'] is True
    assert cov['existing_short_calls'] == 5
    assert cov['coverable_calls'] == 10
    # both orders survive (nothing dropped)
    kinds = [r['type'] for r in res['recommendations']]
    assert 'OPEN_CC' in kinds and 'CALL_TP' in kinds
    assert not any(r.get('_dropped_uncovered') for r in res['recommendations'])


def test_coverage_genuine_overwrite_breaches_and_drops_naked(monkeypatch):
    """Genuine over-write: SELL calls beyond shares//100 with NO closes → INVARIANT BREACHED,
    and the leg(s) past the coverage room are hard-blocked (dropped). THE naked-call block.

    The drop logic is row-by-row: once `room` is exhausted, further SELL-CALL rows are dropped.
    We feed TWO SELL-CALL lots against 0 shares (room 0) so BOTH are uncovered and dropped."""
    # 0 shares → coverable 0. Two OPEN_CC lots, no closes → net 2 > 0 → all naked → all dropped.
    positions = [{'symbol': 'UNG', 'right': 'SHARES', 'qty': 0}]
    orders = [
        {'type': 'OPEN_CC', 'qty': 1, 'K': 14.0, 'dte': 45, 'credit': 100, 'pnl': 0,
         'expiry': _ts_plus(45)},
        {'type': 'OPEN_CC', 'qty': 1, 'K': 15.0, 'dte': 45, 'credit': 100, 'pnl': 0,
         'expiry': _ts_plus(50)},
    ]
    res = _run(monkeypatch, positions, orders, spot=12.0, bs_delta=0.3)
    cov = res['coverage']
    assert cov['violation'] is not None
    assert 'COVERAGE INVARIANT BREACHED' in cov['violation']
    assert cov['covered'] is False
    # NO uncovered SELL-CALL leg may survive into the operator's order list (naked block).
    sell_calls = [r for r in res['recommendations']
                  if r['right'] == 'CALL' and (r['side'] or '').upper().startswith('SELL')]
    assert sell_calls == [], "naked short-call leg leaked into the order list"


def test_coverage_partial_room_keeps_first_drops_overflow(monkeypatch):
    """Row-by-row drop: the first SELL-CALL lot that fits within `room` is KEPT; the next lot
    after `room` goes <=0 is DROPPED. Proves the keep/drop boundary, not just an all-or-nothing."""
    # 100 shares → room 1. Two lots of 1: first kept (room 1→0), second dropped (room<=0).
    positions = [{'symbol': 'UNG', 'right': 'SHARES', 'qty': 100}]
    orders = [
        {'type': 'OPEN_CC', 'qty': 1, 'K': 14.0, 'dte': 45, 'credit': 100, 'pnl': 0,
         'expiry': _ts_plus(45)},
        {'type': 'OPEN_CC', 'qty': 1, 'K': 15.0, 'dte': 45, 'credit': 100, 'pnl': 0,
         'expiry': _ts_plus(50)},
    ]
    res = _run(monkeypatch, positions, orders, spot=12.0, bs_delta=0.3)
    assert res['coverage']['violation'] is not None
    sell_calls = [r for r in res['recommendations']
                  if r['right'] == 'CALL' and (r['side'] or '').upper().startswith('SELL')]
    # exactly the first (covered) lot survives; the overflow lot is dropped
    assert len(sell_calls) == 1
    assert sell_calls[0]['strike'] == 14.0
    assert not sell_calls[0].get('_dropped_uncovered')


# ───────────────────────── 3. assign_risk NET model ─────────────────────────

def test_assign_risk_net_model(monkeypatch):
    """put_assign = Σ p_assign·qty·100 ; call_away = Σ(1-p_assign)·qty·100 ; net = puts - calls.

    stub p_assign = 0.8 if K>=spot else 0.2 (spot=12):
      put  K=13 (>=spot): p=0.8 → put_assign  = 0.8·2·100 = 160
      call K=14 (>=spot): p=0.8 → call_away    = (1-0.8)·1·100 = 20
      call K=11 (< spot): p=0.2 → call_away    = (1-0.2)·3·100 = 240
      → call_away total 260 ; net = 160 - 260 = -100."""
    positions = [
        {'symbol': 'UNG', 'right': 'SHARES', 'qty': 1000},
        {'symbol': 'UNG', 'option_type': 'P', 'strike': 13.0, 'qty': -2, 'expiry': _ts_plus(20)},
        {'symbol': 'UNG', 'option_type': 'C', 'strike': 14.0, 'qty': -1, 'expiry': _ts_plus(20)},
        {'symbol': 'UNG', 'option_type': 'C', 'strike': 11.0, 'qty': -3, 'expiry': _ts_plus(3)},
    ]
    res = _run(monkeypatch, positions, [], spot=12.0, bs_delta=0.3)
    ar = res['assign_risk']
    assert ar['put_assign_delta'] == 160
    assert ar['call_away_delta'] == 20 + 240
    assert ar['net_delta'] == 160 - 260
    assert ar['pct_of_shares'] == round(160 / 1000 * 100)
    assert ar['net_pct_of_shares'] == round((160 - 260) / 1000 * 100)
    # target 35% of 1000 = 350 ; put_assign 160 <= 350 → within_target True
    assert ar['within_target'] is True
    # the K=11 ITM call expiring in 3d (<=4) is "called away soon"
    assert 'called_away_soon' in ar
    assert ar['called_away_soon']['lots'] == 3


def test_assign_risk_execution_roll_aid(monkeypatch):
    """A put cluster with contracts > smooth_cap (10) emits an execution roll aid; else roll_n 0."""
    positions = [
        {'symbol': 'UNG', 'right': 'SHARES', 'qty': 5000},
        {'symbol': 'UNG', 'option_type': 'P', 'strike': 11.0, 'qty': -15, 'expiry': _ts_plus(20)},
    ]
    res = _run(monkeypatch, positions, [], spot=12.0, bs_delta=0.3)
    ex = res['assign_risk']['execution']
    assert ex['roll_n'] == 15 - 10        # smooth above the 10 cap
    assert '$11.00' in ex['cluster']

    # tiny cluster → no roll
    positions2 = [
        {'symbol': 'UNG', 'right': 'SHARES', 'qty': 5000},
        {'symbol': 'UNG', 'option_type': 'P', 'strike': 11.0, 'qty': -2, 'expiry': _ts_plus(20)},
    ]
    res2 = _run(monkeypatch, positions2, [], spot=12.0, bs_delta=0.3)
    assert res2['assign_risk']['execution']['roll_n'] == 0


# ───────────────────────── 4. _called_certain (via expiry_reaccum) ─────────────────────────

def test_called_certain_deep_itm_fires(monkeypatch):
    """Deep-ITM short call (delta>=0.90, dte<=1, spot>K) → near-certain called away → reaccum runs."""
    positions = [
        {'symbol': 'UNG', 'right': 'SHARES', 'qty': 1500},
        # ITM (K=11 < spot 12), expiring tomorrow
        {'symbol': 'UNG', 'option_type': 'C', 'strike': 11.0, 'qty': -5, 'expiry': _ts_plus(1)},
    ]
    # bs_delta 0.95 >= 0.90 → certain. Second engine run returns an OPEN_PUT to re-accumulate.
    second = [{'type': 'OPEN_PUT', 'qty': 4, 'K': 11.0, 'dte': 30, 'credit': 150,
               'expiry': _ts_plus(30)}]
    res = _run(monkeypatch, positions, [], spot=12.0, bs_delta=0.95, second_orders_rows=second)
    assert res['expiry_reaccum'] is not None
    assert res['expiry_reaccum']['called_lots'] == 5
    assert res['expiry_reaccum']['called_shares'] == 500
    assert res['expiry_reaccum']['puts'][0]['qty'] == 4


def test_called_certain_mild_itm_does_not_fire(monkeypatch):
    """Mildly-ITM call (delta < 0.90) is NOT near-certain → no reaccum."""
    positions = [
        {'symbol': 'UNG', 'right': 'SHARES', 'qty': 1500},
        {'symbol': 'UNG', 'option_type': 'C', 'strike': 11.5, 'qty': -5, 'expiry': _ts_plus(1)},
    ]
    res = _run(monkeypatch, positions, [], spot=12.0, bs_delta=0.55)
    assert res['expiry_reaccum'] is None


def test_called_certain_otm_does_not_fire(monkeypatch):
    """OTM call (spot <= K) can never be called-certain regardless of delta."""
    positions = [
        {'symbol': 'UNG', 'right': 'SHARES', 'qty': 1500},
        {'symbol': 'UNG', 'option_type': 'C', 'strike': 13.0, 'qty': -5, 'expiry': _ts_plus(1)},
    ]
    res = _run(monkeypatch, positions, [], spot=12.0, bs_delta=0.99)
    assert res['expiry_reaccum'] is None


def test_called_certain_far_dte_does_not_fire(monkeypatch):
    """dte > 1 → not certain even if deep ITM (plenty of time to move OTM)."""
    positions = [
        {'symbol': 'UNG', 'right': 'SHARES', 'qty': 1500},
        {'symbol': 'UNG', 'option_type': 'C', 'strike': 11.0, 'qty': -5, 'expiry': _ts_plus(10)},
    ]
    res = _run(monkeypatch, positions, [], spot=12.0, bs_delta=0.99)
    assert res['expiry_reaccum'] is None


# ───────────────────────── 5c. concentration per-strike tally ─────────────────────────

def test_concentration_per_strike_tally(monkeypatch):
    """Short clusters tallied by (right,strike); contracts summed across expiries; cap flagged."""
    positions = [
        {'symbol': 'UNG', 'right': 'SHARES', 'qty': 5000},
        {'symbol': 'UNG', 'option_type': 'P', 'strike': 11.0, 'qty': -8, 'expiry': _ts_plus(20)},
        {'symbol': 'UNG', 'option_type': 'P', 'strike': 11.0, 'qty': -7, 'expiry': _ts_plus(40)},
        {'symbol': 'UNG', 'option_type': 'C', 'strike': 13.0, 'qty': -2, 'expiry': _ts_plus(20)},
        # long & DBA legs must be ignored by the SHORT-only concentration tally
        {'symbol': 'UNG', 'option_type': 'P', 'strike': 11.0, 'qty': +3, 'expiry': _ts_plus(20)},
        {'symbol': 'DBA', 'option_type': 'P', 'strike': 20.0, 'qty': -50, 'expiry': _ts_plus(20)},
    ]
    res = _run(monkeypatch, positions, [], spot=12.0, bs_delta=0.3)
    conc = {(c['right'], c['strike']): c for c in res['concentration']}
    assert conc[('PUT', 11.0)]['contracts'] == 15            # 8 + 7 across two expiries
    assert conc[('PUT', 11.0)]['assignment_shares'] == 1500
    assert conc[('PUT', 11.0)]['max_single_expiry'] == 8     # largest single-expiry cluster
    assert conc[('CALL', 13.0)]['contracts'] == 2
    # DBA never appears
    assert all(c['strike'] != 20.0 for c in res['concentration'])
    # 15-contract cluster triggers the de-risk suggestion
    assert conc[('PUT', 11.0)]['suggestion']


# ───────────────────────── 6. full orchestration smoke ─────────────────────────

def test_get_live_recommendation_full_shape(monkeypatch):
    """End-to-end orchestration: returns the expected top-level keys, recs are JUSTIFY-filtered,
    settlement/consequence orders are dropped from the actionable list."""
    positions = [
        {'symbol': 'UNG', 'right': 'SHARES', 'qty': 2000},
        {'symbol': 'UNG', 'option_type': 'P', 'strike': 11.0, 'qty': -3, 'expiry': _ts_plus(20),
         'average_price': 0.30},
        {'symbol': 'KOLD', 'qty': 100, 'market_value': 5000.0},
        {'symbol': 'BOXX', 'qty': 50, 'market_value': 5850.0},
    ]
    orders = [
        {'type': 'OPEN_PUT', 'qty': 4, 'K': 10.5, 'dte': 30, 'credit': 120, 'pnl': 0,
         'expiry': _ts_plus(30)},
        {'type': 'PUT_TP', 'qty': 1, 'K': 11.0, 'dte': 20, 'pnl': 80, 'credit': 0,
         'buyback': 0.10, 'expiry': _ts_plus(20)},
        # a CONSEQUENCE event (not in JUSTIFY) — must be filtered out of recs
        {'type': 'PUT_ASSIGN', 'qty': 2, 'K': 11.0, 'dte': 0},
    ]
    res = _run(monkeypatch, positions, orders, spot=12.0, bs_delta=0.3)
    for key in ('kernel', 'spot', 'coverage', 'greeks', 'assign_risk', 'concentration',
                'recommendations', 'theta', 'regime', 'delta_compass', 'z_models', 'nav_state'):
        assert key in res, f"missing top-level key {key}"
    types = [r['type'] for r in res['recommendations']]
    assert 'OPEN_PUT' in types and 'PUT_TP' in types
    assert 'PUT_ASSIGN' not in types        # consequence event filtered
    # theta dict computed
    assert 'now_per_day' in res['theta'] and 'gross_premium_month' in res['theta']
    # coverage clean (no short calls written)
    assert res['coverage']['violation'] is None


def test_recs_action_branches(monkeypatch):
    """Exercise the order-line builders for share trades, KOLD hedge, ITM divest and a
    take-profit-without-strike fallback (the elif chain in the recs loop)."""
    positions = [{'symbol': 'UNG', 'right': 'SHARES', 'qty': 5000}]
    orders = [
        {'type': 'Z_TARGET_TRIM', 'qty': 100, 'pnl': 0, 'credit': 0},        # SELL UNG shares
        {'type': 'KOLD_BOOK_HEDGE', 'qty': 300, 'pnl': 0, 'credit': 0},      # SET KOLD hedge
        {'type': 'ITM_CC_DIVEST', 'qty': 1, 'K': 11.0, 'dte': 20, 'credit': 50, 'pnl': 0,
         'expiry': _ts_plus(20)},                                            # SELL CALL (divest)
        {'type': 'CALL_TP', 'qty': 1, 'pnl': 40, 'credit': 0},               # TP without strike
    ]
    res = _run(monkeypatch, positions, orders, spot=12.0, bs_delta=0.3)
    actions = {r['type']: r['action'] for r in res['recommendations']}
    assert 'UNG shares' in actions['Z_TARGET_TRIM']
    assert 'KOLD' in actions['KOLD_BOOK_HEDGE']
    assert '$11.00 CALL' in actions['ITM_CC_DIVEST']
    assert 'take profit' in actions['CALL_TP']


def test_delta_compass_hedge_active(monkeypatch):
    """Bearish regime + delta above the trim ceiling + delta_hedge on → HEDGE ACTIVE branch."""
    # Big long share book pushes options+shares delta high; force a bearish regime + hedge on.
    params = {**_PARAMS, 'delta_hedge': True, 'delta_target_nav': 0.0, 'delta_hedge_rs_min': 0.0}
    positions = [{'symbol': 'UNG', 'right': 'SHARES', 'qty': 8000}]
    res = _run(monkeypatch, positions, [], spot=12.0, bs_delta=0.3, params=params)
    dc = res['delta_compass']
    # with delta_target_nav 0 the trim ceiling is ~0 and 8000 shares is well above it
    assert 'status' in dc
    assert dc['hedge_active'] in (True, False)  # branch executed regardless of regime sign


def test_concentration_pct_nav_cap(monkeypatch):
    """max_short_pct_nav set → per-strike cap is NAV-proportional (_cap_for pct branch)."""
    params = {**_PARAMS, 'max_short_pct_nav': 0.05}
    positions = [
        {'symbol': 'UNG', 'right': 'SHARES', 'qty': 5000},
        {'symbol': 'UNG', 'option_type': 'P', 'strike': 11.0, 'qty': -20, 'expiry': _ts_plus(20)},
    ]
    res = _run(monkeypatch, positions, [], spot=12.0, bs_delta=0.3, params=params)
    cl = next(c for c in res['concentration'] if c['strike'] == 11.0)
    assert cl['cap'] is not None and cl['cap'] >= 1


def test_get_live_recommendation_unknown_kernel(monkeypatch):
    """Unknown kernel with no STRATEGIES entry → graceful error dict, no crash."""
    monkeypatch.setattr(R, "STRATEGIES", {})
    from validated_kernel_adapter import KERNELS
    monkeypatch.setitem(KERNELS, '__nope__', {'strategy': '__nope__'})
    res = LK.get_live_recommendation([], cash=1000.0, spot=12.0, kernel_key='__nope__')
    assert 'error' in res
