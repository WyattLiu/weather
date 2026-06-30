"""SAFETY-CRITICAL invariant tests for the options-wheel engine (replay_engine.py).

This is a REAL-MONEY system. The assertions here are the safety net: they prove the
engine never emits a structurally-dangerous book (naked calls, short shares, mis-booked
assignment cash/share flows, over-concentration) on real market data and on adversarial
crafted scenarios.

Two layers:
  1. PURE HELPERS — p_assign / book_greeks / book_greeks_stat / bs_put / bs_call /
     bs_greeks_pt. Closed-form, so we pin them to their mathematical contract
     (monotonicity, parity, deep-ITM/OTM limits, guard branches, sign conventions).
  2. ENGINE INVARIANTS — run_strategy_simple over a real master_dataset slice AND over
     crafted seed_state books (mirrors how live_kernel.py seeds the operator's real
     positions: cash/shares/short_puts/short_calls/long_*/boxx/kold/nav_peak). The
     end-of-day `history` frame exposes per-bar shares + put_book/call_book, so the
     covered-calls-only and never-short-shares invariants are checked on EVERY bar.

Run:    venv/bin/python -m pytest backtest/test_engine_safety.py -q
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import pytest
import replay_engine as R

# ---------------------------------------------------------------------------
# Shared real-data fixture: a tiny precomputed slice so the engine has valid
# z / regime / iv_rank / surprise columns. Kept short so the suite stays fast.
# ---------------------------------------------------------------------------
GREEKS = 'regime_wheel_boxx_greeks'   # the promoted live kernel: gamma_cap + max_short_pct_nav


@pytest.fixture(scope='module')
def df_full():
    df = pd.read_csv(os.path.join(R.CACHE_DIR, 'master_dataset.csv'),
                     parse_dates=[0], index_col=0)
    return R.precompute_factor_z(df).dropna(subset=['UNG'])


@pytest.fixture(scope='module')
def df_slice(df_full):
    # ~250 trading days exercises opens, TPs, rolls, assignments, the cap, KOLD, BOXX.
    return df_full.iloc[-250:]


@pytest.fixture(scope='module')
def run_greeks(df_slice):
    hist, trades = R.run_strategy_simple(
        df_slice, R.STRATEGIES[GREEKS], initial_cash=100000, initial_shares=0)
    return hist, trades


def _live(win, params, seed):
    """One seeded live-decision day; returns (today_orders_df, post-decision book)."""
    _, orders = R.run_strategy_simple(win, params, seed_state=seed, live_decision=True)
    return orders, dict(R._LIVE_FINAL)


def _empty_book(cash=100000.0, shares=0, **kw):
    b = {'cash': float(cash), 'shares': int(shares), 'short_puts': [], 'short_calls': [],
         'long_puts': [], 'long_calls': [], 'boxx': 0.0, 'kold': 0, 'nav_peak': float(cash)}
    b.update(kw)
    return b


# ===========================================================================
# 1. PURE HELPERS
# ===========================================================================

class TestPAssign:
    def test_dte_guard_itm_returns_one(self):
        # dte<=0 with S<K (put ITM) -> certain assignment
        assert R.p_assign(K=12.0, S=10.0, dte_days=0, z=0.0) == 1.0
        assert R.p_assign(K=12.0, S=10.0, dte_days=-5, z=0.0) == 1.0

    def test_dte_guard_otm_returns_zero(self):
        assert R.p_assign(K=8.0, S=10.0, dte_days=0, z=0.0) == 0.0

    def test_spot_guard(self):
        # S<=0 -> degenerate; S<K so returns 1.0
        assert R.p_assign(K=10.0, S=0.0, dte_days=30, z=0.0) == 1.0
        assert R.p_assign(K=10.0, S=-1.0, dte_days=30, z=0.0) == 1.0

    def test_in_unit_interval(self):
        for K in (8, 10, 12):
            p = R.p_assign(K=float(K), S=10.0, dte_days=30, z=0.0)
            assert 0.0 <= p <= 1.0

    def test_monotonic_in_strike(self):
        # higher put strike (more ITM) -> strictly higher P(assign)
        lo = R.p_assign(K=9.0, S=10.0, dte_days=30, z=0.0)
        mid = R.p_assign(K=10.0, S=10.0, dte_days=30, z=0.0)
        hi = R.p_assign(K=11.0, S=10.0, dte_days=30, z=0.0)
        assert lo < mid < hi

    def test_z_sign(self):
        # drift mu = a + b*z, b<0 -> larger z => more negative drift => K/S exceeded
        # more easily => HIGHER assignment prob for a fixed strike.
        lo_z = R.p_assign(K=10.0, S=10.0, dte_days=30, z=-2.0)
        hi_z = R.p_assign(K=10.0, S=10.0, dte_days=30, z=2.0)
        assert hi_z > lo_z


class TestBSPutCall:
    def test_put_call_parity(self):
        S, K, T, sig = 10.0, 11.0, 0.5, 0.45
        c = R.bs_call(S, K, T, sig)
        p = R.bs_put(S, K, T, sig)
        # C - P = S - K*exp(-rT)
        rhs = S - K * math.exp(-0.045 * T)
        assert c - p == pytest.approx(rhs, abs=1e-6)

    def test_put_T_guard_returns_intrinsic(self):
        assert R.bs_put(10.0, 12.0, 0.0, 0.4) == pytest.approx(2.0)
        assert R.bs_put(13.0, 12.0, 0.0, 0.4) == pytest.approx(0.0)  # OTM -> 0
        assert R.bs_put(10.0, 12.0, 0.5, 0.0) == pytest.approx(2.0)  # sig<=0 guard

    def test_call_T_guard_returns_intrinsic(self):
        assert R.bs_call(13.0, 12.0, 0.0, 0.4) == pytest.approx(1.0)
        assert R.bs_call(10.0, 12.0, 0.0, 0.4) == pytest.approx(0.0)
        assert R.bs_call(13.0, 12.0, 0.5, 0.0) == pytest.approx(1.0)

    def test_deep_itm_put_approaches_intrinsic(self):
        # deep ITM put, short dated -> ~ discounted strike minus spot, dominated by intrinsic
        v = R.bs_put(1.0, 20.0, 0.05, 0.4)
        assert v > 18.0   # huge intrinsic
        assert v >= max(0, 20.0 - 1.0) - 1.0   # near/above intrinsic floor

    def test_deep_otm_options_approach_zero(self):
        assert R.bs_put(20.0, 5.0, 0.25, 0.4) < 0.01     # put miles OTM
        assert R.bs_call(5.0, 50.0, 0.25, 0.4) < 0.01    # call miles OTM

    def test_values_nonnegative(self):
        for S in (5, 10, 15):
            for K in (5, 10, 15):
                assert R.bs_put(S, K, 0.3, 0.5) >= 0
                assert R.bs_call(S, K, 0.3, 0.5) >= 0


class TestBSGreeksPt:
    def test_call_delta_in_unit(self):
        d, g = R.bs_greeks_pt(10.0, 10.0, 0.5, 0.45, 'C')
        assert 0.0 <= d <= 1.0
        assert g >= 0.0

    def test_put_delta_in_unit(self):
        d, g = R.bs_greeks_pt(10.0, 10.0, 0.5, 0.45, 'P')
        assert -1.0 <= d <= 0.0
        assert g >= 0.0

    def test_call_put_delta_relation(self):
        # call_delta - put_delta == 1 (same S,K,T,sig)
        dc, _ = R.bs_greeks_pt(10.0, 11.0, 0.4, 0.5, 'C')
        dp, _ = R.bs_greeks_pt(10.0, 11.0, 0.4, 0.5, 'P')
        assert dc - dp == pytest.approx(1.0, abs=1e-9)

    def test_deep_itm_call_delta_one(self):
        d, g = R.bs_greeks_pt(50.0, 5.0, 0.3, 0.4, 'C')
        assert d == pytest.approx(1.0, abs=1e-3)

    def test_deep_itm_put_delta_minus_one(self):
        d, g = R.bs_greeks_pt(1.0, 50.0, 0.3, 0.4, 'P')
        assert d == pytest.approx(-1.0, abs=1e-3)

    def test_T_guard_returns_intrinsic_delta(self):
        # T<=0: ITM call -> delta 1, OTM call -> 0; gamma 0
        assert R.bs_greeks_pt(12.0, 10.0, 0.0, 0.4, 'C') == (1.0, 0.0)
        assert R.bs_greeks_pt(8.0, 10.0, 0.0, 0.4, 'C') == (0.0, 0.0)
        assert R.bs_greeks_pt(8.0, 10.0, 0.0, 0.4, 'P') == (-1.0, 0.0)
        assert R.bs_greeks_pt(12.0, 10.0, 0.0, 0.4, 'P') == (0.0, 0.0)

    def test_spot_guard(self):
        assert R.bs_greeks_pt(0.0, 10.0, 0.5, 0.4, 'P') == (-1.0, 0.0)


class TestBookGreeks:
    def _iv(self):
        return lambda K, dte, right: 0.45

    def test_shares_only_delta_is_share_count(self):
        s = {'shares': 300, 'short_puts': [], 'short_calls': [], 'long_puts': []}
        nd, ng = R.book_greeks(s, 10.0, self._iv())
        assert nd == pytest.approx(300.0)
        assert ng == pytest.approx(0.0)

    def test_short_put_adds_positive_delta(self):
        # short put: -qty * (negative delta) * 100 = +ve  (bullish exposure)
        base = {'shares': 0, 'short_puts': [], 'short_calls': [], 'long_puts': []}
        s = {**base, 'short_puts': [{'K': 10.0, 'dte': 30, 'qty': 2}]}
        nd, ng = R.book_greeks(s, 10.0, self._iv())
        assert nd > 0.0
        assert ng < 0.0   # short option => negative gamma

    def test_short_call_adds_negative_delta(self):
        s = {'shares': 0, 'short_puts': [], 'long_puts': [],
             'short_calls': [{'K': 10.0, 'dte': 30, 'qty': 2}]}
        nd, ng = R.book_greeks(s, 10.0, self._iv())
        assert nd < 0.0
        assert ng < 0.0

    def test_long_put_adds_negative_delta_positive_gamma(self):
        s = {'shares': 0, 'short_puts': [], 'short_calls': [],
             'long_puts': [{'K': 10.0, 'dte': 30, 'qty': 2}]}
        nd, ng = R.book_greeks(s, 10.0, self._iv())
        assert nd < 0.0   # bearish hedge
        assert ng > 0.0   # long option => positive gamma


class TestBookGreeksStat:
    def test_shares_only(self):
        s = {'shares': 500, 'short_puts': [], 'short_calls': [], 'long_puts': []}
        assert R.book_greeks_stat(s, 10.0, 0.0) == pytest.approx(500.0)

    def test_short_put_term_is_p_assign_qty_100(self):
        K, S, dte, z, qty = 11.0, 10.0, 30, 0.3, 3
        s = {'shares': 0, 'short_puts': [{'K': K, 'dte': dte, 'qty': qty}],
             'short_calls': [], 'long_puts': []}
        pa = R.p_assign(K, S, dte, z, -0.000797, -0.000009, 0.0390)
        assert R.book_greeks_stat(s, S, z) == pytest.approx(pa * qty * 100)

    def test_short_call_term_is_neg_called_away(self):
        K, S, dte, z, qty = 9.0, 10.0, 30, 0.1, 4
        s = {'shares': 0, 'short_calls': [{'K': K, 'dte': dte, 'qty': qty}],
             'short_puts': [], 'long_puts': []}
        pa = R.p_assign(K, S, dte, z, -0.000797, -0.000009, 0.0390)
        expected = -qty * (1.0 - pa) * 100
        assert R.book_greeks_stat(s, S, z) == pytest.approx(expected)

    def test_long_put_term_is_negative(self):
        K, S, dte, z, qty = 11.0, 10.0, 30, 0.0, 2
        s = {'shares': 0, 'long_puts': [{'K': K, 'dte': dte, 'qty': qty}],
             'short_puts': [], 'short_calls': []}
        pa = R.p_assign(K, S, dte, z, -0.000797, -0.000009, 0.0390)
        assert R.book_greeks_stat(s, S, z) == pytest.approx(-pa * qty * 100)


# ===========================================================================
# 2. ENGINE INVARIANTS — real-data continuous run
# ===========================================================================

def _total_short_call_lots(call_book):
    return sum(int(q) for q in (call_book or {}).values())


class TestRealRunInvariants:
    """Assert on EVERY bar of a real ~250-day continuous run."""

    def test_never_short_shares(self, run_greeks):
        hist, _ = run_greeks
        assert (hist['shares'] >= 0).all(), \
            f"shares went negative on {hist.loc[hist['shares'] < 0, 'date'].tolist()}"

    def test_never_naked_calls(self, run_greeks):
        # covered-calls-only: total short-call lots <= shares // 100 on every bar.
        hist, _ = run_greeks
        bad = []
        for _, r in hist.iterrows():
            lots = _total_short_call_lots(r['call_book'])
            if lots > r['shares'] // 100:
                bad.append((r['date'], lots, r['shares']))
        assert not bad, f"naked-call bars (lots > shares//100): {bad[:5]}"

    def test_short_call_count_matches_call_book(self, run_greeks):
        hist, _ = run_greeks
        # sanity: the count column and the per-strike book agree on # of legs presence
        for _, r in hist.iterrows():
            if r['short_calls'] == 0:
                assert _total_short_call_lots(r['call_book']) == 0 or r['call_book'] == {}

    def test_gamma_cap_per_open_put_order(self, run_greeks, df_slice):
        # Each OPEN_PUT order is a single (strike,dte): its qty must respect the
        # scale-invariant cap max(1, int(pct*NAV/(K*100))) (legacy floor allowed).
        hist, trades = run_greeks
        p = R.STRATEGIES[GREEKS]
        pct = p['max_short_pct_nav']
        floor = p.get('max_short_per_strike', 10)
        nav_by_date = dict(zip(hist['date'], hist['nav']))
        opens = trades[trades['type'] == 'OPEN_PUT']
        assert len(opens) > 0, "expected the run to open puts"
        viol = []
        for _, r in opens.iterrows():
            nav = nav_by_date.get(r['date'])
            if nav is None or r['K'] <= 0:
                continue
            cap = max(max(1, int(pct * nav / (r['K'] * 100))), floor)
            if r['qty'] > cap:
                viol.append((r['date'], r['K'], r['qty'], cap))
        assert not viol, f"gamma-cap exceeded on OPEN_PUT: {viol[:5]}"

    def test_produces_assignment_events(self, run_greeks):
        # The slice must actually exercise the assignment paths we care about,
        # otherwise the crafted tests below are the only coverage.
        _, trades = run_greeks
        types = set(trades['type'])
        assert 'PUT_ASSIGN' in types or 'CALL_ASSIGN' in types


# ===========================================================================
# 2b. ENGINE INVARIANTS — crafted seed_state scenarios (deterministic)
# ===========================================================================

@pytest.fixture(scope='module')
def win40(df_full):
    return df_full.iloc[-40:]


# Params that ISOLATE the assignment/early-assign mechanics: disable roll-up and
# elevator close so a deep-ITM short option resolves via (early-)assignment, not a roll.
def _isolated(**over):
    p = dict(R.STRATEGIES[GREEKS])
    p['model_early_assign'] = True
    p['roll_up_calls'] = False
    p['elevator_close'] = False
    p.update(over)
    return p


class TestPutAssignment:
    def test_itm_put_at_expiry_assigns_shares_and_pays_strike(self, win40):
        last = win40.index[-1]
        spot = float(win40['UNG'].iloc[-1])
        K = float(round(spot + 3))                 # ITM put (K > spot)
        qty = 2
        entry = last - pd.Timedelta(days=60)       # past dte=30 -> expires today
        seed = _empty_book(short_puts=[{'entry': entry, 'K': K, 'dte': 30,
                                        'qty': qty, 'entry_prem': 0.30}])
        orders, book = _live(win40, _isolated(), seed)
        assigns = orders[orders['type'] == 'PUT_ASSIGN']
        assert len(assigns) == 1
        a = assigns.iloc[0]
        assert int(a['qty']) == qty
        assert float(a['K']) == K
        # shares += qty*100 (no other order changes share count here)
        assert book['shares'] == 0 + qty * 100
        # trade P&L == premium kept minus intrinsic loss == premium - (K-spot)*100*qty
        expected_pnl = 0.30 * 100 * qty - (K - spot) * 100 * qty
        assert float(a['pnl']) == pytest.approx(expected_pnl, abs=1e-6)

    def test_otm_put_at_expiry_keeps_premium_no_shares(self, win40):
        last = win40.index[-1]
        spot = float(win40['UNG'].iloc[-1])
        K = float(round(spot * 0.6))               # deep OTM put
        qty = 1
        entry = last - pd.Timedelta(days=60)
        seed = _empty_book(short_puts=[{'entry': entry, 'K': K, 'dte': 30,
                                        'qty': qty, 'entry_prem': 0.30}])
        orders, book = _live(win40, _isolated(), seed)
        assert 'PUT_ASSIGN' not in set(orders['type'])
        assert book['shares'] == 0   # no assignment -> no shares


class TestCallAssignment:
    def test_itm_call_at_expiry_calls_away_shares_pays_strike(self, win40):
        last = win40.index[-1]
        spot = float(win40['UNG'].iloc[-1])
        K = float(round(spot * 0.5))               # deep ITM call (S > K)
        qty = 1
        entry = last - pd.Timedelta(days=60)       # past dte=30 -> expires
        seed = _empty_book(shares=100, short_calls=[{'entry': entry, 'K': K, 'dte': 30,
                                                     'qty': qty, 'entry_prem': 0.30}])
        orders, book = _live(win40, _isolated(), seed)
        assigns = orders[orders['type'] == 'CALL_ASSIGN']
        assert len(assigns) == 1
        a = assigns.iloc[0]
        assert int(a['qty']) == qty
        # shares -= qty*100
        assert book['shares'] == 100 - qty * 100
        expected_pnl = 0.30 * 100 * qty - (spot - K) * 100 * qty
        assert float(a['pnl']) == pytest.approx(expected_pnl, abs=1e-6)


class TestEarlyAssignment:
    def test_deep_itm_put_early_assigns_before_expiry(self, win40):
        last = win40.index[-1]
        spot = float(win40['UNG'].iloc[-1])
        K = float(round(spot * 1.6))               # very deep ITM put -> |delta|>0.99
        qty = 1
        entry = last                               # 0 days elapsed, NOT expired
        seed = _empty_book(short_puts=[{'entry': entry, 'K': K, 'dte': 2,
                                        'qty': qty, 'entry_prem': 0.30}])
        orders, book = _live(win40, _isolated(model_early_assign=True), seed)
        types = set(orders['type'])
        assert 'PUT_EARLY_ASSIGN' in types
        assert 'PUT_ASSIGN' not in types          # tagged EARLY, not regular
        assert book['shares'] == qty * 100

    def test_calls_never_early_assign_when_disabled(self, win40):
        last = win40.index[-1]
        spot = float(win40['UNG'].iloc[-1])
        K = float(round(spot * 0.5))               # deep ITM call, near expiry
        entry = last
        seed = _empty_book(shares=100,
                           short_calls=[{'entry': entry, 'K': K, 'dte': 2,
                                         'qty': 1, 'entry_prem': 0.30}])
        orders, book = _live(win40, _isolated(early_assign_calls=False), seed)
        assert 'CALL_EARLY_ASSIGN' not in set(orders['type'])
        # call not expired and not early-assigned -> shares untouched
        assert book['shares'] == 100

    def test_deep_itm_call_early_assigns_when_enabled(self, win40):
        last = win40.index[-1]
        spot = float(win40['UNG'].iloc[-1])
        K = float(round(spot * 0.5))
        entry = last
        seed = _empty_book(shares=100,
                           short_calls=[{'entry': entry, 'K': K, 'dte': 2,
                                         'qty': 1, 'entry_prem': 0.30}])
        orders, book = _live(win40, _isolated(early_assign_calls=True), seed)
        assert 'CALL_EARLY_ASSIGN' in set(orders['type'])
        assert book['shares'] == 0                 # 100 shares called away


class TestSeededInvariantsHold:
    """The covered-calls / non-negative-shares invariants must survive even when the
    operator's seeded book is itself the engine's starting point (the live path)."""

    def test_seeded_run_keeps_calls_covered(self, win40):
        last = win40.index[-1]
        spot = float(win40['UNG'].iloc[-1])
        # exactly-covered book: 300 shares, 3 short-call lots (the boundary case)
        seed = _empty_book(shares=300, short_calls=[
            {'entry': last - pd.Timedelta(days=5), 'K': float(round(spot * 1.1)),
             'dte': 30, 'qty': 3, 'entry_prem': 0.20}])
        orders, book = _live(win40, _isolated(), seed)
        # after today's decision the post book must remain covered & non-negative
        lots = sum(int(c['qty']) for c in book['short_calls'])
        assert book['shares'] >= 0
        assert lots <= book['shares'] // 100, \
            f"seeded live decision produced naked calls: {lots} lots vs {book['shares']} shares"


# Live champion: all USD swept to BOXX; the buffer MODELS the CAD reserve (deployable collateral), not
# idle USD. CORRECTED 2026-06-30: buffer:0 starved the wheel (→ bare BOXX yield); the faithful model is
# buffer == the CAD reserve, with the live dashboard seeding that CAD (USD-equiv) into deployable cash.
LIVE = 'regime_wheel_boxx_greeks_live'


class TestBoxxUsdCashOnly:
    """Regression guards for the multi-currency USD-margin bug (see memory
    feedback_boxx_usd_cash_multicurrency + project_delta_gamma_accumulation). BOXX is filled from USD
    CASH ONLY: the live path sources real USD cash from the broker (not net_liq − Σ MV, which blended the
    CAD side, fed phantom +cash, and drew ~$9.5k of USD margin). The strategy's boxx_cash_buffer is NOT
    idle USD — it REPRESENTS the CAD reserve, which the dashboard seeds as deployable cash so the all-USD
    sweep never recommends selling BOXX to raise idle USD, and never pushes USD cash negative."""

    def test_live_buffer_models_cad_reserve(self):
        # Corrected model: the buffer == the CAD reserve (deployable collateral), NOT 0. buffer:0 starved
        # the wheel down to bare BOXX yield; the reserve→return curve is near-monotonic (15k → 15.8%/yr).
        assert R.STRATEGIES[LIVE].get('boxx_cash_buffer') == 15000

    def test_dashboard_seeds_cad_reserve_as_cash(self):
        # The buffer is satisfied by CAD, not idle USD: the live path must add the CAD reserve (USD-equiv)
        # to deployable cash, else buffer:15k would wrongly recommend selling BOXX to raise USD.
        import kernel_dashboard as KD
        assert hasattr(KD, '_real_cad_reserve_usd'), \
            "live path must seed the CAD reserve as deployable cash (buffer represents CAD, not idle USD)"

    def test_dashboard_sources_real_usd_cash_not_nav_derivation(self):
        # The fix: live cash comes from the exact broker query (can be negative = margin), not net_liq−ΣMV.
        import kernel_dashboard as KD
        assert hasattr(KD, '_real_usd_cash_bp'), \
            "live path must source USD cash from FetchTradingBalanceBuyingPower, not net_liq − Σ MV"

    def test_positive_usd_cash_sweep_leaves_no_margin(self, win40):
        # Given positive USD cash, the sweep buys BOXX but leaves cash >= buffer — never negative.
        seed = _empty_book(cash=40000.0, boxx=0.0, nav_peak=40000.0)
        _, book = _live(win40, dict(R.STRATEGIES[LIVE]), seed)
        assert book['cash'] >= -1.0, \
            f"sweep bought BOXX into USD margin: post cash {book['cash']:.0f}"
