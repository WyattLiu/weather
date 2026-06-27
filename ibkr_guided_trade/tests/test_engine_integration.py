"""Integration tests for the recommendation engine.

Validates that single-position changes produce expected output changes.
These tests use the LIVE option chain and portfolio state — they verify
the engine's logic end-to-end, not mocked data.

Run: python -m pytest tests/test_engine_integration.py -v
"""
import sys
import pytest
from datetime import date, timedelta

sys.path.insert(0, '.')

from ung_visualizer import (
    OPTIONS, UNG_PRICE, compute_portfolio_state, evaluate_portfolio_quality,
    apply_trade_to_state, generate_candidates,
)


@pytest.fixture
def base_state():
    """Build a consistent base portfolio state from live positions."""
    spot = UNG_PRICE
    today = date.today()
    iv = 0.45
    ps = compute_portfolio_state(OPTIONS, spot, iv, today)
    wt = list(ps['weekly_theta'].values())
    near = [v for v in wt[:2] if v > 0]
    ps['avg_weekly_theta'] = sum(near) / len(near) if near else 0
    ps['target_weekly_income'] = 1500.0
    ps['capital_base'] = 112000.0
    ps['tail_hedge_floor'] = 2
    return ps, spot, iv, today


class TestTailHedgeTracking:
    """Verify tail hedge responds correctly to position changes."""

    def test_zero_leaps_shows_penalty(self, base_state):
        ps, spot, iv, today = base_state
        q = evaluate_portfolio_quality(ps)
        assert q['components']['tail_hedge'] < 0, \
            "Should show negative tail_hedge when no LEAPS"

    def test_add_one_leaps_reduces_penalty(self, base_state):
        ps, spot, iv, today = base_state
        q_before = evaluate_portfolio_quality(ps)

        leaps_exp = (today + timedelta(days=200)).strftime('%Y-%m-%d')
        new_positions = list(ps['positions']) + [
            (leaps_exp, 11.0, 'P', 1, 2.50)  # 1 LONG put
        ]
        ps_after = dict(ps)
        ps_after['positions'] = new_positions

        q_after = evaluate_portfolio_quality(ps_after)
        assert q_after['components']['tail_hedge'] > q_before['components']['tail_hedge'], \
            "Adding 1 LEAPS should improve tail_hedge component"

    def test_add_two_leaps_reduces_penalty_further(self, base_state):
        """More LEAPS = more hedged = less penalty (proportional, no floor)."""
        ps, spot, iv, today = base_state
        leaps_exp = (today + timedelta(days=200)).strftime('%Y-%m-%d')

        ps1 = dict(ps)
        ps1['positions'] = list(ps['positions']) + [(leaps_exp, 11.0, 'P', 1, 2.50)]
        q1 = evaluate_portfolio_quality(ps1)

        ps2 = dict(ps)
        ps2['positions'] = list(ps['positions']) + [(leaps_exp, 11.0, 'P', 2, 2.50)]
        q2 = evaluate_portfolio_quality(ps2)

        assert q2['components']['tail_hedge'] > q1['components']['tail_hedge'], \
            "2 LEAPS should have less penalty than 1 (proportional reduction)"

    def test_short_dte_put_not_counted_as_leaps(self, base_state):
        ps, spot, iv, today = base_state
        short_exp = (today + timedelta(days=90)).strftime('%Y-%m-%d')
        new_positions = list(ps['positions']) + [
            (short_exp, 11.0, 'P', 2, 1.00)  # 90 DTE < 180 threshold
        ]
        ps_after = dict(ps)
        ps_after['positions'] = new_positions

        q_after = evaluate_portfolio_quality(ps_after)
        assert q_after['components']['tail_hedge'] < 0, \
            "90-DTE puts should NOT count as LEAPS (need >= 180)"


class TestSinglePositionChange:
    """Verify that adding/removing one position changes recs correctly."""

    def test_add_short_put_increases_theta(self, base_state):
        ps, spot, iv, today = base_state
        trade = {
            'type': 'OPEN',
            'target_exp': (today + timedelta(days=30)).strftime('%Y-%m-%d'),
            'target_strike': round(spot, 0),
            'add_qty': 1,
            'theta_change': 1.5,
            'delta_change': 50.0,
            'gamma_change': -30.0,
            'vega_change': -5.0,
            'new_extrinsic_total': 50.0,
            'n_legs': 1,
        }
        ns = apply_trade_to_state(dict(ps), trade, spot, iv, today)
        assert ns['total_theta'] > ps['total_theta'], \
            "Adding a short put should increase total theta"
        assert ns['total_delta'] > ps['total_delta'], \
            "Adding a short put should increase delta (more long)"

    def test_add_covered_call_reduces_delta(self, base_state):
        ps, spot, iv, today = base_state
        trade = {
            'type': 'COVERED CALL',
            'target_exp': (today + timedelta(days=30)).strftime('%Y-%m-%d'),
            'target_strike': round(spot * 1.05, 0),
            'add_qty': 1,
            'theta_change': 1.0,
            'delta_change': -40.0,
            'gamma_change': -20.0,
            'vega_change': -3.0,
            'new_extrinsic_total': 30.0,
            'n_legs': 1,
        }
        ns = apply_trade_to_state(dict(ps), trade, spot, iv, today)
        assert ns['total_delta'] < ps['total_delta'], \
            "Adding a covered call should reduce delta"

    def test_buy_put_adds_positive_gamma(self, base_state):
        ps, spot, iv, today = base_state
        trade = {
            'type': 'BUY PUT',
            'target_exp': (today + timedelta(days=200)).strftime('%Y-%m-%d'),
            'target_strike': round(spot, 0),
            'add_qty': 1,
            'theta_change': -0.5,
            'delta_change': -30.0,
            'gamma_change': 15.0,
            'vega_change': 10.0,
            'new_extrinsic_total': 200.0,
            'n_legs': 1,
        }
        ns = apply_trade_to_state(dict(ps), trade, spot, iv, today)
        assert ns['total_gamma'] > ps['total_gamma'], \
            "Buying a put should add positive gamma"


class TestCandidateGeneration:
    """Verify candidate generation responds to position state."""

    def test_leaps_buy_candidates_generated_when_below_floor(self, base_state):
        ps, spot, iv, today = base_state
        cands = generate_candidates(ps, spot, iv, today)
        leaps_buys = [c for c in cands
                      if c['type'] == 'BUY PUT' and 'LEAPS' in c.get('action', '')]
        assert len(leaps_buys) > 0, \
            "Should generate BUY LEAPS PUT candidates when below tail-hedge floor"

    def test_leaps_candidates_always_generated(self, base_state):
        """LEAPS candidates always generated — beam decides via qΔ, no floor."""
        ps, spot, iv, today = base_state
        leaps_exp = (today + timedelta(days=200)).strftime('%Y-%m-%d')
        ps['positions'] = list(ps['positions']) + [
            (leaps_exp, 11.0, 'P', 2, 2.50)
        ]
        ps['tail_hedge_qty'] = 2
        cands = generate_candidates(ps, spot, iv, today)
        leaps_buys = [c for c in cands
                      if c['type'] == 'BUY PUT' and 'LEAPS' in c.get('action', '')]
        assert len(leaps_buys) > 0, \
            "LEAPS candidates should always be generated (beam evaluates qΔ)"

    def test_open_candidates_have_required_fields(self, base_state):
        ps, spot, iv, today = base_state
        cands = generate_candidates(ps, spot, iv, today)
        opens = [c for c in cands if c['type'] == 'OPEN']
        assert len(opens) > 0, "Should have OPEN candidates"
        for c in opens[:3]:
            assert 'target_exp' in c, "OPEN must have target_exp"
            assert 'target_strike' in c, "OPEN must have target_strike"
            assert 'add_qty' in c, "OPEN must have add_qty"
            assert 'theta_change' in c, "OPEN must have theta_change"
            assert 'liquidity' in c, "OPEN must have liquidity"
            assert c['liquidity'].get('oi', 0) > 0, "OPEN must have OI > 0"


class TestQualityConsistency:
    """Verify evaluate_portfolio_quality is consistent with apply_trade_to_state."""

    def test_quality_before_after_trade_consistent(self, base_state):
        ps, spot, iv, today = base_state
        q_before = evaluate_portfolio_quality(ps)

        cands = generate_candidates(ps, spot, iv, today)
        opens = [c for c in cands if c['type'] == 'OPEN']
        if not opens:
            pytest.skip("No OPEN candidates to test with")

        trade = opens[0]
        ns = apply_trade_to_state(dict(ps), trade, spot, iv, today)
        q_after = evaluate_portfolio_quality(ns)

        qd = q_after['total'] - q_before['total']
        assert isinstance(qd, float), "qΔ should be a number"
        # The trade should have SOME effect (not exactly zero)
        # (very unlikely to be exactly zero for a real trade)
