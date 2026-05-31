"""Generate trade candidates from a state, parameterized by Params.

A Candidate is just a dict describing a potential trade. Pure function
of (state, params) — no I/O.
"""
from typing import List, Dict, Any
from .params import Params
from .state import PortfolioState
from .signals import regime_for_state
from .options import bs_put, bs_call


def generate_candidates(state: PortfolioState, params: Params) -> List[Dict[str, Any]]:
    """Return candidate trades for the current state + regime."""
    r = regime_for_state(state, params)
    spot = state.spot
    iv = max(0.01, state.iv)
    cands = []

    # === Regime-driven strategy params ===
    otm_put_map = {
        'EXTREME_CHEAP': params.otm_put_extreme_cheap,
        'CHEAP': params.otm_put_cheap,
        'NEUTRAL': params.otm_put_neutral,
        'RICH': params.otm_put_rich,
        'EXTREME_RICH': params.otm_put_extreme_rich,
    }
    qty_put_map = {
        'EXTREME_CHEAP': params.put_qty_extreme_cheap,
        'CHEAP': params.put_qty_cheap,
        'NEUTRAL': params.put_qty_neutral,
        'RICH': params.put_qty_rich,
        'EXTREME_RICH': params.put_qty_extreme_rich,
    }
    otm_call_map = {
        'EXTREME_CHEAP': params.otm_call_extreme_cheap,
        'CHEAP': params.otm_call_cheap,
        'NEUTRAL': params.otm_call_neutral,
        'RICH': params.otm_call_rich,
        'EXTREME_RICH': params.otm_call_extreme_rich,
    }

    otm_put = otm_put_map.get(r, 0.10)
    put_qty = qty_put_map.get(r, 3)
    otm_call = otm_call_map.get(r, 0.05)

    # === SELL PUT candidate ===
    if put_qty > 0 and otm_put < 0.5:
        K = round(spot * (1 - otm_put), 1)
        prem = bs_put(spot, K, 30/365, iv)
        if prem > 0.05:
            cands.append({
                'type': 'OPEN_PUT',
                'action': f"Sell {put_qty}x ${K}P 30d",
                'strike': K,
                'dte': 30,
                'qty': put_qty,
                'premium': prem,
                'cash_delta': prem * 100 * put_qty - put_qty * params.option_spread_per_share * 100,
                'regime': r,
            })

    # === COVERED CALL candidate ===
    if state.shares >= 100:
        call_qty = min(params.call_qty, state.shares // 100)
        K_c = round(spot * (1 + otm_call), 1)
        prem_c = bs_call(spot, K_c, 30/365, iv)
        if prem_c > 0.05 and call_qty > 0:
            cands.append({
                'type': 'COVERED_CALL',
                'action': f"Sell {call_qty}x ${K_c}C 30d covered",
                'strike': K_c,
                'dte': 30,
                'qty': call_qty,
                'premium': prem_c,
                'cash_delta': prem_c * 100 * call_qty - call_qty * params.option_spread_per_share * 100,
                'regime': r,
            })

    # === BOXX candidate ===
    excess = state.cash - params.min_cash_buffer
    if excess > params.boxx_deploy_threshold:
        deploy = excess * params.boxx_deploy_fraction
        n_boxx = int(deploy / 117)
        if n_boxx >= 10:
            cost = n_boxx * 117
            if not params.never_negative_cash or state.cash - cost >= 0:
                cands.append({
                    'type': 'BUY_BOXX',
                    'action': f"Buy {n_boxx} BOXX @ $117",
                    'qty': n_boxx,
                    'cash_delta': -cost,
                    'annual_yield': cost * params.boxx_yield,
                    'regime': r,
                })

    # === BEARISH STACK in EXTREME_RICH ===
    if r == 'EXTREME_RICH' and params.bearish_stack_enabled:
        K_p = round(spot * (1 - params.long_put_otm_pct), 1)
        cost = bs_put(spot, K_p, params.long_put_dte/365, iv)
        if cost > 0.05:
            total_cost = cost * 100 * params.long_put_qty
            cands.append({
                'type': 'BUY_PUT',
                'action': f"Buy {params.long_put_qty}x ${K_p}P {params.long_put_dte}d (bearish hedge)",
                'strike': K_p,
                'dte': params.long_put_dte,
                'qty': params.long_put_qty,
                'cost_per_share': cost,
                'cash_delta': -total_cost,
                'regime': r,
            })

        if state.kold_spot > 0 and state.nlv > 0:
            target_qty = int(state.nlv * params.kold_nav_fraction / state.kold_spot)
            if target_qty > 5 and state.cash > target_qty * state.kold_spot:
                cands.append({
                    'type': 'BUY_KOLD',
                    'action': f"Buy {target_qty} KOLD @ ${state.kold_spot:.2f} (tactical short NG)",
                    'qty': target_qty,
                    'cash_delta': -target_qty * state.kold_spot,
                    'regime': r,
                })

    return cands
