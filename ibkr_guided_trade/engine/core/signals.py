"""Pure functions for z-score, regime classification, factor breakdown.

Used identically by live engine and backtest. No I/O, no side effects.
"""
from typing import List, Dict, Any, Optional
from .params import Params
from .state import PortfolioState


def compute_z_components(state: PortfolioState, params: Params) -> List[Dict[str, Any]]:
    """Compute per-factor z contributions. Returns list of dicts.
    Used by BOTH z_score() and the factor-breakdown display."""
    out = []

    # Storage level
    if state.storage_bcf:
        s_z = (state.storage_bcf - 2500) / 500
        out.append({
            'name': 'Storage Level',
            'value': f"{state.storage_bcf:.0f} Bcf",
            'z_raw': -s_z,                          # raw signed z
            'z_contrib': -s_z * params.w_storage_level,  # weighted
            'weight': params.w_storage_level,
        })

    # Days of supply
    if state.days_supply and state.days_supply > 0:
        ds_z = (state.days_supply - 31) / 5
        out.append({
            'name': 'Days of Supply',
            'value': f"{state.days_supply:.1f} days",
            'z_raw': -ds_z,
            'z_contrib': -ds_z * params.w_days_supply,
            'weight': params.w_days_supply,
        })

    # NG trend vs MA200
    if state.ng_trend is not None:
        out.append({
            'name': 'NG Trend vs MA200',
            'value': f"{state.ng_trend*100:+.1f}%",
            'z_raw': -state.ng_trend * 3,
            'z_contrib': -state.ng_trend * 3 * params.w_ng_trend,
            'weight': params.w_ng_trend,
        })

    # VIX
    if state.vix:
        vix_norm = (state.vix - 20) / 10
        out.append({
            'name': 'VIX',
            'value': f"{state.vix:.1f}",
            'z_raw': -vix_norm,
            'z_contrib': -vix_norm * params.w_vix,
            'weight': params.w_vix,
        })

    # Oil/NG ratio
    if state.cl_price and state.ng_price > 0:
        ratio = state.cl_price / state.ng_price
        r_z = (ratio - 25) / 10
        out.append({
            'name': 'Oil/NG Ratio',
            'value': f"{ratio:.1f}",
            'z_raw': r_z,
            'z_contrib': r_z * params.w_oil_ng_ratio,
            'weight': params.w_oil_ng_ratio,
        })

    return out


def z_score(state: PortfolioState, params: Params) -> float:
    """Composite z-score normalized to full weight coverage."""
    comps = compute_z_components(state, params)
    if not comps:
        return 0.0
    target_weight = (params.w_storage_level + params.w_days_supply +
                     params.w_ng_trend + params.w_vix + params.w_oil_ng_ratio)
    weight_sum = sum(c['weight'] for c in comps)
    scale = target_weight / weight_sum if weight_sum > 0 else 1.0
    return sum(c['z_contrib'] for c in comps) * scale


def regime(z: float, params: Params) -> str:
    """Classify z-score into a regime label."""
    if z > params.z_extreme_cheap: return 'EXTREME_CHEAP'
    if z > params.z_cheap: return 'CHEAP'
    if z > params.z_neutral_lower: return 'NEUTRAL'
    if z > params.z_extreme_rich: return 'RICH'
    return 'EXTREME_RICH'


def regime_for_state(state: PortfolioState, params: Params) -> str:
    """Convenience: state → regime."""
    return regime(z_score(state, params), params)
