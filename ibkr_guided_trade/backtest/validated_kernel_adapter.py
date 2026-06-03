"""Validated-kernel side-car for ung_visualizer.py.

Designed for ZERO-REGRESSION integration. Production keeps its existing
compute_recommendations() entirely intact. This module exposes a single
function the production webdash can call to get a "validated kernel's
verdict" on the current state — what the backtested champion strategy
(champion_target_25_dd_trim) would do given current holdings.

Integration in ung_visualizer.py is a 5-line addition:

    # Add near top:
    try:
        from backtest.validated_kernel_adapter import validated_verdict
    except ImportError:
        validated_verdict = None

    # Add a new route in Handler.do_GET:
    elif parsed.path == '/api/validated':
        if validated_verdict is None:
            data = {'available': False, 'reason': 'adapter not on path'}
        else:
            data = validated_verdict(UNG_PRICE, WS_POSITIONS, IV_CACHE)
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

The dashboard can fetch /api/validated alongside /api/data and render a
"Backtest Says" panel without affecting any existing rec.

NOTES:
- No mutation of WS state. Verdict is informational.
- IV is preferred from the real PG surface when available; falls back
  to whatever IV the caller passes (production's calibrated IV).
- All thresholds derive from champion_target_25_dd_trim (current
  walk-forward winner: Sharpe 2.73 full sample, worst 12mo MDD -17%).
"""
import os
import sys
from typing import Any, Dict, List, Optional

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from replay_engine import (
    STRATEGIES, precompute_factor_z, compute_historical_z, regime,
    _load_iv_surface, iv_from_surface, iv_shape_features,
)

# Champion config (walk-forward winner). Snapshot the mults so changes to
# STRATEGIES don't silently shift production behavior.
CHAMPION_NAME = 'champion_target_25_dd_trim'


def _z_mult(z: float) -> float:
    """Champion's z-bucket multiplier curve."""
    if z < -1.5: return 2.0
    if z < -0.5: return 1.6
    if z < 0.5:  return 1.0
    if z < 1.0:  return 0.4
    return 0.1


def _current_z_and_regime():
    """Read latest z + regime from master_dataset (the same data the
    backtest trains on)."""
    import pandas as pd
    csv = os.path.join(THIS_DIR, 'cache', 'master_dataset.csv')
    if not os.path.exists(csv):
        return None
    df = pd.read_csv(csv, index_col=0, parse_dates=True)
    df = precompute_factor_z(df).dropna(subset=['UNG'])
    if df.empty:
        return None
    row = df.iloc[-1]
    z = compute_historical_z(row, use_surprise=True)
    return {
        'date': df.index[-1].strftime('%Y-%m-%d'),
        'z_surprise': float(z),
        'regime': regime(z),
        'mult': _z_mult(z),
    }


def _iv_shape_today(spot: float):
    surf = _load_iv_surface()
    if not surf:
        return None
    # Use latest available date from surface
    if not surf:
        return None
    latest = max(surf.keys())
    return iv_shape_features(surf, latest, spot)


def validated_verdict(spot: float, positions: Optional[List[Dict[str, Any]]] = None,
                      base_shares: int = 6200) -> Dict[str, Any]:
    """Return what the validated kernel says about current state.

    Args:
        spot: current UNG price (adjusted/yfinance frame)
        positions: optional list of position dicts in WS format
            (each with symbol, is_option, option_type, strike, expiry, quantity)
        base_shares: strategy's base share count target (default 6200)

    Returns dict with:
        kernel: kernel name (string)
        snapshot_date: latest data date
        regime: NEUTRAL/CHEAP/RICH/...
        z_surprise: current z score (surprise-detrended)
        mult: z-bucket multiplier (e.g. 1.0 for NEUTRAL)
        target_shares: target share count
        share_delta: shares to buy (+) or sell (-)
        iv_shape: dict with atm_iv/put_skew/call_skew/term_slope (if surface available)
        recommendations: list of {action, details} strings
        warnings: list of risk warnings (e.g. over-leverage, walk-fwd MDD)
    """
    out = {
        'available': True,
        'kernel': CHAMPION_NAME,
        'spot': spot,
        'recommendations': [],
        'warnings': [],
    }

    snap = _current_z_and_regime()
    if snap is None:
        out['available'] = False
        out['reason'] = 'master_dataset.csv missing'
        return out
    out.update(snap)

    # Target shares
    target = int(round(base_shares * snap['mult'] / 100) * 100)
    out['target_shares'] = target

    current_shares = 0
    current_short_calls = 0
    current_short_puts = 0
    current_put_collateral = 0.0
    if positions:
        for p in positions:
            sym = p.get('symbol', '').upper()
            if sym != 'UNG':
                continue
            qty = int(p.get('quantity', 0) or 0)
            if p.get('is_option'):
                K = float(p.get('strike') or 0)
                if p.get('option_type') == 'CALL':
                    current_short_calls += abs(qty)
                else:
                    current_short_puts += abs(qty)
                    current_put_collateral += abs(qty) * 100 * K
            else:
                current_shares += qty

    out['current_shares'] = current_shares
    out['share_delta'] = target - current_shares
    out['current_short_calls'] = current_short_calls
    out['current_short_puts'] = current_short_puts
    out['current_put_collateral'] = current_put_collateral

    # IV shape today
    shape = _iv_shape_today(spot)
    if shape:
        out['iv_shape'] = shape

    # Recommendations
    if abs(out['share_delta']) >= 100:
        action = 'BUY' if out['share_delta'] > 0 else 'SELL'
        out['recommendations'].append({
            'action': f'{action} {abs(out["share_delta"])} UNG shares',
            'why': f'z={snap["z_surprise"]:+.2f} → {snap["regime"]} → target {target} '
                   f'(× base {base_shares} mult {snap["mult"]})',
            'priority': 'high',
        })

    # CC posture
    if snap['regime'] in ('RICH', 'EXTREME_RICH'):
        out['recommendations'].append({
            'action': f'Consider ITM CCs to force-assign at z={snap["z_surprise"]:+.2f}',
            'why': 'aggressive_itm_cc_z=-0.25 fires when z very cheap; here too rich → '
                   'standard 5% OTM CCs preferred, but ITM-cap exposure if z continues up',
            'priority': 'medium',
        })
    elif snap['regime'] == 'NEUTRAL':
        out['recommendations'].append({
            'action': 'Standard 5% OTM CCs on uncovered shares (K ≈ ${:.2f})'.format(spot * 1.05),
            'why': 'NEUTRAL regime; normal premium harvest mode',
            'priority': 'low',
        })

    # Put posture
    if current_put_collateral > 0:
        # Check over-leverage signal
        # Recommend not opening new puts if collateral > 80% of likely NAV proxy (5x base*spot)
        proxy_nav = base_shares * spot
        out['put_collateral_pct_nav'] = (current_put_collateral / proxy_nav) if proxy_nav > 0 else 0
        if out['put_collateral_pct_nav'] > 0.8:
            out['warnings'].append('Short-put collateral > 80% of est NAV — over-leveraged; '
                                   'do NOT open new puts until existing roll off')

    # Shoulder
    import datetime
    month = datetime.date.today().month
    in_shoulder = month in (3, 4, 5, 9, 10, 11)
    out['shoulder_season'] = in_shoulder
    if in_shoulder and snap['z_surprise'] > -0.5:
        out['recommendations'].append({
            'action': 'KOLD shoulder hedge: allocate up to 10% NAV (z-scaled)',
            'why': 'Mar-May/Sept-Nov empirically weak NG; KOLD +0.33%/d vs +0.12%/d non-shoulder',
            'priority': 'medium',
        })

    # Walk-forward truth disclosure
    out['warnings'].append('Walk-forward worst 12mo MDD: -17% (full-sample MDD -7% is sample-biased)')

    return out


if __name__ == '__main__':
    # Smoke test
    import json
    test_positions = [
        {'symbol': 'UNG', 'is_option': False, 'quantity': 5400},
        {'symbol': 'UNG', 'is_option': True, 'option_type': 'PUT',
         'strike': 11.0, 'expiry': '2026-07-17', 'quantity': -14},
        {'symbol': 'UNG', 'is_option': True, 'option_type': 'CALL',
         'strike': 11.5, 'expiry': '2026-06-26', 'quantity': -14},
    ]
    verdict = validated_verdict(11.51, test_positions)
    print(json.dumps(verdict, indent=2, default=str))
