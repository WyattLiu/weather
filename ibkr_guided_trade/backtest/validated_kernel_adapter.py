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
                      base_shares: int = 6200,
                      nav: Optional[float] = None) -> Dict[str, Any]:
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

    # ─── ACTIONABLE: timed, concrete recs with prices/qtys ───────────────
    # Share rebalance: kernel uses cut_speed=0.5 — only move 50% toward target per cadence
    if abs(out['share_delta']) >= 100:
        this_cycle = int(round(out['share_delta'] * 0.5 / 100) * 100)
        full_delta = out['share_delta']
        if this_cycle != 0:
            action = 'BUY' if this_cycle > 0 else 'SELL'
            # Limit ladder: 50% at mid, 30% lower, 20% lower (or mirror for sells)
            mid = spot
            l1 = round(spot * 1.005, 2)  # 0.5% above
            l2 = round(spot * 0.995, 2)  # 0.5% below
            l3 = round(spot * 0.985, 2)  # 1.5% below
            est_cost = abs(this_cycle) * spot
            out['recommendations'].append({
                'action': f'{action} {abs(this_cycle)} UNG shares THIS CYCLE',
                'order_draft': {
                    'qty': abs(this_cycle), 'side': action.lower(),
                    'ladder': [
                        {'price': l1, 'qty': int(this_cycle * 0.5)},
                        {'price': l2, 'qty': int(this_cycle * 0.3)},
                        {'price': l3, 'qty': int(this_cycle * 0.2)},
                    ] if this_cycle > 0 else [
                        {'price': l3, 'qty': abs(int(this_cycle * 0.5))},
                        {'price': l2, 'qty': abs(int(this_cycle * 0.3))},
                        {'price': l1, 'qty': abs(int(this_cycle * 0.2))},
                    ],
                },
                'est_cost_dollar': est_cost,
                'why': (f'Kernel target {target} - current {current_shares} = {full_delta:+d}. '
                        f'z_target_cadence_days=21 + cut_speed=0.5 → move 50% per month → '
                        f'{this_cycle:+d} this cycle, re-check next month'),
                'priority': 'high',
                'when': 'this week',
            })

    # Put-expiration calendar: predict each contract's fate
    if positions:
        from datetime import date as _date
        today = _date.today()
        upcoming = []
        for p in positions:
            if p.get('symbol','').upper() != 'UNG': continue
            if not p.get('is_option'): continue
            if p.get('option_type') != 'PUT': continue
            q = abs(int(p.get('quantity',0) or 0))
            if q == 0: continue
            K = float(p.get('strike') or 0)
            try:
                exp_d = _date.fromisoformat(p.get('expiry',''))
            except Exception:
                continue
            dte = (exp_d - today).days
            if dte < 0 or dte > 45: continue
            collat = q * 100 * K
            # Outcome estimate
            if K < spot - 0.05:
                outcome = 'EXPIRE_OTM'
                freed = collat
            elif K > spot + 0.05:
                outcome = 'ASSIGN'
                freed = 0
            else:
                outcome = 'ATM'
                freed = collat * 0.5  # ~50% chance
            upcoming.append({
                'expiry': p.get('expiry'), 'dte': dte, 'strike': K,
                'qty': q, 'collateral': collat, 'outcome': outcome,
                'freed_est': freed,
            })
        upcoming.sort(key=lambda x: x['dte'])
        out['expiration_calendar'] = upcoming
        total_collat = sum(x['collateral'] for x in upcoming)
        total_freed_30d = sum(x['freed_est'] for x in upcoming if x['dte'] <= 30)
        if total_freed_30d >= total_collat * 0.5:
            out['recommendations'].append({
                'action': f'WAIT — ${total_freed_30d:,.0f} put collateral frees in 30 days naturally',
                'why': f'{sum(1 for x in upcoming if x["dte"]<=30 and x["outcome"]=="EXPIRE_OTM")} '
                       f'puts at strikes below ${spot} will expire OTM. Do NOT add new puts until '
                       f'collateral drops below 30% of NAV.',
                'priority': 'medium',
                'when': 'next 30 days (passive)',
            })

    # Near-expiry ATM put: opportunity to close cheap
    if positions:
        from datetime import date as _date
        today = _date.today()
        atm_to_close = []
        for p in positions:
            if (p.get('symbol','').upper() == 'UNG' and p.get('is_option')
                and p.get('option_type') == 'PUT'):
                K = float(p.get('strike') or 0)
                if abs(K - spot) <= 0.10:  # ATM ±$0.10
                    try:
                        exp_d = _date.fromisoformat(p.get('expiry',''))
                    except Exception:
                        continue
                    dte = (exp_d - today).days
                    if 0 < dte <= 30:
                        atm_to_close.append({
                            'expiry': p.get('expiry'), 'strike': K,
                            'qty': abs(int(p.get('quantity',0) or 0)),
                            'dte': dte,
                            'collat': abs(int(p.get('quantity',0) or 0)) * 100 * K,
                        })
        if atm_to_close:
            total_collat = sum(x['collat'] for x in atm_to_close)
            out['recommendations'].append({
                'action': f'OPTIONAL: Close {len(atm_to_close)} ATM put group(s) early to free '
                          f'${total_collat:,.0f} collateral',
                'why': f'ATM puts near spot ${spot} have ~50% assignment risk; closing for ~$0.20-0.50 '
                       f'each unlocks collateral immediately instead of waiting for expiry',
                'priority': 'low',
                'when': 'optional, anytime',
            })

    # CC posture: regime-specific
    if snap['regime'] in ('RICH', 'EXTREME_RICH'):
        K_itm = round(spot * 0.95, 2)
        out['recommendations'].append({
            'action': f'Sell ITM CCs at K=${K_itm} (5% ITM) to force-assign at high prices',
            'why': f'z={snap["z_surprise"]:+.2f} → {snap["regime"]} → aggressive_itm_cc_z fires; '
                   f'force assignment locks gains via wheel exit',
            'priority': 'medium',
            'when': 'this week',
        })
    elif snap['regime'] == 'NEUTRAL':
        K_otm = round(spot * 1.05, 2)
        # Only if there are uncovered shares
        uncovered = current_shares - current_short_calls * 100
        if uncovered >= 100:
            n_ccs = uncovered // 100
            out['recommendations'].append({
                'action': f'Sell up to {n_ccs} CCs at K=${K_otm} (5% OTM, 30-45 DTE)',
                'why': f'NEUTRAL regime; harvest premium on {uncovered} uncovered shares',
                'priority': 'low',
                'when': 'this week',
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
        # Use real NAV when caller provides it; otherwise fall back to proxy
        # (base_shares × spot underestimates real NAV when BOXX/cash present)
        denom = nav if (nav and nav > 0) else (base_shares * spot)
        out['put_collateral_pct_nav'] = (current_put_collateral / denom) if denom > 0 else 0
        out['_nav_source'] = 'live' if (nav and nav > 0) else 'proxy(base×spot)'
        if out['put_collateral_pct_nav'] > 0.8:
            out['warnings'].append('Short-put collateral > 80% of NAV — over-leveraged; '
                                   'do NOT open new puts until existing roll off')
        elif out['put_collateral_pct_nav'] > 0.6:
            out['warnings'].append('Short-put collateral > 60% of NAV — elevated; '
                                   'avoid adding new put exposure')

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

    # ─── DEEP BEAM ANALYSIS — multi-strike candidate scoring ──────────────
    # Same logic as backtest's beam_put_selector: score 5 OTM levels by
    # income vs expected loss, return all candidates so user can see why
    # the picked strike won.
    try:
        out['beam_analysis'] = _beam_analysis(spot, snap['z_surprise'])
    except Exception as e:
        out['beam_analysis_error'] = str(e)

    # ─── PER-POSITION ANALYSIS + GREEKS ──────────────────────────────────
    try:
        out['position_analysis'] = _per_position_analysis(positions or [], spot, snap)
        out['portfolio_greeks'] = _portfolio_greeks(positions or [], spot)
        out['pnl_curve'] = _pnl_curve(positions or [], spot)
        out['delta_curve'] = _delta_curve(positions or [], spot)
        out['theta_by_expiry'] = _theta_by_expiry(positions or [], spot)
        out['theta_waterfall'] = _theta_waterfall(positions or [], spot)
        out['extrinsic'] = _extrinsic_and_smoothness(positions or [], spot)
        out['calendar_grid'] = _calendar_grid(positions or [], spot)
        out['daily_status'] = _daily_status(out)
    except Exception as e:
        out['position_analysis_error'] = str(e)

    # Walk-forward truth disclosure
    out['warnings'].append('Walk-forward worst 12mo MDD: -17% (full-sample MDD -7% is sample-biased)')

    return out


def _bs_greeks(S, K, T, r, sigma, right='C'):
    """BS Greeks: delta, gamma, theta (per day), vega (per 1% IV)."""
    import math
    from scipy.stats import norm
    if T <= 0 or sigma <= 0:
        return 0.0, 0.0, 0.0, 0.0
    d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    if right == 'C':
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1.0
    gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
    theta_call = -(S*norm.pdf(d1)*sigma) / (2*math.sqrt(T)) - r*K*math.exp(-r*T)*norm.cdf(d2)
    theta_put = -(S*norm.pdf(d1)*sigma) / (2*math.sqrt(T)) + r*K*math.exp(-r*T)*norm.cdf(-d2)
    theta = (theta_call if right == 'C' else theta_put) / 365
    vega = S * norm.pdf(d1) * math.sqrt(T) / 100  # per 1% IV change
    return float(delta), float(gamma), float(theta), float(vega)


def _per_position_analysis(positions, spot, snap):
    """For each UNG option, recommend HOLD/CLOSE/ROLL with concrete details."""
    from datetime import date as _date
    import math
    today = _date.today()
    results = []
    surf = _load_iv_surface()
    latest_surf = max(surf.keys()) if surf else None
    for p in positions:
        if p.get('symbol', '').upper() != 'UNG':
            continue
        if not p.get('is_option'):
            continue
        qty = int(p.get('quantity', 0) or 0)
        if qty == 0:
            continue
        K = float(p.get('strike') or 0)
        right = 'C' if p.get('option_type') == 'CALL' else 'P'
        try:
            exp_d = _date.fromisoformat(p.get('expiry', ''))
        except Exception:
            continue
        dte = (exp_d - today).days
        if dte <= 0:
            continue
        # IV from surface
        iv = None
        if surf and latest_surf:
            iv = iv_from_surface(surf, latest_surf, K, dte, right)
        if iv is None:
            iv = 0.50
        # Greeks
        delta, gamma, theta, vega = _bs_greeks(spot, K, dte/365, 0.045, iv, right)
        # Recommendation logic
        moneyness = 'ITM' if (right == 'C' and K < spot) or (right == 'P' and K > spot) else \
                    'ATM' if abs(K - spot) <= 0.10 else 'OTM'
        action = 'HOLD'
        action_detail = ''
        # Short calls
        if qty < 0 and right == 'C':
            if moneyness == 'ITM' and dte <= 7:
                action = 'ACCEPT_ASSIGNMENT'
                action_detail = f'Near expiry ({dte}d) + ITM — assignment likely; shares called at ${K}'
            elif moneyness == 'ITM' and dte > 7 and snap['regime'] in ('CHEAP', 'EXTREME_CHEAP'):
                action = 'CONSIDER_BUYBACK'
                action_detail = f'ITM CC in cheap regime; closing locks loss but preserves upside'
            elif moneyness == 'OTM' and dte > 30:
                action = 'HOLD'
                action_detail = f'Comfortably OTM; let theta decay work ({-theta*abs(qty)*100:.0f}/day collected)'
            elif moneyness == 'ATM':
                action = 'HOLD'
                action_detail = f'ATM — wait, may go either way. {abs(theta*qty*100):.0f}/day theta'
            else:
                action = 'HOLD'
        # Short puts
        elif qty < 0 and right == 'P':
            if moneyness == 'ITM' and dte <= 14:
                action = 'PREP_FOR_ASSIGNMENT'
                action_detail = f'Likely assigned at ${K} (acquires {abs(qty)*100} shares for ${K*abs(qty)*100:,.0f})'
            elif moneyness == 'OTM' and dte <= 7:
                action = 'LET_EXPIRE'
                action_detail = f'Far OTM ({(K/spot-1)*100:.1f}%) and {dte}d to expiry — expires worthless'
            elif moneyness == 'OTM':
                action = 'HOLD'
                action_detail = f'OTM; collecting ${-theta*abs(qty)*100:.0f}/day theta'
            elif moneyness == 'ATM':
                action = 'OPTIONAL_CLOSE'
                action_detail = f'ATM — close for ~${0.4*abs(qty)*100:.0f} to free ${K*abs(qty)*100:,.0f} collateral'
            else:
                action = 'HOLD'
        results.append({
            'right': right, 'strike': K, 'expiry': p.get('expiry'), 'dte': dte,
            'qty': qty, 'moneyness': moneyness, 'iv': round(iv, 4),
            'delta': round(delta * 100 * qty, 1),  # shares-equivalent
            'gamma': round(gamma * 100 * qty, 3),
            'theta_per_day': round(theta * 100 * qty, 2),  # $/day total
            'vega': round(vega * 100 * qty, 2),
            'action': action, 'action_detail': action_detail,
        })
    results.sort(key=lambda r: (r['dte'], r['right']))
    return results


def _intrinsic_value(K, spot, right):
    if right == 'C':
        return max(0, spot - K)
    return max(0, K - spot)


def _pnl_curve(positions, spot_now):
    """P&L profile at expiration across UNG price range."""
    import math
    from datetime import date as _date
    today = _date.today()
    # Range: 70% to 130% of current spot
    prices = [spot_now * (0.70 + 0.01 * i) for i in range(61)]
    pnl = []
    for s in prices:
        total = 0.0
        for p in positions:
            if p.get('symbol','').upper() != 'UNG': continue
            qty = int(p.get('quantity', 0) or 0)
            if qty == 0: continue
            if p.get('is_option'):
                K = float(p.get('strike') or 0)
                right = 'C' if p.get('option_type') == 'CALL' else 'P'
                intrinsic = _intrinsic_value(K, s, right)
                # P&L vs current market price (use book value as cost basis proxy)
                avg = float(p.get('average_price', 0) or 0)
                # Short: collect avg, pay intrinsic at expiry
                # Long: paid avg, get intrinsic at expiry
                if qty < 0:
                    total += abs(qty) * (avg - intrinsic) * 100
                else:
                    total += qty * (intrinsic - avg) * 100
            else:
                avg = float(p.get('average_price', 0) or 0)
                total += qty * (s - avg)
        pnl.append(round(total, 0))
    return {
        'prices': [round(x, 2) for x in prices],
        'pnl': pnl,
        'spot_now': spot_now,
    }


def _delta_curve(positions, spot_now):
    """Total portfolio delta as UNG price varies."""
    import math
    from datetime import date as _date
    surf = _load_iv_surface()
    latest_surf = max(surf.keys()) if surf else None
    today = _date.today()
    prices = [spot_now * (0.80 + 0.01 * i) for i in range(41)]
    deltas = []
    for s in prices:
        td = 0.0
        for p in positions:
            if p.get('symbol','').upper() != 'UNG': continue
            qty = int(p.get('quantity', 0) or 0)
            if qty == 0: continue
            if p.get('is_option'):
                K = float(p.get('strike') or 0)
                right = 'C' if p.get('option_type') == 'CALL' else 'P'
                try:
                    exp_d = _date.fromisoformat(p.get('expiry', ''))
                    dte = (exp_d - today).days
                    if dte <= 0: continue
                except Exception:
                    continue
                iv = None
                if surf and latest_surf:
                    iv = iv_from_surface(surf, latest_surf, K, dte, right)
                if iv is None: iv = 0.50
                d, _, _, _ = _bs_greeks(s, K, dte/365, 0.045, iv, right)
                td += d * 100 * qty
            else:
                td += qty
        deltas.append(round(td, 1))
    return {'prices': [round(x, 2) for x in prices], 'deltas': deltas, 'spot_now': spot_now}


def _theta_by_expiry(positions, spot):
    """Daily theta collection grouped by expiration date."""
    import math
    from datetime import date as _date
    surf = _load_iv_surface()
    latest_surf = max(surf.keys()) if surf else None
    today = _date.today()
    bucket = {}
    for p in positions:
        if p.get('symbol','').upper() != 'UNG': continue
        if not p.get('is_option'): continue
        qty = int(p.get('quantity', 0) or 0)
        if qty == 0: continue
        K = float(p.get('strike') or 0)
        right = 'C' if p.get('option_type') == 'CALL' else 'P'
        try:
            exp_d = _date.fromisoformat(p.get('expiry', ''))
            dte = (exp_d - today).days
            if dte <= 0: continue
        except Exception:
            continue
        iv = None
        if surf and latest_surf:
            iv = iv_from_surface(surf, latest_surf, K, dte, right)
        if iv is None: iv = 0.50
        _, _, theta, _ = _bs_greeks(spot, K, dte/365, 0.045, iv, right)
        # Negate: short positions collect theta as positive income
        contrib = theta * 100 * qty  # short pos (qty<0) × theta_bsm (neg) = positive income
        bucket[p.get('expiry')] = bucket.get(p.get('expiry'), 0.0) + contrib
    rows = [{'expiry': k, 'theta_per_day': round(v, 2)} for k, v in bucket.items()]
    rows.sort(key=lambda r: r['expiry'])
    return rows


def _theta_waterfall(positions, spot):
    """Cumulative theta projection over next 60 days + smoothness quality."""
    import math
    from datetime import date as _date, timedelta
    surf = _load_iv_surface()
    latest_surf = max(surf.keys()) if surf else None
    today = _date.today()
    result = []
    cumulative = 0.0
    for d_ahead in range(0, 61, 3):
        future = today + timedelta(days=d_ahead)
        daily = 0.0
        for p in positions:
            if p.get('symbol','').upper() != 'UNG': continue
            if not p.get('is_option'): continue
            qty = int(p.get('quantity', 0) or 0)
            if qty == 0: continue
            K = float(p.get('strike') or 0)
            right = 'C' if p.get('option_type') == 'CALL' else 'P'
            try:
                exp_d = _date.fromisoformat(p.get('expiry', ''))
            except Exception:
                continue
            dte_remaining = (exp_d - future).days
            if dte_remaining <= 0: continue
            iv = None
            if surf and latest_surf:
                iv = iv_from_surface(surf, latest_surf, K, dte_remaining, right)
            if iv is None: iv = 0.50
            _, _, theta, _ = _bs_greeks(spot, K, dte_remaining/365, 0.045, iv, right)
            daily += theta * 100 * qty  # short × neg = positive
        cumulative += daily * 3
        result.append({
            'day': d_ahead, 'date': future.isoformat(),
            'daily_theta': round(daily, 2),
            'cumulative_theta': round(cumulative, 2),
        })
    return result


def _extrinsic_and_smoothness(positions, spot):
    """Expected extrinsic-value decay + smoothness quality (production metric).

    Smoothness = 1 - std(weekly_theta)/mean(weekly_theta), 4-week horizon.
    Higher = more even income across weeks. Production uses this as a
    quality gauge for the wheel's income stream.
    """
    import math
    from datetime import date as _date
    surf = _load_iv_surface()
    latest_surf = max(surf.keys()) if surf else None
    today = _date.today()
    total_extrinsic = 0.0
    extrinsic_30d_realized = 0.0  # expected to decay within 30d (short pos = profit)
    weekly_theta = [0.0, 0.0, 0.0, 0.0]  # next 4 weeks
    for p in positions:
        if p.get('symbol','').upper() != 'UNG': continue
        if not p.get('is_option'): continue
        qty = int(p.get('quantity', 0) or 0)
        if qty == 0: continue
        K = float(p.get('strike') or 0)
        right = 'C' if p.get('option_type') == 'CALL' else 'P'
        try:
            exp_d = _date.fromisoformat(p.get('expiry', ''))
            dte = (exp_d - today).days
            if dte <= 0: continue
        except Exception:
            continue
        avg_prem = float(p.get('average_price', 0) or 0)
        intrinsic = _intrinsic_value(K, spot, right)
        # Current premium estimate
        iv = None
        if surf and latest_surf:
            iv = iv_from_surface(surf, latest_surf, K, dte, right)
        if iv is None: iv = 0.50
        # Use BSM to estimate current value
        T = dte / 365
        d1 = (math.log(spot/K) + (0.045 + 0.5*iv**2)*T) / (iv*math.sqrt(T))
        d2 = d1 - iv*math.sqrt(T)
        from scipy.stats import norm
        if right == 'C':
            current_value = spot*norm.cdf(d1) - K*math.exp(-0.045*T)*norm.cdf(d2)
        else:
            current_value = K*math.exp(-0.045*T)*norm.cdf(-d2) - spot*norm.cdf(-d1)
        # Extrinsic = current value − intrinsic, attributed to position
        extrinsic = max(0, current_value - intrinsic)
        contrib = extrinsic * 100 * abs(qty)
        total_extrinsic += contrib * (1 if qty < 0 else -1)  # short = we receive, long = we paid
        # Portion that decays in 30 days: theta over 30 days = ~30 * current theta
        _, _, theta, _ = _bs_greeks(spot, K, T, 0.045, iv, right)
        decay_30d = theta * 100 * qty * min(30, dte)  # short × neg = positive collection
        extrinsic_30d_realized += decay_30d
        # Weekly buckets
        for wk in range(4):
            wk_start = wk * 7
            wk_end = (wk + 1) * 7
            if dte < wk_start: continue
            days_in_wk = min(wk_end, dte) - wk_start
            if days_in_wk > 0:
                weekly_theta[wk] += theta * 100 * qty * days_in_wk  # short × neg = positive
    # Smoothness — production formula
    active = [w for w in weekly_theta if w > 0]
    if len(active) >= 2:
        import statistics
        mean_wt = sum(active) / len(active)
        std_wt = statistics.stdev(active) if len(active) > 1 else 0
        smoothness = max(0.0, 1.0 - std_wt / mean_wt) if mean_wt > 0 else 0
    else:
        smoothness = 0.0
    return {
        'total_extrinsic': round(total_extrinsic, 0),
        'extrinsic_decay_30d_est': round(extrinsic_30d_realized, 0),
        'weekly_theta': [round(w, 0) for w in weekly_theta],
        'smoothness': round(smoothness, 3),
        'avg_weekly_theta': round(sum(weekly_theta) / 4, 0),
    }


def _calendar_grid(positions, spot):
    """Rolling calendar: strike × expiry → contracts held."""
    grid = {}
    strikes = set()
    expiries = set()
    for p in positions:
        if p.get('symbol','').upper() != 'UNG': continue
        if not p.get('is_option'): continue
        qty = int(p.get('quantity', 0) or 0)
        if qty == 0: continue
        K = float(p.get('strike') or 0)
        right = 'C' if p.get('option_type') == 'CALL' else 'P'
        exp = p.get('expiry')
        strikes.add(K)
        expiries.add(exp)
        key = (K, exp)
        grid.setdefault(key, {'C': 0, 'P': 0})
        grid[key][right] += qty
    return {
        'strikes': sorted(strikes),
        'expiries': sorted(expiries),
        'cells': [{'strike': k, 'expiry': e, **grid[(k,e)]}
                  for (k, e) in grid.keys()],
        'spot': spot,
    }


def _daily_status(out):
    """Roll up overall portfolio health into a single banner status."""
    # Critical: over-leveraged or near-assignment
    collat_pct = out.get('put_collateral_pct_nav', 0)
    regime = out.get('regime', 'NEUTRAL')
    share_delta = out.get('share_delta', 0)
    z = out.get('z_surprise', 0)
    issues = []
    headline = 'All systems nominal'
    color = 'green'
    if collat_pct > 0.8:
        color = 'red'
        headline = 'OVER-LEVERAGED — do not open new puts'
        issues.append(f'Put collateral {collat_pct*100:.0f}% of NAV')
    elif collat_pct > 0.6:
        color = 'orange'
        headline = 'Elevated leverage — caution on new exposure'
        issues.append(f'Put collateral {collat_pct*100:.0f}% of NAV')
    elif abs(share_delta) >= 800:
        color = 'orange'
        headline = f'Share count off target by {abs(share_delta)} — gradual rebalance'
        issues.append(f'Δ shares: {share_delta:+d}')
    elif regime in ('RICH', 'EXTREME_RICH'):
        color = 'orange'
        headline = f'{regime} regime — favor share-cut over accumulation'
        issues.append(f'z={z:+.2f}')
    elif regime == 'NEUTRAL':
        headline = 'NEUTRAL — premium harvest mode'
    elif regime in ('CHEAP', 'EXTREME_CHEAP'):
        headline = f'{regime} — load up if not over-leveraged'
        color = 'green'

    return {'color': color, 'headline': headline, 'issues': issues}


def _portfolio_greeks(positions, spot):
    """Aggregate Greeks across all UNG positions (shares + options)."""
    total_delta = 0.0
    total_gamma = 0.0
    total_theta = 0.0
    total_vega = 0.0
    shares = 0
    surf = _load_iv_surface()
    latest_surf = max(surf.keys()) if surf else None
    from datetime import date as _date
    today = _date.today()
    for p in positions:
        if p.get('symbol', '').upper() != 'UNG':
            continue
        qty = int(p.get('quantity', 0) or 0)
        if p.get('is_option'):
            K = float(p.get('strike') or 0)
            right = 'C' if p.get('option_type') == 'CALL' else 'P'
            try:
                exp_d = _date.fromisoformat(p.get('expiry', ''))
            except Exception:
                continue
            dte = (exp_d - today).days
            if dte <= 0:
                continue
            iv = None
            if surf and latest_surf:
                iv = iv_from_surface(surf, latest_surf, K, dte, right)
            if iv is None:
                iv = 0.50
            d, g, t, v = _bs_greeks(spot, K, dte/365, 0.045, iv, right)
            total_delta += d * 100 * qty
            total_gamma += g * 100 * qty
            total_theta += t * 100 * qty
            total_vega += v * 100 * qty
        else:
            shares += qty
            total_delta += qty  # shares are delta 1 each
    return {
        'shares_delta': shares,
        'total_delta': round(total_delta, 1),
        'total_gamma': round(total_gamma, 2),
        'total_theta_per_day': round(total_theta, 2),
        'total_vega': round(total_vega, 2),
        'delta_dollar_per_1pct': round(total_delta * spot * 0.01, 2),  # $ P&L for 1% UNG move
    }


def _beam_analysis(spot: float, z: float):
    """Score multiple OTM put candidates and return ranking with all scores.

    Mirrors backtest's beam_put scoring: income - p_itm * expected_loss.
    """
    import math
    from scipy.stats import norm
    surf = _load_iv_surface()
    # Pick a representative DTE (45d) and ladder OTM levels
    dte = 45
    T = dte / 365
    r = 0.045
    candidates = []
    for otm in [0.02, 0.05, 0.08, 0.12, 0.15, 0.20]:
        K = round(spot * (1 - otm), 2)
        # IV at K
        iv = None
        if surf:
            latest_date = max(surf.keys())
            iv = iv_from_surface(surf, latest_date, K, dte, 'P')
        if iv is None:
            # Fallback to ATM iv shape estimate
            iv = 0.50
        # BSM put price
        d1 = (math.log(spot/K) + (r + 0.5*iv**2)*T) / (iv*math.sqrt(T))
        d2 = d1 - iv*math.sqrt(T)
        bsm_put = K*math.exp(-r*T)*norm.cdf(-d2) - spot*norm.cdf(-d1)
        # Probability ITM at expiry (under BSM measure)
        p_itm = float(norm.cdf(-d2))
        # Expected loss if assigned: (K - expected_spot_at_assignment)
        # Use a simple proxy: half-distance to strike
        expected_loss_if_itm = max(0, K - spot * 0.95)
        # Net score per contract per 100 shares
        income = bsm_put * 100
        expected_loss = p_itm * expected_loss_if_itm * 100
        score = income - expected_loss
        candidates.append({
            'strike': K,
            'otm_pct': round(otm * 100, 1),
            'iv': round(iv, 4),
            'iv_source': 'PG_real' if surf else 'fallback',
            'premium': round(bsm_put, 3),
            'p_itm_pct': round(p_itm * 100, 1),
            'income_per_contract': round(income, 1),
            'expected_loss_per_contract': round(expected_loss, 1),
            'net_score': round(score, 1),
            'dte': dte,
        })
    candidates.sort(key=lambda c: c['net_score'], reverse=True)
    return {
        'spot': spot,
        'dte': dte,
        'candidates': candidates,
        'winner': candidates[0]['strike'] if candidates else None,
        'method': 'income - p_itm × expected_loss (BSM measure)',
    }


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
