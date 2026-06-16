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

# ─── KERNEL REGISTRY ─────────────────────────────────────────────────────────
# Each entry: (strategy_name, OOS train/test metrics, why-to-use blurb).
# OOS metrics from backtest/honest_walkforward.py with sealed test data
# (2024-01 → 2026-06) + IBKR cost model + leak-free z.
KERNELS = {
    'kold15_ivrank_kbh': {
        'strategy': 'champion_kold15_ivrank_kbh',
        'label': 'KOLD-15 + IV-Rank + Book Hedge (gen-8 champion)',
        # OOS sealed test 2024-26, real fills, matched-share controlled:
        # +0.32 Sharpe / -1.9pp MDD / -0.3pp return vs the unhedged kernel.
        'oos_ann': 22.4,   'oos_sharpe': 2.10, 'oos_mdd': -6.6,
        'is_ann': 58.9,    'is_sharpe': 3.03,  'is_mdd': -13.3,
        'why': 'Promoted 2026-06-14. Adds a KOLD 2x-inverse book hedge '
               '(frac 0.5) on the uncovered share book — passed all 4 rigor '
               'gates (confound-free, bootstrap-significant, OOS-validated, '
               'cost ~0). Same return as kold15_ivrank, +0.32 Sharpe, ~2pp '
               'less drawdown. Cheap grind-regime insurance. See KERNEL_LAB.md.',
    },
    'kold15_ivrank': {
        'strategy': 'champion_kold15_ivrank',
        'label': 'KOLD-15 + IV-Rank (gen-2 champion)',
        # OOS from honest_walkforward sealed test 2024-01→2026-06 with
        # slippage + assignment haircuts (commission $0 on WS; the $0.65/ct
        # in the harness is conservative padding)
        'oos_ann': 31.1,   'oos_sharpe': 2.16, 'oos_mdd': -9.2,
        'is_ann': 65.5,    'is_sharpe': 2.99,  'is_mdd': -13.6,
        'why': 'Promoted 2026-06-13. scale_invariant + KOLD hedge 0.15 + '
               'IV-rank z-scaling (real-ATM-IV percentile trims adds at '
               'rich-vol tops, boosts at cheap-vol bottoms). Best sealed-OOS '
               'Sharpe (2.16 vs 2.07) at comparable MDD; floor +2.5% '
               'worst-12mo. See backtest/KERNEL_LAB.md.',
    },
    'premium_harvest_scale_invariant': {
        'strategy': 'champion_premium_harvest_scale_invariant',
        'label': 'Premium Harvest (Scale-Invariant)',
        'oos_ann': 27.9,   'oos_sharpe': 2.34, 'oos_mdd': -7.5,
        'is_ann': 49.2,    'is_sharpe': 3.84,  'is_mdd': -6.0,
        'why': 'Lowest OOS MDD + NAV-pct sizing (same return at $50K → $1M). '
               'Best balanced choice for production at any capital level.',
    },
    'premium_harvest': {
        'strategy': 'champion_premium_harvest',
        'label': 'Premium Harvest (Original)',
        'oos_ann': 34.6,   'oos_sharpe': 2.42, 'oos_mdd': -9.5,
        'is_ann': 52.4,    'is_sharpe': 3.18,  'is_mdd': -7.8,
        'why': 'Highest OOS Sharpe (2.42). ITM-put gate + smaller share base. '
               'Best risk-adjusted; hardcoded sizes (less robust at scale).',
    },
    'target_25_smooth': {
        'strategy': 'champion_target_25_smooth',
        'label': 'Target 25 Smooth (max return)',
        'oos_ann': 41.1,   'oos_sharpe': 2.21, 'oos_mdd': -15.6,
        'is_ann': 61.3,    'is_sharpe': 2.89,  'is_mdd': -9.4,
        'why': 'Highest OOS annual return (41%). Accept wider MDD for the '
               'extra return. Choose if max gains > smooth equity curve.',
    },
    'target_25_dd_trim': {
        'strategy': 'champion_target_25_dd_trim',
        'label': 'Target 25 DD-Trim',
        'oos_ann': 39.1,   'oos_sharpe': 1.97, 'oos_mdd': -17.5,
        'is_ann': 55.5,    'is_sharpe': 2.87,  'is_mdd': -9.4,
        'why': 'Reactive DD-trim, between smooth and walkforward_safe. '
               'Solid return with active risk control.',
    },
}

CHAMPION_KEY = 'kold15_ivrank_kbh'   # promoted 2026-06-14 (+ KOLD book hedge); prior: kold15_ivrank
CHAMPION_NAME = KERNELS[CHAMPION_KEY]['strategy']


def get_kernel_info(key=None):
    """Return the kernel info for the active kernel (or specified key)."""
    return KERNELS.get(key or CHAMPION_KEY, KERNELS[CHAMPION_KEY])


def _snap_strike(K: float) -> float:
    """Snap a computed strike to UNG's real listed grid so every suggestion is
    actually tradeable. UNG lists $0.50 increments near the money (…10.5, 11.0,
    11.5, 12.0…); wider out it's $1, but $0.50 rounding is always a listed strike.
    Returns a float rounded to one decimal (avoids 11.5000001 noise)."""
    if not K or K <= 0:
        return K
    return round(round(K * 2) / 2, 1)


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


def _ag_put_collateral(positions, underlying):
    """Sum K*100*|qty| of SHORT puts on `underlying` from WS positions."""
    total = 0.0
    for p in (positions or []):
        try:
            sym = (p.get('underlying_symbol') or p.get('symbol') or '').upper()
            if not sym.startswith(underlying.upper()):
                continue
            if (p.get('option_type') or '').upper() != 'PUT':
                continue
            qty = float(p.get('quantity') or 0)
            if qty >= 0:
                continue
            total += float(p.get('strike') or 0) * 100 * abs(qty)
        except Exception:
            continue
    return total


def validated_verdict(spot: float, positions: Optional[List[Dict[str, Any]]] = None,
                      base_shares: int = 6200,
                      nav: Optional[float] = None,
                      kernel_key: Optional[str] = None) -> Dict[str, Any]:
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
    # Kernel selection (defaults to champion)
    active_key = kernel_key if kernel_key in KERNELS else CHAMPION_KEY
    kernel_info = KERNELS[active_key]
    out = {
        'available': True,
        'kernel': kernel_info['strategy'],
        'kernel_key': active_key,
        'kernel_label': kernel_info['label'],
        'kernel_why': kernel_info['why'],
        'kernel_params': {
            k: STRATEGIES.get(kernel_info['strategy'], {}).get(k)
            for k in ('kold_shoulder_hedge', 'iv_rank_z_scale', 'cc_gex_floor',
                      'tp_dynamic', 'dd_trim_trigger_pct', 'otm_put')
        },
        'kernel_oos': {
            'ann_pct': kernel_info['oos_ann'],
            'sharpe': kernel_info['oos_sharpe'],
            'mdd_pct': kernel_info['oos_mdd'],
        },
        'kernel_is': {
            'ann_pct': kernel_info['is_ann'],
            'sharpe': kernel_info['is_sharpe'],
            'mdd_pct': kernel_info['is_mdd'],
        },
        'available_kernels': [
            {
                'key': k, 'label': v['label'],
                'oos_ann': v['oos_ann'], 'oos_sharpe': v['oos_sharpe'], 'oos_mdd': v['oos_mdd'],
                'why': v['why'],
            } for k, v in KERNELS.items()
        ],
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

    # Target shares — prefer kernel's scale-invariant pct_nav when defined,
    # otherwise fall back to fixed base_shares.
    try:
        from replay_engine import STRATEGIES as _STR  # type: ignore
        sp_top = _STR.get(kernel_info['strategy'], {})
    except Exception:
        sp_top = {}
    pct_base_top = sp_top.get('z_share_target_pct_nav')
    if pct_base_top and spot > 0 and (nav and nav > 0):
        target = int(round(nav * pct_base_top * snap['mult'] / spot / 100) * 100)
        out['target_basis'] = f'{pct_base_top*100:.0f}% NAV * mult / spot (scale-invariant)'
    else:
        # base_shares-fallback uses kernel's z_share_target_base if set
        bs = sp_top.get('z_share_target_base', base_shares)
        target = int(round(bs * snap['mult'] / 100) * 100)
        out['target_basis'] = f'{bs} base * mult (fixed)'
    out['target_shares'] = target

    current_shares = 0
    current_short_calls = 0
    current_short_puts = 0
    current_put_collateral = 0.0
    # STATISTICAL ASSIGNMENT MODEL — apply to BOTH short calls and short puts
    # so the dashboard surfaces probability per leg, and aggregate buckets
    # feed downstream decisions (don't pay BTC for what assignment does free).
    pending_assign_shares = 0       # call-side: shares auto-divesting via assignment
    pending_assign_calls = 0
    pending_put_assign_shares = 0   # put-side: shares we'd be FORCED to buy
    pending_put_assign_calls = 0
    likely_assign_legs = []
    put_assign_legs = []
    leg_assignments = []            # full per-leg breakdown for dashboard
    from datetime import date as _d_today
    from assignment_model import assignment_probability  # type: ignore
    _today_iso = _d_today.today()
    # Quick IV lookup (fallback 0.50 if no surface). We use this for every
    # short leg's assignment-prob calculation.
    surf = _iv_shape_today(spot) or {}
    base_iv = float(surf.get('atm_iv') or 0.50)
    # SURGE-Z for mean-reversion adjustment in assignment model.
    # Compute spot's z-score against recent 20d MA. Big positive z → spot
    # surged → calls less likely to assign (might revert).
    surge_z = None
    try:
        import pandas as _pd
        _df = _pd.read_csv(os.path.join(THIS_DIR, 'cache', 'master_dataset.csv'),
                            index_col=0, parse_dates=True)
        recent = _df['UNG'].dropna().tail(20)
        if len(recent) >= 10:
            ma, sd = recent.mean(), recent.std()
            if sd > 0:
                surge_z = float((spot - ma) / sd)
    except Exception as _e:
        surge_z = None
    out['surge_z'] = surge_z
    if positions:
        for p in positions:
            sym = p.get('symbol', '').upper()
            if sym != 'UNG':
                continue
            qty = int(p.get('quantity', 0) or 0)
            if p.get('is_option'):
                K = float(p.get('strike') or 0)
                try:
                    exp_d = _d_today.fromisoformat(p.get('expiry', ''))
                    dte = (exp_d - _today_iso).days
                except Exception:
                    dte = 999
                # Market premium per share — convert position market_value
                mv = float(p.get('market_value') or 0)
                prem_per_share = abs(mv) / (abs(qty) * 100) if qty != 0 else None
                right = p.get('option_type', '').upper()
                assign = assignment_probability(K=K, spot=spot, dte=dte, iv=base_iv,
                                                right=right, premium_market=prem_per_share,
                                                mean_reversion_z=surge_z)
                leg_record = {
                    'right': right, 'qty': abs(qty), 'strike': K,
                    'expiry': p.get('expiry'), 'dte': dte,
                    'p_assign': assign['p_assign'],
                    'regime': assign['regime'],
                    'intrinsic': assign['intrinsic'],
                    'extrinsic': assign['extrinsic'],
                    'mkt_prem_per_share': prem_per_share,
                }
                leg_assignments.append(leg_record)

                if right == 'CALL':
                    current_short_calls += abs(qty)
                    # Call assignment → shares auto-sold at K (kernel-favorable
                    # when reducing exposure). Count as "pending assignment" if
                    # p_assign >= 0.55 (likely+ regime).
                    if assign['p_assign'] >= 0.55:
                        pending_assign_shares += abs(qty) * 100
                        pending_assign_calls += abs(qty)
                        likely_assign_legs.append({
                            'qty': abs(qty), 'strike': K, 'expiry': p.get('expiry'),
                            'dte': dte, 'p_assign': assign['p_assign'],
                            'itm_pct': round((spot - K)/spot*100, 2) if spot > 0 else 0,
                        })
                else:
                    current_short_puts += abs(qty)
                    current_put_collateral += abs(qty) * 100 * K
                    # Put assignment → shares auto-BOUGHT at K. This INCREASES
                    # exposure (opposite of what kernel may want during cut).
                    if assign['p_assign'] >= 0.55:
                        pending_put_assign_shares += abs(qty) * 100
                        pending_put_assign_calls += abs(qty)
                        put_assign_legs.append({
                            'qty': abs(qty), 'strike': K, 'expiry': p.get('expiry'),
                            'dte': dte, 'p_assign': assign['p_assign'],
                            'itm_pct': round((K - spot)/spot*100, 2) if spot > 0 else 0,
                        })
            else:
                current_shares += qty

    out['current_shares'] = current_shares
    out['share_delta'] = target - current_shares
    out['current_short_calls'] = current_short_calls
    out['current_short_puts'] = current_short_puts
    out['current_put_collateral'] = current_put_collateral
    out['pending_assign_shares'] = pending_assign_shares
    out['pending_assign_calls'] = pending_assign_calls
    out['pending_put_assign_shares'] = pending_put_assign_shares
    out['pending_put_assign_calls'] = pending_put_assign_calls
    out['likely_assign_legs'] = likely_assign_legs
    out['put_assign_legs'] = put_assign_legs
    out['leg_assignments'] = leg_assignments     # full per-leg detail
    # Constraint enforcement — applies across ALL kernels, not kernel-specific
    out['constraints'] = {
        'covered_calls_only': True,  # never short calls without shares to cover
        'cash_secured_puts': True,    # put collateral always available
        'put_collateral_ceiling_pct_nav': 0.80,
        'shares_uncovered': current_shares - current_short_calls * 100,
        'cc_room': max(0, (current_shares - current_short_calls * 100) // 100),
        'enforced_at': 'kernel_adapter._build_actionable_orders',
    }

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
        K_itm = _snap_strike(spot * 0.95)
        out['recommendations'].append({
            'action': f'Sell ITM CCs at K=${K_itm} (~5% ITM) to force-assign at high prices',
            'why': f'z={snap["z_surprise"]:+.2f} → {snap["regime"]} → aggressive_itm_cc_z fires; '
                   f'force assignment locks gains via wheel exit',
            'priority': 'medium',
            'when': 'this week',
        })
    elif snap['regime'] == 'NEUTRAL':
        K_otm = _snap_strike(spot * 1.05)
        # Only if there are uncovered shares
        uncovered = current_shares - current_short_calls * 100
        if uncovered >= 100:
            n_ccs = uncovered // 100
            out['recommendations'].append({
                'action': f'Sell up to {n_ccs} CCs at K=${K_otm} (~5% OTM, 30-45 DTE)',
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
            'action': 'Standard 5% OTM CCs on uncovered shares (K = ${:.2f})'.format(_snap_strike(spot * 1.05)),
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

    # ─── BOXX CASH OPTIMIZATION (cash + margin are SEPARATE) ────────────
    # Query WS for REAL cash + buying_power (do NOT derive from NAV).
    # Cash = actual liquid USD in account (includes premium collected).
    # Buying power = remaining margin headroom (depleted by puts + BOXX).
    # BOXX requires 50% margin: each $X of BOXX uses $0.5X cash + $0.5X margin.
    # Short puts INCREASE cash (premium collected) but DECREASE margin
    # (collateral requirement).
    boxx_qty = 0
    boxx_mkt_value = 0.0
    for p in (positions or []):
        if p.get('symbol', '').upper() == 'BOXX' and not p.get('is_option'):
            boxx_qty += int(p.get('quantity', 0))
            boxx_mkt_value += float(p.get('market_value', 0))
    # Fetch REAL cash + buying_power via WS
    real_cash = None
    real_buying_power = None
    try:
        from ws_sdk import WSClient, get_session, graphql_query
        from ws_sdk.queries import QUERY_TRADING_BALANCE
        _c = WSClient()
        _session = get_session()
        for _a in _c.list_accounts():
            if 'non-registered' in _a.id and 'MARGIN' in str(_a.type).upper():
                _d = graphql_query(_session, 'FetchTradingBalanceBuyingPower',
                                    QUERY_TRADING_BALANCE,
                                    {'accountCanonicalId': _a.id, 'currency': 'USD'})
                _v = ((_d or {}).get('account') or {}).get('financials', {}).get('current', {}).get('tradingBalanceView') or {}
                _cash_q = float((_v.get('cash') or {}).get('quantity') or 0)
                _bp_q = float((_v.get('buyingPower') or {}).get('quantity') or 0)
                if _cash_q > 0 or _bp_q > 0:
                    real_cash = _cash_q
                    real_buying_power = _bp_q
                    break
    except Exception:
        pass
    # BOXX price from master_dataset
    boxx_spot_est = 117.0
    try:
        import pandas as _pd
        _df = _pd.read_csv(os.path.join(THIS_DIR, 'cache', 'master_dataset.csv'),
                            index_col=0, parse_dates=True)
        _bx = _df['BOXX'].dropna()
        if len(_bx) > 0:
            boxx_spot_est = float(_bx.iloc[-1])
    except Exception:
        pass

    # Decide BUY_BOXX qty based on REAL cash + margin (not derived)
    more_boxx_shares = 0
    cash_buffer = 5000  # keep this much liquid
    max_boxx_dollars = 0
    cash_available = 0
    margin_available = real_buying_power if real_buying_power is not None else 0
    if real_cash is not None and real_buying_power is not None:
        # CASH-ONLY mode (avoid margin interest leakage):
        # Use only cash to buy BOXX. No leverage. Margin only acts as ceiling.
        # WS margin rate (~5.5%) > BOXX yield (4.74%) → margined BOXX loses money.
        # BOXX IS THE RESIDUAL, not the default: reserve cash for unfilled
        # ag-leg gaps first (those legs out-earn BOXX 4.74% whenever a gap
        # exists per real-chain backtests) — only the cash NO leg can
        # absorb gets parked.
        _ag_gap_reserve = 0.0
        try:
            import json as _json2
            _cpath = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'research/dba/cache/composite_state.json')
            with open(_cpath) as _cf:
                _ctgts = (_json2.load(_cf).get('portfolio_targets') or {})
            for _lg in ('DBA', 'CORN', 'CANE'):
                _t = float(_ctgts.get(_lg, 0))
                if _t > 0:
                    _ag_gap_reserve += max(0, (nav or 0) * _t
                                           - _ag_put_collateral(positions, _lg))
        except Exception:
            pass
        cash_available = max(0, real_cash - cash_buffer - _ag_gap_reserve)
        out['ag_gap_reserve'] = round(_ag_gap_reserve, 0)
        max_boxx_from_cash = cash_available             # 1× cash, NO leverage
        max_boxx_from_margin = real_buying_power * 2    # margin ceiling check
        max_boxx_dollars = min(max_boxx_from_cash, max_boxx_from_margin)
        if max_boxx_dollars > 1000:
            more_boxx_shares = int(max_boxx_dollars * 0.7 / boxx_spot_est)

    out['boxx_state'] = {
        'qty': boxx_qty,
        'market_value': round(boxx_mkt_value, 0),
        'pct_nav': round(boxx_mkt_value / nav * 100, 1) if (nav or 0) > 0 else 0,
        'real_cash_usd': real_cash,
        'real_buying_power_usd': real_buying_power,
        'cash_available_after_buffer': round(cash_available, 0),
        'max_boxx_buy_dollars': round(max_boxx_dollars, 0),
        'suggest_more_boxx_shares': more_boxx_shares,
    }

    # ─── BOXX ACTIONABLE ORDERS ──────────────────────────────────────────
    # Emit BUY_BOXX based on REAL cash + buying_power from WS.
    # SELL_BOXX only when cash is genuinely critical (real_cash < buffer).
    BOXX_SEC_ID = 'sec-s-aed53cd42a354b0fa104745054d0daa6'
    if more_boxx_shares >= 10:
        # LIVE bid/ask via yfinance (BOXX tracks daily close; intraday matters)
        live_boxx_price = boxx_spot_est
        bid = round(boxx_spot_est - 0.02, 2)
        ask = round(boxx_spot_est + 0.02, 2)
        last = boxx_spot_est
        try:
            import yfinance as _yf
            _t = _yf.Ticker('BOXX')
            _info = _t.fast_info
            _bid_l = float(getattr(_info, 'last_price', 0) or 0)
            _bid_v = float(getattr(_info, 'bid', 0) or 0) or _bid_l
            _ask_v = float(getattr(_info, 'ask', 0) or 0) or _bid_l
            if _bid_v > 0 and _ask_v > 0:
                bid = round(_bid_v, 2)
                ask = round(_ask_v, 2)
                last = round(_bid_l or (_bid_v + _ask_v) / 2, 2)
                live_boxx_price = round((bid + ask) / 2, 2)
        except Exception:
            pass
        # TIGHT-SPREAD SHORTCUT: if spread ≤ \$0.05, cross to ASK for
        # instant fill (waiting at bid is wasteful; spread cost negligible).
        spread = ask - bid
        if spread <= 0.05:
            boxx_ladder = [
                {'tier': 1, 'qty': more_boxx_shares,
                 'limit_price': ask,  # hit ask = instant fill
                 'kind': 'cross_immediate'},
            ]
        else:
            boxx_ladder = [
                {'tier': 1, 'qty': max(1, int(more_boxx_shares * 0.5)),
                 'limit_price': bid, 'kind': 'passive'},
                {'tier': 2, 'qty': max(1, int(more_boxx_shares * 0.3)),
                 'limit_price': round(live_boxx_price, 2), 'kind': 'mid'},
                {'tier': 3, 'qty': max(1, int(more_boxx_shares * 0.2)),
                 'limit_price': ask, 'kind': 'cross'},
            ]
        # Reconcile ladder qtys to total
        total_q = sum(t['qty'] for t in boxx_ladder)
        if total_q != more_boxx_shares:
            boxx_ladder[0]['qty'] += (more_boxx_shares - total_q)
        out.setdefault('_pending_boxx_orders', []).append({
            'order_type': 'BUY_BOXX',
            'side': 'BUY',
            'symbol': 'BOXX',
            'sec_id': BOXX_SEC_ID,
            'qty': more_boxx_shares,
            'limit_ladder': boxx_ladder,
            'live_bid': bid, 'live_ask': ask, 'live_mid': live_boxx_price,
            'est_cost': round(more_boxx_shares * live_boxx_price, 0),
            'expected_yield_dollars_per_year': round(more_boxx_shares * live_boxx_price * 0.0474, 0),
            'rationale': (f'Park ${more_boxx_shares * live_boxx_price:,.0f} of available cash in BOXX for risk-free ~4.74%/yr. '
                          f'Real cash ${real_cash:,.0f}, buying power ${real_buying_power:,.0f} '
                          f'(50% margin: ${more_boxx_shares * live_boxx_price * 0.5:,.0f} cash + ${more_boxx_shares * live_boxx_price * 0.5:,.0f} margin). '
                          f'Current BOXX position {boxx_qty} shares = ${boxx_mkt_value:,.0f} ({out["boxx_state"]["pct_nav"]:.0f}% NAV). '
                          f'Assignments tonight add ~$4,600 more cash (no interest carry).'),
            'priority': 'high',  # cash sitting at 0% loses yield daily; real action
        })
    elif real_cash is not None and real_cash < 1000 and boxx_qty > 0:
        # SELL BOXX only if real cash is critically low (not derived deficit)
        deficit = max(5000, 1000 - real_cash + cash_buffer)
        live_boxx_price = boxx_spot_est
        bid = round(boxx_spot_est - 0.02, 2)
        ask = round(boxx_spot_est + 0.02, 2)
        try:
            import yfinance as _yf
            _t = _yf.Ticker('BOXX')
            _info = _t.fast_info
            _bid_v = float(getattr(_info, 'bid', 0) or 0)
            _ask_v = float(getattr(_info, 'ask', 0) or 0)
            if _bid_v > 0 and _ask_v > 0:
                bid, ask = round(_bid_v, 2), round(_ask_v, 2)
                live_boxx_price = round((bid + ask) / 2, 2)
        except Exception:
            pass
        sell_shares = min(boxx_qty, int(deficit / live_boxx_price) + 10)
        # SELL: tight spread → hit BID for instant fill
        spread = ask - bid
        if spread <= 0.05:
            boxx_ladder = [
                {'tier': 1, 'qty': sell_shares,
                 'limit_price': bid, 'kind': 'cross_immediate'},
            ]
        else:
            boxx_ladder = [
                {'tier': 1, 'qty': max(1, int(sell_shares * 0.5)),
                 'limit_price': ask, 'kind': 'passive'},
                {'tier': 2, 'qty': max(1, int(sell_shares * 0.3)),
                 'limit_price': round(live_boxx_price, 2), 'kind': 'mid'},
                {'tier': 3, 'qty': max(1, int(sell_shares * 0.2)),
                 'limit_price': bid, 'kind': 'cross'},
            ]
        total_q = sum(t['qty'] for t in boxx_ladder)
        if total_q != sell_shares:
            boxx_ladder[0]['qty'] += (sell_shares - total_q)
        out.setdefault('_pending_boxx_orders', []).append({
            'order_type': 'SELL_BOXX',
            'side': 'SELL',
            'symbol': 'BOXX',
            'sec_id': BOXX_SEC_ID,
            'qty': sell_shares,
            'limit_ladder': boxx_ladder,
            'live_bid': bid, 'live_ask': ask, 'live_mid': live_boxx_price,
            'est_proceeds': round(sell_shares * live_boxx_price, 0),
            'rationale': (f'SELL {sell_shares} BOXX to free ${sell_shares * live_boxx_price:,.0f} '
                          f'cash. Real cash ${real_cash:,.0f} below critical buffer.'),
            'priority': 'high' if real_cash < 500 else 'medium',
        })

    # ─── COMPOSITE DBA EXPOSURE (UNG×DBA weather regime allocator) ───────
    # Reads research/dba/cache/composite_state.json (refreshed daily by
    # research/dba/refresh.sh). When dba_pct > 0.2 AND UNG has no setup,
    # emit SELL_PUT_DBA candidate competing with UNG/BOXX for best_play.
    # See research/dba/composite_edge.py for the allocation logic.
    try:
        import os as _os
        import time as _time_mod
        comp_path = _os.path.join(
            _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
            'research/dba/cache/composite_state.json')
        # STALENESS GUARD: signals decay (DSCI weekly, ONI/CPC monthly).
        # A composite_state older than 48h means refresh.sh isn't running —
        # do NOT trade on dead data; warn instead.
        _comp_fresh = (_os.path.exists(comp_path) and
                       (_time_mod.time() - _os.path.getmtime(comp_path)) < 48 * 3600)
        if _os.path.exists(comp_path) and not _comp_fresh:
            _age_h = (_time_mod.time() - _os.path.getmtime(comp_path)) / 3600
            out['warnings'].append(
                f'composite_state.json is {_age_h:.0f}h old (>48h) — DBA signals '
                f'suppressed; run research/dba/refresh.sh')
        if _comp_fresh:
            import json as _json
            with open(comp_path) as _f:
                _comp = _json.load(_f)
            _dba_pct = float(_comp.get('allocation', {}).get('dba', 0))
            _dba_edge = float(_comp.get('dba_edge', 0))
            _oni = float(_comp.get('oni', 0))
            out['composite_state'] = _comp
            if _dba_pct > 0.2 and (nav or 0) > 0:
                # Get live DBA price
                _dba_spot = None
                try:
                    import yfinance as _yf
                    _dba_t = _yf.Ticker('DBA').fast_info
                    _dba_spot = float(getattr(_dba_t, 'last_price', 0) or 0)
                except Exception:
                    pass
                _tgts = _comp.get('portfolio_targets') or {}
                if (_dba_spot and _dba_spot > 0
                        and float(_tgts.get('DBA', 0)) > 0):
                    # SOFT TARGET ALLOCATOR: saturation target × factor tilt;
                    # recommend only step_per_cycle of the remaining gap.
                    # Target 0 = ag carry disabled (post covered-calls-only
                    # correction: wheel carry < BOXX) — no rec at all.
                    _tilt = _comp.get('dba_wheel_tilt') or {}
                    _size_mult = float(_tilt.get('size_mult', 1.0))
                    _otm = float(_tilt.get('target_otm_pct', 0.02))
                    _dte = int(_tilt.get('target_dte', 60))
                    _target_strike = round(_dba_spot * (1 - _otm))  # $1 grid
                    _est_credit = round(_dba_spot * (0.018 + _otm * 0.3), 2)
                    _alloc_dollars = ((nav or 0) * float(_tgts.get('DBA', 0.10))
                                      * _size_mult)  # saturation level
                    _target_contracts = max(1, int(_alloc_dollars / (_target_strike * 100)))
                    _existing_dba_collateral = _ag_put_collateral(positions, 'DBA') or 19800
                    _gap = max(0, _alloc_dollars - _existing_dba_collateral)
                    _step = float(_tgts.get('step_per_cycle', 0.33))
                    _add_contracts = int(_gap * _step / (_target_strike * 100))
                    if _add_contracts == 0 and _gap >= _target_strike * 100:
                        _add_contracts = 1  # gap exists — soft-step at least 1c
                    out.setdefault('_pending_boxx_orders', []).append({
                        'order_type': 'SELL_PUT_DBA',
                        'side': 'SELL_TO_OPEN',
                        'symbol': 'DBA',
                        'sec_id': None,  # requires WS chain lookup before submission
                        'target_strike': _target_strike,
                        'target_dte_range': f'{_dte}±15',
                        'target_contracts': _target_contracts,
                        'incremental_contracts_vs_existing': _add_contracts,
                        'est_credit_per_contract': _est_credit,
                        'est_total_credit': round(_est_credit * _add_contracts * 100, 0),
                        'dba_spot': round(_dba_spot, 2),
                        'allocation_dollars': round(_alloc_dollars, 0),
                        'existing_dba_collateral': _existing_dba_collateral,
                        'factor_tilt': _tilt,
                        'rationale': (
                            f'DBA WHEEL (soft allocator): saturation '
                            f'${_alloc_dollars:,.0f} ({_tgts.get("DBA", 0.1):.0%} NAV '
                            f'x {_size_mult:.1f}x tilt, score {_tilt.get("score", "?")}); '
                            f'held ${_existing_dba_collateral:,.0f} → gap ${_gap:,.0f}, '
                            f'this cycle +{_add_contracts}c (step {_step:.0%}) at '
                            f'P{_target_strike:.0f} {_dte}d ~${_est_credit:.2f}cr. '
                            f'{_otm:.0%} OTM. Real-chain 8.5y: +36.6%/Sharpe 1.82. '
                            f'REQUIRES CONSULT.'),
                        'priority': 'medium' if _add_contracts > 0 else 'low',
                        'requires_consult': True,
                    })
            # ── ENSO satellite pair (CORN + CANE always-on, ONI size tilt) ─
            _ag = _comp.get('ag_single_leg') or {}
            for _leg in (_ag.get('legs') or []):
                _ag_tk = _leg.get('ticker')
                if not _ag_tk or (nav or 0) <= 0:
                    continue
                _ag_spot = None
                try:
                    import yfinance as _yf2
                    _ag_spot = float(getattr(_yf2.Ticker(_ag_tk).fast_info,
                                             'last_price', 0) or 0)
                except Exception:
                    pass
                if not _ag_spot or _ag_spot <= 0:
                    continue
                _ag_mult = float(_leg.get('size_mult', 1.0))
                _ag_otm = 0.03  # thin chains → wider strike
                _ag_step_K = 0.5 if _ag_spot < 12 else 1.0
                _ag_K = round(_ag_spot * (1 - _ag_otm) / _ag_step_K) * _ag_step_K
                _tgts2 = _comp.get('portfolio_targets') or {}
                _ag_target_pct = float(_tgts2.get(_ag_tk,
                                                  _ag.get('nav_pct_cap_each', 0.05)))
                _ag_alloc = (nav or 0) * _ag_target_pct * _ag_mult  # saturation
                _ag_held = _ag_put_collateral(positions, _ag_tk)
                _ag_gap = max(0, _ag_alloc - _ag_held)
                _ag_step = float(_tgts2.get('step_per_cycle', 0.33))
                _ag_n = min(int(_ag.get('max_contracts', 5)),
                            int(_ag_gap * _ag_step / (_ag_K * 100)))
                if _ag_n == 0 and _ag_gap >= _ag_K * 100:
                    _ag_n = 1  # gap exists — soft-step at least 1c
                if _ag_n <= 0:
                    continue  # saturated — no rec this cycle
                _ag_credit = round(_ag_spot * 0.025, 2)
                out.setdefault('_pending_boxx_orders', []).append({
                    'order_type': f'SELL_PUT_{_ag_tk}',
                    'side': 'SELL_TO_OPEN',
                    'symbol': _ag_tk,
                    'sec_id': None,
                    'target_strike': _ag_K,
                    'target_dte_range': '60±15',
                    'target_contracts': _ag_n,
                    'est_credit_per_contract': _ag_credit,
                    'est_total_credit': round(_ag_credit * _ag_n * 100, 0),
                    'spot': round(_ag_spot, 2),
                    'allocation_dollars': round(_ag_alloc, 0),
                    'size_mult': _ag_mult,
                    'rationale': (
                        f'ENSO SATELLITE ({_ag_tk}, always-on pair): '
                        f'{_ag.get("reason", "")}. {_ag_n}x P{_ag_K:.1f} ~60d '
                        f'(~3% OTM, ~${_ag_credit:.2f}cr, tilt {_ag_mult:.1f}x). '
                        f'Cap {_ag.get("nav_pct_cap_each", 0.05):.0%} NAV/leg, '
                        f'{_ag.get("max_contracts", 5)}c (thin chain). '
                        f'1446(f) qualified-notice exempt (Teucrium). REQUIRES CONSULT.'),
                    'priority': 'low',
                    'requires_consult': True,
                })
    except Exception as _comp_err:
        out['composite_error'] = str(_comp_err)

    # ── LIVE IV-RANK (real ATM IV vs 252d history; new kernels act on it) ─
    try:
        import sys as _ivsys
        _gexd = os.path.join(os.path.dirname(THIS_DIR), 'research/gex')
        if _gexd not in _ivsys.path:
            _ivsys.path.insert(0, _gexd)
        from live_wall import current_iv_rank
        _ivr_live = current_iv_rank('UNG', spot)
        if _ivr_live:
            _r = _ivr_live.get('iv_rank')
            _ivr_live['regime'] = ('RICH-VOL top zone — kernel halves share adds'
                                   if _r is not None and _r > 0.8 else
                                   'elevated — kernel trims adds x0.8'
                                   if _r is not None and _r > 0.6 else
                                   'CHEAP-VOL — kernel boosts accumulation x1.3'
                                   if _r is not None and _r < 0.2 else 'neutral')
            out['iv_rank_live'] = _ivr_live
    except Exception:
        pass

    # ── EXEC TIMING RECOMMENDER (validated: 1,044-day minute study) ─────
    # Thursday bleeds -40bps intraday (print day) → best put-sell strikes
    # arrive in the afternoon; Tue opens -40bps (cheap adds); vol collapses
    # 14:30-15:30 (tightest option quotes); TOM = NG-expiry churn; never
    # trade the first 60s after the 10:30 print (knee-jerk reverses, r=-.19).
    try:
        from datetime import datetime as _dtt
        import pytz as _pytz
        _now = _dtt.now(_pytz.timezone('US/Eastern'))
        _dow, _hm = _now.weekday(), _now.strftime('%H:%M')
        _tips = []
        if _dow == 3:
            _tips.append('THURSDAY: print day bleeds -40bps — SELL PUTS in the '
                         '14:30-15:30 window (post-print dip = better strikes); '
                         'no orders 10:30-11:00; wait 60s+ after the print')
        elif _dow == 1:
            _tips.append('TUESDAY: overnight bled -40bps into the open — '
                         'best day for SHARE ADDS (morning entries land cheap)')
        elif _dow == 4:
            _tips.append('FRIDAY: tape favors longs (+13bps) — good day to let '
                         'runners run; price CALLS Mon close or today')
        elif _dow == 0:
            _tips.append('MONDAY: neutral; sell CALLS near the close (ahead of '
                         'the Tuesday-overnight bleed)')
        if _now.day >= 27 or _now.day <= 2:
            _tips.append('TURN-OF-MONTH: NG expiry churn (-43bps/d) — defer '
                         'share adds until day 3+')
        if _hm < '11:00':
            _tips.append(f'now {_hm} ET: morning vol is 1.5-2x afternoon — '
                         'WAIT for 14:30-15:30 to roll/sell unless urgent')
        elif '14:30' <= _hm <= '15:30':
            _tips.append(f'now {_hm} ET: IN the optimal window (lowest vol, '
                         'tightest quotes) — execute pending rolls/sells now')
        elif _hm > '15:30':
            _tips.append(f'now {_hm} ET: late session — take the bid on '
                         'resting orders rather than carrying overnight')
        out['exec_timing'] = _tips
    except Exception:
        pass

    # ── EXECUTOR BRIEF: directional ag engine state (for the human) ─────
    try:
        import json as _json3
        import time as _t3
        _dpath = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'research/dba/cache/directional_state.json')
        if os.path.exists(_dpath):
            with open(_dpath) as _df:
                _dstate = _json3.load(_df)
            _dstate['age_days'] = round(
                (_t3.time() - os.path.getmtime(_dpath)) / 86400, 1)
            out['directional_ag'] = _dstate
    except Exception:
        pass

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
        # Per-kernel beam — each kernel produces its own ranked candidates
        out['beam_by_kernel'] = _beam_per_kernel(spot, snap['z_surprise'],
                                                  nav or 100000)
    except Exception as e:
        out['beam_analysis_error'] = str(e)

    # ─── DIRECTLY USABLE TRADE ORDERS — kernel-specific, NAV-sized ─────────
    try:
        out['actionable_orders'] = _build_actionable_orders(
            kernel_info, spot, nav or 100000, current_shares,
            current_short_calls, current_put_collateral, snap,
            pending_assign_shares=pending_assign_shares,
            pending_assign_calls=pending_assign_calls,
            likely_assign_legs=likely_assign_legs,
        )
    except Exception as e:
        out['actionable_orders_error'] = str(e)

    # ── STATISTICAL WHAT-IF on recommendations ──────────────────────────
    # Every actionable PUT/CC rec gets an EMPIRICAL outcome distribution
    # (overlapping UNG return windows since 2018 — fat tails included, no
    # normality assumption): E[PnL], P(assign), P(loss), p5/p95, CVaR5.
    # Decisions read off the DISTRIBUTION, not the point EV.
    try:
        import pandas as _wpd
        import numpy as _np
        _mp = _wpd.read_csv(os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'research/dba/cache/master_panel.csv'),
            index_col=0, parse_dates=True)['UNG'].dropna().loc['2018':]
        _hor = _mp.pct_change(24).dropna()      # ~35 cal DTE in trading days
        _ST = spot * (1 + _hor.values)
        for _o in (out.get('actionable_orders') or []):
            if _o.get('order_type') not in ('PUT_SHORT_MIX', 'CALL_SHORT_COVERED'):
                continue
            try:
                _pnl = _np.zeros(len(_ST))
                _hit = _np.zeros(len(_ST), dtype=bool)
                _legs = _o.get('legs') or [{'strike': _o.get('strike'),
                                            'passive_tier_price': _o.get('est_premium_per'),
                                            'qty': _o.get('qty', 1)}]
                for _l in _legs:
                    _K = float(_l.get('strike') or 0)
                    _cr = float(_l.get('passive_tier_price')
                                or _l.get('est_premium_per') or 0)
                    _q = int(_l.get('qty') or 0)
                    if _K <= 0 or _q <= 0:
                        continue
                    if _o['order_type'] == 'PUT_SHORT_MIX':
                        _pnl += (_cr - _np.maximum(_K - _ST, 0)) * 100 * _q
                        _hit |= (_ST < _K)
                    else:
                        _pnl += (_cr - _np.maximum(_ST - _K, 0)) * 100 * _q
                        _hit |= (_ST > _K)
                _p5, _p95 = _np.percentile(_pnl, [5, 95])
                _o['whatif_stats'] = {
                    'e_pnl': round(float(_pnl.mean()), 0),
                    'p_assign': round(float(_hit.mean()), 2),
                    'p_loss': round(float((_pnl < 0).mean()), 2),
                    'p5_pnl': round(float(_p5), 0),
                    'p95_pnl': round(float(_p95), 0),
                    'cvar5': round(float(_pnl[_pnl <= _p5].mean()), 0),
                    'n_scenarios': int(len(_pnl)),
                    'basis': 'empirical 35d UNG windows 2018+ (fat tails, no normality)',
                }
            except Exception:
                continue
    except Exception:
        pass

    # ─── MERGE BOXX ORDERS + RE-SCORE ────────────────────────────────────
    # BOXX orders were computed before _build_actionable_orders; now merge
    # them in and re-run the EV scoring loop so they compete for best_play.
    boxx_orders = out.pop('_pending_boxx_orders', None)
    if boxx_orders:
        existing = out.get('actionable_orders') or []
        for bo in boxx_orders:
            # Score BOXX orders
            kind = bo.get('order_type', '')
            if kind == 'BUY_BOXX':
                ev = float(bo.get('est_cost', 0)) * 0.0474  # 1-yr standing yield
            elif kind == 'SELL_BOXX':
                ev = float(bo.get('est_proceeds', 0)) * 0.01
            elif kind.startswith('SELL_PUT_'):  # DBA core + CORN/CANE satellite
                # EV = total credit if expires worthless; weighted by edge
                # magnitude (higher edge → higher probability of OTM expiry)
                _credit = float(bo.get('est_total_credit', 0))
                _edge = float(out.get('composite_state', {}).get('dba_edge', 0))
                # probability OTM ~ 0.55 + 0.25*edge (clipped); EV = credit * prob
                _p_otm = max(0.4, min(0.85, 0.55 + 0.25 * _edge))
                ev = _credit * _p_otm
            else:
                ev = 0
            pri_mult = {'high': 1.0, 'medium': 0.7, 'low': 0.4}.get(
                bo.get('priority', 'low'), 0.4)
            bo['expected_ev_dollars'] = round(ev, 0)
            bo['ranked_score'] = round(ev * pri_mult, 0)
            existing.append(bo)
        out['actionable_orders'] = existing
        # Re-pick best play across combined set
        actionable = [o for o in existing if o.get('ranked_score', 0) > 0]
        if actionable:
            # Clear prior best_play flag
            for o in existing:
                o.pop('best_play', None)
            best = max(actionable, key=lambda o: o.get('ranked_score', 0))
            best['best_play'] = True

    # ─── PER-POSITION ANALYSIS + GREEKS ──────────────────────────────────
    try:
        out['position_analysis'] = _per_position_analysis(positions or [], spot, snap)
        out['portfolio_greeks'] = _portfolio_greeks(positions or [], spot)
        out['pnl_curve'] = _pnl_curve(positions or [], spot)
        out['delta_curve'] = _delta_curve(positions or [], spot)
        out['theta_by_expiry'] = _theta_by_expiry(positions or [], spot)
        out['theta_waterfall'] = _theta_waterfall(positions or [], spot)
        out['extrinsic'] = _extrinsic_and_smoothness(positions or [], spot)
        out['roll_plan'] = _roll_forward_plan(positions or [], spot, snap)
        out['whatif_matrix'] = _whatif_delta_matrix(positions or [], spot, out.get('portfolio_greeks', {}))
        out['calendar_grid'] = _calendar_grid(positions or [], spot)
        out['daily_status'] = _daily_status(out)
    except Exception as e:
        out['position_analysis_error'] = str(e)

    # Walk-forward truth disclosure
    out['warnings'].append('Walk-forward worst 12mo MDD: -17% (full-sample MDD -7% is sample-biased)')

    return out


def _beam_per_kernel(spot, z, nav):
    """Sophisticated beam for each kernel. Each kernel scores candidates
    using ITS OWN selection logic (OTM%, DTE, IV preference, share-target
    influence). Returns per-kernel ranking so user can compare what each
    kernel would do RIGHT NOW.
    """
    import math
    from scipy.stats import norm
    surf = _load_iv_surface()
    latest = max(surf.keys()) if surf else None
    out = {}
    for key, info in KERNELS.items():
        strat_name = info['strategy']
        # Each kernel has different OTM/DTE preferences (read from STRATEGIES)
        try:
            from replay_engine import STRATEGIES  # type: ignore
            sp = STRATEGIES.get(strat_name, {})
        except Exception:
            sp = {}
        default_otm = sp.get('otm_put', 0.10)
        default_dte = sp.get('open_dte', 45)
        # Check if kernel has ITM put gate enabled
        itm_z = sp.get('aggressive_itm_put_z')
        itm_pct = sp.get('itm_put_pct', -0.05)
        # Generate candidate ladder
        if itm_z is not None and z > itm_z:
            # Kernel would sell ITM here
            otms = [itm_pct, itm_pct + 0.02, itm_pct + 0.05]
            mode = 'ITM (z above kernel threshold)'
        else:
            otms = [default_otm * 0.5, default_otm, default_otm * 1.5, default_otm * 2.0]
            mode = 'OTM standard'

        # Score each candidate
        candidates = []
        for otm in otms:
            K = _snap_strike(spot * (1 - otm))
            iv = None
            if surf and latest:
                iv = iv_from_surface(surf, latest, K, default_dte, 'P')
            if iv is None: iv = 0.50
            T = default_dte / 365
            d1 = (math.log(spot/K) + (0.045 + 0.5*iv**2)*T) / (iv*math.sqrt(T))
            d2 = d1 - iv*math.sqrt(T)
            put_prem = K*math.exp(-0.045*T)*norm.cdf(-d2) - spot*norm.cdf(-d1)
            p_itm = float(norm.cdf(-d2))
            # Kernel-specific sizing: how many contracts?
            put_nav_pct = sp.get('put_qty_nav_pct')
            if put_nav_pct:
                # NAV-pct sizing
                qty = max(1, int(nav * put_nav_pct / (K * 100)))
                qty = min(qty, sp.get('put_qty_max', 100))
            else:
                # Fixed qty (per cycle, scaled by cadence)
                base_qty = sp.get('put_qty', 5)
                cadence = sp.get('entry_cadence', 7)
                qty = max(1, int(base_qty * cadence / 7))
            # Premium income vs expected loss
            income = put_prem * 100 * qty
            collateral = K * 100 * qty
            expected_loss = p_itm * max(0, K - spot * 0.95) * 100 * qty
            net_score = income - expected_loss
            efficiency = income / max(collateral, 1) * 100  # income as % of collateral
            candidates.append({
                'strike': K, 'otm_pct': round(otm * 100, 1),
                'dte': default_dte, 'iv': round(iv, 4),
                'premium_per_contract': round(put_prem, 3),
                'qty_recommended': qty,
                'total_income': round(income, 0),
                'collateral_required': round(collateral, 0),
                'p_itm_pct': round(p_itm * 100, 1),
                'expected_loss': round(expected_loss, 0),
                'net_score': round(net_score, 0),
                'eff_pct': round(efficiency, 2),
            })
        candidates.sort(key=lambda c: c['net_score'], reverse=True)
        out[key] = {
            'label': info['label'],
            'mode': mode,
            'winner_strike': candidates[0]['strike'] if candidates else None,
            'winner_qty': candidates[0]['qty_recommended'] if candidates else 0,
            'candidates': candidates,
        }
    return out


_LISTED_STRIKE_CACHE = {'ts': 0.0, 'strikes': None, 'date': None, 'dte': None}
_LIVE_CHAIN_CACHE = {'ts': 0.0, 'data': None}


def _query_live_chain(target_dte=45, right='P', tolerance_dte=14):
    """Pull TODAY's UNG option chain via yfinance.
    Returns dict: {'strikes': [...], 'expiration': '2026-07-18',
                   'liquidity': {strike: {bid, ask, vol, oi}}, 'source': 'yfinance_live'}
    Falls back to PG ung_iv_surface if yfinance unavailable.
    Cached 5 min.
    """
    import time
    now = time.time()
    cache_key = (target_dte, right)
    if _LIVE_CHAIN_CACHE['data'] and now - _LIVE_CHAIN_CACHE['ts'] < 300:
        cached = _LIVE_CHAIN_CACHE['data'].get(cache_key)
        if cached:
            return cached
    try:
        import yfinance as yf
        from datetime import date as _date, timedelta as _td
        ung = yf.Ticker('UNG')
        expirations = ung.options
        # Pick expiry closest to target_dte
        today = _date.today()
        best_exp = None
        best_diff = 10**9
        for exp_str in expirations:
            try:
                exp_d = _date.fromisoformat(exp_str)
                diff = abs((exp_d - today).days - target_dte)
                if diff < best_diff:
                    best_diff = diff
                    best_exp = exp_str
            except Exception:
                continue
        if not best_exp:
            return None
        chain = ung.option_chain(best_exp)
        side = chain.puts if right == 'P' else chain.calls
        strikes = sorted(float(s) for s in side['strike'].unique() if 1 <= s <= 50)
        def _safe_int(v):
            try:
                import math
                if v is None: return 0
                v = float(v)
                if math.isnan(v): return 0
                return int(v)
            except Exception:
                return 0
        def _safe_float(v):
            try:
                import math
                if v is None: return 0.0
                v = float(v)
                if math.isnan(v): return 0.0
                return v
            except Exception:
                return 0.0
        liquidity = {}
        for _, row in side.iterrows():
            K = float(row['strike'])
            liquidity[K] = {
                'bid': _safe_float(row.get('bid')),
                'ask': _safe_float(row.get('ask')),
                'vol': _safe_int(row.get('volume')),
                'oi': _safe_int(row.get('openInterest')),
            }
        result = {
            'strikes': strikes, 'expiration': best_exp,
            'liquidity': liquidity, 'source': 'yfinance_live',
            'dte': (_date.fromisoformat(best_exp) - today).days,
        }
        if _LIVE_CHAIN_CACHE['data'] is None:
            _LIVE_CHAIN_CACHE['data'] = {}
        _LIVE_CHAIN_CACHE['data'][cache_key] = result
        _LIVE_CHAIN_CACHE['ts'] = now
        return result
    except Exception as e:
        return None


def _query_real_listed_strikes(target_dte=45, right='P', tolerance_dte=14):
    """Query PG ung_iv_surface for the actual listed strikes most recently
    available near target_dte. Returns sorted list of strike_adj values
    (in yfinance/adjusted units the engine uses).

    Caches result for 10 minutes to avoid hammering DB on every refresh.
    """
    import time
    now = time.time()
    cache_key = (target_dte, right)
    cached = _LISTED_STRIKE_CACHE
    if (cached['strikes'] is not None and now - cached['ts'] < 600
            and cached.get('dte') == target_dte):
        return cached['strikes']
    try:
        import psycopg2
        conn = psycopg2.connect(
            host='192.168.1.172', port=5432, database='market_scanner',
            user='postgres', password='shinobi2025', connect_timeout=5,
        )
        cur = conn.cursor()
        # Get the latest date in surface with dte close to target
        cur.execute('SELECT MAX(date) FROM ung_iv_surface')
        latest_date = cur.fetchone()[0]
        if not latest_date:
            conn.close()
            return None
        cur.execute(
            """
            SELECT DISTINCT strike_adj
            FROM ung_iv_surface
            WHERE date = %s AND option_right = %s
              AND ABS(dte - %s) <= %s
            ORDER BY strike_adj
            """,
            (latest_date, right, target_dte, tolerance_dte),
        )
        rows = cur.fetchall()
        conn.close()
        strikes = sorted(float(r[0]) for r in rows)
        if not strikes:
            return None
        _LISTED_STRIKE_CACHE.update({
            'ts': now, 'strikes': strikes,
            'date': str(latest_date), 'dte': target_dte,
        })
        return strikes
    except Exception:
        return None


def _real_strikes_near(target_K, spot, increment=0.5, available=None):
    """Real UNG strikes. If `available` (sorted list) provided, snap to those.
    Otherwise fall back to fixed-increment grid.
    """
    if available:
        # Find bracketing strikes in the real available list
        below = [s for s in available if s <= target_K]
        above = [s for s in available if s > target_K]
        if not below and above:
            return [(above[0], 1.0)]
        if not above and below:
            return [(below[-1], 1.0)]
        if not above or not below:
            return [(target_K, 1.0)]  # shouldn't happen
        lo = below[-1]
        hi = above[0]
        if abs(target_K - lo) < 0.01:
            return [(round(lo, 2), 1.0)]
        w_upper = (target_K - lo) / (hi - lo)
        return [(round(lo, 2), 1.0 - w_upper), (round(hi, 2), w_upper)]
    # Fallback: fixed increment
    lower = int(target_K / increment) * increment
    upper = lower + increment
    if abs(target_K - lower) < 0.01:
        return [(round(lower, 2), 1.0)]
    w_upper = (target_K - lower) / increment
    return [(round(lower, 2), 1.0 - w_upper), (round(upper, 2), w_upper)]


def _strike_mix_for_target_otm(target_otm_pct, spot, total_qty,
                                increment=0.5, available_strikes=None):
    """Build a strike mix that approximates the ideal target OTM%.
    Uses REAL listed strikes from PG if available, else fixed increments.
    """
    target_K = spot * (1 - target_otm_pct)
    weighted = _real_strikes_near(target_K, spot, increment, available_strikes)
    if total_qty < 2 or len(weighted) == 1:
        best_K = min((s for s, _ in weighted),
                     key=lambda s: abs(s - target_K))
        return [{'strike': best_K, 'qty': total_qty,
                 'effective_otm_pct': round((1 - best_K/spot) * 100, 2),
                 'source': 'pg_real' if available_strikes else 'fixed_increment'}]
    qty_upper = max(1, round(total_qty * weighted[1][1]))
    qty_lower = total_qty - qty_upper
    if qty_lower < 1:
        qty_lower = 1
        qty_upper = total_qty - 1
    return [
        {'strike': weighted[0][0], 'qty': qty_lower,
         'effective_otm_pct': round((1 - weighted[0][0]/spot) * 100, 2),
         'target_weight': round(weighted[0][1], 2),
         'source': 'pg_real' if available_strikes else 'fixed_increment'},
        {'strike': weighted[1][0], 'qty': qty_upper,
         'effective_otm_pct': round((1 - weighted[1][0]/spot) * 100, 2),
         'target_weight': round(weighted[1][1], 2),
         'source': 'pg_real' if available_strikes else 'fixed_increment'},
    ]


def _premium_limit_ladder(mid_estimate, qty, side='SELL', max_tiers=4):
    """Build a passive→aggressive limit-price ladder so ONE order set fills
    without modification. Avoids spread chasing.

    Auto-shrinks tier count when qty < max_tiers so we never emit ladder
    rows with no real qty backing.
    """
    if qty < 1: return []
    # Use min(qty, max_tiers) tiers — never more rows than contracts
    n_tiers = min(qty, max_tiers)
    if n_tiers == 1:
        # Single contract: just one passive tier at mid
        return [{'tier': 1, 'qty': 1, 'limit_price': round(max(0.01, mid_estimate), 2),
                 'kind': 'mid'}]
    # Quantity split favors passive (40/30/20/10 for 4-tier; auto-prorated)
    full_weights = [0.4, 0.3, 0.2, 0.1]
    weights = full_weights[:n_tiers]
    weights = [w / sum(weights) for w in weights]
    qtys = [max(1, int(round(qty * w))) for w in weights]
    diff = qty - sum(qtys)
    if diff > 0: qtys[0] += diff
    elif diff < 0:
        # Need to subtract — pull from largest-qty tier
        for i in range(n_tiers):
            while qtys[i] > 1 and sum(qtys) > qty:
                qtys[i] -= 1
            if sum(qtys) == qty: break
    if side == 'SELL':
        full_offsets = [0.05, 0.02, -0.01, -0.04]
    else:
        full_offsets = [-0.05, -0.02, 0.01, 0.04]
    offsets = full_offsets[:n_tiers]
    full_labels = ['passive', 'near-mid', 'mid', 'cross']
    labels = full_labels[:n_tiers]
    return [
        {'tier': i + 1, 'qty': q,
         'limit_price': round(max(0.01, mid_estimate * (1 + off)), 2),
         'kind': label}
        for i, (q, off, label) in enumerate(zip(qtys, offsets, labels))
    ]


def _build_actionable_orders(kernel_info, spot, nav, current_shares,
                              current_short_calls, current_put_collateral, snap,
                              pending_assign_shares=0, pending_assign_calls=0,
                              likely_assign_legs=None):
    """Build directly-executable orders specific to the active kernel.
    Includes OSI option symbols, limit prices, qtys based on current NAV.

    Assignment-aware: pending_assign_shares is the count of shares that will
    auto-divest within ~7d via ITM call assignment. The builder subtracts
    these from any BTC-to-free-shares precondition so we don't pay BTC for
    work that assignment does for free.
    """
    likely_assign_legs = likely_assign_legs or []
    from datetime import date as _date, timedelta
    import math
    try:
        from replay_engine import STRATEGIES  # type: ignore
        sp = STRATEGIES.get(kernel_info['strategy'], {})
    except Exception:
        sp = {}

    orders = []
    z = snap['z_surprise']
    today = _date.today()
    # Target expiry: 45 DTE → Friday closest to that date
    # LIQUIDITY-AWARE EXPIRY SELECTION:
    # Try live chain (yfinance) first — picks an expiration that actually
    # has open contracts. Fallback to "nearest 3rd-Friday monthly" which
    # is the most reliably liquid expiry for UNG.
    target_dte_pref = sp.get('open_dte', 45)
    target_date = None
    try:
        chain = _query_live_chain(target_dte=target_dte_pref, right='P')
        if chain and chain.get('strikes'):
            target_date = _date.fromisoformat(chain['expiration'])
    except Exception:
        target_date = None
    if target_date is None:
        # Fallback: nearest monthly = 3rd Friday of nearest month to target DTE
        ideal = today + timedelta(days=target_dte_pref)
        # Find 3rd Friday of ideal.month
        def _third_friday(year, month):
            from datetime import date as _date2
            d = _date2(year, month, 1)
            # 1st Friday
            offset = (4 - d.weekday()) % 7
            return _date2(year, month, 1 + offset + 14)
        candidates = []
        for delta in (-1, 0, 1):
            y, m = ideal.year, ideal.month + delta
            if m < 1:  y -= 1; m += 12
            if m > 12: y += 1; m -= 12
            try:
                candidates.append(_third_friday(y, m))
            except Exception:
                continue
        # Pick the candidate closest to ideal (prefer post-ideal if tie)
        target_date = min(candidates, key=lambda d: (abs((d - ideal).days), -d.toordinal()))
    exp_str = target_date.isoformat()
    exp_osi = target_date.strftime('%y%m%d')

    def _bs_call_put(K, T_y, iv_=0.50):
        """BSM call & put prices."""
        from scipy.stats import norm
        d1 = (math.log(spot/K) + (0.045 + 0.5*iv_**2)*T_y) / (iv_*math.sqrt(T_y))
        d2 = d1 - iv_*math.sqrt(T_y)
        call = spot*norm.cdf(d1) - K*math.exp(-0.045*T_y)*norm.cdf(d2)
        put = K*math.exp(-0.045*T_y)*norm.cdf(-d2) - spot*norm.cdf(-d1)
        return call, put

    # SHARE order: regime-aware target
    z_mults = sp.get('z_target_mults', {
        'extreme_cheap': 1.7, 'cheap': 1.4, 'neutral': 1.0,
        'rich': 0.4, 'extreme_rich': 0.1})
    if z < -1.5: mult = z_mults['extreme_cheap']
    elif z < -0.5: mult = z_mults['cheap']
    elif z < 0.5: mult = z_mults['neutral']
    elif z < 1.0: mult = z_mults['rich']
    else: mult = z_mults['extreme_rich']

    pct_base = sp.get('z_share_target_pct_nav')
    if pct_base and spot > 0:
        target_total = int(nav * pct_base * mult / spot / 100) * 100
    else:
        target_total = int(sp.get('z_share_target_base', 6200) * mult / 100) * 100

    # Apply gradual move (50% toward target per cycle)
    share_delta = (target_total - current_shares) // 2
    share_delta = (share_delta // 100) * 100
    if abs(share_delta) >= 100:
        side = 'BUY' if share_delta > 0 else 'SELL'

        # COVERED-CALLS-ONLY GUARD: cannot sell shares below short-call coverage.
        # max safe sell = current_shares - current_short_calls * 100
        desired_sell = -share_delta if side == 'SELL' else 0
        max_safe_sell = max(0, current_shares - current_short_calls * 100)
        capped_sell = min(desired_sell, max_safe_sell) if side == 'SELL' else 0
        shortfall = desired_sell - capped_sell  # shares we WANT to sell but can't yet

        # ASSIGNMENT-AWARE: imminent ITM short calls will auto-divest shares
        # for free. Subtract that from the shortfall before suggesting any BTC.
        assignment_will_handle = min(pending_assign_shares, shortfall) if side == 'SELL' else 0
        residual_shortfall = max(0, shortfall - assignment_will_handle)

        if side == 'SELL' and assignment_will_handle > 0:
            # Surface the "wait for assignment" plan as a first-class order
            legs_desc = '; '.join(
                f"{l['qty']}× ${l['strike']:.2f} {l['expiry']} ({l['dte']}d, +{l['itm_pct']}% ITM)"
                for l in likely_assign_legs[:5]
            )
            orders.append({
                'order_type': 'WAIT_FOR_ASSIGNMENT',
                'side': 'HOLD',
                'symbol': 'UNG (ITM short calls near expiry)',
                'qty': pending_assign_calls,
                'qty_total': pending_assign_shares,
                'rationale': (f'Kernel wants to sell {desired_sell} shares. '
                              f'{pending_assign_calls} ITM short calls (≤14d DTE) will '
                              f'auto-assign {pending_assign_shares} shares — '
                              f'don\'t pay BTC for what assignment does for free. '
                              f'Legs: {legs_desc}.'),
                'priority': 'medium',  # passive — assignment auto-happens, no order needed
            })

        if side == 'SELL' and residual_shortfall > 0:
            # Only after assignment, suggest BTC for whatever still doesn't fit
            calls_to_close = (residual_shortfall + 99) // 100
            calls_to_close = min(calls_to_close, current_short_calls - pending_assign_calls)
            if calls_to_close > 0:
                orders.append({
                    'order_type': 'CC_BTC_TO_FREE_SHARES',
                    'side': 'BUY_TO_CLOSE',
                    'symbol': 'UNG short calls (pick lowest-extrinsic, non-assigning leg)',
                    'qty': calls_to_close,
                    'rationale': (f'RESIDUAL precondition (after assignment): kernel wants to sell '
                                  f'{desired_sell} shares; assignment handles {assignment_will_handle}; '
                                  f'still short {residual_shortfall} shares. '
                                  f'BTC {calls_to_close} more calls (pick legs that are NOT in the '
                                  f'pending-assignment list — those will assign anyway). '
                                  f'Prefer lowest-extrinsic legs to minimize buyback cost.'),
                    'priority': 'medium',
                })

        # Build the SHARES order at the SAFE qty (capped, may be 0)
        effective_qty = capped_sell if side == 'SELL' else abs(share_delta)
        if effective_qty >= 100:
            l1 = round(spot * (1.005 if side == 'BUY' else 0.995), 2)
            l2 = round(spot, 2)
            l3 = round(spot * (0.995 if side == 'BUY' else 1.005), 2)
            per_tier_qty = effective_qty // 3
            cap_note = ''
            if side == 'SELL' and capped_sell < desired_sell:
                cap_note = (f' [CAPPED from {desired_sell} → {capped_sell} by covered-calls '
                            f'constraint; close calls first to sell the rest]')
            orders.append({
                'order_type': 'SHARES',
                'side': side,
                'symbol': 'UNG',
                'qty_total': effective_qty,
                'limit_ladder': [
                    {'qty': per_tier_qty, 'limit_price': l1},
                    {'qty': per_tier_qty, 'limit_price': l2},
                    {'qty': effective_qty - 2*per_tier_qty, 'limit_price': l3},
                ],
                'rationale': (f'Rebalance toward {target_total} shares (current {current_shares}, '
                              f'z={z:+.2f}, mult={mult}). Move 50%/cycle.{cap_note}'),
                'priority': 'high' if effective_qty >= 200 else 'medium',
            })
        elif side == 'SELL' and capped_sell == 0 and desired_sell > 0:
            # 100% blocked — surface clearly
            orders.append({
                'order_type': 'SHARES_SELL_BLOCKED',
                'side': 'BLOCKED',
                'symbol': 'UNG',
                'qty': 0,
                'rationale': (f'Kernel wants to sell {desired_sell} shares to reach {target_total}, '
                              f'but ALL {current_shares} shares are covering {current_short_calls} '
                              f'short calls. Must BTC calls first (see CC_BTC_TO_FREE_SHARES order). '
                              f'Covered-calls-only rule.'),
                'priority': 'high',
            })

        # ─── PUT-CALL PARITY SYNTHETIC ALTERNATIVE ────────────────────────
        # Same delta exposure without paying for shares upfront.
        # Synthetic long stock = +1 call + (-1) put at same K/exp
        # Synthetic short stock = (-1) call + (+1) put — needs share coverage for call leg
        from datetime import timedelta as _td
        synth_target = today + _td(days=30)
        while synth_target.weekday() != 4:
            synth_target += _td(days=1)
        synth_exp = synth_target.isoformat()
        synth_exp_osi = synth_target.strftime('%y%m%d')
        K_synth = round(spot, 0)  # ATM
        T_synth = 30 / 365
        c_prem, p_prem = _bs_call_put(K_synth, T_synth)
        net_debit = c_prem - p_prem  # synthetic-long net (typically small at ATM)
        synth_qty = max(1, abs(share_delta) // 100)
        synth_qty = min(synth_qty, 50)  # cap

        if share_delta > 0:
            # Synthetic LONG: buy call + sell put. Net delta = +100 per contract.
            # No coverage constraint on either leg (BTO call + STO put are fine).
            orders.append({
                'order_type': 'SYNTHETIC_LONG_PARITY',
                'side': 'BUY_CALL + SELL_PUT',
                'legs': [
                    {'side': 'BUY_TO_OPEN',  'symbol': f'UNG   {synth_exp_osi}C{int(K_synth*1000):08d}',
                     'right': 'CALL', 'strike': K_synth, 'qty': synth_qty,
                     'est_premium_per': round(c_prem, 3)},
                    {'side': 'SELL_TO_OPEN', 'symbol': f'UNG   {synth_exp_osi}P{int(K_synth*1000):08d}',
                     'right': 'PUT',  'strike': K_synth, 'qty': synth_qty,
                     'est_premium_per': round(p_prem, 3)},
                ],
                'underlying': 'UNG',
                'expiry': synth_exp,
                'qty': synth_qty,
                'net_debit_per_pair': round(net_debit, 3),
                'net_delta_per_pair': 100,
                'capital_efficiency': 'No shares purchased; only put collateral required',
                'collateral_required': round(K_synth * 100 * synth_qty, 0),
                'rationale': (f'Put-call parity LONG: same +{synth_qty*100}Δ as buying shares '
                              f'but no upfront stock purchase. Net debit ~${net_debit*100*synth_qty:+.0f}. '
                              f'Useful when cash is short or you want optional 30d exit.'),
                'priority': 'low',
            })
        else:
            # Synthetic SHORT: sell call + buy put. The short call REQUIRES share coverage.
            uncovered_synth = current_shares - current_short_calls * 100
            if uncovered_synth >= synth_qty * 100:
                orders.append({
                    'order_type': 'SYNTHETIC_SHORT_PARITY',
                    'side': 'SELL_CALL + BUY_PUT',
                    'legs': [
                        {'side': 'SELL_TO_OPEN', 'symbol': f'UNG   {synth_exp_osi}C{int(K_synth*1000):08d}',
                         'right': 'CALL', 'strike': K_synth, 'qty': synth_qty,
                         'est_premium_per': round(c_prem, 3)},
                        {'side': 'BUY_TO_OPEN',  'symbol': f'UNG   {synth_exp_osi}P{int(K_synth*1000):08d}',
                         'right': 'PUT',  'strike': K_synth, 'qty': synth_qty,
                         'est_premium_per': round(p_prem, 3)},
                    ],
                    'underlying': 'UNG',
                    'expiry': synth_exp,
                    'qty': synth_qty,
                    'net_credit_per_pair': round(c_prem - p_prem, 3),
                    'net_delta_per_pair': -100,
                    'capital_efficiency': 'Avoids tax realization vs share sale',
                    'cc_coverage_check': f'OK: {uncovered_synth} uncovered ≥ {synth_qty*100} needed',
                    'rationale': (f'Put-call parity SHORT: synthetic -{synth_qty*100}Δ without '
                                  f'selling shares (avoids realized gain/loss). Covered by '
                                  f'{uncovered_synth} uncovered shares.'),
                    'priority': 'low',
                })
            else:
                orders.append({
                    'order_type': 'SYNTHETIC_SHORT_BLOCKED',
                    'side': 'BLOCKED',
                    'rationale': (f'Cannot create synthetic short: need {synth_qty*100} uncovered '
                                  f'shares to cover the short-call leg, only have {uncovered_synth}. '
                                  f'Covered-calls-only rule.'),
                    'priority': 'low',
                })

    # PUT order: short put with kernel's strike preference
    itm_z = sp.get('aggressive_itm_put_z')
    if itm_z is not None and z > itm_z:
        strike_pct = sp.get('itm_put_pct', -0.05)  # negative = ITM
        K = _snap_strike(spot * (1 - strike_pct))
        order_label = f'SHORT_PUT_ITM (z>{itm_z} threshold met)'
    else:
        K = _snap_strike(spot * (1 - sp.get('otm_put', 0.10)))
        order_label = 'SHORT_PUT_OTM (standard)'
    # Estimate premium for limit price
    iv = 0.50  # fallback; surface lookup would be better but kernel handles that
    T = sp.get('open_dte', 45) / 365
    from scipy.stats import norm
    d1 = (math.log(spot/K) + (0.045 + 0.5*iv**2)*T) / (iv*math.sqrt(T))
    d2 = d1 - iv*math.sqrt(T)
    bsm_prem = K*math.exp(-0.045*T)*norm.cdf(-d2) - spot*norm.cdf(-d1)
    limit_low = round(bsm_prem * 0.90, 2)
    limit_high = round(bsm_prem * 1.10, 2)

    # Qty sizing
    put_nav_pct = sp.get('put_qty_nav_pct')
    if put_nav_pct:
        max_qty = max(1, int(nav * put_nav_pct / (K * 100)))
        max_qty = min(max_qty, sp.get('put_qty_max', 100))
    else:
        max_qty = max(1, int(sp.get('put_qty', 5) * sp.get('entry_cadence', 7) / 7))

    # ─── HIGH-CONVICTION ASSIGNMENT ANTICIPATION ─────────────────────────
    # Statistical model: only act when p_assign per leg is HUGELY convinced
    # (≥0.85 = 'cert' regime). Low/medium conviction → no action.
    # For each leg passing the threshold, treat its full qty as "expected to
    # be lost." Sum to expected_share_loss, then scale puts to replace a
    # fraction (default 50%) of those losses.
    #
    # This is gated per-leg (asynchronous belief): the 4× 1-DTE deep-ITM
    # calls trigger; the 14× 22-DTE coin-flip calls do not.
    huge_conviction_threshold = 0.85
    replacement_factor = 0.5  # replace 50% of expected losses
    expected_loss_shares = 0
    contributing_legs = []
    for leg in (likely_assign_legs or []):
        # likely_assign_legs already filters p_assign ≥ 0.55; re-check ≥ 0.85
        if leg.get('p_assign', 0) >= huge_conviction_threshold:
            expected_loss_shares += leg.get('qty', 0) * 100
            contributing_legs.append(leg)
    anticipation_bonus = 0
    if expected_loss_shares >= 100:  # at least one full contract worth
        anticipation_bonus = int((expected_loss_shares / 100) * replacement_factor)
        # Don't go nuts — cap bonus at +5 contracts and at 50% of base qty
        anticipation_bonus = min(anticipation_bonus, 5, max_qty)
        if anticipation_bonus > 0:
            max_qty += anticipation_bonus

    # Check current collateral usage (account for anticipated puts too)
    proposed_collateral = max_qty * K * 100
    total_after = current_put_collateral + proposed_collateral
    collateral_pct_after = total_after / nav if nav > 0 else 0
    if collateral_pct_after > 0.80:
        # Throttle to keep below 80% leverage
        room = nav * 0.80 - current_put_collateral
        max_qty = max(0, int(room / (K * 100)))

    if max_qty >= 1:
        # MIX-AND-MATCH: split across nearest real strikes so effective
        # OTM matches the kernel's ideal target. CRITICAL: use real strikes
        # for the CHOSEN expiry, not a generic grid — different expiries
        # list different strikes (e.g., monthly expiries are integer-only
        # while weeklies may include half-strikes).
        target_otm = (spot - K) / spot
        live_strikes_for_exp = None
        try:
            live_chain = _query_live_chain(target_dte=(target_date - today).days, right='P')
            if live_chain and live_chain.get('expiration') == exp_str:
                live_strikes_for_exp = live_chain.get('strikes')
        except Exception:
            live_strikes_for_exp = None
        # Infer increment from the strike list (median spacing) — falls back to 0.5
        inc = 0.5
        if live_strikes_for_exp and len(live_strikes_for_exp) >= 2:
            diffs = [live_strikes_for_exp[i+1] - live_strikes_for_exp[i]
                     for i in range(len(live_strikes_for_exp)-1)]
            diffs.sort()
            inc = diffs[len(diffs)//2]  # median
        strike_mix = _strike_mix_for_target_otm(target_otm, spot, max_qty,
                                                 increment=inc,
                                                 available_strikes=live_strikes_for_exp)
        # Build OSI legs for each strike in the mix
        legs = []
        total_credit = 0
        total_collateral = 0
        for slot in strike_mix:
            sK = slot['strike']
            sQ = slot['qty']
            # Re-price for this real strike
            d1s = (math.log(spot/sK) + (0.045 + 0.5*iv**2)*T) / (iv*math.sqrt(T))
            d2s = d1s - iv*math.sqrt(T)
            slot_prem = sK*math.exp(-0.045*T)*norm.cdf(-d2s) - spot*norm.cdf(-d1s)
            strike_osi = f'{int(sK * 1000):08d}'
            # LIMIT LADDER (4 tiers, passive → aggressive) — submit as a single
            # ladder so one fills without modifying. Avoids spread chase.
            ladder = _premium_limit_ladder(slot_prem, sQ, side='SELL')
            legs.append({
                'symbol': f'UNG   {exp_osi}P{strike_osi}',
                'strike': sK,
                'qty': sQ,
                'effective_otm_pct': slot['effective_otm_pct'],
                'est_premium_per': round(slot_prem, 3),
                'limit_low': round(slot_prem * 0.90, 2),
                'limit_high': round(slot_prem * 1.10, 2),
                'limit_ladder': ladder,   # NEW: 4-tier price ladder
                'credit_total': round(slot_prem * 100 * sQ, 0),
                'collateral': round(sK * 100 * sQ, 0),
            })
            total_credit += slot_prem * 100 * sQ
            total_collateral += sK * 100 * sQ
        avg_eff_otm = sum(l['effective_otm_pct'] * l['qty'] for l in legs) / max_qty
        orders.append({
            'order_type': 'PUT_SHORT_MIX',
            'side': 'SELL_TO_OPEN',
            'underlying': 'UNG',
            'expiry': exp_str,
            'right': 'PUT',
            'symbol': f'{len(legs)} real strikes',  # for display
            'qty': max_qty,
            'target_otm_pct': round(target_otm * 100, 2),
            'achieved_otm_pct': round(avg_eff_otm, 2),
            'est_credit_total': round(total_credit, 0),
            'collateral_required': round(total_collateral, 0),
            'legs': legs,
            'anticipation': {
                'bonus_qty': anticipation_bonus,
                'expected_loss_shares': expected_loss_shares,
                'contributing_legs': [
                    {'qty': l['qty'], 'strike': l['strike'],
                     'expiry': l['expiry'], 'p_assign': l['p_assign']}
                    for l in contributing_legs
                ],
                'huge_conviction_threshold': huge_conviction_threshold,
                'replacement_factor': replacement_factor,
            } if anticipation_bonus > 0 else None,
            'rationale': (
                f'{order_label}. Kernel target {target_otm*100:.1f}% OTM; '
                f'real strikes split into {len(legs)} legs to match. '
                f'Achieved avg {avg_eff_otm:.1f}% OTM.'
                + (f' +{anticipation_bonus} contracts anticipating high-conviction '
                   f'assignment of {expected_loss_shares} shares (p≥{huge_conviction_threshold}, '
                   f'replacement factor {replacement_factor}).' if anticipation_bonus > 0 else '')
            ),
            'priority': 'medium',
        })

    # CC order — sell calls ONLY if we have uncovered shares (constraint:
    # covered-calls-only, never naked). short_calls_qty * 100 shares are
    # already covering existing CCs; uncovered = shares - committed.
    uncovered = current_shares - current_short_calls * 100
    if uncovered < 100:
        # No room for new CCs — explicitly skip and tell user why
        orders.append({
            'order_type': 'CC_SKIPPED',
            'side': 'HOLD',
            'symbol': 'UNG',
            'qty': 0,
            'rationale': (f'No new CC: {current_short_calls} short calls already '
                          f'cover {current_short_calls*100} of {current_shares} shares '
                          f'(uncovered: {uncovered}). Covered-calls-only constraint.'),
            'priority': 'low',
        })
    elif uncovered >= 100:
        Kc = _snap_strike(spot * (1 + sp.get('otm_call', 0.05)))
        # GEX WALL FLOOR: never sell CCs below the dealer call wall.
        # Backtest (100 monthlies 2018-2026): final-week high stayed under
        # the wall 74% vs 69% for naive same-OTM strikes (+5pp hold rate).
        # Strike-selection overlay only — never affects qty/sizing.
        gex_note = ''
        _wall = None
        try:
            import os as _gos
            import sys as _gsys
            _gex_dir = _gos.path.join(
                _gos.path.dirname(_gos.path.dirname(_gos.path.abspath(__file__))),
                'research/gex')
            if _gex_dir not in _gsys.path:
                _gsys.path.insert(0, _gex_dir)
            from live_wall import current_gex_wall
            _wall = current_gex_wall('UNG', spot)
            if _wall and _wall['wall'] > Kc:
                gex_note = (f' Strike floored at GEX call wall ${_wall["wall"]:.1f} '
                            f'(was ${Kc:.2f}; wall +GEX ${_wall["wall_gex"]:,.0f}, '
                            f'74% final-week hold rate).')
                Kc = float(_wall['wall'])
        except Exception:
            pass
        cc_dte = sp.get('open_dte', 45)
        T = cc_dte / 365
        d1c = (math.log(spot/Kc) + (0.045 + 0.5*iv**2)*T) / (iv*math.sqrt(T))
        d2c = d1c - iv*math.sqrt(T)
        bsm_cc = spot*norm.cdf(d1c) - Kc*math.exp(-0.045*T)*norm.cdf(d2c)
        # qty: limited by uncovered shares
        if put_nav_pct:
            max_cc = max(1, int(nav * sp.get('call_qty_nav_pct', 0.04) / (Kc * 100)))
        else:
            max_cc = max(1, int(sp.get('call_qty', 5)))
        max_cc = min(max_cc, uncovered // 100)
        if max_cc >= 1:
            cc_osi = f'UNG   {exp_osi}C{int(Kc * 1000):08d}'
            orders.append({
                'order_type': 'CALL_SHORT_COVERED',
                'side': 'SELL_TO_OPEN',
                'symbol': cc_osi,
                'underlying': 'UNG',
                'strike': Kc,
                'expiry': exp_str,
                'right': 'CALL',
                'qty': max_cc,
                'limit_low': round(bsm_cc * 0.90, 2),
                'limit_high': round(bsm_cc * 1.10, 2),
                'limit_ladder': _premium_limit_ladder(bsm_cc, max_cc, side='SELL'),
                'est_premium_per': round(bsm_cc, 3),
                'est_credit_total': round(bsm_cc * 100 * max_cc, 0),
                'shares_covered': max_cc * 100,
                'gex_wall': _wall,
                'rationale': (f'CC at K=${Kc}, {cc_dte}d, covers {max_cc * 100} shares.'
                              + gex_note),
                'priority': 'low',
            })

    # ─── BEST PLAY SCORING — rank orders by expected dollar value ────────
    # Each actionable order gets a score = expected $EV * priority weight.
    # Surface the top pick at the front of the list with a "best_play" flag.
    pri_mult = {'high': 1.0, 'medium': 0.7, 'low': 0.4}
    for o in orders:
        ev = 0.0
        kind = o.get('order_type', '')
        if kind == 'PUT_SHORT_MIX':
            # EV = credit collected * (1 - p_assign_avg * 0.5)
            # Conservatively assume avg p_assign 25% for OTM puts
            ev = float(o.get('est_credit_total', 0)) * 0.75
        elif kind == 'CALL_SHORT_COVERED':
            ev = float(o.get('est_credit_total', 0)) * 0.65
        elif kind == 'SHARES':
            ev = abs(float(o.get('qty_total', 0))) * spot * 0.001  # tiny score
        elif kind == 'CC_BTC_TO_FREE_SHARES':
            # Negative EV — we're paying to free shares — but enables high-priority sell
            ev = -100 * o.get('qty', 0)
        elif kind == 'WAIT_FOR_ASSIGNMENT':
            # Implicit gain from assignment = (spot - K) avoided BTC cost
            ev = o.get('qty', 0) * 50  # heuristic
        elif kind == 'BUY_BOXX':
            # ONGOING annual yield on idle cash (this is standing income, not
            # one-shot). Cash sitting at 0% loses this yield every year.
            ev = float(o.get('est_cost', 0)) * 0.0474  # 1-yr yield captured
        elif kind == 'SELL_BOXX':
            # Enables put-collateral relief — score by deficit unlocked
            ev = float(o.get('est_proceeds', 0)) * 0.01  # 1% value for unlock
        elif kind in ('SYNTHETIC_SHORT_BLOCKED', 'CC_SKIPPED', 'SHARES_SELL_BLOCKED'):
            ev = 0  # not actionable
        o['expected_ev_dollars'] = round(ev, 0)
        o['ranked_score'] = round(ev * pri_mult.get(o.get('priority', 'low'), 0.4), 0)

    # Mark the best play
    actionable = [o for o in orders if o['ranked_score'] > 0]
    if actionable:
        best = max(actionable, key=lambda o: o['ranked_score'])
        best['best_play'] = True
        best['rationale'] = '⭐ BEST PLAY TODAY: ' + best.get('rationale', '')

    return orders


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


# Spread-friction half-spreads as a fraction of mid, calibrated to measured UNG
# chains (2026-06-12): near-term legs ~15% wide → ~8% half-spread; 45d legs
# ~28-30% wide → ~14% half-spread. Tunable if WS fills come in tighter/wider.
_ROLL_CLOSE_SPREAD_FRAC = 0.08
_ROLL_OPEN_SPREAD_FRAC = 0.14


def _roll_forward_plan(positions, spot, snap):
    """For each near-term contract, suggest a roll target + projected smoothness.

    Roll philosophy:
    - Near-DTE OTM puts → roll to 45d at same strike (let theta decay,
      then re-write). Only roll if extrinsic > 30% of premium target.
    - Near-DTE ITM CCs → likely assignment; suggest waiting unless we
      want to recover cost basis.
    - Net effect: shifts week-1 theta into week-4 and beyond, raising
      smoothness toward 0.75 target.
    """
    import math
    from datetime import date as _date, timedelta
    surf = _load_iv_surface()
    latest_surf = max(surf.keys()) if surf else None
    today = _date.today()
    roll_actions = []
    # Project hypothetical post-roll positions
    rolled_positions = []
    for p in positions:
        if p.get('symbol', '').upper() != 'UNG':
            continue
        if not p.get('is_option'):
            rolled_positions.append(p)
            continue
        qty = int(p.get('quantity', 0) or 0)
        if qty == 0:
            continue
        K = float(p.get('strike') or 0)
        right = 'C' if p.get('option_type') == 'CALL' else 'P'
        try:
            exp_d = _date.fromisoformat(p.get('expiry', ''))
            dte = (exp_d - today).days
        except Exception:
            continue
        if dte <= 0:
            continue
        moneyness = 'ITM' if ((right == 'C' and K < spot) or (right == 'P' and K > spot)) else \
                    'ATM' if abs(K - spot) <= 0.10 else 'OTM'
        # Roll candidates: near-DTE (<= 14d) OTM puts/calls
        if dte <= 14 and moneyness == 'OTM' and qty < 0:
            # Suggested target: 45d out, same OTM% from new spot
            new_dte = 45
            new_exp = (today + timedelta(days=new_dte)).isoformat()
            # Keep ~same OTM offset
            otm_pct = (spot - K) / spot if right == 'P' else (K - spot) / spot
            new_K = _snap_strike(spot * (1 - otm_pct) if right == 'P' else spot * (1 + otm_pct))
            # Estimate new premium
            iv = None
            if surf and latest_surf:
                iv = iv_from_surface(surf, latest_surf, new_K, new_dte, right)
            if iv is None:
                iv = 0.50
            T = new_dte / 365
            d1 = (math.log(spot/new_K) + (0.045 + 0.5*iv**2)*T) / (iv*math.sqrt(T))
            d2 = d1 - iv*math.sqrt(T)
            from scipy.stats import norm
            if right == 'C':
                new_prem = spot*norm.cdf(d1) - new_K*math.exp(-0.045*T)*norm.cdf(d2)
            else:
                new_prem = new_K*math.exp(-0.045*T)*norm.cdf(-d2) - spot*norm.cdf(-d1)
            # Close cost of existing position (intrinsic + small extrinsic)
            old_intrinsic = _intrinsic_value(K, spot, right)
            old_iv = iv_from_surface(surf, latest_surf, K, max(1, dte), right) if (surf and latest_surf) else 0.50
            if old_iv is None: old_iv = 0.50
            T_old = max(1, dte) / 365
            d1o = (math.log(spot/K) + (0.045 + 0.5*old_iv**2)*T_old) / (old_iv*math.sqrt(T_old))
            d2o = d1o - old_iv*math.sqrt(T_old)
            if right == 'C':
                old_val = spot*norm.cdf(d1o) - K*math.exp(-0.045*T_old)*norm.cdf(d2o)
            else:
                old_val = K*math.exp(-0.045*T_old)*norm.cdf(-d2o) - spot*norm.cdf(-d1o)
            # REAL-FILL FRICTION (measured UNG chains 2026-06-12): you BUY-to-close
            # the near-term leg at the ASK and SELL-to-open the 45d leg at the BID,
            # so each leg loses ~half its bid/ask spread vs mid. Near-term legs run
            # ~15% wide ($0.04 on $0.29), but 45d legs run ~28-30% wide ($0.17-0.19) —
            # the wide far leg dominates. The old model used BS mid for BOTH legs
            # (zero friction), overstating net credit by ~1 full far-spread/contract.
            # Half-spreads below; floored at $0.02/sh for thin options.
            close_hs = max(0.02, old_val * _ROLL_CLOSE_SPREAD_FRAC)   # pay above mid
            open_hs = max(0.02, new_prem * _ROLL_OPEN_SPREAD_FRAC)    # receive below mid
            eff_close = old_val + close_hs
            eff_open = max(0.0, new_prem - open_hs)
            close_cost = eff_close * 100 * abs(qty)
            new_credit = eff_open * 100 * abs(qty)
            # EXPIRE-AND-REOPEN (user insight, friction-aware): this leg is OTM and
            # ≤14d out — it will most likely expire WORTHLESS. Rolling it means
            # BUYING IT BACK (paying its residual value + close-leg friction) just
            # to re-sell a far leg. That's paying to close a penny. Instead: LET IT
            # EXPIRE ($0 cost, keep the premium already collected) and SELL the new
            # far leg fresh — friction is paid ONCE (the open), not twice. We keep
            # the close-cost math only to show what rolling would have wasted.
            # (Matches the backtest, which only rolls ITM puts and lets OTM expire.)
            open_friction = open_hs * 100 * abs(qty)
            roll_would_cost = close_cost  # buyback (incl. its friction) avoided
            roll_actions.append({
                'mode': 'EXPIRE_REOPEN',
                'old': {'right': right, 'strike': K, 'expiry': p.get('expiry'), 'qty': qty, 'dte': dte},
                'new': {'right': right, 'strike': new_K, 'expiry': new_exp, 'qty': qty, 'dte': new_dte},
                'new_credit_per_contract': round(eff_open, 3),
                'mid_open_per_contract': round(new_prem, 3),
                'mid_close_per_contract': round(old_val, 3),
                'friction_total': round(open_friction, 0),     # open leg only
                'net_credit_total': round(new_credit, 0),       # the new open credit
                'roll_would_cost': round(roll_would_cost, 0),   # what rolling wastes
                'savings_vs_roll': round(roll_would_cost, 0),
                'rationale': f'OTM, {dte}d to expiry → LET ${K} {right} EXPIRE ($0, keep premium), '
                             f'then SELL {abs(qty)}x ${new_K} {right} {new_dte}d for ~${new_credit:.0f} '
                             f'(open friction ${open_friction:.0f}). Rolling would waste '
                             f'~${roll_would_cost:.0f} buying back a near-worthless leg.',
            })
            # Synthetic rolled position for smoothness projection
            rolled_positions.append({
                **p,
                'strike': new_K,
                'expiry': new_exp,
                'average_price': new_prem,
            })
        else:
            rolled_positions.append(p)

    # Project smoothness assuming the new (reopened) legs are sold
    projected = _extrinsic_and_smoothness(rolled_positions, spot)
    return {
        'rolls': roll_actions,
        'roll_count': len(roll_actions),
        'mode': 'EXPIRE_REOPEN',
        'projected_smoothness': projected['smoothness'],
        'projected_weekly_theta': projected['weekly_theta'],
        'projected_avg_weekly': projected['avg_weekly_theta'],
        # net = sum of NEW open credits (the expiring OTM legs cost $0 to retire)
        'net_credit_total': round(sum(r['net_credit_total'] for r in roll_actions), 0),
        'friction_total': round(sum(r.get('friction_total', 0) for r in roll_actions), 0),
        # what you SAVE by letting OTM expire instead of rolling (penny-buybacks avoided)
        'savings_vs_roll_total': round(sum(r.get('savings_vs_roll', 0) for r in roll_actions), 0),
    }


def _whatif_delta_matrix(positions, spot, current_greeks):
    """2D what-if grid: for each (strike, DTE), what does selling 1 put OR
    1 call do to portfolio delta and theta? Surfaces which side (CC vs CSP)
    is more 'delta-efficient' right now — i.e., maximizes premium per
    unit of delta exposure added.

    Output: matrix shape (strikes × dtes), two slices (put, call), with
    delta_change, theta_change, theta_per_delta efficiency per cell.
    Also computes the AGGREGATE tendency: lean PUT or CALL.
    """
    import math
    surf = _load_iv_surface()
    latest_surf = max(surf.keys()) if surf else None
    # Strike grid: -8% to +8% from spot, 7 levels
    otm_levels = [-0.08, -0.05, -0.02, 0.0, 0.02, 0.05, 0.08]
    # DTE grid: 7 / 14 / 30 / 45 / 60
    dtes = [7, 14, 30, 45, 60]
    current_delta = current_greeks.get('total_delta', 0)
    matrix_put = []
    matrix_call = []
    best_put = None
    best_call = None
    for otm in otm_levels:
        # Put strike = spot * (1 - |otm|) when otm <= 0 (below spot)
        # Call strike = spot * (1 + |otm|) when otm >= 0 (above spot)
        K_put = round(spot * (1 + otm), 2) if otm <= 0 else round(spot * (1 - otm), 2)
        K_call = round(spot * (1 + otm), 2) if otm >= 0 else round(spot * (1 - otm), 2)
        row_put = []
        row_call = []
        for dte in dtes:
            iv_p = iv_from_surface(surf, latest_surf, K_put, dte, 'P') if (surf and latest_surf) else 0.50
            iv_c = iv_from_surface(surf, latest_surf, K_call, dte, 'C') if (surf and latest_surf) else 0.50
            if iv_p is None: iv_p = 0.50
            if iv_c is None: iv_c = 0.50
            # Compute Greeks for selling 1 contract (qty = -1)
            d_p, _, t_p, _ = _bs_greeks(spot, K_put, dte/365, 0.045, iv_p, 'P')
            d_c, _, t_c, _ = _bs_greeks(spot, K_call, dte/365, 0.045, iv_c, 'C')
            # Sold 1 put: position delta = d_p * 100 * -1 = -d_p * 100
            # (put delta is negative, so -d_p > 0 → adds positive delta)
            delta_put_chg = -d_p * 100
            theta_put_chg = t_p * 100 * -1  # short × negative = positive income
            delta_call_chg = -d_c * 100
            theta_call_chg = t_c * 100 * -1
            # Efficiency: $ theta per unit |delta change|
            eff_put = abs(theta_put_chg) / max(abs(delta_put_chg), 1)
            eff_call = abs(theta_call_chg) / max(abs(delta_call_chg), 1)
            put_cell = {
                'strike': K_put, 'dte': dte,
                'delta_chg': round(delta_put_chg, 1),
                'theta_chg': round(theta_put_chg, 2),
                'eff': round(eff_put, 3),
                'iv': round(iv_p, 4),
                'otm_pct': round(otm * 100, 1),
            }
            call_cell = {
                'strike': K_call, 'dte': dte,
                'delta_chg': round(delta_call_chg, 1),
                'theta_chg': round(theta_call_chg, 2),
                'eff': round(eff_call, 3),
                'iv': round(iv_c, 4),
                'otm_pct': round(otm * 100, 1),
            }
            row_put.append(put_cell)
            row_call.append(call_cell)
            # Track best (highest theta per unit delta moved)
            if best_put is None or put_cell['eff'] > best_put['eff']:
                best_put = put_cell
            if best_call is None or call_cell['eff'] > best_call['eff']:
                best_call = call_cell
        matrix_put.append(row_put)
        matrix_call.append(row_call)
    # Aggregate tendency
    flat_put = [c for row in matrix_put for c in row]
    flat_call = [c for row in matrix_call for c in row]
    avg_eff_put = sum(c['eff'] for c in flat_put) / len(flat_put)
    avg_eff_call = sum(c['eff'] for c in flat_call) / len(flat_call)
    # Bias: if portfolio delta is too LONG (positive), favor calls (cuts delta)
    # If delta is too short (negative), favor puts (raises delta)
    delta_imbalance = current_delta - 6200  # 6200 = neutral target
    if delta_imbalance > 500:
        tendency = 'LEAN_CALL'
        tendency_reason = f'Portfolio +{delta_imbalance:.0f}Δ above target; selling CCs reduces delta'
    elif delta_imbalance < -500:
        tendency = 'LEAN_PUT'
        tendency_reason = f'Portfolio {delta_imbalance:.0f}Δ below target; selling CSPs raises delta'
    else:
        tendency = 'BALANCED'
        tendency_reason = f'Portfolio Δ within ±500 of target (current {current_delta:.0f}); follow IV richness'
    return {
        'otm_levels': [round(x*100, 1) for x in otm_levels],
        'dtes': dtes,
        'put_matrix': matrix_put,
        'call_matrix': matrix_call,
        'best_put': best_put,
        'best_call': best_call,
        'avg_eff_put': round(avg_eff_put, 3),
        'avg_eff_call': round(avg_eff_call, 3),
        'tendency': tendency,
        'tendency_reason': tendency_reason,
        'current_delta': round(current_delta, 0),
        'delta_imbalance': round(delta_imbalance, 0),
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
    seen_K = set()
    for otm in [0.02, 0.05, 0.08, 0.12, 0.15, 0.20]:
        # REAL STRIKE GRID: UNG trades $0.50 increments (integers on
        # monthlies) — synthetic 2-decimal strikes were unexecutable
        K = round(spot * (1 - otm) * 2) / 2
        if K in seen_K:
            continue
        seen_K.add(K)
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
