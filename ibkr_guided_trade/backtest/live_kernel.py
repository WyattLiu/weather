"""Single-source-of-truth live recommendation.

Runs the ACTUAL champion kernel (replay_engine, live_decision) on the operator's
real positions and returns today's orders the engine decided — each justified —
plus the projected theta stream and the Z signal models. No re-implementation, so
what you see is exactly what the validated backtest does. Zero noise.
"""
import os
import sys
import math
import pandas as pd

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
import replay_engine as R
from validated_kernel_adapter import CHAMPION_KEY, KERNELS  # noqa: E402

# trade-type → (human action template, why-it-fires)
JUSTIFY = {
    'OPEN_PUT':            ('Sell {qty} cash-secured put(s) @ ${K} ({dte}d)',
                            'Premium harvest on UNG real bid (${credit:,.0f}); gap-driven size steers the book toward target via assignment, not direct buys'),
    'CONVICTION_ITM_PUT':  ('Sell {qty} ITM put(s) @ ${K}',
                            'Deep-cheap z + low IV-rank → accumulate at a cushioned basis'),
    'OPEN_PUT_RATIO_FLOOR':('Buy {qty} put(s) @ ${K} as a floor',
                            'Defines downside risk under the aggressive put accumulation'),
    'OPEN_LONG_PUT_FLOOR': ('Buy {qty} put(s) @ ${K} as a floor',
                            'Defined-risk floor under the put accumulation'),
    'OPEN_UPSIDE_WING':    ('Buy {qty} call(s) @ ${K} (wing)',
                            'Recovers capped upside if UNG spikes through the short call'),
    'PUT_TP':              ('Buy back {qty} short put(s) — take profit',
                            'Captured ${pnl:,.0f}; recycle the collateral into a fresh cycle'),
    'PUT_ROLL_DOWN':       ('Roll {qty} ITM put(s) down',
                            'UNG dropped through the strike — roll rather than assign into a downtrend (avoids catching a falling knife)'),
    'OPEN_CC':             ('Sell {qty} covered call(s) @ ${K}',
                            'Income on uncovered shares at/above the GEX wall'),
    'OPEN_ITM_CC':         ('Sell {qty} ITM covered call(s) @ ${K}',
                            'Rich z / hot → monetize and pre-commit a called-away exit'),
    'ITM_CC_DIVEST':       ('Sell {qty} deep-ITM call(s) @ ${K} to divest',
                            'Rich-and-hot → divest into strength at premium'),
    'CALL_TP':             ('Buy back {qty} short call(s) — take profit',
                            'Captured ${pnl:,.0f}'),
    'KOLD_BOOK_HEDGE':     ('Set KOLD inverse-ETF hedge to {qty} sh',
                            'Hedges the uncovered share book (cheap grind-regime insurance)'),
    'KOLD_SHOULDER_ENTRY': ('Add KOLD {qty} sh (shoulder season)',
                            'Mar-May / Sep-Nov structural NG weakness'),
    'Z_TARGET_ADD':        ('Buy {qty} UNG shares', 'Below share target'),
    'Z_TARGET_TRIM':       ('Sell {qty} UNG shares', 'Above share target'),
}


# trade-type → (side verb, instrument) for the explicit order line
_LEG = {
    'OPEN_PUT': ('SELL', 'PUT'), 'CONVICTION_ITM_PUT': ('SELL', 'PUT'),
    'OPEN_CC': ('SELL', 'CALL'), 'OPEN_ITM_CC': ('SELL', 'CALL'),
    'ITM_CC_DIVEST': ('SELL', 'CALL'),
    'PUT_TP': ('BUY-TO-CLOSE', 'PUT'), 'CALL_TP': ('BUY-TO-CLOSE', 'CALL'),
    'OPEN_LONG_PUT_FLOOR': ('BUY', 'PUT'), 'OPEN_PUT_RATIO_FLOOR': ('BUY', 'PUT'),
    'OPEN_UPSIDE_WING': ('BUY', 'CALL'), 'PUT_ROLL_DOWN': ('ROLL DOWN', 'PUT'),
    'CALL_ROLL_UP': ('ROLL UP', 'CALL'), 'KOLD_BOOK_HEDGE': ('SET', 'KOLD'),
    'KOLD_SHOULDER_ENTRY': ('BUY', 'KOLD'),
    'Z_TARGET_ADD': ('BUY', 'UNG'), 'Z_TARGET_TRIM': ('SELL', 'UNG'),
}


def _opt_expiry(dte):
    """Concrete option expiry: today + dte, snapped to the next Friday."""
    d = pd.Timestamp.today().normalize() + pd.Timedelta(days=int(dte or 30))
    while d.weekday() != 4:
        d += pd.Timedelta(days=1)
    return d.date().isoformat()


def _to_engine_positions(positions):
    """Map WS-style positions → engine short_puts/short_calls/long_* lists."""
    sp, sc, lp, lc = [], [], [], []
    today = pd.Timestamp.today().normalize()
    for p in positions or []:
        # UNG ONLY — never pull DBA/other-underlying options into the UNG engine (they
        # contaminate sizing/theta/TP/roll). Default to UNG when symbol absent (demo books).
        if 'UNG' not in str(p.get('symbol') or 'UNG').upper():
            continue
        try:
            qty = int(p.get('qty') or p.get('quantity') or 0)
            K = float(p.get('strike') or 0)
            right = (p.get('option_type') or p.get('right') or '').upper()[:1]
            exp = p.get('expiry') or p.get('expiration')
            dte = max(1, (pd.Timestamp(exp).normalize() - today).days) if exp else 30
            entry = today - pd.Timedelta(days=10)
            rec = {'entry': entry, 'K': K, 'dte': dte, 'qty': abs(qty),
                   'expiry': (str(pd.Timestamp(exp).date()) if exp else None),
                   'entry_prem': float(p.get('average_price') or p.get('avg_price') or 0.3)}
        except Exception:
            continue
        if not right or not K:
            continue
        short = qty < 0
        if right == 'P':
            (sp if short else lp).append(rec)
        elif right == 'C':
            (sc if short else lc).append(rec)
    return sp, sc, lp, lc


def _est_theta(short_puts, short_calls, spot):
    """Daily EXTRINSIC (time-value) decay = real theta. Only the time value decays;
    INTRINSIC (moneyness) does not. The gap-wheel sells ITM puts whose premium is
    mostly intrinsic, so real theta << premium/DTE. Gross of assignment."""
    th = 0.0
    for leg in short_puts:
        K = leg.get('K', spot) or spot
        intrinsic = max(0.0, K - spot)            # short put ITM when K > spot
        extr = max(0.0, leg.get('entry_prem', 0) - intrinsic)
        th += extr * 100 * leg.get('qty', 0) / max(1, leg.get('dte', 30))
    for leg in short_calls:
        K = leg.get('K', spot) or spot
        intrinsic = max(0.0, spot - K)            # short call ITM when K < spot
        extr = max(0.0, leg.get('entry_prem', 0) - intrinsic)
        th += extr * 100 * leg.get('qty', 0) / max(1, leg.get('dte', 30))
    return th


def get_live_recommendation(positions=None, cash=100000.0, spot=None, kernel_key=None):
    key = kernel_key or CHAMPION_KEY
    params = R.STRATEGIES.get(KERNELS.get(key, {}).get('strategy', key))
    if params is None:
        return {'error': f'kernel {key} not in STRATEGIES'}
    df = pd.read_csv(os.path.join(THIS, 'cache', 'master_dataset.csv'),
                     index_col=0, parse_dates=True)
    df = R.precompute_factor_z(df).dropna(subset=['UNG'])
    row = df.iloc[-1]
    spot = float(spot if spot else row['UNG'])

    sp, sc, lp, lc = _to_engine_positions(positions)
    # UNG share count — read BOTH 'qty' and 'quantity' (the live feed normalizes shares as
    # 'quantity'; reading only 'qty' silently zeroed shares → false 'naked' coverage). UNG
    # only (exclude BOXX/KOLD/option legs), so the covered-call cap is computed correctly.
    def _is_ung_shares(p):
        if p.get('is_option'):
            return False
        if (p.get('option_type') or p.get('right') or '') not in ('', 'SHARES', 'STOCK'):
            return False
        return 'UNG' in str(p.get('symbol') or 'UNG').upper()
    shares = sum(int(p.get('qty') or p.get('quantity') or 0)
                 for p in (positions or []) if _is_ung_shares(p))
    seed = {'cash': float(cash), 'shares': int(shares), 'short_puts': sp,
            'short_calls': sc, 'long_puts': lp, 'long_calls': lc}

    _h, orders = R.run_strategy_simple(df, params, seed_state=seed, live_decision=True)

    recs = []
    for _, o in orders.iterrows() if len(orders) else []:
        ty = o.get('type', '')
        # ACTIONABLE-ONLY: drop settlement/consequence events (PUT_ASSIGN,
        # *_EXPIRE_*, *_DEFER_*, *_SKIP_*) — those happen TO the book, not actions.
        if ty not in JUSTIFY:
            continue
        side, right = _LEG.get(ty, ('', ''))
        qty = int(abs(o['qty'])) if ('qty' in o and o['qty'] == o['qty']) else None
        K = round(float(o['K']), 2) if ('K' in o and o['K'] == o['K']) else None
        dte = (int(o['dte']) if ('dte' in o and o['dte'] == o['dte'])
               else (int(params.get('open_dte', 30)) if right in ('PUT', 'CALL') else None))
        credit = float(o['credit']) if ('credit' in o and o['credit'] == o['credit']) else 0.0
        pnl = float(o['pnl']) if ('pnl' in o and o['pnl'] == o['pnl']) else 0.0
        _exp = o.get('expiry')
        expiry = (_exp if (isinstance(_exp, str) and _exp)
                  else (_opt_expiry(dte) if (right in ('PUT', 'CALL') and dte) else None))
        # DTE must match the EXPIRY DATE vs TODAY (the sim clock differs from real today,
        # which made exp/DTE inconsistent, e.g. '2026-07-02 (6d)' on 2026-06-17). Recompute.
        if expiry:
            try:
                dte = max(0, (pd.Timestamp(expiry).normalize()
                              - pd.Timestamp.today().normalize()).days)
            except Exception:
                pass
        _, why = JUSTIFY[ty]
        # Build a fully-specified order line: qty × strike right, expiry, DTE.
        if ty in ('PUT_TP', 'CALL_TP') and K:
            d = f' exp {expiry} ({dte}d)' if (expiry or dte) else ''
            bb = f' @ ~${o["buyback"]:.2f}' if ('buyback' in o and o['buyback'] == o['buyback']) else ''
            action = (f"BUY-TO-CLOSE {qty or ''}× UNG ${K:.2f} {right}{d}{bb} "
                      f"— take profit +${pnl:,.0f}").strip()
        elif right in ('PUT', 'CALL') and K:
            d = f' exp {expiry} ({dte}d)' if expiry else ''
            action = f"{side} {qty or ''}× UNG ${K:.2f} {right}{d}".strip()
        elif ty in ('PUT_TP', 'CALL_TP'):
            action = f"BUY-TO-CLOSE winning {right}(s) — take profit +${pnl:,.0f}"
        elif right == 'KOLD':
            action = f"SET KOLD hedge → {qty} sh"
        elif right == 'UNG':
            action = f"{side} {qty} UNG shares"
        else:
            action = ty
        recs.append({'action': action, 'side': side, 'right': right, 'qty': qty,
                     'strike': K, 'expiry': expiry, 'dte': dte,
                     'credit': round(credit, 0), 'pnl': round(pnl, 0),
                     'why': (why.format(credit=credit, pnl=pnl) if '{' in why else why),
                     'type': ty})

    # ── HARD COVERAGE CAP (covered-calls-only safety, [[feedback_covered_calls_only]]) ──
    # NEVER suggest selling more calls than the share book covers. Existing short calls +
    # any newly-suggested call sales must be ≤ shares//100. Cap/drop offending recs at the
    # output boundary, regardless of engine path. This guarantees the book stays covered.
    _existing_calls = sum(int(abs(c.get('qty') or 0)) for c in sc)   # sc = seeded short calls
    _cap = shares // 100 - _existing_calls
    _capped = []
    for r in recs:
        if (r.get('right') == 'CALL' and (r.get('side') or '').upper().startswith('SELL')
                and r.get('qty')):
            allow = max(0, min(int(r['qty']), _cap))
            _cap -= allow
            if allow <= 0:
                r['_dropped_uncovered'] = True
                continue
            if allow < r['qty']:
                r['qty'] = allow
                r['action'] = r['action'].replace(str(int(r.get('qty', 0))), str(allow), 1)
                r['why'] = (r.get('why', '') + ' [capped to stay covered]')
        _capped.append(r)
    recs = _capped
    coverage = {'shares': int(shares), 'coverable_calls': int(shares // 100),
                'existing_short_calls': int(_existing_calls),
                'covered': _existing_calls <= shares // 100}
    # EXTRINSIC-only theta, BEFORE (your current book) vs AFTER today's orders
    # (the engine's actual post-decision book). Gross of assignment.
    final = getattr(R, '_LIVE_FINAL', {}) or {}
    theta_now = _est_theta(sp, sc, spot)
    theta_after = _est_theta(final.get('short_puts', sp), final.get('short_calls', sc), spot)

    z = R.compute_historical_z(row)
    # ── REGIME STATE (regime_wheel_boxx driver) — what regime we're in TODAY and the
    #    posture it implies, so the operator knows accumulate vs neutral vs distribute.
    _ssz = float(row.get('storage_surprise_z') or 0.0)
    _rs = float(row.get('regime_strength') or 0.0)
    _dd60 = float(row.get('ung_dd_60') or 0.0)
    _state = 'ACCUMULATE' if _ssz < -0.5 else ('DISTRIBUTE' if _ssz > 0.5 else 'NEUTRAL')
    _posture = {'ACCUMULATE': 'lean long — shares ~17% NAV, BOXX ~50%, full puts, far CCs',
                'NEUTRAL': 'balanced — shares ~16%, BOXX ~48%, harvest ITM/OTM premium',
                'DISTRIBUTE': 'defensive — shares ~12%, BOXX ~56%, sell ITM calls, sweep cash'}[_state]
    regime = {'state': _state, 'storage_surprise_z': round(_ssz, 2),
              'regime_strength': round(_rs, 2), 'price_dd_60d': round(_dd60 * 100, 1),
              'posture': _posture}
    return {
        'kernel': key, 'kernel_label': KERNELS.get(key, {}).get('label', key),
        'spot': round(spot, 2),
        # ── TWO-CLOCK BRIDGE (elegant common ground) ──────────────────────────────
        # as_of  = data/sim time → drives SIGNALS & valuation (regime/z/IV/storage).
        # today  = real-world time → drives SCHEDULING & decay (DTE/expiry/exec/theta).
        # Surfaced explicitly; staleness flagged. In a pure backtest as_of==today.
        'asof': str(df.index[-1].date()),
        'today': str(pd.Timestamp.today().normalize().date()),
        'data_stale_days': int(max(0, (pd.Timestamp.today().normalize()
                                       - df.index[-1].normalize()).days)),
        'regime': regime, 'coverage': coverage,
        'recommendations': recs,
        'theta': {'now_per_day': round(theta_now, 0), 'after_per_day': round(theta_after, 0),
                  'after_per_month': round(theta_after * 30, 0),
                  'note': 'extrinsic (time-value) decay only — intrinsic excluded; gross of assignment',
                  'per_day': round(theta_after, 0), 'per_week': round(theta_after * 7, 0),
                  'per_month': round(theta_after * 30, 0)},
        'z_models': {
            'z_valuation': round(z, 2),
            'surge_z_momentum': round(float(row.get('ung_surge_z') or 0), 2),
            'iv_rank': (round(float(row['iv_rank']), 2)
                        if 'iv_rank' in row and row['iv_rank'] == row['iv_rank'] else None),
            'regime': ('CHEAP' if z < -1 else 'RICH' if z > 1 else 'NEUTRAL'),
            'hh_basis': round(float(row.get('hh_basis') or 0), 3),
        },
    }


if __name__ == '__main__':
    import json
    demo = [{'option_type': 'P', 'strike': 11.0, 'qty': -7, 'expiry': '2026-06-26', 'average_price': 0.3},
            {'option_type': 'C', 'strike': 12.0, 'qty': -14, 'expiry': '2026-07-17', 'average_price': 0.4},
            {'right': 'SHARES', 'qty': 3400}]
    print(json.dumps(get_live_recommendation(demo, cash=120000), indent=2, default=str))
