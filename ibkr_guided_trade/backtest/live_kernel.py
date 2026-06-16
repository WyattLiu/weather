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
        try:
            qty = int(p.get('qty') or p.get('quantity') or 0)
            K = float(p.get('strike') or 0)
            right = (p.get('option_type') or p.get('right') or '').upper()[:1]
            exp = p.get('expiry') or p.get('expiration')
            dte = max(1, (pd.Timestamp(exp).normalize() - today).days) if exp else 30
            entry = today - pd.Timedelta(days=10)
            rec = {'entry': entry, 'K': K, 'dte': dte, 'qty': abs(qty),
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
    """Rough daily theta ($/day) from outstanding shorts: premium/DTE per contract,
    summed. Positive = income accruing to you."""
    th = 0.0
    for leg in list(short_puts) + list(short_calls):
        prem = leg.get('entry_prem', 0.3)
        dte = max(1, leg.get('dte', 30))
        th += prem * 100 * leg.get('qty', 0) / dte
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
    shares = sum(int(p.get('qty') or 0) for p in (positions or [])
                 if (p.get('option_type') or p.get('right') or '') in ('', 'SHARES', 'STOCK'))
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
        expiry = _opt_expiry(dte) if (right in ('PUT', 'CALL') and dte) else None
        _, why = JUSTIFY[ty]
        # Build a fully-specified order line: qty × strike right, expiry, DTE.
        if right in ('PUT', 'CALL') and K:
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

    # projected theta from resulting shorts (seed shorts kept + new opens)
    new_sp = sp + [{'entry': pd.Timestamp.today(), 'K': o.get('K', spot),
                    'dte': o.get('dte', 30), 'qty': abs(o.get('qty', 0)),
                    'entry_prem': (o.get('credit', 0) / max(1, abs(o.get('qty', 1)) * 100))}
                   for _, o in (orders.iterrows() if len(orders) else [])
                   if o.get('type') == 'OPEN_PUT']
    theta_day = _est_theta(new_sp, sc, spot)

    z = R.compute_historical_z(row)
    return {
        'kernel': key, 'kernel_label': KERNELS.get(key, {}).get('label', key),
        'spot': round(spot, 2), 'asof': str(df.index[-1].date()),
        'recommendations': recs,
        'theta': {'per_day': round(theta_day, 0), 'per_week': round(theta_day * 7, 0),
                  'per_month': round(theta_day * 30, 0)},
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
