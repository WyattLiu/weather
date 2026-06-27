"""Kernel Dashboard — standalone webdash for the validated backtest kernel.

Runs on port 10001 by default (9998 = research dash, 9999 = production).

Visually matches production dashboard (same CSS variables, layout, table
styles). Logic backed entirely by champion_target_25_dd_trim, the
walk-forward-validated kernel.

Start:
    cd /home/wyatt/weather/ibkr_guided_trade
    venv/bin/python kernel_dashboard.py

URLs:
    http://localhost:10001 ← kernel (default; override with KERNEL_DASH_PORT)
    http://localhost:9999  ← production (untouched, reference)
    http://localhost:9998  ← research dashboard
"""
from __future__ import annotations

import http.server
import json
import os
import socketserver
import sys
import threading
import time
import urllib.parse

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS_DIR, 'backtest'))
sys.path.insert(0, '/home/wyatt/ibkr_guided_trade')

from validated_kernel_adapter import validated_verdict, CHAMPION_NAME  # type: ignore

try:
    from spy_vega_alert import spy_vega_signal  # SPY vega-scrape VIX<=16 setup alert
except Exception:
    def spy_vega_signal(force=False):  # type: ignore
        return {'error': 'spy_vega_alert unavailable'}

# Live single-source recommendation cache (the engine run is ~30-40s, so cache it
# and warm it in the background — the panel then loads instantly).
_LIVE_CACHE = {'ts': 0.0, 'data': None}
_LIVE_TTL = 180.0


def _real_usd_cash_bp():
    """EXACT settled USD cash + buying power from the broker's FetchTradingBalanceBuyingPower query.
    NOT net_liq − Σ MV — that is WRONG for a MULTI-CURRENCY margin account: it blends the CAD side and
    ignores a USD margin LOAN, so the BOXX sweep oversized the buy into ~$9.5k of USD margin. Cash can
    be NEGATIVE here (= margin loan); the engine must see that so it never buys more into margin."""
    try:
        from ws_sdk import WSClient, get_session, graphql_query
        from ws_sdk.queries import QUERY_TRADING_BALANCE
        _c = WSClient(); _s = get_session()
        for _a in _c.list_accounts():
            if 'non-registered' in _a.id and 'MARGIN' in str(_a.type).upper():
                _d = graphql_query(_s, 'FetchTradingBalanceBuyingPower', QUERY_TRADING_BALANCE,
                                   {'accountCanonicalId': _a.id, 'currency': 'USD'})
                _v = ((_d or {}).get('account') or {}).get('financials', {}).get('current', {}).get('tradingBalanceView') or {}
                _cash = (_v.get('cash') or {}).get('quantity')
                _bp = (_v.get('buyingPower') or {}).get('quantity')
                if _cash is not None:
                    return float(_cash), (float(_bp) if _bp is not None else None)
    except Exception:
        pass
    return None, None


def _compute_live():
    try:
        from live_kernel import get_live_recommendation
        with _STATE_LOCK:
            spot = _STATE_CACHE.get('spot')
            pos = _STATE_CACHE.get('positions', [])
            bal = _STATE_CACHE.get('balance') or {}
            # REAL USD CASH from the broker's exact query (can be NEGATIVE = margin loan). The old
            # `cash = net_liq − Σ MV` was WRONG for this multi-currency margin account — it blended the
            # CAD side and hid the USD margin, so the BOXX sweep over-bought ~$9.5k into USD margin.
            # Fall back to the derivation ONLY if the query fails.
            _cash, _bp = _real_usd_cash_bp()
            if _cash is None:
                _cash = bal.get('cash') or bal.get('total_cash')
                if not _cash:
                    _nl = bal.get('net_liquidation')
                    if _nl:
                        _pos_mv = sum(float(p.get('market_value') or 0) for p in pos)
                        _cash = float(_nl) - _pos_mv
            cash = _cash if _cash is not None else 100000
        data = get_live_recommendation(pos, cash=cash, spot=spot,
                                       kernel_key='regime_wheel_boxx_greeks_live')  # PROMOTED champion (greeks-managed)
        # EXECUTION ADVISOR: annotate each order with a manual-execution plan
        # (which minute to work it + limit ladder mid→touch). Operator runs these by hand.
        try:
            from execution_advisor import plan_for_recs
            if isinstance(data, dict) and data.get('recommendations'):
                data['recommendations'] = plan_for_recs(data['recommendations'],
                                                        spot=data.get('spot'))
        except Exception as _e:
            data['exec_advisor_error'] = repr(_e)[:160]
        _LIVE_CACHE['data'] = data
        _LIVE_CACHE['ts'] = time.time()
    except Exception as e:
        _LIVE_CACHE['data'] = {'error': repr(e)}
        _LIVE_CACHE['ts'] = time.time()
    return _LIVE_CACHE['data']


def _live_warm_loop():
    while True:
        _compute_live()
        time.sleep(_LIVE_TTL)

try:
    from ws_sdk.client import WSClient
    WS_AVAILABLE = True
except Exception:
    WSClient = None
    WS_AVAILABLE = False


# ─── State cache ─────────────────────────────────────────────────────────────
_STATE_LOCK = threading.Lock()
_STATE_CACHE = {
    'last_refresh': 0.0, 'spot': None, 'positions': [],
    'balance': None, 'verdict': None, 'error': None,
}
REFRESH_SEC = 30


def _normalize(p) -> dict:
    d = {
        'symbol': p.symbol, 'is_option': p.is_option,
        'quantity': int(p.quantity) if p.quantity is not None else 0,
        'market_value': float(p.market_value) if p.market_value is not None else 0.0,
        'unrealized_pnl': float(p.unrealized_pnl) if p.unrealized_pnl is not None else 0.0,
    }
    if p.is_option:
        d['option_type'] = p.option_type
        d['strike'] = float(p.strike) if p.strike is not None else None
        d['expiry'] = p.expiry
    else:
        d['average_price'] = float(p.average_price) if p.average_price is not None else 0.0
    return d


def _refresh():
    with _STATE_LOCK:
        try:
            if not WS_AVAILABLE:
                _STATE_CACHE['error'] = 'WSClient unavailable — kernel-only mode'
                spot = _STATE_CACHE.get('spot') or 11.51
            else:
                c = WSClient()
                positions = c.list_positions()
                ung_share = next((p for p in positions if p.symbol == 'UNG' and not p.is_option), None)
                spot = None
                if ung_share:
                    raw_q = ung_share.raw.get('security', {}).get('quoteV2', {})
                    bid = float(raw_q.get('bid') or 0)
                    ask = float(raw_q.get('ask') or 0)
                    last = float(raw_q.get('last') or 0)
                    quote_spot = (bid + ask) / 2 if (bid > 0 and ask > 0) else last
                    if not quote_spot or quote_spot <= 0:
                        quote_spot = float(raw_q.get('price') or 0)
                    # BROKER MARK = market_value / qty — the authoritative, internally-consistent
                    # price. The quoteV2 bid/ask midpoint is unreliable when the market is CLOSED
                    # (holiday/after-hours): a stale or wide book gave a bad spot ($12.34 vs an
                    # $11.73 mark on a 2026-06-19 holiday). Trust the quote only if it's within 3%
                    # of the mark; otherwise use the mark. Keeps spot consistent with positions.
                    mark_spot = None
                    try:
                        q = float(ung_share.quantity or 0)
                        mv = float(ung_share.market_value or 0)
                        if q and mv:
                            mark_spot = abs(mv / q)
                    except Exception:
                        pass
                    if mark_spot and mark_spot > 0:
                        spot = (quote_spot if (quote_spot and quote_spot > 0
                                               and abs(quote_spot / mark_spot - 1) <= 0.03)
                                else mark_spot)
                    else:
                        spot = quote_spot
                if spot is None or spot <= 0:
                    spot = 11.51
                _STATE_CACHE['spot'] = spot
                _STATE_CACHE['positions'] = [_normalize(p) for p in positions]
                bal = c.get_balance('USD')
                _STATE_CACHE['balance'] = {
                    'net_liquidation': float(bal.net_liquidation),
                    'total_return': float(bal.total_return),
                    'total_return_pct': float(bal.total_return_pct),
                }
                _STATE_CACHE['error'] = None
            nav_live = (_STATE_CACHE.get('balance') or {}).get('net_liquidation')
            _STATE_CACHE['verdict'] = validated_verdict(spot, _STATE_CACHE['positions'], nav=nav_live)
            _STATE_CACHE['last_refresh'] = time.time()
        except Exception as e:
            _STATE_CACHE['error'] = f'refresh failed: {e}'


def _refresh_loop():
    while True:
        _refresh()
        time.sleep(REFRESH_SEC)


# ─── Series + analytics endpoints ────────────────────────────────────────────
_SERIES_CACHE = {'ts': 0.0, 'data': None}


def _build_series():
    """Build time-series JSON for charts (UNG, z, IV30, regime bands)."""
    import pandas as pd
    from replay_engine import precompute_factor_z, compute_historical_z  # type: ignore

    csv = os.path.join(THIS_DIR, 'backtest', 'cache', 'master_dataset.csv')
    df = pd.read_csv(csv, index_col=0, parse_dates=True)
    df = precompute_factor_z(df).dropna(subset=['UNG'])
    # Compute z timeseries
    z = df.apply(lambda r: compute_historical_z(r, use_surprise=True), axis=1)
    out = {
        'dates': [d.strftime('%Y-%m-%d') for d in df.index],
        'ung': [float(x) for x in df['UNG'].tolist()],
        'iv30': [float(x) if pd.notna(x) else None for x in df.get('iv_30d', pd.Series([None]*len(df))).tolist()],
        'z': [float(x) if pd.notna(x) else None for x in z.tolist()],
        'rv30': [float(x) if pd.notna(x) else None for x in df.get('rv_30', pd.Series([None]*len(df))).tolist()],
    }
    return out


def _live_champion_strat(STRATEGIES):
    """Resolve the CURRENT live champion strategy, resilient to renames/stale
    filtering. Old code hardcoded 'champion_target_25_dd_trim', which the
    _KEEP_STRATEGIES/lifecycle filter now drops → KeyError → blank charts.
    Prefer the adapter's promoted CHAMPION_KEY, then known fallbacks."""
    candidates = []
    try:
        from validated_kernel_adapter import CHAMPION_KEY  # type: ignore
        candidates.append(f'champion_{CHAMPION_KEY}')
    except Exception:
        pass
    candidates += ['champion_kold15_ivrank_kbh', 'champion_kold15_ivrank',
                   'champion_target_25_smooth', 'champion_target_25_dd_trim']
    for k in candidates:
        if k in STRATEGIES:
            return STRATEGIES[k], k
    for k in STRATEGIES:  # last resort: any champion
        if k.startswith('champion_'):
            return STRATEGIES[k], k
    raise KeyError('no champion strategy available in STRATEGIES')


def _build_walkforward():
    """Run the live champion on rolling 12-month windows."""
    import pandas as pd
    import math
    from replay_engine import run_strategy_simple, STRATEGIES, precompute_factor_z  # type: ignore
    csv = os.path.join(THIS_DIR, 'backtest', 'cache', 'master_dataset.csv')
    df = pd.read_csv(csv, index_col=0, parse_dates=True)
    df = precompute_factor_z(df).dropna(subset=['UNG'])
    strat, _ = _live_champion_strat(STRATEGIES)
    windows = []
    start_dates = pd.date_range('2021-07-01', '2025-04-01', freq='3MS')
    for start in start_dates:
        end = start + pd.DateOffset(years=1)
        if end > df.index[-1]: continue
        sub = df.loc[start:end]
        if len(sub) < 200: continue
        try:
            hist, _ = run_strategy_simple(sub, strat, 100000, 0)
            hist = hist.set_index(pd.to_datetime(hist['date']))
            init = 100000
            fret = (float(hist.iloc[-1]['nav'])/init - 1)*100
            y = (sub.index[-1]-sub.index[0]).days/365.25
            ann = (1+fret/100)**(1/y)*100 - 100
            rets = hist['nav'].pct_change().dropna()
            sh = rets.mean()/(rets.std()+1e-9)*math.sqrt(252)
            mdd = ((hist['nav']-hist['nav'].cummax())/hist['nav'].cummax()*100).min()
            windows.append({
                'start': start.strftime('%Y-%m-%d'),
                'end': end.strftime('%Y-%m-%d'),
                'ret': round(fret, 1), 'ann': round(ann, 1),
                'sharpe': round(sh, 2), 'mdd': round(mdd, 1),
            })
        except Exception:
            pass
    return windows


def _build_backtest_curve():
    """Full backtest equity curve for champion strategy at $100K cash start."""
    import pandas as pd
    from replay_engine import run_strategy_simple, STRATEGIES, precompute_factor_z  # type: ignore
    csv = os.path.join(THIS_DIR, 'backtest', 'cache', 'master_dataset.csv')
    df = pd.read_csv(csv, index_col=0, parse_dates=True)
    df = precompute_factor_z(df).dropna(subset=['UNG'])
    strat, _ = _live_champion_strat(STRATEGIES)
    hist, _ = run_strategy_simple(df, strat, 100000, 0)
    hist = hist.set_index(pd.to_datetime(hist['date']))
    nav = hist['nav'].tolist()
    peak = hist['nav'].cummax()
    dd = ((hist['nav']-peak)/peak*100).tolist()
    return {
        'dates': [d.strftime('%Y-%m-%d') for d in hist.index],
        'nav': [float(x) for x in nav],
        'drawdown_pct': [float(x) for x in dd],
        'shares': [int(x) for x in hist['shares'].tolist()],
        'cash': [float(x) for x in hist['cash'].tolist()],
    }


def _build_yearly_pnl():
    """Year-by-year P&L breakdown for the champion strategy."""
    import pandas as pd
    import math
    from replay_engine import run_strategy_simple, STRATEGIES, precompute_factor_z  # type: ignore
    csv = os.path.join(THIS_DIR, 'backtest', 'cache', 'master_dataset.csv')
    df = pd.read_csv(csv, index_col=0, parse_dates=True)
    df = precompute_factor_z(df).dropna(subset=['UNG'])
    strat, _ = _live_champion_strat(STRATEGIES)
    hist, _ = run_strategy_simple(df, strat, 100000, 0)
    hist = hist.set_index(pd.to_datetime(hist['date']))
    years = []
    for yr in sorted(hist.index.year.unique()):
        ydf = hist[hist.index.year == yr]
        if len(ydf) < 2: continue
        ystart = ydf['nav'].iloc[0]; yend = ydf['nav'].iloc[-1]
        yret = (yend/ystart - 1) * 100
        yrets = ydf['nav'].pct_change().dropna()
        ysh = yrets.mean()/(yrets.std()+1e-9)*math.sqrt(252) if len(yrets) else 0
        ymdd = ((ydf['nav'] - ydf['nav'].cummax())/ydf['nav'].cummax()*100).min()
        years.append({
            'year': int(yr),
            'pnl_pct': round(yret, 1),
            'pnl_dollar': round(yend - ystart, 0),
            'sharpe': round(ysh, 2),
            'mdd_pct': round(ymdd, 1),
            'days': len(ydf),
        })
    return years


_ANALYTICS_LOCK = threading.Lock()


def _compute_analytics():
    """Heavy compute (~76s of backtests). LOCKED so only one runs at a time — without this, concurrent
    cold-cache requests each kicked off a 76s backtest, piled up, contended, and NONE returned in time
    → empty /api/analytics → every lower chart rendered with no data / broken axes. Called only by the
    warm loop, never on the request path."""
    with _ANALYTICS_LOCK:
        if time.time() - _SERIES_CACHE['ts'] < 600 and _SERIES_CACHE['data']:
            return _SERIES_CACHE['data']
        data = {
            'series': _build_series(),
            'walkforward': _build_walkforward(),
            'backtest_curve': _build_backtest_curve(),
            'yearly': _build_yearly_pnl(),
        }
        _SERIES_CACHE['ts'] = time.time()
        _SERIES_CACHE['data'] = data
        return data


def _cached_analytics():
    """NON-BLOCKING: serve whatever the warm loop has cached (or {} until the first warm completes).
    NEVER computes synchronously — a 76s backtest on the request path hangs the endpoint and, with
    concurrent loads, breaks all the charts. The warm loop keeps this fresh off the request path."""
    return _SERIES_CACHE['data'] or {}


# ─── HTML (matches production CSS variables and layout) ──────────────────────
HTML = r"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=5">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<title>UNG Kernel</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
:root {
    --bg: #0d1117;
    --panel: #161b22;
    --border: #30363d;
    --text: #e6edf3;
    --text-dim: #8b949e;
    --green: #3fb950;
    --red: #f85149;
    --blue: #58a6ff;
    --purple: #bc8cff;
    --orange: #d29922;
    --cyan: #39d2c0;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.5;
}
.container { max-width: 1600px; margin: 0 auto; padding: 16px; }
h1 { font-size: 1.5rem; margin-bottom: 4px; }
.sub { font-size: 0.85rem; color: var(--text-dim); margin-bottom: 16px; }
.sub a { color: var(--blue); text-decoration: none; }
.sub a:hover { text-decoration: underline; }
h2 { font-size: 1.1rem; margin-bottom: 12px; color: var(--text-dim); font-weight: 500; }

.summary-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px;
    margin-bottom: 20px;
}
.card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
}
.card-label { font-size: 0.75rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.05em; }
.card-value { font-size: 1.5rem; font-weight: 600; margin-top: 4px; font-variant-numeric: tabular-nums; }
.card-value.positive { color: var(--green); }
.card-value.negative { color: var(--red); }
.card-value.neutral { color: var(--blue); }
.card-value.warn { color: var(--orange); }
.card-sub { font-size: 0.75rem; color: var(--text-dim); margin-top: 4px; }

.section {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 20px;
}

.rec-list { display: flex; flex-direction: column; gap: 8px; }
.rec {
    background: var(--bg);
    border-left: 3px solid var(--blue);
    padding: 12px;
    border-radius: 4px;
}
.rec-h { border-left-color: var(--red); }
.rec-m { border-left-color: var(--orange); }
.rec-action { font-weight: 600; color: var(--text); margin-bottom: 4px; }
.rec-why { color: var(--text-dim); font-size: 0.85rem; }
.priority {
    display: inline-block;
    padding: 1px 8px;
    border-radius: 12px;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    margin-left: 8px;
    vertical-align: middle;
}
.priority-h { background: var(--red); color: white; }
.priority-m { background: var(--orange); color: black; }
.priority-l { background: var(--border); color: var(--text-dim); }

.warning {
    background: rgba(210,153,34,0.1);
    border-left: 3px solid var(--orange);
    padding: 10px 12px;
    border-radius: 4px;
    color: var(--orange);
    font-size: 0.85rem;
    margin-top: 8px;
}
.error {
    background: rgba(248,81,73,0.1);
    border-left: 3px solid var(--red);
    padding: 10px 12px;
    border-radius: 4px;
    color: var(--red);
    font-size: 0.85rem;
    margin-bottom: 16px;
}

table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
th {
    text-align: left;
    padding: 8px 12px;
    border-bottom: 2px solid var(--border);
    color: var(--text-dim);
    font-weight: 600;
    white-space: nowrap;
}
td {
    padding: 6px 12px;
    border-bottom: 1px solid var(--border);
    white-space: nowrap;
    font-variant-numeric: tabular-nums;
}
tr:hover { background: rgba(88,166,255,0.05); }
td.positive { color: var(--green); }
td.negative { color: var(--red); }
td.neutral { color: var(--blue); }
.mono { font-family: ui-monospace, SFMono-Regular, monospace; }
.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.tag {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    background: var(--border);
    color: var(--text);
    font-size: 0.75rem;
    font-weight: 500;
}
.tag-call { background: rgba(63,185,80,0.2); color: var(--green); }
.tag-put { background: rgba(248,81,73,0.2); color: var(--red); }
.tag-share { background: rgba(88,166,255,0.2); color: var(--blue); }
.tag-neutral { background: var(--border); color: var(--text); }
.tag-rich { background: rgba(248,81,73,0.2); color: var(--red); }
.tag-cheap { background: rgba(63,185,80,0.2); color: var(--green); }

/* ─── PRODUCTION PARITY: daily-banner, expiry-cards, rec-card ─────────── */
.daily-banner {
    display: flex; align-items: center; gap: 16px;
    padding: 18px 24px; border-radius: 8px; margin-bottom: 20px;
    border: 1px solid var(--border);
}
.daily-banner.status-green {
    background: linear-gradient(135deg, rgba(63,185,80,0.12), rgba(63,185,80,0.04));
    border-color: rgba(63,185,80,0.4);
}
.daily-banner.status-yellow {
    background: linear-gradient(135deg, rgba(210,153,34,0.15), rgba(210,153,34,0.04));
    border-color: rgba(210,153,34,0.5);
}
.daily-banner.status-red {
    background: linear-gradient(135deg, rgba(248,81,73,0.15), rgba(248,81,73,0.04));
    border-color: rgba(248,81,73,0.5);
}
.status-icon { font-size: 2rem; flex-shrink: 0; }
.status-headline { font-size: 1.15rem; font-weight: 700; letter-spacing: 0.02em; }
.status-green .status-headline { color: var(--green); }
.status-yellow .status-headline { color: var(--orange); }
.status-red .status-headline { color: var(--red); }
.status-detail { font-size: 0.88rem; color: var(--text-dim); margin-top: 4px; }

.expiry-cards {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(380px, 1fr));
    gap: 12px;
    margin-bottom: 20px;
}
.expiry-card {
    background: rgba(13, 17, 23, 0.6);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px 18px;
    border-left: 4px solid var(--green);
}
.expiry-card.critical { border-left-color: var(--red); }
.expiry-card.warning { border-left-color: var(--orange); }
.expiry-card.caution { border-left-color: #e3b341; }
.expiry-card.ok { border-left-color: var(--green); }
.expiry-card h3 {
    font-size: 1.05rem; margin-bottom: 10px;
    display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
}
.expiry-card h3 .e-badge {
    font-size: 0.72rem; padding: 3px 10px; border-radius: 12px;
    font-weight: 600; text-transform: uppercase;
}
.e-badge-critical { background: var(--red); color: #fff; }
.e-badge-warning { background: var(--orange); color: #000; }
.e-badge-caution { background: #e3b341; color: #000; }
.e-badge-ok { background: rgba(63,185,80,0.2); color: var(--green); }
.detail-row {
    font-size: 0.9rem; color: var(--text-dim);
    padding: 4px 0; line-height: 1.7; word-wrap: break-word;
}
.detail-row strong { color: var(--text); }
.rec-item {
    display: block; padding: 3px 0 3px 12px;
    border-left: 2px solid var(--border); margin: 4px 0;
    font-size: 0.88rem; line-height: 1.6;
}
.rec-item.rec-expire { border-left-color: var(--green); }
.rec-item.rec-assign { border-left-color: var(--orange); }
.rec-item.rec-roll { border-left-color: var(--cyan); }
.rec-item.rec-monitor { border-left-color: var(--text-dim); }

.rec-card {
    padding: 12px 14px; margin-bottom: 10px;
    background: rgba(13, 17, 23, 0.6);
    border-radius: 6px; border: 1px solid var(--border);
}
.rec-header {
    display: flex; align-items: center; gap: 8px;
    margin-bottom: 6px; flex-wrap: wrap;
}
.rec-rank { font-weight: 700; color: var(--text); font-size: 0.9rem; }
.rec-type-badge {
    font-size: 0.7rem; padding: 2px 8px; border-radius: 10px; font-weight: 600;
    background: rgba(88,166,255,0.15); color: var(--blue);
}
.rec-urgency-badge {
    font-size: 0.65rem; padding: 2px 6px; border-radius: 10px;
    font-weight: 600; text-transform: uppercase;
}
.rec-urgency-badge.high { background: var(--red); color: #fff; }
.rec-urgency-badge.medium { background: var(--orange); color: #000; }
.rec-urgency-badge.low { background: rgba(139,148,158,0.2); color: #8b949e; }
.rec-theta {
    margin-left: auto; color: var(--green); font-weight: 600; font-size: 0.85rem;
}

/* MOBILE (matches production breakpoint @ 768px) */
@media (max-width: 768px) {
    .expiry-cards {
        grid-template-columns: 1fr !important;
        gap: 8px;
    }
    .expiry-card { padding: 12px 14px; }
    .expiry-card h3 { font-size: 0.95rem; margin-bottom: 8px; }
    .expiry-card .detail-row { font-size: 0.82rem; }
    .expiry-card .rec-item { font-size: 0.8rem; }
    .rec-card { padding: 10px 12px; }
    .rec-action { font-size: 0.88rem; }
    .rec-header { font-size: 0.78rem; }
    .daily-banner { padding: 12px 14px; gap: 10px; }
    .status-icon { font-size: 1.5rem; }
    .status-headline { font-size: 1rem; }
    .status-detail { font-size: 0.8rem; }
    .container { padding: 8px; }
    h1 { font-size: 1.2rem; margin-bottom: 4px; }
    .sub { font-size: 0.75rem; margin-bottom: 12px; }
    h2 { font-size: 0.95rem; margin-bottom: 8px; }

    .summary-row {
        grid-template-columns: repeat(2, 1fr);
        gap: 8px;
    }
    .card { padding: 10px; }
    .card-label { font-size: 0.65rem; }
    .card-value { font-size: 1.1rem; }
    .card-sub { font-size: 0.65rem; }

    .section { padding: 12px; margin-bottom: 12px; }

    .grid-2 { grid-template-columns: 1fr; gap: 12px; }

    .rec { padding: 10px; }
    .rec-action { font-size: 0.88rem; }
    .rec-why { font-size: 0.75rem; }
    .priority { font-size: 0.6rem; padding: 1px 6px; }

    /* Phone: tables scroll horizontally with momentum */
    .section table { font-size: 0.75rem; }
    .section table th, .section table td {
        padding: 4px 6px;
    }
    .scrollable, .section > table, #beam-content, #iv-shape {
        overflow-x: auto;
        -webkit-overflow-scrolling: touch;
        display: block;
    }
    #beam-content table, #iv-shape table {
        min-width: 100%;
        display: table;
    }
    /* Per-kernel beam: keep tables scrollable inside their containers */
    #beam-by-kernel > div {
        overflow-x: auto;
        -webkit-overflow-scrolling: touch;
    }
    #beam-by-kernel table { font-size: 0.7rem; min-width: 460px; }
    /* Actionable orders: keep OSI symbol from overflowing */
    #actionable-orders .rec-card { word-break: break-all; }
    #actionable-orders .mono { font-size: 0.78rem; }
    /* Kernel selector readable */
    #kernel-selector { font-size: 0.78rem; }
    #kernel-label { font-size: 1.1rem; }

    /* Charts: shorter on phone to fit more above the fold */
    #chart-regime { height: 280px !important; }
    #chart-iv { height: 220px !important; }
    #chart-equity { height: 300px !important; }
    #chart-walkforward { height: 280px !important; }
    #chart-yearly { height: 240px !important; }

    /* Order ladder readable */
    .rec table { font-size: 0.78rem; }
    .badge { font-size: 0.65rem !important; }
    .warning { font-size: 0.78rem; padding: 8px 10px; }
}

/* Very small phones */
@media (max-width: 380px) {
    .summary-row { grid-template-columns: 1fr; }
}
</style>
</head><body>
<div class="container">
  <h1>UNG Kernel Dashboard <span class="tag" id="kernel-tag"></span></h1>
  <section class="card" id="spy-vega-alert" style="margin:14px 0; padding:13px 16px; border:2px solid #555">
    <h2 style="margin:0 0 6px">🌊 SPY vega-scrape setup <span id="sv-verdict" class="tag"></span>
       <span class="sub" style="font-weight:400">long ~45D ATM straddle when VIX≤16 &amp; IV≥RV (crash-aware overlay)</span></h2>
    <div id="sv-body" class="rec-why">loading…</div>
  </section>
  <section class="card" id="sot-panel" style="border:2px solid #39d2c0; margin:14px 0; padding:16px;">
    <h2 style="color:#39d2c0; margin-bottom:8px">⚡ TODAY — Single Source of Truth <span class="tag" id="sot-kernel"></span></h2>
    <div class="sub" style="margin-bottom:10px">Orders below come straight from the live champion engine run on your real positions — same code as the backtest, every order justified. No re-implementation, no noise.</div>
    <div id="sot-z" class="summary-row" style="margin-bottom:8px"></div>
    <div id="sot-theta" class="summary-row" style="margin-bottom:8px"></div>
    <h3 style="margin:14px 0 6px">📐 Book greeks — current → post-order <span class="sub" style="font-weight:400">(what-if from fitted vol surface, never a fill · incl 3rd-order speed/color)</span></h3>
    <div id="sot-greeks"></div>
    <div id="sot-compass"></div>
    <div id="sot-concentration"></div>
    <div id="sot-settlement"></div>
    <h3 style="margin:14px 0 6px">Orders for today</h3>
    <div id="sot-reaccum"></div>
    <div id="sot-orders"></div>
  </section>
  <div class="sub">
    Validated by walk-forward backtest · <a href="http://localhost:9999">Production dashboard</a> ·
    Refresh: <span id="freshness">–</span> ·
    Kernel: <span id="kernel-fullname">–</span>
  </div>

  <div id="error-row"></div>

  <!-- ACTIVE KERNEL: prominently displayed with OOS metrics + selector -->
  <div class="section" style="border:2px solid var(--blue);background:linear-gradient(135deg,rgba(88,166,255,0.08),rgba(88,166,255,0.02))">
    <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:8px">
      <div style="font-size:0.7rem;color:var(--text-dim);text-transform:uppercase;letter-spacing:0.05em">Active Kernel</div>
      <select id="kernel-selector" style="flex:1;min-width:240px;background:var(--bg);color:var(--text);border:1px solid var(--border);padding:6px 8px;border-radius:4px;font-size:0.9rem">
      </select>
    </div>
    <div id="kernel-label" style="font-size:1.3rem;font-weight:700;color:var(--blue);margin-bottom:4px">Loading...</div>
    <div id="kernel-why" style="font-size:0.85rem;color:var(--text-dim);margin-bottom:8px;line-height:1.5"></div>
    <div class="summary-row" style="margin-bottom:0;grid-template-columns:repeat(3,1fr)">
      <div class="card">
        <div class="card-label">Out-of-Sample Ann %</div>
        <div class="card-value positive" id="oos-ann">–</div>
        <div class="card-sub" id="is-ann">(in-sample: –)</div>
      </div>
      <div class="card">
        <div class="card-label">Out-of-Sample Sharpe</div>
        <div class="card-value positive" id="oos-sharpe">–</div>
        <div class="card-sub" id="is-sharpe">(in-sample: –)</div>
      </div>
      <div class="card">
        <div class="card-label">Out-of-Sample MDD</div>
        <div class="card-value negative" id="oos-mdd">–</div>
        <div class="card-sub" id="is-mdd">(in-sample: –)</div>
      </div>
    </div>
  </div>

  <!-- EXECUTOR BRIEF — SUPERSEDED by SOT panel (Z models live there now) -->
  <div class="section" style="display:none">
    <h2>🎯 Executor Brief</h2>
    <div id="executor-brief">Loading...</div>
  </div>

  <!-- DIRECTLY USABLE ORDERS — SUPERSEDED by SOT panel (adapter recs caused the
       phantom 'buy shares'/fake strikes; the SOT panel is the single source now) -->
  <div class="section" style="display:none">
    <h2>📋 Directly Usable Orders (from active kernel)</h2>
    <div id="actionable-orders">Loading kernel orders...</div>
    <div class="rec-why" style="margin-top:8px">
      Each order shows: side, OSI symbol, qty, limit ladder. Sized to current NAV.
      Risk-aware: throttles puts to keep collateral &lt; 80% of NAV.
    </div>
  </div>

  <!-- SOPHISTICATED BEAM — SUPERSEDED (hidden) -->
  <div class="section" style="display:none">
    <h2>🎯 Sophisticated Beam Search — what each kernel would do</h2>
    <div id="beam-by-kernel">Loading kernel beams...</div>
    <div class="rec-why" style="margin-top:8px">
      For each kernel option, scores its candidate strikes using its OWN selection logic
      (OTM%, DTE, IV preference). Winner highlighted. Lets you compare alternative
      kernels side-by-side before switching.
    </div>
  </div>

  <!-- Daily Status Banner (production verbatim) -->
  <div class="daily-banner status-green" id="daily-status-banner">
    <div class="status-icon" id="status-icon">✅</div>
    <div style="flex:1">
      <div class="status-headline" id="status-headline">Loading...</div>
      <div class="status-detail" id="status-detail">Fetching live state...</div>
    </div>
  </div>

  <!-- Top summary cards -->
  <div class="summary-row">
    <div class="card">
      <div class="card-label">UNG Spot</div>
      <div class="card-value neutral" id="spot-val">–</div>
      <div class="card-sub" id="spot-date">–</div>
    </div>
    <div class="card">
      <div class="card-label">Regime</div>
      <div class="card-value" id="regime-val">–</div>
      <div class="card-sub" id="regime-z">z = –</div>
    </div>
    <div class="card">
      <div class="card-label">Target Shares</div>
      <div class="card-value" id="target-val">–</div>
      <div class="card-sub" id="target-mult">×– mult</div>
    </div>
    <div class="card">
      <div class="card-label">Share Δ</div>
      <div class="card-value" id="delta-val">–</div>
      <div class="card-sub">vs current</div>
    </div>
    <div class="card">
      <div class="card-label">Net Liquidation</div>
      <div class="card-value neutral" id="nav-val">–</div>
      <div class="card-sub" id="nav-pct">–</div>
    </div>
    <div class="card">
      <div class="card-label">Put Collat / NAV</div>
      <div class="card-value" id="collat-val">–</div>
      <div class="card-sub" id="collat-warn">healthy</div>
    </div>
  </div>

  <!-- Warnings only (recommendations unified into Directly Usable Orders section) -->
  <div id="warnings" style="margin-bottom:12px"></div>

  <!-- Expiration timeline: production-style per-expiry cards -->
  <div class="section">
    <h2>Expiration Timeline &amp; Roll Planner</h2>
    <div class="expiry-cards" id="expiry-cards-grid">–</div>
  </div>

  <!-- Deep beam analysis -->
  <div class="section" style="display:none">
    <h2>🎯 Deep Beam Analysis — why this strike?</h2>
    <div id="beam-content">–</div>
    <div class="rec-why" style="margin-top:8px">
      Each candidate strike scored as <strong>income − P(ITM) × expected_loss</strong> under BSM measure with real PG IV.
      The winner is the best risk-adjusted premium per contract.
    </div>
  </div>

  <!-- Extrinsic + theta smoothness (production quality metrics) -->
  <div class="section">
    <h2>Theta Smoothness &amp; Extrinsic Value</h2>
    <div class="summary-row" style="margin-bottom:0" id="extrinsic-cards"></div>
    <div id="weekly-theta-bars" style="height:180px;margin-top:12px"></div>
    <div class="rec-why" style="margin-top:8px">
      Smoothness = <code>1 − σ(weekly_theta) / μ(weekly_theta)</code> across next 4 weeks.
      Higher = more even income. Production target ≥ 0.75.
    </div>
  </div>

  <!-- Roll forward planner (kernel's wheel-rhythm projection) -->
  <div class="section">
    <h2>🔄 Expire &amp; Reopen Planner — let OTM near-DTE expire, sell fresh (no penny-buybacks)</h2>
    <div class="summary-row" style="margin-bottom:0" id="roll-summary"></div>
    <div id="roll-theta-comparison" style="height:200px;margin-top:12px"></div>
    <div class="scrollable" style="margin-top:12px">
      <table id="roll-table"></table>
    </div>
    <div class="rec-why" style="margin-top:8px">
      For each near-DTE OTM contract (≤14d), planner suggests rolling to ~45 DTE at similar OTM%.
      Shifts week-1 theta into weeks 3-4, raising smoothness toward 0.75 target.
    </div>
  </div>

  <!-- What-If Delta Matrix (strike × DTE for sell-put vs sell-call) -->
  <div class="section">
    <h2>⚖️ What-If Effective-Delta Matrix — sell put vs sell call?</h2>
    <div class="summary-row" style="margin-bottom:0" id="whatif-summary"></div>
    <div class="grid-2" style="margin-top:12px">
      <div>
        <h3 style="font-size:0.95rem;color:var(--red);margin-bottom:8px">📉 Sell PUT (θ per |Δ|)</h3>
        <div class="scrollable"><table id="whatif-put-table"></table></div>
      </div>
      <div>
        <h3 style="font-size:0.95rem;color:var(--green);margin-bottom:8px">📈 Sell CALL (θ per |Δ|)</h3>
        <div class="scrollable"><table id="whatif-call-table"></table></div>
      </div>
    </div>
    <div class="rec-why" style="margin-top:8px">
      Cell value = <strong>theta-$ per |delta-shift|</strong> for selling 1 contract there.
      Higher = more premium for the delta you take on. Best cells highlighted in green.
      <strong>Tendency</strong> uses portfolio Δ vs neutral (6,200) to bias call vs put side.
    </div>
  </div>

  <!-- Position analytics charts (matches production Delta/Theta/PnL panels) -->
  <div class="section">
    <h2>📈 P&amp;L Profile at Expiration</h2>
    <div id="chart-pnl" style="height:300px"></div>
    <div class="rec-why" style="margin-top:8px">
      Portfolio P&amp;L if held to expiration across UNG range 70%-130% of current spot. Vertical line = current spot.
    </div>
  </div>

  <div class="grid-2">
    <div class="section">
      <h2>Δ Delta Exposure vs UNG Price</h2>
      <div id="chart-delta" style="height:280px"></div>
    </div>
    <div class="section">
      <h2>θ Daily Theta by Expiry</h2>
      <div id="chart-theta-bar" style="height:280px"></div>
    </div>
  </div>

  <div class="section">
    <h2>⌛ Theta Decay Waterfall (next 60 days)</h2>
    <div id="chart-theta-waterfall" style="height:280px"></div>
    <div class="rec-why" style="margin-top:8px">
      Cumulative theta projected forward, assuming positions held flat. Total = passive income from time decay.
    </div>
  </div>

  <div class="section">
    <h2>🗓 Rolling Calendar Grid (Strike × Expiry)</h2>
    <div class="scrollable">
      <table id="calendar-grid-table"></table>
    </div>
    <div class="rec-why" style="margin-top:8px">
      Cell value: net short qty (negative = short, positive = long). Color = C (call) vs P (put).
    </div>
  </div>

  <!-- Portfolio Greeks summary — fed from the SAME greeks-managed live engine as the
       SOT panel (current → post-order, incl 3rd-order). Single greeks truth. -->
  <div class="section">
    <h2>📐 Portfolio Greeks — full book, current → post-order</h2>
    <div class="sub" style="margin-bottom:10px">Same greeks-managed engine as the TODAY panel above (regime_wheel_boxx_greeks) — current book vs the book once today's orders fill. What-if from the fitted vol surface, never a fill. Includes 2nd-order (vanna, charm) and 3rd-order (speed ∂Γ/∂S, color ∂Γ/∂t).</div>
    <div class="summary-row" id="greeks-cards" style="margin-bottom:0"></div>
  </div>

  <!-- Per-position analysis — SUPERSEDED by SOT panel (hidden) -->
  <div class="section" style="display:none">
    <h2>🎯 Per-position action — what to do with each contract</h2>
    <div class="scrollable">
      <table>
        <thead><tr>
          <th>Right</th><th>K</th><th>Exp</th><th>DTE</th><th>Qty</th><th>Money</th>
          <th>Δ</th><th>θ/day</th><th>Action</th><th>Detail</th>
        </tr></thead>
        <tbody id="position-analysis"></tbody>
      </table>
    </div>
  </div>

  <!-- 30-day put expiration calendar -->
  <div class="section">
    <h2>🗓 Next 45 days — put expiration calendar</h2>
    <div class="scrollable"><table id="expiry-calendar">
      <thead><tr>
        <th>Expiry</th><th>DTE</th><th>Strike</th><th>Qty</th>
        <th>Collateral</th><th>Outcome</th><th>$ freed</th>
      </tr></thead>
      <tbody></tbody>
    </table></div>
    <div class="rec-why" style="margin-top:8px" id="calendar-summary">–</div>
  </div>

  <!-- IV Shape + Account -->
  <div class="grid-2">
    <div class="section">
      <h2>IV Shape (PG Real Surface)</h2>
      <div id="iv-shape">–</div>
    </div>
    <div class="section">
      <h2>Account</h2>
      <table id="account-table"></table>
    </div>
  </div>

  <!-- Charts behind the reasons -->
  <div class="section">
    <h2>Why this regime? — UNG price + Z-score history</h2>
    <div id="chart-regime" style="height:380px"></div>
    <div class="rec-why" style="margin-top:8px">
      Z = surprise-detrended storage z. <span style="color:var(--red)">z &gt; +1.5 = EXTREME_RICH</span> (mult 0.1),
      <span style="color:var(--orange)">+0.5 to +1.5 = RICH</span> (0.4), grey ±0.5 = NEUTRAL (1.0),
      <span style="color:var(--green)">-0.5 to -1.5 = CHEAP</span> (1.4), bright green &lt; -1.5 = EXTREME_CHEAP (2.0).
    </div>
  </div>

  <div class="section">
    <h2>Why this IV? — IV30 history (PG real surface)</h2>
    <div id="chart-iv" style="height:280px"></div>
    <div class="rec-why" style="margin-top:8px">
      Real market IV30 from PG <code>ung_iv_surface</code> (16,517 rows, 878 dates). Median 0.55,
      range 0.20-1.22 over 2017-2026.
    </div>
  </div>

  <div class="section">
    <h2>Backtest equity curve — champion_target_25_dd_trim @ $100K cash start</h2>
    <div id="chart-equity" style="height:380px"></div>
    <div class="rec-why" style="margin-top:8px">
      Full 5yr backtest used to validate the kernel. Drawdown overlay shows real DD episodes.
      Walk-forward 12mo worst window: <strong>-17%</strong> MDD (vs full-sample -7%).
    </div>
  </div>

  <div class="section">
    <h2>Walk-forward validation — rolling 12mo windows</h2>
    <div id="chart-walkforward" style="height:340px"></div>
    <div class="rec-why" style="margin-top:8px">
      Each bar is a 1-year run starting at a different date. The headline "Sharpe 2.58 / MDD -7%"
      is for the full window; the rolling test exposes that real worst-case MDD on any 12mo period
      reaches -17%. Lower bars = harder windows.
    </div>
  </div>

  <div class="section">
    <h2>Year-by-year P&amp;L</h2>
    <div id="chart-yearly" style="height:300px"></div>
    <table style="margin-top:12px" id="yearly-table">
      <thead><tr><th>Year</th><th>P&L $</th><th>P&L %</th><th>Sharpe</th><th>MDD %</th><th>Days</th></tr></thead>
      <tbody></tbody>
    </table>
  </div>

  <!-- Positions -->
  <div class="section">
    <h2>UNG Position Detail</h2>
    <div class="scrollable"><table>
      <thead><tr>
        <th>Type</th><th>Qty</th><th>Strike</th><th>Expiry</th>
        <th>Market Value</th><th>Unrealized P&amp;L</th>
      </tr></thead>
      <tbody id="positions">–</tbody>
    </table></div>
  </div>
</div>

<script>
function fmt(n, d=0) {
  return n == null || isNaN(n) ? '–' : Number(n).toLocaleString(undefined, {minimumFractionDigits:d, maximumFractionDigits:d});
}
function $(id) { return document.getElementById(id); }
function regimeClass(r) {
  if (!r) return 'tag-neutral';
  if (r.includes('RICH')) return 'tag-rich';
  if (r.includes('CHEAP')) return 'tag-cheap';
  return 'tag-neutral';
}
async function refresh() {
  try {
    const urlParams = new URLSearchParams(window.location.search);
    const k = urlParams.get('kernel');
    const url = k ? `/api/state?kernel=${encodeURIComponent(k)}` : '/api/state';
    const r = await fetch(url);
    const s = await r.json();
    const v = s.verdict || {};

    if (s.error) {
      $('error-row').innerHTML = `<div class="error">${s.error}</div>`;
    } else {
      $('error-row').innerHTML = '';
    }

    const age = s.last_refresh ? Math.round(Date.now()/1000 - s.last_refresh) : null;
    $('freshness').innerText = age != null ? age + 's ago' : 'never';
    $('kernel-tag').innerText = v.kernel || '?';
    $('kernel-fullname').innerText = v.kernel || '?';

    // Top cards
    $('spot-val').innerText = '$' + fmt(s.spot, 2);
    $('spot-date').innerText = v.date || '–';
    const regEl = $('regime-val');
    regEl.innerHTML = `<span class="tag ${regimeClass(v.regime)}">${v.regime || '–'}</span>`;
    $('regime-z').innerText = 'z = ' + (v.z_surprise == null ? '–' : v.z_surprise.toFixed(2));
    $('target-val').innerText = fmt(v.target_shares);
    $('target-mult').innerText = '× ' + (v.mult == null ? '–' : v.mult.toFixed(2)) + ' mult';
    const dv = $('delta-val');
    dv.innerText = v.share_delta == null ? '–' : ((v.share_delta>0?'+':'')+fmt(v.share_delta));
    dv.className = 'card-value ' + (v.share_delta>0?'positive':v.share_delta<0?'negative':'');
    const b = s.balance || {};
    $('nav-val').innerText = '$' + fmt(b.net_liquidation, 0);
    $('nav-pct').innerText = b.total_return_pct != null ? fmt(b.total_return_pct, 2) + '% return' : '–';
    const cv = $('collat-val');
    const cp = v.put_collateral_pct_nav;
    cv.innerText = cp == null ? '–' : (cp*100).toFixed(1) + '%';
    cv.className = 'card-value ' + (cp>0.8?'negative':cp>0.5?'warn':'positive');
    $('collat-warn').innerText = cp>0.8 ? '⚠ over-leveraged' : cp>0.5 ? 'elevated' : 'healthy';

    // (Recommendations section removed in unified layout — orders surface in
    //  "Directly Usable Orders" panel below via v.actionable_orders)

    // Expiry cards: group positions by expiry, classify card priority
    const positions = (s.positions || []).filter(p => p.symbol === 'UNG' && p.is_option);
    const byExpiry = {};
    positions.forEach(p => {
      if (!byExpiry[p.expiry]) byExpiry[p.expiry] = [];
      byExpiry[p.expiry].push(p);
    });
    const today = new Date();
    const exCards = Object.keys(byExpiry).sort().map(exp => {
      const ps = byExpiry[exp];
      const expDate = new Date(exp);
      const dte = Math.round((expDate - today) / 86400000);
      // Classify card priority
      let cls = 'ok', badge = 'OK', badgeCls = 'e-badge-ok';
      if (dte <= 3) { cls = 'critical'; badge = `${dte}D LEFT`; badgeCls = 'e-badge-critical'; }
      else if (dte <= 7) { cls = 'warning'; badge = `${dte}D`; badgeCls = 'e-badge-warning'; }
      else if (dte <= 14) { cls = 'caution'; badge = `${dte}D`; badgeCls = 'e-badge-caution'; }
      else { badge = `${dte}D`; }
      const calls = ps.filter(p => p.option_type === 'CALL').map(p => ({K: p.strike, qty: p.quantity, mv: p.market_value}));
      const puts = ps.filter(p => p.option_type === 'PUT').map(p => ({K: p.strike, qty: p.quantity, mv: p.market_value}));
      const callsHtml = calls.map(c => {
        const itm = c.K < s.spot;
        const cls2 = itm ? 'rec-assign' : 'rec-expire';
        return `<div class="rec-item ${cls2}"><strong>${c.qty}C</strong> @ $${c.K} ${itm?'(ITM)':'(OTM)'}</div>`;
      }).join('');
      const putsHtml = puts.map(p => {
        const itm = p.K > s.spot;
        const cls2 = itm ? 'rec-assign' : 'rec-expire';
        return `<div class="rec-item ${cls2}"><strong>${p.qty}P</strong> @ $${p.K} ${itm?'(ITM)':'(OTM)'}</div>`;
      }).join('');
      // Total notional (collateral for puts, share-coverage for calls)
      const put_collat = puts.reduce((sum, p) => sum + Math.abs(p.qty) * p.K * 100, 0);
      const call_lots = calls.reduce((sum, c) => sum + Math.abs(c.qty), 0);
      return `<div class="expiry-card ${cls}">
        <h3>${exp} <span class="e-badge ${badgeCls}">${badge}</span></h3>
        <div class="detail-row"><strong>${ps.length}</strong> contracts: ${calls.length}C / ${puts.length}P</div>
        ${put_collat > 0 ? `<div class="detail-row">Put collateral: <strong>$${fmt(put_collat, 0)}</strong></div>` : ''}
        ${call_lots > 0 ? `<div class="detail-row">Calls cover: <strong>${call_lots*100}</strong> shares</div>` : ''}
        ${callsHtml}${putsHtml}
      </div>`;
    }).join('');
    $('expiry-cards-grid').innerHTML = exCards || '<div class="rec-why" style="grid-column:1/-1">No UNG options expiring</div>';

    // Expiration calendar (next 45 days)
    const cal = v.expiration_calendar || [];
    const calRows = cal.map(c => {
      const cls = c.outcome === 'EXPIRE_OTM' ? 'positive' : c.outcome === 'ASSIGN' ? 'negative' : 'neutral';
      return `<tr>
        <td class="mono">${c.expiry}</td>
        <td class="mono">${c.dte}d</td>
        <td class="mono">$${c.strike}</td>
        <td class="mono">${c.qty}</td>
        <td class="mono">$${fmt(c.collateral,0)}</td>
        <td><span class="tag ${c.outcome==='EXPIRE_OTM'?'tag-cheap':c.outcome==='ASSIGN'?'tag-rich':'tag-neutral'}">${c.outcome}</span></td>
        <td class="mono ${cls}">$${fmt(c.freed_est,0)}</td>
      </tr>`;
    }).join('');
    const calBody = document.querySelector('#expiry-calendar tbody');
    if (calBody) calBody.innerHTML = calRows || '<tr><td colspan=7 class="rec-why">No upcoming put expirations</td></tr>';
    const totalCollat = cal.reduce((s,c) => s + c.collateral, 0);
    const totalFreed30 = cal.filter(c => c.dte <= 30).reduce((s,c) => s + c.freed_est, 0);
    if ($('calendar-summary')) {
      $('calendar-summary').innerHTML = cal.length === 0 ? 'No puts expiring in 45 days.' :
        `<strong>$${fmt(totalCollat,0)}</strong> total put collateral; <strong style="color:var(--green)">$${fmt(totalFreed30,0)}</strong> likely frees in next 30 days at current spot ($${fmt(s.spot,2)}).`;
    }

    // Warnings
    $('warnings').innerHTML = (v.warnings || []).map(w => `<div class="warning">⚠ ${w}</div>`).join('');

    // Active kernel banner + selector population
    if (v.available_kernels) {
      const sel = $('kernel-selector');
      if (sel && sel.options.length === 0) {
        v.available_kernels.forEach(k => {
          const opt = document.createElement('option');
          opt.value = k.key;
          opt.text = `${k.label}  (OOS: ${k.oos_ann.toFixed(0)}% / Sh ${k.oos_sharpe.toFixed(2)} / MDD ${k.oos_mdd.toFixed(0)}%)`;
          sel.appendChild(opt);
        });
        sel.value = v.kernel_key;
        sel.addEventListener('change', () => {
          const newKey = sel.value;
          window.location.href = '?kernel=' + encodeURIComponent(newKey);
        });
      } else if (sel) {
        sel.value = v.kernel_key;
      }
    }
    if (v.kernel_oos) {
      $('kernel-label').innerText = v.kernel_label || v.kernel;
      $('kernel-why').innerText = v.kernel_why || '';
      $('oos-ann').innerText = '+' + v.kernel_oos.ann_pct.toFixed(1) + '%';
      $('oos-sharpe').innerText = '+' + v.kernel_oos.sharpe.toFixed(2);
      $('oos-mdd').innerText = v.kernel_oos.mdd_pct.toFixed(1) + '%';
      if (v.kernel_is) {
        $('is-ann').innerText = `(in-sample: ${v.kernel_is.ann_pct.toFixed(1)}%)`;
        $('is-sharpe').innerText = `(in-sample: ${v.kernel_is.sharpe.toFixed(2)})`;
        $('is-mdd').innerText = `(in-sample: ${v.kernel_is.mdd_pct.toFixed(1)}%)`;
      }
    }

    // Actionable orders — concrete trades with OSI + limit ladder
    // ── EXECUTOR BRIEF ──
    try {
      const cs = v.composite_state || {};
      const tilt = cs.dba_wheel_tilt || {};
      const gw = v.gex_wall || (v.actionable_orders||[]).map(o=>o.gex_wall).find(x=>x) || null;
      const da = v.directional_ag || {};
      const daW = da.weights || {};
      const activeSleeves = Object.entries(daW).filter(([k,x]) => (x.weight_now||0) > 0);
      const cpc = cs.cpc_outlook || {};
      let brief = '<table style="font-size:0.85rem;line-height:1.7">';
      brief += `<tr><td style="color:var(--text-dim);padding-right:14px">ENSO</td><td>ONI ${cs.oni != null ? (cs.oni>0?'+':'')+cs.oni : '?'} · CPC peak El Niño ${cpc.peak_el_nino_pct ?? '?'}% (${cpc.issue_date ?? '?'})</td></tr>`;
      brief += `<tr><td style="color:var(--text-dim)">DBA tilt</td><td>score ${tilt.score ?? '?'} (${Object.entries(tilt.score_parts||{}).filter(([k,x])=>x===true).map(([k])=>k).join('+')||'none'}) · warn ${tilt.macro_warn_count ?? '?'} · ag carry targets ZEROED (carry &lt; BOXX)</td></tr>`;
      brief += `<tr><td style="color:var(--text-dim)">GEX wall</td><td>${gw ? `call wall <strong>$${gw.wall}</strong> (+$${Number(gw.wall_gex).toLocaleString()}/1%) · put wall $${gw.put_wall} — sell CCs AT/ABOVE the wall (74% final-week hold)` : 'computed on CC candidates only (none this cycle)'}</td></tr>`;
      brief += `<tr><td style="color:var(--text-dim)">Ag directional</td><td>${activeSleeves.length ? activeSleeves.map(([k,x])=>`<strong>${k}</strong> w=${x.weight_now}`).join(' · ') + ' — BUY shares per weight' : 'all sleeves FLAT (no confluence ≥2) — cash stays in BOXX'}${da.age_days != null ? ` <span style="color:var(--text-dim)">(state ${da.age_days}d old)</span>` : ''}</td></tr>`;
      const ivr = v.iv_rank_live || {};
      if (ivr.atm_iv != null) {
        const rcol = ivr.iv_rank > 0.8 ? 'var(--red)' : ivr.iv_rank < 0.2 ? 'var(--green)' : 'var(--text)';
        brief += `<tr><td style="color:var(--text-dim)">IV-rank</td><td><strong style="color:${rcol}">${(ivr.iv_rank*100).toFixed(0)}%</strong> (ATM IV ${(ivr.atm_iv*100).toFixed(1)}%) — ${ivr.regime ?? ''} <span style="color:var(--text-dim)">[real-chain factor: top-quintile → -23% fwd-63d]</span></td></tr>`;
      }
      brief += `<tr><td style="color:var(--text-dim)">Kernel knobs</td><td>${v.kernel_label ?? v.kernel ?? '?'} — KOLD hedge ${(v.kernel_params||{}).kold_shoulder_hedge ?? '?'} · IV-rank scaling ${(v.kernel_params||{}).iv_rank_z_scale ? 'ON' : 'off'} · GEX floor ${(v.kernel_params||{}).cc_gex_floor ? 'ON' : 'off'}</td></tr>`;
      const tips = v.exec_timing || [];
      if (tips.length) {
        brief += `<tr><td style="color:var(--text-dim)">Sell timing</td><td>${tips.map(t=>`• ${t}`).join('<br>')}</td></tr>`;
      }
      brief += `<tr><td style="color:var(--text-dim)">Cash rule</td><td>reserve $${fmt(v.ag_gap_reserve ?? 0,0)} for leg gaps → rest to BOXX ladder below</td></tr>`;
      brief += '</table>';
      $('executor-brief').innerHTML = brief;
    } catch (e) { $('executor-brief').innerHTML = '<span style="color:var(--red)">brief error: '+e+'</span>'; }

    const orders = v.actionable_orders;
    if (orders && orders.length) {
      $('actionable-orders').innerHTML = orders.map(o => {
        const priClass = o.priority === 'high' ? 'priority-h' : o.priority === 'medium' ? 'priority-m' : 'priority-l';
        let detail = '';
        if (o.order_type === 'SHARES' && o.limit_ladder) {
          detail = '<div style="margin-top:6px;padding:8px;background:var(--bg);border-radius:4px;border-left:2px solid var(--cyan)">'
                 + '<div style="font-size:0.72rem;color:var(--text-dim);margin-bottom:4px">Limit ladder:</div>'
                 + '<table style="font-size:0.85rem">'
                 + o.limit_ladder.map(l => `<tr><td class="mono" style="text-align:right">${l.qty}</td><td>shares @</td><td class="mono">$${l.limit_price}</td></tr>`).join('')
                 + '</table></div>';
        } else if (o.order_type.startsWith('SYNTHETIC')) {
          const legs = (o.legs || []).map(l => `<tr><td>${l.side}</td><td>${l.qty}</td><td class="mono">${l.symbol}</td><td class="mono">@~$${l.est_premium_per}</td></tr>`).join('');
          detail = `<div style="margin-top:6px;padding:8px;background:var(--bg);border-radius:4px;border-left:2px solid var(--purple)">`
                 + `<div style="font-size:0.72rem;color:var(--text-dim);margin-bottom:4px">Legs (put-call parity):</div>`
                 + `<table style="font-size:0.78rem">${legs}</table>`
                 + (o.net_debit_per_pair != null ? `<div style="font-size:0.78rem;margin-top:4px;color:var(--text-dim)">Net debit/pair: $${o.net_debit_per_pair} · Δ per pair: ${o.net_delta_per_pair}</div>` : '')
                 + (o.net_credit_per_pair != null ? `<div style="font-size:0.78rem;margin-top:4px;color:var(--text-dim)">Net credit/pair: $${o.net_credit_per_pair} · Δ per pair: ${o.net_delta_per_pair}</div>` : '')
                 + (o.cc_coverage_check ? `<div style="font-size:0.72rem;margin-top:4px;color:var(--green)">✓ ${o.cc_coverage_check}</div>` : '')
                 + (o.capital_efficiency ? `<div style="font-size:0.72rem;margin-top:4px;color:var(--text-dim)">💡 ${o.capital_efficiency}</div>` : '')
                 + `</div>`;
        } else if (o.order_type === 'CC_SKIPPED' || o.order_type === 'SYNTHETIC_SHORT_BLOCKED') {
          detail = `<div style="margin-top:6px;padding:8px;background:rgba(248,81,73,0.08);border-radius:4px;border-left:2px solid var(--red);font-size:0.78rem;color:var(--red)">⛔ CONSTRAINT ENFORCED — covered-calls-only rule</div>`;
        } else if (o.order_type === 'PUT_SHORT_MIX' && o.legs) {
          // Mix-and-match real-strike rendering
          const legsTable = o.legs.map(l => `<tr>
            <td class="mono">${l.qty}</td><td class="mono">${l.symbol}</td>
            <td class="mono">$${l.est_premium_per}</td>
            <td class="mono" style="color:var(--text-dim)">${l.effective_otm_pct}% OTM</td>
            <td class="mono">$${fmt(l.credit_total,0)}</td>
          </tr>`).join('');
          detail = `<div style="margin-top:6px;padding:8px;background:var(--bg);border-radius:4px;border-left:2px solid var(--blue)">`
                 + `<div style="font-size:0.72rem;color:var(--text-dim);margin-bottom:4px">`
                 + `Strike mix → target <strong>${o.target_otm_pct}%</strong> OTM, achieved <strong>${o.achieved_otm_pct}%</strong></div>`
                 + `<table style="font-size:0.78rem"><thead><tr><th>Qty</th><th>OSI</th><th>Prem/ct</th><th>Eff OTM</th><th>Credit</th></tr></thead><tbody>${legsTable}</tbody></table>`
                 + `<div style="font-size:0.78rem;margin-top:4px;color:var(--text-dim)">Total credit: $${fmt(o.est_credit_total,0)} · Collateral: $${fmt(o.collateral_required,0)}</div>`
                 + (o.whatif_stats ? `<div style="font-size:0.76rem;margin-top:6px;padding:6px;background:rgba(88,166,255,0.07);border-radius:4px"><strong>What-if (${o.whatif_stats.n_scenarios} historical scenarios):</strong> E[PnL] $${fmt(o.whatif_stats.e_pnl,0)} · P(assign) ${(o.whatif_stats.p_assign*100).toFixed(0)}% · P(loss) ${(o.whatif_stats.p_loss*100).toFixed(0)}% · p5/p95 $${fmt(o.whatif_stats.p5_pnl,0)}/$${fmt(o.whatif_stats.p95_pnl,0)} · CVaR5 $${fmt(o.whatif_stats.cvar5,0)}</div>` : '')
                 + `</div>`;
        } else if (o.order_type.startsWith('SELL_PUT_') && o.requires_consult) {
          // Consult-only ag-wheel candidates (DBA core + CORN/CANE satellites)
          const tilt = o.factor_tilt || {};
          const tiltStr = tilt.score != null ? `score ${tilt.score} → ${tilt.size_mult}x` : (o.size_mult ? `tilt ${o.size_mult}x` : '');
          detail = `<div style="margin-top:6px;padding:8px;background:var(--bg);border-radius:4px;border-left:2px solid var(--yellow)">`
                 + `<div class="mono" style="font-size:0.85rem">Target: <strong>${o.target_contracts ?? '?'}×</strong> ${o.symbol} P${o.target_strike ?? '?'} · ${o.target_dte_range ?? ''} DTE</div>`
                 + `<div class="mono" style="font-size:0.82rem;margin-top:4px"><span style="color:var(--text-dim)">Est credit:</span> $${o.est_credit_per_contract ?? '?'}/ct · total $${fmt(o.est_total_credit,0)}</div>`
                 + `<div class="mono" style="font-size:0.82rem"><span style="color:var(--text-dim)">Collateral target:</span> $${fmt(o.allocation_dollars,0)}` + (tiltStr ? ` · <span style="color:var(--cyan)">${tiltStr}</span>` : '') + `</div>`
                 + `<div style="font-size:0.72rem;margin-top:4px;color:var(--yellow)">⚠ CONSULT — chain lookup + manual submission required</div></div>`;
        } else if (o.order_type.includes('PUT') || o.order_type.includes('CALL')) {
          detail = `<div style="margin-top:6px;padding:8px;background:var(--bg);border-radius:4px">`
                 + `<div class="mono" style="font-size:0.85rem"><span style="color:var(--text-dim)">OSI:</span> ${o.symbol}</div>`
                 + `<div class="mono" style="font-size:0.82rem;margin-top:4px"><span style="color:var(--text-dim)">Limit range:</span> $${o.limit_low ?? '?'} – $${o.limit_high ?? '?'}/contract</div>`
                 + `<div class="mono" style="font-size:0.82rem"><span style="color:var(--text-dim)">Est credit:</span> $${fmt(o.est_credit_total ?? o.est_total_credit,0)}` +
                   (o.collateral_required ? ` &nbsp; <span style="color:var(--text-dim)">Collateral:</span> $${fmt(o.collateral_required,0)}` : '') + `</div></div>`;
        }
        return `<div class="rec-card">
          <div class="rec-header">
            <span class="rec-type-badge">${o.order_type}</span>
            <span class="rec-urgency-badge ${o.priority}">${o.priority}</span>
            <span style="margin-left:auto;font-weight:600">${o.side ?? ''}</span>
            <span class="mono">${o.symbol ?? ''}</span>
            <span class="mono">×${o.qty ?? o.qty_total ?? o.target_contracts ?? '–'}</span>
          </div>
          <div class="rec-why">${o.rationale || ''}</div>
          ${detail}
        </div>`;
      }).join('');
    } else {
      $('actionable-orders').innerHTML = '<div class="rec-why">No active orders from this kernel right now.</div>';
    }

    // Per-kernel beam — comparison table
    const beamByK = v.beam_by_kernel;
    if (beamByK) {
      let html = '';
      Object.entries(beamByK).forEach(([key, b]) => {
        const isActive = key === v.kernel_key;
        const borderColor = isActive ? 'var(--blue)' : 'var(--border)';
        html += `<div style="background:var(--bg);border:1px solid ${borderColor};border-radius:6px;padding:12px;margin-bottom:8px${isActive ? ';box-shadow:0 0 0 1px var(--blue)' : ''}">`;
        html += `<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px"><strong>${b.label}</strong>`;
        if (isActive) html += ' <span class="tag tag-cheap">ACTIVE</span>';
        html += `<span style="color:var(--text-dim);font-size:0.72rem;margin-left:auto">mode: ${b.mode}</span></div>`;
        html += '<table style="font-size:0.78rem"><thead><tr><th>Strike</th><th>OTM%</th><th>DTE</th><th>IV</th><th>Prem/ct</th><th>Qty</th><th>Income</th><th>P(ITM)</th><th>Net</th></tr></thead><tbody>';
        b.candidates.forEach((c, i) => {
          const isWinner = c.strike === b.winner_strike;
          const bg = isWinner ? 'style="background:rgba(63,185,80,0.1)"' : '';
          html += `<tr ${bg}><td class="mono">${isWinner ? '🏆' : ''} $${c.strike}</td><td class="mono">${c.otm_pct}%</td><td class="mono">${c.dte}d</td><td class="mono">${(c.iv*100).toFixed(1)}%</td><td class="mono">$${c.premium_per_contract}</td><td class="mono">${c.qty_recommended}</td><td class="mono positive">$${fmt(c.total_income, 0)}</td><td class="mono">${c.p_itm_pct}%</td><td class="mono ${isWinner?'positive':''}">$${fmt(c.net_score,0)}</td></tr>`;
        });
        html += '</tbody></table></div>';
      });
      $('beam-by-kernel').innerHTML = html;
    }

    // Daily status banner (production status-green/yellow/red classes)
    const ds = v.daily_status;
    if (ds) {
      const banner = $('daily-status-banner');
      const icons = {green: '✅', orange: '⚠️', red: '🚨'};
      const cls = {green: 'status-green', orange: 'status-yellow', red: 'status-red'};
      banner.className = 'daily-banner ' + (cls[ds.color] || 'status-green');
      $('status-icon').innerText = icons[ds.color] || '✅';
      $('status-headline').innerText = ds.headline;
      $('status-detail').innerText = ds.issues && ds.issues.length ? ds.issues.join(' · ') : 'No issues detected.';
    }

    // Extrinsic + smoothness cards
    const ex = v.extrinsic;
    if (ex) {
      $('extrinsic-cards').innerHTML = `
        <div class="card">
          <div class="card-label">Smoothness</div>
          <div class="card-value ${ex.smoothness>=0.75?'positive':ex.smoothness>=0.5?'warn':'negative'}">${ex.smoothness.toFixed(3)}</div>
          <div class="card-sub">target ≥ 0.75</div>
        </div>
        <div class="card">
          <div class="card-label">Total Extrinsic</div>
          <div class="card-value ${ex.total_extrinsic>0?'positive':'negative'}">$${fmt(ex.total_extrinsic,0)}</div>
          <div class="card-sub">time value remaining</div>
        </div>
        <div class="card">
          <div class="card-label">Avg Weekly θ</div>
          <div class="card-value neutral">$${fmt(ex.avg_weekly_theta,0)}</div>
          <div class="card-sub">across next 4 weeks</div>
        </div>
        <div class="card">
          <div class="card-label">30d Decay Est</div>
          <div class="card-value ${ex.extrinsic_decay_30d_est>0?'positive':'negative'}">$${fmt(ex.extrinsic_decay_30d_est,0)}</div>
          <div class="card-sub">expected to realize</div>
        </div>`;
      // Weekly theta bars
      if (ex.weekly_theta) {
        const _tmax = Math.max(...ex.weekly_theta.map(v=>+v||0), 1);
        Plotly.newPlot('weekly-theta-bars', [
          {x: ['Week 1','Week 2','Week 3','Week 4'], y: ex.weekly_theta, type:'bar',
           marker: {color: '#39d2c0'}, text: ex.weekly_theta.map(v=>'$'+fmt(v,0)),
           textposition: 'outside', textfont: {color: '#e6edf3'}, cliponaxis: false},
        ], {
          ...PLOTLY_LAYOUT_BASE,
          // headroom so the tallest bar's outside label isn't clipped at top
          yaxis: { ...PLOTLY_LAYOUT_BASE.yaxis, title: '$', range: [0, _tmax * 1.18] },
          margin: { l: 50, r: 10, t: 26, b: 30 },
        }, {displayModeBar: false, responsive: true});
      }
    }

    // Roll forward planner
    const rp = v.roll_plan;
    if (rp) {
      const currS = ex ? ex.smoothness : 0;
      const projS = rp.projected_smoothness;
      const sDelta = (projS - currS).toFixed(3);
      $('roll-summary').innerHTML = `
        <div class="card">
          <div class="card-label">Expire + Reopen</div>
          <div class="card-value neutral">${rp.roll_count}</div>
          <div class="card-sub">OTM legs: let expire, sell fresh</div>
        </div>
        <div class="card">
          <div class="card-label">Current Smoothness</div>
          <div class="card-value ${currS>=0.75?'positive':'warn'}">${currS.toFixed(3)}</div>
          <div class="card-sub">today</div>
        </div>
        <div class="card">
          <div class="card-label">Projected (post-roll)</div>
          <div class="card-value ${projS>=0.75?'positive':projS>currS?'neutral':'warn'}">${projS.toFixed(3)}</div>
          <div class="card-sub">Δ ${sDelta>0?'+':''}${sDelta}</div>
        </div>
        <div class="card">
          <div class="card-label">New-Open Credit</div>
          <div class="card-value ${rp.net_credit_total>0?'positive':'negative'}">$${fmt(rp.net_credit_total,0)}</div>
          <div class="card-sub">reopen credit (friction −$${fmt(rp.friction_total||0,0)}) · saves $${fmt(rp.savings_vs_roll_total||0,0)} vs rolling</div>
        </div>`;
      // Compare current vs projected weekly theta
      const projWk = rp.projected_weekly_theta || [0,0,0,0];
      const currWk = ex ? ex.weekly_theta : [0,0,0,0];
      const _rmax = Math.max(...currWk.map(v=>+v||0), ...projWk.map(v=>+v||0), 1);
      Plotly.newPlot('roll-theta-comparison', [
        {x: ['W1','W2','W3','W4'], y: currWk, type:'bar', name:'Current',
         marker: {color: 'rgba(57,210,192,0.6)'}, text: currWk.map(v=>'$'+fmt(v,0)),
         textposition: 'outside', textfont: {color: '#e6edf3'}, cliponaxis: false},
        {x: ['W1','W2','W3','W4'], y: projWk, type:'bar', name:'After rolls',
         marker: {color: 'rgba(63,185,80,0.7)'}, text: projWk.map(v=>'$'+fmt(v,0)),
         textposition: 'outside', textfont: {color: '#e6edf3'}, cliponaxis: false},
      ], {
        ...PLOTLY_LAYOUT_BASE,
        // headroom for outside labels; legend sits above the headroom, not the bars
        yaxis: { ...PLOTLY_LAYOUT_BASE.yaxis, title: '$ weekly θ', range: [0, _rmax * 1.22] },
        barmode: 'group',
        legend: { x: 0, y: 1.16, orientation: 'h' },
        margin: { l: 50, r: 10, t: 40, b: 30 },
      }, {displayModeBar: false, responsive: true});
      // Roll table — expire-and-reopen framing
      const rows = (rp.rolls || []).map(r => `
        <tr>
          <td><span class="tag ${r.old.right==='C'?'tag-call':'tag-put'}">${r.old.right}</span></td>
          <td class="mono">$${r.old.strike} <span style="color:var(--text-dim)">exp→sell</span> $${r.new.strike}</td>
          <td class="mono">${r.old.dte}d <span style="color:var(--text-dim)">→</span> ${r.new.dte}d</td>
          <td class="mono">${r.old.qty}</td>
          <td class="mono positive">+$${fmt(r.new_credit_per_contract,2)}</td>
          <td class="mono positive">$${fmt(r.net_credit_total,0)}</td>
          <td class="mono" style="color:var(--text-dim)">saves $${fmt(r.savings_vs_roll||0,0)}</td>
        </tr>`).join('');
      $('roll-table').innerHTML = `
        <thead><tr>
          <th>R</th><th>K (expire→sell)</th><th>DTE (old→new)</th><th>Qty</th>
          <th>New ea</th><th>Reopen $</th><th>vs roll</th>
        </tr></thead>
        <tbody>${rows || '<tr><td colspan="7" class="rec-why">No near-DTE rolls suggested</td></tr>'}</tbody>`;
    }

    // What-If delta matrix
    const wi = v.whatif_matrix;
    if (wi) {
      $('whatif-summary').innerHTML = `
        <div class="card">
          <div class="card-label">Tendency</div>
          <div class="card-value ${wi.tendency==='LEAN_CALL'?'neutral':wi.tendency==='LEAN_PUT'?'warn':'positive'}">${wi.tendency}</div>
          <div class="card-sub">${wi.tendency_reason}</div>
        </div>
        <div class="card">
          <div class="card-label">Avg Eff (PUT)</div>
          <div class="card-value ${wi.avg_eff_put>wi.avg_eff_call?'positive':''}">${wi.avg_eff_put.toFixed(3)}</div>
          <div class="card-sub">θ per |Δ|</div>
        </div>
        <div class="card">
          <div class="card-label">Avg Eff (CALL)</div>
          <div class="card-value ${wi.avg_eff_call>wi.avg_eff_put?'positive':''}">${wi.avg_eff_call.toFixed(3)}</div>
          <div class="card-sub">θ per |Δ|</div>
        </div>
        <div class="card">
          <div class="card-label">Best PUT cell</div>
          <div class="card-value neutral">$${wi.best_put.strike} / ${wi.best_put.dte}d</div>
          <div class="card-sub">eff ${wi.best_put.eff.toFixed(3)}, θ +$${fmt(wi.best_put.theta_chg,2)}</div>
        </div>
        <div class="card">
          <div class="card-label">Best CALL cell</div>
          <div class="card-value neutral">$${wi.best_call.strike} / ${wi.best_call.dte}d</div>
          <div class="card-sub">eff ${wi.best_call.eff.toFixed(3)}, θ +$${fmt(wi.best_call.theta_chg,2)}</div>
        </div>`;
      // Build matrices
      const buildTable = (mtx, side) => {
        const max_eff = Math.max(...mtx.flat().map(c => c.eff));
        const head = '<thead><tr><th>OTM%</th>' +
          wi.dtes.map(d => `<th class="mono">${d}d</th>`).join('') + '</tr></thead>';
        const body = mtx.map(row => {
          const otm = row[0].otm_pct;
          const cells = row.map(c => {
            const isBest = c.eff >= max_eff * 0.95;
            const bg = isBest ? 'style="background:rgba(63,185,80,0.15)"' : '';
            return `<td class="mono" ${bg} title="θ $${c.theta_chg}, Δ ${c.delta_chg}, IV ${(c.iv*100).toFixed(1)}%">
              ${c.eff.toFixed(2)}<br/>
              <span style="font-size:0.65rem;color:var(--text-dim)">$${c.strike}</span>
            </td>`;
          }).join('');
          return `<tr><td class="mono">${otm>0?'+':''}${otm}%</td>${cells}</tr>`;
        }).join('');
        return head + '<tbody>' + body + '</tbody>';
      };
      $('whatif-put-table').innerHTML = buildTable(wi.put_matrix, 'P');
      $('whatif-call-table').innerHTML = buildTable(wi.call_matrix, 'C');
    }

    // P&L Curve at expiration
    const pnl = v.pnl_curve;
    if (pnl && pnl.prices) {
      const layout = {
        ...PLOTLY_LAYOUT_BASE,
        yaxis: { ...PLOTLY_LAYOUT_BASE.yaxis, title: 'P&L $' },
        xaxis: { ...PLOTLY_LAYOUT_BASE.xaxis, title: 'UNG at expiry $' },
        shapes: [{type:'line', x0:pnl.spot_now, x1:pnl.spot_now, yref:'paper', y0:0, y1:1,
                  line:{color:'#58a6ff', width:1, dash:'dash'}}],
      };
      const colors = pnl.pnl.map(v => v >= 0 ? '#3fb950' : '#f85149');
      Plotly.newPlot('chart-pnl', [
        {x: pnl.prices, y: pnl.pnl, type: 'scatter', mode: 'lines',
         line: {color: '#58a6ff', width: 2}, fill: 'tozeroy',
         fillcolor: 'rgba(63,185,80,0.1)'},
      ], layout, {displayModeBar: false, responsive: true});
    }

    // Delta curve
    const dc = v.delta_curve;
    if (dc && dc.prices) {
      Plotly.newPlot('chart-delta', [
        {x: dc.prices, y: dc.deltas, type: 'scatter', mode: 'lines',
         line: {color: '#bc8cff', width: 2}},
      ], {
        ...PLOTLY_LAYOUT_BASE,
        yaxis: { ...PLOTLY_LAYOUT_BASE.yaxis, title: 'Δ' },
        xaxis: { ...PLOTLY_LAYOUT_BASE.xaxis, title: 'UNG $' },
        shapes: [{type:'line', x0:dc.spot_now, x1:dc.spot_now, yref:'paper', y0:0, y1:1,
                  line:{color:'#58a6ff', width:1, dash:'dash'}}],
      }, {displayModeBar: false, responsive: true});
    }

    // Theta by expiry bar
    const tbe = v.theta_by_expiry;
    if (tbe && tbe.length) {
      const _tbmax = Math.max(...tbe.map(t => +t.theta_per_day || 0), 1);
      Plotly.newPlot('chart-theta-bar', [
        {x: tbe.map(t => t.expiry), y: tbe.map(t => t.theta_per_day),
         type: 'bar', marker: {color: '#39d2c0'},
         text: tbe.map(t => '$' + t.theta_per_day.toFixed(0)), textposition: 'outside',
         textfont: {color: '#e6edf3'}, cliponaxis: false},
      ], {
        ...PLOTLY_LAYOUT_BASE,
        yaxis: { ...PLOTLY_LAYOUT_BASE.yaxis, title: '$/day', range: [0, _tbmax * 1.18] },
        xaxis: { ...PLOTLY_LAYOUT_BASE.xaxis, type: 'category', tickangle: -45 },
        margin: { ...PLOTLY_LAYOUT_BASE.margin, t: 22 },
      }, {displayModeBar: false, responsive: true});
    }

    // Theta waterfall
    const tw = v.theta_waterfall;
    if (tw && tw.length) {
      Plotly.newPlot('chart-theta-waterfall', [
        {x: tw.map(t => t.day), y: tw.map(t => t.cumulative_theta),
         type: 'scatter', mode: 'lines+markers',
         line: {color: '#3fb950', width: 2}, fill: 'tozeroy',
         fillcolor: 'rgba(63,185,80,0.15)', name: 'Cumulative'},
        {x: tw.map(t => t.day), y: tw.map(t => t.daily_theta * 3),
         type: 'bar', marker: {color: 'rgba(57,210,192,0.5)'}, name: 'Daily x3', yaxis: 'y2'},
      ], {
        ...PLOTLY_LAYOUT_BASE,
        yaxis: { ...PLOTLY_LAYOUT_BASE.yaxis, title: 'Cumulative $' },
        yaxis2: { title: '$/3d', overlaying: 'y', side: 'right', gridcolor: 'transparent' },
        xaxis: { ...PLOTLY_LAYOUT_BASE.xaxis, title: 'days ahead' },
        legend: { x: 0, y: 1.1, orientation: 'h' },
      }, {displayModeBar: false, responsive: true});
    }

    // Calendar grid
    const cg = v.calendar_grid;
    if (cg && cg.strikes && cg.expiries) {
      const head = '<thead><tr><th>Strike\\Expiry</th>' +
        cg.expiries.map(e => `<th class="mono">${e.slice(5)}</th>`).join('') + '</tr></thead>';
      const cells = {};
      cg.cells.forEach(c => { cells[c.strike + '|' + c.expiry] = c; });
      const body = cg.strikes.map(k => {
        const itm = (k > cg.spot * 0.95 && k < cg.spot * 1.05) ? 'style="background:rgba(88,166,255,0.05)"' : '';
        return '<tr ' + itm + '><td class="mono">$' + k.toFixed(1) + '</td>' +
          cg.expiries.map(e => {
            const c = cells[k + '|' + e];
            if (!c) return '<td>–</td>';
            const parts = [];
            if (c.C !== 0) parts.push(`<span style="color:${c.C>0?'var(--green)':'var(--red)'}">${c.C>0?'+':''}${c.C}C</span>`);
            if (c.P !== 0) parts.push(`<span style="color:${c.P>0?'var(--green)':'var(--red)'}">${c.P>0?'+':''}${c.P}P</span>`);
            return `<td class="mono" style="font-size:0.7rem">${parts.join('<br/>') || '–'}</td>`;
          }).join('') + '</tr>';
      }).join('');
      document.querySelector('#calendar-grid-table').innerHTML = head + '<tbody>' + body + '</tbody>';
    }

    // Portfolio Greeks cards are populated by drawSOT() from /api/live (the SAME
    // greeks-managed engine as the TODAY panel) so there is ONE greeks truth — see
    // renderGreeksCards(). Nothing to do here from the /api/state verdict path.

    // Per-position analysis table
    const pa = v.position_analysis || [];
    const actionClass = (a) => {
      if (a.includes('CLOSE') || a.includes('ASSIGNMENT') || a.includes('BUYBACK')) return 'tag-rich';
      if (a.includes('EXPIRE') || a.includes('HOLD')) return 'tag-cheap';
      return 'tag-neutral';
    };
    const paRows = pa.map(r => `
      <tr>
        <td><span class="tag ${r.right==='C'?'tag-call':'tag-put'}">${r.right}</span></td>
        <td class="mono">$${r.strike}</td>
        <td class="mono">${r.expiry}</td>
        <td class="mono">${r.dte}d</td>
        <td class="mono">${r.qty}</td>
        <td><span class="tag tag-${r.moneyness==='ITM'?'rich':r.moneyness==='OTM'?'cheap':'neutral'}">${r.moneyness}</span></td>
        <td class="mono ${r.delta>0?'positive':'negative'}">${fmt(r.delta,0)}</td>
        <td class="mono ${r.theta_per_day>0?'positive':'negative'}">$${fmt(r.theta_per_day,1)}</td>
        <td><span class="tag ${actionClass(r.action)}">${r.action}</span></td>
        <td class="rec-why">${r.action_detail || ''}</td>
      </tr>`).join('');
    $('position-analysis').innerHTML = paRows || '<tr><td colspan="10" class="rec-why">No UNG options held</td></tr>';

    // Deep beam analysis table
    const beam = v.beam_analysis;
    if (beam && beam.candidates) {
      const rows = beam.candidates.map((c, i) => {
        const isWinner = c.strike === beam.winner;
        const cls = isWinner ? 'style="background:rgba(63,185,80,0.1)"' : '';
        return `<tr ${cls}>
          <td class="mono">${isWinner ? '🏆' : ''} $${c.strike}</td>
          <td class="mono">${c.otm_pct}%</td>
          <td class="mono">${(c.iv*100).toFixed(1)}%</td>
          <td class="mono">$${c.premium}</td>
          <td class="mono">${c.p_itm_pct}%</td>
          <td class="mono positive">$${fmt(c.income_per_contract,1)}</td>
          <td class="mono negative">$${fmt(c.expected_loss_per_contract,1)}</td>
          <td class="mono ${isWinner?'positive':''}">$${fmt(c.net_score,1)}</td>
        </tr>`;
      }).join('');
      $('beam-content').innerHTML = `
        <table>
          <thead><tr>
            <th>Strike</th><th>OTM%</th><th>IV</th><th>Premium</th>
            <th>P(ITM)</th><th>Income</th><th>Exp Loss</th><th>Net Score</th>
          </tr></thead>
          <tbody>${rows}</tbody>
        </table>
        <div style="margin-top:8px;font-size:0.8rem;color:var(--text-dim)">
          Method: <strong>${beam.method}</strong> · DTE: ${beam.dte}d · Winner: <strong style="color:var(--green)">$${beam.winner}</strong>
          · IV source: <span class="badge">${beam.candidates[0].iv_source}</span>
        </div>`;
    } else {
      $('beam-content').innerHTML = '<div class="rec-why">No beam analysis available</div>';
    }

    // IV shape
    const iv = v.iv_shape || {};
    if (iv.atm_iv == null) {
      $('iv-shape').innerHTML = '<div class="rec-why">No PG surface for latest date</div>';
    } else {
      $('iv-shape').innerHTML = `
        <table>
          <thead><tr><th>ATM IV</th><th>Put Skew</th><th>Call Skew</th><th>P-C Skew</th><th>Term Slope</th></tr></thead>
          <tbody><tr>
            <td class="mono">${(iv.atm_iv*100).toFixed(1)}%</td>
            <td class="mono ${iv.put_skew>0.05?'positive':iv.put_skew<0?'negative':''}">${(iv.put_skew*100).toFixed(2)}pp</td>
            <td class="mono ${iv.call_skew>0.05?'positive':iv.call_skew<0?'negative':''}">${(iv.call_skew*100).toFixed(2)}pp</td>
            <td class="mono">${(iv.pc_skew*100).toFixed(2)}pp</td>
            <td class="mono ${iv.term_slope<-0.05?'negative':''}">${(iv.term_slope*100).toFixed(2)}pp</td>
          </tr></tbody>
        </table>`;
    }

    // Account
    $('account-table').innerHTML = `
      <thead><tr><th>Metric</th><th style="text-align:right">Value</th></tr></thead>
      <tbody>
        <tr><td>Net liquidation</td><td class="mono" style="text-align:right">$${fmt(b.net_liquidation, 2)}</td></tr>
        <tr><td>Total return</td><td class="mono ${b.total_return>0?'positive':'negative'}" style="text-align:right">$${fmt(b.total_return, 2)}</td></tr>
        <tr><td>Return %</td><td class="mono ${b.total_return_pct>0?'positive':'negative'}" style="text-align:right">${fmt(b.total_return_pct, 2)}%</td></tr>
        <tr><td>Short calls</td><td class="mono" style="text-align:right">${fmt(v.current_short_calls)}</td></tr>
        <tr><td>Short puts</td><td class="mono" style="text-align:right">${fmt(v.current_short_puts)}</td></tr>
        <tr><td>Put collateral $</td><td class="mono" style="text-align:right">$${fmt(v.current_put_collateral)}</td></tr>
        <tr><td>Shoulder season</td><td style="text-align:right">${v.shoulder_season ? '<span class="tag tag-rich">YES</span>' : '<span class="tag tag-neutral">no</span>'}</td></tr>
      </tbody>`;

    // Positions
    const ungPos = (s.positions || []).filter(p => p.symbol === 'UNG');
    const rows = ungPos.map(p => {
      if (p.is_option) {
        const tag = p.option_type === 'CALL' ? 'tag-call' : 'tag-put';
        return `<tr>
          <td><span class="tag ${tag}">${p.option_type}</span></td>
          <td class="mono">${p.quantity}</td>
          <td class="mono">$${p.strike}</td>
          <td>${p.expiry}</td>
          <td class="mono">$${fmt(p.market_value, 0)}</td>
          <td class="mono ${p.unrealized_pnl>0?'positive':'negative'}">$${fmt(p.unrealized_pnl, 0)}</td>
        </tr>`;
      }
      return `<tr>
        <td><span class="tag tag-share">SHARES</span></td>
        <td class="mono">${fmt(p.quantity)}</td>
        <td class="mono">avg $${fmt(p.average_price, 2)}</td>
        <td>–</td>
        <td class="mono">$${fmt(p.market_value, 0)}</td>
        <td class="mono ${p.unrealized_pnl>0?'positive':'negative'}">$${fmt(p.unrealized_pnl, 0)}</td>
      </tr>`;
    }).join('');
    $('positions').innerHTML = rows || '<tr><td colspan="6" class="rec-why">No UNG positions</td></tr>';

  } catch (e) {
    console.error('refresh() threw', e);
    const stack = (e.stack || '').split('\n').slice(0,3).join(' | ');
    $('error-row').innerHTML = `<div class="error">render failed: ${e.message} <span style="font-size:0.7rem;opacity:0.7">[${stack}]</span></div>`;
  }
}
async function refreshWithKernel() {
  // Force refresh with active kernel key in URL
  const key = window._activeKernel;
  const url = key ? `/api/state?kernel=${encodeURIComponent(key)}` : '/api/state';
  try {
    const r = await fetch(url);
    const s = await r.json();
    window._lastState = s;
    // Re-render by calling refresh path (it reads same JSON shape)
    refresh._renderFrom(s);
  } catch (e) {
    console.error('kernel switch failed', e);
  }
}

refresh();
setInterval(refresh, 30000);

// ── SINGLE SOURCE OF TRUTH panel (orders straight from the live engine) ──
function sotCard(label, val, sub, cls){
  return '<div class="card"><div class="card-label">'+label+'</div>'+
    '<div class="card-value '+(cls||'')+'">'+(val==null?'–':val)+'</div>'+
    '<div class="card-sub">'+(sub||'')+'</div></div>';
}
async function drawSOT(){
  try{
    const r = await fetch('/api/live'); const d = await r.json();
    const oel = document.getElementById('sot-orders');
    if(d.error){ oel.innerHTML='<div class="rec-why">live error: '+d.error+'</div>'; return; }
    document.getElementById('sot-kernel').textContent = d.kernel_label || d.kernel || '';
    // ── OPTIONS-DATA STALENESS banner: the options feed (IV surface + minute quotes) is
    //    SEPARATE from the daily price feed. If it has lagged, model prices use a carried-
    //    forward smile and 'live' quotes are N days old — warn loudly. ──
    const od = d.options_data;
    let oban = document.getElementById('sot-optstale');
    if (!oban) { oban=document.createElement('div'); oban.id='sot-optstale';
                 const k0=document.getElementById('sot-kernel'); k0.parentNode.insertBefore(oban, k0.nextSibling); }
    if (od && od.stale_days != null && od.stale_days > 1) {
      oban.innerHTML = '<div style="margin:8px 0;padding:9px 13px;border-radius:6px;background:#c6282814;'+
        'border-left:4px solid #c62828"><span style="font-weight:700;color:#c62828">⚠ OPTIONS DATA '+
        od.stale_days+' DAYS STALE</span> <span style="color:var(--text-dim)">— IV surface &amp; quotes as of '+
        (od.surface_asof||od.minute_asof)+' (price feed is current). Model prices use a carried-forward '+
        'smile; live quotes below are NOT today\'s market. Refresh ThetaData→PG before trusting option prices.</span></div>';
    } else { oban.innerHTML=''; }
    // ── REGIME banner: the state driving today's posture (accumulate/neutral/distribute) ──
    const rg = d.regime;
    if (rg) {
      const col = rg.state==='ACCUMULATE'?'#2e7d32':(rg.state==='DISTRIBUTE'?'#c62828':'#666');
      let rel = document.getElementById('sot-regime');
      if (!rel) { rel=document.createElement('div'); rel.id='sot-regime';
                  const z0=document.getElementById('sot-z'); z0.parentNode.insertBefore(rel, z0); }
      rel.innerHTML = '<div style="margin:6px 0 10px;padding:10px 14px;border-radius:6px;'+
        'background:'+col+'14;border-left:4px solid '+col+'">'+
        '<span style="font-weight:700;color:'+col+';font-size:1.05rem">REGIME: '+rg.state+'</span>'+
        '<span style="margin-left:14px;color:var(--text-dim)">storage-surprise z '+rg.storage_surprise_z+
        ' · strength '+rg.regime_strength+' · 60d price-dd '+rg.price_dd_60d+'%</span>'+
        '<div style="margin-top:4px;font-size:.9rem">↳ '+rg.posture+'</div>'+
        ((d.coverage)?('<div style="margin-top:4px;font-size:.85rem;font-weight:600;color:'+
          (d.coverage.covered?'#2e7d32':'#c62828')+'">🛡 COVERED-CALLS-ONLY: '+
          d.coverage.existing_short_calls+' short calls vs '+d.coverage.coverable_calls+
          ' coverable ('+d.coverage.shares+' shares) — '+
          (d.coverage.covered?'covered ✓':'OVER-WRITTEN ⚠')+'</div>'):'')+
        ((d.coverage&&d.coverage.violation)?('<div style="margin-top:4px;padding:6px 10px;'+
          'background:#c62828;color:#fff;font-weight:700;border-radius:4px;font-size:.85rem">🚨 '+
          d.coverage.violation+'</div>'):'')+'</div>';
    }
    const z = d.z_models||{};
    document.getElementById('sot-z').innerHTML =
      sotCard('Z — valuation', z.z_valuation, z.regime, z.regime==='CHEAP'?'positive':z.regime==='RICH'?'warn':'neutral')+
      sotCard('Surge-Z — momentum', z.surge_z_momentum, z.surge_z_momentum<-1?'dumping':z.surge_z_momentum>1?'ripping':'calm','')+
      sotCard('IV-rank', z.iv_rank==null?'–':z.iv_rank, (z.iv_rank!=null&&z.iv_rank<0.2)?'cheap vol':(z.iv_rank>0.6?'rich vol':''),'')+
      sotCard('HH basis', z.hh_basis, z.hh_basis>0.33?'backwardation⚠':'normal', z.hh_basis>0.33?'warn':'');
    const t = d.theta||{};
    document.getElementById('sot-theta').innerHTML =
      sotCard('Theta / day (now)', '$'+fmt(t.now_per_day,0), 'BS decay · front-loaded, don\'t ×30','')+
      sotCard('Time-value in book', '$'+fmt(t.extrinsic_today,0), 'max theta from today\'s legs','')+
      sotCard('Gross premium / mo', '$'+fmt(t.gross_premium_month,0), (t.gross_premium_pct||0)+'% NAV (backtest) · net ~break-even','')+
      sotCard('Signals as of', d.asof,
              (d.data_stale_days>0 ? ('⚠ '+d.data_stale_days+'d stale · today '+d.today) : ('today '+(d.today||''))),
              d.data_stale_days>1?'warn':'')+
      sotCard('Spot', '$'+fmt(d.spot,2), 'DTE/expiry use real today','');
    // ── BOOK GREEKS: current → post-order, incl 3rd-order speed/color ──
    const gel = document.getElementById('sot-greeks');
    if (d.greeks && gel) {
      const G = d.greeks, n = G.now, a = G.after, ch = G.change, ex = G.explain||{};
      // signed delta-arrow: green if the order moves the greek toward LESS risk
      const arrow = (was, now, lessIsSafer) => {
        const d2 = now - was; if (Math.abs(d2) < 1e-9) return '<span style="color:var(--text-dim)">→ flat</span>';
        const safer = lessIsSafer ? (Math.abs(now) < Math.abs(was)) : (now > was);
        const col = safer ? '#2e7d32' : '#c62828';
        return '<span style="color:'+col+'">'+(d2>0?'▲':'▼')+' '+(d2>0?'+':'')+fmt(d2,Math.abs(d2)<1?3:1)+'</span>';
      };
      const row = (label, was, now, lessIsSafer, sub) =>
        '<tr><td style="padding:3px 12px 3px 0;font-weight:600">'+label+'</td>'+
        '<td style="padding:3px 12px 3px 0;text-align:right;font-variant-numeric:tabular-nums">'+fmt(was,Math.abs(was)<1?3:1)+'</td>'+
        '<td style="padding:3px 8px;color:var(--text-dim)">→</td>'+
        '<td style="padding:3px 12px 3px 0;text-align:right;font-weight:600;font-variant-numeric:tabular-nums">'+fmt(now,Math.abs(now)<1?3:1)+'</td>'+
        '<td style="padding:3px 12px 3px 0">'+arrow(was,now,lessIsSafer)+'</td>'+
        '<td style="padding:3px 0;color:var(--text-dim);font-size:.82rem">'+(sub||'')+'</td></tr>';
      gel.innerHTML =
        '<table style="border-collapse:collapse;margin:2px 0 6px;font-size:.9rem">'+
        '<thead><tr style="color:var(--text-dim);font-size:.78rem;text-transform:uppercase">'+
        '<th style="text-align:left;padding-right:12px">Greek</th><th style="text-align:right">Now</th><th></th>'+
        '<th style="text-align:right;padding-right:12px">After orders</th><th style="text-align:left">Δ</th><th style="text-align:left">meaning</th></tr></thead><tbody>'+
        row('Δ delta (sh-eq)', n.delta, a.delta, true, '$'+fmt(a.delta_dollar_1pct,0)+' per +1% UNG · '+(Math.abs(a.delta)<Math.abs(n.delta)?'hedged toward neutral':'directional retained'))+
        row('Γ gamma', n.gamma, a.gamma, true, 'Δ shifts '+fmt(a.gamma_dollar_1pct,1)+' sh per +1% — short Γ hedges INTO moves')+
        row('θ theta /day', n.theta, a.theta, false, 'extrinsic decay in your favor')+
        row('vega /vol-pt', n.vega, a.vega, true, a.vega<0?'short vol — hurt by IV spike':'long vol')+
        '<tr><td colspan="6" style="padding:6px 0 2px;color:#39d2c0;font-size:.78rem;text-transform:uppercase;letter-spacing:.04em">— 2nd / 3rd-order —</td></tr>'+
        row('vanna ∂Δ/∂σ', n.vanna, a.vanna, true, 'Δ drift per +1 vol-pt — IV spike silently re-aims you')+
        row('charm ∂Δ/∂t', n.charm, a.charm, true, 'Δ drift per day even if UNG sits still — re-hedge daily')+
        row('speed ∂Γ/∂S', n.speed, a.speed, true, 'pin-risk acceleration as spot nears the strike wall')+
        row('color ∂Γ/∂t', n.color, a.color, true, 'front-week Γ blow-up rate as expiry nears')+
        '</tbody></table>'+
        '<div style="font-size:.82rem;color:var(--text-dim);line-height:1.5;border-left:2px solid #39d2c0;padding-left:10px;margin-top:4px">'+
        '<b style="color:var(--text)">Read:</b> '+(ex.delta||'')+' '+(ex.gamma||'')+'<br>'+
        '<b style="color:var(--text)">3rd-order:</b> '+(ex.speed||'')+' '+(ex.color||'')+'</div>';
    }
    // ── Dedicated "📐 Portfolio Greeks" section: SAME source (d.greeks), card layout
    //    so the standalone greeks panel can never diverge from the TODAY panel. ──
    const gc = document.getElementById('greeks-cards');
    if (d.greeks && gc) {
      const n = d.greeks.now, a = d.greeks.after;
      // each card: NOW → AFTER, colored by whether the orders move toward LESS risk
      const card = (label, was, now, lessIsSafer, fmtfn, sub) => {
        const f = fmtfn || (x => fmt(x, Math.abs(x)<1?3:1));
        const moved = Math.abs(now - was) > 1e-9;
        const safer = lessIsSafer ? (Math.abs(now) < Math.abs(was)) : (now > was);
        const cls = !moved ? 'neutral' : (safer ? 'positive' : 'negative');
        const arr = !moved ? '→' : (now>was?'▲':'▼');
        return '<div class="card"><div class="card-label">'+label+'</div>'+
          '<div class="card-value '+cls+'" style="font-size:1.05rem">'+f(was)+
          ' <span style="opacity:.5">'+arr+'</span> '+f(now)+'</div>'+
          '<div class="card-sub">'+sub+'</div></div>';
      };
      const usd = x => '$'+fmt(x,0);
      gc.innerHTML =
        card('Δ delta (sh-eq)', n.delta, a.delta, true, x=>fmt(x,0), 'net directional · '+usd(a.delta_dollar_1pct)+' per +1% UNG')+
        card('Γ gamma', n.gamma, a.gamma, true, x=>fmt(x,1), 'Δ-accel · '+fmt(a.gamma_dollar_1pct,1)+' sh per +1%')+
        card('θ theta /day', n.theta, a.theta, false, x=>'$'+fmt(x,0), 'extrinsic decay collected')+
        card('vega /vol-pt', n.vega, a.vega, true, x=>'$'+fmt(x,0), a.vega<0?'short vol':'long vol')+
        card('vanna ∂Δ/∂σ', n.vanna, a.vanna, true, x=>fmt(x,2), 'Δ drift per +1 vol-pt')+
        card('charm ∂Δ/∂t', n.charm, a.charm, true, x=>fmt(x,3), 'Δ drift per day (re-hedge)')+
        card('speed ∂Γ/∂S ⁳', n.speed, a.speed, true, x=>fmt(x,1), '3rd-order · pin-risk acceleration')+
        card('color ∂Γ/∂t ⁳', n.color, a.color, true, x=>fmt(x,2), '3rd-order · front-week Γ blow-up rate');
    }
    // ── DELTA COMPASS: net delta vs the engine's NAV-scaled, regime-cut target + WHY the
    //    hedge is dormant/active. Shows the glide-to-target logic (incremental, not rebuild). ──
    const cel = document.getElementById('sot-compass');
    const dc = d.delta_compass;
    if (cel && dc) {
      const gate = dc.gate_delta!=null ? dc.gate_delta : dc.net_delta;   // engine gates on this
      const kold = dc.kold_delta || 0;
      const lo = Math.min(0, gate, dc.net_delta, dc.target), hi = Math.max(gate, dc.net_delta, dc.target, dc.target+dc.band);
      const span = (hi-lo)||1, pct = x => (100*(x-lo)/span).toFixed(1)+'%';
      const ceil = dc.trim_ceiling!=null ? dc.trim_ceiling : dc.target;
      const col = dc.hedge_active ? '#c62828' : '#2e7d32';
      cel.innerHTML =
        '<h3 style="margin:14px 0 6px">🧭 Delta compass — net Δ vs <b>trim ceiling</b> '+
        '<span class="sub" style="font-weight:400">('+dc.glide+')</span></h3>'+
        // track: band just below the ceiling, ceiling tick, GATE-Δ marker (what fires the
        // hedge), dashed marker for the TRUE total Δ incl KOLD. Region BELOW the ceiling is
        // the safe 'no trim' zone (green); the hedge only fires to the RIGHT of the ceiling.
        '<div style="position:relative;height:26px;margin:6px 0 4px;background:var(--bg-alt,rgba(128,128,128,.12));border-radius:4px">'+
          '<div style="position:absolute;left:0;width:'+pct(ceil)+';top:0;bottom:0;background:rgba(46,125,50,.12);border-radius:4px 0 0 4px" title="below ceiling — no trim (safe)"></div>'+
          '<div style="position:absolute;left:'+pct(ceil)+';top:-2px;bottom:-2px;width:2px;background:#e08a00" title="trim ceiling (trim only if ABOVE, in a bear regime)"></div>'+
          '<div style="position:absolute;left:'+pct(gate)+';top:-4px;bottom:-4px;width:3px;background:'+col+'" title="options+shares Δ (engine gate)"></div>'+
          (Math.abs(kold)>=1?'<div style="position:absolute;left:'+pct(dc.net_delta)+';top:1px;bottom:1px;width:0;border-left:2px dashed #58a6ff" title="true net Δ incl KOLD"></div>':'')+
        '</div>'+
        '<div style="font-size:.86rem">options+shares Δ <b style="color:'+col+'">'+fmt(gate,0)+'</b> '+
          ' · trim ceiling <b style="color:#e08a00">'+fmt(ceil,0)+'</b> · '+
          (dc.headroom!=null?('<b style="color:#2e7d32">'+fmt(dc.headroom,0)+' headroom below</b>'):'gap '+fmt(dc.gap,0))+
          ' · regime '+dc.regime_strength+' vs '+dc.rs_min+
          ' · <b style="color:'+col+'">'+(dc.hedge_active?'HEDGE ACTIVE (trimming)':'hedge dormant')+'</b></div>'+
          (dc.share_pct_nav!=null?('<div style="font-size:.82rem;color:var(--text-dim);margin-top:1px">exposure set by the share target — shares ≈ '+dc.share_pct_nav+'% NAV (regime posture); the ceiling is a one-sided risk cap, not a goal to reach</div>'):'')+
          (Math.abs(kold)>=1?('<div style="font-size:.84rem;margin-top:2px">true net Δ <b style="color:#58a6ff">'+fmt(dc.net_delta,0)+
            '</b> = '+fmt(gate,0)+' options+shares '+(kold<0?'−':'+')+' '+fmt(Math.abs(kold),0)+' KOLD hedge '+
            '<span class="sub">(inverse-ETF, now counted — was a blind spot)</span></div>'):'')+
        '<div style="font-size:.82rem;color:var(--text-dim);border-left:2px solid '+col+';padding-left:10px;margin-top:4px">↳ '+dc.status+'</div>';
    }
    // ── PER-STRIKE CONCENTRATION: short clusters vs the gamma-cap (forward-only, so legacy
    //    clusters persist). Flag over-cap strikes with an incremental de-risk suggestion. ──
    const xel = document.getElementById('sot-concentration');
    const conc = d.concentration || [];
    if (xel) {
      const ar = d.assign_risk || {};
      const era = d.expiry_reaccum;
      let eraHtml = '';
      if (era && era.puts && era.puts.length) {
        const plist = era.puts.map(p=>'<b>SELL '+p.qty+'× $'+p.K.toFixed(2)+' put</b> ('+p.dte+'d, +$'+fmt(p.credit,0)+')').join(' · ');
        eraHtml = '<div style="margin:0 0 10px;padding:10px 13px;border-radius:6px;background:#1565c018;border-left:3px solid #1565c0">'+
          '<div style="font-weight:700;color:#1565c0">⏰ FRIDAY RE-ACCUMULATION — what to SELL pre-close ('+era.clusters.join(', ')+' called away)</div>'+
          '<div style="margin-top:5px;font-size:1.02rem">'+plist+'</div>'+
          '<div style="margin-top:5px;font-size:.82rem;color:var(--text-dim)">'+era.note+'</div></div>';
      }
      // surface the SELL at the TOP of "Orders for today" (not buried in the concentration panel)
      const reEl = document.getElementById('sot-reaccum');
      if (reEl) reEl.innerHTML = eraHtml;
      const flagged = conc.filter(c => c.over_cap);
      if (!conc.length) { xel.innerHTML = ''; }
      else {
        const within = ar.within_target;
        const arcol = within ? '#26a269' : '#e08a00';
        const ex = ar.execution || {};
        const ca = ar.called_away_soon;
        const net = ar.net_delta||0;
        let head =
          '<h3 style="margin:14px 0 6px">🎯 Assignment-risk — <b>statistical model</b> '+
          '<span style="font-size:.76rem;color:var(--text-dim)">(prob-weighted, DTE-aware · z='+(ar.z!=null?ar.z:'–')+')</span></h3>'+
          '<div style="margin:0 0 9px;padding:9px 13px;border-radius:6px;background:'+arcol+'14;border-left:3px solid '+arcol+'">'+
          '<div><b>Net expected share-Δ '+(net>=0?'+':'')+fmt(net,0)+'</b> '+
          '<span style="color:var(--text-dim);font-size:.85rem">= puts <span style="color:#26a269">+'+fmt(ar.put_assign_delta,0)+'</span> (assign, down) − calls <span style="color:#c62828">'+fmt(ar.call_away_delta,0)+'</span> (called away, up)</span></div>'+
          '<div style="margin-top:4px"><b style="color:'+arcol+'">Put-assignment Δ +'+fmt(ar.put_assign_delta,0)+'</b> = '+ar.pct_of_shares+'% of '+fmt(ar.shares,0)+' shares '+
          '<span style="color:'+arcol+';font-weight:600">('+(within?'WITHIN':'OVER')+' '+ar.target_pct+'% target)</span></div>'+
          (ca ? '<div style="margin-top:6px;font-size:.88rem;color:#c62828">📤 <b>Called away soon:</b> '+ca.note+' <span style="color:var(--text-dim)">['+ca.clusters.join(', ')+']</span></div>' : '')+
          (ex.roll_n>0
            ? '<div style="margin-top:6px;font-size:.88rem">🔧 <b>Execution aid:</b> '+ex.how+' <span style="color:var(--text-dim)">('+ex.reason+')</span></div>'
            : '<div style="margin-top:6px;font-size:.86rem;color:var(--text-dim)">✓ put spread is fine — no roll needed.</div>')+
          '</div>';
        const rows = conc.map(c => {
          const col = c.over_cap ? '#e08a00' : 'var(--text-dim)';
          const isP = c.right==='PUT';
          const ed = (c.exp_assign_delta!=null) ? (isP?'+':'−')+fmt(c.exp_assign_delta,0)+' Δ' : '–';
          const edcol = isP ? '#26a269' : '#c62828';
          const lbl = isP ? 'assign' : 'called away';
          const det = (c.exp_detail||[]).map(e=>e.contracts+'@'+e.dte+'d→'+Math.round((e.prob||0)*100)+'%').join('  ');
          return '<tr style="border-bottom:1px solid rgba(128,128,128,.12)">'+
            '<td style="padding:3px 12px 3px 0;font-weight:600;color:'+col+'">'+c.contracts+'× $'+c.strike.toFixed(2)+' '+c.right+'</td>'+
            '<td style="padding:3px 12px 3px 0;font-weight:600;color:'+edcol+'">'+ed+'<span style="color:var(--text-dim);font-weight:400;font-size:.72rem"> '+lbl+'</span></td>'+
            '<td style="padding:3px 12px 3px 0;font-size:.78rem;color:var(--text-dim)">'+(det||'–')+'</td>'+
            '<td style="padding:3px 0;color:'+col+';font-size:.8rem">'+(c.over_cap?'⚠ ':'')+c.max_single_expiry+'/exp vs notional '+(c.cap||'–')+'</td></tr>';
        }).join('');
        xel.innerHTML = head +
          '<table style="border-collapse:collapse;font-size:.9rem"><thead><tr style="color:var(--text-dim);font-size:.76rem;text-transform:uppercase">'+
          '<th style="text-align:left;padding-right:12px">cluster</th><th style="text-align:left">expected Δ</th><th style="text-align:left">prob by expiry</th><th style="text-align:left">notional cap (conservative)</th></tr></thead><tbody>'+rows+'</tbody></table>'+
          (flagged.length?'<div style="font-size:.74rem;color:var(--text-dim);margin-top:5px">⚠ = over the flat <i>notional</i> cap (assumes <i>certain</i> assignment → over-conservative). The <b>expected Δ</b> above (weighted by real odds within each DTE) is the honest risk.</div>':'');
      }
    }
    // ── SETTLEMENT WATCH: expiring options are ACTIONS too (await-worthless / assign /
    //    called-away / pin). They move shares/cash/coverage — surface them for execution. ──
    const sel2 = document.getElementById('sot-settlement');
    const sett = d.settlement || [];
    if (sel2) {
      if (!sett.length) { sel2.innerHTML = ''; }
      else {
        const meta = {
          EXPECT_ASSIGNMENT:  {ic:'📥', col:'#c62828', tag:'ASSIGNMENT'},
          EXPECT_CALLED_AWAY: {ic:'📤', col:'#e08a00', tag:'CALLED AWAY'},
          UNCERTAIN:          {ic:'⚠',  col:'#e08a00', tag:'PIN RISK'},
          DECIDE_LONG:        {ic:'🎯', col:'#2e7d32', tag:'EXERCISE'},
          AWAIT_WORTHLESS:    {ic:'✓',  col:'#2e7d32', tag:'EXPIRES OTM'},
          ABANDON_LONG:       {ic:'✓',  col:'var(--text-dim)', tag:'ABANDON'},
        };
        const needsDecision = sett.filter(s=>['EXPECT_ASSIGNMENT','EXPECT_CALLED_AWAY','UNCERTAIN'].includes(s.kind)).length;
        let netCash=0, netSh=0; sett.forEach(s=>{netCash+=s.cash_impact||0; netSh+=s.share_impact||0;});
        const rows = sett.map(s=>{
          const m = meta[s.kind]||{ic:'•',col:'var(--text-dim)',tag:s.kind};
          const impact = (s.cash_impact||s.share_impact)
            ? ('<span style="color:'+m.col+'">'+(s.share_impact?(s.share_impact>0?'+':'')+fmt(s.share_impact,0)+' sh':'')+
               (s.cash_impact?'  '+(s.cash_impact>0?'+':'')+'$'+fmt(s.cash_impact,0):'')+'</span>')
            : '<span style="color:var(--text-dim)">no cash/share move</span>';
          return '<div class="rec" style="border-left:3px solid '+m.col+';padding-left:10px;margin-bottom:6px">'+
            '<div class="rec-action" style="font-weight:600">'+m.ic+' <span style="color:'+m.col+
              ';font-size:.75rem;font-weight:700;letter-spacing:.04em">['+m.tag+']</span> '+s.action+'</div>'+
            '<div class="rec-why" style="color:var(--text-dim)">↳ '+(s.why||'')+'  ·  '+impact+'</div></div>';
        }).join('');
        sel2.innerHTML =
          '<h3 style="margin:14px 0 6px">⏳ Settlement today — expiring positions (these are actions: monitor / let settle / decide)'+
          (needsDecision?(' <span style="color:#c62828;font-size:.8rem">· '+needsDecision+' need a decision before the close</span>'):'')+'</h3>'+
          rows+
          '<div style="font-size:.82rem;color:var(--text-dim);margin-top:2px">Net if all settle as-is: '+
            (netSh?((netSh>0?'+':'')+fmt(netSh,0)+' shares'):'no share change')+
            (netCash?('  ·  '+(netCash>0?'+':'')+'$'+fmt(netCash,0)+' cash'):'')+
            '  ·  coverage recomputes after settlement.</div>';
      }
    }
    const recs = d.recommendations||[];
    oel.innerHTML = recs.length ? recs.map(o => {
      const ep = o.exec_plan;
      let exec = '';
      if (ep && ep.live_quote){
        const q = ep.live_quote, lad = ep.ladder||{};
        const rungs = (lad.rungs||[]).map(rg =>
          '<tr><td style="padding:1px 10px 1px 0;color:var(--text-dim)">'+rg.clock+'</td>'+
          '<td style="padding:1px 10px 1px 0;font-weight:600">$'+rg.limit.toFixed(2)+'</td>'+
          '<td style="color:var(--text-dim)">'+rg.rung+'</td></tr>').join('');
        exec = '<div class="exec-plan" style="margin:6px 0 2px 14px;padding:8px 10px;'+
          'border-left:2px solid #4a9;background:rgba(74,153,136,.07);border-radius:4px;font-size:.82rem">'+
          '<div style="margin-bottom:3px">⏱ <b>'+((ep.timing.recommended||'').replace(/^\s*work near\s*/i,'Work near '))+'</b></div>'+
          '<div style="color:var(--text-dim);margin-bottom:4px">Live: bid $'+q.bid.toFixed(2)+
            ' / ask $'+q.ask.toFixed(2)+' / mid $'+q.mid.toFixed(2)+'  ('+q.spread_pct+
            '% wide · P(mid) '+lad.p_mid+')</div>'+
          '<div style="margin-bottom:2px">Limit ladder (patient→cross):</div>'+
          '<table style="margin-left:6px">'+rungs+'</table>'+
          '<div style="margin-top:3px">Expected fill ≈ <b>$'+(lad.expected_fill||0).toFixed(2)+'</b></div>'+
          ((ep.caveats&&ep.caveats.length)?'<div class="warn" style="margin-top:3px">⚠ '+ep.caveats[0]+'</div>':'')+
          '</div>';
      } else if (ep && ep.note){
        exec = '<div style="margin:4px 0 2px 14px;color:var(--text-dim);font-size:.8rem">⏱ '+
          (ep.timing?ep.timing.recommended:'')+' · '+ep.note+'</div>';
      }
      // ── RECONCILE: model price vs REAL executable price (accuracy guard) ──
      let recon = '';
      const rc = o.reconcile;
      if (rc && rc.real_buyback != null && rc.model_buyback != null) {
        // STALE data = real red warning (don't trust the price). A price-update on FRESH
        // data is NOT a warning — the order is a validated decision; we just fill it at the
        // real market. Green/info, never 'reconsider'. (live = backtest: execute every order.)
        const stale = (rc.quote_stale_days||0) >= 1;
        const col = stale ? '#c62828' : '#2e7d32';
        const qlabel = stale ? ('real fill ('+rc.quote_stale_days+'d-STALE, '+rc.quote_asof+')') : 'real fill';
        recon = '<div style="margin:5px 0 2px 14px;padding:6px 10px;border-left:2px solid '+col+
          ';background:'+col+'12;border-radius:4px;font-size:.82rem">'+
          '<b style="color:'+col+'">'+(stale?'⚠ STALE price':'✓ fill at real price')+'</b>  '+
          'engine $'+fmt(rc.model_buyback,2)+' (+$'+fmt(rc.model_pnl,0)+') '+
          '→ <b>'+qlabel+' ≈$'+fmt(rc.real_buyback,2)+'</b> → realistic <b>+$'+fmt(rc.real_pnl,0)+'</b>'+
          (rc.stale_warning?('<div style="color:#c62828;margin-top:2px">'+rc.stale_warning+'</div>'):'')+
          (rc.flag && !stale?('<div style="color:var(--text-dim);margin-top:2px">'+rc.flag+'</div>'):'')+'</div>';
      }
      return '<div class="rec"><div class="rec-action" style="font-weight:600">'+o.action+
        (o.credit?' <span class="positive">+$'+fmt(o.credit,0)+'</span>':'')+'</div>'+
        '<div class="rec-why" style="color:var(--text-dim)">↳ '+(o.why||'')+'</div>'+recon+exec+'</div>';
    }).join('')
      : '<div class="rec-why">Engine holds — no new orders today.</div>';
  }catch(e){ document.getElementById('sot-orders').innerHTML='<div class="rec-why">live fetch failed: '+e+'</div>'; }
}
drawSOT();
setInterval(drawSOT, 60000);

// ── SPY vega-scrape VIX<=16 setup alert ──
async function drawSpyVega(){
  try{
    const d = await (await fetch('/api/spy_vega')).json();
    const sec = document.getElementById('spy-vega-alert');
    const vEl = document.getElementById('sv-verdict');
    const bEl = document.getElementById('sv-body');
    if(d.error){ bEl.innerHTML='alert error: '+d.error; return; }
    const COL = {GREEN:'#26a269', CAUTION:'#e5a50a', WARNING:'#e8841a', RED:'#666'};
    const col = COL[d.verdict] || '#666';
    sec.style.borderColor = col;
    vEl.textContent = d.verdict + (d.size && d.size!=='0' ? ' · '+d.size+' size' : '');
    vEl.style.background = col; vEl.style.color = '#fff';
    const chk = (ok)=> ok ? '<span style="color:#26a269">✓</span>' : '<span style="color:#c01c28">✗</span>';
    const pct = (x)=> (x*100).toFixed(0)+'%';
    bEl.innerHTML =
      `<b style="color:${col}">${d.msg}</b><br>`+
      `VIX <b>${d.vix.toFixed(1)}</b> ${chk(d.low_vix)} (≤16) &nbsp;|&nbsp; `+
      `IV ${pct(d.iv)} vs RV20 ${pct(d.rv20)} ${chk(d.not_cheap)} (not-cheap) &nbsp;|&nbsp; `+
      `10d-std ${d.vix_std10.toFixed(2)} ${chk(d.consolidated)} (consolidated) &nbsp;|&nbsp; `+
      `dist-from-high ${pct(d.dist_high)} &nbsp;|&nbsp; SPY ${d.spy.toFixed(2)}`+
      `<span class="sub" style="display:block;margin-top:4px">src: ${d.src} · asof ${d.asof}</span>`;
  }catch(e){ document.getElementById('sv-body').innerHTML='alert fetch failed: '+e; }
}
drawSpyVega();
setInterval(drawSpyVega, 600000); // 10 min

// ── SCRATCH NOISE PANELS — keep only the genuinely useful ones ──
(function scrubNoise(){
  // Hide only the genuinely-noisy / stale-kernel panels. KEEP the execution-relevant
  // DETAIL: Per-position action, Portfolio Greeks, Delta Exposure, Theta-by-Expiry,
  // Theta Waterfall, P&L Profile, Expiration Timeline, Rolling Calendar, UNG Position Detail.
  const HIDE = ['Executor Brief','Directly Usable','Beam','Deep Beam',
    'What-If','Expire &amp; Reopen','Expire & Reopen','Theta Smoothness',
    'champion_target_25_dd_trim'];  // stale backtest curve (old kernel)
  document.querySelectorAll('.section').forEach(sec=>{
    const h=sec.querySelector('h2'); if(!h) return;
    if(HIDE.some(k=>h.textContent.includes(k.replace('&amp;','&')))) sec.style.display='none';
  });
})();

// Chart rendering (10-minute cache on backend)
const PLOTLY_LAYOUT_BASE = {
  paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
  font: { color: '#e6edf3', family: 'ui-monospace, SFMono-Regular, monospace', size: 11 },
  margin: { l: 50, r: 30, t: 10, b: 40 },
  xaxis: { gridcolor: '#21262d', linecolor: '#30363d' },
  yaxis: { gridcolor: '#21262d', linecolor: '#30363d' },
};
async function drawCharts() {
  try {
    const r = await fetch('/api/analytics');
    const a = await r.json();
    if (a.error) { console.error(a.error); return; }
    // cache still warming (first ~76s after a restart) → {} ; skip and let the next poll draw.
    if (!a.series || !a.backtest_curve || !a.walkforward || !a.yearly) { setTimeout(drawCharts, 8000); return; }

    // Regime: UNG price + z overlay
    const s = a.series;
    Plotly.newPlot('chart-regime', [
      { x: s.dates, y: s.ung, type: 'scatter', mode: 'lines', name: 'UNG ($)',
        line: {color: '#58a6ff', width: 1.5}, yaxis: 'y' },
      { x: s.dates, y: s.z, type: 'scatter', mode: 'lines', name: 'Z (surprise)',
        line: {color: '#bc8cff', width: 1.5, dash: 'dot'}, yaxis: 'y2' },
    ], {
      ...PLOTLY_LAYOUT_BASE,
      yaxis: { ...PLOTLY_LAYOUT_BASE.yaxis, title: 'UNG $', side: 'left' },
      yaxis2: { title: 'Z', overlaying: 'y', side: 'right',
                gridcolor: 'transparent', zeroline: true, zerolinecolor: '#30363d',
                range: [Math.min(...s.z, -3) * 1.05, Math.max(...s.z, 3) * 1.05] },
      shapes: [
        // Regime bands as horizontal stripes on z axis
        {type:'rect', xref:'paper', yref:'y2', x0:0, x1:1, y0:1.5, y1:3, fillcolor:'rgba(248,81,73,0.08)', line:{width:0}},
        {type:'rect', xref:'paper', yref:'y2', x0:0, x1:1, y0:-1.5, y1:-3, fillcolor:'rgba(63,185,80,0.08)', line:{width:0}},
      ],
      legend: { x: 0, y: 1.1, orientation: 'h' },
    }, {displayModeBar: false, responsive: true});

    // IV30 history
    Plotly.newPlot('chart-iv', [
      { x: s.dates, y: s.iv30, type: 'scatter', mode: 'lines', name: 'IV30',
        line: {color: '#39d2c0', width: 1.5}, fill: 'tozeroy', fillcolor: 'rgba(57,210,192,0.1)' },
    ], {
      ...PLOTLY_LAYOUT_BASE,
      yaxis: { ...PLOTLY_LAYOUT_BASE.yaxis, title: 'IV30 (annualized)', tickformat: '.0%' },
    }, {displayModeBar: false, responsive: true});

    // Equity curve + drawdown overlay
    const bc = a.backtest_curve;
    Plotly.newPlot('chart-equity', [
      { x: bc.dates, y: bc.nav, type: 'scatter', mode: 'lines', name: 'NAV ($)',
        line: {color: '#3fb950', width: 1.8}, yaxis: 'y' },
      { x: bc.dates, y: bc.drawdown_pct, type: 'scatter', mode: 'lines', name: 'Drawdown %',
        line: {color: '#f85149', width: 1.2}, fill: 'tozeroy', fillcolor: 'rgba(248,81,73,0.15)',
        yaxis: 'y2' },
    ], {
      ...PLOTLY_LAYOUT_BASE,
      yaxis: { ...PLOTLY_LAYOUT_BASE.yaxis, title: 'NAV ($)', side: 'left' },
      yaxis2: { title: 'DD %', overlaying: 'y', side: 'right', gridcolor: 'transparent',
                range: [Math.min(...bc.drawdown_pct, 0) * 1.1 - 1, 0] },
      legend: { x: 0, y: 1.1, orientation: 'h' },
    }, {displayModeBar: false, responsive: true});

    // Walk-forward windows (bar chart, color by MDD severity)
    const wf = a.walkforward;
    const wfColors = wf.map(w => w.mdd < -15 ? '#f85149' : w.mdd < -10 ? '#d29922' : '#3fb950');
    Plotly.newPlot('chart-walkforward', [
      { x: wf.map(w => w.start), y: wf.map(w => w.ann), type: 'bar', name: 'Annualized %',
        marker: { color: wfColors }, yaxis: 'y' },
      { x: wf.map(w => w.start), y: wf.map(w => w.mdd), type: 'scatter', mode: 'lines+markers',
        name: 'MDD %', line: { color: '#bc8cff' }, marker: {size: 6}, yaxis: 'y2' },
    ], {
      ...PLOTLY_LAYOUT_BASE,
      yaxis: { ...PLOTLY_LAYOUT_BASE.yaxis, title: 'Annualized %' },
      yaxis2: { title: 'MDD %', overlaying: 'y', side: 'right', gridcolor: 'transparent',
                range: [Math.min(...wf.map(w => w.mdd), 0) * 1.1 - 1, 0] },
      legend: { x: 0, y: 1.1, orientation: 'h' },
    }, {displayModeBar: false, responsive: true});

    // Yearly P&L bars
    const yrs = a.yearly;
    const yrColors = yrs.map(y => y.pnl_pct > 0 ? '#3fb950' : '#f85149');
    const _ymax = Math.max(...yrs.map(y => +y.pnl_pct || 0), 0);
    const _ymin = Math.min(...yrs.map(y => +y.pnl_pct || 0), 0);
    const _ypad = Math.max((_ymax - _ymin) * 0.15, 5);
    Plotly.newPlot('chart-yearly', [
      { x: yrs.map(y => y.year), y: yrs.map(y => y.pnl_pct), type: 'bar',
        marker: { color: yrColors }, text: yrs.map(y => y.pnl_pct.toFixed(1) + '%'),
        textposition: 'outside', textfont: { color: '#e6edf3' }, cliponaxis: false },
    ], {
      ...PLOTLY_LAYOUT_BASE,
      // pad both ends: positive bars label above, negative bars label below
      yaxis: { ...PLOTLY_LAYOUT_BASE.yaxis, title: 'P&L %', range: [_ymin - _ypad, _ymax + _ypad] },
      xaxis: { ...PLOTLY_LAYOUT_BASE.xaxis, type: 'category' },
      margin: { ...PLOTLY_LAYOUT_BASE.margin, t: 20 },
    }, {displayModeBar: false, responsive: true});

    // Yearly table
    const tbody = document.querySelector('#yearly-table tbody');
    tbody.innerHTML = yrs.map(y => `<tr>
      <td>${y.year}</td>
      <td class="mono ${y.pnl_dollar>0?'positive':'negative'}">$${y.pnl_dollar.toLocaleString()}</td>
      <td class="mono ${y.pnl_pct>0?'positive':'negative'}">${y.pnl_pct.toFixed(1)}%</td>
      <td class="mono">${y.sharpe.toFixed(2)}</td>
      <td class="mono ${y.mdd_pct<-10?'negative':'neutral'}">${y.mdd_pct.toFixed(1)}%</td>
      <td>${y.days}</td>
    </tr>`).join('');
  } catch (e) {
    console.error('charts failed:', e);
  }
}
drawCharts();
setInterval(drawCharts, 600000); // 10 min
</script>
</body></html>"""


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == '/' or parsed.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML.encode('utf-8'))
        elif parsed.path == '/api/state':
            # Support ?kernel=KEY param to swap kernel without restart
            params = urllib.parse.parse_qs(parsed.query or '')
            kernel_key = params.get('kernel', [None])[0]
            if kernel_key:
                # Re-compute verdict with requested kernel
                with _STATE_LOCK:
                    spot = _STATE_CACHE.get('spot') or 11.51
                    pos = _STATE_CACHE.get('positions', [])
                    nav_live = (_STATE_CACHE.get('balance') or {}).get('net_liquidation')
                    verdict = validated_verdict(spot, pos, nav=nav_live, kernel_key=kernel_key)
                    body = json.dumps({**_STATE_CACHE, 'verdict': verdict}, default=str)
            else:
                with _STATE_LOCK:
                    body = json.dumps(_STATE_CACHE, default=str)
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body.encode('utf-8'))
        elif parsed.path == '/api/live':
            # SINGLE SOURCE OF TRUTH: orders straight from the champion engine on
            # the real positions — justified, + theta stream + Z models. No adapter.
            try:
                # serve the warmed cache (engine run is ~40s); compute on miss
                if _LIVE_CACHE['data'] is None or (time.time() - _LIVE_CACHE['ts']) > _LIVE_TTL:
                    _compute_live()
                body = json.dumps(_LIVE_CACHE['data'], default=str)
            except Exception as e:
                body = json.dumps({'error': repr(e)})
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body.encode('utf-8'))
        elif parsed.path == '/api/refresh':
            _refresh()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"refreshed": true}')
        elif parsed.path == '/api/analytics':
            try:
                body = json.dumps(_cached_analytics(), default=str)
            except Exception as e:
                body = json.dumps({'error': str(e)})
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Cache-Control', 'public, max-age=600')
            self.end_headers()
            self.wfile.write(body.encode('utf-8'))
        elif parsed.path == '/api/spy_vega':
            try:
                body = json.dumps(spy_vega_signal(), default=str)
            except Exception as e:
                body = json.dumps({'error': repr(e)})
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body.encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()


class ReusableTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def _spy_vega_warm_loop():
    """Pre-compute the SPY vega alert off the request path (the live IBKR snapshot can take ~8s) so
    the /api/spy_vega handler always serves a warm cache and never blocks the dashboard."""
    while True:
        try:
            spy_vega_signal(force=True)
        except Exception:
            pass
        time.sleep(540)


def _analytics_warm_loop():
    """Keep the heavy backtest analytics (curve / walk-forward / yearly / series) warm OFF the request
    path so /api/analytics is always instant and the lower charts always have data."""
    while True:
        try:
            _compute_analytics()
        except Exception:
            pass
        time.sleep(540)


def main():
    port = int(os.environ.get('KERNEL_DASH_PORT', '10001'))
    print('[kernel-dash] Bootstrapping state from WS / PG...')
    _refresh()
    threading.Thread(target=_refresh_loop, daemon=True).start()
    threading.Thread(target=_live_warm_loop, daemon=True).start()  # warm the SOT cache
    threading.Thread(target=_spy_vega_warm_loop, daemon=True).start()  # warm the SPY vega alert
    threading.Thread(target=_analytics_warm_loop, daemon=True).start()  # warm the backtest charts
    server = ReusableTCPServer(('0.0.0.0', port), Handler)
    print(f'[kernel-dash] Serving on http://localhost:{port}  (kernel: {CHAMPION_NAME})')
    print('[kernel-dash] Production dashboard untouched at :9999')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\n[kernel-dash] shutting down')
        server.server_close()


if __name__ == '__main__':
    main()
