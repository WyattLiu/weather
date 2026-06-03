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
                    spot = (bid + ask) / 2 if (bid > 0 and ask > 0) else last
                    if not spot or spot <= 0:
                        spot = float(raw_q.get('price') or 0) or 11.51
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


def _build_walkforward():
    """Run champion_target_25_dd_trim on rolling 12-month windows."""
    import pandas as pd
    import math
    from replay_engine import run_strategy_simple, STRATEGIES, precompute_factor_z  # type: ignore
    csv = os.path.join(THIS_DIR, 'backtest', 'cache', 'master_dataset.csv')
    df = pd.read_csv(csv, index_col=0, parse_dates=True)
    df = precompute_factor_z(df).dropna(subset=['UNG'])
    strat = STRATEGIES['champion_target_25_dd_trim']
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
    strat = STRATEGIES['champion_target_25_dd_trim']
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
    strat = STRATEGIES['champion_target_25_dd_trim']
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


def _cached_analytics():
    """Refresh analytics every 10 minutes (heavy compute)."""
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


# ─── HTML (matches production CSS variables and layout) ──────────────────────
HTML = r"""<!doctype html>
<html><head>
<meta charset="utf-8"/>
<title>UNG Kernel Dashboard</title>
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
</style>
</head><body>
<div class="container">
  <h1>UNG Kernel Dashboard <span class="tag" id="kernel-tag"></span></h1>
  <div class="sub">
    Validated by walk-forward backtest · <a href="http://localhost:9999">Production dashboard</a> ·
    Refresh: <span id="freshness">–</span> ·
    Kernel: <span id="kernel-fullname">–</span>
  </div>

  <div id="error-row"></div>

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

  <!-- ACTION PLAN: timed concrete recommendations -->
  <div class="section" style="border-color:var(--blue); border-width:2px">
    <h2 style="color:var(--blue)">📋 Action Plan — what to do, when, at what price</h2>
    <div class="rec-list" id="recs">–</div>
    <div id="warnings"></div>
  </div>

  <!-- Deep beam analysis -->
  <div class="section">
    <h2>🎯 Deep Beam Analysis — why this strike?</h2>
    <div id="beam-content">–</div>
    <div class="rec-why" style="margin-top:8px">
      Each candidate strike scored as <strong>income − P(ITM) × expected_loss</strong> under BSM measure with real PG IV.
      The winner is the best risk-adjusted premium per contract.
    </div>
  </div>

  <!-- 30-day put expiration calendar -->
  <div class="section">
    <h2>🗓 Next 45 days — put expiration calendar</h2>
    <table id="expiry-calendar">
      <thead><tr>
        <th>Expiry</th><th>DTE</th><th>Strike</th><th>Qty</th>
        <th>Collateral</th><th>Outcome</th><th>$ freed</th>
      </tr></thead>
      <tbody></tbody>
    </table>
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
    <table>
      <thead><tr>
        <th>Type</th><th>Qty</th><th>Strike</th><th>Expiry</th>
        <th>Market Value</th><th>Unrealized P&amp;L</th>
      </tr></thead>
      <tbody id="positions">–</tbody>
    </table>
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
    const r = await fetch('/api/state');
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

    // Recommendations — fully actionable
    const recs = (v.recommendations || []).map(r => {
      const p = (r.priority || 'l').charAt(0);
      const whenLine = r.when ? `<div class="badge" style="background:var(--border);color:var(--text-dim);padding:2px 6px;border-radius:4px;font-size:0.7rem;display:inline-block;margin-left:8px">${r.when}</div>` : '';
      let ladderHtml = '';
      if (r.order_draft && r.order_draft.ladder) {
        const ladder = r.order_draft.ladder.map(l => `<tr><td style="text-align:right" class="mono">${l.qty}</td><td>@</td><td class="mono">$${l.price}</td></tr>`).join('');
        ladderHtml = `<div style="margin-top:8px;padding:8px;background:var(--bg);border-radius:4px"><div style="font-size:0.75rem;color:var(--text-dim);margin-bottom:4px">📝 Order ladder (split fills):</div><table style="font-size:0.85rem">${ladder}</table>${r.est_cost_dollar?'<div style="margin-top:4px;font-size:0.75rem;color:var(--text-dim)">est cost: $'+fmt(r.est_cost_dollar,0)+'</div>':''}</div>`;
      }
      return `<div class="rec rec-${p}">
        <div class="rec-action">${r.action}<span class="priority priority-${p}">${r.priority||'–'}</span>${whenLine}</div>
        <div class="rec-why">${r.why || ''}</div>
        ${ladderHtml}
      </div>`;
    }).join('');
    $('recs').innerHTML = recs || '<div class="rec-why">No active recommendations</div>';

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
    $('error-row').innerHTML = `<div class="error">fetch failed: ${e.message}</div>`;
  }
}
refresh();
setInterval(refresh, 30000);

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
                range: [-3, 3] },
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
                range: [-25, 0] },
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
                range: [-30, 0] },
      legend: { x: 0, y: 1.1, orientation: 'h' },
    }, {displayModeBar: false, responsive: true});

    // Yearly P&L bars
    const yrs = a.yearly;
    const yrColors = yrs.map(y => y.pnl_pct > 0 ? '#3fb950' : '#f85149');
    Plotly.newPlot('chart-yearly', [
      { x: yrs.map(y => y.year), y: yrs.map(y => y.pnl_pct), type: 'bar',
        marker: { color: yrColors }, text: yrs.map(y => y.pnl_pct.toFixed(1) + '%'),
        textposition: 'outside', textfont: { color: '#e6edf3' } },
    ], {
      ...PLOTLY_LAYOUT_BASE,
      yaxis: { ...PLOTLY_LAYOUT_BASE.yaxis, title: 'P&L %' },
      xaxis: { ...PLOTLY_LAYOUT_BASE.xaxis, type: 'category' },
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
            with _STATE_LOCK:
                body = json.dumps(_STATE_CACHE, default=str)
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
        else:
            self.send_response(404)
            self.end_headers()


class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


def main():
    port = int(os.environ.get('KERNEL_DASH_PORT', '10001'))
    print(f'[kernel-dash] Bootstrapping state from WS / PG...')
    _refresh()
    threading.Thread(target=_refresh_loop, daemon=True).start()
    server = ReusableTCPServer(('0.0.0.0', port), Handler)
    print(f'[kernel-dash] Serving on http://localhost:{port}  (kernel: {CHAMPION_NAME})')
    print(f'[kernel-dash] Production dashboard untouched at :9999')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\n[kernel-dash] shutting down')
        server.server_close()


if __name__ == '__main__':
    main()
