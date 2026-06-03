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
            _STATE_CACHE['verdict'] = validated_verdict(spot, _STATE_CACHE['positions'])
            _STATE_CACHE['last_refresh'] = time.time()
        except Exception as e:
            _STATE_CACHE['error'] = f'refresh failed: {e}'


def _refresh_loop():
    while True:
        _refresh()
        time.sleep(REFRESH_SEC)


# ─── HTML (matches production CSS variables and layout) ──────────────────────
HTML = r"""<!doctype html>
<html><head>
<meta charset="utf-8"/>
<title>UNG Kernel Dashboard</title>
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

  <!-- Recommendations -->
  <div class="section">
    <h2>Validated Kernel Recommendations</h2>
    <div class="rec-list" id="recs">–</div>
    <div id="warnings"></div>
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

    // Recommendations
    const recs = (v.recommendations || []).map(r => {
      const p = (r.priority || 'l').charAt(0);
      return `<div class="rec rec-${p}">
        <div class="rec-action">${r.action}<span class="priority priority-${p}">${r.priority||'–'}</span></div>
        <div class="rec-why">${r.why || ''}</div>
      </div>`;
    }).join('');
    $('recs').innerHTML = recs || '<div class="rec-why">No active recommendations</div>';

    // Warnings
    $('warnings').innerHTML = (v.warnings || []).map(w => `<div class="warning">⚠ ${w}</div>`).join('');

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
