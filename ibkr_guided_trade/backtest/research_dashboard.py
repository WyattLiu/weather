"""NG Research Dashboard — port 9998, separate from live trading.

Beautiful study tool for natural gas markets and the wheel engine.

Panels:
  1. Market State Now — prices, IV, regime, z-score breakdown
  2. Storage & Supply — EIA data, days_supply, deviation vs 5-yr avg
  3. Z-Score Timeline — composite + factor contributions over time
  4. Regime Transitions — when did regime change, time-in-regime stats
  5. UNG vs KOLD vs NG — relative behavior, correlation, decay
  6. IV Surface — realized vs implied, percentiles
  7. Engine Insights — auto-generated observations + opportunity ideas
  8. Position What-If — quick simulation of different position sizes

Run: python backtest/research_dashboard.py
"""
import os
import sys
import json
import math
import csv
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from datetime import datetime

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


def load_master_dataset():
    """Load master dataset CSV, return as list of dicts."""
    path = os.path.join(CACHE_DIR, 'master_dataset.csv')
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            new_row = {}
            for k, v in row.items():
                if k in ('Date', '', 'index'):
                    new_row['date'] = v
                else:
                    try:
                        new_row[k] = float(v) if v not in ('', 'nan') else None
                    except ValueError:
                        new_row[k] = None
            rows.append(new_row)
    return rows


def compute_insights(rows):
    """Generate research observations from current data."""
    if not rows:
        return []
    recent = rows[-30:]  # last 30 days
    insights = []
    latest = rows[-1]

    # Price changes
    if latest.get('UNG') and rows[-22].get('UNG'):
        m_change = (latest['UNG'] / rows[-22]['UNG'] - 1) * 100
        w_change = (latest['UNG'] / rows[-6]['UNG'] - 1) * 100
        if abs(w_change) > 5:
            insights.append({
                'type': 'price_move',
                'severity': 'high' if abs(w_change) > 10 else 'med',
                'text': f"UNG moved {w_change:+.1f}% this week ({m_change:+.1f}% this month). "
                        f"Large moves often produce mean-reversion opportunities within 2-3 weeks."
            })

    # Storage trend
    storage_recent = [r.get('eia_storage_weekly') for r in rows[-90:] if r.get('eia_storage_weekly')]
    if len(storage_recent) > 4:
        recent_chg = storage_recent[-1] - storage_recent[-4]
        if abs(recent_chg) > 200:
            direction = 'BUILD' if recent_chg > 0 else 'DRAW'
            insights.append({
                'type': 'storage',
                'severity': 'med',
                'text': f"Storage {direction}: {recent_chg:+.0f} Bcf over past 4 weeks. "
                        f"Sustained {direction.lower()}s alter NG fair value by ~$0.30/MMBtu per 500 Bcf."
            })

    # Days of supply
    days_supply = latest.get('days_supply')
    if days_supply:
        if days_supply < 25:
            insights.append({
                'type': 'supply_tight',
                'severity': 'high',
                'text': f"Days of supply at {days_supply:.1f} (vs 5yr avg ~31). Bullish — "
                        f"historically when DOS drops below 25, NG rallies 15-30% in following quarter."
            })
        elif days_supply > 40:
            insights.append({
                'type': 'supply_loose',
                'severity': 'high',
                'text': f"Days of supply at {days_supply:.1f} (vs 5yr avg ~31). Bearish — "
                        f"oversupply tends to persist; consider reducing long exposure."
            })

    # IV regime
    iv_30 = latest.get('iv_30d')
    if iv_30:
        if iv_30 > 0.80:
            insights.append({
                'type': 'iv_high',
                'severity': 'high',
                'text': f"30-day IV at {iv_30*100:.0f}% (very high). "
                        f"Premium-rich environment — selling puts captures excess vol. "
                        f"Historical mean: 60%, current 75th percentile threshold: ~76%."
            })
        elif iv_30 < 0.35:
            insights.append({
                'type': 'iv_low',
                'severity': 'med',
                'text': f"30-day IV at {iv_30*100:.0f}% (low). "
                        f"Premium-poor — wheel income suppressed. Consider waiting or smaller size."
            })

    # VIX context
    vix = latest.get('VIX')
    if vix:
        if vix > 25:
            insights.append({
                'type': 'vix',
                'severity': 'med',
                'text': f"VIX at {vix:.1f} (elevated). Cross-asset vol high — NG IV likely "
                        f"trades richer than implied; favorable for premium sellers."
            })

    # Correlation check
    if latest.get('NG') and latest.get('UNG'):
        ratio = latest['UNG'] / latest['NG']
        # Typical UNG = ~4-5x NG (UNG tracks 1-month NG futures × ~4x for share price)
        insights.append({
            'type': 'ratio',
            'severity': 'info',
            'text': f"UNG/NG ratio: {ratio:.2f}. UNG is a 1-month NG futures ETF — "
                    f"ratio drift indicates contango/backwardation roll cost."
        })

    return insights


def compute_z_factor_breakdown(latest_row):
    """Return current z-score and per-factor contributions."""
    if not latest_row:
        return {'z': 0, 'factors': []}
    factors = []

    # Storage z (proxy)
    storage = latest_row.get('eia_storage_weekly')
    if storage:
        # Quick z: how far from 2500 (rough mean)?
        s_z = (storage - 2500) / 500
        factors.append({
            'name': 'Storage Level',
            'value': f"{storage:.0f} Bcf",
            'z_contrib': -s_z * 0.30,  # high storage = bearish, weight 0.30
            'weight': 0.30,
            'direction': 'bearish' if s_z > 0 else 'bullish',
        })

    # Days supply
    ds = latest_row.get('days_supply')
    if ds:
        ds_z = (ds - 31) / 5
        factors.append({
            'name': 'Days of Supply',
            'value': f"{ds:.1f} days",
            'z_contrib': -ds_z * 0.25,
            'weight': 0.25,
            'direction': 'bearish' if ds_z > 0 else 'bullish',
        })

    # NG trend
    trend = latest_row.get('ng_trend')
    if trend is not None:
        factors.append({
            'name': 'NG Trend vs MA200',
            'value': f"{trend*100:+.1f}%",
            'z_contrib': -trend * 3 * 0.20,
            'weight': 0.20,
            'direction': 'bearish' if trend > 0 else 'bullish',
        })

    # VIX
    vix = latest_row.get('VIX')
    if vix:
        vix_norm = (vix - 20) / 10
        factors.append({
            'name': 'VIX (market fear)',
            'value': f"{vix:.1f}",
            'z_contrib': -vix_norm * 0.10,
            'weight': 0.10,
            'direction': 'bearish' if vix > 20 else 'bullish',
        })

    # Oil/NG ratio
    if latest_row.get('CL') and latest_row.get('NG'):
        ratio = latest_row['CL'] / latest_row['NG']
        r_z = (ratio - 25) / 10
        factors.append({
            'name': 'Oil/NG Ratio',
            'value': f"{ratio:.1f}",
            'z_contrib': r_z * 0.15,
            'weight': 0.15,
            'direction': 'bullish' if r_z > 0 else 'bearish',
        })

    # Composite
    z_total = sum(f['z_contrib'] for f in factors)
    return {
        'z': z_total,
        'factors': factors,
        'regime': (
            'EXTREME_CHEAP' if z_total > 1 else
            'CHEAP' if z_total > 0.5 else
            'NEUTRAL' if z_total > -0.5 else
            'RICH' if z_total > -1 else 'EXTREME_RICH'
        ),
    }


HTML = r"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NG Research Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
:root {
  --bg: #0d1117; --bg2: #161b22; --border: #30363d;
  --text: #c9d1d9; --dim: #8b949e;
  --accent: #58a6ff; --gain: #3fb950; --loss: #f85149;
  --warn: #d29922; --info: #79c0ff;
}
body { margin: 0; padding: 16px; background: var(--bg); color: var(--text);
       font: 14px/1.4 -apple-system, BlinkMacSystemFont, sans-serif; }
header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; }
h1 { margin: 0; font-size: 1.4rem; color: var(--accent); }
.updated { color: var(--dim); font-size: 0.75rem; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(380px, 1fr)); gap: 16px; }
.panel { background: var(--bg2); border: 1px solid var(--border); border-radius: 8px; padding: 16px; }
.panel h2 { margin: 0 0 12px; font-size: 0.85rem; color: var(--accent); text-transform: uppercase; letter-spacing: 0.06em; }
.kpi { display: grid; grid-template-columns: repeat(auto-fit, minmax(110px, 1fr)); gap: 8px; margin-bottom: 12px; }
.kpi .item { padding: 8px; background: var(--bg); border-radius: 4px; }
.kpi .label { color: var(--dim); font-size: 0.7rem; text-transform: uppercase; }
.kpi .value { font-size: 1.1rem; font-weight: 600; margin-top: 2px; }
.gain { color: var(--gain); }
.loss { color: var(--loss); }
.warn { color: var(--warn); }
.info { color: var(--info); }
.factor-row { display: grid; grid-template-columns: 1fr 80px 80px 100px; gap: 8px; padding: 6px 0; border-bottom: 1px solid var(--border); font-size: 0.85rem; }
.factor-row:last-child { border-bottom: 0; }
.regime-badge { display: inline-block; padding: 3px 10px; border-radius: 4px; font-weight: 600; font-size: 0.8rem; }
.regime-EXTREME_CHEAP { background: rgba(63,185,80,0.3); color: var(--gain); }
.regime-CHEAP { background: rgba(63,185,80,0.15); color: var(--gain); }
.regime-NEUTRAL { background: rgba(139,148,158,0.2); color: var(--dim); }
.regime-RICH { background: rgba(210,153,34,0.2); color: var(--warn); }
.regime-EXTREME_RICH { background: rgba(248,81,73,0.3); color: var(--loss); }
.insight { padding: 10px; margin-bottom: 8px; border-radius: 6px; border-left: 3px solid var(--info); background: rgba(121,192,255,0.05); }
.insight.high { border-left-color: var(--warn); background: rgba(210,153,34,0.05); }
.insight .type { font-size: 0.7rem; color: var(--dim); text-transform: uppercase; margin-bottom: 4px; }
.chart { height: 280px; }
.bigchart { height: 380px; }
table { width: 100%; font-size: 0.85rem; border-collapse: collapse; }
th, td { padding: 6px 8px; border-bottom: 1px solid var(--border); text-align: right; }
th { color: var(--dim); font-weight: 500; text-transform: uppercase; font-size: 0.7rem; }
th:first-child, td:first-child { text-align: left; }
</style></head><body>
<header>
  <h1>🔬 NG Research Dashboard</h1>
  <span class="updated" id="updated">—</span>
</header>

<div class="grid">
  <div class="panel">
    <h2>📊 Market Now</h2>
    <div class="kpi" id="kpiMarket"></div>
    <div id="regimeBadge"></div>
  </div>

  <div class="panel">
    <h2>🎯 Z-Score Factor Breakdown</h2>
    <div id="factorTable"></div>
  </div>

  <div class="panel" style="grid-column: span 2;">
    <h2>📈 Composite Z-Score Over Time</h2>
    <div class="bigchart" id="zChart"></div>
  </div>

  <div class="panel" style="grid-column: span 2;">
    <h2>⛽ Storage & Days of Supply</h2>
    <div class="bigchart" id="storageChart"></div>
  </div>

  <div class="panel">
    <h2>📉 UNG / KOLD / BOIL Relative</h2>
    <div class="chart" id="relativeChart"></div>
  </div>

  <div class="panel">
    <h2>📊 IV Surface</h2>
    <div class="chart" id="ivChart"></div>
  </div>

  <div class="panel" style="grid-column: span 2;">
    <h2>💡 Engine Insights & Opportunities</h2>
    <div id="insights"></div>
  </div>

  <div class="panel" style="grid-column: span 2;">
    <h2>🏁 Strategy Performance Comparison</h2>
    <div id="strategyTable"></div>
  </div>
</div>

<script>
const COLORS = { UNG: '#58a6ff', KOLD: '#f85149', BOIL: '#3fb950', NG: '#d29922', VIX: '#a5a5a5' };
const LAYOUT = {
  paper_bgcolor: '#161b22', plot_bgcolor: '#0d1117',
  font: { color: '#c9d1d9', size: 11 },
  xaxis: { gridcolor: '#30363d', zerolinecolor: '#30363d' },
  yaxis: { gridcolor: '#30363d', zerolinecolor: '#30363d' },
  margin: { l: 55, r: 30, t: 20, b: 40 },
  legend: { x: 0.02, y: 0.98, bgcolor: 'rgba(0,0,0,0)', font: { size: 10 } },
};

async function load() {
  const data = await fetch('/api/research').then(r => r.json());
  document.getElementById('updated').textContent = 'Updated ' + new Date(data.updated_at).toLocaleString();
  renderMarket(data);
  renderFactors(data.z_breakdown);
  renderZChart(data.history);
  renderStorage(data.history);
  renderRelative(data.history);
  renderIV(data.history);
  renderInsights(data.insights);
  renderStrategies(data.strategy_summary);
}

function renderMarket(d) {
  const m = d.latest || {};
  const kpis = document.getElementById('kpiMarket');
  const items = [
    ['UNG', m.UNG ? `$${m.UNG.toFixed(2)}` : '—'],
    ['NG (futures)', m.NG ? `$${m.NG.toFixed(2)}` : '—'],
    ['KOLD', m.KOLD ? `$${m.KOLD.toFixed(2)}` : '—'],
    ['BOIL', m.BOIL ? `$${m.BOIL.toFixed(2)}` : '—'],
    ['VIX', m.VIX ? m.VIX.toFixed(1) : '—'],
    ['IV 30d', m.iv_30d ? (m.iv_30d * 100).toFixed(0) + '%' : '—'],
    ['IV 60d', m.iv_60d ? (m.iv_60d * 100).toFixed(0) + '%' : '—'],
    ['Days Supply', m.days_supply ? m.days_supply.toFixed(1) : '—'],
  ];
  kpis.innerHTML = items.map(([l, v]) =>
    `<div class="item"><div class="label">${l}</div><div class="value">${v}</div></div>`).join('');

  const z = d.z_breakdown || {};
  document.getElementById('regimeBadge').innerHTML =
    `<div style="margin-top:8px;"><span class="regime-badge regime-${z.regime || 'NEUTRAL'}">${z.regime || 'NEUTRAL'}</span>
     <span style="margin-left:8px;color:var(--dim);">composite z = ${(z.z || 0).toFixed(2)}</span></div>`;
}

function renderFactors(z) {
  if (!z || !z.factors) return;
  let html = '<div class="factor-row" style="color:var(--dim);font-size:0.7rem;text-transform:uppercase;border-bottom:1px solid var(--border);"><div>Factor</div><div>Value</div><div>Weight</div><div>Z Contrib</div></div>';
  for (const f of z.factors) {
    const contribClass = f.z_contrib > 0 ? 'gain' : (f.z_contrib < 0 ? 'loss' : '');
    html += `<div class="factor-row">
      <div>${f.name}</div>
      <div>${f.value}</div>
      <div>${(f.weight * 100).toFixed(0)}%</div>
      <div class="${contribClass}">${f.z_contrib >= 0 ? '+' : ''}${f.z_contrib.toFixed(2)}</div>
    </div>`;
  }
  html += `<div class="factor-row" style="font-weight:600;border-top:2px solid var(--border);margin-top:6px;">
    <div>COMPOSITE Z</div><div></div><div></div>
    <div class="${z.z > 0 ? 'gain' : 'loss'}">${z.z >= 0 ? '+' : ''}${z.z.toFixed(2)}</div></div>`;
  document.getElementById('factorTable').innerHTML = html;
}

function renderZChart(hist) {
  if (!hist || hist.length === 0) return;
  // Compute synthetic z over time (price-based proxy)
  const dates = hist.map(h => h.date);
  const ung = hist.map(h => h.UNG);
  // 200-day MA
  const ma200 = ung.map((v, i) => {
    if (i < 200) return null;
    const window = ung.slice(i-200, i).filter(x => x);
    return window.reduce((a,b)=>a+b,0) / window.length;
  });
  const std200 = ung.map((v, i) => {
    if (i < 200 || !ma200[i]) return null;
    const window = ung.slice(i-200, i).filter(x => x);
    const mean = window.reduce((a,b)=>a+b,0) / window.length;
    return Math.sqrt(window.map(x => (x-mean)**2).reduce((a,b)=>a+b,0) / window.length);
  });
  const z = ung.map((v, i) => (v && ma200[i] && std200[i]) ? -((v - ma200[i]) / std200[i]) : null);

  Plotly.react('zChart', [
    { x: dates, y: z, type: 'scatter', mode: 'lines', name: 'Z-Score', line: { color: '#58a6ff' } },
  ], {
    ...LAYOUT,
    shapes: [
      { type: 'line', x0: dates[0], x1: dates[dates.length-1], y0: 1, y1: 1, line: { color: '#3fb950', dash: 'dash', width: 1 } },
      { type: 'line', x0: dates[0], x1: dates[dates.length-1], y0: -1, y1: -1, line: { color: '#f85149', dash: 'dash', width: 1 } },
      { type: 'line', x0: dates[0], x1: dates[dates.length-1], y0: 0, y1: 0, line: { color: '#8b949e', dash: 'dot', width: 1 } },
    ],
    annotations: [
      { x: dates[Math.floor(dates.length*0.05)], y: 1.1, text: 'CHEAP →', showarrow: false, font: { color: '#3fb950', size: 9 } },
      { x: dates[Math.floor(dates.length*0.05)], y: -1.1, text: '← RICH', showarrow: false, font: { color: '#f85149', size: 9 } },
    ],
    yaxis: { ...LAYOUT.yaxis, title: 'Z-Score' },
  }, { responsive: true, displayModeBar: false });
}

function renderStorage(hist) {
  if (!hist) return;
  const dates = hist.map(h => h.date);
  Plotly.react('storageChart', [
    { x: dates, y: hist.map(h => h.eia_storage_weekly), name: 'Storage (Bcf)', type: 'scatter', mode: 'lines',
      line: { color: '#58a6ff' }, yaxis: 'y' },
    { x: dates, y: hist.map(h => h.days_supply), name: 'Days of Supply', type: 'scatter', mode: 'lines',
      line: { color: '#d29922', dash: 'dot' }, yaxis: 'y2' },
  ], {
    ...LAYOUT,
    yaxis: { ...LAYOUT.yaxis, title: 'Storage (Bcf)', titlefont: { color: '#58a6ff' } },
    yaxis2: { title: 'Days of Supply', overlaying: 'y', side: 'right', titlefont: { color: '#d29922' } },
  }, { responsive: true, displayModeBar: false });
}

function renderRelative(hist) {
  if (!hist) return;
  const dates = hist.map(h => h.date);
  // Normalize all to 100 at start
  const norm = (key) => {
    const first = hist.find(h => h[key])?.[key];
    if (!first) return [];
    return hist.map(h => h[key] ? (h[key] / first * 100) : null);
  };
  Plotly.react('relativeChart', [
    { x: dates, y: norm('UNG'), name: 'UNG', type: 'scatter', mode: 'lines', line: { color: COLORS.UNG } },
    { x: dates, y: norm('KOLD'), name: 'KOLD', type: 'scatter', mode: 'lines', line: { color: COLORS.KOLD } },
    { x: dates, y: norm('BOIL'), name: 'BOIL', type: 'scatter', mode: 'lines', line: { color: COLORS.BOIL } },
    { x: dates, y: norm('NG'), name: 'NG futures', type: 'scatter', mode: 'lines', line: { color: COLORS.NG, dash: 'dot' } },
  ], { ...LAYOUT, yaxis: { ...LAYOUT.yaxis, title: 'Normalized (start=100, log)', type: 'log' } },
  { responsive: true, displayModeBar: false });
}

function renderIV(hist) {
  if (!hist) return;
  const dates = hist.map(h => h.date);
  Plotly.react('ivChart', [
    { x: dates, y: hist.map(h => h.iv_30d ? h.iv_30d * 100 : null), name: '30d IV', type: 'scatter', mode: 'lines', line: { color: '#58a6ff' } },
    { x: dates, y: hist.map(h => h.iv_60d ? h.iv_60d * 100 : null), name: '60d IV', type: 'scatter', mode: 'lines', line: { color: '#3fb950' } },
    { x: dates, y: hist.map(h => h.iv_90d ? h.iv_90d * 100 : null), name: '90d IV', type: 'scatter', mode: 'lines', line: { color: '#d29922' } },
    { x: dates, y: hist.map(h => h.VIX), name: 'VIX', type: 'scatter', mode: 'lines', line: { color: '#a5a5a5', dash: 'dot' }, yaxis: 'y2' },
  ], {
    ...LAYOUT,
    yaxis: { ...LAYOUT.yaxis, title: 'UNG Realized IV (%)' },
    yaxis2: { title: 'VIX', overlaying: 'y', side: 'right' },
  }, { responsive: true, displayModeBar: false });
}

function renderInsights(insights) {
  if (!insights) return;
  document.getElementById('insights').innerHTML = insights.map(i => `
    <div class="insight ${i.severity || ''}">
      <div class="type">${i.type}</div>
      <div>${i.text}</div>
    </div>
  `).join('') || '<div style="color:var(--dim);">No insights yet. Run replay_engine.py first.</div>';
}

function renderStrategies(s) {
  if (!s) {
    document.getElementById('strategyTable').innerHTML = '<div style="color:var(--dim);">Run replay_engine.py to populate.</div>';
    return;
  }
  let html = '<table><tr><th>Strategy</th><th>Final NAV</th><th>Return</th><th>Annual</th><th>Max DD</th><th>Sharpe</th></tr>';
  for (const [name, r] of Object.entries(s)) {
    if (typeof r !== 'object' || !r.final) continue;
    const retClass = r.return_pct >= 0 ? 'gain' : 'loss';
    html += `<tr>
      <td>${name}</td>
      <td>$${Math.round(r.final).toLocaleString()}</td>
      <td class="${retClass}">${r.return_pct?.toFixed(1)}%</td>
      <td>${r.annual_pct?.toFixed(1)}%</td>
      <td class="loss">${r.max_dd_pct?.toFixed(1)}%</td>
      <td>${r.sharpe?.toFixed(2)}</td>
    </tr>`;
  }
  document.getElementById('strategyTable').innerHTML = html;
}

load();
setInterval(load, 120000);
</script>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            return self._send(HTML, 'text/html')
        if self.path == '/api/research':
            rows = load_master_dataset()
            # Use most recent row with UNG data (skip null/weekend rows)
            latest = {}
            for r in reversed(rows):
                if r.get('UNG'):
                    latest = r
                    break
            z_break = compute_z_factor_breakdown(latest)
            insights = compute_insights(rows)
            # Strategy summary
            sum_path = os.path.join(RESULTS_DIR, 'summary.json')
            strategy_summary = {}
            if os.path.exists(sum_path):
                with open(sum_path) as f:
                    strategy_summary = json.load(f)
            return self._send_json({
                'latest': latest,
                'z_breakdown': z_break,
                'insights': insights,
                'history': rows,
                'strategy_summary': strategy_summary,
                'updated_at': datetime.now().isoformat(),
            })
        self.send_error(404)

    def _send(self, content, ctype):
        self.send_response(200)
        self.send_header('Content-Type', f'{ctype}; charset=utf-8')
        self.end_headers()
        self.wfile.write(content.encode('utf-8'))

    def _send_json(self, data):
        try:
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            def default(o):
                if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
                    return None
                return str(o)
            self.wfile.write(json.dumps(data, default=default).encode('utf-8'))
        except (BrokenPipeError, ConnectionResetError):
            pass  # client disconnected, no need to crash

    def log_message(self, *a):
        pass


def main():
    PORT = 9998
    print(f"NG Research Dashboard at http://localhost:{PORT}")
    ThreadingHTTPServer(('0.0.0.0', PORT), Handler).serve_forever()


if __name__ == '__main__':
    main()
