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


# Engine playbook: what the engine WOULD do at each regime
ENGINE_PLAYBOOK = {
    'EXTREME_CHEAP': {
        'thesis': 'NG fundamentals very bullish — supply tight, mean-reversion to higher prices',
        'actions': [
            'Sell ATM puts aggressively (5+ contracts, 30 DTE)',
            'Hold full UNG share position',
            'Sell modest 5%+ OTM covered calls',
            'Skip BOXX deployment — capital tied up in wheel',
            'Don\'t buy KOLD or UNG puts',
        ],
        'delta_target': '100% NAV-equivalent long',
        'income_pace': 'Maximum — collect every premium opportunity',
    },
    'CHEAP': {
        'thesis': 'Still bullish but less extreme — wheel grinding',
        'actions': [
            'Sell ATM/5%OTM puts (3-5 contracts)',
            'Hold UNG shares',
            'Standard covered calls 5% OTM',
            'Small BOXX deployment for excess cash',
        ],
        'delta_target': '80-90% NAV long',
        'income_pace': 'Strong',
    },
    'NEUTRAL': {
        'thesis': 'No clear directional edge — collect premium, manage risk',
        'actions': [
            'Sell 5-10% OTM puts (2-4 contracts)',
            'Hold core shares',
            'OTM covered calls 3-5% above',
            'Build BOXX position with excess cash',
        ],
        'delta_target': '60-75% NAV long',
        'income_pace': 'Moderate',
    },
    'RICH': {
        'thesis': 'NG getting expensive — reduce long exposure via assignment',
        'actions': [
            'STOP new put writing (avoid catching falling knife)',
            'Sell ATM/slightly-ITM covered calls (force assignment)',
            'Let shares get called away — accumulate cash',
            'Increase BOXX position',
            'Consider small UNG long puts (defined risk)',
        ],
        'delta_target': '30-50% NAV long, transitioning',
        'income_pace': 'Reduced — defensive',
    },
    'EXTREME_RICH': {
        'thesis': 'Mean reversion likely — tactical bearish positioning',
        'actions': [
            'BUY 90-day 5%OTM UNG puts (3-5 contracts) — defined-risk short',
            'BUY small KOLD shares (3% NAV) — leveraged bearish',
            'NO new short put writing',
            'Continue selling ITM CCs on remaining shares',
            'Park most cash in BOXX',
            'Set exit triggers: close positions when z reverts to -0.3',
        ],
        'delta_target': '0-30% NAV long (mostly cash/hedges)',
        'income_pace': 'Minimal — playing for capital appreciation on reversion',
    },
}


def compute_regime_history(rows):
    """Compute per-day regime using the SAME _z_components as the badge.
    Both will always agree."""
    out = []
    target_weight = sum(Z_WEIGHTS.values())
    for r in rows:
        comps = _z_components(r)
        if not comps:
            out.append({'date': r.get('date'), 'z': 0, 'regime': 'NEUTRAL', 'ung': r.get('UNG')})
            continue
        weight_sum = sum(c['weight'] for c in comps)
        scale = target_weight / weight_sum if weight_sum > 0 else 1.0
        z_val = sum(c['z_contrib'] for c in comps) * scale
        if z_val > 1: reg = 'EXTREME_CHEAP'
        elif z_val > 0.5: reg = 'CHEAP'
        elif z_val > -0.5: reg = 'NEUTRAL'
        elif z_val > -1: reg = 'RICH'
        else: reg = 'EXTREME_RICH'
        out.append({'date': r.get('date'), 'z': round(z_val, 3), 'regime': reg, 'ung': r.get('UNG')})
    return out


def compute_regime_stats(regime_history):
    """Aggregate stats: time in each regime, transitions, avg outcome."""
    if not regime_history:
        return {}
    from collections import Counter
    regime_counts = Counter(r['regime'] for r in regime_history)
    total = len(regime_history)

    # Find transitions and what happened next
    transitions = []
    for i in range(1, len(regime_history)):
        prev_r = regime_history[i-1]['regime']
        curr_r = regime_history[i]['regime']
        if prev_r != curr_r:
            transitions.append({
                'date': regime_history[i]['date'],
                'from': prev_r,
                'to': curr_r,
                'ung_at_transition': regime_history[i].get('ung'),
            })

    # For each EXTREME_RICH/RICH start, find 30-day UNG return
    extreme_signals = []
    for i in range(len(regime_history) - 30):
        if regime_history[i]['regime'] == 'EXTREME_RICH' and (i == 0 or regime_history[i-1]['regime'] != 'EXTREME_RICH'):
            entry_ung = regime_history[i].get('ung')
            exit_ung = regime_history[i+30].get('ung')
            if entry_ung and exit_ung:
                extreme_signals.append({
                    'entry_date': regime_history[i]['date'],
                    'entry_ung': entry_ung,
                    'ung_30d_later': exit_ung,
                    'return_30d_pct': (exit_ung / entry_ung - 1) * 100,
                })

    return {
        'days_per_regime': dict(regime_counts),
        'pct_per_regime': {k: round(v / total * 100, 1) for k, v in regime_counts.items()},
        'n_transitions': len(transitions),
        'last_5_transitions': transitions[-5:] if transitions else [],
        'extreme_rich_signals': extreme_signals,
        'avg_extreme_rich_30d_return': (
            round(sum(s['return_30d_pct'] for s in extreme_signals) / len(extreme_signals), 1)
            if extreme_signals else None
        ),
    }


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


# UNIFIED Z-SCORE WEIGHTS (used by BOTH badge and chart)
Z_WEIGHTS = {
    'storage_level': 0.30,
    'days_supply': 0.25,
    'ng_trend': 0.20,
    'vix': 0.10,
    'oil_ng_ratio': 0.15,
}


def _z_components(row):
    """Compute z contributions for a single row. Returns list of dicts
    or empty list if essential data missing. Used by BOTH compute_z_factor_breakdown
    AND compute_regime_history so they always agree."""
    if not row:
        return []
    out = []

    storage = row.get('eia_storage_weekly')
    if storage:
        s_z = (storage - 2500) / 500
        out.append({
            'name': 'Storage Level', 'value': f"{storage:.0f} Bcf",
            'z_contrib': -s_z * Z_WEIGHTS['storage_level'],
            'weight': Z_WEIGHTS['storage_level'],
            'direction': 'bearish' if s_z > 0 else 'bullish',
        })

    ds = row.get('days_supply')
    if ds and ds > 0:
        ds_z = (ds - 31) / 5
        out.append({
            'name': 'Days of Supply', 'value': f"{ds:.1f} days",
            'z_contrib': -ds_z * Z_WEIGHTS['days_supply'],
            'weight': Z_WEIGHTS['days_supply'],
            'direction': 'bearish' if ds_z > 0 else 'bullish',
        })

    trend = row.get('ng_trend')
    if trend is not None:
        out.append({
            'name': 'NG Trend vs MA200', 'value': f"{trend*100:+.1f}%",
            'z_contrib': -trend * 3 * Z_WEIGHTS['ng_trend'],
            'weight': Z_WEIGHTS['ng_trend'],
            'direction': 'bearish' if trend > 0 else 'bullish',
        })

    vix = row.get('VIX')
    if vix:
        vix_norm = (vix - 20) / 10
        out.append({
            'name': 'VIX (market fear)', 'value': f"{vix:.1f}",
            'z_contrib': -vix_norm * Z_WEIGHTS['vix'],
            'weight': Z_WEIGHTS['vix'],
            'direction': 'bearish' if vix > 20 else 'bullish',
        })

    if row.get('CL') and row.get('NG') and row['NG'] > 0:
        ratio = row['CL'] / row['NG']
        r_z = (ratio - 25) / 10
        out.append({
            'name': 'Oil/NG Ratio', 'value': f"{ratio:.1f}",
            'z_contrib': r_z * Z_WEIGHTS['oil_ng_ratio'],
            'weight': Z_WEIGHTS['oil_ng_ratio'],
            'direction': 'bullish' if r_z > 0 else 'bearish',
        })

    return out


def compute_z_factor_breakdown(latest_row):
    """Return current z-score and per-factor contributions. Uses _z_components
    for consistency with compute_regime_history."""
    factors = _z_components(latest_row)
    # Composite: sum of contributions (weights already applied)
    # Normalize by weight coverage (so missing factors don't inflate/deflate)
    weight_sum = sum(f['weight'] for f in factors)
    target_weight = sum(Z_WEIGHTS.values())  # 1.00 by construction
    if weight_sum < target_weight and weight_sum > 0:
        scale = target_weight / weight_sum
    else:
        scale = 1.0
    z_total = sum(f['z_contrib'] for f in factors) * scale
    return {
        'z': z_total,
        'factors': factors,
        'weight_coverage_pct': round(weight_sum / target_weight * 100, 1),
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
    <h2>📜 Engine Playbook by Regime — What the engine does at each Z</h2>
    <div id="playbook"></div>
  </div>

  <div class="panel">
    <h2>🔄 Regime Statistics (Historical)</h2>
    <div id="regimeStats"></div>
  </div>

  <div class="panel">
    <h2>⚠️ EXTREME_RICH Signal Track Record</h2>
    <div id="extremeStats"></div>
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
  renderPlaybook(data.playbook, data.z_breakdown?.regime);
  renderRegimeStats(data.regime_stats);
  renderExtremeStats(data.regime_stats);
}

function renderPlaybook(pb, currentRegime) {
  if (!pb) return;
  let html = '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px;">';
  for (const [regime, data] of Object.entries(pb)) {
    const isCurrent = regime === currentRegime;
    const borderColor = isCurrent ? '#58a6ff' : 'transparent';
    html += `<div style="border:2px solid ${borderColor};border-radius:6px;padding:10px;background:rgba(0,0,0,0.2);">
      <div style="margin-bottom:6px;"><span class="regime-badge regime-${regime}">${regime}</span>
        ${isCurrent ? '<span style="color:var(--accent);font-size:0.7rem;margin-left:8px;">← CURRENT</span>' : ''}</div>
      <div style="color:var(--dim);font-size:0.78rem;margin-bottom:6px;font-style:italic;">${data.thesis}</div>
      <div style="font-size:0.8rem;">
        <strong style="color:var(--accent);">Delta target:</strong> ${data.delta_target}<br>
        <strong style="color:var(--accent);">Income pace:</strong> ${data.income_pace}<br>
        <strong style="color:var(--accent);">Actions:</strong>
        <ul style="margin:4px 0 0 18px;padding:0;color:var(--text);font-size:0.78rem;">
          ${data.actions.map(a => `<li>${a}</li>`).join('')}
        </ul>
      </div>
    </div>`;
  }
  html += '</div>';
  document.getElementById('playbook').innerHTML = html;
}

function renderRegimeStats(s) {
  if (!s || !s.pct_per_regime) {
    document.getElementById('regimeStats').innerHTML = '<div style="color:var(--dim);">Loading...</div>';
    return;
  }
  let html = '<table>';
  html += '<tr><th>Regime</th><th>Days</th><th>%</th></tr>';
  for (const [r, days] of Object.entries(s.days_per_regime || {})) {
    html += `<tr>
      <td><span class="regime-badge regime-${r}" style="font-size:0.65rem;padding:2px 6px;">${r}</span></td>
      <td>${days}</td>
      <td>${s.pct_per_regime[r]}%</td>
    </tr>`;
  }
  html += '</table>';
  html += `<div style="margin-top:10px;color:var(--dim);font-size:0.8rem;">
    ${s.n_transitions} regime transitions in dataset
  </div>`;
  if (s.last_5_transitions?.length) {
    html += '<div style="margin-top:8px;font-size:0.75rem;color:var(--dim);">Recent transitions:</div>';
    for (const t of s.last_5_transitions.slice(-3)) {
      html += `<div style="font-size:0.75rem;padding:2px 0;">${(t.date || '').slice(0,10)}: ${t.from} → ${t.to}</div>`;
    }
  }
  document.getElementById('regimeStats').innerHTML = html;
}

function renderExtremeStats(s) {
  if (!s || !s.extreme_rich_signals) return;
  const sigs = s.extreme_rich_signals;
  if (!sigs.length) {
    document.getElementById('extremeStats').innerHTML = '<div style="color:var(--dim);">No EXTREME_RICH signals fired in available history.</div>';
    return;
  }
  let html = `<div style="font-size:0.85rem;margin-bottom:8px;">
    <strong>Avg 30d UNG return after EXTREME_RICH signal:
      <span class="${s.avg_extreme_rich_30d_return < 0 ? 'gain' : 'loss'}">${s.avg_extreme_rich_30d_return}%</span></strong>
    (negative = bearish thesis worked; we'd profit on UNG puts/KOLD longs)
  </div>`;
  html += '<table><tr><th>Entry Date</th><th>Entry UNG</th><th>30d Later</th><th>30d Return</th></tr>';
  for (const sig of sigs.slice(-10)) {
    const retClass = sig.return_30d_pct < 0 ? 'gain' : 'loss';  // bearish thesis: negative = win
    html += `<tr>
      <td>${(sig.entry_date || '').slice(0,10)}</td>
      <td>$${sig.entry_ung?.toFixed(2)}</td>
      <td>$${sig.ung_30d_later?.toFixed(2)}</td>
      <td class="${retClass}">${sig.return_30d_pct?.toFixed(1)}%</td>
    </tr>`;
  }
  html += '</table>';
  document.getElementById('extremeStats').innerHTML = html;
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
  // Use the regime_history from the server (already computed with all factors)
  // Fall back to client-side if missing
  fetch('/api/research').then(r => r.json()).then(d => {
    const rh = d.regime_history || [];
    if (!rh.length) return;
    const dates = rh.map(r => r.date);
    const zs = rh.map(r => r.z);
    Plotly.react('zChart', [
      { x: dates, y: zs, type: 'scatter', mode: 'lines', name: 'Composite Z',
        line: { color: '#58a6ff' } },
    ], {
      ...LAYOUT,
      shapes: [
        { type: 'rect', x0: dates[0], x1: dates[dates.length-1], y0: 1, y1: 3, fillcolor: 'rgba(63,185,80,0.10)', line: { width: 0 } },
        { type: 'rect', x0: dates[0], x1: dates[dates.length-1], y0: 0.5, y1: 1, fillcolor: 'rgba(63,185,80,0.05)', line: { width: 0 } },
        { type: 'rect', x0: dates[0], x1: dates[dates.length-1], y0: -0.5, y1: -1, fillcolor: 'rgba(210,153,34,0.05)', line: { width: 0 } },
        { type: 'rect', x0: dates[0], x1: dates[dates.length-1], y0: -1, y1: -3, fillcolor: 'rgba(248,81,73,0.10)', line: { width: 0 } },
        { type: 'line', x0: dates[0], x1: dates[dates.length-1], y0: 1, y1: 1, line: { color: '#3fb950', dash: 'dash', width: 1 } },
        { type: 'line', x0: dates[0], x1: dates[dates.length-1], y0: -1, y1: -1, line: { color: '#f85149', dash: 'dash', width: 1 } },
        { type: 'line', x0: dates[0], x1: dates[dates.length-1], y0: 0, y1: 0, line: { color: '#8b949e', dash: 'dot', width: 1 } },
      ],
      annotations: [
        { x: dates[Math.floor(dates.length*0.05)], y: 1.5, text: 'EXTREME CHEAP', showarrow: false, font: { color: '#3fb950', size: 10 } },
        { x: dates[Math.floor(dates.length*0.05)], y: -1.5, text: 'EXTREME RICH', showarrow: false, font: { color: '#f85149', size: 10 } },
      ],
      yaxis: { ...LAYOUT.yaxis, title: 'Composite Z-Score' },
    }, { responsive: true, displayModeBar: false });
  });
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
            regime_history = compute_regime_history(rows)
            regime_stats = compute_regime_stats(regime_history)
            sum_path = os.path.join(RESULTS_DIR, 'summary.json')
            strategy_summary = {}
            if os.path.exists(sum_path):
                with open(sum_path) as f:
                    strategy_summary = json.load(f)
            # Trim history to essentials to reduce payload
            slim_history = [{
                'date': r.get('date'),
                'UNG': r.get('UNG'),
                'KOLD': r.get('KOLD'),
                'BOIL': r.get('BOIL'),
                'NG': r.get('NG'),
                'VIX': r.get('VIX'),
                'iv_30d': r.get('iv_30d'),
                'iv_60d': r.get('iv_60d'),
                'iv_90d': r.get('iv_90d'),
                'eia_storage_weekly': r.get('eia_storage_weekly'),
                'days_supply': r.get('days_supply'),
            } for r in rows]
            # Data freshness per source
            from datetime import date as _date_t
            today_d = _date_t.today()
            freshness = {}
            for col in ['UNG', 'NG', 'VIX', 'KOLD', 'BOIL', 'CL',
                        'eia_storage_weekly', 'eia_consumption', 'eia_production',
                        'eia_lng_exports', 'days_supply', 'iv_30d']:
                last_date = None
                last_value = None
                for r in reversed(rows):
                    if r.get(col) is not None:
                        last_date = r['date'][:10] if r.get('date') else None
                        last_value = r[col]
                        break
                if last_date:
                    try:
                        d = datetime.strptime(last_date, '%Y-%m-%d').date()
                        age = (today_d - d).days
                    except Exception:
                        age = None
                    freshness[col] = {'last_date': last_date, 'age_days': age, 'last_value': last_value}
            return self._send_json({
                'latest': latest,
                'z_breakdown': z_break,
                'insights': insights,
                'history': slim_history,
                'regime_history': regime_history,
                'regime_stats': regime_stats,
                'playbook': ENGINE_PLAYBOOK,
                'strategy_summary': strategy_summary,
                'data_freshness': freshness,
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
            # Sanitize: replace NaN/Infinity with None recursively (JSON-valid)
            def clean(obj):
                if isinstance(obj, dict):
                    return {k: clean(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [clean(v) for v in obj]
                if isinstance(obj, float):
                    if math.isnan(obj) or math.isinf(obj):
                        return None
                    return obj
                return obj
            cleaned = clean(data)
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            # allow_nan=False forces it to fail loudly rather than emit NaN
            payload = json.dumps(cleaned, default=str, allow_nan=False)
            self.wfile.write(payload.encode('utf-8'))
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *a):
        pass


def main():
    PORT = 9998
    print(f"NG Research Dashboard at http://localhost:{PORT}")
    ThreadingHTTPServer(('0.0.0.0', PORT), Handler).serve_forever()


if __name__ == '__main__':
    main()
