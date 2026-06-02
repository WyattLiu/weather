"""Render feature_tracker.json as a self-contained HTML dashboard.

Usage: python render_tracker.py
Writes: results/feature_tracker.html
"""
import json
import os
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'feature_tracker.json')
OUT = os.path.join(HERE, 'results', 'feature_tracker.html')


STATUS_COLORS = {
    'not_started': '#666',
    'in_progress': '#d29922',
    'tested': '#58a6ff',
    'ported': '#3fb950',
    'negative_result': '#bc8cff',
    'blocked': '#f85149',
}

PRIORITY_BADGE = {
    'high': '🔴', 'medium': '🟡', 'low': '🟢',
}


def render():
    with open(SRC) as f:
        d = json.load(f)

    best = d.get('current_best_kernel', {})
    items = d.get('items', [])
    ported = d.get('ported_features', [])

    counts = {'not_started': 0, 'in_progress': 0, 'ported': 0,
              'tested': 0, 'negative_result': 0, 'blocked': 0}
    for it in items:
        counts[it.get('status', 'not_started')] += 1

    items_html = []
    for it in items:
        status = it.get('status', 'not_started')
        color = STATUS_COLORS.get(status, '#666')
        pri = PRIORITY_BADGE.get(it.get('priority', 'medium'), '⚪')
        tr = it.get('test_result') or {}
        test_block = ''
        if tr:
            sh_delta = tr.get('sharpe_delta', 0)
            ret_delta = tr.get('return_delta_pct', 0)
            test_block = f"""
            <div class="test-result">
              <strong>Test:</strong>
              ret Δ {ret_delta:+.1f}pp ·
              Sharpe Δ {sh_delta:+.2f} ·
              {tr.get('verdict', '?')}
            </div>"""
        items_html.append(f"""
        <div class="card" style="border-left: 4px solid {color}">
          <div class="card-head">
            <span class="pri">{pri}</span>
            <h3>{it['id']}. {it['name']}</h3>
            <span class="status" style="background:{color}">{status.replace('_',' ')}</span>
          </div>
          <p class="desc">{it['description']}</p>
          <div class="meta">
            effort: <strong>{it.get('estimated_effort', '?')}</strong> ·
            impact: <em>{it.get('expected_impact', '?')}</em>
          </div>
          {test_block}
        </div>""")

    ported_html = ''.join(f'<span class="pill">{p}</span>' for p in ported)

    html = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"><title>Production-Feature Port Tracker</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif;
         background: #0d1117; color: #e6edf3; padding: 20px;
         line-height: 1.5; max-width: 1100px; margin: 0 auto; }}
  h1 {{ color: #58a6ff; margin: 0 0 8px; }}
  h3 {{ margin: 0; flex: 1; }}
  .subtitle {{ color: #8b949e; margin-bottom: 16px; }}
  .summary {{ background: #161b22; padding: 20px; border-radius: 8px;
              margin-bottom: 20px; border: 1px solid #30363d; }}
  .summary h2 {{ margin: 0 0 12px; color: #3fb950; }}
  .summary .row {{ display: flex; gap: 24px; flex-wrap: wrap; }}
  .summary .stat {{ font-size: 1.5rem; font-weight: 600; }}
  .summary .stat-label {{ color: #8b949e; font-size: 0.8rem; }}
  .progress-row {{ display: flex; gap: 8px; margin-top: 10px; }}
  .progress-row > div {{ flex: 1; text-align: center; padding: 8px 4px;
                        border-radius: 6px; background: #21262d; font-size: 0.85rem; }}
  .pill {{ display: inline-block; background: #21262d; color: #8b949e;
           padding: 2px 8px; margin: 2px; border-radius: 12px;
           font-size: 0.75rem; }}
  .card {{ background: #161b22; padding: 16px; margin: 12px 0;
           border-radius: 8px; border: 1px solid #30363d; }}
  .card-head {{ display: flex; align-items: center; gap: 12px; }}
  .pri {{ font-size: 1.2rem; }}
  .status {{ color: white; padding: 4px 12px; border-radius: 4px;
             font-size: 0.75rem; text-transform: uppercase; font-weight: 600; }}
  .desc {{ color: #c9d1d9; margin: 8px 0 4px; }}
  .meta {{ color: #8b949e; font-size: 0.85rem; }}
  .test-result {{ background: #0d1117; border-radius: 4px; padding: 8px 12px;
                  margin-top: 8px; font-size: 0.9rem; }}
  footer {{ color: #8b949e; text-align: center; margin-top: 30px;
            font-size: 0.85rem; }}
</style></head>
<body>
  <h1>🏗️ Production-Feature Port Tracker</h1>
  <p class="subtitle">Absorbing production engine features into the backtest, one at a time.</p>

  <div class="summary">
    <h2>🏆 Current Best Kernel: {best.get('name', '?')}</h2>
    <div class="row">
      <div><div class="stat-label">Full return (5yr)</div>
           <div class="stat" style="color:#3fb950">{best.get('full_return_pct', 0):+.1f}%</div></div>
      <div><div class="stat-label">Sharpe</div>
           <div class="stat" style="color:#58a6ff">{best.get('sharpe', 0):+.2f}</div></div>
      <div><div class="stat-label">Max drawdown</div>
           <div class="stat" style="color:#f85149">{best.get('max_dd_pct', 0):+.1f}%</div></div>
      <div><div class="stat-label">Calm regime (3yr)</div>
           <div class="stat" style="color:#bc8cff">{best.get('calm_return_pct', 0):+.1f}%</div></div>
      <div><div class="stat-label">Calm annualized</div>
           <div class="stat" style="color:#d29922">{best.get('calm_annual_pct', 0):+.1f}%</div></div>
    </div>
    <div class="progress-row">
      <div>not started: <strong>{counts['not_started']}</strong></div>
      <div style="color:#d29922">in progress: <strong>{counts['in_progress']}</strong></div>
      <div style="color:#58a6ff">tested: <strong>{counts['tested']}</strong></div>
      <div style="color:#3fb950">ported: <strong>{counts['ported']}</strong></div>
      <div style="color:#bc8cff">neg result: <strong>{counts['negative_result']}</strong></div>
    </div>
  </div>

  <h2>📋 Items to Port (7)</h2>
  {''.join(items_html)}

  <h2>✅ Already Ported ({len(ported)})</h2>
  <div>{ported_html}</div>

  <footer>Updated {d.get('updated', '')} · {datetime.now().strftime('%Y-%m-%d %H:%M')} ·
  Generated from feature_tracker.json</footer>
</body></html>"""

    with open(OUT, 'w') as f:
        f.write(html)
    print(f"Wrote: {OUT}")
    return OUT


if __name__ == '__main__':
    render()
