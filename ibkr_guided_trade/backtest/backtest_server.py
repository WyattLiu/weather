"""Standalone backtest visualization web server (port 9998).

Separate from the live trading engine (port 9999). Serves:
- Equity curves overlaid
- Regime distribution
- Position evolution (shares, cash, BOXX, KOLD over time)
- Trade log
- Strategy comparison table

Run: python backtest/backtest_server.py
"""
import os
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

HTML_PAGE = r"""<!DOCTYPE html>
<html><head><title>UNG Wheel Backtest</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
body { font-family: -apple-system, sans-serif; margin: 0; padding: 12px; background: #0d1117; color: #c9d1d9; }
h1 { margin: 0 0 12px; font-size: 1.2rem; color: #58a6ff; }
.summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 8px; margin-bottom: 16px; }
.card { background: #161b22; padding: 10px; border-radius: 6px; border: 1px solid #30363d; }
.card h3 { margin: 0 0 6px; font-size: 0.8rem; color: #8b949e; text-transform: uppercase; }
.card .val { font-size: 1.3rem; font-weight: 600; }
.gain { color: #3fb950; }
.loss { color: #f85149; }
table { width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 0.85rem; }
th, td { padding: 6px 8px; border-bottom: 1px solid #30363d; text-align: right; }
th:first-child, td:first-child { text-align: left; }
.chart { background: #161b22; border-radius: 6px; padding: 12px; margin-bottom: 16px; }
.update { color: #8b949e; font-size: 0.75rem; }
</style></head>
<body>
<h1>UNG Wheel Backtest Replay <span class="update" id="updated"></span></h1>

<div class="summary" id="summary"></div>

<div class="chart"><div id="equityChart" style="height:400px"></div></div>
<div class="chart"><div id="regimeChart" style="height:280px"></div></div>
<div class="chart"><div id="positionsChart" style="height:280px"></div></div>

<h3 style="margin-top:24px">Strategy Comparison</h3>
<table id="comparisonTable"></table>

<script>
const PLOT_LAYOUT = {
    paper_bgcolor: '#161b22', plot_bgcolor: '#0d1117',
    font: { color: '#c9d1d9', size: 11 },
    xaxis: { gridcolor: '#30363d', zerolinecolor: '#30363d' },
    yaxis: { gridcolor: '#30363d', zerolinecolor: '#30363d' },
    margin: { l: 60, r: 30, t: 30, b: 40 },
    legend: { x: 0, y: 1, bgcolor: 'rgba(0,0,0,0)' },
};

async function load() {
    const s = await fetch('/api/summary').then(r => r.json());
    const histories = {};
    for (const name of Object.keys(s)) {
        if (typeof s[name] === 'object') {
            histories[name] = await fetch('/api/history/' + name).then(r => r.json()).catch(() => []);
        }
    }
    renderSummary(s);
    renderEquity(histories, s.initial_nav);
    renderRegime(histories);
    renderPositions(histories);
    renderTable(s);
}

function renderSummary(s) {
    const sum = document.getElementById('summary');
    sum.innerHTML = '';
    document.getElementById('updated').textContent = '— Updated ' + (s.updated_at || '').slice(0,16);

    const items = [
        ['Initial NAV', `$${Math.round(s.initial_nav || 0).toLocaleString()}`, ''],
        ['Period', `${(s.years || 0).toFixed(1)} years`, ''],
        ['UNG Return', `${(s.ung_return_pct || 0).toFixed(1)}%`, s.ung_return_pct >= 0 ? 'gain' : 'loss'],
    ];
    for (const [label, val, cls] of items) {
        sum.innerHTML += `<div class="card"><h3>${label}</h3><div class="val ${cls}">${val}</div></div>`;
    }
}

function renderEquity(histories, initialNav) {
    const traces = [];
    const colors = { naive_atm: '#f85149', otm_managed: '#d29922', regime_aware: '#3fb950', deep_otm_passive: '#a5a5a5' };
    for (const [name, hist] of Object.entries(histories)) {
        if (!Array.isArray(hist) || !hist.length) continue;
        traces.push({
            x: hist.map(h => h.date),
            y: hist.map(h => h.nav),
            type: 'scatter', mode: 'lines', name,
            line: { color: colors[name] || '#58a6ff', width: 2 },
        });
    }
    // UNG benchmark
    const first = Object.values(histories)[0];
    if (first && first.length) {
        const startSpot = first[0].spot;
        traces.push({
            x: first.map(h => h.date),
            y: first.map(h => h.spot / startSpot * initialNav),
            type: 'scatter', mode: 'lines', name: 'UNG buy-hold',
            line: { color: '#8b949e', width: 1, dash: 'dot' },
        });
    }
    Plotly.react('equityChart', traces, {
        ...PLOT_LAYOUT,
        title: 'Equity Curves',
        xaxis: { ...PLOT_LAYOUT.xaxis, title: 'Date' },
        yaxis: { ...PLOT_LAYOUT.yaxis, title: 'NAV ($)', tickformat: '$,.0f' },
    }, { responsive: true, displayModeBar: false });
}

function renderRegime(histories) {
    const first = Object.values(histories)[0];
    if (!first || !first.length) return;
    // Use first strategy's regime timeline (all share spot)
    const colorMap = { EXTREME_CHEAP: '#3fb950', CHEAP: '#7dd87f', NEUTRAL: '#8b949e', RICH: '#d29922', EXTREME_RICH: '#f85149' };
    const traces = [{
        x: first.map(h => h.date),
        y: first.map(h => h.z),
        type: 'scatter', mode: 'lines', name: 'Composite z',
        line: { color: '#58a6ff', width: 1 },
    }];
    Plotly.react('regimeChart', traces, {
        ...PLOT_LAYOUT,
        title: 'Z-Score Over Time',
        xaxis: { ...PLOT_LAYOUT.xaxis, title: 'Date' },
        yaxis: { ...PLOT_LAYOUT.yaxis, title: 'Z-Score (negative=rich, positive=cheap)' },
        shapes: [
            { type: 'line', x0: first[0].date, x1: first[first.length-1].date, y0: 1, y1: 1, line: { color: '#3fb950', dash: 'dash', width: 1 } },
            { type: 'line', x0: first[0].date, x1: first[first.length-1].date, y0: -1, y1: -1, line: { color: '#f85149', dash: 'dash', width: 1 } },
            { type: 'line', x0: first[0].date, x1: first[first.length-1].date, y0: 0, y1: 0, line: { color: '#8b949e', dash: 'dot', width: 1 } },
        ],
    }, { responsive: true, displayModeBar: false });
}

function renderPositions(histories) {
    const ra = histories.regime_aware;
    if (!ra) return;
    Plotly.react('positionsChart', [
        { x: ra.map(h => h.date), y: ra.map(h => h.shares), type: 'scatter', mode: 'lines', name: 'UNG Shares', line: { color: '#58a6ff' } },
        { x: ra.map(h => h.date), y: ra.map(h => h.boxx), type: 'scatter', mode: 'lines', name: 'BOXX Shares', yaxis: 'y2', line: { color: '#3fb950' } },
        { x: ra.map(h => h.date), y: ra.map(h => h.kold), type: 'scatter', mode: 'lines', name: 'KOLD Shares', yaxis: 'y2', line: { color: '#f85149' } },
    ], {
        ...PLOT_LAYOUT,
        title: 'Position Evolution (Regime-Aware Strategy)',
        xaxis: { ...PLOT_LAYOUT.xaxis, title: 'Date' },
        yaxis: { ...PLOT_LAYOUT.yaxis, title: 'UNG Shares', titlefont: { color: '#58a6ff' } },
        yaxis2: { title: 'BOXX/KOLD Shares', overlaying: 'y', side: 'right', titlefont: { color: '#3fb950' } },
    }, { responsive: true, displayModeBar: false });
}

function renderTable(s) {
    const table = document.getElementById('comparisonTable');
    table.innerHTML = '<tr><th>Strategy</th><th>Final NAV</th><th>Return</th><th>Annual</th><th>Max DD</th><th>Sharpe</th></tr>';
    for (const [name, r] of Object.entries(s)) {
        if (typeof r !== 'object' || !r.final) continue;
        const retClass = r.return_pct >= 0 ? 'gain' : 'loss';
        table.innerHTML += `<tr>
            <td>${name}</td>
            <td>$${Math.round(r.final).toLocaleString()}</td>
            <td class="${retClass}">${r.return_pct.toFixed(1)}%</td>
            <td>${r.annual_pct.toFixed(1)}%</td>
            <td class="loss">${r.max_dd_pct.toFixed(1)}%</td>
            <td>${r.sharpe.toFixed(2)}</td>
        </tr>`;
    }
}

load();
setInterval(load, 60000);  // refresh every minute
</script>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self._send_html(HTML_PAGE)
            return
        if self.path == '/api/summary':
            path = os.path.join(RESULTS_DIR, 'summary.json')
            if os.path.exists(path):
                with open(path) as f:
                    self._send_json(json.load(f))
            else:
                self._send_json({'error': 'no summary yet — run replay_engine.py'})
            return
        if self.path.startswith('/api/history/'):
            name = self.path.split('/')[-1]
            path = os.path.join(RESULTS_DIR, f'{name}_history.csv')
            if os.path.exists(path):
                import csv
                rows = []
                with open(path) as f:
                    for row in csv.DictReader(f):
                        # Convert numeric strings
                        for k in row:
                            if k != 'date' and k != 'regime':
                                try: row[k] = float(row[k])
                                except: pass
                        rows.append(row)
                self._send_json(rows)
            else:
                self._send_json([])
            return
        self.send_error(404)

    def _send_html(self, content):
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(content.encode('utf-8'))

    def _send_json(self, data):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, default=str).encode('utf-8'))

    def log_message(self, format, *args):
        pass  # silence


def main():
    PORT = 9998
    print(f"Backtest server running at http://localhost:{PORT}")
    HTTPServer(('0.0.0.0', PORT), Handler).serve_forever()


if __name__ == '__main__':
    main()
