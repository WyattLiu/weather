"""TRADER'S DIARY dashboard — the ENTIRE backtest history as an interactive web book.
Port 10002. Run the full backtest once, then serve a day-by-day diary: equity curve,
daily BOOK (shares/BOXX/puts-by-strike/calls), VALUE + daily P&L, every TRADE with its
WHY, the regime, and stand-down signals. Filter by regime / flagged days / date range.

  venv/bin/python journal_dashboard.py            # serves http://0.0.0.0:10002
"""
import os, sys, json, math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import pandas as pd

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS, 'backtest'))
import replay_engine as R

KERNEL = os.environ.get('JOURNAL_KERNEL', 'regime_wheel_boxx')
START = os.environ.get('JOURNAL_START', '2021-06-17')
END = os.environ.get('JOURNAL_END', '2026-06-16')

WHY = {
    'OPEN_PUT': 'Sell cash-secured put — harvest premium / accumulate on the dip',
    'OPEN_CC': 'Sell covered call — premium income on uncovered shares',
    'OPEN_ITM_CC': 'Sell IN-THE-MONEY covered call — fat premium, the income engine in a decline',
    'PUT_TP': 'Take profit — buy the put back cheap, lock the gain, free the collateral',
    'CALL_TP': 'Take profit — buy the call back cheap, lock the gain',
    'PUT_ROLL_DOWN': 'Roll the put down — defend against assignment as price falls',
    'PUT_ASSIGN': 'Put assigned — acquired 100×qty shares at the strike',
    'CALL_ASSIGN': 'Call assigned — shares called away at the strike (distribute high)',
    'Z_TARGET_ADD': 'Buy shares — valuation share-target (cheap, build the book)',
    'OPEN_REBUILD_PUT': 'Re-sell put after a trim — rebuild premium income',
    'DD_TRIM_SHARES': 'Trim shares — drawdown control',
    'ELEVATOR_CLOSE': 'Elevator close — lock the rally gain before mean reversion',
    'KOLD_SHOULDER_ENTRY': 'KOLD inverse hedge — shoulder-season protection',
    'KOLD_REGIME_BUY': 'KOLD inverse hedge — distribute regime',
    'OPEN_LONG_PUT_FLOOR': 'Buy long-put floor — defined-downside crash insurance',
    'STAND_DOWN_GRIND': 'STAND DOWN — confirmed grind-down, do not add risk',
    'SKIP_PUT_FALLING_KNIFE': 'SKIP put — falling knife, do not sell into the drop',
    'OPEN_PUT_REJECTED_MARGIN': 'Put skipped — margin limit reached (never over-extend)',
    'PUT_ROLL_SKIP_SURGE': 'Roll skipped — momentum surge (let it ride)',
}
ACTIONS = {'OPEN_PUT', 'OPEN_CC', 'OPEN_ITM_CC', 'PUT_TP', 'CALL_TP', 'PUT_ROLL_DOWN',
           'PUT_ASSIGN', 'CALL_ASSIGN', 'OPEN_LONG_PUT_FLOOR', 'KOLD_REGIME_BUY',
           'KOLD_SHOULDER_ENTRY', 'Z_TARGET_ADD', 'DD_TRIM_SHARES', 'OPEN_REBUILD_PUT',
           'ELEVATOR_CLOSE', 'CALL_ROLL_UP'}
WHY_DAY = {'ACC': 'Tight supply (bullish storage surprise) — accumulate, harvest premium.',
           'DIST': 'Oversupply (bearish surprise) — defensive: trim exposure, BOXX yield, ITM-call income.',
           'NEU': 'Balanced regime — harvest premium, hold low share exposure, stay covered.'}

_CACHE = {'data': None}


def _apply(sp, sc, ty, K, q, fk, tk):
    K = round(float(K), 1) if (K == K and K) else None
    if ty == 'OPEN_PUT' and K: sp[K] += q
    elif ty in ('OPEN_CC', 'OPEN_ITM_CC') and K: sc[K] += q
    elif ty in ('PUT_TP', 'PUT_ASSIGN', 'PUT_EXPIRE_OTM') and K: sp[K] = max(0, sp[K] - q)
    elif ty in ('CALL_TP', 'CALL_ASSIGN', 'CALL_EXPIRE_OTM') and K: sc[K] = max(0, sc[K] - q)
    elif ty == 'PUT_ROLL_DOWN':
        if fk == fk and fk: sp[round(float(fk), 1)] = max(0, sp[round(float(fk), 1)] - q)
        if tk == tk and tk: sp[round(float(tk), 1)] += q


def build():
    df = pd.read_csv(os.path.join(R.CACHE_DIR, 'master_dataset.csv'), parse_dates=[0], index_col=0)
    df = R.precompute_factor_z(df).dropna(subset=['UNG']).loc[START:END]
    p = {**R.STRATEGIES[KERNEL], 'intraday_exec': False, 'real_chain_pricing': False}
    h, t = R.run_strategy_simple(df, p, 100000, 0)
    h = h.set_index(pd.to_datetime(h['date']))
    t['d'] = pd.to_datetime(t['date'])
    bx = df['BOXX'].reindex(h.index, method='ffill').fillna(117)
    spk = df['KOLD'].reindex(h.index, method='ffill').fillna(0)
    ssz = df['storage_surprise_z'].reindex(h.index, method='ffill')
    rs = df['regime_strength'].reindex(h.index, method='ffill')
    days, prev, equity = [], None, []
    for dt, hr in h.iterrows():
        td = t[t['d'] == dt]
        trades, signals = [], []
        for _, r in td.iterrows():
            ty = r['type']; K = r.get('K'); q = abs(int(r['qty'])) if ('qty' in r and r['qty'] == r['qty']) else 0
            ks = (f"{q}×${float(K):.1f}" if (K == K and K) else '')
            rec = {'type': ty, 'label': ks, 'why': WHY.get(ty, ty),
                   'credit': round(float(r['credit']), 0) if ('credit' in r and r['credit'] == r['credit']) else 0,
                   'pnl': round(float(r['pnl']), 0) if ('pnl' in r and r['pnl'] == r['pnl']) else 0}
            (trades if ty in ACTIONS else signals).append(rec)
        # REAL book from the engine (no reconstruction leak)
        pbk = {round(float(k), 1): int(v) for k, v in (hr.get('put_book') or {}).items() if v}
        cbk = {round(float(k), 1): int(v) for k, v in (hr.get('call_book') or {}).items() if v}
        sp = {k: v for k, v in pbk.items()}; sc = {k: v for k, v in cbk.items()}
        sh = int(hr['shares']); ncall = sum(sc.values()); nput = sum(sp.values())
        topK = max(sp.items(), key=lambda x: x[1]) if sp else (None, 0)
        reg = 'ACC' if ssz.get(dt, 0) < -0.5 else 'DIST' if ssz.get(dt, 0) > 0.5 else 'NEU'
        nav = float(hr['nav']); dnav = (nav - prev) if prev else 0.0; prev = nav
        equity.append([str(dt.date()), round(nav, 0), round(hr['boxx'] * bx.get(dt, 117), 0),
                       round(sh * hr['spot'], 0)])
        days.append({'date': str(dt.date()), 'regime': reg, 'spot': round(float(hr['spot']), 2),
                     'ssz': round(float(ssz.get(dt, 0) or 0), 2), 'rstr': round(float(rs.get(dt, 0) or 0), 2),
                     'nav': round(nav, 0), 'dnav': round(dnav, 0), 'cash': round(float(hr['cash']), 0),
                     'shares': sh, 'boxx': round(float(hr['boxx'] * bx.get(dt, 117)), 0),
                     'kold': round(float(hr['kold'] * spk.get(dt, 0)), 0),
                     'nSP': nput, 'nSC': ncall, 'cov': f"{ncall}/{sh // 100}", 'covered': ncall <= sh // 100,
                     'topK': topK[0], 'topKq': topK[1],
                     'put_book': {str(k): v for k, v in sorted(sp.items()) if v > 0},
                     'call_book': {str(k): v for k, v in sorted(sc.items()) if v > 0},
                     'trades': trades, 'signals': signals, 'why_day': WHY_DAY[reg]})
    nav0, nav1 = days[0]['nav'], days[-1]['nav']
    yrs = (pd.Timestamp(days[-1]['date']) - pd.Timestamp(days[0]['date'])).days / 365.25
    navs = pd.Series([d['nav'] for d in days])
    ret = navs.pct_change().dropna()
    summary = {'kernel': KERNEL, 'start': START, 'end': END, 'days': len(days),
               'ann': round(((nav1 / nav0) ** (1 / yrs) - 1) * 100, 1) if yrs > 0 else 0,
               'sharpe': round(ret.mean() / (ret.std() + 1e-12) * math.sqrt(252), 2),
               'mdd': round((navs / navs.cummax() - 1).min() * 100, 1),
               'nav0': nav0, 'nav1': nav1, 'trade_days': sum(1 for d in days if d['trades']),
               'peak_conc': max(d['topKq'] for d in days), 'fill_basis': 'model (diary view)'}
    return {'summary': summary, 'equity': equity, 'days': days}


def get_data():
    if _CACHE['data'] is None:
        _CACHE['data'] = build()
    return _CACHE['data']


PAGE = """<!doctype html><html><head><meta charset=utf-8><title>Trader's Diary — %KERNEL%</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
body{font-family:-apple-system,system-ui,sans-serif;background:#0d1117;color:#e6edf3;margin:0;padding:16px}
h1{font-size:1.2rem;margin:0 0 4px} .dim{color:#8b949e}
#summary{display:flex;gap:18px;flex-wrap:wrap;margin:8px 0 14px}
.kpi{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:8px 14px}
.kpi b{font-size:1.25rem} .kpi .l{font-size:.72rem;color:#8b949e;text-transform:uppercase}
#filters{margin:10px 0;display:flex;gap:8px;align-items:center;flex-wrap:wrap}
button,select,input{background:#21262d;color:#e6edf3;border:1px solid #30363d;border-radius:6px;padding:6px 10px;cursor:pointer}
button.active{background:#1f6feb;border-color:#1f6feb}
#chart{height:300px;background:#161b22;border:1px solid #30363d;border-radius:8px;margin-bottom:12px}
.day{background:#161b22;border:1px solid #30363d;border-radius:8px;margin:6px 0;padding:10px 14px}
.day .hd{display:flex;gap:14px;align-items:baseline;flex-wrap:wrap;cursor:pointer}
.reg{font-weight:700;padding:1px 8px;border-radius:4px;font-size:.8rem}
.ACC{background:#1a3d2a;color:#3fb950} .DIST{background:#3d1a1a;color:#f85149} .NEU{background:#21262d;color:#8b949e}
.pos{color:#3fb950} .neg{color:#f85149}
.body{display:none;margin-top:8px;font-size:.86rem}
.day.open .body{display:block}
.trade{margin:3px 0;padding:4px 8px;border-left:3px solid #1f6feb;background:#0d1117;border-radius:0 4px 4px 0}
.sig{margin:3px 0;padding:4px 8px;border-left:3px solid #8b949e;background:#0d1117;color:#8b949e;border-radius:0 4px 4px 0}
.book{font-family:monospace;font-size:.8rem;color:#8b949e;margin:4px 0}
.warn{color:#d29922}
</style></head><body>
<h1>📓 Trader's Diary — <span id=ktitle></span></h1>
<div class=dim>Entire backtest history · cash start $100k · click any day to expand the book + every trade's why</div>
<div id=summary></div>
<div id=chart></div>
<div id=filters>
  <span class=dim>regime:</span>
  <button class=rf data-r=ALL>all</button><button class=rf data-r=ACC>accumulate</button>
  <button class=rf data-r=NEU>neutral</button><button class=rf data-r=DIST>distribute</button>
  <button id=tradesonly>trade-days only</button>
  <input id=search placeholder="search date / strike…" style="cursor:text">
  <span class=dim id=count></span>
</div>
<div id=diary></div>
<script>
let DATA=null, RF='ALL', TRADESONLY=false, Q='';
fetch('/api/journal').then(r=>r.json()).then(d=>{DATA=d;render()});
function fmt(x){return (x||0).toLocaleString(undefined,{maximumFractionDigits:0})}
function render(){
  const s=DATA.summary; document.getElementById('ktitle').textContent=s.kernel+'  ('+s.start+' → '+s.end+')';
  document.getElementById('summary').innerHTML=[
    ['Annualized', (s.ann>=0?'+':'')+s.ann+'%'],['Sharpe',s.sharpe],['Max DD',s.mdd+'%'],
    ['NAV', '$'+fmt(s.nav0)+' → $'+fmt(s.nav1)],['Days',s.days+' ('+s.trade_days+' active)'],
    ['Peak put-concentration', s.peak_conc+' @ 1 strike']
  ].map(k=>'<div class=kpi><div class=l>'+k[0]+'</div><b>'+k[1]+'</b></div>').join('');
  const e=DATA.equity;
  Plotly.newPlot('chart',[
    {x:e.map(r=>r[0]),y:e.map(r=>r[1]),name:'NAV',line:{color:'#1f6feb',width:2}},
    {x:e.map(r=>r[0]),y:e.map(r=>r[2]),name:'BOXX',line:{color:'#3fb950',width:1},fill:'tozeroy',fillcolor:'rgba(63,185,80,.08)'},
    {x:e.map(r=>r[0]),y:e.map(r=>r[3]),name:'UNG shares $',line:{color:'#d29922',width:1}}
  ],{paper_bgcolor:'#161b22',plot_bgcolor:'#161b22',font:{color:'#8b949e'},margin:{t:10,r:10,b:30,l:50},
     legend:{orientation:'h'},xaxis:{gridcolor:'#21262d'},yaxis:{gridcolor:'#21262d'}},{responsive:true});
  drawDays();
}
function drawDays(){
  let days=DATA.days.filter(d=>(RF=='ALL'||d.regime==RF)&&(!TRADESONLY||d.trades.length));
  if(Q){days=days.filter(d=>JSON.stringify(d).toLowerCase().includes(Q))}
  document.getElementById('count').textContent=days.length+' days';
  const html=days.slice().reverse().map(d=>{
    const dn=d.dnav>=0?'pos':'neg';
    const pb=Object.entries(d.put_book).map(([k,v])=>v+'×$'+k).join('  ')||'—';
    const cb=Object.entries(d.call_book).map(([k,v])=>v+'×$'+k).join('  ')||'—';
    const conc=d.topKq>=8?' <span class=warn>⚠ '+d.topKq+'@$'+d.topK+'</span>':'';
    const trd=d.trades.map(t=>'<div class=trade><b>'+t.type+'</b> '+t.label+
        (t.credit?' <span class=pos>+$'+fmt(t.credit)+'</span>':'')+(t.pnl?' <span class="'+(t.pnl>=0?'pos':'neg')+'">'+(t.pnl>=0?'+':'')+'$'+fmt(t.pnl)+'</span>':'')+
        '<div class=dim>↳ '+t.why+'</div></div>').join('');
    const sig=d.signals.map(t=>'<div class=sig>'+t.type+' — '+t.why+'</div>').join('');
    return '<div class="day"><div class=hd onclick="this.parentNode.classList.toggle(\\'open\\')">'+
      '<span class="reg '+d.regime+'">'+d.regime+'</span>'+
      '<b>'+d.date+'</b> <span class=dim>$'+d.spot+'</span>'+
      '<span>NAV $'+fmt(d.nav)+' <span class='+dn+'>'+(d.dnav>=0?'+':'')+'$'+fmt(d.dnav)+'</span></span>'+
      '<span class=dim>sh '+fmt(d.shares)+' · BOXX $'+fmt(d.boxx)+' · SP '+d.nSP+' · SC '+d.nSC+' · cov '+d.cov+conc+'</span>'+
      (d.trades.length?'<span class=dim> · '+d.trades.length+' trade(s)</span>':'')+'</div>'+
      '<div class=body><div class=dim style="margin-bottom:6px">'+d.why_day+'  (storage-surprise '+d.ssz+', regime-strength '+d.rstr+')</div>'+
      '<div class=book>PUTS: '+pb+'<br>CALLS: '+cb+'</div>'+trd+sig+'</div></div>';
  }).join('');
  document.getElementById('diary').innerHTML=html;
}
document.querySelectorAll('.rf').forEach(b=>b.onclick=()=>{RF=b.dataset.r;document.querySelectorAll('.rf').forEach(x=>x.classList.remove('active'));b.classList.add('active');drawDays()});
document.getElementById('tradesonly').onclick=function(){TRADESONLY=!TRADESONLY;this.classList.toggle('active');drawDays()};
document.getElementById('search').oninput=function(){Q=this.value.toLowerCase();drawDays()};
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith('/api/journal'):
            body = json.dumps(get_data(), default=str).encode()
            self.send_response(200); self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body))); self.end_headers()
            try: self.wfile.write(body)
            except Exception: pass
        else:
            body = PAGE.replace('%KERNEL%', KERNEL).encode()
            self.send_response(200); self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', str(len(body))); self.end_headers()
            try: self.wfile.write(body)
            except Exception: pass


if __name__ == '__main__':
    print(f"Building diary for {KERNEL} ({START}→{END})…", flush=True)
    get_data()
    print(f"Ready: {len(_CACHE['data']['days'])} days. Serving http://0.0.0.0:10002", flush=True)
    ThreadingHTTPServer(('0.0.0.0', 10002), H).serve_forever()
