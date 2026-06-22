"""Live SPY vega-scrape setup alert for the dashboard.

GREEN when the kernel would deploy: VIX<=16 (low absolute vol) AND IV>=RV20 (not just-after a spike).
Reads daily SPY/VIX from research/spy_vol/cache/spy_vix_daily.csv (refreshed by the 5pm cron) and,
if reachable, overrides VIX/SPY with a live IBKR snapshot (cached 10 min). Self-contained + guarded.
"""
import os
import time
import math
import numpy as np
import pandas as pd

THIS = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(THIS, 'research', 'spy_vol', 'cache', 'spy_vix_daily.csv')
_CACHE = {'ts': 0.0, 'data': None}
TTL = 600


def _from_csv():
    df = pd.read_csv(CSV, index_col=0, parse_dates=True).dropna(subset=['SPY', 'VIX'])
    spy, vix = df['SPY'], df['VIX']
    rv20 = float((np.log(spy / spy.shift(1)).rolling(20).std() * math.sqrt(252)).iloc[-1])
    return {'asof': str(df.index[-1].date()), 'spy': float(spy.iloc[-1]), 'vix': float(vix.iloc[-1]),
            'rv20': rv20, 'vix_std10': float(vix.rolling(10).std().iloc[-1]),
            'dist_high': float(spy.iloc[-1] / spy.rolling(252).max().iloc[-1] - 1), 'src': 'csv-close'}


def _live_vix_spy():
    try:
        import sys
        import asyncio
        sys.path.insert(0, THIS)
        try:                                  # ib_insync needs an event loop in this thread
            asyncio.get_event_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())
        from modules.common import connect
        from ib_insync import Stock, Index
        ib = connect(client_id=91)
        try:
            ib.reqMarketDataType(3)
        except Exception:
            pass
        vx = Index('VIX', 'CBOE'); sp = Stock('SPY', 'SMART', 'USD'); ib.qualifyContracts(vx, sp)
        tv = ib.reqMktData(vx, '', False, False); ts = ib.reqMktData(sp, '', False, False)
        for _ in range(8):
            ib.sleep(1)
            if any(v == v and v > 0 for v in (tv.last, tv.close)):
                break

        def px(t):
            for v in (t.last, t.close, t.markPrice, t.bid):
                if v == v and v and v > 0:
                    return float(v)
            return None
        v, s = px(tv), px(ts)
        ib.disconnect()
        return v, s
    except Exception:
        return None


def spy_vega_signal(force=False):
    if not force and _CACHE['data'] and time.time() - _CACHE['ts'] < TTL:
        return _CACHE['data']
    try:
        base = _from_csv()
    except Exception as e:
        return {'error': f'no SPY/VIX data: {e}'}
    live = _live_vix_spy()
    if live and live[0]:
        base['vix'] = live[0]
        if live[1]:
            base['spy'] = live[1]
        base['src'] = 'ibkr-live'; base['asof'] = 'live'
    vix, rv20 = base['vix'], base['rv20']
    iv = vix / 100.0
    low_vix = vix <= 16
    not_cheap = iv >= rv20
    consolidated = base['vix_std10'] < 1.5
    if low_vix and not_cheap:
        verdict, msg = 'GREEN', 'SETUP — buy ~45D ATM straddle (afternoon, combo-mid)'
    elif vix <= 17.5:
        verdict, msg = 'WATCH', 'close — waiting for VIX to settle ≤16'
    else:
        verdict, msg = 'RED', 'no setup — VIX too high (vol mean-reverts down)'
    base.update({'iv': iv, 'low_vix': low_vix, 'not_cheap': not_cheap, 'consolidated': consolidated,
                 'verdict': verdict, 'msg': msg})
    _CACHE['data'] = base; _CACHE['ts'] = time.time()
    return base


if __name__ == '__main__':
    import json
    print(json.dumps(spy_vega_signal(force=True), indent=2, default=str))
