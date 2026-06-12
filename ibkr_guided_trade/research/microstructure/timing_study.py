"""Execution-timing study — when to roll/enter/exit, from minute data.

Inputs: bars_alldays_trades (1-min TRADES, ~1,044 days) and, when
available, bars_alldays_bid_ask (1-min NBBO).

Outputs the executor's playbook details:
  1. SPREAD by 30-min bucket + day-of-week  → cheapest windows to trade
  2. REALIZED VOL by 30-min bucket          → calmest windows (tight
     option quotes track calm + tight underlying)
  3. INTRADAY DRIFT by hour + weekday       → systematic timing of fills
  4. Thursday-specific: full-day pattern vs other days (print day)

Run: venv/bin/python research/microstructure/timing_study.py
"""
import os
import math
import pandas as pd
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def load_all(sub, col_map=None):
    d = os.path.join(THIS_DIR, sub)
    if not os.path.isdir(d):
        return None
    parts = []
    for f in sorted(os.listdir(d)):
        if not f.endswith('.csv'):
            continue
        try:
            df = pd.read_csv(os.path.join(d, f))
            # ib bars carry tz offsets → normalize to US/Eastern
            df['ts'] = pd.to_datetime(df['ts'], utc=True,
                                      errors='coerce').dt.tz_convert('US/Eastern')
        except Exception:
            continue
        df = df.dropna(subset=['ts'])
        if df.empty:
            continue
        df['day'] = f[4:14]
        parts.append(df)
    if not parts:
        return None
    return pd.concat(parts, ignore_index=True)


def bucket(hm):
    h, m = int(hm[:2]), int(hm[3:5])
    return f'{h:02d}:{0 if m < 30 else 30:02d}'


def main():
    px = load_all('bars_alldays_trades')
    if px is None:
        print('no all-days TRADES yet')
        return
    px['hm'] = px['ts'].dt.strftime('%H:%M')
    px['bucket'] = px['hm'].map(bucket)
    px['dow'] = pd.to_datetime(px['day']).dt.dayofweek
    px['ret'] = px.groupby('day')['close'].pct_change()

    print(f'=== TIMING STUDY ({px["day"].nunique()} days, '
          f'{len(px):,} minute bars) ===')

    # 2. realized vol by bucket (annualized from 1-min)
    vol = (px.groupby('bucket')['ret'].std() * math.sqrt(252 * 390) * 100)
    print('\n-- Realized vol by 30-min bucket (ann %) --')
    print('  ' + '  '.join(f'{b}={v:.0f}' for b, v in vol.items()))

    # 3. drift by bucket (mean ret per bucket, bps per 30min)
    drift = px.groupby('bucket')['ret'].mean() * 30 * 1e4
    print('\n-- Mean drift by bucket (bps/30min) --')
    print('  ' + '  '.join(f'{b}={v:+.1f}' for b, v in drift.items()))

    # day-of-week close-to-close + intraday pattern
    daily = px.groupby('day')['close'].agg(['first', 'last'])
    daily.index = pd.to_datetime(daily.index)
    daily['intraday'] = daily['last'] / daily['first'] - 1
    daily['dow'] = daily.index.dayofweek
    dows = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
    print('\n-- Intraday (open→close) return by weekday (bps) --')
    g = daily.groupby('dow')['intraday'].agg(['mean', 'std', 'count'])
    for i, r in g.iterrows():
        print(f'  {dows[int(i)]}: {r["mean"]*1e4:+7.1f} bps  '
              f'(sd {r["std"]*1e4:.0f}, n={int(r["count"])})')

    # vol by weekday
    vw = px.groupby('dow')['ret'].std() * math.sqrt(252 * 390) * 100
    print('\n-- Realized vol by weekday (ann %) --')
    print('  ' + '  '.join(f'{dows[int(i)]}={v:.0f}' for i, v in vw.items()))

    # 1. spreads (if BID_ASK landed)
    ba = load_all('bars_alldays_bid_ask')
    if ba is not None:
        ba['spread'] = ba['close'] - ba['open']
        ba['mid'] = (ba['close'] + ba['open']) / 2
        ba = ba[(ba['spread'] >= 0) & (ba['spread'] < ba['mid'] * 0.02)]
        ba['bps'] = ba['spread'] / ba['mid'] * 1e4
        ba['hm'] = ba['ts'].dt.strftime('%H:%M')
        ba['bucket'] = ba['hm'].map(bucket)
        ba['dow'] = pd.to_datetime(ba['day']).dt.dayofweek
        sp = ba.groupby('bucket')['bps'].median()
        print('\n-- Median UNG spread by 30-min bucket (bps of mid) --')
        print('  ' + '  '.join(f'{b}={v:.1f}' for b, v in sp.items()))
        spw = ba.groupby('dow')['bps'].median()
        print('\n-- Median spread by weekday (bps) --')
        print('  ' + '  '.join(f'{dows[int(i)]}={v:.1f}' for i, v in spw.items()))
        best = sp.idxmin()
        print(f'\n>>> tightest window: {best} ({sp.min():.1f} bps) · '
              f'widest: {sp.idxmax()} ({sp.max():.1f} bps)')
    else:
        print('\n(spread sections pending — all-days BID_ASK still fetching)')


if __name__ == '__main__':
    main()
