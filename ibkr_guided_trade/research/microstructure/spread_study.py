"""Bid-ask spread microstructure — MM fear as a LEADING signal.

User hypothesis: "if something MM is feared they will move before that"
— market makers widen quotes BEFORE events/moves, so abnormal spread is
a leading indicator, not a coincident one.

Tests (data from eia_release_study.py fetches):
  A. THURSDAY EVENT STUDY (bars_ba, 1-min BID_ASK):
     - event-time spread curve 09:30-12:00 around the 10:30 print
     - pre-print spread z (10:15-10:29 vs own-morning baseline)
       → does it PREDICT |jump|?  (MMs sniffing the surprise)
     - post-print normalization half-life
  B. ALL-DAYS LEADING TEST (bars_alldays_bid_ask + _trades):
     - daily abnormal spread (vs trailing 60d, time-of-day matched)
       → next-day |return| (vol prediction) and signed return
     - close-window spread widening (15:30-16:00) → overnight+next-day
     - WEDNESDAY spread → THURSDAY print size (fear before the event)

Run: venv/bin/python research/microstructure/spread_study.py
"""
import os
import pandas as pd
from scipy import stats as sstats

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def load_ba(sub):
    """BID_ASK bars: open=avg bid, close=avg ask → spread = close - open."""
    d = os.path.join(THIS_DIR, sub)
    if not os.path.isdir(d):
        return {}
    out = {}
    for f in sorted(os.listdir(d)):
        if not f.endswith('.csv'):
            continue
        try:
            df = pd.read_csv(os.path.join(d, f), parse_dates=['ts'])
        except Exception:
            continue
        if df.empty:
            continue
        df['spread'] = df['close'] - df['open']
        df['mid'] = (df['close'] + df['open']) / 2
        df = df[(df['spread'] >= 0) & (df['spread'] < df['mid'] * 0.05)]
        df['hm'] = df['ts'].dt.strftime('%H:%M')
        out[f[4:14]] = df
    return out


def load_px(sub):
    d = os.path.join(THIS_DIR, sub)
    if not os.path.isdir(d):
        return {}
    out = {}
    for f in sorted(os.listdir(d)):
        if not f.endswith('.csv'):
            continue
        try:
            df = pd.read_csv(os.path.join(d, f), parse_dates=['ts'])
        except Exception:
            continue
        if df.empty:
            continue
        df['hm'] = df['ts'].dt.strftime('%H:%M')
        out[f[4:14]] = df
    return out


def thursday_event_study():
    ba = load_ba('bars_ba')
    px = load_px('bars')
    if not ba:
        print('A. no Thursday BID_ASK data yet')
        return
    curves, rows = [], []
    for day, df in ba.items():
        s = df.set_index('hm')['spread']
        m = df.set_index('hm')['mid']
        base = s.loc['09:35':'10:00'].mean()
        pre = s.loc['10:15':'10:29'].mean()
        post5 = s.loc['10:30':'10:35'].mean()
        post30 = s.loc['10:55':'11:05'].mean()
        if any(pd.isna(x) or x == 0 for x in (base, pre, post5)):
            continue
        jump = None
        if day in px:
            p = px[day].set_index('hm')['close']
            if '10:29' in p.index and '10:35' in p.index:
                jump = abs(p['10:35'] / p['10:29'] - 1)
        rows.append({'day': day, 'base_bps': base / m.loc['09:35':'10:00'].mean() * 1e4,
                     'pre_ratio': pre / base, 'post5_ratio': post5 / base,
                     'post30_ratio': post30 / base if post30 == post30 else None,
                     'abs_jump': jump})
        curves.append((s / base).rename(day))
    t = pd.DataFrame(rows).dropna(subset=['pre_ratio', 'post5_ratio'])
    print(f'=== A. THURSDAY SPREAD EVENT STUDY ({len(t)} days) ===')
    print(f'  baseline spread:      {t["base_bps"].median():.1f} bps of mid')
    print(f'  pre-print  (10:15-29): {t["pre_ratio"].median():.2f}x baseline')
    print(f'  at print   (10:30-35): {t["post5_ratio"].median():.2f}x baseline')
    print(f'  +30 min    (10:55-05): {t["post30_ratio"].median():.2f}x baseline')
    tj = t.dropna(subset=['abs_jump'])
    if len(tj) > 30:
        r, p = sstats.spearmanr(tj['pre_ratio'], tj['abs_jump'])
        print(f'  PRE-print widening → |jump| size: spearman r={r:+.2f} p={p:.3f} '
              f'(n={len(tj)})  ← "fear moves first" test')
        hi = tj[tj['pre_ratio'] > tj['pre_ratio'].quantile(0.8)]['abs_jump']
        lo = tj[tj['pre_ratio'] < tj['pre_ratio'].quantile(0.2)]['abs_jump']
        print(f'  |jump| when MMs widened pre-print: {hi.mean():.2%} vs calm: {lo.mean():.2%}')
    # average event curve
    ec = pd.concat(curves, axis=1).median(axis=1)
    key_times = ['09:45', '10:00', '10:15', '10:25', '10:29', '10:30',
                 '10:31', '10:33', '10:35', '10:45', '11:00', '11:30']
    print('  median spread curve (x baseline):')
    print('   ' + '  '.join(f'{k}={ec.get(k, float("nan")):.2f}' for k in key_times))


def alldays_leading_test():
    ba = load_ba('bars_alldays_bid_ask')
    px = load_px('bars_alldays_trades')
    if not ba or not px:
        print('\nB. all-days data not complete yet')
        return
    feats = []
    for day, df in ba.items():
        s = df.set_index('hm')['spread']
        m = df['mid'].median()
        if m <= 0 or len(s) < 100:
            continue
        feats.append({'date': pd.Timestamp(day),
                      'spread_bps': s.median() / m * 1e4,
                      'close_spread_bps': s.loc['15:30':'15:59'].median() / m * 1e4})
    F = pd.DataFrame(feats).set_index('date').sort_index()
    closes = {}
    for day, df in px.items():
        closes[pd.Timestamp(day)] = float(df['close'].iloc[-1])
    C = pd.Series(closes).sort_index()
    F['ret_next'] = C.reindex(F.index).pct_change().shift(-1)
    F['absret_next'] = F['ret_next'].abs()
    # abnormal spread vs trailing 60d
    for col in ('spread_bps', 'close_spread_bps'):
        F[f'{col}_z'] = ((F[col] - F[col].rolling(60, min_periods=30).mean())
                         / F[col].rolling(60, min_periods=30).std())
    print(f'\n=== B. ALL-DAYS LEADING TEST ({len(F)} days) ===')
    for col in ('spread_bps_z', 'close_spread_bps_z'):
        f = F.dropna(subset=[col, 'ret_next'])
        if len(f) < 100:
            continue
        rv, pv = sstats.spearmanr(f[col], f['absret_next'])
        rd, pd_ = sstats.spearmanr(f[col], f['ret_next'])
        print(f'  {col:>20}: → next |ret| r={rv:+.2f} (p={pv:.3f})   '
              f'→ next signed ret r={rd:+.2f} (p={pd_:.3f})')
        wide = f[f[col] > 1.5]
        calm = f[f[col] < 0]
        print(f'      after WIDE days (z>1.5, n={len(wide)}): next |ret| '
              f'{wide["absret_next"].mean():.2%}, signed {wide["ret_next"].mean():+.2%}')
        print(f'      after calm days (z<0,  n={len(calm)}): next |ret| '
              f'{calm["absret_next"].mean():.2%}, signed {calm["ret_next"].mean():+.2%}')
    # Wednesday spread → Thursday jump
    F['dow'] = F.index.dayofweek
    wed = F[F['dow'] == 2]
    px_thu = load_px('bars')
    jumps = {}
    for day, df in px_thu.items():
        p = df.set_index('hm')['close']
        if '10:29' in p.index and '10:35' in p.index:
            jumps[pd.Timestamp(day)] = abs(p['10:35'] / p['10:29'] - 1)
    J = pd.Series(jumps).sort_index()
    rowsw = []
    for d, row in wed.iterrows():
        thu = d + pd.Timedelta(days=1)
        if thu in J.index and not pd.isna(row['spread_bps_z']):
            rowsw.append({'wed_z': row['spread_bps_z'], 'thu_jump': J[thu]})
    W = pd.DataFrame(rowsw)
    if len(W) > 30:
        r, p = sstats.spearmanr(W['wed_z'], W['thu_jump'])
        print(f'\n  WEDNESDAY abnormal spread → THURSDAY |print jump|: '
              f'r={r:+.2f} p={p:.3f} (n={len(W)})  ← fear-before-the-event')


if __name__ == '__main__':
    thursday_event_study()
    alldays_leading_test()
