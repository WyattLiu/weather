"""Directional ag blend — pure price alpha, long-only, walk-forward.

Post carry-invalidation ([[project_dba_factor_alpha]] correction):
ag premium is too thin to SELL; the validated PRICE factors monetize
through DIRECTION. This builds per-ticker long-only sleeves.

Per-ticker signal sets (economic sign PRIORS, not mined):
  grains (DBA, CORN, WEAT, SOYB):
    oni_low      ONI < 0           (La Niña wrecks S.American grain)
    cot_washed   own-market MM flow 13w < rolling q20  (capitulation)
    stu_tight    own-commodity world stocks-to-use z < rolling q30
    fpi_mom      Cereals sub-index 3m momentum > rolling q80
  sugar (CANE):
    oni_high     ONI > +0.5        (El Niño dries Brazil/Asia sugar)
    cot_washed   sugar MM flow 13w < q20
    fpi_mom      Sugar sub-index 3m momentum > q80
  DBA uses basket-average COT + Cereals FPI + avg stu.

Macro-squeeze trim (drawdown forensics lifts 3.3x/2.7x/2.3x):
  >=2 of {dxy 3m > +2%, crude 3m > +10%, own-COT flow hot} → halve w.

Walk-forward: ALL thresholds are rolling 756d past-only quantiles;
signals lagged for publication (ONI +1mo, COT +1wk, STU +1mo,
FPI +2mo). Monthly rebalance. Weights by score: {0:0, 1:0.4, 2:0.8,
3+:1.2} (cash floor earns BOXX 4.74%).

Run: venv/bin/python research/dba/directional_blend.py
"""
import os
import math
import json
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(THIS_DIR, 'cache')
FUND = os.path.join(CACHE, 'fundamentals')

BOXX_D = 0.0474 / 252
# CONCENTRATED (validated): only deploy at confluence>=2 — less time
# exposed to the ETFs' roll bleed, higher alpha density. Blend:
# +7.15%/yr Sharpe 0.91 vs BOXX 4.74%. Bear-put sleeve tested: FAILS
# (-1%/yr, theta+spreads > signal) — long-only it is.
W_BY_SCORE = {0: 0.0, 1: 0.0, 2: 0.8, 3: 1.5, 4: 1.5}
ROLL = 756

TICKER_CFG = {
    'DBA':  {'oni_sign': 'low',  'cot': None,      'stu': 'avg',   'fpi': 'Cereals'},
    'CORN': {'oni_sign': 'low',  'cot': 'corn',    'stu': 'Corn',  'fpi': 'Cereals'},
    'WEAT': {'oni_sign': 'low',  'cot': 'wheat',   'stu': 'Wheat', 'fpi': 'Cereals'},
    'SOYB': {'oni_sign': 'low',  'cot': 'soybeans','stu': 'Oilseed, Soybean', 'fpi': 'Oils'},
    'CANE': {'oni_sign': 'high', 'cot': 'sugar',   'stu': None,    'fpi': 'Sugar'},
}

# yfinance closes are split-adjusted; fine for RETURNS (no strike compare)
def load_inputs():
    panel = pd.read_csv(os.path.join(CACHE, 'master_panel.csv'),
                        index_col=0, parse_dates=True)
    oni = pd.read_csv(os.path.join(CACHE, 'oni.csv'),
                      index_col=0, parse_dates=True)['oni']
    oni.index = oni.index + pd.DateOffset(months=1)        # publication lag

    cot = pd.read_csv(os.path.join(FUND, 'cot_ag.csv'), parse_dates=['date'])
    cot['date'] = cot['date'] + pd.Timedelta(days=7)        # publication lag
    cot_by_mkt = {m: g.set_index('date')['mm_net_pct_oi']
                  for m, g in cot.groupby('market')}
    cot_by_mkt['basket'] = cot.groupby('date')['mm_net_pct_oi'].mean()

    stu = pd.read_csv(os.path.join(FUND, 'usda_stocks_to_use.csv'))
    stu_piv = stu.pivot_table(index='market_year', columns='commodity',
                              values='stocks_to_use')
    stu_z = ((stu_piv - stu_piv.rolling(5, min_periods=3).mean())
             / stu_piv.rolling(5, min_periods=3).std())
    stu_z['avg'] = stu_z.mean(axis=1)

    fao = pd.read_csv(os.path.join(FUND, 'fao_fpi.csv'), parse_dates=['date'])
    fao = fao.set_index('date')
    return panel, oni, cot_by_mkt, stu_z, fao


def daily(series, idx):
    return series.reindex(idx, method='ffill')


def build_sleeve(tk, panel, oni, cot_by_mkt, stu_z, fao):
    cfg = TICKER_CFG[tk]
    px = panel[tk].dropna()
    idx = px.index
    ret = px.pct_change().fillna(0)

    oni_d = daily(oni, idx)
    f = pd.DataFrame(index=idx)
    f['s_oni'] = (oni_d < 0) if cfg['oni_sign'] == 'low' else (oni_d > 0.5)

    cot_s = cot_by_mkt.get(cfg['cot'] or 'basket')
    cot_flow = (cot_s - cot_s.shift(13)).dropna()           # 13-wk flow
    cot_d = daily(cot_flow, idx)
    f['s_cot'] = cot_d < cot_d.rolling(ROLL, min_periods=252).quantile(0.2)
    hot = cot_d > cot_d.rolling(ROLL, min_periods=252).quantile(0.9)

    if cfg['stu']:
        sz = stu_z[cfg['stu']]
        # marketing year y known ~Oct of year y → map to daily
        sz_d = pd.Series({d: sz.get(d.year if d.month >= 10 else d.year - 1)
                          for d in idx})
        f['s_stu'] = sz_d < sz_d.rolling(ROLL, min_periods=252).quantile(0.3)
    else:
        f['s_stu'] = False

    fpi = fao[cfg['fpi']].astype(float)
    fpi_d = daily(fpi, idx).shift(42)                       # 2mo lag
    fpi_mom = fpi_d.pct_change(63)
    f['s_fpi'] = fpi_mom > fpi_mom.rolling(ROLL, min_periods=252).quantile(0.8)

    score = f[['s_oni', 's_cot', 's_stu', 's_fpi']].sum(axis=1)

    # macro-squeeze warnings (timing trim, validated by forensics)
    md = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(THIS_DIR)),
                                  'backtest', 'cache', 'master_dataset.csv'),
                     index_col=0, parse_dates=True)
    _i = pd.to_datetime(md.index, utc=True).tz_localize(None)
    md.index = _i.normalize()
    md = md.groupby(md.index).first()
    warn = pd.Series(0, index=idx, dtype=float)
    if 'DX_DXY' in md.columns:
        warn = warn.add((md['DX_DXY'].pct_change(63) > 0.02)
                        .reindex(idx).ffill().fillna(False).astype(int), fill_value=0)
    if 'CL' in md.columns:
        warn = warn.add((md['CL'].pct_change(63) > 0.10)
                        .reindex(idx).ffill().fillna(False).astype(int), fill_value=0)
    warn = warn.add(hot.fillna(False).astype(int), fill_value=0)

    # weight: monthly rebalance, signals lagged 1 more day
    w = score.map(W_BY_SCORE).fillna(0)
    w[warn >= 2] = w[warn >= 2] * 0.5
    w = w.shift(1).fillna(0)
    w = w.iloc[::21].reindex(idx).ffill().fillna(0)         # monthly hold

    sleeve = w * ret + (1 - w.clip(upper=1.0)) * BOXX_D
    return sleeve, w, score


def stats(r, label):
    r = r.dropna()
    if len(r) < 200 or r.std() == 0:
        return None
    nav = (1 + r).cumprod()
    yrs = (r.index[-1] - r.index[0]).days / 365.25
    w12 = nav.pct_change(252).dropna()
    return {'sleeve': label,
            'ann': round(nav.iloc[-1] ** (1 / yrs) - 1, 4),
            'sharpe': round(r.mean() / r.std() * math.sqrt(252), 3),
            'mdd': round((nav / nav.cummax() - 1).min(), 4),
            'worst12mo': round(w12.min(), 4) if len(w12) else None}


def main():
    panel, oni, cot_by_mkt, stu_z, fao = load_inputs()
    rows, sleeves, weights = [], {}, {}
    start = '2013-01-01'
    for tk in TICKER_CFG:
        if tk not in panel.columns:
            continue
        sleeve, w, score = build_sleeve(tk, panel, oni, cot_by_mkt, stu_z, fao)
        sleeve = sleeve.loc[start:]
        sleeves[tk] = sleeve
        weights[tk] = w
        r = stats(sleeve, f'{tk} directional')
        bh = stats(panel[tk].pct_change().loc[start:], f'{tk} buy&hold')
        if r:
            rows.append(r)
        if bh:
            rows.append(bh)

    # equal-weight blend of sleeves (cash overlap earns BOXX inside sleeves)
    common = None
    for s in sleeves.values():
        common = s.index if common is None else common.intersection(s.index)
    blend = sum(s.reindex(common).fillna(0) for s in sleeves.values()) / len(sleeves)
    rows.append(stats(blend, 'EQUAL-WEIGHT BLEND'))
    rows.append({'sleeve': 'BOXX', 'ann': 0.0474, 'sharpe': None,
                 'mdd': 0.0, 'worst12mo': 0.0474})

    df = pd.DataFrame([r for r in rows if r])
    print(f'\n=== DIRECTIONAL AG SLEEVES (walk-forward, {start} → now) ===')
    print(df.to_string(index=False))
    df.to_csv(os.path.join(CACHE, 'directional_blend_results.csv'), index=False)

    # live state
    live = {}
    for tk, w in weights.items():
        live[tk] = {'weight_now': round(float(w.iloc[-1]), 2)}
    print('\ncurrent weights:', json.dumps(live))
    with open(os.path.join(CACHE, 'directional_state.json'), 'w') as fh:
        json.dump({'as_of': str(common[-1].date()), 'weights': live}, fh, indent=2)


if __name__ == '__main__':
    main()
