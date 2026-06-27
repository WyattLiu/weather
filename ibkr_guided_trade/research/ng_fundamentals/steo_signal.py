"""EIA STEO Henry Hub forecast signal — vintage-true, keyless.

Baumeister et al. (NBER w33156): STEO expert forecasts beat the random
walk by 35-38% at 9-12 month horizons — the largest accuracy gain of
any method tested. This builds the signal from the monthly archive
XLSX files (real vintages, no revision lookahead):

  steo_prem_h = STEO forecast for month (vintage + h) / HH spot at
                vintage date - 1     (expected appreciation)

Then tests UNG forward returns by quintile of steo_prem_12m, and
writes the latest signal for live use.

Run: venv/bin/python research/ng_fundamentals/steo_signal.py
"""
import os
import subprocess
import json
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
STEO_DIR = os.path.join(THIS_DIR, 'steo')
os.makedirs(STEO_DIR, exist_ok=True)
DBA_CACHE = os.path.join(os.path.dirname(THIS_DIR), 'dba', 'cache')

MONTHS = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
          'jul', 'aug', 'sep', 'oct', 'nov', 'dec']


def fetch_archives(start_year=2017):
    from datetime import date
    today = date.today()
    got = 0
    for yr in range(start_year, today.year + 1):
        for mi, mon in enumerate(MONTHS, 1):
            if (yr, mi) > (today.year, today.month):
                break
            fname = f'{mon}{str(yr)[2:]}_base.xlsx'
            dest = os.path.join(STEO_DIR, fname)
            if os.path.exists(dest) and os.path.getsize(dest) > 100000:
                continue
            url = f'https://www.eia.gov/outlooks/steo/archives/{fname}'
            subprocess.run(['curl', '-s', '--max-time', '60', '-o', dest, url],
                           check=False)
            if os.path.getsize(dest) < 100000:
                os.remove(dest)   # 404 page
            else:
                got += 1
    print(f'[steo] fetched {got} new archives; total '
          f'{len(os.listdir(STEO_DIR))} on disk')


def parse_archive(path):
    """Return (vintage_date, Series[month → HH $/MMBtu forecast])."""
    df = pd.read_excel(path, sheet_name='2tab', header=None)
    years = df.iloc[2].ffill()
    months = df.iloc[3]
    hh_row = None
    for i in range(len(df)):
        cell = str(df.iloc[i, 1])
        if 'Henry Hub Spot (dollars per million Btu)' in cell:
            hh_row = i
            break
    if hh_row is None:
        return None, None
    out = {}
    for c in range(2, df.shape[1]):
        try:
            mon = str(months[c]).strip()[:3]
            if mon.lower() not in MONTHS:
                continue
            ts = pd.Timestamp(int(float(years[c])), MONTHS.index(mon.lower()) + 1, 1)
            v = float(df.iloc[hh_row, c])
            out[ts] = v
        except (ValueError, TypeError):
            continue
    name = os.path.basename(path)               # e.g. jun26_base.xlsx
    vint = pd.Timestamp(2000 + int(name[3:5]), MONTHS.index(name[:3]) + 1, 1)
    return vint, pd.Series(out).sort_index()


def build_vintage_panel():
    rows = []
    for f in sorted(os.listdir(STEO_DIR)):
        if not f.endswith('_base.xlsx'):
            continue
        try:
            vint, ser = parse_archive(os.path.join(STEO_DIR, f))
        except Exception as e:
            print(f'  {f}: parse failed ({e})')
            continue
        if ser is None:
            continue
        for h in (3, 6, 9, 12):
            tgt = vint + pd.DateOffset(months=h)
            if tgt in ser.index:
                rows.append({'vintage': vint, 'h': h, 'forecast': ser[tgt]})
        # nowcast month value = "current" anchor in same vintage
        if vint in ser.index:
            rows.append({'vintage': vint, 'h': 0, 'forecast': ser[vint]})
    pan = pd.DataFrame(rows)
    pan.to_csv(os.path.join(THIS_DIR, 'steo_vintage_panel.csv'), index=False)
    print(f'[steo] vintage panel: {pan["vintage"].nunique()} vintages')
    return pan


def main():
    fetch_archives()
    pan = build_vintage_panel()
    piv = pan.pivot_table(index='vintage', columns='h', values='forecast')

    # expected appreciation: forecast(h) / nowcast(0) - 1
    sig = pd.DataFrame({f'steo_prem_{h}m': piv[h] / piv[0] - 1
                        for h in (3, 6, 9, 12) if h in piv.columns})
    # available ~mid-month → shift availability by 1 month to be safe
    sig.index = sig.index + pd.DateOffset(months=1)

    panel = pd.read_csv(os.path.join(DBA_CACHE, 'master_panel.csv'),
                        index_col=0, parse_dates=True)
    ung = panel['UNG'].dropna()
    from scipy import stats as sstats
    print('\n=== STEO premium vs UNG forward returns (quintiles, monthly) ===')
    results = []
    for col in sig.columns:
        s_d = sig[col].reindex(ung.index, method='ffill')
        for fwd_n, fwd_lab in ((21, 'fwd21'), (63, 'fwd63'), (126, 'fwd126')):
            fwd = ung.pct_change(fwd_n).shift(-fwd_n)
            f = pd.DataFrame({'sig': s_d, 'fwd': fwd}).dropna().iloc[::21]
            if len(f) < 40:
                continue
            try:
                f['q'] = pd.qcut(f['sig'], 5, labels=False, duplicates='drop')
            except ValueError:
                continue
            top = f[f['q'] == f['q'].max()]['fwd']
            bot = f[f['q'] == f['q'].min()]['fwd']
            t, p = sstats.ttest_ind(top, bot, equal_var=False)
            results.append({'signal': col, 'fwd': fwd_lab,
                            'spread': round(top.mean() - bot.mean(), 4),
                            't': round(t, 2), 'p': round(p, 3), 'n': len(f)})
    res = pd.DataFrame(results).sort_values('p')
    print(res.to_string(index=False))
    res.to_csv(os.path.join(THIS_DIR, 'steo_scan.csv'), index=False)

    latest = sig.iloc[-1]
    state = {'as_of': str(sig.index[-1].date()),
             **{c: round(float(latest[c]), 4) for c in sig.columns}}
    with open(os.path.join(THIS_DIR, 'steo_state.json'), 'w') as f:
        json.dump(state, f, indent=2)
    print('\nlive STEO state:', json.dumps(state))


if __name__ == '__main__':
    main()
