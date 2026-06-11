"""Fetch non-weather DBA fundamentals.

1. CFTC COT disaggregated (weekly, 2010+): managed-money net positioning
   for the DBA basket futures. Factor: positioning extremes mean-revert.
2. FAO Food Price Index (monthly): global food price momentum.
3. USDA PSD stocks-to-use (marketing-year): world corn/wheat/soy
   ending-stocks / consumption — the ag analog of EIA storage.
   CAVEAT: PSD download is CURRENT estimates (final revisions), not
   vintage-as-published — scan results carry revision lookahead bias.
4. Crude (CL) — already in backtest/cache/master_dataset.csv.

Run: venv/bin/python research/dba/fundamentals_fetch.py
"""
import os
import io
import re
import zipfile
import subprocess
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(THIS_DIR, 'cache')
FUND = os.path.join(CACHE, 'fundamentals')
os.makedirs(FUND, exist_ok=True)

# DBA basket markets in COT disaggregated naming
COT_MARKETS = {
    'CORN': 'corn', 'SOYBEANS': 'soybeans', 'WHEAT-SRW': 'wheat',
    'SUGAR NO. 11': 'sugar', 'COFFEE C': 'coffee', 'COCOA': 'cocoa',
    'LIVE CATTLE': 'cattle', 'LEAN HOGS': 'hogs',
}


def _curl(url, dest=None, timeout=120):
    args = ['curl', '-s', '-L', '--max-time', str(timeout), '-A', 'Mozilla/5.0', url]
    if dest:
        args += ['-o', dest]
        subprocess.run(args, check=True)
        return dest
    return subprocess.check_output(args)


def fetch_cot(start_year=2010, end_year=None):
    """Managed-money net positioning per ag market, weekly.
    Current-year zip refreshes if older than 6 days (CFTC posts Fridays)."""
    import time as _t
    from datetime import date as _date
    if end_year is None:
        end_year = _date.today().year
    out_path = os.path.join(FUND, 'cot_ag.csv')
    frames = []
    for yr in range(start_year, end_year + 1):
        zp = os.path.join(FUND, f'cot_{yr}.zip')
        if (yr == end_year and os.path.exists(zp)
                and _t.time() - os.path.getmtime(zp) > 6 * 86400):
            os.remove(zp)  # stale current-year file → re-download
        if not os.path.exists(zp):
            print(f'[cot] {yr} downloading...')
            try:
                _curl(f'https://www.cftc.gov/files/dea/history/fut_disagg_txt_{yr}.zip', zp)
            except Exception as e:
                print(f'  {yr}: failed ({e})')
                continue
        try:
            with zipfile.ZipFile(zp) as z:
                name = z.namelist()[0]
                df = pd.read_csv(z.open(name), low_memory=False)
        except Exception as e:
            print(f'  {yr}: parse failed ({e})')
            continue
        df.columns = [c.strip() for c in df.columns]
        mcol = 'Market_and_Exchange_Names'
        dcol = ('Report_Date_as_YYYY-MM-DD' if 'Report_Date_as_YYYY-MM-DD' in df.columns
                else 'Report_Date_as_MM_DD_YYYY')
        for key, label in COT_MARKETS.items():
            sub = df[df[mcol].str.upper().str.startswith(key)]
            if sub.empty:
                continue
            mm_net = (sub['M_Money_Positions_Long_All']
                      - sub['M_Money_Positions_Short_All'])
            oi = sub['Open_Interest_All'].replace(0, pd.NA)
            frames.append(pd.DataFrame({
                'date': pd.to_datetime(sub[dcol]),
                'market': label,
                'mm_net_pct_oi': (mm_net / oi).astype(float),
            }))
    cot = pd.concat(frames, ignore_index=True).dropna()
    cot = cot.sort_values('date')
    cot.to_csv(out_path, index=False)
    print(f'[cot] {len(cot)} rows {cot["date"].min().date()} → '
          f'{cot["date"].max().date()} → {out_path}')
    return cot


def fetch_fao(month_hint=None):
    """FAO Food Price Index monthly. URL embeds the publication month —
    auto-discover by trying current month then walking back 3."""
    from datetime import date as _date
    out_path = os.path.join(FUND, 'fao_fpi.csv')
    xp = os.path.join(FUND, 'ffpi.xlsx')
    hints = [month_hint] if month_hint else []
    if not hints:
        d = _date.today().replace(day=1)
        for _ in range(4):
            hints.append(d.strftime('%Y-%m'))
            d = (d - pd.Timedelta(days=1)).replace(day=1)
    df = None
    for h in hints:
        url = (f'https://www.fao.org/media/docs/worldfoodsituationlibraries/'
               f'default-document-library/ffpi-data-{h}.xlsx')
        try:
            _curl(url, xp)
            if os.path.getsize(xp) < 10000:
                continue  # 404 page, not an xlsx
            df = pd.read_excel(xp, sheet_name=0, skiprows=2)
            print(f'[fao] got ffpi-data-{h}.xlsx')
            break
        except Exception:
            continue
    if df is None:
        print('[fao] all month hints failed')
        return None
    df = df.rename(columns={df.columns[0]: 'date'})
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    keep = [c for c in df.columns
            if str(c).lower() in ('date', 'food price index', 'meat', 'dairy',
                                  'cereals', 'oils', 'sugar')
            or 'index' in str(c).lower()]
    df = df[keep[:7]]
    df.to_csv(out_path, index=False)
    print(f'[fao] {len(df)} months → {out_path}')
    return df


def fetch_usda_stocks():
    """World stocks-to-use for corn/wheat/soybeans from USDA PSD."""
    out_path = os.path.join(FUND, 'usda_stocks_to_use.csv')
    zp = os.path.join(FUND, 'psd_grains.zip')
    if not os.path.exists(zp):
        print('[usda] downloading PSD grains...')
        _curl('https://apps.fas.usda.gov/psdonline/downloads/psd_grains_pulses_csv.zip', zp)
    with zipfile.ZipFile(zp) as z:
        name = [n for n in z.namelist() if n.endswith('.csv')][0]
        df = pd.read_csv(z.open(name), low_memory=False)
    # also soybeans live in the oilseeds file
    zp2 = os.path.join(FUND, 'psd_oilseeds.zip')
    if not os.path.exists(zp2):
        print('[usda] downloading PSD oilseeds...')
        _curl('https://apps.fas.usda.gov/psdonline/downloads/psd_oilseeds_csv.zip', zp2)
    try:
        with zipfile.ZipFile(zp2) as z:
            name = [n for n in z.namelist() if n.endswith('.csv')][0]
            df = pd.concat([df, pd.read_csv(z.open(name), low_memory=False)],
                           ignore_index=True)
    except Exception as e:
        print(f'[usda] oilseeds failed ({e}) — grains only')

    rows = []
    for commodity in ('Corn', 'Wheat', 'Oilseed, Soybean'):
        sub = df[df['Commodity_Description'] == commodity]
        if sub.empty:
            continue
        # world total = sum across countries per (year, attribute)
        piv = (sub.groupby(['Market_Year', 'Attribute_Description'])['Value']
               .sum().unstack())
        if 'Ending Stocks' not in piv.columns or 'Domestic Consumption' not in piv.columns:
            continue
        stu = piv['Ending Stocks'] / piv['Domestic Consumption'].replace(0, pd.NA)
        for yr, v in stu.dropna().items():
            rows.append({'market_year': int(yr), 'commodity': commodity,
                         'stocks_to_use': round(float(v), 4)})
    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    print(f'[usda] {len(out)} commodity-years → {out_path}')
    return out


if __name__ == '__main__':
    fetch_cot()
    fetch_fao()
    fetch_usda_stocks()
