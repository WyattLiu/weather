"""Pull NOAA CPC daily population/gas-weighted degree days (the HISTORICAL archive that
answers 'how do we get historical weather data') and test as a leading UNG signal.

Source: https://ftp.cpc.ncep.noaa.gov/htdocs/degree_days/weighted/daily_data/<year>/
  UtilityGas.Heating.txt  — gas-customer-weighted HDD by census division (the GWDD gas
                            traders use), daily, archive back to 1981.
  Population.Cooling.txt  — population-weighted CDD by census division.
National = census-division population-weighted sum. ACTUALS only (forecast vintages are
not freely archived — paid provider DTN/CWG, or self-archive the CPC forecast going forward).

Findings (2021-2026, 1992 daily obs):
  - DD anomaly is ORTHOGONAL to storage_surprise_z (corr -0.03) — genuinely new info.
  - It LEADS the storage print: corr with next-week storage_surprise_z -0.12 (correct
    sign: high demand -> tighter storage). The weather->storage chain confirmed.
  - Direct UNG prediction WEAK + CONTRARIAN: fwd-10d corr -0.11 (demand spikes = local
    tops -> fade), decaying to -0.04 by 42d. Consistent with the sell-the-spike finding.
  => Best use: NOWCAST the storage-surprise regime ~1wk early (latency edge on a trusted
     signal), NOT a standalone alpha. Raw -0.11 too weak to wire directly (overfit risk).
"""
import subprocess
import pandas as pd

W = {1: .044, 2: .124, 3: .140, 4: .064, 5: .204, 6: .058, 7: .125, 8: .077, 9: .164}
BASE = "https://ftp.cpc.ncep.noaa.gov/htdocs/degree_days/weighted/daily_data"


def _pull(year, kind):
    txt = subprocess.run(['curl', '-s', '-m', '30', f"{BASE}/{year}/{kind}.txt"],
                         capture_output=True, text=True).stdout
    lines = [l for l in txt.splitlines() if '|' in l]
    dates = lines[0].split('|')[1:]
    rows = {}
    for l in lines[1:]:
        p = l.split('|')
        try:
            r = int(p[0])
        except ValueError:
            continue
        if r in W:
            rows[r] = pd.to_numeric(pd.Series(p[1:]), errors='coerce').values
    df = pd.DataFrame(rows, index=pd.to_datetime(dates, format='%Y%m%d', errors='coerce'))
    return sum(df[r] * W[r] for r in df.columns) / sum(W[r] for r in df.columns)


def national_degree_days(y0=2021, y1=2027):
    hdd = pd.concat([_pull(y, 'UtilityGas.Heating') for y in range(y0, y1)])
    cdd = pd.concat([_pull(y, 'Population.Cooling') for y in range(y0, y1)])
    dd = (hdd.fillna(0) + cdd.fillna(0)).dropna()
    return dd[~dd.index.duplicated()].sort_index()


if __name__ == '__main__':
    dd = national_degree_days()
    dd.to_csv('ibkr_guided_trade/backtest/cache/noaa_degree_days_daily.csv')
    print(f"NOAA daily degree days: {len(dd)} days, {dd.index.min().date()} → {dd.index.max().date()}")
