"""NOAA CPC ENSO probability + strength fetcher.

Pulls the official CPC ENSO outlook (updated monthly ~2nd Thursday) from:
  https://cpc.ncep.noaa.gov/products/analysis_monitoring/enso/roni/probabilities/
  https://cpc.ncep.noaa.gov/products/analysis_monitoring/enso/roni/strengths/
  https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso_advisory/ensodisc.txt

Output:
  cache/enso_outlook.json — structured forecast {season → {la_nina, neutral, el_nino}}
  cache/enso_outlook.csv  — same data, tabular for backtest/research use

Run: venv/bin/python research/dba/fetch_enso_plume.py
"""
import os
import re
import json
import subprocess
import pandas as pd
from datetime import date, timedelta

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, 'cache')
os.makedirs(CACHE, exist_ok=True)

PROB_URL = 'https://cpc.ncep.noaa.gov/products/analysis_monitoring/enso/roni/probabilities/'
STR_URL = 'https://cpc.ncep.noaa.gov/products/analysis_monitoring/enso/roni/strengths/'
DISC_URL = 'https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso_advisory/ensodisc.txt'

SEAS_ORDER = ['DJF', 'JFM', 'FMA', 'MAM', 'AMJ', 'MJJ',
              'JJA', 'JAS', 'ASO', 'SON', 'OND', 'NDJ']
SEAS_MID_MO = {'DJF': 1, 'JFM': 2, 'FMA': 3, 'MAM': 4, 'AMJ': 5, 'MJJ': 6,
               'JJA': 7, 'JAS': 8, 'ASO': 9, 'SON': 10, 'OND': 11, 'NDJ': 12}


def _curl(url, ua='Mozilla/5.0'):
    return subprocess.check_output(
        ['curl', '-s', '--max-time', '30', '-L', '-A', ua, url]
    ).decode(errors='ignore')


_MO_NAME_TO_NUM = {'january': 1, 'february': 2, 'march': 3, 'april': 4,
                   'may': 5, 'june': 6, 'july': 7, 'august': 8,
                   'september': 9, 'october': 10, 'november': 11, 'december': 12}


def _parse_issue_date(disc_text):
    """Extract issue date from discussion text (e.g., '14 May 2026')."""
    m = re.search(r'(\d{1,2})\s+([A-Z][a-z]+)\s+(\d{4})', disc_text)
    if not m:
        return None
    day, mo_name, yr = m.groups()
    mo_num = _MO_NAME_TO_NUM.get(mo_name.lower())
    if not mo_num:
        return None
    return date(int(yr), mo_num, int(day))


def _start_season_from_issue(issue_dt):
    """The first probability row in the CPC table is the 3-month season
    whose RIGHT edge ends in the issue month. e.g., issued mid-May →
    first row is AMJ (Apr-May-Jun, center=May). The seasons are listed by
    their CENTER month so AMJ has center=May=5."""
    if issue_dt is None:
        # Fallback: assume first row = current month - 1
        today = date.today()
        seas = SEAS_ORDER[today.month - 2 if today.month > 1 else 11]
        return seas, today.year
    # Issue month = center month of first season
    seas = SEAS_ORDER[issue_dt.month - 1]
    return seas, issue_dt.year


def fetch_probabilities(issue_dt=None):
    """Parse 9-season × 3-category probability table from CPC ROI page."""
    html = _curl(PROB_URL)
    # Cells appear as <td>NN</td> inside #probabilities-table
    cells = re.findall(r'<td[^>]*>(\d+)</td>', html)
    if len(cells) < 27:
        raise RuntimeError(f'expected ≥27 prob cells, got {len(cells)}')
    cells = [int(c) for c in cells[:27]]  # 9 seasons × 3 cats

    start_seas, start_yr = _start_season_from_issue(issue_dt)
    si = SEAS_ORDER.index(start_seas)
    rows = []
    for i in range(9):
        seas = SEAS_ORDER[(si + i) % 12]
        yr = start_yr + (1 if (si + i) >= 12 else 0)
        la_nina, neutral, el_nino = cells[3*i:3*i+3]
        rows.append({
            'season': seas, 'year': yr,
            'season_label': f'{seas} {yr}',
            'la_nina_pct': la_nina,
            'neutral_pct': neutral,
            'el_nino_pct': el_nino,
        })
    return rows


def fetch_strengths():
    """Parse strength categorization table.
    Format: rows are seasons, columns are strength categories
    (Weak / Moderate / Strong / Very Strong) - probabilities."""
    html = _curl(STR_URL)
    cells = re.findall(r'<td[^>]*>(\d+)</td>', html)
    if not cells:
        return None
    return [int(c) for c in cells]  # raw — schema may vary; preserve for now


def fetch_discussion():
    """Pull the monthly text discussion to extract headline probabilities."""
    txt = _curl(DISC_URL)
    # Look for "X% chance in <season>" patterns
    matches = re.findall(
        r'(\d{1,3})%\s+chance\s+in\s+([A-Z][a-z]+(?:-[A-Z][a-z]+)?\s+\d{4}(?:-\d{2,4})?)',
        txt
    )
    headlines = [{'pct': int(p), 'window': w} for p, w in matches]
    # Status: "ENSO Alert System Status: <status>"
    status_match = re.search(r'ENSO Alert System Status:\s*(.+)', txt)
    status = status_match.group(1).strip() if status_match else 'Unknown'
    # Issue date
    date_match = re.search(r'(\d{1,2}\s+[A-Z][a-z]+\s+\d{4})', txt)
    issue_date = date_match.group(1) if date_match else None
    return {'status': status, 'issue_date': issue_date, 'headlines': headlines}


def build_outlook():
    print(f'[plume] fetching CPC ENSO outlook...')
    disc = fetch_discussion()
    issue_dt = _parse_issue_date(disc['issue_date'] or '')
    probs = fetch_probabilities(issue_dt=issue_dt)
    strengths = fetch_strengths()

    # Convert to forecast date map: middle-month → probability of El Niño
    forecast_index = []
    for r in probs:
        mid_mo = SEAS_MID_MO[r['season']]
        mid_date = date(r['year'], mid_mo, 15)
        forecast_index.append({
            'forecast_date': str(mid_date),
            'season_label': r['season_label'],
            'el_nino_pct': r['el_nino_pct'],
            'neutral_pct': r['neutral_pct'],
            'la_nina_pct': r['la_nina_pct'],
        })

    # Headline summary
    max_en = max(probs, key=lambda r: r['el_nino_pct'])
    peak_window = max_en['season_label']
    peak_pct = max_en['el_nino_pct']

    # Probability El Niño persists through DJF 2026/27 (proxy for full winter)
    djf = [r for r in probs if r['season'] == 'DJF']
    djf_pct = djf[0]['el_nino_pct'] if djf else None

    out = {
        'issue_date': disc['issue_date'],
        'status': disc['status'],
        'discussion_headlines': disc['headlines'],
        'forecast': forecast_index,
        'summary': {
            'peak_el_nino_pct': peak_pct,
            'peak_window': peak_window,
            'djf_el_nino_pct': djf_pct,
            'months_above_80pct': sum(1 for r in probs if r['el_nino_pct'] >= 80),
            'months_above_95pct': sum(1 for r in probs if r['el_nino_pct'] >= 95),
        },
        'strength_raw': strengths,  # preserve until we know the schema
    }

    json_path = os.path.join(CACHE, 'enso_outlook.json')
    with open(json_path, 'w') as f:
        json.dump(out, f, indent=2)

    df = pd.DataFrame(forecast_index)
    csv_path = os.path.join(CACHE, 'enso_outlook.csv')
    df.to_csv(csv_path, index=False)

    print(f'[plume] saved → {json_path}')
    print(f'[plume] saved → {csv_path}')
    return out


def main():
    out = build_outlook()
    print(f'\n=== CPC ENSO Outlook (issued {out["issue_date"]}) ===')
    print(f'Status: {out["status"]}')
    print(f'\nForecast plume (El Niño probability by season):')
    for r in out['forecast']:
        bar = '█' * (r['el_nino_pct'] // 5)
        print(f'  {r["season_label"]:>11s}  {r["el_nino_pct"]:>3d}%  {bar}')
    s = out['summary']
    print(f'\nSummary:')
    print(f'  Peak El Niño: {s["peak_el_nino_pct"]}% in {s["peak_window"]}')
    print(f'  DJF 2026/27 (winter): {s["djf_el_nino_pct"]}% El Niño')
    print(f'  Months ≥80%: {s["months_above_80pct"]} / 9')
    print(f'  Months ≥95%: {s["months_above_95pct"]} / 9')
    if out['discussion_headlines']:
        print(f'\nText discussion headlines:')
        for h in out['discussion_headlines']:
            print(f'  {h["pct"]}% — {h["window"]}')


if __name__ == '__main__':
    main()
