"""Daily option-chain snapshot — builds the historical GEX dataset.

GEX (dealer gamma exposure) needs per-strike OI history, which no free
source provides (dolthub options DB has greeks but no OI). So we collect
it ourselves: every close, snapshot the full UNG + DBA chains (all near
expiries, both rights) and append to a per-symbol CSV.

After ~3-6 months this becomes backtestable: wall-pinning, GEX-flip
regime shifts, expiry-week magnetism.

NOTE: WS openInterest is the OCC prior-day settled value (OI updates
overnight), so a snapshot taken after today's close carries this
morning's official OI. That is the standard convention GEX vendors use.

Cron: 5 18 * * 1-5 (after close, before dba refresh kicks off)
"""
import os
import sys
import csv
import time
from datetime import date, timedelta

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'live'))

from ws_option_resolver import fetch_chain  # noqa: E402

HIST_DIR = os.path.join(THIS_DIR, 'history')
os.makedirs(HIST_DIR, exist_ok=True)

SYMBOLS = ['UNG', 'DBA']
LOOKAHEAD_DAYS = 80


def candidate_expiries(today=None):
    """Fridays in the next LOOKAHEAD_DAYS, plus the Thursday before each
    (holiday-shifted expiries like Juneteenth 2026-06-18)."""
    today = today or date.today()
    out = []
    d = today
    while (d - today).days <= LOOKAHEAD_DAYS:
        if d.weekday() == 4:  # Friday
            out.append(d)
            out.append(d - timedelta(days=1))  # Thursday fallback
        d += timedelta(days=1)
    return [x.isoformat() for x in out]


def _already_snapped(path, run_date):
    """Same-day guard: don't duplicate rows if cron re-fires."""
    if not os.path.exists(path):
        return False
    with open(path, 'rb') as f:
        try:
            f.seek(-200, os.SEEK_END)
        except OSError:
            f.seek(0)
        last = f.read().decode(errors='ignore').strip().split('\n')[-1]
    return last.startswith(run_date)


def snapshot_symbol(symbol, run_date):
    path = os.path.join(HIST_DIR, f'{symbol.lower()}_chain_history.csv')
    if _already_snapped(path, run_date):
        return 0, []
    new_file = not os.path.exists(path)
    rows_written = 0
    expiries_found = []
    with open(path, 'a', newline='') as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(['snap_date', 'expiry', 'right', 'strike',
                        'bid', 'ask', 'last', 'oi'])
        for expiry in candidate_expiries():
            got_any = False
            for right in ('C', 'P'):
                try:
                    chain = fetch_chain(symbol, expiry, right)
                except Exception:
                    chain = {}
                if not chain:
                    continue
                got_any = True
                for K, leg in sorted(chain.items()):
                    w.writerow([run_date, expiry, right, K,
                                leg['bid'], leg['ask'], leg['last'], leg['oi']])
                    rows_written += 1
                time.sleep(0.3)  # be polite to WS
            if got_any:
                expiries_found.append(expiry)
    return rows_written, expiries_found


def main():
    run_date = date.today().isoformat()
    for sym in SYMBOLS:
        n, exps = snapshot_symbol(sym, run_date)
        print(f'[gex-snap] {sym}: {n} rows across {len(exps)} expiries '
              f'({", ".join(exps[:4])}{"..." if len(exps) > 4 else ""})')


if __name__ == '__main__':
    main()
