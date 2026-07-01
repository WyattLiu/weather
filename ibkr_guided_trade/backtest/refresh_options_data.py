"""ONE-COMMAND daily refresh of all UNG options data from ThetaData (authoritative source).

Runs the three ingestions in order, each AUTO-RESUMING from its last ingested date through
today — so the pipeline never silently stops at a hardcoded date again (that froze the feed
at 2026-06-12 and made the engine misprice off stale IV). Safe to run repeatedly / on a
schedule; every step is idempotent and a no-op when already current.

  venv/bin/python backtest/refresh_options_data.py            # IV surface + minute quotes + OI
  venv/bin/python backtest/refresh_options_data.py --no-oi    # skip the (slow) OI pass

Order matters: the master dataset must have today's UNG spot first (the live dashboard's
refresh_to_today does that from ThetaData), since the per-day ingestion reads the spot from it.
"""
import argparse
import subprocess
import sys
import os

THIS = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def _run(label, args):
    print(f"\n{'='*70}\n[{label}]\n{'='*70}", flush=True)
    r = subprocess.run([PY, os.path.join(THIS, args[0])] + args[1:])
    print(f"[{label}] exit {r.returncode}", flush=True)
    return r.returncode


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--no-oi', action='store_true', help='skip the slow daily-OI pass')
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    rc = 0
    # 0) master dataset → today's UNG spot from ThetaData EOD (after close it includes today),
    #    so the per-day ingestion below has a spot for every new session. Self-contained: a
    #    cron run doesn't depend on the live dashboard having stamped today's row.
    try:
        sys.path.insert(0, THIS)
        from historical_data_pipeline import refresh_to_today
        refresh_to_today(persist=True)
        print('[master dataset] refreshed to today from ThetaData', flush=True)
    except Exception as e:
        print(f'[master dataset] refresh skipped ({e!r})', flush=True)
    # 0b) backfill BOXX pre-inception gap (synthesize from BIL T-bill accrual). Idempotent no-op once
    #     complete; guards against a full rebuild re-introducing the flat-117 default + fake seam crash.
    try:
        from backfill_boxx import backfill as _backfill_boxx
        _backfill_boxx()
    except Exception as e:
        print(f'[boxx] backfill skipped ({e!r})', flush=True)
    # 1) IV surface (resume from last surface date → today)
    rc |= _run('IV surface', ['backfill_ung_iv_pg.py', '--workers', str(a.workers)])
    # 1b) iv_rank signal (252d percentile of ATM IV) — recompute from the just-refreshed surface.
    #     VALIDATED live signal (iv_rank_z_scale); previously had NO refresh path and froze at
    #     2026-06-12, silently dropping it from live decisions (NaN → neutral) after ffill(limit=10).
    try:
        sys.path.insert(0, THIS)
        from refresh_iv_rank import refresh as _refresh_iv_rank
        _refresh_iv_rank()
    except Exception as e:
        print(f'[iv_rank] refresh skipped ({e!r})', flush=True)
    # 2) minute quotes (resume from last trade_date → today, 1m)
    rc |= _run('Minute quotes', ['backfill_ung_intraday.py', '--workers', str(a.workers)])
    # 3) open interest (per-contract latest)
    if not a.no_oi:
        rc |= _run('Open interest', ['backfill_ung_oi.py'])
    print(f"\n{'='*70}\nDONE (overall exit {rc})", flush=True)
    sys.exit(rc)


if __name__ == '__main__':
    main()
