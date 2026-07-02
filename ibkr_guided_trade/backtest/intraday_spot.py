"""Intraday UNG spot reconstruction via put-call parity (roadmap FiC).

The minute-reactive engine must react at the EIA print (10:30 ET Thursday) using the UNG spot AT THAT MINUTE.
PG has no intraday UNG ETF spot: etf_spot_minute holds only SPY/QQQ/IWM, and ung_options_history.underlying_price
is a DAILY-CONSTANT reference (verified: identical at 10:30 and 16:00), not an intraday series. But the option
minute bars DO carry the intraday move, so we recover the true intraday spot from put-call parity:

    S = C_mid - P_mid + K * exp(-r * T)

evaluated per strike and taken as the MEDIAN across near-ATM strikes for robustness. Cross-strike agreement
(empirically std ~0.1-0.6% of spot) is the built-in fidelity check that the number is a real price, not noise.

NO-LEAK: only option bars at bar_time == at_time (the event minute) are read; nothing later. A reconstruction
stamped 10:30 is a strict function of the 10:30 quotes, so it can never see the 16:00 (or any later) tape.
"""
import datetime as dt
import math

R_RATE = 0.045          # matches replay_engine bs_put/bs_call/exec_fill discounting

# UNG reverse splits. ung_options_history stores RAW (as-traded) prices/strikes; master_dataset['UNG']
# (and therefore the engine's spot_u) is BACK-ADJUSTED. A reconstruction from raw option quotes must be
# scaled UP by the split factor for pre-split dates or it feeds the engine a ~4x-too-small spot (found
# 2026-07-01: pre-2024 reactive spot was $12 raw vs $47 adjusted — corrupted every pre-split decision).
# Kept in sync with real_chain._SPLITS / intraday_fill._SPLITS.
try:
    from real_chain import _SPLITS as _SPLITS
except Exception:
    _SPLITS = [('2018-01-05', 4.0), ('2024-01-24', 4.0)]


def _raw_to_adjusted_factor(trade_date):
    """Multiplier converting a RAW options-scale price on `trade_date` to the back-adjusted master scale."""
    import pandas as pd
    f = 1.0
    for sd, fac in _SPLITS:
        if pd.Timestamp(trade_date) < pd.Timestamp(sd):
            f *= fac
    return f


def _median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return None
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def reconstruct_spot(trade_date, at_time, conn, r=R_RATE, min_strikes=3):
    """Robust intraday UNG spot at `at_time` (a datetime.time) on `trade_date` via near-ATM put-call parity,
    using the nearest expiration. Returns dict(spot, n, std, rel_std, expiration, dte) or None if the minute
    lacks enough paired quotes. Reads ONLY bars at bar_time == at_time — strictly causal to that instant."""
    if isinstance(at_time, str):
        h, m = at_time.split(':')[:2]
        at_time = dt.time(int(h), int(m))
    cur = conn.cursor()
    cur.execute("select min(expiration) from ung_options_history "
                "where trade_date=%s and expiration>trade_date", (trade_date,))
    row = cur.fetchone()
    if not row or row[0] is None:
        return None
    exp = row[0]
    T = max((exp - trade_date).days, 1) / 365.0
    DF = math.exp(-r * T)
    cur.execute("select strike, option_right, data_type, close from ung_options_history "
                "where trade_date=%s and bar_time::time=%s and expiration=%s",
                (trade_date, at_time, exp))
    px = {}
    for K, right, dtp, close in cur.fetchall():
        px.setdefault((float(K), right), {})[dtp] = float(close)
    recon = []
    for (K, right), v in px.items():
        if right != 'C':
            continue
        p = px.get((K, 'P'))
        if not p:
            continue
        if not (('BID' in v and 'ASK' in v) and ('BID' in p and 'ASK' in p)):
            continue
        cm = (v['BID'] + v['ASK']) / 2.0
        pm = (p['BID'] + p['ASK']) / 2.0
        if cm <= 0 and pm <= 0:                 # both legs empty → dead strike
            continue
        recon.append((K, cm - pm + K * DF))
    if len(recon) < min_strikes:
        return None
    rough = _median([s for _, s in recon])
    atm = [(K, s) for K, s in recon if abs(K - rough) <= 0.15 * rough]   # near-ATM = tightest parity
    use = atm if len(atm) >= min_strikes else recon
    spots = [s for _, s in use]
    spot = _median(spots)
    mean = sum(spots) / len(spots)
    std = (sum((x - mean) ** 2 for x in spots) / len(spots)) ** 0.5
    # RAW → back-adjusted (master) scale: without this the engine gets a ~4x-too-small pre-split spot.
    factor = _raw_to_adjusted_factor(trade_date)
    spot *= factor
    std *= factor
    return {'spot': spot, 'n': len(spots), 'std': std,
            'rel_std': (std / spot) if spot else None,   # scale-invariant (unchanged by factor)
            'expiration': exp, 'dte': (exp - trade_date).days}


def _connect():
    """Best-effort PG connection using the project's params; None if unavailable."""
    try:
        import psycopg2
        from backfill_ung_iv_pg import DB_PARAMS
        return psycopg2.connect(**DB_PARAMS, connect_timeout=6)
    except Exception:
        return None


def event_spot_map(dates, at_time='10:30', max_rel_std=0.02):
    """Batch-reconstruct the intraday event-moment spot for many `dates` in ONE connection (so the
    reactive engine can precompute before its day loop instead of hitting PG per-day).

    Returns {date: spot} containing ONLY dates where the reconstruction succeeded AND is reliable
    (rel_std <= max_rel_std — strikes agree). Dates without minute data / with scattered strikes are
    omitted, so the caller cleanly falls back to the daily spot (non-reactive, never a leak). Empty
    dict when PG is unavailable. Each value is a strict function of `at_time`'s quotes — causal."""
    if isinstance(at_time, str):
        h, m = at_time.split(':')[:2]
        at_time = dt.time(int(h), int(m))
    conn = _connect()
    if conn is None:
        return {}
    out = {}
    try:
        for d in dates:
            dd = d.date() if isinstance(d, dt.datetime) else d
            r = reconstruct_spot(dd, at_time, conn)
            if r and r.get('rel_std') is not None and r['rel_std'] <= max_rel_std:
                out[dd] = r['spot']
    finally:
        conn.close()
    return out
