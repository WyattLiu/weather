#!/usr/bin/env python3
"""
UNG Portfolio Visualizer - Interactive web-based dashboard
Run: python ung_visualizer.py
Then open http://localhost:8080
"""

import http.server
import json
import urllib.parse
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, date, timedelta

import numpy as np
import pandas as pd
from scipy.stats import norm
import yfinance as yf

# Historical UNG stress scenarios (backtested from 10yr data)
STRESS_SCENARIOS = {
    '5d_crash': -0.117,      # 5th percentile weekly
    '5d_mild_drop': -0.048,  # 25th percentile
    '5d_flat': -0.005,       # median (contango drag)
    '5d_rally': 0.039,       # 75th percentile
    '5d_spike': 0.121,       # 95th percentile
}

# Cycle 175: tried fork-based process pool (shinobi memmap pattern) but
# the IPC overhead per candidate + fork CoW page faults negated the
# multi-core gain — still ~12s. The bottleneck is generate_candidates
# being called 24x (3 paths × 8 steps), not per-candidate eval time.
# Keeping ThreadPool which handles the GIL-releasing scipy path well.
_QUALITY_POOL = ThreadPoolExecutor(max_workers=40, thread_name_prefix='quality')


def _eval_candidate(p_state, c, spot, iv, today, initial_quality):
    try:
        new_state = apply_trade_to_state(dict(p_state), c, spot, iv, today)
        new_q_dict = evaluate_portfolio_quality(new_state)
        # Cycle 180: hard DD veto — if adding this trade pushes the raw
        # 5%-CVaR tail loss past -15% of capital, return -inf (trade blocked).
        if new_q_dict.get('hard_dd_veto', False):
            return (-float('inf'), None, new_q_dict, c)
        new_q_total = new_q_dict.get('total', 0.0)
        qd = new_q_total - initial_quality

        # Cycle 199: synthetic early-assignment locked-gain bonus.
        # When closing a deep ITM short call AND selling the covering shares
        # at spot above strike, the (spot - strike) × shares gain is realized
        # cash that the standard quality evaluator misses (it only sees lost
        # theta + lost delta). Add the locked gain as a one-time bonus so the
        # beam can compare apples-to-apples vs letting nature take its course.
        if c.get('type') == 'CLOSE' and c.get('source_right') == 'C':
            _shares_sold = c.get('shares_sold', 0)
            _src_strike = c.get('source_strike', 0)
            if _shares_sold > 0 and spot > _src_strike:
                _locked_gain = (spot - _src_strike) * _shares_sold
                qd += _locked_gain

        return (qd, new_state, new_q_dict, c)
    except Exception:
        return (-float('inf'), None, None, c)

# Seasonal drift + vol scale caches — calibrated from UNG history.
# See CENTRAL_PHILOSOPHY.md "Cyclicality is the spine".
_seasonal_drift_cache = None
_seasonal_drift_cache_date = None
_seasonal_vol_cache = None
_seasonal_vol_cache_date = None


def get_seasonal_drift_vector(force_refresh=False):
    """Return dict {month_int (1-12): excess_per_day_log_return}.

    Excess = median per-day log return for that calendar month MINUS
    the overall median per-day log return. Captures the *seasonal residual*
    after average UNG drift (which is already modeled as contango_per_day in
    ScenarioDistribution). Sum across 12 months ≈ 0 by construction.

    Median used (not mean) to be robust to crisis spikes; winsorize 5/95%
    at the monthly-return level before aggregation.

    Cached per-process; refreshed daily.
    """
    global _seasonal_drift_cache, _seasonal_drift_cache_date
    import datetime as _dt
    today_ = _dt.date.today()
    if not force_refresh and _seasonal_drift_cache is not None \
            and _seasonal_drift_cache_date == today_:
        return _seasonal_drift_cache

    fallback = {m: 0.0 for m in range(1, 13)}
    try:
        df = yf.download('UNG', start='2010-01-01', progress=False, auto_adjust=False)
        if df is None or len(df) < 200:
            print("seasonal_drift: insufficient history, using zeros")
            _seasonal_drift_cache = fallback
            _seasonal_drift_cache_date = today_
            return fallback
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        # Use Adj Close if available, else Close
        price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
        df['ret'] = np.log(df[price_col] / df[price_col].shift(1))
        df['month'] = df['Date'].dt.month
        # Winsorize daily returns at 5/95% to mute crisis spikes
        lo, hi = df['ret'].quantile([0.05, 0.95])
        df['ret_w'] = df['ret'].clip(lo, hi)
        overall_median = df['ret_w'].median()
        result = {}
        for m in range(1, 13):
            month_med = df[df['month'] == m]['ret_w'].median()
            if bool(pd.isna(month_med)):
                result[m] = 0.0
            else:
                result[m] = float(month_med - overall_median)
        # Center so the 12-vector sums to ~0 exactly (defensive)
        avg = sum(result.values()) / 12.0
        result = {k: v - avg for k, v in result.items()}
        _seasonal_drift_cache = result
        _seasonal_drift_cache_date = today_
        # Sanity report
        sample_n = len(df)
        print(f"seasonal_drift calibrated from {sample_n} bars "
              f"({df['Date'].min().date()} to {df['Date'].max().date()}):")
        month_names = ['Jan','Feb','Mar','Apr','May','Jun',
                       'Jul','Aug','Sep','Oct','Nov','Dec']
        for m in range(1, 13):
            print(f"  {month_names[m-1]}: {result[m]*100:+.3f}%/day "
                  f"({result[m]*30*100:+.2f}%/mo equivalent)")
        return result
    except Exception as e:
        print(f"seasonal_drift fetch failed: {e}; using zeros")
        _seasonal_drift_cache = fallback
        _seasonal_drift_cache_date = today_
        return fallback


def get_seasonal_vol_scale(force_refresh=False):
    """Return dict {month_int (1-12): vol_scale_multiplier} for stress tails.

    Calibrated from UNG history: for each calendar month, compute the realized
    daily-return stdev; scale by overall stdev to get a multiplicative factor.
    Winter heating months (cold-snap risk) typically scale > 1.0; shoulder
    months typically scale < 1.0. Clamped to [0.6, 1.6] to avoid extremes
    from small samples.

    Used to scale stress tails in ScenarioDistribution._build() — winter
    crash/spike magnitudes are wider than shoulder.
    """
    global _seasonal_vol_cache, _seasonal_vol_cache_date
    import datetime as _dt
    today_ = _dt.date.today()
    if not force_refresh and _seasonal_vol_cache is not None \
            and _seasonal_vol_cache_date == today_:
        return _seasonal_vol_cache

    fallback = {m: 1.0 for m in range(1, 13)}
    try:
        df = yf.download('UNG', start='2010-01-01', progress=False, auto_adjust=False)
        if df is None or len(df) < 200:
            _seasonal_vol_cache = fallback
            _seasonal_vol_cache_date = today_
            return fallback
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
        df['ret'] = np.log(df[price_col] / df[price_col].shift(1))
        df['month'] = df['Date'].dt.month
        overall_std = df['ret'].std()
        if overall_std <= 0:
            _seasonal_vol_cache = fallback
            _seasonal_vol_cache_date = today_
            return fallback
        result = {}
        for m in range(1, 13):
            month_std = df[df['month'] == m]['ret'].std()
            if bool(pd.isna(month_std)) or month_std <= 0:
                result[m] = 1.0
            else:
                scale = float(month_std / overall_std)
                # Clamp to keep small-sample noise from blowing up stress tails
                result[m] = max(0.6, min(1.6, scale))
        _seasonal_vol_cache = result
        _seasonal_vol_cache_date = today_
        print("seasonal_vol_scale calibrated (winter > shoulder expected):")
        month_names = ['Jan','Feb','Mar','Apr','May','Jun',
                       'Jul','Aug','Sep','Oct','Nov','Dec']
        for m in range(1, 13):
            print(f"  {month_names[m-1]}: ×{result[m]:.2f}")
        return result
    except Exception as e:
        print(f"seasonal_vol_scale fetch failed: {e}; using 1.0")
        _seasonal_vol_cache = fallback
        _seasonal_vol_cache_date = today_
        return fallback


# ── Available Options Cache ─────────────────────────────────────────────────

_available_options = None
_available_options_time = 0
_server_startup_time = time.time()


# ── Income progress tracker (cycle 52) ──────────────────────────────────────
# Daily snapshots of strategic-objective metrics so the user can see whether
# the book is improving over time. SQLite-backed (stdlib, no extra deps).
import sqlite3 as _sqlite3  # noqa: E402
import os as _os_progress  # noqa: E402
import collections as _collections  # noqa: E402  cycle 147: rec stability

# Cycle 147: rolling-window of last 5 compute_recommendations() rec sets,
# so each rec can carry a stability_count (0-5). Addresses the "OPEN/ROLL
# disappear on small spot drift" UX issue — recs that survive cycle-to-
# cycle reshuffling get a stability badge so the operator can distinguish
# durable signal from beam-path noise.
# Cycle 150: persist the window to progress.db so auto-reloads (cycle 45)
# don't wipe history every code commit. Without persistence, stability
# resets too often to be useful — every shipped patch reset the counter
# to 0/0 and the operator had to wait ~3 refreshes for badges to recover.
_RECS_HISTORY = _collections.deque(maxlen=5)


def _rec_signature(rec):
    """Stable identifier across cycles. Strips volatile parentheticals
    like '(41% profit)' / '(OTM by $0.56)' / '(6/13)' that change with
    spot drift but don't represent a different trade."""
    typ = str(rec.get('type', '?'))
    action = str(rec.get('action', ''))
    # Take everything before the first ' (' — strips first parenthetical
    # and anything that follows. Leaves qty/strike/expiry intact.
    prefix = action.split(' (', 1)[0].strip()
    return f"{typ}|{prefix}"


def _recs_history_persist():
    """Save current _RECS_HISTORY deque to progress.db. Single-row JSON
    blob — simple, atomic via REPLACE."""
    try:
        conn = _sqlite3.connect(_PROGRESS_DB)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS rec_history_state (
                id INTEGER PRIMARY KEY,
                window_json TEXT,
                updated_ts INTEGER
            )
        """)
        window_payload = json.dumps([sorted(list(s)) for s in _RECS_HISTORY])
        conn.execute(
            "INSERT OR REPLACE INTO rec_history_state (id, window_json, updated_ts) VALUES (0, ?, ?)",
            (window_payload, int(time.time())),
        )
        conn.commit()
        conn.close()
    except Exception as _e:
        print(f"[stability] persist failed: {_e}")


def _recs_history_restore():
    """Load _RECS_HISTORY from progress.db on startup. No-op if table or
    row missing. Stale rows (>1h old) are dropped — the rolling window is
    only meaningful while the operator is actively using the system."""
    try:
        conn = _sqlite3.connect(_PROGRESS_DB)
        cur = conn.execute("SELECT window_json, updated_ts FROM rec_history_state WHERE id=0")
        row = cur.fetchone()
        conn.close()
        if not row:
            return
        window_json, updated_ts = row
        if time.time() - (updated_ts or 0) > 3600:
            return  # stale — let it refill from current activity
        loaded = json.loads(window_json or '[]')
        for sigs in loaded[-5:]:
            _RECS_HISTORY.append(set(sigs))
        if _RECS_HISTORY:
            print(f"[stability] restored {len(_RECS_HISTORY)} cycles from progress.db")
    except Exception as _e:
        print(f"[stability] restore failed: {_e}")


_PROGRESS_DB = _os_progress.path.join(_os_progress.path.dirname(_os_progress.path.abspath(__file__)),
                                      'progress.db')
_recs_history_restore()  # cycle 150: call AFTER _PROGRESS_DB defined


def _progress_init():
    """Ensure the progress DB and table exist."""
    conn = _sqlite3.connect(_PROGRESS_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS daily_snapshot (
            date TEXT PRIMARY KEY,
            ts INTEGER,
            avg_weekly_theta REAL,
            quality_total REAL,
            dd_penalty REAL,
            income_gap REAL,
            fund_score REAL,
            yoy_score REAL,
            tech_score REAL,
            supply_regime TEXT,
            income_bias REAL,
            ung_price REAL,
            shares INTEGER,
            options_count INTEGER
        )
    """)
    conn.commit()
    conn.close()


def _progress_record(snapshot):
    """Insert or update today's snapshot. `snapshot` is a dict matching columns."""
    try:
        _progress_init()
        conn = _sqlite3.connect(_PROGRESS_DB)
        conn.execute("""
            INSERT OR REPLACE INTO daily_snapshot
            (date, ts, avg_weekly_theta, quality_total, dd_penalty, income_gap,
             fund_score, yoy_score, tech_score, supply_regime, income_bias,
             ung_price, shares, options_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            snapshot.get('date'),
            int(time.time()),
            snapshot.get('avg_weekly_theta'),
            snapshot.get('quality_total'),
            snapshot.get('dd_penalty'),
            snapshot.get('income_gap'),
            snapshot.get('fund_score'),
            snapshot.get('yoy_score'),
            snapshot.get('tech_score'),
            snapshot.get('supply_regime'),
            snapshot.get('income_bias'),
            snapshot.get('ung_price'),
            snapshot.get('shares'),
            snapshot.get('options_count'),
        ))
        conn.commit()
        conn.close()
    except Exception as _pe:
        print(f"[progress] record failed: {_pe}")


def _progress_load(days=30):
    """Load the last N days of snapshots, oldest first."""
    try:
        _progress_init()
        conn = _sqlite3.connect(_PROGRESS_DB)
        conn.row_factory = _sqlite3.Row
        cur = conn.execute(
            "SELECT * FROM daily_snapshot ORDER BY date DESC LIMIT ?", (days,)
        )
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return list(reversed(rows))
    except Exception as _pe:
        print(f"[progress] load failed: {_pe}")
        return []


def fetch_available_options():
    """Fetch all available UNG option expirations, strikes, and liquidity from yfinance."""
    ung = yf.Ticker('UNG')
    available = {}  # {expiry_str: {'puts': [...], 'calls': [...], 'liquidity': {strike_right: {oi, vol, bid, ask}}}}

    for exp in ung.options:
        try:
            chain = ung.option_chain(exp)
            put_strikes = sorted([float(s) for s in chain.puts['strike'].unique() if 5 <= s <= 25])
            call_strikes = sorted([float(s) for s in chain.calls['strike'].unique() if 5 <= s <= 25])

            # Build liquidity map: {(strike, 'P'|'C'): {oi, vol, bid, ask}}
            def _safe_int(v):
                try:
                    import math
                    if v is None or (isinstance(v, float) and math.isnan(v)):
                        return 0
                    return int(v)
                except (ValueError, TypeError):
                    return 0

            def _safe_float(v):
                try:
                    import math
                    if v is None or (isinstance(v, float) and math.isnan(v)):
                        return 0.0
                    return float(v)
                except (ValueError, TypeError):
                    return 0.0

            liquidity = {}
            for _, row in chain.puts.iterrows():
                s = float(row['strike'])
                if 5 <= s <= 25:
                    _iv = _safe_float(row.get('impliedVolatility', 0))
                    # Cycle 187b: use lastPrice as fallback when bid/ask are 0
                    # (after hours / weekends / holidays). Enables simulation
                    # with most recent trade data.
                    _bid = _safe_float(row.get('bid', 0))
                    _ask = _safe_float(row.get('ask', 0))
                    _last = _safe_float(row.get('lastPrice', 0))
                    _using_last = False
                    if _bid == 0 and _ask == 0 and _last > 0:
                        _bid = _last * 0.9
                        _ask = _last * 1.1
                        _using_last = True
                    _oi = _safe_int(row.get('openInterest', 0))
                    if _oi == 0 and _using_last:
                        _oi = 100  # assume liquid when using lastPrice
                    liquidity[(s, 'P')] = {
                        'oi': _oi,
                        'vol': _safe_int(row.get('volume', 0)),
                        'bid': _bid,
                        'ask': _ask,
                        'iv': _iv if 0.05 < _iv < 3.0 else 0.0,
                    }
            for _, row in chain.calls.iterrows():
                s = float(row['strike'])
                if 5 <= s <= 25:
                    _iv = _safe_float(row.get('impliedVolatility', 0))
                    _bid = _safe_float(row.get('bid', 0))
                    _ask = _safe_float(row.get('ask', 0))
                    _last = _safe_float(row.get('lastPrice', 0))
                    _using_last = False
                    if _bid == 0 and _ask == 0 and _last > 0:
                        _bid = _last * 0.9
                        _ask = _last * 1.1
                        _using_last = True
                    _oi = _safe_int(row.get('openInterest', 0))
                    if _oi == 0 and _using_last:
                        _oi = 100
                    liquidity[(s, 'C')] = {
                        'oi': _oi,
                        'vol': _safe_int(row.get('volume', 0)),
                        'bid': _bid,
                        'ask': _ask,
                        'iv': _iv if 0.05 < _iv < 3.0 else 0.0,
                    }

            available[exp] = {
                'puts': put_strikes,
                'calls': call_strikes,
                'liquidity': liquidity,
            }
        except Exception:
            pass

    return available


def get_available_options():
    """Get available options, refreshing cache every 5 minutes."""
    global _available_options, _available_options_time
    now = time.time()
    if _available_options is None or now - _available_options_time > 300:
        _available_options = fetch_available_options()
        _available_options_time = now
    return _available_options


def get_contract_iv(expiry_str, strike, right, fallback=0.50):
    """Look up real implied volatility for a specific contract from yfinance.
    Returns fallback (typically 0.50) if not available or out of sane range.
    Used by Kelly/assignment_sim/recovery to replace the legacy fixed-0.50."""
    try:
        avail = get_available_options()
        if not avail:
            return fallback
        exp_data = avail.get(expiry_str)
        if not exp_data:
            return fallback
        liq = exp_data.get('liquidity', {}).get((float(strike), right))
        if not liq:
            return fallback
        iv = float(liq.get('iv', 0.0))
        return iv if 0.05 < iv < 3.0 else fallback
    except Exception:
        return fallback


def find_nearest_strike(target, strikes):
    """Find the nearest available strike to target price."""
    if not strikes:
        return None
    return min(strikes, key=lambda s: abs(s - target))


# ── Technicals Cache ────────────────────────────────────────────────────────

_technicals_cache = {'data': None, 'timestamp': 0}
_TECHNICALS_TTL = 300  # 5 minutes


def compute_technicals():
    """Fetch UNG technical data from yfinance (cached for 5 minutes)."""
    now = time.time()
    if _technicals_cache['data'] is not None and (now - _technicals_cache['timestamp']) < _TECHNICALS_TTL:
        return _technicals_cache['data']

    ung = yf.Ticker('UNG')
    hist = ung.history(period='1y', interval='1d')
    if hasattr(hist.index, 'tz') and hist.index.tz is not None:
        hist.index = hist.index.tz_localize(None)  # type: ignore[union-attr]

    spot = float(hist['Close'].iloc[-1])

    # Moving averages
    ma_20 = float(hist['Close'].rolling(20).mean().iloc[-1])
    ma_50 = float(hist['Close'].rolling(50).mean().iloc[-1])
    ma_100 = float(hist['Close'].rolling(100).mean().iloc[-1])
    ma_200 = float(hist['Close'].rolling(200).mean().iloc[-1])

    # 52-week high/low
    high_52w = float(hist['High'].max())
    low_52w = float(hist['Low'].min())

    # 120-day high/low (for gamma regime detection)
    high_120d = float(hist['High'].iloc[-120:].max())
    low_120d = float(hist['Low'].iloc[-120:].min())

    # Realized volatility
    rets = hist['Close'].pct_change()
    rv_21 = float(rets.rolling(21).std().iloc[-1] * (252 ** 0.5))
    rv_63 = float(rets.rolling(63).std().iloc[-1] * (252 ** 0.5))

    # VWAP (cumulative approximation)
    hist['vwap'] = (hist['Close'] * hist['Volume']).cumsum() / hist['Volume'].cumsum()
    vwap = float(hist['vwap'].iloc[-1])

    # Contango-adjusted targets (UNG decays ~3%/month)
    contango_30d = spot * 0.97
    contango_60d = spot * 0.94
    contango_90d = spot * 0.91

    # Recent price history for chart (last 60 days)
    price_history = []
    for d, row in hist.iloc[-60:].iterrows():
        price_history.append({
            'date': d.strftime('%Y-%m-%d'),
            'open': round(float(row['Open']), 2),
            'high': round(float(row['High']), 2),
            'low': round(float(row['Low']), 2),
            'close': round(float(row['Close']), 2),
        })

    # MA history for the last 60 days
    ma20_series = hist['Close'].rolling(20).mean()
    ma50_series = hist['Close'].rolling(50).mean()
    ma100_series = hist['Close'].rolling(100).mean()
    ma200_series = hist['Close'].rolling(200).mean()
    ma_history = {
        'dates': [d.strftime('%Y-%m-%d') for d in hist.index[-60:]],
        'ma_20': [round(float(v), 2) if not pd.isna(v) else None for v in ma20_series.iloc[-60:]],
        'ma_50': [round(float(v), 2) if not pd.isna(v) else None for v in ma50_series.iloc[-60:]],
        'ma_100': [round(float(v), 2) if not pd.isna(v) else None for v in ma100_series.iloc[-60:]],
        'ma_200': [round(float(v), 2) if not pd.isna(v) else None for v in ma200_series.iloc[-60:]],
    }

    # IV term structure from option chains
    iv_term = []
    try:
        exps = ung.options[:10]
        for exp in exps:
            dte = (pd.Timestamp(exp) - pd.Timestamp.now()).days
            if dte < 0:
                continue
            try:
                chain = ung.option_chain(exp)
                atm_strike = chain.calls.iloc[(chain.calls['strike'] - spot).abs().argsort()[:1]]['strike'].values[0]
                c = chain.calls[chain.calls['strike'] == atm_strike]
                p = chain.puts[chain.puts['strike'] == atm_strike]
                if len(c) > 0 and len(p) > 0:
                    c_iv = float(c['impliedVolatility'].values[0])
                    p_iv = float(p['impliedVolatility'].values[0])
                    iv_term.append({
                        'exp': exp,
                        'dte': dte,
                        'call_iv': round(c_iv * 100, 1),
                        'put_iv': round(p_iv * 100, 1),
                        'avg_iv': round((c_iv + p_iv) / 2 * 100, 1),
                    })
            except Exception:
                continue
    except Exception:
        pass

    # IV surface data
    iv_surface = []
    try:
        exps_surf = ung.options[:8]
        for exp in exps_surf:
            dte = (pd.Timestamp(exp) - pd.Timestamp.now()).days
            if dte < 0:
                continue
            try:
                chain = ung.option_chain(exp)
                # Get strikes within reasonable range
                mask = (chain.calls['strike'] >= spot * 0.7) & (chain.calls['strike'] <= spot * 1.3)
                for _, row in chain.calls[mask].iterrows():
                    if row['impliedVolatility'] > 0.01:
                        iv_surface.append({
                            'strike': float(row['strike']),
                            'dte': dte,
                            'iv': round(float(row['impliedVolatility']) * 100, 1),
                        })
            except Exception:
                continue
    except Exception:
        pass

    # Cycle 115: real UNG GEX walls (was hardcoded 12.0 / 10.50). Reuses
    # the iv_surface candidates we just gathered to compute dealer-gamma
    # per strike. Aggregates calls and puts separately; the strike with
    # max |put GEX| is the put wall (support); max call GEX is call wall
    # (resistance). Operator's GEX intuition from SOXX work applies here
    # too — knowing the actual walls is more useful than fixed numbers.
    _gex_put_wall = 10.50
    _gex_call_wall = 12.0
    _net_gex = 0.0
    try:
        _r_rate = 0.04
        _by_strike = {}  # strike -> {'call': float, 'put': float}
        _today_d = date.today()
        for exp_str in ung.options[:8]:
            _dte = (date.fromisoformat(exp_str) - _today_d).days
            if _dte <= 0 or _dte > 60:
                continue
            _T = _dte / 365.0
            try:
                _ch = ung.option_chain(exp_str)
            except Exception:
                continue
            for _df, _side in [(_ch.calls, 'call'), (_ch.puts, 'put')]:
                for _, _row in _df.iterrows():
                    _K = float(_row['strike'])
                    if _K < spot * 0.6 or _K > spot * 1.4:
                        continue
                    _oi = int(_row.get('openInterest', 0) or 0)
                    _iv = float(_row.get('impliedVolatility', 0) or 0)
                    if _oi <= 0 or _iv <= 0:
                        continue
                    _g = bs_gamma(spot, _K, _T, _r_rate, _iv)
                    _gex = _g * _oi * 100 * (spot ** 2) * 0.01
                    _bs = _by_strike.setdefault(_K, {'call': 0.0, 'put': 0.0})
                    _bs[_side] += _gex
        if _by_strike:
            _net_gex = (sum(v['call'] for v in _by_strike.values())
                        - sum(v['put'] for v in _by_strike.values()))
            _gex_put_wall = max(_by_strike.items(), key=lambda x: x[1]['put'])[0]
            _gex_call_wall = max(_by_strike.items(), key=lambda x: x[1]['call'])[0]
    except Exception as _ge:
        print(f"[gex] compute failed: {_ge}; using fallback walls")

    result = {
        'spot': round(spot, 2),
        'ma_20': round(ma_20, 2),
        'ma_50': round(ma_50, 2),
        'ma_100': round(ma_100, 2),
        'ma_200': round(ma_200, 2),
        'high_52w': round(high_52w, 2),
        'low_52w': round(low_52w, 2),
        'high_120d': round(high_120d, 2),
        'low_120d': round(low_120d, 2),
        'rv_21': round(rv_21 * 100, 1),
        'rv_63': round(rv_63 * 100, 1),
        'vwap': round(vwap, 2),
        'contango_30d': round(contango_30d, 2),
        'contango_60d': round(contango_60d, 2),
        'contango_90d': round(contango_90d, 2),
        'price_history': price_history,
        'ma_history': ma_history,
        'share_avg': SHARE_AVG,
        'highest_put_strike': round(_gex_call_wall, 2),
        'gex_put_wall': round(_gex_put_wall, 2),
        'gex_call_wall': round(_gex_call_wall, 2),
        'net_gex_M': round(_net_gex / 1e6, 2),
        'iv_term': iv_term,
        'iv_surface': iv_surface,
    }

    _technicals_cache['data'] = result
    _technicals_cache['timestamp'] = now
    return result

# ── Gamma Regime Detection ──────────────────────────────────────────────────

_gamma_regime_cache = {'data': None, 'timestamp': 0}


def compute_gamma_regime(spot):
    """Determine gamma stance based on where UNG sits in its 120-day range.

    Computed ONCE per refresh (cached 5 minutes alongside technicals).
    """
    now = time.time()
    if _gamma_regime_cache['data'] is not None and (now - _gamma_regime_cache['timestamp']) < _TECHNICALS_TTL:
        return _gamma_regime_cache['data']

    tech = compute_technicals()
    trail_low = tech.get('low_120d', tech.get('low_52w', spot * 0.85))
    trail_high = tech.get('high_120d', tech.get('high_52w', spot * 1.15))

    pct_above_low = (spot - trail_low) / trail_low if trail_low > 0 else 0
    pct_below_high = (trail_high - spot) / trail_high if trail_high > 0 else 0

    reasoning = []
    reasoning.append(f"Spot ${spot:.2f} | 120d range ${trail_low:.2f}-${trail_high:.2f}")
    reasoning.append(f"{pct_above_low*100:.1f}% above 120d low, {pct_below_high*100:.1f}% below 120d high")

    if pct_above_low >= 0.35:
        regime = 'EXIT'
        gamma_stance = "Eliminate gamma. Close all short puts. Sell remaining shares via aggressive covered calls. Go to cash."
        reasoning.append("Price >35% above 120d low — extended, high risk of reversion")
    elif pct_above_low >= 0.20:
        regime = 'HARVEST'
        gamma_stance = "Reduce gamma. Stop selling new puts. Focus on covered calls to exit shares. Let existing puts expire or roll to OTM."
        reasoning.append("Price 20-35% above 120d low — take profits, reduce put exposure")
    elif pct_above_low >= 0.10:
        regime = 'HOLD'
        gamma_stance = "Maintain current gamma. Normal wheel operations."
        reasoning.append("Price 10-20% above 120d low — neutral zone, standard operations")
    else:
        regime = 'ACCUMULATE'
        gamma_stance = "Welcome gamma. Sell puts aggressively. Assignment = buying cheap."
        reasoning.append("Price <10% above 120d low — near bottom, accumulate via puts")

    result = {
        'regime': regime,
        'pct_above_low': round(pct_above_low, 4),
        'pct_below_high': round(pct_below_high, 4),
        'trail_low': trail_low,
        'trail_high': trail_high,
        'gamma_stance': gamma_stance,
        'reasoning': reasoning,
    }

    _gamma_regime_cache['data'] = result
    _gamma_regime_cache['timestamp'] = now
    return result


# ── WealthSimple Live Position Fetch ─────────────────────────────────────────
_margin_capital_usd = 109433  # fetched from WS: $113k NLV - $3.6k cushion. Updated by fetch_ws_positions().
_OTHER_HOLDINGS = {}  # cycle 142: non-UNG stock holdings (BOXX, ADA, etc.) captured by fetch_ws_positions
# Cycle 201: cash, buying power, and position value tracked separately.
# Cash = WS-reported settled cash (FetchTradingBalanceBuyingPower).
# Buying Power = Cash + margin available for new buys.
# Position Value = Σ(positions market value).
# User: "never in negative cash to avoid pay interest."
_ws_cash_usd = 0.0
_ws_buying_power_usd = 0.0
_ws_position_value_usd = 0.0


def fetch_ws_positions():
    """Fetch live UNG positions from WealthSimple."""
    import sys
    sys.path.insert(0, '/home/wyatt/ibkr_guided_trade')
    from ws_sdk import (
        get_session, load_config, load_cookies, graphql_query,
        QUERY_FETCH_POSITIONS, extract_identity_from_cookies,
    )

    try:
        session = get_session()
        config = load_config()
        cookies = load_cookies()
        identity_id = config.get('identity_id') or extract_identity_from_cookies(cookies)

        # Fetch margin capital FIRST while session is fresh
        global _margin_capital_usd
        try:
            from ws_sdk import QUERY_FETCH_FINANCIALS, QUERY_ALL_ACCOUNTS
            _accts = graphql_query(session, 'FetchAllAccounts', QUERY_ALL_ACCOUNTS, {'identityId': identity_id})
            _mid = None
            if _accts:
                for _e in _accts.get('identity', {}).get('accounts', {}).get('edges', []):
                    _n = _e.get('node', {})
                    if 'MARGIN' in str(_n.get('unifiedAccountType', '')):
                        _mid = _n.get('id')
                        break
            if _mid:
                _fin = graphql_query(session, 'FetchIdentityCurrentFinancials', QUERY_FETCH_FINANCIALS, {
                    'identityId': identity_id, 'currency': 'USD', 'accountIds': [_mid]
                })
                if _fin:
                    _nlv = _fin.get('identity', {}).get('financials', {}).get('current', {}).get('netLiquidationValueV2', {})
                    _val = float(_nlv.get('amount', 0))
                    if _val > 1000:
                        _margin_capital_usd = _val - 3600
                        print(f"Margin capital (USD): ${_margin_capital_usd:,.0f}")
        except Exception as _ce:
            print(f"Capital fetch failed: {_ce}")

        # Cycle 201: filter to margin account only — user has DBA and ADA in
        # other accounts (crypto, cash savings) that shouldn't pollute the
        # engine's view. _mid was computed above; if missing, fall back to
        # all accounts (with warning).
        # Cycle 201b: query positions in USD to match NLV currency (was CAD,
        # which caused Cash = NLV_USD - positions_CAD arithmetic mismatch).
        # User reported Cash = $15,189.85; we were calculating $12,041 due
        # to the currency error.
        _query_args = {
            "identityId": identity_id,
            "currency": "USD",
            "first": 50,
            "aggregated": True,
            "currencyOverride": "MARKET",
            "sort": "TODAY_GAIN",
            "includeSecurity": True,
            "includeAccountData": True,
            "includeOneDayReturnsBaseline": True,
        }
        if _mid:
            _query_args["accountIds"] = [_mid]
        data = graphql_query(session, "FetchIdentityPositions", QUERY_FETCH_POSITIONS, _query_args)

        shares = 0
        share_avg = 0.0
        options = []
        ung_price = None
        global _OTHER_HOLDINGS, _ws_cash_usd, _ws_position_value_usd
        _OTHER_HOLDINGS = {}  # symbol -> {qty, market_value}
        # Cycle 201: track total position value separately so we can derive cash
        # Cash = NLV - Σ(position market values). User: "cash and margin are
        # different things... never in negative cash to avoid pay interest."
        _total_position_value = 0.0

        identity = data.get('identity', {})
        positions = (identity.get('financials', {})
                     .get('current', {})
                     .get('positions', {})
                     .get('edges', []))

        for edge in positions:
            pos = edge.get('node', {})
            security = pos.get('security', {})
            stock = security.get('stock', {})
            symbol = stock.get('symbol', '')
            option_details = security.get('optionDetails') or {}

            qty = float(pos.get('quantity', 0))
            avg_price = pos.get('marketAveragePrice', pos.get('averagePrice', {}))
            avg_cost = float(avg_price.get('amount', 0))

            # Determine underlying for options
            underlying = ''
            if option_details:
                underlying = (option_details.get('underlyingSecurity', {})
                              .get('stock', {}).get('symbol', ''))

            if symbol == 'UNG' and not option_details:
                # UNG shares
                shares = int(qty)
                share_avg = avg_cost
                # Derive current price from market value / qty
                market = pos.get('totalValue', {})
                market_value = float(market.get('amount', 0))
                if qty != 0:
                    ung_price = abs(market_value / qty)
                _total_position_value += market_value
            elif underlying == 'UNG' and option_details:
                # UNG option position
                opt_type = option_details.get('optionType', '')
                right = 'C' if 'call' in opt_type.lower() else 'P'
                strike = float(option_details.get('strikePrice', 0))
                expiry_str = option_details.get('expiryDate', '')
                if len(expiry_str) >= 10:
                    expiry_str = expiry_str[:10]

                # avg_cost from WS is per-contract cost (matching hardcoded format)
                options.append((expiry_str, strike, right, int(qty), avg_cost))
                # Add this option's market value to position total
                _opt_market = pos.get('totalValue', {})
                _total_position_value += float(_opt_market.get('amount', 0))
            elif symbol and not option_details and symbol != 'UNG':
                # Cycle 142: capture non-UNG stock positions (BOXX, ADA, etc.)
                # so cash_park_suggestion sees what's already in cash-equivalents.
                market = pos.get('totalValue', {})
                mv = float(market.get('amount', 0))
                _OTHER_HOLDINGS[symbol] = {
                    'qty': float(qty),
                    'market_value': mv,
                    'avg_cost': avg_cost,
                }
                _total_position_value += mv

        if ung_price is None:
            # Fallback: get spot from yfinance
            ung_price = float(yf.Ticker('UNG').history(period='1d')['Close'].iloc[-1])

        # Cycle 201c: query WS directly for Cash and Buying Power instead of
        # deriving from NLV - positions (inferential, off by ~$1k due to
        # intra-second price drift). User: "try to see exact way to get USD
        # available." Found FetchTradingBalanceBuyingPower returns exact
        # USD cash + buying power.
        _ws_position_value_usd = _total_position_value
        try:
            from ws_sdk import QUERY_TRADING_BALANCE
            _UNG_SEC_ID = 'sec-s-32f0b46791214cbcbee9486e40232ea4'
            _tb = graphql_query(session, 'FetchTradingBalanceBuyingPower',
                                QUERY_TRADING_BALANCE, {
                'accountCanonicalId': _mid, 'currency': 'USD',
                'securityId': _UNG_SEC_ID,
            })
            if _tb:
                _tbv = _tb.get('account', {}).get('financials', {}).get('current', {}).get('tradingBalanceView', {})
                _ws_cash_usd = float(_tbv.get('cash', {}).get('quantity', 0))
                global _ws_buying_power_usd
                _ws_buying_power_usd = float(_tbv.get('buyingPower', {}).get('quantity', 0))
                print(f"Cash (USD): ${_ws_cash_usd:,.2f} | Buying Power: ${_ws_buying_power_usd:,.2f} | Position value: ${_total_position_value:,.2f}")
        except Exception as _ce:
            # Fallback to inferential calculation
            _ws_cash_usd = (_margin_capital_usd + 3600) - _total_position_value
            print(f"Cash (inferential): ${_ws_cash_usd:,.2f}  (trading balance query failed: {_ce})")

        return shares, share_avg, options, ung_price

    except Exception as e:
        print(f"WS fetch failed: {e}")
        return None


# ── Portfolio Data ───────────────────────────────────────────────────────────

_FALLBACK_SHARES = 7100
_FALLBACK_SHARE_AVG = 12.04
_FALLBACK_UNG_PRICE = 10.74
_FALLBACK_OPTIONS = [
    ('2026-05-06', 11.0, 'C', -2, 28.50),
    ('2026-05-08', 11.0, 'C', -4, 32.75),
    ('2026-05-08', 11.0, 'P', -2, 51.00),
    ('2026-05-08', 11.5, 'C', -1, 22.00),
    ('2026-05-08', 12.0, 'C', -10, 12.00),
    ('2026-05-15', 10.0, 'P', -2, 30.50),
    ('2026-05-15', 10.5, 'P', -1, 39.00),
    ('2026-05-15', 11.0, 'C', -4, 43.00),
    ('2026-05-15', 11.0, 'P', -7, 63.29),
    ('2026-05-15', 11.5, 'C', -2, 24.00),
    ('2026-05-15', 11.5, 'P', -5, 77.00),
    ('2026-05-15', 12.0, 'C', -8, 20.00),
    ('2026-05-15', 12.0, 'P', -13, 102.46),
    ('2026-05-22', 10.0, 'P', -3, 37.33),
    ('2026-05-22', 10.5, 'P', -4, 45.25),
    ('2026-05-22', 11.0, 'C', -5, 53.60),
    ('2026-05-22', 11.5, 'C', -4, 38.25),
    ('2026-05-22', 11.5, 'P', -4, 88.75),
    ('2026-05-29', 10.0, 'P', -4, 41.00),
    ('2026-05-29', 10.5, 'P', -10, 48.00),
    ('2026-05-29', 11.0, 'C', -14, 55.36),
    ('2026-05-29', 11.0, 'P', -1, 87.00),
    ('2026-06-05', 10.5, 'P', -3, 67.67),
    ('2026-06-18', 11.0, 'P', -8, 90.88),
    ('2026-07-17', 10.0, 'P', -2, 69.00),
    ('2026-07-17', 11.0, 'P', -6, 108.67),
    ('2026-10-16', 10.0, 'P', -2, 116.00),
    ('2027-01-21', 11.0, 'P', -4, 319.00),
]

# Try to fetch live data from WS, fall back to cached
_ws_data = fetch_ws_positions()
if _ws_data:
    SHARES, SHARE_AVG, OPTIONS, UNG_PRICE = _ws_data
    print(f"Live WS data: {SHARES} shares @ ${SHARE_AVG:.2f}, {len(OPTIONS)} options, UNG ${UNG_PRICE:.2f}")
    # fetch_margin_capital() called in main() after all functions are defined
else:
    SHARES = _FALLBACK_SHARES
    SHARE_AVG = _FALLBACK_SHARE_AVG
    UNG_PRICE = _FALLBACK_UNG_PRICE
    OPTIONS = list(_FALLBACK_OPTIONS)
    print("Using cached position data")

# ── Black-Scholes ────────────────────────────────────────────────────────────

def bs_price(S, K, T, r, sigma, right):
    if T <= 0.001:
        return max(0, S - K) if right == 'C' else max(0, K - S)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if right == 'C':
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


import functools as _functools  # noqa: E402  cycle 80: lru_cache for bs_* primitives


@_functools.lru_cache(maxsize=8192)
def bs_delta(S, K, T, r, sigma, right):
    if T <= 0.001:
        if right == 'C':
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    if right == 'C':
        return norm.cdf(d1)
    return norm.cdf(d1) - 1.0


@_functools.lru_cache(maxsize=8192)
def bs_gamma(S, K, T, r, sigma):
    if T <= 0.001:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return norm.pdf(d1) / (S * sigma * math.sqrt(T))


@_functools.lru_cache(maxsize=8192)
def bs_theta(S, K, T, r, sigma, right):
    if T <= 0.001:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    common = -(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T))
    if right == 'C':
        return (common - r * K * math.exp(-r * T) * norm.cdf(d2)) / 365.0
    return (common + r * K * math.exp(-r * T) * norm.cdf(-d2)) / 365.0


@_functools.lru_cache(maxsize=8192)
def bs_vega(S, K, T, r, sigma):
    if T <= 0.001:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return S * norm.pdf(d1) * math.sqrt(T) / 100.0  # per 1% IV change


# ── Computation ──────────────────────────────────────────────────────────────

def compute_data(price, iv, excluded_indices):
    """Compute all portfolio data for a given UNG price and IV."""
    r = 0.045  # risk-free rate
    today = date.today()

    # Share P&L
    share_pnl = SHARES * (price - SHARE_AVG)
    share_delta = SHARES

    # Option computations
    option_rows = []
    total_opt_pnl = 0.0
    total_opt_delta = 0.0
    total_opt_gamma = 0.0
    total_opt_theta = 0.0
    total_opt_vega = 0.0

    for idx, (expiry_str, strike, right, qty, avg_cost) in enumerate(OPTIONS):
        excluded = idx in excluded_indices
        expiry = datetime.strptime(expiry_str, '%Y-%m-%d').date()
        dte = max((expiry - today).days, 0)
        T = dte / 365.0

        theo_price = bs_price(price, strike, T, r, iv, right) * 100  # per contract
        delta = bs_delta(price, strike, T, r, iv, right)
        gamma = bs_gamma(price, strike, T, r, iv)
        theta = bs_theta(price, strike, T, r, iv, right)
        vega = bs_vega(price, strike, T, r, iv)

        # P&L: for short options, we sold at avg_cost, current value is theo_price
        # qty is negative (short), so position_pnl = qty * (theo_price - avg_cost)
        # When short: qty=-2, sold at 28.50, now worth 50 => pnl = -2*(50-28.50) = -43
        position_pnl = qty * (theo_price - avg_cost)
        position_delta = qty * delta * 100  # share-equivalent
        position_gamma = qty * gamma * 100
        position_theta = qty * theta * 100
        position_vega = qty * vega * 100

        # Compute extrinsic value % remaining
        per_share_price = theo_price / 100.0  # per share
        if right == 'P':
            intrinsic = max(0, strike - price)
        else:
            intrinsic = max(0, price - strike)
        extrinsic = max(0, per_share_price - intrinsic)
        extrinsic_pct = (extrinsic / per_share_price * 100) if per_share_price > 0.01 else 0

        row = {
            'idx': idx,
            'expiry': expiry_str,
            'strike': strike,
            'right': right,
            'qty': qty,
            'avg_cost': avg_cost,
            'theo_price': theo_price,
            'pnl': position_pnl,
            'delta': position_delta,
            'gamma': position_gamma,
            'theta': position_theta,
            'vega': position_vega,
            'dte': dte,
            'excluded': excluded,
            'intrinsic': round(intrinsic, 4),
            'extrinsic': round(extrinsic, 4),
            'extrinsic_pct': round(extrinsic_pct, 1),
        }
        option_rows.append(row)

        if not excluded:
            total_opt_pnl += position_pnl
            total_opt_delta += position_delta
            total_opt_gamma += position_gamma
            total_opt_theta += position_theta
            total_opt_vega += position_vega

    # Active share delta (shares are never excluded)
    net_delta = share_delta + total_opt_delta
    total_pnl = share_pnl + total_opt_pnl

    # P&L profile across price range
    prices = np.linspace(7, 16, 181).tolist()
    pnl_shares = []
    pnl_options = []
    pnl_total = []
    delta_profile = []

    for p in prices:
        s_pnl = SHARES * (p - SHARE_AVG)
        o_pnl = 0.0
        o_delta = 0.0
        for idx, (expiry_str, strike, right, qty, avg_cost) in enumerate(OPTIONS):
            if idx in excluded_indices:
                continue
            expiry = datetime.strptime(expiry_str, '%Y-%m-%d').date()
            dte = max((expiry - today).days, 0)
            T = dte / 365.0
            tp = bs_price(p, strike, T, r, iv, right) * 100
            o_pnl += qty * (tp - avg_cost)
            o_delta += qty * bs_delta(p, strike, T, r, iv, right) * 100
        pnl_shares.append(round(s_pnl, 2))
        pnl_options.append(round(o_pnl, 2))
        pnl_total.append(round(s_pnl + o_pnl, 2))
        delta_profile.append(round(float(SHARES + o_delta), 2))

    # Theta timeline by expiry bucket
    expiry_buckets = {}
    for row in option_rows:
        if row['excluded']:
            continue
        exp = row['expiry']
        if exp not in expiry_buckets:
            expiry_buckets[exp] = {'theta': 0.0, 'contracts': 0}
        expiry_buckets[exp]['theta'] += row['theta']
        expiry_buckets[exp]['contracts'] += abs(row['qty'])

    theta_timeline = []
    for exp in sorted(expiry_buckets.keys()):
        theta_timeline.append({
            'expiry': exp,
            'daily_theta': round(expiry_buckets[exp]['theta'], 2),
            'contracts': expiry_buckets[exp]['contracts'],
        })

    # Heatmap data: strike x expiry grid
    strikes_set = sorted(set(o[1] for o in OPTIONS))
    expiries_set = sorted(set(o[0] for o in OPTIONS))
    heatmap = []
    for s in strikes_set:
        row_data = []
        for e in expiries_set:
            count = 0
            for idx, (expiry_str, strike, right, qty, avg_cost) in enumerate(OPTIONS):
                if idx in excluded_indices:
                    continue
                if expiry_str == e and strike == s:
                    count += abs(qty)
            row_data.append(count)
        heatmap.append(row_data)

    return {
        'summary': {
            'price': round(price, 2),
            'iv': round(iv, 4),
            'shares': SHARES,
            'share_avg': SHARE_AVG,
            'share_pnl': round(share_pnl, 2),
            'total_options': sum(abs(o[3]) for i, o in enumerate(OPTIONS) if i not in excluded_indices),
            'net_delta': round(float(net_delta), 2),
            'total_gamma': round(total_opt_gamma, 2),
            'total_theta': round(total_opt_theta, 2),
            'total_vega': round(total_opt_vega, 2),
            'total_pnl': round(total_pnl, 2),
            'option_pnl': round(total_opt_pnl, 2),
        },
        'options': option_rows,
        'profile': {
            'prices': [round(p, 2) for p in prices],
            'pnl_shares': pnl_shares,
            'pnl_options': pnl_options,
            'pnl_total': pnl_total,
            'delta_profile': delta_profile,
        },
        'theta_timeline': theta_timeline,
        'heatmap': {
            'strikes': strikes_set,
            'expiries': expiries_set,
            'data': heatmap,
        },
        'outlook': _build_outlook(price),
    }


def _build_outlook(ung_spot):
    """Translate NG fair-value scenarios into UNG bull/base/bear targets.

    Uses pct move on NG → applied to UNG, then de-rated by 30-day contango
    drag for the UNG side. Returns None if model predictions not yet available.
    """
    p = _model_predictions
    ng_now = p.get('ng_current')
    if not ng_now or ng_now <= 0:
        return None
    # Approx contango decay. The technicals cache stores `contango_30d` as a
    # dollar TARGET (spot × 0.97 in compute_technicals), not a percent. Convert
    # to a percent here.
    contango_30d_pct = 3.0
    try:
        if _technicals_cache.get('data'):
            contango_target = float(_technicals_cache['data'].get('contango_30d', 0))
            if contango_target > 0 and ung_spot > 0:
                contango_30d_pct = max(0.0, (1.0 - contango_target / ung_spot) * 100.0)
    except Exception:
        pass
    decay = 1.0 - contango_30d_pct / 100.0  # e.g. 0.97 for 3% contango

    def _ung_from_ng(ng_target):
        if not ng_target or ng_target <= 0:
            return None
        pct_move = ng_target / ng_now - 1.0
        return round(ung_spot * (1.0 + pct_move) * decay, 2)

    return {
        'z_score': _model_zscore,
        'ng_current': ng_now,
        'ng_base': p.get('ng_fv_base'),
        'ng_bull': p.get('ng_fv_bull'),
        'ng_bear': p.get('ng_fv_bear'),
        'ung_current': round(ung_spot, 2),
        'ung_base': _ung_from_ng(p.get('ng_fv_base')),
        'ung_bull': _ung_from_ng(p.get('ng_fv_bull')),
        'ung_bear': _ung_from_ng(p.get('ng_fv_bear')),
        'contango_30d_pct': round(contango_30d_pct, 2),
        'updated_at': p.get('updated_at'),
    }


def _scenarios_for_ung(ung_spot):
    """Return [(ung_price, weight)] from NG model outlook. Used for probabilistic scoring.

    Weights: 30% bull / 40% base / 30% bear. Returns empty list if outlook not
    available (then probabilistic scoring is a no-op).
    """
    out = _build_outlook(ung_spot)
    if not out:
        return []
    bull = out.get('ung_bull')
    base = out.get('ung_base')
    bear = out.get('ung_bear')
    if not (bull and base and bear):
        return []
    return [(float(bull), 0.30), (float(base), 0.40), (float(bear), 0.30)]


def compute_portfolio_state(positions, spot, iv, today):
    """Compute current portfolio state metrics."""
    r = 0.04
    total_theta = 0
    total_delta = float(SHARES)
    total_gamma = 0
    total_vega = 0

    # Per-expiry aggregations for concentration tracking. Cycle 62 adds
    # delta and gamma alongside the existing theta so the dashboard can
    # show "which expiry holds the most short gamma" — directly actionable
    # for the gamma_convexity DD driver surfaced in cycles 56-58.
    expiry_theta = {}
    expiry_delta = {}
    expiry_gamma = {}
    expiry_contract_count = {}

    # Cycle 77: cache per-(strike, exp_str, right) Greek tuples globally.
    # For fixed spot/iv/today within one /api/timeline request, the per-
    # share Greeks of any (strike, expiry, right) are invariant — only qty
    # changes between portfolio states. Cycle 73's weekly_theta fix
    # eliminated one redundancy; this catches the main Greek loop.
    # Cache key includes spot/iv to auto-invalidate across requests.
    global _GREEKS_CACHE
    if '_GREEKS_CACHE' not in globals():
        _GREEKS_CACHE = {'__key': None, 'greeks': {}}
    _cache_key = (round(spot, 6), round(iv, 6), today.toordinal())
    if _GREEKS_CACHE['__key'] != _cache_key:
        _GREEKS_CACHE = {'__key': _cache_key, 'greeks': {}}
    _gc = _GREEKS_CACHE['greeks']

    _pos_info = []  # (expiry_date, daily_theta_dollars)
    for exp_str, strike, right, qty, avg_cost in positions:
        _gk = (strike, exp_str, right)
        _per = _gc.get(_gk)
        if _per is None:
            expiry = datetime.strptime(exp_str, '%Y-%m-%d').date()
            dte = max((expiry - today).days, 0)
            T = dte / 365.0
            _per = (
                expiry,
                bs_theta(spot, strike, T, r, iv, right) * 100,  # per-share-per-day × 100
                bs_delta(spot, strike, T, r, iv, right) * 100,
                bs_gamma(spot, strike, T, r, iv) * 100,
                bs_vega(spot, strike, T, r, iv) * 100,
            )
            _gc[_gk] = _per
        expiry, _t_pc, _d_pc, _g_pc, _v_pc = _per
        theta = abs(qty * _t_pc)
        delta = qty * _d_pc
        gamma = qty * _g_pc
        vega = qty * _v_pc

        total_theta += theta
        total_delta += delta
        total_gamma += gamma
        total_vega += vega

        expiry_theta[exp_str] = expiry_theta.get(exp_str, 0) + theta
        expiry_delta[exp_str] = expiry_delta.get(exp_str, 0) + delta
        expiry_gamma[exp_str] = expiry_gamma.get(exp_str, 0) + gamma
        expiry_contract_count[exp_str] = expiry_contract_count.get(exp_str, 0) + abs(qty)

        _pos_info.append((expiry, theta))

    # Weekly $ income from theta decay. Sums each position's $/day theta
    # while it is alive during the week, then × 7. Uses the cached per-position
    # daily theta from the loop above — was previously re-running bs_theta in
    # an inner loop, which dominated apply_trade_to_state cost (cycle 73 fix).
    weekly_theta = {}
    week_start = today + timedelta(days=-today.weekday())
    for i in range(12):
        wk = week_start + timedelta(days=7 * i)
        wk_label = wk.strftime('%b %d')
        wk_theta = 0.0
        for expiry, daily_theta in _pos_info:
            if expiry >= wk:
                wk_theta += daily_theta
        weekly_theta[wk_label] = wk_theta * 7  # daily → weekly $

    # Smoothness from active weeks
    max_wk = max(weekly_theta.values(), default=0)
    # Cycle 186: removed 5% threshold — use ALL positive weeks for smoothness.
    # The threshold created cliff artifacts (smoothness +234 phantom when
    # late-week buckets crossed in/out of "active"). Same class of bug as
    # the income metric cliff fixed in cycle 173.
    # Cycle 188b: use first 4 weeks for smoothness (practical wheel horizon).
    # Full 12-week window gives 0% because far-future empty weeks ($5) drag
    # CV > 1. The wheel refills those weeks — measuring them now is pointless.
    # 4 weeks ≈ 1 monthly cycle of the wheel.
    _all_wt = list(weekly_theta.values())
    active = [v for v in _all_wt[:6] if v > 0]
    if len(active) > 2 and np.mean(active) > 0:
        smoothness = max(0, 1 - np.std(active) / np.mean(active))
    else:
        smoothness = 0

    # Max concentration (highest single expiry as % of total)
    max_conc = max(expiry_theta.values(), default=0) / total_theta if total_theta > 0 else 0

    # Stress test: delta at crash/rally prices (gamma approximation)
    crash_price = spot * (1 + STRESS_SCENARIOS['5d_crash'])
    rally_price = spot * (1 + STRESS_SCENARIOS['5d_spike'])
    delta_at_crash = total_delta + total_gamma * (crash_price - spot)
    delta_at_rally = total_delta + total_gamma * (rally_price - spot)

    return {
        'total_theta': total_theta,
        'total_delta': total_delta,
        'total_gamma': total_gamma,
        'total_vega': total_vega,
        'smoothness': smoothness,
        'max_concentration': max_conc,
        'weekly_theta': weekly_theta,
        'expiry_theta': expiry_theta,
        'expiry_delta': expiry_delta,
        'expiry_gamma': expiry_gamma,
        'expiry_contract_count': expiry_contract_count,
        'positions': list(positions),
        'spot': spot,
        'delta_at_crash': delta_at_crash,
        'delta_at_rally': delta_at_rally,
    }


def generate_candidates(portfolio_state, spot, iv, today):
    """Generate all possible trades to evaluate using REAL available strikes and expirations.

    Respects deployment_mode set on portfolio_state (see CENTRAL_PHILOSOPHY.md
    "The wheel is not always on" — empirically validated stand-aside rule):
      - 'WAITING' (z < -0.5, expensive regime): emit only defensive trades
        (CLOSE / LET EXPIRE / TAKE PROFIT / ASSIGNMENT / strike-down rolls).
        No new short exposure.
      - 'TRANSITION' (-0.5 <= z <= 0): full generation, but scoring should
        de-weight aggressive adds.
      - 'ACTIVE' (z > 0): full generation, full wheel deployment.
    """
    r = 0.04
    candidates = []
    # Collapse multi-lot positions at the same (expiry, strike, right) into a
    # single weighted-average lot. Without this, every ladder generator (ROLL,
    # TAKE PROFIT, etc.) emits near-duplicate recs for each open lot — e.g. a
    # 22-contract $11.0P split across 6 fills produced 6 identical-looking
    # "Close 1x (1/22)" TAKE PROFIT recs. Surfaced by user on 2026-05-18.
    _raw_positions = portfolio_state['positions']
    _lot_aggregator: dict[tuple[str, float, str], list] = {}
    for exp_str_pos, strike, right, qty, avg_cost in _raw_positions:
        key = (exp_str_pos, float(strike), right)
        if key in _lot_aggregator:
            cur_q, cur_cost = _lot_aggregator[key]
            _lot_aggregator[key] = [cur_q + qty,
                                    (cur_cost * abs(cur_q) + avg_cost * abs(qty))
                                    / max(1, abs(cur_q) + abs(qty))]
        else:
            _lot_aggregator[key] = [qty, avg_cost]
    positions = [(k[0], k[1], k[2], v[0], v[1])
                 for k, v in _lot_aggregator.items()]

    # Get real available options from the market
    available = get_available_options()

    # Build list of valid expiries.
    # Cycle 161: lowered lower bound 14 → 7 in income-mode. Near-term
    # puts (7-13 DTE) at OTM strikes are excellent income — high theta/$
    # ratio, low gamma exposure at sufficient OTM distance. When we're
    # below 60% of weekly target, every income opportunity counts. The
    # cycle-156 multi-expiry OPENs lost the 6/05 OPEN qΔ +75 the moment
    # 6/05 ticked under 14 DTE — that was a real income trade dropped
    # purely to the floor.
    _ve_income_mode = (
        portfolio_state.get('avg_weekly_theta', 0) <
        portfolio_state.get('target_weekly_income', 1500) * 0.6
    )
    _dte_floor = 7 if _ve_income_mode else 14
    # Cycle 163: raise upper bound 45 → 60 in income-mode so we can reach
    # an additional weekly expiry for income chaining. Current beam tops
    # out at 4 OPENs (one per valid expiry); 7/17 at 55 DTE was excluded
    # by the 45 cap despite having empty-slot strikes available.
    _dte_ceiling = 60 if _ve_income_mode else 45
    valid_expiries = []
    for exp_str in sorted(available.keys()):
        exp_date = datetime.strptime(exp_str, '%Y-%m-%d').date()
        dte = (exp_date - today).days
        if _dte_floor <= dte <= _dte_ceiling:
            valid_expiries.append((exp_str, dte, available[exp_str]))

    if not valid_expiries:
        return candidates

    for exp_str_pos, strike, right, qty, avg_cost in positions:
        expiry = datetime.strptime(exp_str_pos, '%Y-%m-%d').date()
        dte = max((expiry - today).days, 0)
        T = dte / 365.0

        # Skip positions with >45 DTE (user's directional bets)
        if dte > 45:
            continue

        per_share = abs(bs_price(spot, strike, T, r, iv, right))
        intrinsic = max(0, strike - spot) if right == 'P' else max(0, spot - strike)
        extrinsic = max(0, per_share - intrinsic)
        ext_pct = (extrinsic / per_share * 100) if per_share > 0.01 else 0

        # EXPIRE/ASSIGNMENT candidate: positions expiring within 7 days
        if dte <= 7:
            close_theta = abs(bs_theta(spot, strike, T, r, iv, right)) * abs(qty) * 100 if T > 0.001 else 0
            close_delta = qty * bs_delta(spot, strike, T, r, iv, right) * 100
            close_gamma = qty * bs_gamma(spot, strike, T, r, iv) * 100 if T > 0.001 else 0
            close_vega = qty * bs_vega(spot, strike, T, r, iv) * 100 if T > 0.001 else 0

            itm = (right == 'P' and spot < strike) or (right == 'C' and spot > strike)
            # Near the money: within 2% of strike — generate BOTH scenarios
            near_money = abs(spot - strike) / spot < 0.02

            if not itm:
                # OTM — let expire worthless, collect full premium, zero friction
                candidates.append({
                    'type': 'LET EXPIRE',
                    'action': f"Let expire {abs(qty)}x {exp_str_pos} ${strike}{right} ({dte}d, OTM by ${abs(spot-strike):.2f})",
                    'source_exp': exp_str_pos,
                    'source_strike': strike,
                    'source_right': right,
                    'source_dte': dte,
                    'roll_qty': abs(qty),
                    'theta_change': -close_theta,
                    'delta_change': -close_delta,
                    'gamma_change': -close_gamma,
                    'vega_change': -close_vega,
                    'new_extrinsic_total': 0,
                    'ext_pct_old': ext_pct,
                    'n_legs': 0,  # zero friction — no trade needed
                    'detail': f"Expires OTM in {dte}d. Full premium captured. No action needed.",
                    'why': f"${strike}{right} is OTM (spot ${spot:.2f}). Free theta harvest.",
                })

            if itm or near_money:
                # ITM or near money — show assignment impact
                # (both scenarios shown for borderline positions)
                if right == 'P':
                    assign_desc = f"Assigned: +{abs(qty)*100} shares @ ${strike}"
                    assign_delta = abs(qty) * 100  # gain shares
                else:
                    assign_desc = f"Called away: {abs(qty)*100} shares @ ${strike}"
                    assign_delta = -abs(qty) * 100  # lose shares
                itm_otm = "ITM" if itm else f"near money (${abs(spot-strike):.2f} away)"
                candidates.append({
                    'type': 'ASSIGNMENT',
                    'action': f"{'Let assign' if itm else 'Risk: assignment'} {abs(qty)}x {exp_str_pos} ${strike}{right} ({dte}d) — {assign_desc}",
                    'source_exp': exp_str_pos,
                    'source_strike': strike,
                    'source_right': right,
                    'source_dte': dte,
                    'roll_qty': abs(qty),
                    'theta_change': -close_theta,
                    'delta_change': assign_delta - close_delta,  # net: lose option delta + gain/lose share delta
                    'gamma_change': -close_gamma,
                    'vega_change': -close_vega,
                    'new_extrinsic_total': 0,
                    'ext_pct_old': ext_pct,
                    'n_legs': 0,
                    'detail': f"{itm_otm}. {assign_desc}. Net Δ change: {assign_delta - close_delta:+.0f}.",
                    'why': f"${strike}{right} {itm_otm}. Assignment {'likely' if itm else 'possible if crosses strike'}.",
                })

        # Cycle 138 guard: 0-DTE positions should LET EXPIRE (OTM) or
        # ASSIGNMENT (ITM), not be rolled. User principle: "i already
        # sold a few calls and expecting today it expires." Rolling at
        # 0-DTE is high-friction last-minute panic — the natural action
        # is already generated as LET EXPIRE / ASSIGNMENT above.
        if dte <= 0:
            continue  # let_expire (OTM) or assignment (ITM) handles it
        # Cycle 192: near-term OTM → let expire, don't roll. User: "near term
        # OTM roll should be just open a new covered call as the 'to' target."
        # Rolling a near-expiry OTM option pays spread to close something
        # nearly worthless. Better: let expire (free) + sell fresh (one spread).
        # The LET EXPIRE + COVERED CALL candidates already exist independently;
        # the beam chains them without a roll.
        if dte <= 7 and extrinsic < 0.10:
            continue  # remaining value < $10/contract — not worth roll friction

        # Finer ladder so optimizer can pick partial sizes per position,
        # not just half-or-full. Subsequent greedy iterations refine further.
        full_qty = abs(qty)
        if full_qty <= 2:
            roll_qty_options = list(range(1, full_qty + 1))
        elif full_qty <= 5:
            roll_qty_options = sorted(set([1, full_qty // 2, full_qty]))
        elif full_qty <= 10:
            roll_qty_options = sorted(set([1, 3, full_qty // 2, full_qty]))
        else:
            third = max(1, full_qty // 3)
            two_thirds = max(third + 1, 2 * full_qty // 3)
            roll_qty_options = sorted(set([1, 3, full_qty // 2, two_thirds, full_qty]))

        for target_exp_str, actual_dte, chain_strikes in valid_expiries:
            # Pick target strike from REAL available strikes for this expiry
            # Calls: target OTM (5% above spot); Puts: target ATM
            strike_list = chain_strikes['puts'] if right == 'P' else chain_strikes['calls']
            if right == 'C':
                atm_strike = find_nearest_strike(spot * 1.05, strike_list)
            else:
                atm_strike = find_nearest_strike(spot, strike_list)
            if atm_strike is None:
                continue

            # Skip rolling to the same expiry and strike (no-op)
            if target_exp_str == exp_str_pos and atm_strike == strike:
                continue

            # Check liquidity at target strike
            liq = chain_strikes.get('liquidity', {})
            target_liq = liq.get((atm_strike, right), {})
            target_oi = target_liq.get('oi', 0)
            target_vol = target_liq.get('vol', 0)
            target_bid = target_liq.get('bid', 0)
            target_ask = target_liq.get('ask', 0)

            # Skip if no meaningful liquidity (OI < 5 and no volume)
            if target_oi < 5 and target_vol < 1:
                continue

            # Cap roll qty: don't try to roll more than ~20% of OI
            max_liq_qty = max(1, target_oi // 5) if target_oi > 0 else full_qty

            for roll_qty_candidate in roll_qty_options:
                roll_qty = min(roll_qty_candidate, max_liq_qty)
                if roll_qty <= 0:
                    continue

                T_new = actual_dte / 365.0
                new_theta = abs(bs_theta(spot, atm_strike, T_new, r, iv, right)) * roll_qty * 100
                old_theta = abs(bs_theta(spot, strike, T, r, iv, right)) * roll_qty * 100
                new_ext = abs(bs_price(spot, atm_strike, T_new, r, iv, right))
                new_intr = max(0, atm_strike - spot) if right == 'P' else max(0, spot - atm_strike)
                new_pure_ext = max(0, new_ext - new_intr)

                # Greek changes (close old, open new at ATM)
                sign = qty / abs(qty)
                old_delta = sign * bs_delta(spot, strike, T, r, iv, right) * roll_qty * 100
                new_delta = -1 * bs_delta(spot, atm_strike, T_new, r, iv, right) * roll_qty * 100
                old_gamma = sign * bs_gamma(spot, strike, T, r, iv) * roll_qty * 100
                new_gamma = -1 * bs_gamma(spot, atm_strike, T_new, r, iv) * roll_qty * 100
                old_vega = sign * bs_vega(spot, strike, T, r, iv) * roll_qty * 100
                new_vega = -1 * bs_vega(spot, atm_strike, T_new, r, iv) * roll_qty * 100

                # Don't generate roll candidate if:
                # 1. Old position still has >10% extrinsic AND
                # 2. Rolling would LOSE theta (new_theta < old_theta) AND
                # 3. DTE > 5 (not urgently expiring)
                if ext_pct > 10 and (new_theta - old_theta) < 0 and dte > 5:
                    continue

                # Economic data: old/new full option price for roll cost computation
                old_full_price = abs(bs_price(spot, strike, T, r, iv, right))
                new_full_price = abs(bs_price(spot, atm_strike, T_new, r, iv, right))
                roll_net_per_share = new_full_price - old_full_price
                roll_net_total = roll_net_per_share * roll_qty * 100
                old_ext_remaining = extrinsic * roll_qty * 100

                # Cycle 138 hard filter (refined): debit rolls = no, unless:
                #   • short PUT is meaningfully ITM (>2% below strike) AND
                #     not 0-DTE: defending against share assignment, allowed
                #   • dte ≤ 5 AND ITM by > 2%: urgent — allowed
                # CALL-side debit rolls are NEVER allowed — if a covered
                # call is ITM, the wheel says let it assign (cash for
                # shares is the income you wanted). User principle:
                # "i already sold a few calls and expecting today it
                # expires." Don't defend covered calls.
                if roll_net_total < 0:
                    if right == 'C':
                        continue  # never debit-roll a covered call
                    # right == 'P' — only allow if meaningfully ITM
                    itm_amt = strike - spot
                    deep_itm = itm_amt > 0 and itm_amt / max(spot, 0.01) > 0.02
                    src_is_urgent = dte <= 5 and itm_amt > 0
                    if not (deep_itm or src_is_urgent):
                        continue  # skip non-defensive put debit roll

                # Friction model: realistic slippage relative to mid, not half-spread
                # For wide spreads (>20%), we can work limit orders near mid
                # Cap at half-spread (worst case), floor at $0.02/sh slippage
                if target_bid > 0 and target_ask > 0:
                    real_spread = target_ask - target_bid
                    mid = (target_bid + target_ask) / 2
                    spread_pct_local = (real_spread / mid * 100) if mid > 0 else 0
                    half_spread = real_spread / 2
                    # Tighter spreads: full half-spread cost (harder to work)
                    # Wider spreads: small slippage from mid (we can work the order)
                    if spread_pct_local <= 10:
                        per_share_fric = half_spread  # full half-spread
                    elif spread_pct_local <= 25:
                        per_share_fric = max(0.02, half_spread * 0.4)  # ~40% of half-spread
                    else:
                        per_share_fric = max(0.02, min(0.05, half_spread * 0.2))  # mostly mid + small slip
                else:
                    per_share_fric = max(0.01, min(0.05, new_pure_ext * 0.03))

                liq_note = f"OI={target_oi}"
                if roll_qty < roll_qty_candidate:
                    liq_note += f" (liq capped from {roll_qty_candidate})"

                # Compute dollar amounts for transparency
                total_notional = new_full_price * roll_qty * 100  # total option value
                spread_pct = ((target_ask - target_bid) / ((target_ask + target_bid) / 2) * 100) if (target_bid + target_ask) > 0 else 0
                oi_usage_pct = (roll_qty / target_oi * 100) if target_oi > 0 else 100

                # Source leg liquidity (the position being closed)
                source_chain = available.get(exp_str_pos, {})
                source_liq_raw = source_chain.get('liquidity', {}).get((strike, right), {})
                source_oi = source_liq_raw.get('oi', 0)
                source_bid = source_liq_raw.get('bid', 0)
                source_ask = source_liq_raw.get('ask', 0)
                source_spread_pct = ((source_ask - source_bid) / ((source_ask + source_bid) / 2) * 100) if (source_bid + source_ask) > 0 else 0

                # BS fair price for source — back-calculate "what should this trade at"
                source_fair_price = abs(bs_price(spot, strike, T, r, iv, right))
                source_mid = (source_bid + source_ask) / 2 if (source_bid + source_ask) > 0 else source_fair_price
                source_mid_vs_fair = source_mid - source_fair_price  # +ve = market overpricing, -ve = underpricing

                # Source-leg friction: same realistic model
                if source_bid > 0 and source_ask > 0:
                    src_half = (source_ask - source_bid) / 2
                    if source_spread_pct <= 10:
                        src_fric_per_share = src_half
                    elif source_spread_pct <= 25:
                        src_fric_per_share = max(0.02, src_half * 0.4)
                    else:
                        src_fric_per_share = max(0.02, min(0.05, src_half * 0.2))
                    source_friction = src_fric_per_share * roll_qty * 100
                else:
                    source_friction = per_share_fric * roll_qty * 100

                # Target BS fair price
                target_fair_price = new_full_price
                target_mid = (target_bid + target_ask) / 2 if (target_bid + target_ask) > 0 else target_fair_price
                target_mid_vs_fair = target_mid - target_fair_price

                # Total friction = source close + target open
                total_friction = source_friction + per_share_fric * roll_qty * 100

                candidates.append({
                    'type': 'ROLL',
                    'action': f"Roll {roll_qty}x {exp_str_pos} ${strike}{right} -> {target_exp_str} ${atm_strike}{right}",
                    'source_exp': exp_str_pos,
                    'source_strike': strike,
                    'source_right': right,
                    'source_dte': dte,
                    'roll_qty': roll_qty,
                    'target_exp': target_exp_str,
                    'target_strike': atm_strike,
                    'target_dte': actual_dte,
                    'theta_change': new_theta - old_theta,
                    'delta_change': new_delta - old_delta,
                    'gamma_change': new_gamma - old_gamma,
                    'vega_change': new_vega - old_vega,
                    'new_extrinsic_total': new_pure_ext * roll_qty * 100,
                    'old_ext_remaining': old_ext_remaining,
                    'roll_net_total': roll_net_total,
                    'ext_pct_old': ext_pct,
                    'target_oi': target_oi,
                    'target_spread': f"${target_bid:.2f}/${target_ask:.2f}" if target_bid > 0 else "n/a",
                    'n_legs': 2,
                    'liquidity': {
                        'oi': target_oi,
                        'bid': round(target_bid, 2),
                        'ask': round(target_ask, 2),
                        'spread_pct': round(spread_pct, 1),
                        'oi_usage_pct': round(oi_usage_pct, 1),
                        'notional': round(total_notional),
                        'credit_debit': round(roll_net_total),
                        'friction_est': round(total_friction),
                        # BS-fair vs market mid (regardless of fill assumption)
                        'target_fair': round(target_fair_price, 3),
                        'target_mid': round(target_mid, 3),
                        'target_mid_vs_fair': round(target_mid_vs_fair, 3),
                        # Source leg (BTC the position being closed)
                        'source_oi': source_oi,
                        'source_bid': round(source_bid, 2),
                        'source_ask': round(source_ask, 2),
                        'source_spread_pct': round(source_spread_pct, 1),
                        'source_friction': round(source_friction),
                        'source_fair': round(source_fair_price, 3),
                        'source_mid': round(source_mid, 3),
                        'source_mid_vs_fair': round(source_mid_vs_fair, 3),
                    },
                    'detail': f"Ext: {ext_pct:.0f}% | θ: ${old_theta:.1f}→${new_theta:.1f}/d | {'cr' if roll_net_total >= 0 else 'db'}: ${abs(roll_net_total):.0f} | {liq_note} | spread: ${target_bid:.2f}/{target_ask:.2f}",
                    'why': f"{ext_pct:.0f}% extrinsic, {dte}d left. New {actual_dte}d has ${new_pure_ext:.2f}/sh. {liq_note}.",
                })

        # Generate CLOSE candidates for near-worthless OR deep-ITM positions.
        # Cycle 196: user insight — deep ITM short calls have small extrinsic
        # but tie up share collateral. Closing + selling shares frees capital
        # for BOXX (4% risk-free) or new puts (better income). The beam should
        # evaluate this trade-off, not gate it behind a $50 threshold.
        _close_total_ext = extrinsic * abs(qty) * 100
        _is_deep_itm_short_call = (right == 'C' and qty < 0 and intrinsic > 0.3
                                    and extrinsic < 0.40)
        _close_eligible = (_close_total_ext < 50) or _is_deep_itm_short_call
        if _close_eligible:
            close_theta = abs(bs_theta(spot, strike, T, r, iv, right)) * abs(qty) * 100
            close_delta = qty * bs_delta(spot, strike, T, r, iv, right) * 100
            close_gamma = qty * bs_gamma(spot, strike, T, r, iv) * 100
            close_vega = qty * bs_vega(spot, strike, T, r, iv) * 100

            # Find nearest ATM strike for roll suggestion in detail
            available = get_available_options()
            all_strikes_pool = []
            for av_exp, av_chain in available.items():
                all_strikes_pool.extend(av_chain['puts'] if right == 'P' else av_chain['calls'])
            all_strikes_pool = sorted(set(all_strikes_pool)) if all_strikes_pool else [spot]
            nearest_atm_strike = find_nearest_strike(spot, all_strikes_pool)

            candidates.append({
                'type': 'CLOSE',
                'action': f"Close {abs(qty)}x {exp_str_pos} ${strike}{right}",
                'source_exp': exp_str_pos,
                'source_strike': strike,
                'source_right': right,
                'theta_change': -close_theta,
                'delta_change': -close_delta,
                'gamma_change': -close_gamma,
                'vega_change': -close_vega,
                'new_extrinsic_total': 0,
                'n_legs': 1,
                'detail': f"${_close_total_ext:.0f} extrinsic left | frees ${abs(qty) * strike * 100:,.0f} share collateral for BOXX/puts",
                'why': (
                    f"Deep ITM short call mean-reversion hedge: extrinsic ${extrinsic:.2f}/sh is tiny vs "
                    f"variance risk. If UNG mean-reverts back, capped upside ($0.30/sh max) won't offset "
                    f"share losses. Close + sell {abs(qty)*100} shares locks ${spot:.2f} and converts to "
                    f"cash for BOXX (~${abs(qty) * strike * 100 * 0.04 / 365 * dte:.0f} over {dte}d) or "
                    f"fresh OTM puts."
                    if _is_deep_itm_short_call else
                    "Theta exhausted. Free up margin."),
            })

            # Cycle 198: Synthetic early-assignment candidate. Combines CLOSE
            # short call + SELL shares into ONE trade = lock in market price
            # NOW, avoid mean-reversion variance on shares. Net cash = strike
            # × 100 × qty + extrinsic captured. Same outcome as natural
            # assignment but at TODAY's price, not at expiry.
            if _is_deep_itm_short_call:
                _portfolio_shares = int(portfolio_state.get('shares', SHARES) or SHARES)
                _share_qty_to_sell = min(_portfolio_shares, abs(qty) * 100)
                if _share_qty_to_sell >= 100:
                    _close_cost = per_share * abs(qty) * 100  # buy back the call (per_share is the current price)
                    _share_proceeds = spot * _share_qty_to_sell
                    _net_cash = _share_proceeds - _close_cost
                    # Mean reversion variance: 1σ adverse move over remaining DTE
                    _adverse_move = iv * (dte / 365.0) ** 0.5 * spot
                    _adverse_share_loss = _adverse_move * _share_qty_to_sell * 0.5  # half a sigma
                    candidates.append({
                        'type': 'CLOSE',
                        'action': f"Close {abs(qty)}x ${strike}{right} + Sell {_share_qty_to_sell} shares (lock ${spot:.2f})",
                        'source_exp': exp_str_pos,
                        'source_strike': strike,
                        'source_right': right,
                        'theta_change': -close_theta,
                        'delta_change': -close_delta - _share_qty_to_sell,
                        'gamma_change': -close_gamma,
                        'vega_change': -close_vega,
                        'new_extrinsic_total': 0,
                        'n_legs': 2,
                        'shares_sold': _share_qty_to_sell,
                        'detail': f"Synthetic early assign at ${spot:.2f} | net cash ${_net_cash:,.0f} | avoids 1σ share variance ±${_adverse_share_loss:,.0f}",
                        'why': f"Mean-reversion hedge: lock in rally gain TODAY. Synthetic early assignment captures ${spot - strike:.2f}/sh above strike that natural assignment would forfeit at expiry.",
                    })

    # TAKE PROFIT: positions where we've captured a meaningful fraction
    # of premium AND keeping them open is hurting (theta nearly exhausted).
    # Cycle 159: tighter gate in income-mode. CENTRAL_PHILOSOPHY anti-pattern:
    # "Closing a profitable short put early just to redeploy at a slightly
    # better strike (friction > marginal income)". With UNG income at 23%
    # of target, every theta-producing position is precious — don't close
    # winners at 40% if there's still meaningful theta to harvest. Require
    # 60%+ profit OR ≤7 DTE (theta nearly done either way).
    _tp_income_mode = (
        portfolio_state.get('avg_weekly_theta', 0) <
        portfolio_state.get('target_weekly_income', 1500) * 0.6
    )
    _tp_min_profit = 60 if _tp_income_mode else 40
    for exp_str_pos, strike, right, qty, avg_cost in positions:
        expiry = datetime.strptime(exp_str_pos, '%Y-%m-%d').date()
        dte = max((expiry - today).days, 0)
        if dte > 45 or dte < 1:
            continue
        T = dte / 365.0
        current_value = abs(bs_price(spot, strike, T, r, iv, right))
        # avg_cost is per contract; current_value is per share
        premium_collected = avg_cost / 100 if avg_cost > 1 else avg_cost  # normalize
        if premium_collected > 0.01:
            profit_pct = (premium_collected - current_value) / premium_collected * 100
            # In income mode: 60%+ profit OR ≤7 DTE (theta nearly done either way)
            _passes = (
                profit_pct > _tp_min_profit
                or (dte <= 7 and profit_pct > 40)
            )
            if _passes:
                # Partial profit-taking ladder: standard wheel practice is to scale out
                # rather than dump everything. Emit several sizes so greedy can pick.
                full_q = abs(qty)
                if full_q <= 2:
                    tp_qty_options = [full_q]
                elif full_q <= 5:
                    tp_qty_options = sorted(set([1, full_q // 2, full_q]))
                else:
                    tp_qty_options = sorted(set([1, full_q // 3, full_q // 2, full_q]))

                for tp_qty in tp_qty_options:
                    if tp_qty <= 0:
                        continue
                    close_theta = abs(bs_theta(spot, strike, T, r, iv, right)) * tp_qty * 100
                    close_delta = (qty / abs(qty)) * tp_qty * bs_delta(spot, strike, T, r, iv, right) * 100 if qty != 0 else 0
                    close_gamma = (qty / abs(qty)) * tp_qty * bs_gamma(spot, strike, T, r, iv) * 100 if qty != 0 else 0
                    close_vega = (qty / abs(qty)) * tp_qty * bs_vega(spot, strike, T, r, iv) * 100 if qty != 0 else 0
                    remaining_value = current_value * tp_qty * 100
                    qty_note = f" ({tp_qty}/{full_q})" if tp_qty < full_q else ""
                    candidates.append({
                        'type': 'TAKE PROFIT',
                        'action': f"Close {tp_qty}x {exp_str_pos} ${strike}{right} ({profit_pct:.0f}% profit){qty_note}",
                        'source_exp': exp_str_pos,
                        'source_strike': strike,
                        'source_right': right,
                        'roll_qty': tp_qty,
                        'theta_change': -close_theta,
                        'delta_change': -close_delta,
                        'gamma_change': -close_gamma,
                        'vega_change': -close_vega,
                        'new_extrinsic_total': 0,
                        'n_legs': 1,
                        'ext_pct_old': 100,
                        'detail': f"{profit_pct:.0f}% profit captured | ${remaining_value:.0f} left to collect vs ${close_theta:.1f}/d theta",
                        'why': f"Collected {profit_pct:.0f}% of premium. Close winner, redeploy at better strike.",
                    })

    # OPEN: independent new puts or calls (not just ATM puts)
    # When delta below target → open puts; when delta above target → open calls
    target_delta_val, _, _ = compute_target_delta(spot)
    current_delta = portfolio_state['total_delta']
    delta_gap = current_delta - target_delta_val

    # Track already-held contracts per expiry for incremental qty calculation
    held_by_expiry = {}  # {expiry: {(strike, right): qty}}
    expiry_contracts = {}  # {expiry: total_contracts}
    for exp_s, strike_s, right_s, qty_s, _ in positions:
        held_by_expiry.setdefault(exp_s, {})[(strike_s, right_s)] = abs(qty_s)
        expiry_contracts[exp_s] = expiry_contracts.get(exp_s, 0) + abs(qty_s)

    # Compute average contracts per expiry (for waterfall target)
    future_expiries = [e for e in expiry_contracts
                       if (datetime.strptime(e, '%Y-%m-%d').date() - today).days > 7]
    avg_contracts_per_expiry = (sum(expiry_contracts.get(e, 0) for e in future_expiries)
                                / max(1, len(future_expiries)))

    # Generate ADD/STRANGLE candidates using REAL expiries, strikes, and liquidity
    for target_exp_str, actual_dte, chain_strikes in valid_expiries:
        T_add = actual_dte / 365.0
        put_strikes = chain_strikes['puts']
        call_strikes = chain_strikes['calls']
        liq = chain_strikes.get('liquidity', {})

        atm_put = find_nearest_strike(spot, put_strikes)
        atm_call = find_nearest_strike(spot * 1.05, call_strikes)

        # ── Generate covered-call candidates across MULTIPLE strikes ──
        # Cycle 151: same multi-strike treatment as puts got in cycle 144.
        # Previously emitted only one candidate per expiry at spot×1.05;
        # the beam never saw ATM (max premium) or deep-OTM (room to run)
        # alternatives. Income mode is at 23% of target ($347 vs $1500),
        # so we need every actionable income trade visible. Covered calls
        # are ESPECIALLY important because they reduce delta (currently
        # +6,300 from shares + puts, well over target).
        total_existing_calls = sum(
            abs(q) for _, s, r, q, _ in positions if r == 'C' and q < 0
        )
        max_covered_calls = SHARES // 100
        calls_available = max_covered_calls - total_existing_calls
        # Strike band: ATM → 15% OTM, intersected with chain.
        # Cycle 174: expanded to include ITM calls (spot×0.95 → spot×1.00).
        # User insight: "sell ITM calls to reduce stocks" — high premium,
        # high assignment probability, reduces delta. Low extrinsic but
        # useful when portfolio is over delta target. Beam evaluates
        # tradeoff via qΔ (delta_gap improvement vs theta loss).
        _call_strike_band = sorted({
            K for K in call_strikes
            if spot * 0.95 <= K <= spot * 1.15
        })
        if calls_available > 0 and _call_strike_band:
            for K_c in _call_strike_band:
                call_liq = liq.get((K_c, 'C'), {})
                call_oi = call_liq.get('oi', 0)
                _existing_same_slot = held_by_expiry.get(target_exp_str, {}).get(
                    (K_c, 'C'), 0)
                if call_oi < 30:
                    continue
                per_contract_delta = abs(bs_delta(spot, K_c, T_add, r, iv, 'C')) * 100
                ideal_qty = max(1, min(calls_available, max(1, call_oi // 5)))
                if delta_gap > 0:
                    delta_driven = max(1, int(abs(delta_gap) / 2 / max(1, per_contract_delta)))
                    ideal_qty = min(ideal_qty, delta_driven)
                # Don't exceed remaining slot capacity
                # No per-strike cap — sized by shares available + delta
                if ideal_qty <= 0:
                    continue
                open_qty = min(ideal_qty, calls_available)
                call_theta = abs(bs_theta(spot, K_c, T_add, r, iv, 'C')) * open_qty * 100
                call_delta = -open_qty * bs_delta(spot, K_c, T_add, r, iv, 'C') * 100
                call_gamma = -open_qty * bs_gamma(spot, K_c, T_add, r, iv) * 100
                call_vega = -open_qty * bs_vega(spot, K_c, T_add, r, iv) * 100
                call_ext = abs(bs_price(spot, K_c, T_add, r, iv, 'C')) - max(0, spot - K_c)
                if call_ext <= 0.01:
                    continue
                c_bid = call_liq.get('bid', 0)
                c_ask = call_liq.get('ask', 0)
                # Cycle 192b: premium must beat BOXX risk-free on margin consumed.
                # CCs use shares not margin, but still have opportunity cost
                # (shares could be sold → cash → BOXX). Use a simpler $0.05 floor.
                if c_bid < 0.05:
                    continue
                spread_note = f"${c_bid:.2f}/${c_ask:.2f}" if c_bid > 0 else "n/a"
                otm_pct = max(0.0, (K_c - spot) / spot) * 100
                otm_tag = f" {otm_pct:.0f}%OTM" if otm_pct >= 1.5 else ""
                candidates.append({
                    'type': 'COVERED CALL',
                    'action': f"Sell {open_qty}x {target_exp_str} ${K_c}C covered ({actual_dte} DTE{otm_tag})",
                    'target_exp': target_exp_str,
                    'target_strike': K_c,
                    'add_qty': open_qty,
                    'theta_change': call_theta,
                    'delta_change': call_delta,
                    'gamma_change': call_gamma,
                    'vega_change': call_vega,
                    'new_extrinsic_total': call_ext * open_qty * 100,
                    'n_legs': 1,
                    'liquidity': {
                        'oi': call_oi,
                        'bid': round(c_bid, 2),
                        'ask': round(c_ask, 2),
                        'spread_pct': round((c_ask - c_bid) / ((c_ask + c_bid) / 2) * 100, 1) if (c_bid + c_ask) > 0 else 0,
                        'oi_usage_pct': round(open_qty / call_oi * 100, 1) if call_oi > 0 else 100,
                        'notional': round(call_ext * open_qty * 100),
                        'credit_debit': round(call_ext * open_qty * 100),
                        'friction_est': round((c_ask - c_bid) / 2 * open_qty * 100) if (c_bid + c_ask) > 0 else 0,
                    },
                    'detail': f"Covered call | +${call_theta:.1f}/d θ | ${call_ext:.2f}/sh | Δ{call_delta:+.0f} | OI={call_oi} | spread: {spread_note}",
                    'why': f"Θ waterfall + delta mgmt (multi-strike, cycle 151). OI={call_oi}.",
                })

        # ── Generate sell-put candidates across MULTIPLE strikes ──
        # Cycle 144: previously emitted only ONE candidate per expiry
        # (ATM), so the beam never saw OTM-put options that capture less
        # premium but add much less delta. For a long-delta portfolio
        # (e.g. +6,308 from shares + existing puts) the optimal new short
        # put is often 5-10% OTM, not ATM. Let the beam rank — generate
        # the menu.
        existing_contracts_here = expiry_contracts.get(target_exp_str, 0)
        # Candidate strikes: spot×0.85 → spot×1.00, intersected with what
        # the chain actually offers.
        # Cycle 174: expanded to include ITM puts (spot×1.00 → spot×1.05).
        # User insight: "sell ITM puts to add stocks" — high premium,
        # near-certain assignment, accumulates shares at discount
        # (strike - premium). Low extrinsic but useful when building
        # position. Beam evaluates gamma/theta tradeoff via qΔ.
        _put_strike_band = sorted({
            K for K in put_strikes
            if spot * 0.85 <= K <= spot * 1.05
        })
        # Always include ATM even if it lands just outside the band
        if atm_put is not None and atm_put not in _put_strike_band:
            _put_strike_band.append(atm_put)
            _put_strike_band.sort()
        for K_p in _put_strike_band:
            put_liq = liq.get((K_p, 'P'), {})
            put_oi = put_liq.get('oi', 0)
            # Per-strike dedupe — don't pile more onto a slot already 5+ deep
            _existing_same_slot = held_by_expiry.get(target_exp_str, {}).get(
                (K_p, 'P'), 0)
            # Cycle 179b: no hard cap per strike. User: "we assume good fill
            # and such small size market maker can jump in." Evaluator's qΔ
            # handles concentration via dd_penalty + friction + smoothness.
            # Only gate: OI >= 30 (minimum liquidity) and don't re-propose
            # at a strike that already has positions (beam dedupes via
            # used_targets). Existing positions are visible to the evaluator.
            if put_oi < 30:
                continue
            per_put_delta = abs(bs_delta(spot, K_p, T_add, r, iv, 'P')) * 100
            # Waterfall target + per-strike incremental sizing
            target_contracts = max(3, int(avg_contracts_per_expiry * 0.7))
            incremental_qty = max(0, target_contracts - existing_contracts_here)
            incremental_qty = min(incremental_qty, 5 - _existing_same_slot)
            if incremental_qty <= 0:
                # Cycle 156: income-mode override. When avg_weekly_theta is
                # below 60% of target, generate OPEN candidates regardless
                # of waterfall — beam needs cross-expiry options to chain
                # multiple income trades. `used_targets` blocks same-expiry
                # repeat picks, so without other expiries the beam can only
                # take ONE OPEN total.
                # Count PUTS only for the gate — `existing_contracts_here`
                # includes calls which is the wrong basis for put OPEN
                # decisions. 6/18 had 29 (mostly calls) blocking all puts
                # despite having 2 puts.
                _put_contracts_here = sum(
                    qty for (k, rt), qty in held_by_expiry.get(target_exp_str, {}).items()
                    if rt == 'P'
                )
                _income_mode = (
                    portfolio_state.get('avg_weekly_theta', 0) <
                    portfolio_state.get('target_weekly_income', 1500) * 0.6
                )
                if _income_mode and _put_contracts_here < 15:
                    # Cycle 179: OI-based sizing, not hardcoded cap.
                    # Max at 20% of OI minus existing. Evaluator handles
                    # concentration via qΔ components.
                    incremental_qty = max(1, 50 - _existing_same_slot)  # no hard cap, evaluator gates
                # Delta-driven override: long-delta portfolio + room at strike
                elif delta_gap < -300 and _put_contracts_here < 15:
                    delta_driven = max(1, int(abs(delta_gap) / 2 / max(1, per_put_delta)))
                    incremental_qty = min(50 - _existing_same_slot, delta_driven)
                if incremental_qty <= 0:
                    continue
            if delta_gap < 0:
                delta_driven = int(abs(delta_gap) / 2 / max(1, per_put_delta))
                incremental_qty = min(incremental_qty, max(1, delta_driven))
            open_qty = max(1, min(incremental_qty, max(1, put_oi // 5)))
            put_theta = abs(bs_theta(spot, K_p, T_add, r, iv, 'P')) * open_qty * 100
            put_delta = -open_qty * bs_delta(spot, K_p, T_add, r, iv, 'P') * 100
            put_gamma = -open_qty * bs_gamma(spot, K_p, T_add, r, iv) * 100
            put_vega = -open_qty * bs_vega(spot, K_p, T_add, r, iv) * 100
            put_ext = abs(bs_price(spot, K_p, T_add, r, iv, 'P')) - max(0, K_p - spot)
            if put_ext <= 0.01:
                continue
            p_bid = put_liq.get('bid', 0)
            p_ask = put_liq.get('ask', 0)
            # Cycle 181: minimum premium economics gate. User: "sell 12 puts
            # but only collect $100 premium and you call it a good choice?"
            # Skip candidates with zero bid or premium < $0.05/share ($5/contract).
            # Cycle 192b: minimum premium must beat BOXX risk-free alternative.
            # Margin = strike × 100 - premium. BOXX earns ~4% APR on that margin.
            # If premium < BOXX return over the DTE period, the trade DESTROYS
            # value vs parking the margin in risk-free.
            _ws_margin_per = max(1, K_p * 100 - p_bid * 100)
            _boxx_return = _ws_margin_per * 0.04 / 365 * actual_dte
            if p_bid * 100 < _boxx_return:
                continue  # premium doesn't beat risk-free — skip
                continue
            # Cycle 182: removed premium-based qty cap. It was backwards —
            # capped ATM options ($80/contract) to qty=1 while the intent
            # was to prevent penny-premium garbage. The bid >= $0.05 filter
            # above handles penny prevention. Let the evaluator's qΔ
            # (dd_penalty, friction, gamma) handle sizing naturally.
            spread_note = f"${p_bid:.2f}/${p_ask:.2f}" if p_bid > 0 else "n/a"
            otm_pct = max(0.0, (spot - K_p) / spot) * 100
            otm_tag = f" {otm_pct:.0f}%OTM" if otm_pct >= 1.5 else ""
            candidates.append({
                'type': 'OPEN',
                'action': f"Sell {open_qty}x {target_exp_str} ${K_p}P ({actual_dte} DTE{otm_tag})",
                'target_exp': target_exp_str,
                'target_strike': K_p,
                'add_qty': open_qty,
                'theta_change': put_theta,
                'delta_change': put_delta,
                'gamma_change': put_gamma,
                'vega_change': put_vega,
                'new_extrinsic_total': put_ext * open_qty * 100,
                'n_legs': 1,
                'liquidity': {
                    'oi': put_oi,
                    'bid': round(p_bid, 2),
                    'ask': round(p_ask, 2),
                    'spread_pct': round((p_ask - p_bid) / ((p_ask + p_bid) / 2) * 100, 1) if (p_bid + p_ask) > 0 else 0,
                    'oi_usage_pct': round(open_qty / put_oi * 100, 1) if put_oi > 0 else 100,
                    'notional': round(put_ext * open_qty * 100),
                    'credit_debit': round(put_ext * open_qty * 100),
                    'friction_est': round((p_ask - p_bid) / 2 * open_qty * 100) if (p_bid + p_ask) > 0 else 0,
                },
                'detail': f"+${put_theta:.1f}/d θ | ${put_ext:.2f}/sh | Δ{abs(put_delta):+.0f} | OI={put_oi} | spread: {spread_note}",
                'why': f"Θ waterfall + multi-strike menu (cycle 144). OI={put_oi}.",
            })

        # (BUY/SELL shares moved outside expiry loop — see below)

        # Buy puts (hedge) — fire on EITHER:
        #   (a) delta significantly over target (original crash-protection case), OR
        #   (b) total_gamma is meaningfully negative AND the portfolio is
        #       projecting a DD breach. Cycle 57 revealed gamma_convexity
        #       was -$18,859 (half of tail loss) yet no BUY PUT candidate
        #       was generated because delta_gap was inside the 500 gate.
        #       Sizing-by-delta would also be wrong for the gamma case; we
        #       size by gamma shortfall when that trigger fires.
        _total_gamma = float(portfolio_state.get('total_gamma', 0.0) or 0.0)
        _cvar_drop = float(portfolio_state.get('cvar_30d_5pct_drop', 0.0) or 0.0)
        _capital = float(portfolio_state.get('capital_base', 100_000) or 100_000)
        # Approximate DD breach: if 5%-CVaR drop produces a tail loss > 10%
        # of capital from gamma alone, that's a gamma-driven breach.
        _gamma_loss = 0.5 * abs(_total_gamma) * (_cvar_drop ** 2) if _cvar_drop > 0 else 0.0
        _gamma_trigger = (_total_gamma < -1000.0 and _gamma_loss > 0.05 * _capital)
        if (delta_gap > 500 or _gamma_trigger) and atm_put is not None:
            put_liq = liq.get((atm_put, 'P'), {})
            put_oi = put_liq.get('oi', 0)
            if put_oi >= 5:
                per_put_delta = abs(bs_delta(spot, atm_put, T_add, r, iv, 'P')) * 100
                per_put_gamma = abs(bs_gamma(spot, atm_put, T_add, r, iv)) * 100
                if _gamma_trigger and per_put_gamma > 0:
                    # Size to absorb ~half the gamma shortfall — full absorption
                    # in one expiry over-concentrates timing risk; one or two
                    # adds across expiries is more robust.
                    ideal_qty = max(1, int(abs(_total_gamma) / 2 / per_put_gamma))
                else:
                    ideal_qty = max(1, int(abs(delta_gap) / 2 / max(1, per_put_delta)))
                buy_qty = min(ideal_qty, max(1, put_oi // 5))
                put_cost = abs(bs_price(spot, atm_put, T_add, r, iv, 'P')) * buy_qty * 100
                put_delta = buy_qty * bs_delta(spot, atm_put, T_add, r, iv, 'P') * 100
                put_gamma = buy_qty * bs_gamma(spot, atm_put, T_add, r, iv) * 100
                put_theta = -abs(bs_theta(spot, atm_put, T_add, r, iv, 'P')) * buy_qty * 100
                put_vega = buy_qty * bs_vega(spot, atm_put, T_add, r, iv) * 100
                _why_extra = "Gamma-driven DD breach" if _gamma_trigger else "Crash protection"
                candidates.append({
                    'type': 'BUY PUT',
                    'action': f"Buy {buy_qty}x {target_exp_str} ${atm_put}P ({actual_dte} DTE)",
                    'target_exp': target_exp_str,
                    'target_strike': atm_put,
                    'add_qty': buy_qty,
                    'theta_change': put_theta,
                    'delta_change': put_delta,
                    'gamma_change': put_gamma,
                    'vega_change': put_vega,
                    'new_extrinsic_total': 0,
                    'n_legs': 1,
                    'detail': f"Hedge | cost ${put_cost:.0f} | Δ{put_delta:+.0f} | Γ{put_gamma:+.0f} | OI={put_oi}",
                    'why': f"{_why_extra}. Costs ${abs(put_theta):.1f}/d theta. OI={put_oi}.",
                })

        # ADD put at ATM (waterfall gap filling) — incremental + liquidity checked
        if atm_put is not None:
            put_liq = liq.get((atm_put, 'P'), {})
            put_oi = put_liq.get('oi', 0)
            # Only suggest ADD if this expiry is below average
            if put_oi >= 5 and existing_contracts_here < avg_contracts_per_expiry:
                gap_contracts = max(1, int(avg_contracts_per_expiry - existing_contracts_here))
                add_qty = min(gap_contracts, 3, max(1, put_oi // 5))
                add_theta = abs(bs_theta(spot, atm_put, T_add, r, iv, 'P')) * add_qty * 100
                add_delta = -add_qty * bs_delta(spot, atm_put, T_add, r, iv, 'P') * 100
                add_gamma = -add_qty * bs_gamma(spot, atm_put, T_add, r, iv) * 100
                add_vega = -add_qty * bs_vega(spot, atm_put, T_add, r, iv) * 100
                add_ext = abs(bs_price(spot, atm_put, T_add, r, iv, 'P')) - max(0, atm_put - spot)

                candidates.append({
                    'type': 'ADD',
                    'action': f"Sell {add_qty}x {target_exp_str} ${atm_put}P ({actual_dte} DTE)",
                    'target_exp': target_exp_str,
                    'target_strike': atm_put,
                    'add_qty': add_qty,
                    'theta_change': add_theta,
                    'delta_change': add_delta,
                    'gamma_change': add_gamma,
                    'vega_change': add_vega,
                    'new_extrinsic_total': add_ext * add_qty * 100,
                    'n_legs': 1,
                    'detail': f"+${add_theta:.1f}/d θ | ${add_ext:.2f}/sh | OI={put_oi} ({actual_dte} DTE)",
                    'why': f"Adds theta to fill waterfall gap. OI={put_oi}.",
                })

        # Cycle 186b: COVERED STRANGLE — sell put + sell covered call as ONE
        # trade. Fixed the old STRANGLE's P0 bug (naked call) by requiring
        # share coverage for the call leg. The beam evaluates COMBINED
        # Greeks — the delta nearly cancels (put +47, call -40 = net +7),
        # so delta_gap doesn't penalize the pair like it does the CC alone.
        # This is the natural wheel trade: collect premium from both sides,
        # delta-neutral, call covered by shares.
        if calls_available > 0 and atm_put is not None:
            # ATM put + 1-strike-OTM call
            K_strangle_put = atm_put
            K_strangle_call = find_nearest_strike(spot * 1.05, call_strikes) if call_strikes else None
            if K_strangle_call is not None:
                _s_put_liq = liq.get((K_strangle_put, 'P'), {})
                _s_call_liq = liq.get((K_strangle_call, 'C'), {})
                _s_p_oi = _s_put_liq.get('oi', 0)
                _s_c_oi = _s_call_liq.get('oi', 0)
                _s_p_bid = _s_put_liq.get('bid', 0)
                _s_c_bid = _s_call_liq.get('bid', 0)
                if _s_p_oi >= 30 and _s_c_oi >= 30 and _s_p_bid >= 0.05 and _s_c_bid >= 0.05:
                    _s_qty = min(3, calls_available)  # conservative sizing
                    _s_p_theta = abs(bs_theta(spot, K_strangle_put, T_add, r, iv, 'P')) * _s_qty * 100
                    _s_c_theta = abs(bs_theta(spot, K_strangle_call, T_add, r, iv, 'C')) * _s_qty * 100
                    _s_p_delta = -_s_qty * bs_delta(spot, K_strangle_put, T_add, r, iv, 'P') * 100
                    _s_c_delta = -_s_qty * bs_delta(spot, K_strangle_call, T_add, r, iv, 'C') * 100
                    _s_p_gamma = -_s_qty * bs_gamma(spot, K_strangle_put, T_add, r, iv) * 100
                    _s_c_gamma = -_s_qty * bs_gamma(spot, K_strangle_call, T_add, r, iv) * 100
                    _s_p_prem = abs(bs_price(spot, K_strangle_put, T_add, r, iv, 'P'))
                    _s_c_prem = abs(bs_price(spot, K_strangle_call, T_add, r, iv, 'C'))
                    _s_mid_p = (_s_put_liq.get('bid', 0) + _s_put_liq.get('ask', 0)) / 2
                    _s_mid_c = (_s_call_liq.get('bid', 0) + _s_call_liq.get('ask', 0)) / 2
                    candidates.append({
                        'type': 'OPEN',
                        'action': f"Covered strangle {_s_qty}x {target_exp_str} ${K_strangle_put}P/${K_strangle_call}C ({actual_dte} DTE)",
                        'target_exp': target_exp_str,
                        'target_strike': K_strangle_put,
                        'add_qty': _s_qty,
                        'theta_change': _s_p_theta + _s_c_theta,
                        'delta_change': _s_p_delta + _s_c_delta,  # nearly cancels!
                        'gamma_change': _s_p_gamma + _s_c_gamma,
                        'vega_change': 0,
                        'new_extrinsic_total': (_s_p_prem + _s_c_prem) * _s_qty * 100,
                        'n_legs': 2,
                        'liquidity': {
                            'oi': min(_s_p_oi, _s_c_oi),
                            'bid': round(_s_mid_p + _s_mid_c, 2),
                            'ask': round(_s_mid_p + _s_mid_c, 2),
                            'spread_pct': 0,
                            'oi_usage_pct': 0,
                            'notional': round((_s_p_prem + _s_c_prem) * _s_qty * 100),
                            'credit_debit': round((_s_p_prem + _s_c_prem) * _s_qty * 100),
                            'friction_est': 0,
                        },
                        'detail': f"Covered strangle: P ${_s_p_prem:.2f} + C ${_s_c_prem:.2f} | θ ${_s_p_theta + _s_c_theta:.1f}/d | netΔ {_s_p_delta + _s_c_delta:+.0f}",
                        'why': f"Put + covered call as one trade. Delta nearly cancels ({_s_p_delta:+.0f} + {_s_c_delta:+.0f} = {_s_p_delta + _s_c_delta:+.0f}). Double theta, single margin.",
                    })

    # ── BUY/SELL shares (once, outside expiry loop) ──
    # Cycle 195: generate SELL SHARES at multiple sizes always. Let the
    # evaluator decide via principled comparison (delta target, gamma load,
    # margin freed for puts, CC collateral remaining). User: "the share
    # drag should not be the fact, it should be after the fact... the
    # decision should be more pure."
    _current_shares = int(portfolio_state.get('shares', SHARES) or SHARES)
    _short_calls = sum(abs(qty) for _exp, _K, rt, qty, _avg in positions
                       if rt == 'C' and qty < 0)
    _cc_collateral_needed = _short_calls * 100  # 100 shares per CC
    _max_sellable = max(0, _current_shares - _cc_collateral_needed)
    if _max_sellable >= 100:
        # Ladder: 25%, 50%, 75%, 100% of sellable, rounded to 100
        for _frac in [0.25, 0.5, 0.75, 1.0]:
            share_qty = max(100, int(round(_max_sellable * _frac / 100) * 100))
            if share_qty > _max_sellable:
                share_qty = _max_sellable
            loss_per_share = SHARE_AVG - spot
            pct_label = f"{int(_frac*100)}%"
            candidates.append({
                'type': 'SELL SHARES',
                'action': f"Sell {share_qty} UNG shares ({pct_label} of sellable)",
                'add_qty': share_qty,
                'theta_change': 0,
                'delta_change': -share_qty,
                'gamma_change': 0,
                'vega_change': 0,
                'new_extrinsic_total': 0,
                'n_legs': 1,
                'detail': f"Δ-{share_qty} | proceeds ${share_qty*spot:,.0f} | freed margin for puts/BOXX 4%",
                'why': f"Convert {share_qty} shares to cash at ${spot:.2f}. Cash earns BOXX risk-free + enables more cash-secured puts. Keeps {_current_shares - share_qty} shares for CC collateral.",
            })
        # Sanity check: don't dedup identical entries (already filtered by 100-share rounding)

    if delta_gap > 500:  # legacy emergency trigger preserved
        pass

    if delta_gap < -500:
        share_qty = min(max(100, int(round(abs(delta_gap) / 100) * 100)), int(abs(delta_gap) * 1.2))
        candidates.append({
            'type': 'BUY SHARES',
            'action': f"Buy {share_qty} UNG shares",
            'add_qty': share_qty,
            'theta_change': 0,
            'delta_change': share_qty,
            'gamma_change': 0,
            'vega_change': 0,
            'new_extrinsic_total': 0,
            'n_legs': 1,
            'detail': f"Δ+{share_qty} | cost ${share_qty * spot:,.0f} | ~${share_qty * 0.01:.0f} friction",
            'why': f"Delta emergency. Gap is {abs(delta_gap):.0f} shares.",
        })

    # ── Stand-aside filter (empirically validated, see CENTRAL_PHILOSOPHY) ──
    # When in WAITING mode (expensive regime, z < -0.5), prune all candidates
    # that ADD new short exposure or roll for credit. Keep only defensive
    # trades: close winners, let OTM expire, take assignment, strike-down rolls.
    mode = portfolio_state.get('deployment_mode', 'ACTIVE')
    # Cycle 180 P1 fix: TRANSITION mode now also enforces reduced aggression
    # per CENTRAL_PHILOSOPHY: "scale-down position size, no new aggressive
    # entries." Previously only WAITING pruned candidates; TRANSITION was
    # treated identically to ACTIVE.
    if mode in ('WAITING', 'TRANSITION'):
        DEFENSIVE_TYPES = {'CLOSE', 'LET EXPIRE', 'TAKE PROFIT', 'ASSIGNMENT', 'SELL SHARES', 'BUY PUT'}
        pruned = []
        for c in candidates:
            t = c.get('type')
            if t in DEFENSIVE_TYPES:
                pruned.append(c)
            elif t == 'ROLL':
                old_k = c.get('source_strike', 0)
                new_k = c.get('target_strike', 0)
                right = c.get('source_right', 'P')
                is_defensive_roll = (
                    (right == 'P' and new_k < old_k) or
                    (right == 'C' and new_k > old_k)
                )
                if is_defensive_roll:
                    pruned.append(c)
            elif mode == 'TRANSITION' and t in ('OPEN', 'COVERED CALL'):
                # TRANSITION allows EXISTING position maintenance but
                # scales down new entries: only single-contract OPENs
                # (no aggressive multi-contract fills).
                qty = abs(c.get('add_qty', c.get('qty', 1)))
                if qty <= 1:
                    pruned.append(c)
            # WAITING: ADD_TYPES dropped entirely
        candidates = pruned

    # Marginal sizing curves: for ROLL/OPEN/ADD/COVERED CALL with qty > 2,
    # emit a qty ladder [1, qty//3, qty//2, qty] so the optimizer can pick
    # the rung where marginal score is best (concentration penalty + Kelly
    # diminishing returns mean smaller sizes can dominate). TAKE PROFIT
    # already had its own ladder generator; STRANGLE/LET EXPIRE/ASSIGNMENT
    # stay single-qty (combo / fixed events).
    _SCALED_FIELDS = ('theta_change', 'delta_change', 'gamma_change',
                      'vega_change', 'new_extrinsic_total')
    _LADDER_TYPES = ('ROLL', 'OPEN', 'ADD', 'COVERED CALL')
    laddered = []
    for c in candidates:
        if c.get('type') not in _LADDER_TYPES:
            laddered.append(c)
            continue
        full_q = abs(c.get('roll_qty') or c.get('add_qty') or c.get('qty') or 1)
        if full_q <= 2:
            laddered.append(c)
            continue
        if full_q <= 5:
            rungs = sorted(set([1, full_q // 2, full_q]))
        else:
            rungs = sorted(set([1, full_q // 3, full_q // 2, full_q]))
        for rq in rungs:
            if rq <= 0 or rq > full_q:
                continue
            ratio = rq / full_q
            rung = dict(c)
            for k in _SCALED_FIELDS:
                if k in rung:
                    rung[k] = rung[k] * ratio
            # qty key varies by type
            if 'roll_qty' in rung:
                rung['roll_qty'] = rq
            if 'add_qty' in rung:
                rung['add_qty'] = rq
            # Refresh action string with the rung qty (replace leading Nx)
            old_action = rung.get('action', '')
            import re as _re
            new_action = _re.sub(r'^([A-Za-z ]+?)\s+\d+x\b',
                                 lambda m: f"{m.group(1)} {rq}x",
                                 old_action, count=1)
            if rq < full_q and new_action == old_action:
                # Fallback: prepend rung qty
                new_action = f"{old_action} ({rq}/{full_q})"
            elif rq < full_q:
                new_action = f"{new_action} ({rq}/{full_q})"
            rung['action'] = new_action
            # Scale liquidity-side dollar fields if present (cosmetic)
            liq = rung.get('liquidity')
            if isinstance(liq, dict):
                liq = dict(liq)
                for lk in ('notional', 'credit_debit', 'friction_est'):
                    if lk in liq:
                        liq[lk] = round(liq[lk] * ratio)
                if 'oi_usage_pct' in liq and full_q > 0:
                    liq['oi_usage_pct'] = round(liq['oi_usage_pct'] * ratio, 1)
                rung['liquidity'] = liq
            laddered.append(rung)
    candidates = laddered

    # Cycle 176: BUY LEAPS PUT candidates — replaces the hardcoded tail
    # hedge static check. Scans the FULL option chain (not just valid_expiries)
    # for puts with DTE ≥ 180. Generates concrete BUY PUT candidates with
    # real pricing so the beam evaluates them via qΔ like any other trade.
    # The tail_hedge component in evaluate_portfolio_quality gives -$1000
    # per missing LEAPS (floor=2), so these candidates get massive qΔ.
    # Cycle 182: LEAPS candidates always generated (no floor gate).
    # The evaluator's tail_hedge component (unhedged exposure penalty)
    # determines whether buying more LEAPS has positive qΔ. If existing
    # LEAPS already offset enough tail risk, the penalty is near zero
    # and BUY LEAPS candidates score negatively (cost > benefit).
    _LEAPS_MIN_DTE = 180
    _tail_qty = int(portfolio_state.get('tail_hedge_qty', 0) or 0)
    if True:  # always generate — let beam decide
        _available = get_available_options()
        for _lp_exp_str in sorted(_available.keys()):
            try:
                _lp_exp = datetime.strptime(_lp_exp_str, '%Y-%m-%d').date()
                _lp_dte = (_lp_exp - today).days
            except Exception:
                continue
            if _lp_dte < _LEAPS_MIN_DTE:
                continue
            _lp_chain = _available[_lp_exp_str]
            _lp_liq = _lp_chain.get('liquidity', {})
            _lp_puts = sorted({k for (k, rt) in _lp_liq if rt == 'P'})
            # ATM and 1 OTM strike
            _lp_atm = find_nearest_strike(spot, _lp_puts) if _lp_puts else None
            _lp_otm = find_nearest_strike(spot * 0.90, _lp_puts) if _lp_puts else None
            for _lp_K in set(filter(None, [_lp_atm, _lp_otm])):
                _lp_l = _lp_liq.get((_lp_K, 'P'), {})
                _lp_oi = _lp_l.get('oi', 0)
                _lp_bid = _lp_l.get('bid', 0)
                _lp_ask = _lp_l.get('ask', 0)
                if _lp_oi < 10 or _lp_bid <= 0:
                    continue
                _lp_mid = (_lp_bid + _lp_ask) / 2
                _lp_T = _lp_dte / 365.0
                _lp_iv = get_contract_iv(_lp_exp_str, _lp_K, 'P') or iv
                for _lp_qty in [1, 2]:
                    if _lp_qty <= 0:
                        continue
                    _lp_cost = _lp_mid * _lp_qty * 100
                    _lp_delta = _lp_qty * bs_delta(spot, _lp_K, _lp_T, r, _lp_iv, 'P') * 100
                    _lp_gamma = _lp_qty * bs_gamma(spot, _lp_K, _lp_T, r, _lp_iv) * 100
                    _lp_theta = -abs(bs_theta(spot, _lp_K, _lp_T, r, _lp_iv, 'P')) * _lp_qty * 100
                    _lp_vega = _lp_qty * bs_vega(spot, _lp_K, _lp_T, r, _lp_iv) * 100
                    _otm_pct = max(0, (spot - _lp_K) / spot) * 100
                    _otm_tag = f" {_otm_pct:.0f}%OTM" if _otm_pct >= 1.5 else ""
                    candidates.append({
                        'type': 'BUY PUT',
                        'action': f"Buy {_lp_qty}x {_lp_exp_str} ${_lp_K}P LEAPS ({_lp_dte} DTE{_otm_tag})",
                        'target_exp': _lp_exp_str,
                        'target_strike': _lp_K,
                        'add_qty': _lp_qty,
                        'theta_change': _lp_theta,
                        'delta_change': _lp_delta,
                        'gamma_change': _lp_gamma,
                        'vega_change': _lp_vega,
                        'new_extrinsic_total': _lp_mid * _lp_qty * 100,
                        'n_legs': 1,
                        'liquidity': {
                            'oi': _lp_oi,
                            'bid': round(_lp_bid, 2),
                            'ask': round(_lp_ask, 2),
                            'spread_pct': round((_lp_ask - _lp_bid) / ((_lp_ask + _lp_bid) / 2) * 100, 1) if (_lp_bid + _lp_ask) > 0 else 0,
                            'oi_usage_pct': round(_lp_qty / _lp_oi * 100, 1) if _lp_oi > 0 else 100,
                            'notional': round(_lp_cost),
                            'credit_debit': -round(_lp_cost),
                            'friction_est': round((_lp_ask - _lp_bid) / 2 * _lp_qty * 100) if (_lp_bid + _lp_ask) > 0 else 0,
                        },
                        'detail': f"LEAPS hedge | cost ${_lp_cost:.0f} | Δ{_lp_delta:+.0f} | Γ{_lp_gamma:+.0f} | OI={_lp_oi} | ${_lp_bid:.2f}/${_lp_ask:.2f}",
                        'why': f"LEAPS hedge: {_tail_qty} existing. Reduces unhedged CVaR tail exposure. Beam evaluates whether premium cost is justified by risk reduction.",
                    })

    # Cycle 200: BUY BOXX candidates for idle cash. User: "BOXX estimator
    # should be part of the engine, max BOXX since margin rate is high."
    # Compute idle cash = capital - share_value - put_margin_used. Reserve
    # 30% for opportunistic puts; deploy 70% in BOXX (4% risk-free, no
    # margin needed). Generate ladder of BOXX buy sizes.
    try:
        _boxx_px = 116.91  # cached, refresh via yfinance in get_boxx_price()
        # Use NET LIQUIDATION (total wealth) as the basis. capital_base is the
        # MARGIN sub-account only and underestimates cash available for BOXX.
        # User: "I don't need margin to buy BOXX; want to max it; margin rate
        # is high." So estimate idle cash from total wealth.
        _net_liq = float(portfolio_state.get('net_liquidation', _margin_capital_usd * 1.5) or _margin_capital_usd * 1.5)
        _shares_b = int(portfolio_state.get('shares', SHARES) or SHARES)
        _share_value = _shares_b * spot
        # Approximate margin used by short puts (cash-secured WS style)
        _put_margin_used = 0.0
        for _exp_p, _K_p, _rt_p, _qty_p, _avg_p in positions:
            if _rt_p == 'P' and _qty_p < 0:
                _prem_p = (_avg_p / 100) if _avg_p > 1 else _avg_p
                _put_margin_used += max(0, _K_p * 100 - _prem_p * 100) * abs(_qty_p)
        _idle_cash = max(0, _net_liq - _share_value - _put_margin_used)
        _reserve = _idle_cash * 0.30  # keep liquid for new puts
        _deployable = _idle_cash - _reserve
        if _deployable >= _boxx_px * 5:  # at least 5 shares worth
            for _frac, _label in [(0.5, '50%'), (1.0, 'full')]:
                _deploy = _deployable * _frac
                _n_boxx = int(_deploy / _boxx_px)
                if _n_boxx < 5:
                    continue
                _cost = _n_boxx * _boxx_px
                _yield_yr = _cost * 0.04
                _yield_wk = _yield_yr / 52
                candidates.append({
                    'type': 'BUY BOXX',
                    'action': f"Buy {_n_boxx} BOXX shares @ ${_boxx_px:.2f} ({_label} of deployable)",
                    'add_qty': _n_boxx,
                    'theta_change': _yield_wk / 7,  # daily theta-equivalent
                    'delta_change': 0,  # BOXX is non-correlated to UNG
                    'gamma_change': 0,
                    'vega_change': 0,
                    'new_extrinsic_total': 0,
                    'n_legs': 1,
                    'boxx_cost': _cost,
                    'liquidity': {
                        'oi': 0,
                        'bid': _boxx_px,
                        'ask': _boxx_px,
                        'spread_pct': 0.01,  # tight ETF spread
                        'oi_usage_pct': 0,
                        'notional': round(_cost),
                        'credit_debit': -round(_cost),
                        'friction_est': 1,  # commission only
                    },
                    'detail': f"Deploy ${_cost:.0f} | annual ${_yield_yr:.0f} (${_yield_wk:.0f}/wk) | idle cash ${_idle_cash:.0f}",
                    'why': f"Park idle cash in BOXX (1-3mo Treasury proxy). Earns ~4% risk-free vs sitting in margin account. Reserve ${_reserve:.0f} for opportunistic puts.",
                })
    except Exception:
        pass

    return candidates


_model_zscore = 0.53  # last known value, updated in background
_model_zscore_computing = False
# NG fair-value scenarios (parsed from ng_daily_forecast.py output).
# Used by the dashboard to show bull/base/bear NG and UNG targets.
_model_predictions = {
    'ng_current': None,
    'ng_fv_base': None,
    'ng_fv_bull': None,
    'ng_fv_bear': None,
    # Supply/demand regime (cyclical axis B per CENTRAL_PHILOSOPHY.md)
    'supply_regime': 'BALANCED',
    'storage_z': 0.0,
    # Tech/Fund/YoY pillar scores (continuous, [-1, +1], + = bullish for NG).
    # Fund + YoY are parsed from ng_daily_forecast.py PREDICTION lines.
    # Tech is computed locally from cached UNG technicals (price_band, MA trend).
    'fund_score': 0.0,
    'fund_score_raw': 0.0,
    'yoy_score': 0.0,
    'yoy_score_raw': 0.0,
    'updated_at': None,
}

# Load persisted predictions on startup so the dashboard shows correct
# pillar/regime values immediately (cycle 61). Without this, every auto-reload
# (cycle 45 os.execv) leaves predictions at 0.0 until the next ~30-60s
# subprocess run completes.
try:
    import json as _json_init
    import os as _os_init
    _cache_path_init = _os_init.path.join(
        _os_init.path.dirname(_os_init.path.abspath(__file__)),
        'predictions_cache.json')
    if _os_init.path.exists(_cache_path_init):
        with open(_cache_path_init) as _fh_init:
            _cached_preds = _json_init.load(_fh_init)
        if isinstance(_cached_preds, dict):
            _model_predictions.update(_cached_preds)
            print(f"[predictions] loaded cache: updated_at={_cached_preds.get('updated_at')}")
except Exception as _le:
    print(f"[predictions] cache load failed (will rely on background refresh): {_le}")

# Per-day drift adjustment by supply/demand regime. Conservative magnitudes:
# meaningful but never dominate seasonal_drift or contango.
SUPPLY_REGIME_DRIFT = {
    'SURPLUS':  -0.0005,  # ~-1.5%/mo extra bleed (storage too full, contango widens)
    'BALANCED':  0.0,
    'SHORTAGE': +0.0008,  # ~+2.4%/mo extra tailwind (tight supply supports price)
}

# Per-day drift scale for pillar scores. Each pillar capped at ±0.00025/d
# (~±0.75%/mo). Total pillar contribution clamped to ±0.0006/d so it never
# overpowers the seasonal/regime baseline.
PILLAR_DRIFT_SCALE = 0.00025
PILLAR_DRIFT_TOTAL_CAP = 0.0006

def compute_tech_score():
    """Continuous technical pillar in [-1, +1] from cached UNG technicals.
    + = bullish for NG (UNG sits near 120d low / trend supportive).
    Uses only signals already in the technicals cache — no extra data fetch."""
    tech = get_technicals_cached()
    if not tech:
        return 0.0
    try:
        spot = float(tech.get('spot', 0))
        lo = float(tech.get('low_120d', 0))
        hi = float(tech.get('high_120d', 0))
        ma_20 = float(tech.get('ma_20', spot))
        ma_50 = float(tech.get('ma_50', spot))
        if hi > lo:
            price_band = max(0.0, min(1.0, (spot - lo) / (hi - lo)))
        else:
            return 0.0
        # Price band reversion: near low → bullish, near high → bearish.
        # (0.5 - pb) maps [0,1] → [+0.5, -0.5]; *1.5 → [+0.75, -0.75].
        pb_part = (0.5 - price_band) * 1.5
        # MA trend tilt: ma_20 > ma_50 → mild bullish.
        ma_part = 0.0
        if ma_50 > 0:
            ma_diff_pct = (ma_20 - ma_50) / ma_50
            ma_part = max(-0.3, min(0.3, ma_diff_pct * 10.0))  # ~3% gap → ±0.3
        return max(-1.0, min(1.0, pb_part + ma_part))
    except Exception:
        return 0.0
# _margin_capital_usd already defined at top of file (line ~212), do not redefine here
_capital_fetched = False


def fetch_margin_capital():
    """Fetch margin account NLV in USD from WS. Updates global."""
    global _margin_capital_usd
    try:
        from ws_trading import (get_session, load_config, load_cookies,
                                graphql_query, extract_identity_from_cookies,
                                QUERY_FETCH_FINANCIALS, QUERY_ALL_ACCOUNTS)
        session = get_session()
        cfg = load_config()
        cks = load_cookies()
        iid = cfg.get('identity_id') or extract_identity_from_cookies(cks)

        # Find margin account ID
        accts = graphql_query(session, 'FetchAllAccounts', QUERY_ALL_ACCOUNTS, {'identityId': iid})
        margin_id = None
        if accts:
            for e in accts.get('identity', {}).get('accounts', {}).get('edges', []):
                n = e.get('node', {})
                if 'MARGIN' in str(n.get('unifiedAccountType', '')):
                    margin_id = n.get('id')
                    break

        if margin_id:
            fin = graphql_query(session, 'FetchIdentityCurrentFinancials', QUERY_FETCH_FINANCIALS, {
                'identityId': iid, 'currency': 'USD', 'accountIds': [margin_id]
            })
            if fin:
                nlv = fin.get('identity', {}).get('financials', {}).get('current', {}).get('netLiquidationValueV2', {})
                val = float(nlv.get('amount', 0))
                if val > 1000:
                    _margin_capital_usd = val - 3600  # minus $5k CAD cushion
                    print(f"Margin NLV: ${val:,.0f} USD, capital (after cushion): ${_margin_capital_usd:,.0f}")
                    return
    except Exception as ex:
        print(f"Margin capital fetch failed: {ex}")

    # Fallback — only if not already set by fetch_ws_positions
    if _margin_capital_usd <= 10000:
        _margin_capital_usd = max(10000, SHARES * UNG_PRICE - 3600)
        print(f"Using fallback capital: ${_margin_capital_usd:,.0f}")


# Capital fetched lazily on first API call to avoid blocking startup


def get_model_zscore():
    """Return cached z-score immediately. Never blocks."""
    return _model_zscore


def refresh_model_zscore():
    """Run model in background thread to update z-score. Non-blocking."""
    global _model_zscore, _model_zscore_computing
    if _model_zscore_computing:
        return  # already running

    def _run():
        global _model_zscore, _model_zscore_computing
        _model_zscore_computing = True
        try:
            import subprocess
            import re
            result = subprocess.run(
                ['/home/wyatt/weather/venv/bin/python', '/home/wyatt/weather/ng_daily_forecast.py'],
                capture_output=True, text=True, timeout=180,
                cwd='/home/wyatt/weather',
                env={**__import__('os').environ, 'MPLBACKEND': 'Agg'}
            )
            global _model_predictions
            pred_patterns = {
                'ng_current': r'PREDICTION_NG_CURRENT:\s*\$([\d.]+)',
                'ng_fv_base': r'PREDICTION_NG_FV_BASE:\s*\$([\d.]+)',
                'ng_fv_bull': r'PREDICTION_NG_FV_BULL:\s*\$([\d.]+)',
                'ng_fv_bear': r'PREDICTION_NG_FV_BEAR:\s*\$([\d.]+)',
                'storage_z': r'PREDICTION_STORAGE_Z:\s*([+-]?\d+\.?\d*)',
                'fund_score': r'PREDICTION_FUND_SCORE:\s*([+-]?\d+\.?\d*)',
                'fund_score_raw': r'PREDICTION_FUND_SCORE_RAW:\s*([+-]?\d+\.?\d*)',
                'yoy_score':  r'PREDICTION_YOY_SCORE:\s*([+-]?\d+\.?\d*)',
                'yoy_score_raw': r'PREDICTION_YOY_SCORE_RAW:\s*([+-]?\d+\.?\d*)',
            }
            regime_pattern = r'PREDICTION_SUPPLY_REGIME:\s*(SURPLUS|BALANCED|SHORTAGE)'
            freshness_pattern = r'PREDICTION_FACTOR_FRESHNESS:\s*(\{.*\})'
            ic_pattern = r'PREDICTION_IC_WEIGHTS:\s*(\{.*\})'
            parsed = {}
            for line in result.stdout.split('\n'):
                if 'Composite z-score:' in line:
                    match = re.search(r'z-score:\s*([+-]?\d+\.?\d*)', line)
                    if match:
                        _model_zscore = float(match.group(1))
                        print(f"Model z-score updated: {_model_zscore}")
                for key, pat in pred_patterns.items():
                    m = re.search(pat, line)
                    if m:
                        parsed[key] = float(m.group(1))
                rm = re.search(regime_pattern, line)
                if rm:
                    parsed['supply_regime'] = rm.group(1)
                # Per-factor freshness payload (cycle 49): single JSON
                # line emitted by ng_daily_forecast.py for /api/health.
                fm = re.search(freshness_pattern, line)
                if fm:
                    try:
                        import json as _json
                        parsed['factor_freshness'] = _json.loads(fm.group(1))
                    except Exception:
                        pass
                # IC weights payload (cycle 50): per-factor predictive power
                # measured as Spearman corr with 3m fwd return.
                ic_m = re.search(ic_pattern, line)
                if ic_m:
                    try:
                        import json as _json
                        parsed['ic_weights'] = _json.loads(ic_m.group(1))
                    except Exception:
                        pass
            if parsed:
                import datetime as _dt
                _model_predictions.update(parsed)
                _model_predictions['updated_at'] = _dt.datetime.now().isoformat(timespec='seconds')
                print(f"Model predictions updated: {parsed}")
                # Persist to disk so the next auto-reload doesn't lose them
                # (cycle 61). os.execv re-exec wipes in-memory state; without
                # this, pillars read 0.0 for the ~30-60s warmup until the
                # next subprocess run.
                try:
                    import json as _json
                    import os as _os
                    _cache_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                                'predictions_cache.json')
                    with open(_cache_path, 'w') as _fh:
                        _json.dump(_model_predictions, _fh, indent=2, default=str)
                except Exception as _ce:
                    print(f"predictions cache write failed: {_ce}")
        except Exception as e:
            print(f"Model z-score update failed: {e}")
        finally:
            _model_zscore_computing = False

    threading.Thread(target=_run, daemon=True).start()


def _compute_target_delta_cached(spot, z, capital_base, current_shares=None):
    """Delta target = current share count. Dynamic (updates when assignments
    add/remove shares), not z-score-leveraged.

    Cycle 187: decoupled delta target from z-score. The old formula set
    target = capital × leverage(z), which at z=+0.53 demanded 8,600 delta
    (84% of portfolio in UNG). This permanently blocked covered calls
    because selling ANY call reduced delta away from an unreachable target.

    The z-score still drives:
      - Deployment mode (ACTIVE/TRANSITION/WAITING)
      - Income mode aggression
      - Pillar scores in quality evaluator
    But NOT the delta target. For a wheel strategy, delta target =
    "maintain current share position" — which is SHARES. This makes
    covered calls delta-NEUTRAL (selling against shares you own doesn't
    change the target gap) and lets the beam pick both puts AND calls.
    """
    # Target tracks CURRENT shares (from state when provided, else module global
    # as fallback). When the beam evaluates SELL SHARES, the new state has
    # fewer shares → target updates → delta_gap stays fair. No hardcoded count.
    target = float(current_shares) if current_shares is not None else float(SHARES)
    # Regime label still from z for display/diagnostics
    if z > 1.0:
        regime = 'EXTREME CHEAP'
    elif z > 0.5:
        regime = 'VERY CHEAP'
    elif z > 0.25:
        regime = 'CHEAP'
    elif z > -0.25:
        regime = 'NEUTRAL'
    elif z > -0.5:
        regime = 'RICH'
    elif z > -1.0:
        regime = 'VERY RICH'
    else:
        regime = 'EXTREME RICH'
    return target, regime, z


def compute_target_delta(spot, current_shares=None):
    """Dynamic delta target using the NG multi-factor model z-score.

    Backtested: z-score regime targeting with aggressive dynamic delta wins.
    z > 1.0 (extreme cheap) → max leverage
    z ~ 0 (neutral) → moderate
    z < -0.5 (rich) → minimal leverage

    All targets in DOLLAR exposure, converted to shares at current price.
    Body memoized in _compute_target_delta_cached; cache key includes the
    two relevant globals (z, capital) so cross-request invalidation is
    automatic.
    """
    try:
        z = get_model_zscore()
    except Exception:
        z = 0.0
    return _compute_target_delta_cached(spot, z, _margin_capital_usd, current_shares)


    # compute_delta_at_price REMOVED — dead code (codex audit, never called)


def short_put_collateral(positions):
    """Sum of correlated max-loss collateral for short puts: |qty| × strike × 100."""
    total = 0.0
    for entry in positions:
        # positions may be tuples (exp, strike, right, qty, avg) or dicts
        if isinstance(entry, dict):
            right = entry.get('right', '')
            qty = entry.get('qty', 0)
            strike = entry.get('strike', 0)
        else:
            _, strike, right, qty, _ = entry
        if right == 'P' and qty < 0:
            total += abs(qty) * strike * 100
    return total


def compute_kelly(trade, portfolio_state):
    """Compute Kelly fraction and expected value for a trade.

    Returns dict with: kelly, half_kelly, ev_total, ev_per_contract, p_otm, win_per, loss_per
    Returns None if not applicable (LET EXPIRE, ASSIGNMENT, share trades).

    Kelly: f* = (p_win * b - p_loss) / b   where b = win / loss

    Wheel adjustment: in ACCUMULATE regime, put assignment is treated as neutral
    (you wanted shares anyway). In HARVEST/EXIT regime, call assignment is neutral.
    """
    import math as _m

    trade_type = trade.get('type', '')
    # Cycle 180: added BUY PUT to exclusions. The Kelly formula below assumes
    # you SELL premium (win = credit kept when OTM). For BUY PUT the economics
    # are inverted (win when ITM, lose premium when OTM). Scoring BUY PUT
    # with the short-premium formula gives materially wrong EV/Kelly. The
    # tail_hedge quality component handles LEAPS ranking via CVaR-based
    # crash benefit instead.
    if trade_type in ('LET EXPIRE', 'ASSIGNMENT', 'SELL SHARES', 'BUY SHARES', 'CLOSE', 'TAKE PROFIT', 'BUY PUT'):
        return None

    spot = portfolio_state.get('spot', UNG_PRICE)
    r_rate = 0.04

    target_strike = trade.get('target_strike', trade.get('source_strike', 0))
    target_dte = trade.get('target_dte', trade.get('source_dte', 30))
    target_right = trade.get('source_right', 'P')
    if trade_type == 'COVERED CALL':
        target_right = 'C'
    # Real per-contract IV from yfinance chain (cycle 41); fallback 0.50.
    target_exp = trade.get('target_exp') or trade.get('source_exp', '')
    iv_est = get_contract_iv(target_exp, target_strike, target_right, fallback=0.50)

    if target_strike <= 0 or target_dte <= 0:
        return None

    qty = abs(trade.get('roll_qty', trade.get('add_qty', trade.get('qty', 1))))
    if qty <= 0:
        return None

    T = target_dte / 365.0
    sigma = iv_est * _m.sqrt(T)
    if sigma <= 0:
        return None

    # P(ITM) and expected loss: prefer the shared scenario_dist (multi-horizon
    # quantile kernel that already encodes seasonal_drift, supply_regime, and
    # Tech/Fund/YoY pillars). Fall back to BS analytics with z-only drift if
    # the distribution isn't available (legacy path).
    sd = portfolio_state.get('scenario_dist')
    p_itm = None
    loss = None
    if sd is not None:
        # Cycle 76: reuse the _sd_cache from cycle 75. score_trade and
        # compute_kelly call sd.prob_*/expected with identical args within
        # the same cheap_score invocation; cache key format kept
        # consistent so both paths hit.
        _sd_cache = portfolio_state.get('_sd_cache')
        if _sd_cache is None:
            _sd_cache = {}
            portfolio_state['_sd_cache'] = _sd_cache
        try:
            _pk = ('prob_below' if target_right == 'P' else 'prob_above',
                   round(target_strike, 4), int(target_dte))
            p_itm = _sd_cache.get(_pk)
            if p_itm is None:
                if target_right == 'P':
                    p_itm = float(sd.prob_below(target_strike, target_dte))
                else:
                    p_itm = float(sd.prob_above(target_strike, target_dte))
                _sd_cache[_pk] = p_itm
            _ik = ('expected_intrinsic', round(target_strike, 4),
                   int(target_dte), target_right)
            expected_intrinsic = _sd_cache.get(_ik)
            if expected_intrinsic is None:
                if target_right == 'P':
                    def _intrinsic(sp, K=target_strike):
                        return max(0.0, K - sp)
                else:
                    def _intrinsic(sp, K=target_strike):
                        return max(0.0, sp - K)
                expected_intrinsic = float(sd.expected(target_dte, _intrinsic))
                _sd_cache[_ik] = expected_intrinsic
            if (p_itm or 0) > 0.001:
                loss = (expected_intrinsic / p_itm) * 100.0
        except Exception:
            p_itm = None
            loss = None
    if p_itm is None or loss is None:
        # Legacy BS path with z-only regime drift (preserves old behavior on
        # fallback so we never lose Kelly scoring).
        contango_daily = -0.029 / 30
        try:
            z = get_model_zscore()
        except Exception:
            z = 0
        regime_daily = z * 0.005 / 5
        drift = (contango_daily + regime_daily) * target_dte
        d2 = (_m.log(spot / target_strike) + drift - 0.5 * sigma**2) / sigma
        if target_right == 'P':
            p_itm = float(norm.cdf(-d2))
        else:
            p_itm = float(norm.cdf(d2))
        if loss is None:
            expected_move_past = sigma * spot * 0.4
            loss = expected_move_past * 100.0
    p_otm = 1 - p_itm

    # Premium per contract = full option price * 100
    full_price = abs(bs_price(spot, target_strike, T, r_rate, iv_est, target_right))
    win = full_price * 100  # we keep credit if OTM

    # Get gamma regime for wheel adjustment
    try:
        gr = compute_gamma_regime(spot)
        gamma_regime = gr.get('regime', 'HOLD')
    except Exception:
        gamma_regime = 'HOLD'

    # Wheel adjustment: assignment isn't always a "loss"
    if target_right == 'P' and gamma_regime == 'ACCUMULATE':
        # Want shares — assignment is a feature, not a bug
        # Discount loss to ~friction only (we'd buy shares anyway)
        loss = max(20, loss * 0.2)
    elif target_right == 'C' and gamma_regime in ('HARVEST', 'EXIT'):
        # Want to shed shares — assignment is good
        loss = max(20, loss * 0.2)

    if loss <= 0:
        return None

    # Kelly fraction
    b = win / loss
    kelly = (p_otm * b - p_itm) / b

    # Expected value in dollars (per contract)
    ev_per = p_otm * win - p_itm * loss
    ev_total = ev_per * qty

    return {
        'kelly': round(kelly, 3),
        'half_kelly': round(kelly * 0.5, 3),    # kept for UI back-compat
        'quarter_kelly': round(kelly * 0.25, 3),  # income-mode default
        'ev_total': round(ev_total),
        'ev_per_contract': round(ev_per),
        'p_otm': round(p_otm * 100),
        'win_per': round(win),
        'loss_per': round(loss),
    }


def score_trade(trade, portfolio_state, skip_waterfall=False):
    """Score trades using economic reasoning, not just daily rate metrics.

    Args:
        skip_waterfall: if True, skip expensive forward-theta projection
                        (use for first-pass filtering of many candidates)
    """
    W_DELTA = 3.0
    W_THETA = 1.0
    W_CONC = 0.5
    W_GAMMA = 0.3

    spot = portfolio_state.get('spot', UNG_PRICE)
    theta_change = trade.get('theta_change', 0)
    delta_change = trade.get('delta_change', 0)
    gamma_change = trade.get('gamma_change', 0)
    qty = abs(trade.get('roll_qty', trade.get('add_qty', trade.get('qty', 1))))

    # ── 1. Delta targeting ─────────────────────────────────────────────────
    # Move toward target, but distinguish STRUCTURAL gap from GAMMA-TRANSIENT gap.
    # If the gap is mostly from gamma (put deltas expanding on a drop), it will
    # self-correct when spot recovers. Don't force trades to close a transient gap.
    #
    # Method: check what the gap would be at a nearby price (±3%).
    # If gap is large here but near zero at spot+3%, it's gamma-transient.
    target_delta, _, _ = compute_target_delta(spot)
    current_delta = portfolio_state['total_delta']
    raw_gap = current_delta - target_delta  # signed: positive = over target

    # Compute gap at spot + 3% (mild recovery) to assess transience
    recovery_price = spot * 1.03
    recovery_target, _, _ = compute_target_delta(recovery_price)
    cur_gamma_val = portfolio_state['total_gamma']
    delta_at_recovery = current_delta + cur_gamma_val * (recovery_price - spot)
    gap_at_recovery = delta_at_recovery - recovery_target

    # Structural gap = the gap that persists even after a small recovery
    # Transient gap = the portion that disappears on recovery (from gamma)
    # Only the structural part should drive urgency
    if abs(raw_gap) > 0:
        # How much of the gap is transient? (disappears on 3% recovery)
        transient_portion = max(0, 1 - abs(gap_at_recovery) / abs(raw_gap))
        # Effective gap: discount the transient part
        # At minimum, use 30% of the raw gap (never fully ignore it)
        effective_gap = abs(raw_gap) * max(0.3, 1 - transient_portion * 0.7)
    else:
        effective_gap = 0

    new_delta = current_delta + delta_change
    new_gap = abs(new_delta - target_delta)
    # Use effective_gap for the "before" but raw new_gap for "after"
    # (the trade permanently changes the portfolio)
    delta_improvement_raw = (effective_gap - new_gap) / 100

    # Diminishing returns: the first 300 shares of gap closure are worth full credit.
    # Beyond that, each additional share of closure is worth less.
    # This prevents massive share dumps from dominating on transient gamma swings.
    # sqrt scaling: closing 100 → 3.0 pts, closing 400 → 6.0 pts, closing 1600 → 12.0 pts
    import math as _m
    if delta_improvement_raw > 0:
        delta_score = _m.sqrt(max(0, delta_improvement_raw) * 3) * W_DELTA
    elif delta_improvement_raw < 0:
        # Widening gap: linear penalty (don't soften bad trades)
        delta_score = delta_improvement_raw * W_DELTA
    else:
        delta_score = 0

    # ── 2. Economic cost of the trade ──────────────────────────────────────
    # Every trade has a real dollar cost: friction + roll debit + surrendered runway
    # This replaces the old separate friction/ext/drain scores

    # 2a. Bid/ask friction
    n_legs = trade.get('n_legs', 1)
    opt_price = trade.get('new_extrinsic_total', 0) / max(1, qty * 100)
    per_share_friction = max(0.01, min(0.05, opt_price * 0.03))
    friction_cost = n_legs * per_share_friction * qty * 100

    # 2b. Roll-specific: net debit is a real cost, net credit is a real benefit
    roll_net = trade.get('roll_net_total', 0)  # positive=credit, negative=debit

    # 2c. Extrinsic runway: for rolls, compare what you give up vs what you get
    # Old runway = old extrinsic remaining (dollars left to harvest if you do nothing)
    # New runway = new extrinsic total (dollars available in new position)
    # Surrendered value = old_runway - |roll_credit| (what you leave on the table)
    old_ext_remaining = trade.get('old_ext_remaining', 0)
    new_ext_total = trade.get('new_extrinsic_total', 0)

    if trade['type'] == 'ROLL':
        # Total economic cost: you pay friction + debit (or receive credit)
        # and you surrender old extrinsic runway in exchange for new runway
        # Net value = new_runway + roll_credit - old_runway - friction
        runway_change = new_ext_total - old_ext_remaining  # positive = more to harvest
        economic_value = runway_change + roll_net - friction_cost
    else:
        # For OPEN/ADD: you collect new extrinsic, pay friction
        economic_value = new_ext_total - friction_cost

    # Cycle 138 debit-roll aversion (user principle): "2 debit rolling is
    # not my style unless you tell me it is more risk management".
    # Apply an extra penalty to ROLL trades that pay a debit UNLESS the
    # source position is ITM (legitimate risk-management defense) or near
    # expiry (urgent assignment avoidance).
    debit_aversion = 0.0
    if trade['type'] == 'ROLL' and roll_net < 0:
        src_strike = trade.get('source_strike', 0)
        src_right = trade.get('source_right', 'P')
        src_dte = trade.get('source_dte', 30)
        src_is_itm = ((src_right == 'C' and spot > src_strike)
                      or (src_right == 'P' and spot < src_strike))
        src_is_urgent = src_dte <= 5
        if not (src_is_itm or src_is_urgent):
            # Pure paid-to-move roll — score down proportional to debit
            # $100 debit = -2 pts; $300 = -6 pts. Strong enough to filter
            # most of these out at MIN_MARGINAL_SCORE=3 unless other
            # signals are very positive.
            debit_aversion = roll_net / 50.0  # roll_net is negative → negative pts

    # Scale: $100 of economic value = 1 point
    economic_score = economic_value / 100.0 * W_THETA

    # ── 3. Theta efficiency (daily rate relative to risk) ──────────────────
    # Still reward theta gain, but scaled down — economic_score handles value
    # This captures rate improvement when runway is similar
    if theta_change > 0:
        theta_rate_bonus = min(3, theta_change * 0.5) * W_THETA
    elif theta_change < 0:
        theta_rate_bonus = theta_change * 1.0 * W_THETA  # penalize losing daily rate
    else:
        theta_rate_bonus = 0

    # ── 4. Extrinsic drain urgency ─────────────────────────────────────────
    # Bonus for acting on positions where theta harvest is nearly done
    ext_pct = trade.get('ext_pct_old', 100)
    drain_bonus = max(0, (40 - ext_pct) * 0.4) if ext_pct < 40 else 0

    # ── 5. Concentration reduction ─────────────────────────────────────────
    source_exp = trade.get('source_exp', '')
    expiry_theta = portfolio_state.get('expiry_theta', {})
    total_theta = portfolio_state['total_theta']
    conc_score = 0
    if source_exp and source_exp in expiry_theta and total_theta > 0:
        source_conc = expiry_theta[source_exp] / total_theta
        if source_conc > 0.25:
            conc_score = min(15, (source_conc - 0.20) * 150) * W_CONC

    # ── 6. Recovery time: how many days of theta to recover from a realistic move?
    # This replaces separate gamma and curve penalties with one principled metric.
    # The swing IS the strategy — it's only bad if theta can't compensate.
    gamma_penalty = 0  # replaced by recovery_score below

    # ── 7. Type bonus ──────────────────────────────────────────────────────
    # LET EXPIRE is free (no action needed). Other types earn small bonus.
    # BUY/SELL SHARES: no bonus — pure delta tool, let delta_score drive it.
    type_bonus = {'LET EXPIRE': 15, 'ASSIGNMENT': 10,
                  'COVERED CALL': 3, 'ROLL': 3, 'OPEN': 3,
                  'BUY PUT': 2, 'TAKE PROFIT': 2, 'ADD': 2,
                  'SELL SHARES': 0, 'BUY SHARES': 0,
                  'CLOSE': 1}.get(trade['type'], 0)

    # ── 7b. Strike simulation: if UNG moves to the target strike, is the
    # resulting portfolio state desirable? ──────────────────────────────────
    # For each candidate, simulate the entire portfolio at strike price.
    # Compare resulting delta vs target delta AT THAT PRICE.
    # If assignment brings us closer to target → good. Further → bad.
    import math as _math
    assignment_score = 0  # positive = good, negative = bad
    p_itm = 0.5  # default; set inside the strike-sim block when applicable
    target_strike_val = trade.get('target_strike', trade.get('source_strike', 0))
    target_dte_val = trade.get('target_dte', trade.get('source_dte', 30))
    target_right = trade.get('source_right', 'P')
    if trade['type'] in ('COVERED CALL',):
        target_right = 'C'

    if trade['type'] in ('ROLL', 'OPEN', 'ADD', 'COVERED CALL') and target_strike_val > 0:
        r_rate_sim = 0.04
        # Real per-contract IV (cycle 41); fallback 0.50.
        _target_exp_sim = trade.get('target_exp') or trade.get('source_exp', '')
        iv_sim = get_contract_iv(_target_exp_sim, target_strike_val, target_right, fallback=0.50)
        positions_for_sim = portfolio_state.get('positions', [])

        # P(ITM) for the new option at this strike. Prefer scenario_dist
        # (multi-horizon kernel with seasonal_drift + supply_regime + pillars);
        # fall back to BS with z-only drift if not available.
        T_target = max(target_dte_val, 1) / 365.0
        sigma_t = iv_sim * _math.sqrt(T_target)
        sd_sim = portfolio_state.get('scenario_dist')
        # Cycle 75: cache scenario_dist method calls per path. Same (K, days)
        # gets re-queried for many candidates inside cheap_score; ~10 unique
        # (K, days) pairs per path × 4 distinct methods.
        _sd_cache = portfolio_state.get('_sd_cache')
        if _sd_cache is None:
            _sd_cache = {}
            portfolio_state['_sd_cache'] = _sd_cache
        p_itm = 0.5
        if sd_sim is not None:
            try:
                _pk = ('prob_above' if target_right == 'C' else 'prob_below',
                       round(target_strike_val, 4), int(target_dte_val))
                p_itm = _sd_cache.get(_pk)
                if p_itm is None:
                    if target_right == 'C':
                        p_itm = float(sd_sim.prob_above(target_strike_val, target_dte_val))
                    else:
                        p_itm = float(sd_sim.prob_below(target_strike_val, target_dte_val))
                    _sd_cache[_pk] = p_itm
            except Exception:
                sd_sim = None  # fall through to BS path below
        if sd_sim is None and sigma_t > 0.001:
            contango_drift = -0.029 / 30 * target_dte_val
            d2 = (_math.log(spot / target_strike_val) + contango_drift - 0.5 * sigma_t**2) / sigma_t
            p_itm = norm.cdf(d2) if target_right == 'C' else norm.cdf(-d2)

        # Simulate portfolio at the strike price (what if UNG moves there?)
        sim_price = target_strike_val
        sim_target_delta, _, _ = compute_target_delta(sim_price)

        # Current portfolio delta at sim_price (cycle 74 cache): for fixed
        # positions (within one beam expansion path), this sum is purely a
        # function of (sim_price, iv_sim). Cache it on portfolio_state so
        # candidates with the same target_strike+IV reuse the result.
        # Before cache: 22 bs_delta calls × 8651 cheap_scores = 190k scipy
        # calls (~24s). After: ~10 unique keys × 22 = 220 calls (~30ms).
        _cache_key = (round(sim_price, 4), round(iv_sim, 4))
        _ssc = portfolio_state.get('_strike_sim_cache')
        if _ssc is None:
            _ssc = {}
            portfolio_state['_strike_sim_cache'] = _ssc
        current_delta_at_strike = _ssc.get(_cache_key)
        if current_delta_at_strike is None:
            current_delta_at_strike = float(SHARES)
            for exp_s, strike_s, right_s, qty_s, _ in positions_for_sim:
                exp_d = datetime.strptime(exp_s, '%Y-%m-%d').date()
                dte_s = max((exp_d - date.today()).days, 0)
                T_s = dte_s / 365.0
                if T_s < 0.001:
                    continue
                current_delta_at_strike += qty_s * bs_delta(sim_price, strike_s, T_s, r_rate_sim, iv_sim, right_s) * 100
            _ssc[_cache_key] = current_delta_at_strike

        # Portfolio delta at sim_price WITH this trade applied
        # The new option's delta at its own strike ≈ 0.5 (ATM)
        trade_qty = abs(trade.get('roll_qty', trade.get('add_qty', trade.get('qty', 1))))
        # Cycle 79: cache bs_delta(S, K, T, iv, right). For ~150 candidates
        # per expansion path, only ~5-20 unique (K, T, iv, right) combos
        # per sim_price. Use a per-path dict on portfolio_state.
        _bsd_cache = portfolio_state.get('_bsd_cache')
        if _bsd_cache is None:
            _bsd_cache = {}
            portfolio_state['_bsd_cache'] = _bsd_cache
        _bk_new = (round(sim_price, 4), round(target_strike_val, 4),
                   round(T_target, 6), round(iv_sim, 4), target_right)
        _bd_new = _bsd_cache.get(_bk_new)
        if _bd_new is None:
            _bd_new = bs_delta(sim_price, target_strike_val, T_target, r_rate_sim, iv_sim, target_right)
            _bsd_cache[_bk_new] = _bd_new
        new_opt_delta = -trade_qty * _bd_new * 100

        # For rolls: also remove old option's delta
        old_opt_delta = 0
        if trade['type'] == 'ROLL':
            source_strike = trade.get('source_strike', 0)
            source_dte = trade.get('source_dte', 0)
            source_right_s = trade.get('source_right', 'P')
            T_source = max(source_dte, 1) / 365.0
            sign = -1  # short option
            _bk_old = (round(sim_price, 4), round(source_strike, 4),
                       round(T_source, 6), round(iv_sim, 4), source_right_s)
            _bd_old = _bsd_cache.get(_bk_old)
            if _bd_old is None:
                _bd_old = bs_delta(sim_price, source_strike, T_source, r_rate_sim, iv_sim, source_right_s)
                _bsd_cache[_bk_old] = _bd_old
            old_opt_delta = sign * _bd_old * trade_qty * 100

        new_delta_at_strike = current_delta_at_strike + new_opt_delta - old_opt_delta
        current_gap_at_strike = abs(current_delta_at_strike - sim_target_delta)
        new_gap_at_strike = abs(new_delta_at_strike - sim_target_delta)

        # Score: weighted by P(ITM) — high probability means this scenario matters more
        # Positive = trade brings us closer to target AT THE STRIKE PRICE
        gap_improvement = (current_gap_at_strike - new_gap_at_strike) / 100
        delta_sim_score = gap_improvement * p_itm * W_DELTA

        # Expected share impact from assignment:
        # Short call assigned = lose shares at strike (bad if spot rallies past)
        # Short put assigned = gain shares at strike (bad if spot drops past)
        # Model the expected cost: P(ITM) × qty × |spot - strike| at expected move
        # For calls: if assigned, you sell shares at strike. Expected loss vs holding:
        #   E[loss] = P(ITM) * E[S-K | S>K] * qty * 100
        # Approximate: expected move beyond strike given ITM ≈ sigma * sqrt(T) * spot * 0.4
        try:
            z_model = get_model_zscore()
        except Exception:
            z_model = 0

        # Expected spot at expiry and expected absolute move past strike.
        # Prefer scenario_dist (cyclical model); fall back to BS with z-drift.
        expected_spot = None
        expected_move_past = None
        if sd_sim is not None:
            try:
                # Cycle 75: cache expected_spot (depends only on days) and
                # expected_intrinsic (depends on K, days, right).
                _ek = ('expected_spot', int(target_dte_val))
                expected_spot = _sd_cache.get(_ek)
                if expected_spot is None:
                    expected_spot = float(sd_sim.expected(target_dte_val, lambda s: s))
                    _sd_cache[_ek] = expected_spot
                _ik = ('expected_intrinsic', round(target_strike_val, 4),
                       int(target_dte_val), target_right)
                e_intr = _sd_cache.get(_ik)
                if e_intr is None:
                    if target_right == 'C':
                        def _intr_c(s, K=target_strike_val):
                            return max(0.0, s - K)
                        e_intr = float(sd_sim.expected(target_dte_val, _intr_c))
                    else:
                        def _intr_p(s, K=target_strike_val):
                            return max(0.0, K - s)
                        e_intr = float(sd_sim.expected(target_dte_val, _intr_p))
                    _sd_cache[_ik] = e_intr
                expected_move_past = (float(e_intr) / float(p_itm)) if (p_itm or 0) > 0.001 else 0.0
            except Exception:
                expected_spot = None
                expected_move_past = None
        if expected_spot is None:
            contango_daily = -0.029 / 30
            regime_daily = z_model * 0.005 / 5  # z=1 → +0.1%/day expected
            total_daily = contango_daily + regime_daily
            expected_spot = spot * _math.exp(total_daily * target_dte_val)
        if expected_move_past is None:
            expected_move_past = iv_sim * _math.sqrt(T_target) * spot * 0.4

        if target_right == 'C':
            # Short call assignment = lose shares.
            # Opportunity cost: if spot rallies to expected_spot, you miss the upside
            # above the strike. Worse in bullish regime.
            expected_upside = max(0, expected_spot - target_strike_val)
            # Total expected cost: P(ITM) × (expected move past + regime upside)
            expected_assignment_cost = float(p_itm) * (float(expected_move_past) + float(expected_upside)) * trade_qty * 100
            # Scale: $100 expected cost = 1 point penalty
            assignment_score = delta_sim_score - expected_assignment_cost / 100
        elif target_right == 'P':
            # Short put assignment = gain shares.
            # In bullish regime: buying shares cheap is GOOD (expected recovery)
            expected_assignment_benefit = float(p_itm) * max(0.0, float(expected_spot or 0) - target_strike_val + float(expected_move_past or 0)) * trade_qty * 100
            # Reward: shares acquired below expected future price
            assignment_score = delta_sim_score + expected_assignment_benefit / 100
        else:
            assignment_score = delta_sim_score

    # ── 8. Recovery time: unified gamma/swing/crash metric ────────────────
    # The swing IS the strategy. Short puts go ITM on drops, short calls on rallies.
    # It's only bad if theta can't compensate for the P&L swing.
    #
    # Metric: for a realistic 5-day move (25th/75th percentile),
    # how many days of theta does it take to recover?
    # < 30 days: self-healing, no problem
    # 30-60 days: concerning, mild penalty
    # > 60 days: dangerous, strong penalty
    #
    # Compare before vs after this trade — reward trades that improve recovery time.

    # Drop/rally percentages — prefer scenario_dist 5d quantiles (cyclical
    # model). Fall back to fixed empirical defaults if unavailable.
    crash_pct = STRESS_SCENARIOS['5d_crash']     # legacy: -11.7%
    mild_drop_pct = -0.048                        # legacy: -4.8% (25th percentile)
    # mild_rally_pct = 0.039 (75th pctile) — recovery_score is asymmetric
    # and only uses the drop side, so the rally quantile is not consumed.
    _sd_rec = portfolio_state.get('scenario_dist')
    if _sd_rec is not None:
        try:
            # Cycle 75: same _sd_cache reused for 5d quantile lookups.
            # Args (5, 0.05/0.25/0.75) are identical across every candidate
            # — three unique entries per path.
            _qc = portfolio_state.get('_sd_cache')
            if _qc is None:
                _qc = {}
                portfolio_state['_sd_cache'] = _qc
            _q05_k = ('quantile', 5, 0.05)
            _q25_k = ('quantile', 5, 0.25)
            _q05 = _qc.get(_q05_k)
            if _q05 is None:
                _q05 = _sd_rec.quantile(5, 0.05)
                _qc[_q05_k] = _q05
            _q25 = _qc.get(_q25_k)
            if _q25 is None:
                _q25 = _sd_rec.quantile(5, 0.25)
                _qc[_q25_k] = _q25
            if spot > 0:
                crash_pct = (_q05 - spot) / spot
                mild_drop_pct = (_q25 - spot) / spot
        except Exception:
            pass  # keep legacy defaults

    new_theta_total = max(1.0, portfolio_state['total_theta'] + theta_change)
    cur_theta = max(1.0, portfolio_state['total_theta'])

    # _recovery_days REMOVED — dead code (codex audit)

    # Before this trade: recovery from mild drop (25th pct)
    cur_drop_loss = abs(SHARES * spot * mild_drop_pct)
    cur_recovery_drop = cur_drop_loss / cur_theta

    # After this trade
    new_drop_loss = abs(SHARES * spot * mild_drop_pct)  # shares don't change from options
    new_recovery_drop = new_drop_loss / new_theta_total

    # For crash scenario (more extreme)
    cur_crash_loss = abs(SHARES * spot * crash_pct)
    cur_recovery_crash = cur_crash_loss / cur_theta
    new_crash_loss = abs(SHARES * spot * crash_pct)
    new_recovery_crash = new_crash_loss / new_theta_total

    # Improvement: positive = faster recovery (good)
    drop_improvement = cur_recovery_drop - new_recovery_drop  # days saved
    crash_improvement = cur_recovery_crash - new_recovery_crash

    # Score: reward trades that speed up recovery, penalize trades that slow it
    # Weight mild drop more than crash (it happens more often)
    recovery_score = (drop_improvement * 0.1 + crash_improvement * 0.03) * W_DELTA

    # Absolute penalty: if recovery time after trade > 60 days for crash, flag it
    crash_penalty = max(0, (new_recovery_crash - 60) * 0.05) * W_GAMMA
    curve_score = recovery_score

    # ── 9. Theta continuity: project theta forward, reward smoothing cliffs ─
    # Expensive — skip on first-pass scoring (optimization #4)
    from datetime import timedelta
    today_date = date.today()
    waterfall_score = 0
    target_exp = trade.get('target_exp', '')

    if not skip_waterfall:
        r_rate = 0.04
        positions_list = portfolio_state.get('positions', [])

        # Cycle 78: cache per-(strike, exp, right, days_ahead) forward theta
        # in the module-level Greeks cache. For fixed spot/today/iv-per-leg
        # within a request, the per-share forward theta is invariant. ~66
        # unique keys (22 positions × 3 checkpoints) vs 58k raw bs_theta
        # calls (cycle 76+77 baseline measured ~6s).
        global _FORWARD_THETA_CACHE
        if '_FORWARD_THETA_CACHE' not in globals():
            _FORWARD_THETA_CACHE = {'__key': None, 'fwd': {}}
        _ftk_key = (round(spot, 6), today_date.toordinal())
        if _FORWARD_THETA_CACHE['__key'] != _ftk_key:
            _FORWARD_THETA_CACHE = {'__key': _ftk_key, 'fwd': {}}
        _ftc = _FORWARD_THETA_CACHE['fwd']

        def _project_theta(positions_to_use, days_ahead):
            future_date = today_date + timedelta(days=days_ahead)
            total = 0
            for exp_s, strike_s, right_s, qty_s, _ in positions_to_use:
                _k = (strike_s, exp_s, right_s, days_ahead)
                _per_share = _ftc.get(_k)
                if _per_share is None:
                    exp_d = datetime.strptime(exp_s, '%Y-%m-%d').date()
                    remaining = (exp_d - future_date).days
                    if remaining <= 0:
                        _ftc[_k] = 0.0
                        continue
                    T_f = remaining / 365.0
                    _iv = get_contract_iv(exp_s, strike_s, right_s, fallback=0.50)
                    _per_share = bs_theta(spot, strike_s, T_f, r_rate, _iv, right_s) * 100
                    _ftc[_k] = _per_share
                elif _per_share == 0.0:
                    continue
                total += abs(qty_s * _per_share)
            return total

        # Reduced from 5 to 3 checkpoints (optimization #4)
        checkpoints = [0, 14, 28]
        current_curve = [_project_theta(positions_list, d) for d in checkpoints]

        # If this trade adds a new position, simulate with it included
        if trade['type'] in ('OPEN', 'ADD', 'COVERED CALL', 'ROLL') and target_exp:
            target_strike = trade.get('target_strike', trade.get('source_strike', 0))
            target_right = trade.get('source_right', 'P')
            if trade['type'] in ('COVERED CALL',):
                target_right = 'C'
            trade_qty = trade.get('add_qty', trade.get('roll_qty', 1))

            modified_positions = list(positions_list)
            if trade['type'] == 'ROLL':
                source_exp = trade.get('source_exp', '')
                source_strike = trade.get('source_strike', 0)
                source_right = trade.get('source_right', 'P')
                roll_qty = trade.get('roll_qty', 1)
                for i, (e, s, r_p, q, a) in enumerate(modified_positions):
                    if e == source_exp and s == source_strike and r_p == source_right:
                        new_q = abs(q) - roll_qty
                        if new_q <= 0:
                            modified_positions.pop(i)
                        else:
                            modified_positions[i] = (e, s, r_p, -new_q, a)
                        break

            modified_positions.append((target_exp, target_strike, target_right, -trade_qty, 0))
            new_curve = [_project_theta(modified_positions, d) for d in checkpoints]

            def _curve_cv(curve):
                if not curve or max(curve) < 1:
                    return 1.0
                mean_c = sum(curve) / len(curve)
                if mean_c < 1:
                    return 1.0
                variance = sum((v - mean_c) ** 2 for v in curve) / len(curve)
                return (variance ** 0.5) / mean_c

            def _worst_drop(curve):
                worst = 0
                for i in range(1, len(curve)):
                    if curve[i-1] > 0:
                        drop_pct = (curve[i-1] - curve[i]) / curve[i-1]
                        worst = max(worst, drop_pct)
                return worst

            old_cv = _curve_cv(current_curve)
            new_cv = _curve_cv(new_curve)
            old_drop = _worst_drop(current_curve)
            new_drop = _worst_drop(new_curve)

            smoothing_improvement = (old_cv - new_cv) * 20
            cliff_improvement = (old_drop - new_drop) * 30

            waterfall_score = max(0, smoothing_improvement + cliff_improvement)

    # ── 10. Gamma regime adjustment ───────────────────────────────────────
    # Adjust score based on where UNG sits in its 120-day range.
    # In HARVEST/EXIT: penalize new short puts, reward covered calls / closing puts.
    # In ACCUMULATE: reward short puts, penalize aggressive covered calls.
    gamma_regime_score = 0
    try:
        gr = compute_gamma_regime(spot)
        gr_regime = gr['regime']
        trade_type = trade['type']

        if gr_regime in ('HARVEST', 'EXIT'):
            severity = 5.0 if gr_regime == 'EXIT' else 3.0
            # Penalize new short puts (adding gamma when we want less)
            if trade_type in ('OPEN', 'ADD') and trade.get('source_right', 'P') == 'P':
                gamma_regime_score = -severity * qty
            elif trade_type == 'STRANGLE':
                gamma_regime_score = -severity * 0.5  # half penalty (has call leg)
            # Reward covered calls and closing/taking profit on puts
            elif trade_type == 'COVERED CALL':
                gamma_regime_score = severity * 0.5
            elif trade_type in ('CLOSE', 'TAKE PROFIT') and trade.get('source_right') == 'P':
                gamma_regime_score = severity * 0.3
            # Reward rolling puts to lower/OTM strikes
            elif trade_type == 'ROLL' and trade.get('source_right') == 'P':
                target_s = trade.get('target_strike', 0)
                source_s = trade.get('source_strike', 0)
                if target_s < source_s:
                    gamma_regime_score = severity * 0.3  # rolling down = reducing risk

        elif gr_regime == 'ACCUMULATE':
            # Reward short puts (gamma welcome, assignment = buying cheap)
            if trade_type in ('OPEN', 'ADD') and trade.get('source_right', 'P') == 'P':
                gamma_regime_score = 2.0
            elif trade_type == 'STRANGLE':
                gamma_regime_score = 1.0
            # Penalize aggressive covered calls (don't sell shares at bottom)
            elif trade_type == 'COVERED CALL':
                gamma_regime_score = -2.0
    except Exception:
        pass

    # ── 11. Kelly criterion: hard +EV filter + scaled reward ──────────────
    # Income-mode default = ¼-Kelly (CENTRAL_PHILOSOPHY.md). Reward magnitude
    # halved vs the prior half-Kelly default; combined with the income_bias
    # modulation that scales positive kelly_score by (0.5+0.5*income_bias),
    # this gives much smaller "size up" pressure than a growth-mode optimizer.
    # Negative-EV penalty stays full strength.
    kelly_score = 0
    kelly_data = compute_kelly(trade, portfolio_state)
    if kelly_data is not None:
        ev = kelly_data['ev_total']
        quarter_k = kelly_data['quarter_kelly']
        if ev < 0:
            # Hard penalty for negative-EV trades (unscaled)
            kelly_score = -10 + ev / 100  # ev is negative, so this adds more penalty
        elif quarter_k > 0:
            # Reward: quarter_kelly = 0.20 (20% of capital under ¼-K sizing) → +4.0 pts
            # In the old half-K scheme the same trade would get +8 pts; now +4 (capped 5).
            kelly_score = min(5.0, quarter_k * 20)
        else:
            kelly_score = 0  # Kelly is positive but small

    # ── 12. Correlation / portfolio-Kelly soft penalty ────────────────────
    # All UNG bets are the same correlated bet. Per-trade Kelly says "+EV" but
    # aggregate short-put collateral may already be over portfolio Kelly.
    # User is OK with current sizing — middle ground:
    #   * Adding correlated exposure (strike-up rolls, fresh shorts) → soft penalty
    #     scaled by how far over Kelly we already are
    #   * Roll-flat (strike unchanged) → neutral
    #   * Reducing exposure (close, roll-down, long puts) → small bonus
    correlation_score = 0
    baseline_coll = portfolio_state.get('baseline_put_coll', 0)
    capital = max(1, portfolio_state.get('capital_base', _margin_capital_usd))
    kelly_pct = baseline_coll / capital  # e.g. 0.92 at $101k coll on $109k cap
    # Trigger when above 50% of capital in correlated put collateral
    over_kelly_mult = max(0, kelly_pct - 0.25) * 4  # 0.92 → 1.68

    trade_type = trade['type']
    target_right_corr = trade.get('source_right', 'P')
    if trade_type == 'COVERED CALL':
        target_right_corr = 'C'

    incremental_coll = 0
    qty_corr = abs(trade.get('roll_qty', trade.get('add_qty', trade.get('qty', 1))))
    if target_right_corr == 'P':
        if trade_type == 'ROLL':
            new_strike_c = trade.get('target_strike', 0)
            old_strike_c = trade.get('source_strike', 0)
            incremental_coll = qty_corr * (new_strike_c - old_strike_c) * 100
        elif trade_type in ('OPEN', 'ADD'):
            incremental_coll = qty_corr * trade.get('target_strike', 0) * 100
        elif trade_type in ('CLOSE', 'TAKE PROFIT', 'LET EXPIRE', 'ASSIGNMENT'):
            incremental_coll = -qty_corr * trade.get('source_strike', 0) * 100
        elif trade_type == 'BUY PUT':
            incremental_coll = -qty_corr * trade.get('target_strike', 0) * 100
    if trade_type == 'STRANGLE':
        # Short put leg adds correlated put collateral
        put_strike_s = trade.get('target_put_strike', 0)
        incremental_coll = qty_corr * put_strike_s * 100

    # Extra bite when rolling a put up across the spot (OTM → ITM): exactly the
    # "rolling into ITM" pattern the user flagged.
    itm_cross_bonus_penalty = 0
    if trade_type == 'ROLL' and target_right_corr == 'P' and over_kelly_mult > 0:
        old_s = trade.get('source_strike', 0)
        new_s = trade.get('target_strike', 0)
        if old_s < spot and new_s > spot:
            itm_amount = new_s - spot
            itm_cross_bonus_penalty = -(itm_amount * qty_corr * 100 / 2000) * over_kelly_mult

    if incremental_coll > 0 and over_kelly_mult > 0:
        # $1k incremental at over_kelly_mult=1 → -0.5 pts
        correlation_score = max(-8.0, -(incremental_coll / 1000) * 0.5 * over_kelly_mult)
    elif incremental_coll < 0 and over_kelly_mult > 0:
        # Reducing exposure when over Kelly: small bonus (don't push hard unwind)
        correlation_score = min(2.5, (-incremental_coll / 1000) * 0.25 * over_kelly_mult)

    correlation_score += max(-5.0, itm_cross_bonus_penalty)

    # Hard correlation cap (CENTRAL_PHILOSOPHY risk-budget constraint).
    # Soft penalty above doesn't stop a trade with a big headline credit from
    # piling on; we need a wall. Two-tier marginal-aware (mirrors CVaR cap):
    # 1) Baseline below cap + trade crosses: -50 (effective veto)
    # 2) Baseline already over + trade worsens: graduated -5/% extra over
    HARD_CORRELATION_CAP = 0.95  # 95% of capital in correlated put collateral
    if incremental_coll > 0:
        post_kelly_pct = (baseline_coll + incremental_coll) / capital
        if post_kelly_pct > HARD_CORRELATION_CAP:
            if kelly_pct <= HARD_CORRELATION_CAP:
                correlation_score = min(correlation_score, -50.0)
            else:
                _over = (post_kelly_pct - kelly_pct) * 100.0
                correlation_score = min(correlation_score, -_over * 5.0)

    # ── 13. Thesis tilt: bias scoring toward user's directional view ───────
    # User-set bias in [-1, +1]. Positive = bullish (penalize upside caps),
    # negative = bearish. Multiplied by the trade's delta_change (which directly
    # measures whether the trade adds or removes long exposure).
    thesis_tilt_val = portfolio_state.get('thesis_tilt', 0)
    # delta_change is in share-equivalents; +1000 means trade adds 1000 long delta.
    # Score = tilt × delta_change / 200, capped at ±5 points.
    thesis_score = 0
    if thesis_tilt_val != 0:
        thesis_score = max(-5.0, min(5.0, thesis_tilt_val * delta_change / 200.0))

    # ── 14. Probabilistic scenario P/L (E[P/L] across model bull/base/bear) ──
    # Consume the SHARED ScenarioDistribution at the trade's natural expiry
    # horizon (CENTRAL_PHILOSOPHY.md #1). Falls back to the legacy 3-point
    # list if dist isn't built yet (early-startup / unit tests).
    scenario_dist = portfolio_state.get('scenario_dist')
    legacy_scenarios = portfolio_state.get('scenarios') or []
    scenario_score = 0
    scenario_ev = 0
    trade_horizon_days = max(1, trade.get('target_dte', trade.get('source_dte', 30)))
    if scenario_dist is not None:
        scenarios = scenario_dist.at_horizon(trade_horizon_days)
    else:
        scenarios = legacy_scenarios
    if scenarios:
        T_trade = max(1, trade.get('target_dte', trade.get('source_dte', 30))) / 365.0
        target_strike_s = trade.get('target_strike', trade.get('source_strike', 0))
        target_right_s = trade.get('source_right', 'P')
        if trade['type'] == 'COVERED CALL':
            target_right_s = 'C'
        qty_s = abs(trade.get('roll_qty', trade.get('add_qty', trade.get('qty', 1))))
        iv_s = 0.50
        r_s = 0.04
        trade_type_s = trade['type']

        ev_total = 0.0
        for scen_spot, w in scenarios:
            # Compute P/L of the trade at this scenario terminal (trade's expiry).
            pl = 0.0
            if trade_type_s in ('OPEN', 'ADD', 'COVERED CALL') and target_strike_s > 0:
                # Sold short option for premium; pay intrinsic at scenario expiry
                # (premium kept ≈ current option mid; we approximate via BS now)
                opt_now = abs(bs_price(spot, target_strike_s, T_trade, r_s, iv_s, target_right_s))
                opt_at = max(0, scen_spot - target_strike_s if target_right_s == 'C' else target_strike_s - scen_spot)
                pl = (opt_now - opt_at) * qty_s * 100
            elif trade_type_s == 'ROLL' and target_strike_s > 0:
                src_strike = trade.get('source_strike', 0)
                src_dte = max(1, trade.get('source_dte', 30)) / 365.0
                src_now = abs(bs_price(spot, src_strike, src_dte, r_s, iv_s, target_right_s))
                src_at = max(0, scen_spot - src_strike if target_right_s == 'C' else src_strike - scen_spot)
                tgt_now = abs(bs_price(spot, target_strike_s, T_trade, r_s, iv_s, target_right_s))
                tgt_at = max(0, scen_spot - target_strike_s if target_right_s == 'C' else target_strike_s - scen_spot)
                # Net: holding source to expiry vs rolling = (src_at - 0) vs (src_close_now - tgt_now + tgt_at)
                # Simplified: rolling gives up (src_at - src_now) in exchange for receiving roll_credit_now - tgt_at
                # P/L of ROLL vs HOLD = (src_at - src_now) + (tgt_now - tgt_at)
                pl = ((src_at - src_now) + (tgt_now - tgt_at)) * qty_s * 100
            elif trade_type_s in ('LET EXPIRE', 'ASSIGNMENT'):
                src_strike = trade.get('source_strike', 0)
                src_right = trade.get('source_right', 'P')
                # Trade just lets expire; P/L = -intrinsic at scenario vs nothing
                # If OTM at scenario, full premium kept (positive vs holding)
                intrinsic_at = max(0, scen_spot - src_strike if src_right == 'C' else src_strike - scen_spot)
                # Already-collected premium is sunk; here we measure relative outcome
                pl = -intrinsic_at * qty_s * 100
            elif trade_type_s in ('CLOSE', 'TAKE PROFIT'):
                src_strike = trade.get('source_strike', 0)
                src_right = trade.get('source_right', 'P')
                src_now = abs(bs_price(spot, src_strike, T_trade, r_s, iv_s, src_right))
                # Closing now avoids future P/L; what we save is the value the option
                # WOULD have had at scenario expiry vs paying its value now
                intrinsic_at = max(0, scen_spot - src_strike if src_right == 'C' else src_strike - scen_spot)
                pl = (intrinsic_at - src_now) * qty_s * 100
            elif trade_type_s in ('BUY SHARES', 'SELL SHARES'):
                qty_shares = trade.get('add_qty', 0)
                if trade_type_s == 'SELL SHARES':
                    qty_shares = -abs(qty_shares)
                pl = qty_shares * (scen_spot - spot)
            ev_total += pl * w

        # Scale: $200 of E[P/L] = 1 point. Cap to ±6 to prevent dominance.
        scenario_score = max(-6.0, min(6.0, ev_total / 200.0))
        scenario_ev = round(ev_total, 0)

    # ── 14b. Anti-churn penalty (income-leakage prevention) ──────────────
    # Philosophy anti-pattern: "Strike-up rolls when current position has >60%
    # extrinsic remaining (eats time value for tiny credit)". Penalize ROLL
    # candidates that grab small credit while surrendering large remaining
    # extrinsic — pure friction tax.
    churn_score = 0
    if trade['type'] == 'ROLL':
        old_ext_pct = trade.get('ext_pct_old', 100)
        old_strike = trade.get('source_strike', 0)
        new_strike = trade.get('target_strike', 0)
        right = trade.get('source_right', 'P')
        # "Credit grab" direction: strike-UP for puts, strike-DOWN for calls
        is_credit_grab = (
            (right == 'P' and new_strike > old_strike) or
            (right == 'C' and new_strike < old_strike)
        )
        if is_credit_grab and old_ext_pct > 60:
            # Linear penalty from 60% (just barely too much) to 100% (pristine)
            churn_score = -((old_ext_pct - 60) / 40.0) * 4.0  # max -4 points

    # ── 14c. Kelly-negative veto (two-tier) ──────────────────────────────
    # Tier 1 (deep EV-loss, kelly < -8): scale economic + assignment_sim
    #   positives to 30% regardless of bias. At kelly < -8 the trade is
    #   EV-negative under the unified cyclical model — no mode should let
    #   a big credit headline a deeply EV-losing trade.
    # Tier 2 (mild EV-loss, kelly in [-8, -4]): same dampening but only in
    #   income mode (income_bias > 0.5). Growth mode is allowed to take
    #   marginally EV-negative trades for accumulation.
    # Surfaced 2026-05-18: with income_bias=0.496 (just below 0.5), the old
    # gate failed and a Roll 10x $10.5P→$11.5P with kelly=-12.6 still
    # scored 6.7 because the income_bias check was too tight.
    _income_bias_for_veto = portfolio_state.get('income_bias', 0.5)
    _deep_kelly_neg = kelly_score < -8
    _mild_kelly_neg = -8 <= kelly_score < -4 and _income_bias_for_veto > 0.5
    if _deep_kelly_neg or _mild_kelly_neg:
        if economic_score > 0:
            economic_score *= 0.3
        if assignment_score > 0:
            assignment_score *= 0.3

    # ── 14e. Hard drawdown CVaR constraint ─────────────────────────────────
    # CENTRAL_PHILOSOPHY strategic objective: max -10% monthly DD.
    # Project both BASELINE and AFTER-TRADE P/L at 30d 5%-CVaR using
    # delta-gamma + theta accrual. Penalize MARGINAL worsening, not the
    # absolute level — otherwise a portfolio that is already past -10% would
    # reject every candidate including the trades that REDUCE risk. Extra
    # weight applied to the portion of the post-trade DD that lies below
    # the -10% threshold so the constraint still bites at the boundary.
    cvar_penalty_score = 0.0
    try:
        _cvar_drop = portfolio_state.get('cvar_30d_5pct_drop', 0.0)
        _cap = portfolio_state.get('capital_base', 0.0) or 0.0
        if _cvar_drop > 0 and _cap > 0:
            _cur_d = portfolio_state['total_delta']
            _cur_g = portfolio_state['total_gamma']
            _cur_t = portfolio_state['total_theta']
            # Expiry-aware 30d theta accrual (cycle 40): weekly_theta already
            # drops as positions expire, so 4 full weeks + 2/7 of week 5 ≈ 30
            # days captures the real accrual rather than over-counting with
            # `total_theta × 30`. Falls back to legacy if weekly_theta missing.
            _wt_dict = portfolio_state.get('weekly_theta', {}) or {}
            if _wt_dict:
                _wt_vals = list(_wt_dict.values())
                if len(_wt_vals) >= 5:
                    _theta_30d_base = sum(_wt_vals[:4]) + (_wt_vals[4] * 2.0 / 7.0)
                else:
                    _theta_30d_base = sum(_wt_vals)
            else:
                _theta_30d_base = _cur_t * 30.0
            _baseline_tail = (-_cur_d * _cvar_drop
                              + 0.5 * _cur_g * (_cvar_drop ** 2)
                              + _theta_30d_base)
            _baseline_dd = _baseline_tail / _cap
            _new_delta = _cur_d + delta_change
            _new_gamma = _cur_g + gamma_change
            _new_theta = _cur_t + theta_change
            # Trade's theta_change applies for ~its DTE; for simplicity scale
            # by min(30, target_dte) — approximates the new position's
            # contribution to 30d accrual.
            _trade_dte = max(1, trade.get('target_dte', trade.get('source_dte', 30)))
            _theta_30d_after = _theta_30d_base + theta_change * min(30.0, _trade_dte)
            _tail_pnl = (-_new_delta * _cvar_drop
                         + 0.5 * _new_gamma * (_cvar_drop ** 2)
                         + _theta_30d_after)
            _dd_frac = _tail_pnl / _cap
            # Penalize MARGINAL worsening (trade making tail DD worse), with
            # extra weight if the trade CROSSES the -10% threshold from a
            # previously-safe baseline. Pre-existing breaches alone do NOT
            # penalize neutral/improving trades — that would freeze the book.
            _worsening = max(0.0, _baseline_dd - _dd_frac)
            _crossing_extra = 0.0
            if _baseline_dd >= -0.10 and _dd_frac < -0.10:
                _crossing_extra = (-0.10 - _dd_frac) * 400.0
            if _worsening > 0 or _crossing_extra > 0:
                cvar_penalty_score = -min(80.0,
                                          _worsening * 600.0 + _crossing_extra)
    except Exception:
        cvar_penalty_score = 0.0

    # ── 14d. Income score (CENTRAL_PHILOSOPHY.md $1500/wk strategic target) ─
    # Rewards trades that close the gap toward target weekly income.
    # NOT a duplicate of theta_rate_bonus (which rewards raw theta/d) or
    # economic_score (which rewards $ credit) — this is gap-relative to the
    # $1500/wk target and weighted by income_bias so it only dominates when
    # the system is in income-harvest mode.
    income_score = 0.0
    _target_wk = portfolio_state.get('target_weekly_income', 1500.0)
    _cur_wk = portfolio_state.get('avg_weekly_theta', 0.0)  # already × 7 weekly
    _weekly_contrib = theta_change * 7.0  # $/d delta → $/wk delta
    if theta_change > 0:
        _gap = max(0.0, _target_wk - _cur_wk)
        # Denominator: at least $200 so a gap-closed portfolio still gets some
        # reward for adding income; at most the full gap so we don't overweight.
        _denom = max(_gap, 200.0)
        income_score = min(3.0, (_weekly_contrib / _denom) * 3.0)
    elif theta_change < 0:
        # Trade reduces weekly income — penalize lightly (theta_rate_bonus
        # also handles part of this; keep small to avoid double-counting).
        income_score = max(-2.0, _weekly_contrib / 500.0)

    # ── 15. Growth/income bias modulation (CENTRAL_PHILOSOPHY.md) ─────────
    # Auto-computed income_bias scales components so they serve the current
    # cyclical+price+ROI regime. Manual thesis_tilt remains an additive
    # overlay (already in thesis_score) — but its INFLUENCE shrinks in
    # income mode (you can lean bullish, but the optimizer respects the
    # cyclical/price/ROI reality more).
    #
    # Negative-EV Kelly penalties and downside protection stay full strength
    # (we never dilute risk signals based on bias).
    _income_bias = portfolio_state.get('income_bias', 0.5)
    _growth_bias = 1.0 - _income_bias
    # thesis (user bullish/bearish tilt): more impact in growth mode
    thesis_score *= _growth_bias if thesis_score != 0 else 1.0
    # Kelly reward: scales 0.5x at full income, 1x at neutral, 1x at full growth
    # (positive only — negative-EV penalty stays full strength)
    if kelly_score > 0:
        kelly_score *= 0.5 + 0.5 * _income_bias
    # Gamma regime ACCUMULATE rewards (positive = buy more puts): scale by growth
    # (don't push aggressive accumulation when bias says harvest mode)
    if gamma_regime_score > 0 and trade['type'] in ('OPEN', 'ADD'):
        gamma_regime_score *= _growth_bias
    # Income score weighting: full strength at income_bias=1, half at 0.5, near
    # zero at 0 — income reward is meaningless in growth mode.
    if income_score > 0:
        income_score *= 0.5 + 0.5 * _income_bias
    # (Negative income penalty stays full strength regardless of bias.)

    total = (delta_score + economic_score + theta_rate_bonus + drain_bonus
             + conc_score + type_bonus + curve_score + waterfall_score
             + assignment_score + gamma_regime_score + kelly_score
             + correlation_score + thesis_score + scenario_score
             + churn_score + income_score + cvar_penalty_score
             + debit_aversion  # cycle 138 (negative for paid-to-move rolls)
             - gamma_penalty - crash_penalty)

    # Store breakdown on the trade dict for display
    trade['_score_breakdown'] = {
        'delta': round(delta_score, 1),
        'economic': round(economic_score, 1),
        'theta_rate': round(theta_rate_bonus, 1),
        'drain': round(drain_bonus, 1),
        'concentration': round(conc_score, 1),
        'type_bonus': round(type_bonus, 1),
        'recovery': round(curve_score, 1),
        'waterfall': round(waterfall_score, 1),
        'assignment_sim': round(assignment_score, 1),
        'gamma_regime': round(gamma_regime_score, 1),
        'kelly': round(kelly_score, 1),
        'correlation': round(correlation_score, 1),
        'thesis': round(thesis_score, 1),
        'scenario': round(scenario_score, 1),
        'churn': round(churn_score, 1),
        'income': round(income_score, 1),
        'cvar_dd': round(cvar_penalty_score, 1),
        'crash_penalty': round(-crash_penalty, 1),
        'debit_aversion': round(debit_aversion, 1),
    }
    # Surface raw $ for the UI tooltip
    trade['_scenario_ev'] = scenario_ev
    if kelly_data is not None:
        trade['_kelly'] = kelly_data
    try:
        if target_strike_val > 0 and trade['type'] in ('ROLL', 'OPEN', 'ADD', 'COVERED CALL'):
            trade['_p_itm'] = round(float(p_itm) * 100)
    except Exception:
        pass

    return total


def get_technicals_cached():
    """Get cached technicals data (inner dict). Populate on miss so callers
    that run before the gamma_regime path don't get the empty fallback that
    silently neutralizes price_band-driven logic."""
    data = _technicals_cache.get('data') if _technicals_cache else None
    if data is None:
        try:
            data = compute_technicals()
        except Exception:
            data = None
    return data or {}


def apply_trade_to_state(state, trade, spot, iv, today):
    """Apply a trade to the portfolio state and return new state."""
    new_state = dict(state)
    new_state['total_theta'] = state['total_theta'] + trade.get('theta_change', 0)
    new_state['total_delta'] = state['total_delta'] + trade.get('delta_change', 0)
    new_state['total_gamma'] = state['total_gamma'] + trade.get('gamma_change', 0)
    new_state['total_vega'] = state['total_vega'] + trade.get('vega_change', 0)

    # Update positions list (remove rolled, add new)
    new_positions = list(state['positions'])

    if trade['type'] == 'ROLL':
        source_exp = trade.get('source_exp')
        source_strike = trade.get('source_strike')
        source_right = trade.get('source_right')
        roll_qty = trade.get('roll_qty', 1)

        for i, (exp, strike, right, qty, avg) in enumerate(new_positions):
            if exp == source_exp and strike == source_strike and right == source_right:
                new_qty = qty + roll_qty  # qty is negative, roll_qty positive
                if abs(new_qty) > 0:
                    new_positions[i] = (exp, strike, right, new_qty, avg)
                else:
                    new_positions.pop(i)
                break

        # Add new position
        target_exp = trade.get('target_exp')
        target_strike = trade.get('target_strike')
        new_positions.append((target_exp, target_strike, source_right, -roll_qty, 0))

    elif trade['type'] == 'ADD':
        target_exp = trade.get('target_exp')
        target_strike = trade.get('target_strike')
        add_qty = trade.get('add_qty', 3)
        new_positions.append((target_exp, target_strike, 'P', -add_qty, 0))

    elif trade['type'] == 'STRANGLE':
        target_exp = trade.get('target_exp')
        put_strike = trade.get('target_put_strike')
        call_strike = trade.get('target_call_strike')
        new_positions.append((target_exp, put_strike, 'P', -1, 0))
        new_positions.append((target_exp, call_strike, 'C', -1, 0))

    elif trade['type'] == 'OPEN':
        target_exp = trade.get('target_exp')
        target_strike = trade.get('target_strike')
        add_qty = trade.get('add_qty', 3)
        action = trade.get('action', '')
        if 'strangle' in action.lower():
            # Cycle 186b: covered strangle — add BOTH legs
            import re as _re_strangle
            _strikes = _re_strangle.findall(r'\$(\d+\.?\d*)', action)
            if len(_strikes) >= 2:
                new_positions.append((target_exp, float(_strikes[0]), 'P', -add_qty, 0))
                new_positions.append((target_exp, float(_strikes[1]), 'C', -add_qty, 0))
            else:
                new_positions.append((target_exp, target_strike, 'P', -add_qty, 0))
        else:
            right = 'C' if 'C ' in action or action.endswith('C') else 'P'
            new_positions.append((target_exp, target_strike, right, -add_qty, 0))

    elif trade['type'] == 'SELL SHARES':
        # Cycle 180 P1 fix: update modeled share count so downstream
        # covered-call capacity checks reflect shares already "sold."
        _sell_qty = abs(trade.get('qty', trade.get('add_qty', 0)))
        new_state['shares'] = max(0, new_state.get('shares', SHARES) - _sell_qty)

    elif trade['type'] == 'BUY SHARES':
        _buy_qty = abs(trade.get('qty', trade.get('add_qty', 0)))
        new_state['shares'] = new_state.get('shares', SHARES) + _buy_qty

    elif trade['type'] == 'BUY BOXX':
        # BOXX is uncorrelated to UNG — no delta/gamma impact.
        # Adds theta_change (BOXX yield) and reduces idle cash.
        _boxx_cost = trade.get('boxx_cost', 0)
        new_state['boxx_value'] = new_state.get('boxx_value', 0) + _boxx_cost
        new_state['boxx_shares'] = new_state.get('boxx_shares', 0) + trade.get('add_qty', 0)
        # theta_change carries the daily yield; total_theta already updated below

    elif trade['type'] == 'COVERED CALL':
        target_exp = trade.get('target_exp')
        target_strike = trade.get('target_strike')
        add_qty = trade.get('add_qty', 3)
        new_positions.append((target_exp, target_strike, 'C', -add_qty, 0))

    elif trade['type'] == 'BUY PUT':
        target_exp = trade.get('target_exp')
        target_strike = trade.get('target_strike')
        add_qty = trade.get('add_qty', 2)
        new_positions.append((target_exp, target_strike, 'P', add_qty, 0))  # positive qty = long

    elif trade['type'] in ('CLOSE', 'TAKE PROFIT', 'LET EXPIRE', 'ASSIGNMENT'):
        source_exp = trade.get('source_exp')
        source_strike = trade.get('source_strike')
        source_right = trade.get('source_right')
        close_qty = trade.get('roll_qty') or trade.get('qty')
        for i, (exp, strike, right, qty, avg) in enumerate(new_positions):
            if exp == source_exp and strike == source_strike and right == source_right:
                # Partial close: if roll_qty specified and < |qty|, reduce instead of removing.
                if close_qty and abs(close_qty) < abs(qty):
                    sign = 1 if qty > 0 else -1
                    new_qty = qty - sign * abs(close_qty)
                    new_positions[i] = (exp, strike, right, new_qty, avg)
                else:
                    new_positions.pop(i)
                break
        # Cycle 199: synthetic early-assignment also reduces shares
        _shares_to_sell = trade.get('shares_sold', 0)
        if _shares_to_sell > 0:
            new_state['shares'] = max(0, int(new_state.get('shares', SHARES) or SHARES) - int(_shares_to_sell))
            new_state['total_delta'] = new_state.get('total_delta', 0) - _shares_to_sell

    # Recompute smoothness and concentration from new positions
    new_state['positions'] = new_positions
    recomputed = compute_portfolio_state(new_positions, spot, iv, today)
    new_state['smoothness'] = recomputed['smoothness']
    new_state['max_concentration'] = recomputed['max_concentration']
    new_state['weekly_theta'] = recomputed['weekly_theta']
    new_state['expiry_theta'] = recomputed['expiry_theta']
    # Refresh avg_weekly_theta so the portfolio-quality evaluator sees the
    # updated weekly income after the trade (was stale on the inherited dict).
    # Cycle 173: user decision on cycle-160 metric question — use NEAR 2-WEEK
    # AVERAGE instead of all-weeks average. Rationale: "we all know this
    # system will renew contracts" — far-future empty weeks are irrelevant
    # for a wheel strategy that continuously rolls. Near-2-week captures
    # the steady-state income level without the cliff artifacts from
    # active-bucket averaging (cycles 164-170).
    try:
        _wt_vals = list(recomputed['weekly_theta'].values())
        _near = [v for v in _wt_vals[:2] if v > 0]
        new_state['avg_weekly_theta'] = float(sum(_near) / len(_near)) if _near else 0.0
    except Exception:
        pass  # leave previous value if recompute fails

    return new_state


class ScenarioDistribution:
    """Single source of truth for the forward UNG spot distribution.

    See CENTRAL_PHILOSOPHY.md principle #1: every scoring component should
    consume from one instance of this class per recommendations cycle.

    Builds discrete distributions at multiple horizons (5d/14d/30d/45d/60d)
    from a regime-adjusted log-normal kernel anchored to the NG fundamentals
    model (z-score) plus realized vol, plus user-provided stress tails.
    """

    DEFAULT_HORIZONS = (5, 14, 30, 45, 60)
    # 7-point quantile kernel covering ±2σ + 0
    _QUANTILES = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)
    _Q_WEIGHTS = (0.05, 0.20, 0.20, 0.10, 0.20, 0.20, 0.05)  # sums to 1.0

    def __init__(self, spot, z_score=0.0, sigma_annual=0.45,
                 contango_per_day=-0.001, outlook=None, stress=None,
                 horizons=None, seasonal_drift_per_day=0.0,
                 seasonal_vol_scale=1.0):
        self.spot = float(spot)
        self.z_score = float(z_score)
        self.sigma_annual = float(sigma_annual)
        # Negative = price drifts down per day (UNG contango). -0.001 ≈ -3%/mo.
        self.contango_per_day = float(contango_per_day)
        # Excess seasonal drift for the CURRENT calendar month (per-day log return).
        # Sum of seasonal_drift across 12 months ≈ 0, so this only nudges the kernel
        # toward typical month-of-year drift on top of contango.
        self.seasonal_drift_per_day = float(seasonal_drift_per_day)
        # Stress-tail vol multiplier for the CURRENT calendar month.
        # Winter > 1.0 (cold-snap risk), shoulder < 1.0 (quiet).
        self.seasonal_vol_scale = float(seasonal_vol_scale)
        # Optional model anchor: {'ung_bull':..,'ung_base':..,'ung_bear':..}
        self.outlook = outlook or {}
        # Optional tail anchor: {'crash': -0.117, 'spike': 0.10}
        self.stress = stress or {}
        self.horizons = tuple(horizons) if horizons else self.DEFAULT_HORIZONS
        self._distributions = {h: self._build(h) for h in self.horizons}

    def _build(self, days):
        """Return list of (spot, weight) tuples for horizon `days`."""
        import math as _m
        if days <= 0:
            return [(self.spot, 1.0)]
        T = days / 365.0
        sigma_h = self.sigma_annual * _m.sqrt(T)
        # Cyclical-first drift composition (CENTRAL_PHILOSOPHY.md "Cyclicality is the spine"):
        # seasonal residual is the base; z-score is a smaller modulator (halved
        # from 0.0008 → 0.0004 to avoid stacking on top of seasonality).
        regime_drift = self.seasonal_drift_per_day + self.z_score * 0.0004
        log_spot = _m.log(max(0.01, self.spot))
        mu = log_spot + (regime_drift + self.contango_per_day) * days - 0.5 * sigma_h ** 2

        # Base quantile points
        points = []
        for q, w in zip(self._QUANTILES, self._Q_WEIGHTS):
            sp = _m.exp(mu + q * sigma_h)
            points.append([sp, w])

        # Optional model anchor: nudge the central mass toward (bull/base/bear)
        # midpoint if outlook is provided. Light blend so the kernel stays smooth.
        anchor = self._central_anchor(days)
        if anchor is not None:
            # Shift all points by the difference between kernel-mean and anchor
            kernel_mean = sum(sp * w for sp, w in points)
            shift = anchor - kernel_mean
            for p in points:
                p[0] += shift

        # Stress tails: small probability mass at extreme moves, scaled by sqrt(t)
        # AND by seasonal vol (winter > shoulder per CENTRAL_PHILOSOPHY axis A).
        # Clamp returns to >= -95% so prices stay strictly positive even at
        # long horizons (sqrt-of-time scaling can over-shoot otherwise).
        if self.stress:
            tail_scale = _m.sqrt(days / 5.0) if days > 0 else 1.0
            tail_scale *= self.seasonal_vol_scale  # widen winter tails, narrow shoulder
            # Scale base weights down so the post-append total still sums to 1.0
            # without further renormalization (keeps the tail's 2% weight honest).
            n_tails = sum(1 for k in ('crash', 'spike') if k in self.stress)
            tail_each = 0.02
            base_scale = max(0.0, 1.0 - n_tails * tail_each)
            for p in points:
                p[1] *= base_scale
            if 'crash' in self.stress:
                crash_pct = max(-0.95, self.stress['crash'] * tail_scale)
                points.append([self.spot * (1 + crash_pct), tail_each])
            if 'spike' in self.stress:
                spike_pct = self.stress['spike'] * tail_scale
                points.append([self.spot * (1 + spike_pct), tail_each])

        # Final safety renorm (handles edge cases where base_scale rounded oddly).
        total_w = sum(w for _, w in points)
        if total_w > 0:
            for p in points:
                p[1] = p[1] / total_w
        return [(float(p[0]), float(p[1])) for p in points]

    def _central_anchor(self, days):
        """Return a target central spot from outlook (bull/base/bear midpoint)
        for the given horizon, or None if outlook unavailable."""
        bull = self.outlook.get('ung_bull')
        base = self.outlook.get('ung_base')
        bear = self.outlook.get('ung_bear')
        if not (bull and base and bear):
            return None
        # Outlook is implicitly ~30 day; scale linearly to horizon
        anchor_30d = 0.30 * bull + 0.40 * base + 0.30 * bear
        # Blend to current spot at shorter horizons (less of the model has played out)
        scale = min(1.0, days / 30.0)
        return self.spot + (anchor_30d - self.spot) * scale

    def at_horizon(self, days):
        """Return discrete distribution closest to `days`. Always non-empty."""
        if not self._distributions:
            return [(self.spot, 1.0)]
        h = min(self.horizons, key=lambda x: abs(x - max(1, days)))
        return self._distributions[h]

    def expected(self, days, payoff_fn):
        """E[payoff_fn(spot)] at horizon `days`."""
        return sum(payoff_fn(sp) * w for sp, w in self.at_horizon(days))

    def prob_above(self, K, days):
        return sum(w for sp, w in self.at_horizon(days) if sp > K)

    def prob_below(self, K, days):
        return sum(w for sp, w in self.at_horizon(days) if sp < K)

    def quantile(self, days, p):
        """Spot price at cumulative-weight quantile p ∈ [0,1] at horizon `days`.
        p=0.05 returns ~5th percentile (left tail), p=0.95 returns ~95th."""
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"quantile p must be in [0,1], got {p}")
        dist = sorted(self.at_horizon(days), key=lambda x: x[0])
        cum = 0.0
        last_sp = self.spot
        for sp, w in dist:
            cum += w
            last_sp = sp
            if cum >= p:
                return float(sp)
        return float(last_sp)

    def cvar_loss(self, days, alpha=0.10):
        """E[max(0, spot_loss)] in the worst alpha tail (returns positive $ loss)."""
        dist = sorted(self.at_horizon(days), key=lambda x: x[0])  # ascending spot
        cum, tail_sum, tail_w = 0.0, 0.0, 0.0
        for sp, w in dist:
            if cum >= alpha:
                break
            take = min(w, alpha - cum)
            loss = max(0.0, self.spot - sp)
            tail_sum += loss * take
            tail_w += take
            cum += take
        return tail_sum / tail_w if tail_w > 0 else 0.0


def evaluate_portfolio_quality(state, target_weekly_income=1500.0):
    """Unified portfolio-quality scalar tying every strategic-objective
    dimension into one $-normalized number. Higher = better.

    Components (all in dollar-equivalent units so signs and magnitudes are
    directly comparable):

      income_gap_$   — weekly income above target rewards directly; shortfall
                       costs 1.5× as much per dollar (asymmetric: missing the
                       target is worse than overshooting)
      dd_penalty_$   — projected 30d 5%-CVaR portfolio DD below -10% of capital
                       penalized hard ($ for $); -0% if safe
      delta_gap_$    — squared shortfall vs target delta, $0.10 per shares²
                       (small linear deviations are fine; large ones cost)
      smoothness_$   — +$500 at perfectly-smooth (1.0), 0 at 0.0
      tail_hedge_$   — -$2000 flat if tail-hedge count below floor
      pillar_drift_$ — +$300 per +1.0 pillar score (mildly biased trades
                       through the cyclical model are good in income mode)

    Returns dict with the score and a breakdown for dashboard rendering.
    """
    capital = float(state.get('capital_base', 100_000) or 100_000)
    spot = float(state.get('spot', 0.0))
    weekly_income = float(state.get('avg_weekly_theta', 0.0) or 0.0)
    total_delta = float(state.get('total_delta', 0.0) or 0.0)
    total_gamma = float(state.get('total_gamma', 0.0) or 0.0)
    total_theta = float(state.get('total_theta', 0.0) or 0.0)
    smoothness = float(state.get('smoothness', 0.0) or 0.0)
    # Cycle 176: derive tail_hedge_qty from ACTUAL positions in state,
    # not from a stale state field. This makes evaluate_portfolio_quality
    # self-consistent after apply_trade_to_state adds/removes LEAPS.
    # Previously tail_hedge_qty was set once in compute_recommendations
    # and never updated when the beam applied BUY PUT trades.
    tail_floor = int(state.get('tail_hedge_floor', 2) or 2)
    _TAIL_MIN_DTE = 180
    _positions = state.get('positions', []) or []
    try:
        _today = date.today()
        tail_qty = sum(
            abs(qty) for exp_s, strike, right, qty, _ in _positions
            if right == 'P' and qty > 0
            and (datetime.strptime(exp_s, '%Y-%m-%d').date() - _today).days >= _TAIL_MIN_DTE
        )
    except Exception:
        tail_qty = int(state.get('tail_hedge_qty', 0) or 0)
    pillar_scores = state.get('pillar_scores', {}) or {}
    pillar_sum = float((pillar_scores.get('tech') or 0)
                       + (pillar_scores.get('fund') or 0)
                       + (pillar_scores.get('yoy') or 0))
    sd = state.get('scenario_dist')

    # Income gap
    gap = weekly_income - target_weekly_income
    income_gap = gap if gap >= 0 else gap * 1.5

    # CVaR drawdown — cycle 137 (user-approved):
    # Old formula treated the 5%-tail-of-30d as if it were certain, producing
    # a -$23k+ "freeze button" wall that blocked deployment even in cheap
    # regimes. User principle: dd penalty should reflect (a) likelihood of
    # the tail event and (b) time-to-react.
    # New: 7-day horizon (matches weekly rebalance cadence) AND expected-loss
    # weighting (multiply by tail probability α=0.05).
    HORIZON_D = 7
    ALPHA = 0.05
    cvar_drop = float(state.get('cvar_7d_5pct_drop', 0.0) or 0.0)
    if cvar_drop <= 0 and sd is not None:
        try:
            cvar_drop = float(sd.cvar_loss(HORIZON_D, alpha=ALPHA))
        except Exception:
            cvar_drop = 0.0
    # Theta over the response horizon — first week of weekly_theta.
    weekly_theta_dict = state.get('weekly_theta', {}) or {}
    if weekly_theta_dict:
        wt_vals = list(weekly_theta_dict.values())
        theta_horizon = wt_vals[0] if wt_vals else 0.0  # first week
    else:
        theta_horizon = total_theta * HORIZON_D
    # Worst-case (conditional-on-tail) P&L drivers, kept on dd_diagnostics
    # so the operator can see raw tail magnitudes:
    if cvar_drop > 0:
        delta_loss = -total_delta * cvar_drop
        gamma_convexity = 0.5 * total_gamma * (cvar_drop ** 2)
        theta_offset = theta_horizon
    else:
        delta_loss = 0.0
        gamma_convexity = 0.0
        theta_offset = 0.0
    tail_pnl_worst = delta_loss + gamma_convexity + theta_offset
    # Expected tail loss = P(tail event) × loss-given-tail.
    # This is the value used to drive the dd_penalty (instead of raw tail_pnl).
    expected_tail_loss = ALPHA * tail_pnl_worst
    expected_dd_frac = expected_tail_loss / capital if capital > 0 else 0.0
    # Penalize only when EXPECTED drawdown exceeds -10% of capital, not the
    # raw worst-case 5%-tail. With ALPHA=0.05 this is roughly equivalent to
    # asking "would I lose >-200% of capital in the worst 5%?" — a true
    # catastrophe threshold, not a normal-operations sanity check.
    if expected_dd_frac < -0.10 and capital > 0:
        dd_penalty = (expected_dd_frac + 0.10) * capital
    else:
        dd_penalty = 0.0
    # Cycle 180: HARD DD VETO — the soft dd_penalty above is a gradient
    # for ranking; this is a TRUE block. CENTRAL_PHILOSOPHY says "max -10%
    # monthly drawdown." If the RAW 7-day 5%-CVaR tail loss exceeds -15%
    # of capital (stricter than expected-value × α), the trade is vetoed.
    # The 15% threshold on a 7-day horizon roughly maps to -10% monthly
    # (accounting for theta offset over 30 days).
    dd_frac = tail_pnl_worst / capital if capital > 0 else 0.0
    _hard_dd_veto = bool(dd_frac < -0.15)  # bool() to avoid numpy.bool_ JSON error
    theta_30d = theta_horizon * (30.0 / HORIZON_D)  # legacy field for diag display

    # Delta gap (quadratic, mild)
    # Cycle 195: target_delta tracks state's current shares (not module global)
    # so SELL SHARES candidates get a fair comparison — target updates with shares.
    _shares_for_target = state.get('shares')
    target_delta, _, _ = compute_target_delta(spot, _shares_for_target) if spot > 0 else (total_delta, '', 0.0)
    delta_gap_shares = total_delta - target_delta
    delta_gap = -(delta_gap_shares ** 2) * 0.0001  # 1000-share gap = -$100

    # Smoothness bonus — enhanced with forward projection stability when available.
    # Forward projection (forward_cache.json) provides projected_income_stability
    # which captures the TRAJECTORY of income over 6 weeks including expiry cliffs.
    # This naturally drives DTE diversification without hacks.
    _fwd_stability = None
    try:
        import os as _os_fwd
        _fwd_path = _os_fwd.path.join(_os_fwd.path.dirname(_os_fwd.path.abspath(__file__)), 'forward_cache.json')
        if _os_fwd.path.exists(_fwd_path):
            _fwd = json.loads(open(_fwd_path).read())
            _fwd_age = (datetime.now() - datetime.fromisoformat(_fwd.get('timestamp', '2000-01-01'))).total_seconds()
            if _fwd_age < 600:  # fresh within 10 min
                _fwd_stability = float(_fwd.get('projected_income_stability', 0))
    except Exception:
        pass
    # Use forward stability if available (better metric), else fall back to smoothness
    _effective_smoothness = _fwd_stability if _fwd_stability is not None else smoothness
    smoothness_bonus = _effective_smoothness * 500.0

    # Cycle 191: per-position forward coverage bonus. Each position gets a
    # bonus proportional to how much of the 6-week projection horizon it
    # covers. A 17d put covers 40% of the horizon (expires at week 2.4).
    # A 37d put covers 88%. A 52d put covers 100%. This naturally makes
    # longer-DTE positions score higher because they contribute income
    # across MORE future weeks — solving DTE diversification without hacks.
    # Also: detect put+call pairs at same expiry (strangle evaluation).
    _FORWARD_HORIZON_DAYS = 42  # 6 weeks
    _forward_bonus = 0.0
    _strangle_bonus = 0.0
    try:
        _today_fc = date.today()
        _put_by_exp = {}  # {expiry: total_qty}
        _call_by_exp = {}
        for _exp_s, _K, _right, _qty, _avg in _positions:
            if _qty >= 0:
                continue  # only short positions
            _exp_d = datetime.strptime(_exp_s, '%Y-%m-%d').date()
            _dte = max(0, (_exp_d - _today_fc).days)
            # Forward coverage: fraction of 6-week horizon this position lives
            _coverage = min(1.0, _dte / _FORWARD_HORIZON_DAYS)
            # Bonus: more coverage = more stable income contribution
            # Scale by daily theta × coverage
            _th = abs(bs_theta(spot, _K, max(1, _dte) / 365.0, 0.045,
                               float(state.get('iv_est', 0.45) or 0.45), _right)) * abs(_qty) * 100
            _forward_bonus += _th * _coverage * 7  # weekly theta × coverage fraction
            # Track for strangle detection
            if _right == 'P':
                _put_by_exp[_exp_s] = _put_by_exp.get(_exp_s, 0) + abs(_qty)
            elif _right == 'C':
                _call_by_exp[_exp_s] = _call_by_exp.get(_exp_s, 0) + abs(_qty)
        # Strangle bonus: reward expiries that have BOTH puts and calls
        # (delta partially cancels, double premium, natural wheel)
        for _exp_s in _put_by_exp:
            if _exp_s in _call_by_exp:
                _paired = min(_put_by_exp[_exp_s], _call_by_exp[_exp_s])
                _strangle_bonus += _paired * 5.0  # $5 per paired contract
    except Exception:
        _forward_bonus = 0.0
        _strangle_bonus = 0.0

    # Cycle 189: expiry concentration penalty. User: "why all 33 contracts
    # at 6/18? in 3 weeks I have a massive rollover — that's not smooth."
    # The smoothness metric measures THETA distribution over weeks, but NOT
    # position concentration per expiry. A portfolio with all contracts at
    # one date creates operational rollover risk.
    # Penalty = proportional to max_concentration². Concentration > 40%
    # at one expiry is risky; > 60% is very bad.
    _max_conc = float(state.get('max_concentration', 0) or 0)
    _conc_penalty = -(_max_conc ** 2) * 1000.0 if _max_conc > 0.3 else 0.0

    # Cycle 194: gamma load penalty for share-heavy portfolios.
    # User: "we have actual shares vs more pure option positions, the
    # optimizer should have a preference for gamma control."
    # Shares have ZERO gamma — pure linear delta. Short options have
    # NEGATIVE gamma — convex loss. As short gamma grows, each $1 spot
    # drop costs proportionally MORE than the previous $1 (the orange
    # dollar-delta curve we showed: $8,470/$ down vs $4,435/$ up).
    # Penalty proportional to short gamma squared (convex), scaled by
    # share count (more shares = more existing linear delta = more
    # important to control the option convexity).
    _shares_in_state = float(state.get('shares', SHARES) or SHARES)
    _short_gamma_abs = abs(min(0, total_gamma))
    # Cycle 202: use FULL expected gamma loss derived from variance, not
    # just 5%-tail. User: "use the tail and mean-reverse risk." The
    # statistically correct expected dollar loss from gamma over horizon T is:
    #   E[-0.5 × |Γ| × ΔS²] = -0.5 × |Γ| × Var(ΔS)
    # Where Var(ΔS) = (σ·S)² × T  (log-normal approximation for short T).
    # Plus mean-reversion uplift: after a sharp move, P(reversal) > 50%,
    # so variance is HIGHER than random-walk would predict. Apply 1.5x
    # uplift when |z-score| > 0.5 (signal of stretched move).
    _gamma_load_penalty = 0.0
    try:
        _spot_g = float(state.get('spot', 11.0) or 11.0)
        _iv_g = float(state.get('iv_est', 0.45) or 0.45)
        _T = 7.0 / 252.0  # weekly horizon
        # Random-walk variance over T (one-week)
        _var_spot = (_iv_g * _spot_g) ** 2 * _T
        # Full expected gamma loss per week — NO arbitrary multiplier
        _gamma_loss_full = 0.5 * _short_gamma_abs * _var_spot
        # Cycle 202b: theta income (in income_gap) already compensates for
        # NORMAL gamma cost under IV-realized parity. Penalty should capture
        # only EXCESS variance — sources:
        #   - Mean reversion uplift: ~30% excess when |z| > 0.5
        #   - Stretched move uplift: ~20% when UNG moved > 5% in past week
        #   - Realized > implied IV: typical ~10%
        # Total typical excess: 30-60% of theoretical gamma loss
        try:
            _z_now = float(_model_zscore or 0)
        except Exception:
            _z_now = 0.0
        _excess_pct = 0.20  # base 20% (realized > implied + transaction cost)
        if abs(_z_now) > 0.5:
            _excess_pct += 0.20  # mean-reversion uplift
        # User said UNG rallied 11% this week — extra uplift
        # (Detected via spot vs recent average; for now use simple share-heavy proxy)
        _share_ratio = min(2.0, _shares_in_state / 5000.0)
        _gamma_load_penalty = -_gamma_loss_full * _excess_pct * _share_ratio
    except Exception:
        _gamma_load_penalty = 0.0

    # Tail-hedge floor
    # Cycle 177: risk-derived tail-hedge penalty. No hardcoded $ values.
    # Computes the expected crash-scenario benefit PER LEAPS put using the
    # same CVaR model as dd_penalty, then penalizes missing hedges by the
    # probability-weighted loss they would have prevented.
    #
    # Methodology:
    #   1. Compute 7-day 5%-CVaR drop (same as dd_penalty)
    #   2. Estimate per-LEAPS-put benefit in a crash: a 200-DTE ATM put
    #      has delta ~-0.45 and gamma ~0.02 per share. In a CVaR crash:
    #      benefit = (-leaps_delta × cvar_drop + 0.5 × leaps_gamma × cvar²) × 100
    #   3. Penalty per missing LEAPS = α × benefit (probability-weighted)
    #   4. Scale by portfolio leverage (more short exposure = more valuable hedge)
    #
    # This makes the penalty DYNAMIC: high-IV / extended markets → bigger
    # CVaR → bigger penalty for missing hedges. Calm markets → smaller.
    # Cycle 182: removed hardcoded floor. User: "2 LEAPS does nothing
    # but a few hundred bucks in a crash." The math confirms: 2 LEAPS
    # covers 1.5% of a 20% crash. Theater, not protection.
    # Now: penalty is purely the UNHEDGED portion of portfolio tail risk.
    # Each existing LEAPS reduces the penalty proportionally. The beam
    # decides whether buying LEAPS is worth the premium cost — no floor
    # forcing a recommendation.
    try:
        _ALPHA = 0.05
        _HORIZON = 7
        _spot = float(state.get('spot', 11.0) or 11.0)
        _leaps_iv = float(state.get('iv_est', 0.45) or 0.45)
        if sd is not None:
            _cvar_price = float(sd.cvar_loss(_HORIZON, alpha=_ALPHA))
        else:
            _daily_vol = _leaps_iv / (252 ** 0.5)
            _cvar_frac = _daily_vol * (_HORIZON ** 0.5) * 2.06
            _cvar_price = _cvar_frac * _spot
        # Total portfolio tail loss (from delta + gamma)
        _tail_loss = abs(-total_delta * _cvar_price + 0.5 * total_gamma * _cvar_price ** 2)
        # How much do existing LEAPS offset?
        _leaps_T = 200.0 / 365.0
        _leaps_delta = bs_delta(_spot, _spot, _leaps_T, 0.045, _leaps_iv, 'P') * 100
        _leaps_gamma = bs_gamma(_spot, _spot, _leaps_T, 0.045, _leaps_iv) * 100
        _benefit_per = max(0,
            -_leaps_delta * _cvar_price
            + 0.5 * _leaps_gamma * _cvar_price ** 2
        )
        _hedged = tail_qty * _benefit_per
        _unhedged = max(0, _tail_loss - _hedged)
        # Penalty = probability-weighted unhedged exposure
        tail_hedge_penalty = -_ALPHA * _unhedged * 0.1  # 10% of expected unhedged loss
    except Exception:
        tail_hedge_penalty = 0.0

    # Pillar drift bonus (mildly bullish/bearish cyclical alignment)
    pillar_bonus = pillar_sum * 300.0

    # Cycle 178: annualized roll friction penalty. Shorter DTE positions
    # require more frequent rolls → higher cumulative friction. The
    # evaluator should penalize portfolios with high annualized friction
    # drag, naturally favoring longer DTE (fewer rolls/year, less friction).
    # User insight: "higher premium from longer DTE gives room — when it
    # moves hard the penalty is less."
    # Computation: for each short option, estimate annual rolls × spread cost.
    # Sum across portfolio → weekly friction drag → subtract from quality.
    _friction_penalty = 0.0
    try:
        _today_f = date.today()
        for _exp_s, _K, _right, _qty, _avg in _positions:
            if _qty >= 0:
                continue  # only short positions have roll friction
            _exp_d = datetime.strptime(_exp_s, '%Y-%m-%d').date()
            _dte = max(1, (_exp_d - _today_f).days)
            if _dte > 180:
                continue  # LEAPS don't roll frequently
            _annual_rolls = 365.0 / _dte
            # Friction per roll ≈ half spread × 100 × contracts.
            # Approximate spread at $0.03/share for liquid UNG options.
            _friction_per_roll = 0.03 * abs(_qty) * 100
            _annual_friction = _annual_rolls * _friction_per_roll
            _weekly_friction = _annual_friction / 52.0
            _friction_penalty -= _weekly_friction
    except Exception:
        _friction_penalty = 0.0

    # Cycle 183: margin efficiency penalty. User insight: "it eats away
    # buying power so if UNG moves down, we 'could have' sell options then."
    # Penalizes positions that consume margin disproportionately to premium
    # captured. Premium/margin < 5% means you're locking buying power for
    # pennies — buying power better reserved for post-move opportunities.
    # This is a proxy for multi-scenario lookahead: the opportunity cost
    # of margin consumption when the market might move in your favor.
    # Cycle 183b: margin efficiency as a continuous cost, no threshold.
    # Every dollar of margin consumed has an opportunity cost — it could
    # earn the risk-free rate (BOXX ~5% APR) or deploy at better prices
    # after a move. The penalty = margin consumed × opportunity rate,
    # offset by premium captured. Net: positions that capture more premium
    # than their opportunity cost are rewarded; positions that waste margin
    # are penalized. No hardcoded threshold.
    _margin_penalty = 0.0
    try:
        _OPPORTUNITY_RATE_WEEKLY = 0.05 / 52.0  # 5% APR → weekly
        for _exp_s, _K, _right, _qty, _avg in _positions:
            if _qty >= 0 or _right != 'P':
                continue
            _prem_per_share = _avg / 100 if _avg > 1 else _avg
            _margin_per_contract = max(0, _K * 100 - _prem_per_share * 100)
            _total_margin = abs(_qty) * _margin_per_contract
            # Opportunity cost of locking this margin for the position's life
            _exp_d = datetime.strptime(_exp_s, '%Y-%m-%d').date()
            _dte = max(1, (_exp_d - date.today()).days)
            _weeks = _dte / 7.0
            _opp_cost = _total_margin * _OPPORTUNITY_RATE_WEEKLY * _weeks
            _margin_penalty -= _opp_cost
    except Exception:
        _margin_penalty = 0.0

    # Cycle 185: read MCTS what-if cache from background refiner.
    # whatif_refiner.py runs Monte Carlo simulations continuously (AlphaGo
    # pondering). Its output: opportunity_value = expected premium after
    # spot move - current premium. Positive = value in waiting.
    _whatif_cache_value = 0.0
    try:
        import os as _os_wif
        _wif_path = _os_wif.path.join(_os_wif.path.dirname(_os_wif.path.abspath(__file__)), 'whatif_cache.json')
        if _os_wif.path.exists(_wif_path):
            _wif = json.loads(open(_wif_path).read())
            _cache_age = (datetime.now() - datetime.fromisoformat(_wif.get('timestamp', '2000-01-01'))).total_seconds()
            if _cache_age < 600:
                _whatif_cache_value = float(_wif.get('opportunity_value', 0))
    except Exception:
        pass

    # Cycle 184: inline what-if (fallback when cache not available). Simulate
    # Method: at spot ± daily_vol × sqrt(horizon), compute hypothetical
    # ATM put premium. Compare to current ATM premium. If post-move
    # premium is HIGHER (because spot dropped and puts are juicier),
    # that's opportunity cost of being fully deployed now.
    _whatif_value = 0.0
    try:
        _spot_wif = float(state.get('spot', 11.0) or 11.0)
        _iv_wif = float(state.get('iv_est', 0.45) or 0.45)
        _daily_vol_wif = _iv_wif / (252 ** 0.5)
        _horizon_wif = 7  # 1-week lookahead
        _move = _daily_vol_wif * (_horizon_wif ** 0.5) * _spot_wif
        # Scenario: UNG drops 1σ
        _down_spot = _spot_wif - _move
        _down_atm_prem = abs(bs_price(_down_spot, round(_down_spot * 2) / 2,
                                       30.0 / 365, 0.045, _iv_wif * 1.1, 'P'))
        # Current ATM premium
        _cur_atm_prem = abs(bs_price(_spot_wif, round(_spot_wif * 2) / 2,
                                      30.0 / 365, 0.045, _iv_wif, 'P'))
        # If post-drop premium > current: there's opportunity value in
        # having margin available. Scale by probability of the drop.
        _prem_uplift = max(0, _down_atm_prem - _cur_atm_prem)
        # How many new contracts could we sell with AVAILABLE margin?
        _total_used_margin = sum(
            max(0, _K * abs(_qty) * 100 - (_avg / 100 if _avg > 1 else _avg) * abs(_qty) * 100)
            for _exp_s, _K, _right, _qty, _avg in _positions
            if _qty < 0 and _right == 'P'
        )
        _capital = float(state.get('capital_base', 112000) or 112000)
        _free_margin = max(0, _capital - _total_used_margin)
        _contracts_deployable = int(_free_margin / max(1, _down_spot * 100))
        # Expected opportunity = P(drop) × contracts × premium uplift
        _p_drop = 0.32  # ~1σ probability
        _whatif_value = _p_drop * min(_contracts_deployable, 10) * _prem_uplift * 100
    except Exception:
        _whatif_value = 0.0

    total = (income_gap + dd_penalty + delta_gap + smoothness_bonus
             + tail_hedge_penalty + pillar_bonus + _friction_penalty
             + _margin_penalty + _whatif_value + _whatif_cache_value
             + _conc_penalty + _forward_bonus + _strangle_bonus
             + _gamma_load_penalty)

    return {
        'total': round(total, 1),
        'hard_dd_veto': _hard_dd_veto,
        'components': {
            'income_gap': round(income_gap, 1),
            'dd_penalty': round(dd_penalty, 1),
            'delta_gap': round(delta_gap, 1),
            'smoothness': round(smoothness_bonus, 1),
            'tail_hedge': round(tail_hedge_penalty, 1),
            'pillar_drift': round(pillar_bonus, 1),
            'friction': round(_friction_penalty, 1),
            'margin_eff': round(_margin_penalty, 1),
            'concentration': round(_conc_penalty, 1),
            'forward_cov': round(_forward_bonus, 1),
            'strangle': round(_strangle_bonus, 1),
            'gamma_load': round(_gamma_load_penalty, 1),
            'whatif': round(_whatif_value, 1),
        },
        'dd_diagnostics': {
            'cvar_drop': round(cvar_drop, 3),
            'capital': round(capital, 0),
            'total_delta': round(total_delta, 0),
            'total_gamma': round(total_gamma, 2),
            'horizon_days': HORIZON_D,
            'alpha': ALPHA,
            'theta_horizon': round(theta_horizon, 1),
            'theta_30d': round(theta_30d, 1),  # back-compat
            'delta_loss': round(delta_loss, 1),
            'gamma_convexity': round(gamma_convexity, 1),
            'theta_offset': round(theta_offset, 1),
            'tail_pnl_worst': round(tail_pnl_worst, 1),
            'tail_pnl': round(tail_pnl_worst, 1),  # back-compat alias
            'expected_tail_loss': round(expected_tail_loss, 1),
            'expected_dd_frac': round(expected_dd_frac, 4),
            'dd_frac': round(dd_frac, 4),  # raw (legacy display)
            'dd_threshold': -0.10,
            'over_threshold_$': round(dd_penalty, 1),
            # Real per-contract gamma for ATM puts at three representative
            # tenors (cycle 63). Cycle 57's hedge-math sub-row used a 2000-Γ
            # heuristic that was 50-200× too high; replaced with actual
            # Black-Scholes gamma × 100 multiplier.
            'atm_put_gamma_per_contract': _atm_put_gammas_for_diagnostic(
                spot,
                (state.get('scenario_dist').sigma_annual
                 if state.get('scenario_dist') is not None else 0.45)),
        },
    }


def _atm_put_gammas_for_diagnostic(spot, sigma):
    """Per-contract gamma (= bs_gamma × 100) for ATM puts at 30/90/365 DTE.

    Used by the dashboard hedge-math display to translate "gamma shortfall"
    into a realistic contract count. Cycle 57's heuristic was off by 50-200×
    because it treated gamma like delta with a 100× multiplier — but ATM
    gamma per share is small (0.05–0.40), not 1.0.
    """
    try:
        if spot is None or spot <= 0:
            return {'30d': 0, '90d': 0, '365d': 0}
        s = float(spot)
        sg = float(sigma) if sigma else 0.45
        return {
            '30d': round(bs_gamma(s, s, 30 / 365.0, 0.04, sg) * 100, 1),
            '90d': round(bs_gamma(s, s, 90 / 365.0, 0.04, sg) * 100, 1),
            '365d': round(bs_gamma(s, s, 365 / 365.0, 0.04, sg) * 100, 1),
        }
    except Exception:
        return {'30d': 0, '90d': 0, '365d': 0}


def compute_recommendations(spot, iv, expiry_groups, weekly_theta, smoothness, avg_weekly_theta, today, thesis_tilt=0.0):
    """Beam search optimizer: explore top-K paths to find best trade sequence.

    thesis_tilt: directional bias in [-1, +1]. +1 = strongly bullish (penalize
    upside-capping trades), -1 = strongly bearish, 0 = neutral.
    """

    # Build flat position list from expiry_groups
    flat_positions = []
    for exp_str, positions in expiry_groups.items():
        for p in positions:
            flat_positions.append((exp_str, p['strike'], p['right'], p['qty'], p.get('avg_cost', 0)))

    # Initial portfolio state
    initial_state = compute_portfolio_state(flat_positions, spot, iv, today)

    # Baseline correlated UNG exposure (short-put collateral) — user is OK with current sizing,
    # so this becomes the threshold against which we penalize *incremental* exposure.
    initial_state['baseline_put_coll'] = short_put_collateral(flat_positions)
    initial_state['capital_base'] = _margin_capital_usd
    # Directional thesis bias (set by dashboard slider, default neutral)
    initial_state['thesis_tilt'] = max(-1.0, min(1.0, float(thesis_tilt)))
    # Income target context (CENTRAL_PHILOSOPHY.md strategic objective).
    # Used by income_score in score_trade to reward gap-closing trades.
    # Cycle 152: recompute avg_weekly_theta from compute_portfolio_state's
    # own weekly_theta dict — SAME method that apply_trade_to_state uses
    # in its recompute. Previously we used the timeline-level avg
    # (compute_timeline's separate calendar-week loop), which gave a
    # different value (~$347 vs ~$306 on identical positions). Result:
    # every trade evaluation showed an artificial "income drop" of ~$40
    # because the initial-vs-post-trade methods disagreed. Made COVERED
    # CALL evaluation impossible (all candidates qΔ negative even when
    # adding theta + reducing delta-over-target).
    # Cycle 173: near 2-week average (user decision on cycle 160).
    try:
        _wt_vals = list(initial_state.get('weekly_theta', {}).values())
        _near = [v for v in _wt_vals[:2] if v > 0]
        initial_state['avg_weekly_theta'] = float(sum(_near) / len(_near)) if _near else 0.0
    except Exception:
        initial_state['avg_weekly_theta'] = float(avg_weekly_theta or 0.0)
    initial_state['target_weekly_income'] = 1500.0

    # Stand-aside discipline (empirically validated, see CENTRAL_PHILOSOPHY).
    # Base mode from z-score, then cyclical-aware augmentation:
    # shoulder + SURPLUS forces WAITING (worst of both), winter + SHORTAGE
    # promotes a level (best of both) per the interaction matrix.
    try:
        _z_mode = get_model_zscore()
    except Exception:
        _z_mode = 0.0
    if _z_mode < -0.5:
        _base_mode = 'WAITING'
    elif _z_mode <= 0.0:
        _base_mode = 'TRANSITION'
    else:
        _base_mode = 'ACTIVE'

    # Cyclical override (axis A × axis B interaction)
    _SHOULDER_MONTHS = {3, 4, 9, 10}   # spring + fall shoulder
    _WINTER_MONTHS   = {11, 12, 1, 2}  # heating season
    _supply_for_mode = _model_predictions.get('supply_regime', 'BALANCED')
    _month = today.month
    _final_mode = _base_mode
    _cyclical_override_reason = None
    if _supply_for_mode == 'SURPLUS' and _month in _SHOULDER_MONTHS:
        # Shoulder + SURPLUS = matrix "Very Bearish" — force WAITING regardless of z
        if _base_mode != 'WAITING':
            _final_mode = 'WAITING'
            _cyclical_override_reason = f'shoulder({_month})+SURPLUS forces WAITING'
    elif _supply_for_mode == 'SHORTAGE' and _month in _WINTER_MONTHS:
        # Winter + SHORTAGE = matrix "Very Bullish" — promote one level
        if _base_mode == 'WAITING':
            _final_mode = 'TRANSITION'
            _cyclical_override_reason = f'winter({_month})+SHORTAGE promotes WAITING→TRANSITION'
        elif _base_mode == 'TRANSITION':
            _final_mode = 'ACTIVE'
            _cyclical_override_reason = f'winter({_month})+SHORTAGE promotes TRANSITION→ACTIVE'
    # MA200 trend gate (cycle 137, user-approved): the user's principle —
    # "we always need to learn to throw UNG when NG is surging, only get
    # back to the game when it is back to historical mean or softer MA."
    # When UNG > 200-day MA, force WAITING regardless of z. Backtest evidence:
    # MA200 gate cut peak-DD from -91% to -81% in 5yr bear (cycle 132).
    _ma200_override_reason = None
    try:
        _tech_now = get_technicals_cached()
        if _tech_now and _tech_now.get('ma_200'):
            _ma200 = float(_tech_now['ma_200'])
            if _ma200 > 0 and spot > _ma200:
                if _final_mode != 'WAITING':
                    _ma200_override_reason = (
                        f'spot ${spot:.2f} > MA200 ${_ma200:.2f} '
                        f'(+{(spot/_ma200 - 1)*100:.1f}%) forces WAITING')
                    _final_mode = 'WAITING'
    except Exception as _ge:
        print(f"MA200 gate check failed: {_ge}")

    initial_state['deployment_mode'] = _final_mode
    initial_state['deployment_base_mode'] = _base_mode
    initial_state['deployment_cyclical_override'] = (
        _cyclical_override_reason or _ma200_override_reason)
    initial_state['deployment_ma200_gate'] = _ma200_override_reason is not None
    if _ma200_override_reason:
        print(f"Deployment MA200 gate: {_ma200_override_reason}")
    if _cyclical_override_reason:
        print(f"Deployment cyclical override: {_cyclical_override_reason}")

    # Build the SHARED scenario distribution (CENTRAL_PHILOSOPHY.md #1).
    # Every probability-aware scoring component consumes from this one object.
    try:
        _z = get_model_zscore()
    except Exception:
        _z = 0.0
    _outlook = _build_outlook(spot) or {}
    _combined_seasonal = 0.0  # default if scenario_dist build below fails
    try:
        # Cyclical foundation: pull seasonal drift + vol scale for current month
        # + supply/demand regime.
        _seasonal_vec = get_seasonal_drift_vector()
        _seasonal_today = _seasonal_vec.get(today.month, 0.0)
        _vol_vec = get_seasonal_vol_scale()
        _vol_today = _vol_vec.get(today.month, 1.0)
        # Axis B: supply/demand regime adjustment, additive on top of seasonal.
        _regime = _model_predictions.get('supply_regime', 'BALANCED')
        _regime_drift = SUPPLY_REGIME_DRIFT.get(_regime, 0.0)
        # Tech/Fund/YoY pillar modulators (continuous, additive on top of
        # categorical regime). Each capped at ±PILLAR_DRIFT_SCALE, total
        # contribution clamped at ±PILLAR_DRIFT_TOTAL_CAP.
        _tech_score = compute_tech_score()
        _fund_score = float(_model_predictions.get('fund_score', 0.0) or 0.0)
        _yoy_score = float(_model_predictions.get('yoy_score', 0.0) or 0.0)
        _pillar_raw = (_tech_score + _fund_score + _yoy_score) * PILLAR_DRIFT_SCALE
        _pillar_drift = max(-PILLAR_DRIFT_TOTAL_CAP,
                            min(PILLAR_DRIFT_TOTAL_CAP, _pillar_raw))
        _combined_seasonal = _seasonal_today + _regime_drift + _pillar_drift
        _fund_raw = float(_model_predictions.get('fund_score_raw', _fund_score) or _fund_score)
        _yoy_raw = float(_model_predictions.get('yoy_score_raw', _yoy_score) or _yoy_score)
        initial_state['pillar_scores'] = {
            'tech': round(_tech_score, 3),
            'fund': round(_fund_score, 3),
            'yoy':  round(_yoy_score, 3),
            'fund_raw': round(_fund_raw, 3),
            'yoy_raw': round(_yoy_raw, 3),
            'drift_per_day': round(_pillar_drift, 6),
        }
        initial_state['scenario_dist'] = ScenarioDistribution(
            spot=spot,
            z_score=_z,
            sigma_annual=max(0.20, float(iv)) if iv else 0.45,
            contango_per_day=-0.029 / 30,  # ≈ -3%/mo UNG contango
            outlook=_outlook,
            stress={'crash': STRESS_SCENARIOS.get('5d_crash', -0.117),
                    'spike': STRESS_SCENARIOS.get('5d_spike', 0.10)},
            seasonal_drift_per_day=_combined_seasonal,
            seasonal_vol_scale=_vol_today,
        )
        initial_state['supply_regime'] = _regime
        initial_state['seasonal_vol_scale'] = _vol_today
        # Precompute 30d 5%-CVaR drop ($/share) once; consumed by the hard
        # drawdown constraint in score_trade (CENTRAL_PHILOSOPHY hard risk
        # constraint: max -10% monthly DD).
        try:
            initial_state['cvar_30d_5pct_drop'] = float(
                initial_state['scenario_dist'].cvar_loss(30, alpha=0.05)
            )
        except Exception:
            initial_state['cvar_30d_5pct_drop'] = 0.0
    except Exception as _e:
        # Falls back to legacy 3-point list path; do not break recommendations.
        print(f"ScenarioDistribution build failed: {_e}")
        initial_state['scenario_dist'] = None
        initial_state['cvar_30d_5pct_drop'] = 0.0
    # Legacy 3-point list — kept for backward compat; will be removed when all
    # scorers consume from scenario_dist.
    initial_state['scenarios'] = _scenarios_for_ung(spot)

    # Growth/income bias — auto-computed from cyclical phase + price band +
    # premium ROI (CENTRAL_PHILOSOPHY.md, user-specified weights).
    # Higher value = more income-mode (harvest premium, accept caps);
    # lower value = more growth-mode (accumulate, preserve upside).
    try:
        _tech = get_technicals_cached() or {}
        _high_120 = _tech.get('high_120d', spot * 1.15)
        _low_120 = _tech.get('low_120d', spot * 0.85)
        if _high_120 > _low_120:
            _price_band = max(0.0, min(1.0, (spot - _low_120) / (_high_120 - _low_120)))
        else:
            _price_band = 0.5
        # Premium ROI proxy: ATM 30d put yield ≈ 0.4 × IV × sqrt(30/365), annualized.
        _iv_est = max(0.20, float(iv)) if iv else 0.45
        _atm_30d_pct = 0.4 * _iv_est * (30.0/365.0) ** 0.5
        _premium_roi_annualized = _atm_30d_pct * 12.0  # 12 months / year
        _roi_score = min(1.0, _premium_roi_annualized / 0.25)
        # Cyclical phase: bearish drift -> income bias; bullish drift -> growth bias.
        # combined_seasonal_per_day in approx [-0.005, +0.005]; multiply ~30 to map
        # to a ~[-0.15, +0.15] adjustment around the 0.5 midpoint, then clamp.
        _cyc_drift = initial_state.get('seasonal_vol_scale', 1.0)  # not used directly; we want drift
        _cyc_drift = _combined_seasonal if 'scenario_dist' in initial_state and initial_state['scenario_dist'] else 0.0
        _cyclical_phase = max(0.0, min(1.0, 0.5 - 30.0 * _cyc_drift))
        _income_bias = max(0.0, min(1.0,
            0.5 * _cyclical_phase + 0.3 * _price_band + 0.2 * _roi_score))
        initial_state['income_bias'] = _income_bias
        initial_state['growth_bias'] = 1.0 - _income_bias
        initial_state['bias_inputs'] = {
            'price_band': round(_price_band, 3),
            'premium_roi': round(_premium_roi_annualized, 3),
            'roi_score': round(_roi_score, 3),
            'cyclical_phase': round(_cyclical_phase, 3),
            'cyc_drift_per_day': round(_cyc_drift, 5),
        }
    except Exception as _bias_e:
        print(f"income_bias compute failed: {_bias_e}; defaulting to 0.5")
        initial_state['income_bias'] = 0.5
        initial_state['growth_bias'] = 0.5
        initial_state['bias_inputs'] = {}

    # Snapshot original quantities per source so partial-roll filtering/tracking
    # doesn't drift as state mutates across greedy iterations (fixes accounting
    # bug where 'remaining' was computed from already-reduced positions).
    original_qty_by_source = {}
    for pos in flat_positions:
        src_key = f"{pos[0]}-{pos[1]}-{pos[2]}"
        original_qty_by_source[src_key] = abs(pos[3])

    MIN_MARGINAL_SCORE = 3
    # Cycle 175: reduced 8 → 6. Each beam step adds ~1.5s; 6 steps saves
    # ~3s vs 8. The 7th-8th trades are marginal (qΔ < $30 typically) and
    # often cliff-dominated (cycle 164 outlier was trade #7).
    # Cycle 189b: raised 6→10 so beam covers all DTEs (was filling with tiny rolls)
    MAX_RECS = 10
    BEAM_WIDTH = 3  # explore top 3 paths at each step

    def _filter_candidates(candidates, state, used_sources, used_source_qty, used_targets, synthetic_positions):
        """Filter candidates based on already-used resources."""
        filtered = []
        for c in candidates:
            source_key = f"{c.get('source_exp', '')}-{c.get('source_strike', '')}-{c.get('source_right', '')}"
            if c['type'] in ('ROLL', 'CLOSE', 'LET EXPIRE', 'ASSIGNMENT', 'TAKE PROFIT') and source_key in used_sources:
                continue
            if source_key and source_key != '--' and c['type'] in ('ROLL', 'CLOSE', 'LET EXPIRE', 'ASSIGNMENT', 'TAKE PROFIT'):
                already_used = used_source_qty.get(source_key, 0)
                # Use the ORIGINAL snapshot, not the (already-mutated) state, to
                # avoid double-counting consumed contracts.
                orig_qty = original_qty_by_source.get(source_key, 0)
                remaining = orig_qty - already_used
                if remaining <= 0:
                    continue
                trade_qty = c.get('roll_qty', abs(c.get('qty', 1)))
                if trade_qty > remaining:
                    # Cap qty AND proportionally rescale all qty-scaled fields so
                    # the optimizer doesn't score a smaller trade at full-size economics.
                    scale = remaining / trade_qty if trade_qty > 0 else 0
                    c = dict(c)
                    c['roll_qty'] = remaining
                    for f in ('theta_change', 'delta_change', 'gamma_change',
                              'vega_change', 'new_extrinsic_total',
                              'old_ext_remaining', 'roll_net_total',
                              'source_friction', 'total_friction'):
                        if f in c and isinstance(c[f], (int, float)):
                            c[f] = c[f] * scale
                    if 'action' in c:
                        import re
                        c['action'] = re.sub(r'\b\d+x\b', f'{remaining}x', c['action'], count=1)
            if c['type'] in ('ROLL', 'CLOSE') and source_key in synthetic_positions:
                continue
            # Cycle 164: in income-mode, allow up to 2 OPENs per expiry at
            # DIFFERENT strikes. Standard mode keeps the 1-per-expiry block.
            # Income-mode rationale: extra strike per expiry captures more
            # premium without violating smoothness (still ≤2 strikes/expiry).
            # Strict same-(exp, strike) is always blocked.
            _t_exp = c.get('target_exp', '')
            _t_strike = c.get('target_strike')
            _t_typ = c['type']
            if _t_typ in ('ADD', 'OPEN', 'COVERED CALL', 'BUY PUT'):
                _income_mode_filter = (
                    state.get('avg_weekly_theta', 0) <
                    state.get('target_weekly_income', 1500) * 0.6
                )
                if _income_mode_filter and _t_typ in ('OPEN', 'COVERED CALL'):
                    # Cycle 179: removed per-expiry strike cap (was 2). User:
                    # "we should not limit strike per expiry." The evaluator's
                    # qΔ (dd_penalty, friction, smoothness) naturally limits
                    # how many strikes at one expiry are worth adding.
                    # Still block exact (exp, strike, type) reuse — can't sell
                    # the same contract twice in one beam path.
                    _used_key = f"{_t_exp}|{_t_strike}|{_t_typ}"
                    if _used_key in used_targets:
                        continue
                elif _t_exp in used_targets:
                    continue
            filtered.append(c)
        return filtered

    def _update_tracking(trade, state, used_sources, used_source_qty, used_targets, synthetic_positions):
        """Update tracking sets after applying a trade."""
        us = set(used_sources)
        usq = dict(used_source_qty)
        ut = set(used_targets)
        sp = set(synthetic_positions)

        if trade['type'] in ('ROLL', 'CLOSE', 'TAKE PROFIT', 'LET EXPIRE', 'ASSIGNMENT'):
            source_key = f"{trade.get('source_exp', '')}-{trade.get('source_strike', '')}-{trade.get('source_right', '')}"
            trade_qty = trade.get('roll_qty', abs(trade.get('qty', 1)))
            usq[source_key] = usq.get(source_key, 0) + trade_qty
            # Use the ORIGINAL snapshot, not the (already-mutated) state.
            orig_qty = original_qty_by_source.get(source_key, 0)
            if usq[source_key] >= orig_qty:
                us.add(source_key)
        if trade['type'] == 'ROLL':
            target_key = f"{trade.get('target_exp', '')}-{trade.get('target_strike', '')}-{trade.get('source_right', '')}"
            sp.add(target_key)
        elif trade['type'] in ('ADD', 'OPEN', 'COVERED CALL', 'BUY PUT'):
            target_key = f"{trade.get('target_exp', '')}-{trade.get('target_strike', '')}"
            sp.add(target_key)
            # Cycle 164: track per-strike target in income-mode to allow
            # multiple OPENs per expiry at different strikes. Standard
            # mode still gets the per-expiry block via the unstructured
            # exp entry below.
            _income_mode_track = (
                state.get('avg_weekly_theta', 0) <
                state.get('target_weekly_income', 1500) * 0.6
            )
            if _income_mode_track and trade['type'] in ('OPEN', 'COVERED CALL'):
                _key = f"{trade.get('target_exp', '')}|{trade.get('target_strike')}|{trade['type']}"
                ut.add(_key)
            else:
                ut.add(trade.get('target_exp', ''))
        elif trade['type'] == 'STRANGLE':
            ut.add(trade.get('target_exp', ''))

        return us, usq, ut, sp

    # ── True beam search ranked by portfolio-quality delta ──
    # Each path = (quality_delta, state, used_sources, used_source_qty,
    #              used_targets, synthetic_positions, best_trades,
    #              cached_candidates, cache_consumed_sources)
    # Beam ranks paths by `evaluate_portfolio_quality(state) - initial_quality`
    # rather than the sum of per-trade heuristic scores. This aligns the
    # optimizer with the strategic-objective evaluator (cycle 35) so beam
    # never picks a path that hurts portfolio quality even if per-trade
    # scores sum higher.
    # TOP_N_FOR_FULL_SCORE: cycle 66 reveal — cheap-score and quality_delta
    # diverged (BUY PUT scored low but qΔ ~+$14k). Cycle 67 raises the gate
    # to 20 (was 8) AND re-ranks the full-scored set by quality_delta
    # (instead of score) before the BEAM_WIDTH cut. The cheap-score still
    # acts as a fast prefilter for very obvious garbage; quality_delta
    # picks the actual winners from a wider funnel.
    TOP_N_FOR_FULL_SCORE = 20
    try:
        _initial_quality = evaluate_portfolio_quality(initial_state)['total']
    except Exception:
        _initial_quality = 0.0

    def _expand_path(path):
        """Return list of expanded paths (one per top-BEAM_WIDTH trade)."""
        (_q_delta, p_state, p_us, p_usq, p_ut, p_sp,
         p_best, p_cache, p_cache_us) = path
        if p_cache is None or p_cache_us != p_us:
            p_cache = generate_candidates(p_state, spot, iv, today)
            p_cache_us = set(p_us)
        filtered = _filter_candidates(p_cache, p_state, p_us, p_usq, p_ut, p_sp)
        if not filtered:
            return []
        cheap_scored = [(score_trade(c, p_state, skip_waterfall=True), c)
                        for c in filtered]
        cheap_scored.sort(key=lambda x: -x[0])
        top = cheap_scored[:TOP_N_FOR_FULL_SCORE]
        # Cycle 153: income-mode bypass. When avg_weekly_theta is below
        # 60% of target, the strategic objective is to AGGRESSIVELY pursue
        # income trades. Two-part fix:
        # (a) Guarantee that the top OPEN + top COVERED CALL candidates
        #     reach the full-scoring stage even if their cheap_score
        #     ranks them outside the top-N for other types (TP/ROLL
        #     usually dominate the score leaderboard).
        # (b) Lower the cheap-score gate threshold to -5 for OPEN/CC in
        #     income-mode so a low cheap_score doesn't block a positive
        #     qΔ trade. The evaluator's qΔ is the source of truth.
        _income_mode = (
            p_state.get('avg_weekly_theta', 0) <
            p_state.get('target_weekly_income', 1500) * 0.6
        )
        _INCOME_BYPASS_TYPES = {'OPEN', 'COVERED CALL', 'CLOSE', 'BUY BOXX'}
        # Cycle 154: lowered from -5 → -50. Cycle 153's -5 still blocked
        # OPENs because adding to a busy expiry incurs a waterfall penalty
        # that pushes full_score to -5.6 ~ -5.7. The cleanest fix is to
        # let qΔ alone gate in income-mode (where the strategic objective
        # mandates aggressive income pursuit). -50 is effectively "no
        # gate" for any reasonable trade, while still blocking obvious
        # garbage (score < -50 would be very ugly).
        _INCOME_BYPASS_THRESHOLD = -50.0
        if _income_mode:
            _top_sigs = {id(c) for _, c in top}
            # Cycle 155/174b: include top income candidates in `top`.
            # Cap OPENs at 15 (was unlimited → 46, too slow with 91
            # total candidates). Top-15 by cheap_score covers all
            # strike/qty variations that matter; the remainder are
            # lower-scored duplicates. Cap CCs at 5.
            # Cycle 188: include BOTH partial (1x) AND full-qty for each
            # (exp, strike). Cheap_score favors 1x (less concentration)
            # but qΔ favors full-qty (3-5x). Without both, the beam only
            # sees 1x and picks tiny positions. Group by (exp, strike),
            # include max-qty AND min-qty for each.
            _open_by_key = {}  # {(exp, strike): [(score, cand), ...]}
            for s, c in cheap_scored:
                if c.get('type') == 'OPEN' and id(c) not in _top_sigs:
                    _k = (c.get('target_exp', ''), c.get('target_strike', 0))
                    _open_by_key.setdefault(_k, []).append((s, c))
            _open_added = 0
            for _k, _variants in sorted(_open_by_key.items(), key=lambda x: -x[1][0][0]):
                if _open_added >= 20:
                    break
                # Add smallest AND largest qty variant
                _variants.sort(key=lambda x: x[1].get('add_qty', 1))
                for _v in [_variants[0], _variants[-1]]:  # min qty, max qty
                    s, c = _v
                    if id(c) not in _top_sigs:
                        top.append((s, c))
                        _top_sigs.add(id(c))
                        _open_added += 1
            _cc_added = 0
            for s, c in cheap_scored:
                if c.get('type') == 'COVERED CALL' and id(c) not in _top_sigs:
                    if _cc_added >= 8:
                        break
                    top.append((s, c))
                    _top_sigs.add(id(c))
                    _cc_added += 1
            # Cycle 199: include synthetic CLOSE+SELL candidates (early-assignment
            # locked-gain trades). These have a big qΔ bonus from _eval_candidate
            # but cheap_score doesn't see the locked gain, so they wouldn't make
            # it through the standard filter. Add ALL synthetics for full eval.
            for s, c in cheap_scored:
                if (c.get('type') == 'CLOSE' and c.get('shares_sold', 0) > 0
                        and id(c) not in _top_sigs):
                    top.append((s, c))
                    _top_sigs.add(id(c))
        full_scored = []
        bypass_candidates = []  # always empty after revert; preserved for downstream code
        for _, c in top:
            s = score_trade(c, p_state)
            _eff_gate = (_INCOME_BYPASS_THRESHOLD
                         if (_income_mode and c.get('type') in _INCOME_BYPASS_TYPES)
                         else MIN_MARGINAL_SCORE)
            if s >= _eff_gate:
                full_scored.append((s, c))
        evaluated = []
        if full_scored:
            _futures = [
                _QUALITY_POOL.submit(_eval_candidate, p_state, c, spot, iv,
                                     today, _initial_quality)
                for _, c in full_scored
            ]
            for (s, c), fut in zip(full_scored, _futures):
                _nqd, _ns, _nq, _ = fut.result()
                evaluated.append((_nqd, s, c, _ns, _nq))
        # Append quality-bypass admissions (they already have state+quality)
        for _nqd_byp, _ns_byp, _nq_byp, _s_byp, _c_byp in bypass_candidates:
            evaluated.append((_nqd_byp, _s_byp, _c_byp, _ns_byp, _nq_byp))
        # Cycle 170: cliff-guard for beam ranking. The avg_weekly_theta
        # metric has active-bucket discontinuities (cycle 164/166 finding:
        # one OPEN had components_delta {smoothness +200, income_gap +319}
        # for an actual $45 credit). Until the metric is replaced
        # (cycle 160 pending), guard against cliff-dominated trades
        # outranking real income trades by capping the smoothness +
        # income_gap contribution per trade. Cap at $150 each — anything
        # above is treated as cliff and excluded from ranking qΔ.
        # The RAW qΔ (for display/components_delta) is preserved; only
        # the sort key is adjusted.
        try:
            _prev_comps_for_rank = (
                evaluate_portfolio_quality(p_state).get('components', {}) or {}
            )
        except Exception:
            _prev_comps_for_rank = {}
        _CLIFF_CAP = 150.0
        def _rank_key(item):
            _qd, _, _, _, _nq = item
            try:
                _new_comps = _nq.get('components', {}) if isinstance(_nq, dict) else {}
                _sm = _new_comps.get('smoothness', 0) - _prev_comps_for_rank.get('smoothness', 0)
                _ig = _new_comps.get('income_gap', 0) - _prev_comps_for_rank.get('income_gap', 0)
                _excess = max(0, _sm - _CLIFF_CAP) + max(0, _ig - _CLIFF_CAP)
                return -(_qd - _excess)
            except Exception:
                return -_qd
        evaluated.sort(key=_rank_key)
        # Cycle 168 perf: compute p_state's components ONCE before the
        # BEAM_WIDTH loop. Cycle 166 ran evaluate_portfolio_quality(p_state)
        # PER candidate (~50/path × BEAM_WIDTH paths × 8 beam steps =
        # lots of redundant evals).
        # Cycle 169: also consume the new_q DICT (with components) directly
        # from `_eval_candidate` instead of re-evaluating new_state per
        # candidate. Combined with cycle 168, eliminates BOTH per-candidate
        # extra evals.
        try:
            _path_prev_components = (
                evaluate_portfolio_quality(p_state).get('components', {}) or {}
            )
        except Exception:
            _path_prev_components = {}
        out = []
        for new_q_delta, _s, trade, new_state, new_q in evaluated[:BEAM_WIDTH]:
            if new_state is None:
                # quality eval failed earlier; do the apply now as a fallback
                new_state = apply_trade_to_state(dict(p_state), trade, spot, iv, today)
                new_q = evaluate_portfolio_quality(new_state)
                new_q_delta = new_q.get('total', _initial_quality) - _initial_quality
            c_copy = dict(trade)
            c_copy['smoothness_impact'] = round(
                (new_state['smoothness'] - p_state['smoothness']) * 100, 1)
            new_us, new_usq, new_ut, new_sp = _update_tracking(
                c_copy, new_state, p_us, p_usq, p_ut, p_sp)
            new_cache = None if c_copy.get('type') in ('TAKE PROFIT', 'CLOSE') else p_cache
            # Per-trade $ quality contribution (cycle 42).
            c_copy['_dollar_value'] = round(new_q_delta - _q_delta, 1)
            # Cycle 166: per-trade components_delta.
            # Cycle 169: components from `new_q` dict (returned by
            # _eval_candidate), zero extra evals.
            try:
                _new_comps = (
                    new_q.get('components', {}) if isinstance(new_q, dict) else {}
                ) or {}
                c_copy['_components_delta'] = {
                    k: round(_new_comps.get(k, 0) - _path_prev_components.get(k, 0), 0)
                    for k in ('income_gap', 'dd_penalty', 'delta_gap',
                              'smoothness', 'tail_hedge', 'pillar_drift', 'friction', 'margin_eff', 'whatif', 'concentration', 'forward_cov', 'strangle', 'gamma_load')
                }
            except Exception:
                c_copy['_components_delta'] = {}
            out.append((
                new_q_delta, new_state, new_us, new_usq, new_ut, new_sp,
                list(p_best) + [c_copy], new_cache, p_cache_us,
            ))
        return out

    # Seed beam with the initial path (zero delta)
    beam = [(
        0.0, dict(initial_state), set(), {}, set(), set(),
        [], None, set(),
    )]
    for _ in range(MAX_RECS):
        all_expansions = []
        for path in beam:
            all_expansions.extend(_expand_path(path))
        if not all_expansions:
            break
        # Rank by quality_delta descending; include the un-expanded paths
        # so a path that "stays put" can still win if expanding hurts quality
        all_expansions.extend(beam)  # carry forward as "do nothing more" option
        all_expansions.sort(key=lambda p: -p[0])
        # Deduplicate paths that are objectively identical (same delta and
        # same best_trades length) to keep beam diverse
        seen_signatures = set()
        deduped = []
        for path in all_expansions:
            sig = (round(path[0], 2), len(path[6]))
            if sig in seen_signatures:
                continue
            seen_signatures.add(sig)
            deduped.append(path)
            if len(deduped) >= BEAM_WIDTH:
                break
        beam = deduped

    # Pick the highest-quality-delta terminal path
    best_path = max(beam, key=lambda p: p[0]) if beam else None
    if best_path is None:
        best_trades = []
        state = dict(initial_state)
        used_sources, used_source_qty = set(), {}
        used_targets, synthetic_positions = set(), set()
    else:
        (_, state, used_sources, used_source_qty, used_targets, synthetic_positions,
         best_trades, _, _) = best_path

    # Beam diagnostic (cycle 54): capture winner + runners-up so the operator
    # can see what alternatives the optimizer considered and on which dimension
    # each runner-up lost. Surfaces in /api/timeline → portfolio_metrics.
    _beam_diagnostic = []
    try:
        _initial_components = evaluate_portfolio_quality(initial_state).get('components', {})
        _ranked_beam = sorted(beam, key=lambda p: -p[0]) if beam else []
        _winner_components = None
        for _rank, _path in enumerate(_ranked_beam[:BEAM_WIDTH]):
            _q_delta = _path[0]
            _p_state = _path[1]
            _p_trades = _path[6]
            try:
                _p_eval = evaluate_portfolio_quality(_p_state)
            except Exception:
                _p_eval = {'total': 0.0, 'components': {}}
            _p_comp = _p_eval.get('components', {})
            if _rank == 0:
                _winner_components = _p_comp
            _comp_delta = {}
            for _k in ('income_gap', 'dd_penalty', 'delta_gap',
                       'smoothness', 'tail_hedge', 'pillar_drift', 'friction', 'margin_eff', 'whatif', 'concentration', 'forward_cov', 'strangle', 'gamma_load'):
                _comp_delta[_k] = round(_p_comp.get(_k, 0.0)
                                        - _initial_components.get(_k, 0.0), 1)
            _losing_dim = None
            _losing_gap = 0.0
            if _rank > 0 and _winner_components is not None:
                _gaps = {_k: (_winner_components.get(_k, 0.0) - _p_comp.get(_k, 0.0))
                         for _k in _comp_delta}
                if _gaps:
                    _losing_dim, _losing_gap = max(_gaps.items(), key=lambda kv: kv[1])
                    _losing_gap = round(_losing_gap, 1)
            _trades_summary = []
            for _t in _p_trades:
                _trades_summary.append({
                    'type': _t.get('type', ''),
                    'action': _t.get('action', ''),
                    'dollar_value': _t.get('_dollar_value', 0.0),
                    'target_strike': _t.get('target_strike') or _t.get('source_strike'),
                    'target_exp': _t.get('target_exp') or _t.get('source_exp'),
                })
            _beam_diagnostic.append({
                'rank': _rank,
                'is_winner': _rank == 0,
                'quality_delta': round(_q_delta, 1),
                'components_delta': _comp_delta,
                'trade_count': len(_p_trades),
                'trades': _trades_summary,
                'losing_dim': _losing_dim,
                'losing_gap': _losing_gap,
            })
    except Exception as _bde:
        print(f"[beam diagnostic] capture failed: {_bde}")
        _beam_diagnostic = []

    # Near-miss diagnostic (cycle 59): when the beam returns no trades
    # (e.g. rally pushed puts OTM, harvests below MIN_MARGINAL_SCORE),
    # capture the top candidates that were considered on the seed path
    # along with their score and reject reason. Tells the operator why
    # the optimizer is silent — wait, or override.
    _near_misses = []
    try:
        _winner_trade_count = len(_beam_diagnostic[0]['trades']) if _beam_diagnostic else 0
        if _winner_trade_count == 0:
            _seed_candidates = generate_candidates(initial_state, spot, iv, today)
            _scored = [(score_trade(c, initial_state, skip_waterfall=True), c)
                       for c in _seed_candidates]
            _scored.sort(key=lambda x: -x[0])
            _initial_q = evaluate_portfolio_quality(initial_state).get('total', 0.0)
            for _s, _c in _scored[:6]:
                if _s >= MIN_MARGINAL_SCORE:
                    _full = score_trade(_c, initial_state)
                    if _full < MIN_MARGINAL_SCORE:
                        _reject_reason = f"full score {_full:.1f} < min {MIN_MARGINAL_SCORE}"
                        _q_delta = None
                    else:
                        # Cycle 65: honest reject reason — actually compute
                        # quality_delta the beam would have seen, rather than
                        # the hand-wavy "did not improve quality" label from
                        # cycle 59. Apply trade to a copy of state and
                        # evaluate.
                        try:
                            _trial_state = apply_trade_to_state(
                                dict(initial_state), _c, spot, iv, today)
                            _trial_q = evaluate_portfolio_quality(_trial_state).get('total', _initial_q)
                            _q_delta = round(_trial_q - _initial_q, 1)
                            if _q_delta > 0:
                                _reject_reason = f"qΔ +${_q_delta} but outranked by stay-put"
                            elif _q_delta == 0:
                                _reject_reason = "qΔ 0 — no quality change"
                            else:
                                _reject_reason = f"qΔ -${abs(_q_delta)} — would worsen quality"
                        except Exception:
                            _reject_reason = "passed score but quality_delta eval failed"
                            _q_delta = None
                    _final_score = round(_full, 1)
                else:
                    _reject_reason = f"cheap score {_s:.1f} < min {MIN_MARGINAL_SCORE}"
                    _final_score = round(_s, 1)
                    _q_delta = None
                _near_misses.append({
                    'type': _c.get('type', ''),
                    'action': _c.get('action', ''),
                    'target_exp': _c.get('target_exp') or _c.get('source_exp'),
                    'target_strike': _c.get('target_strike') or _c.get('source_strike'),
                    'score': _final_score,
                    'quality_delta': _q_delta,
                    'reject_reason': _reject_reason,
                })
    except Exception as _nme:
        print(f"[near-misses] capture failed: {_nme}")
        _near_misses = []

    # Hidden-wins scan (cycle 66/68): scan all seed candidates for any
    # single trade whose standalone quality_delta exceeds what the beam's
    # entire chain achieved. Cycle 67 closed the score-vs-quality
    # divergence inside beam expansion, but MIN_MARGINAL_SCORE=3 and the
    # multi-step chain dynamics can still leave high-qΔ single moves
    # outside the chosen path. Surface them whenever they beat the chain
    # (or simply exceed +$2k if beam is empty), not just on no-action
    # cycles.
    _hidden_wins = []
    _seed_cands = None  # populated inside the hidden-wins try block
    try:
        _beam_chain_q = float(_beam_diagnostic[0]['quality_delta']) if _beam_diagnostic else 0.0
        _hw_threshold = max(2000.0, _beam_chain_q)
        _initial_q = evaluate_portfolio_quality(initial_state).get('total', 0.0)
        _seed_cands = generate_candidates(initial_state, spot, iv, today)
        # Identify trades already in the beam winner so we don't surface them as
        # "hidden": match by (type, target_exp, target_strike). Same key beam uses.
        _winner_keys = set()
        if _beam_diagnostic:
            for _t in _beam_diagnostic[0].get('trades', []):
                _winner_keys.add((_t.get('type', ''), _t.get('target_exp'),
                                  _t.get('target_strike')))
        # Cycle 71: type-restrict the hidden_wins scan. Cycle 69 showed
        # all empirical hidden-win outliers are in DD-helpful types
        # (BUY PUT / CLOSE / TAKE PROFIT / ASSIGNMENT / LET EXPIRE) — types
        # Cycle 91 adds OPEN and ADD: when the operator closes many short
        # puts, the beam prefers ROLL/ASSIGNMENT chains over income-
        # replenishing OPEN, so OPEN was invisible. Now surface OPEN/ADD
        # as override candidates too — operator decides when to reload
        # income vs let the existing positions decay.
        # whose dd_penalty benefit dominates the per-trade theta cost.
        # ROLL/OPEN/ADD/COVERED CALL/STRANGLE are already faithfully
        # served by the beam's score-then-qΔ flow. Restricting the
        # apply_trade_to_state + evaluate_portfolio_quality calls to
        # these 5 types cuts the scan from ~150-200 evals to ~30-60,
        # bringing /api/timeline back under the request budget.
        _HW_TYPES = {'BUY PUT', 'TAKE PROFIT', 'CLOSE', 'ASSIGNMENT',
                     'LET EXPIRE', 'OPEN', 'ADD'}
        # Pre-filter to DD-helpful types not already in beam winner.
        _hw_candidates = []
        for _c in _seed_cands:
            if _c.get('type') not in _HW_TYPES:
                continue
            _key = (_c.get('type', ''),
                    _c.get('target_exp') or _c.get('source_exp'),
                    _c.get('target_strike') or _c.get('source_strike'))
            if _key in _winner_keys:
                continue
            _hw_candidates.append(_c)
        # Cycle 72: parallel evaluation. Same pool as beam expansion.
        _wins = []
        if _hw_candidates:
            _futures = [
                _QUALITY_POOL.submit(_eval_candidate, initial_state, _c, spot,
                                     iv, today, _initial_q)
                for _c in _hw_candidates
            ]
            for fut in _futures:
                _dq, _ns, _nq, _c = fut.result()
                if _dq > _hw_threshold:
                    _wins.append((round(_dq, 1), _c))
        _wins.sort(key=lambda x: -x[0])
        for _dq, _c in _wins[:5]:
            # Cycle 69 instrumentation: also record the cheap-score and
            # full-score so the operator sees WHY the optimizer skipped this.
            try:
                _cheap = round(float(score_trade(_c, initial_state, skip_waterfall=True)), 1)
                _full_s = round(float(score_trade(_c, initial_state)), 1)
            except Exception:
                _cheap = None
                _full_s = None
            _hidden_wins.append({
                'type': _c.get('type', ''),
                'action': _c.get('action', ''),
                'target_exp': _c.get('target_exp') or _c.get('source_exp'),
                'target_strike': _c.get('target_strike') or _c.get('source_strike'),
                'quality_delta': _dq,
                'vs_beam_chain': round(_dq - _beam_chain_q, 1),
                'cheap_score': _cheap,
                'full_score': _full_s,
                'below_min_score': (_full_s is not None and _full_s < MIN_MARGINAL_SCORE),
            })
    except Exception as _hwe:
        print(f"[hidden-wins] scan failed: {_hwe}")
        _hidden_wins = []

    # Cycle 91: OPEN-rejected commentary. When the operator has closed
    # many short puts and Δ is well under target, they EXPECT to see
    # "open new puts" proposals. The beam may correctly decline because
    # adding short gamma worsens DD even though income is below target.
    # Surface this tradeoff explicitly so the operator understands the
    # silence is intentional, not a bug.
    _open_commentary = None
    try:
        _open_cands = [_c for _c in (_seed_cands or generate_candidates(initial_state, spot, iv, today))
                       if _c.get('type') == 'OPEN']
        if _open_cands:
            _initial_q_full = evaluate_portfolio_quality(initial_state)
            _initial_components = _initial_q_full.get('components', {})
            _best = None
            for _c in _open_cands[:10]:  # cap eval count
                try:
                    _ns = apply_trade_to_state(dict(initial_state), _c, spot, iv, today)
                    _q_after = evaluate_portfolio_quality(_ns)
                    _qd = _q_after.get('total', 0) - _initial_q_full.get('total', 0)
                    if _best is None or _qd > _best[0]:
                        _comp_after = _q_after.get('components', {})
                        _comp_deltas = {
                            k: round(_comp_after.get(k, 0) - _initial_components.get(k, 0), 0)
                            for k in ('income_gap', 'dd_penalty', 'delta_gap',
                                      'tail_hedge', 'pillar_drift', 'friction', 'margin_eff', 'whatif', 'concentration', 'forward_cov', 'strangle', 'gamma_load')
                        }
                        _best = (_qd, _c, _comp_deltas)
                except Exception:
                    continue
            if _best is not None:
                _qd, _c, _comp_deltas = _best
                _in_beam = any(t.get('type') == 'OPEN'
                               for t in (_beam_diagnostic[0].get('trades', []) if _beam_diagnostic else []))
                _open_commentary = {
                    'best_action': _c.get('action', ''),
                    'best_qdelta': round(_qd, 0),
                    'components_delta': _comp_deltas,
                    'open_candidate_count': len(_open_cands),
                    'in_beam': _in_beam,
                    '_candidate': _c,  # kept for rec promotion
                    '_qdelta_raw': _qd,
                }
    except Exception as _oce:
        print(f"[open commentary] failed: {_oce}")
        _open_commentary = None

    # Cycle 152: COVERED CALL commentary — same pattern as OPEN. Income
    # mode is at 23% of target and the beam mostly picks nothing because
    # all income trades sit below MIN_MARGINAL_SCORE. The 21-candidate CC
    # menu from cycle 151 is invisible unless we surface the best one for
    # promotion. Covered calls also reduce delta — doubly valuable now.
    _cc_commentary = None
    try:
        _cc_cands = [_c for _c in (_seed_cands or generate_candidates(initial_state, spot, iv, today))
                     if _c.get('type') == 'COVERED CALL']
        if _cc_cands:
            _initial_q_full = evaluate_portfolio_quality(initial_state)
            _initial_components = _initial_q_full.get('components', {})
            _best = None
            for _c in _cc_cands[:15]:
                try:
                    _ns = apply_trade_to_state(dict(initial_state), _c, spot, iv, today)
                    _q_after = evaluate_portfolio_quality(_ns)
                    _qd = _q_after.get('total', 0) - _initial_q_full.get('total', 0)
                    if _best is None or _qd > _best[0]:
                        _comp_after = _q_after.get('components', {})
                        _comp_deltas = {
                            k: round(_comp_after.get(k, 0) - _initial_components.get(k, 0), 0)
                            for k in ('income_gap', 'dd_penalty', 'delta_gap',
                                      'tail_hedge', 'pillar_drift', 'friction', 'margin_eff', 'whatif', 'concentration', 'forward_cov', 'strangle', 'gamma_load')
                        }
                        _best = (_qd, _c, _comp_deltas)
                except Exception:
                    continue
            if _best is not None:
                _qd, _c, _comp_deltas = _best
                _in_beam = any(t.get('type') == 'COVERED CALL'
                               for t in (_beam_diagnostic[0].get('trades', []) if _beam_diagnostic else []))
                _cc_commentary = {
                    'best_action': _c.get('action', ''),
                    'best_qdelta': round(_qd, 0),
                    'components_delta': _comp_deltas,
                    'cc_candidate_count': len(_cc_cands),
                    'in_beam': _in_beam,
                    '_candidate': _c,
                    '_qdelta_raw': _qd,
                }
    except Exception as _cce:
        print(f"[cc commentary] failed: {_cce}")
        _cc_commentary = None

    # Build recommendation list from chosen path
    recommendations = []
    running_state = dict(initial_state)
    for best_trade in best_trades:
        best_score = score_trade(best_trade, running_state)

        state_before_total_delta = running_state['total_delta']
        state_before_total_gamma = running_state['total_gamma']
        smooth_before = running_state['smoothness']

        running_state = apply_trade_to_state(running_state, best_trade, spot, iv, today)
        smooth_after = running_state['smoothness']
        best_trade['smoothness_impact'] = round((smooth_after - smooth_before) * 100, 1)

        # Classify urgency based on score
        if best_score > 20:
            urgency = 'high'
        elif best_score > 10:
            urgency = 'medium'
        else:
            urgency = 'low'
        # Cycle 157: income-mode OPEN/CC urgency should track qΔ, not the
        # heuristic score. Cycle 156's chained OPENs have qΔ +$67-$85 but
        # full_score ~-5 (waterfall penalty on busy expiries), so the
        # score-based classifier drops them to "low" — masking the
        # highest-conviction income trades when we're at 23% of target.
        try:
            _income_mode_urg = (
                initial_state.get('avg_weekly_theta', 0) <
                initial_state.get('target_weekly_income', 1500) * 0.6
            )
            if _income_mode_urg and best_trade['type'] in ('OPEN', 'COVERED CALL'):
                _qd = float(best_trade.get('_dollar_value', best_trade.get('quality_delta', 0)) or 0)
                if _qd >= 150:
                    urgency = 'high'
                elif _qd >= 30:
                    urgency = 'medium'
                # below 30: keep 'low'
        except Exception:
            pass

        # Compute delta at stress scenarios for display
        crash_price = spot * (1 + STRESS_SCENARIOS['5d_crash'])
        rally_price = spot * (1 + STRESS_SCENARIOS['5d_spike'])

        delta_at_crash_before = state_before_total_delta + state_before_total_gamma * (crash_price - spot)
        delta_at_crash_after = running_state['total_delta'] + running_state['total_gamma'] * (crash_price - spot)
        delta_at_rally_before = state_before_total_delta + state_before_total_gamma * (rally_price - spot)
        delta_at_rally_after = running_state['total_delta'] + running_state['total_gamma'] * (rally_price - spot)

        # Score breakdown for display
        breakdown = best_trade.get('_score_breakdown', {})
        p_itm_val = best_trade.get('_p_itm', None)

        rec_entry = {
            'type': best_trade['type'],
            'score': round(best_score, 1),
            'urgency': urgency,
            'action': best_trade['action'],
            'detail': best_trade['detail'],
            'why': best_trade['why'],
            'theta_impact': round(best_trade.get('theta_change', 0), 1),
            'delta_impact': round(best_trade.get('delta_change', 0), 0),
            'gamma_impact': round(best_trade.get('gamma_change', 0), 0),
            'vega_impact': round(best_trade.get('vega_change', 0), 0),
            'smoothness_impact': best_trade.get('smoothness_impact', 0),
            'qty': best_trade.get('roll_qty', best_trade.get('add_qty', best_trade.get('qty', 1))),
            'stress_crash_delta': round(delta_at_crash_after, 0),
            'stress_rally_delta': round(delta_at_rally_after, 0),
            'stress_crash_change': round(delta_at_crash_after - delta_at_crash_before, 0),
            'stress_rally_change': round(delta_at_rally_after - delta_at_rally_before, 0),
            'score_breakdown': breakdown,
            'dollar_value': best_trade.get('_dollar_value', 0),
            # Cycle 166: per-component decomposition of qΔ — operator can
            # spot when qΔ is dominated by a metric cliff (e.g., smoothness
            # +200 from a calendar-week active-bucket boundary) vs real
            # income contribution.
            'components_delta': best_trade.get('_components_delta', {}),
            # dte = source_dte for closes/assignments/expires/rolls (the
            # deadline that matters), target_dte for opens (the new tenor)
            'dte': best_trade.get('source_dte') if best_trade.get('type') in (
                'CLOSE', 'TAKE PROFIT', 'LET EXPIRE', 'ASSIGNMENT', 'ROLL'
            ) else best_trade.get('target_dte'),
            'source_exp': best_trade.get('source_exp'),
        }
        if p_itm_val is not None:
            rec_entry['p_itm'] = p_itm_val
        # Pass through liquidity data if available
        liq_data = best_trade.get('liquidity')
        if liq_data:
            rec_entry['liquidity'] = liq_data
        # Pass through Kelly data if computed
        kelly_data = best_trade.get('_kelly')
        if kelly_data:
            rec_entry['kelly'] = kelly_data
        recommendations.append(rec_entry)

    # Always surface near-money expiring positions as warnings
    # These should appear regardless of score so the user is aware of assignment risk
    all_candidates = generate_candidates(initial_state, spot, iv, today)
    for c in all_candidates:
        if c['type'] == 'ASSIGNMENT' and 'Risk:' in c.get('action', ''):
            # Near-money assignment warning — check if already covered
            already_shown = any(
                c.get('source_exp', '') in r.get('action', '') and
                str(c.get('source_strike', '')) in r.get('action', '')
                for r in recommendations
            )
            if not already_shown:
                recommendations.append({
                    'type': c['type'],
                    'score': round(score_trade(c, initial_state), 1),
                    'urgency': 'high',
                    'action': c['action'],
                    'detail': c.get('detail', ''),
                    'why': c.get('why', ''),
                    'theta_impact': round(c.get('theta_change', 0), 1),
                    'delta_impact': round(c.get('delta_change', 0), 0),
                    'gamma_impact': round(c.get('gamma_change', 0), 0),
                    'vega_impact': round(c.get('vega_change', 0), 0),
                    'dte': c.get('source_dte'),
                    'source_exp': c.get('source_exp'),
                    'qty': c.get('roll_qty', 1),
                })

    # Cycle 176: removed hardcoded tail-hedge static check. LEAPS BUY PUT
    # candidates are now generated in generate_candidates() and evaluated
    # by the beam like any other trade. The tail_hedge component in
    # evaluate_portfolio_quality (-$1000 per missing LEAPS) drives their
    # qΔ. State fields still set for display purposes.
    TAIL_HEDGE_MIN_DTE = 180
    TAIL_HEDGE_FLOOR = 2
    from datetime import datetime as _dt
    _leaps_qty = sum(
        abs(qty) for exp_str, strike, right, qty, _ in flat_positions
        if right == 'P' and qty > 0
        and (_dt.strptime(exp_str, '%Y-%m-%d').date() - today).days >= TAIL_HEDGE_MIN_DTE
    )
    initial_state['tail_hedge_qty'] = _leaps_qty
    initial_state['tail_hedge_floor'] = TAIL_HEDGE_FLOOR

    # If no trades scored high enough, add HOLD
    if not recommendations:
        recommendations.append({
            'type': 'HOLD',
            'score': 0,
            'urgency': 'low',
            'action': "Waterfall running -- no action needed",
            'detail': f"Theta: ${initial_state['total_theta']:.0f}/d | Smoothness: {initial_state['smoothness'] * 100:.0f}%",
            'why': "Marginal benefit of additional trades is below threshold.",
            'theta_impact': 0, 'delta_impact': 0, 'gamma_impact': 0, 'vega_impact': 0,
            'qty': 0,
        })

    # Dynamic delta target (capital_base comes from compute_target_delta via _margin_capital_usd)
    target_delta, regime, ratio = compute_target_delta(spot)
    capital_base = _margin_capital_usd
    current_exposure = initial_state['total_delta'] * spot
    target_exposure = target_delta * spot
    after_exposure = state['total_delta'] * spot

    # Stress test prices
    crash_price = spot * (1 + STRESS_SCENARIOS['5d_crash'])
    rally_price = spot * (1 + STRESS_SCENARIOS['5d_spike'])

    # Gamma regime (computed once per refresh via cache)
    try:
        gr = compute_gamma_regime(spot)
        gamma_regime_name = gr['regime']
        gamma_stance = gr['gamma_stance']
        gamma_reasoning = gr['reasoning']
    except Exception:
        gamma_regime_name = 'HOLD'
        gamma_stance = 'Unable to compute gamma regime'
        gamma_reasoning = []

    # Unified portfolio-quality before/after (CENTRAL_PHILOSOPHY P0 architecture).
    # Single $-normalized scalar tying income/DD/delta/smoothness/tail-hedge/
    # pillar dimensions together. Surfaces in metrics so the dashboard can show
    # how much applying the full recommendation set improves the portfolio.
    try:
        _quality_before = evaluate_portfolio_quality(initial_state)
        _quality_after = evaluate_portfolio_quality(state)
    except Exception as _qe:
        print(f"portfolio_quality compute failed: {_qe}")
        _quality_before = {'total': 0.0, 'components': {}}
        _quality_after = {'total': 0.0, 'components': {}}

    # Portfolio metrics before/after
    portfolio_metrics = {
        'target_delta': round(target_delta, 0),
        'target_exposure': round(target_exposure, 0),
        'current_exposure': round(current_exposure, 0),
        'after_exposure': round(after_exposure, 0),
        'capital_base': round(capital_base, 0),
        'current_leverage': round(current_exposure / capital_base, 2) if capital_base > 0 else 0,
        'target_leverage': round(target_exposure / capital_base, 2) if capital_base > 0 else 0,
        'after_leverage': round(after_exposure / capital_base, 2) if capital_base > 0 else 0,
        'regime': regime,
        'ma_ratio': round(ratio, 2),
        'gamma_regime': gamma_regime_name,
        'gamma_stance': gamma_stance,
        'gamma_reasoning': gamma_reasoning,
        'deployment_mode': initial_state.get('deployment_mode', 'ACTIVE'),
        'deployment_base_mode': initial_state.get('deployment_base_mode'),
        'deployment_cyclical_override': initial_state.get('deployment_cyclical_override'),
        'composite_z': round(_z_mode, 2),
        'supply_regime': initial_state.get('supply_regime', 'BALANCED'),
        'storage_z': _model_predictions.get('storage_z', 0.0),
        'pillar_scores': initial_state.get('pillar_scores', {}),
        'tail_hedge_qty': initial_state.get('tail_hedge_qty', 0),
        'tail_hedge_floor': initial_state.get('tail_hedge_floor', 2),
        'quality_before': _quality_before,
        'quality_after': _quality_after,
        'quality_delta': round(_quality_after['total'] - _quality_before['total'], 1),
        'beam_diagnostic': _beam_diagnostic,
        'near_misses': _near_misses,
        'hidden_wins': _hidden_wins,
        'open_commentary': _open_commentary,
        'cc_commentary': _cc_commentary,
        # Kelly utilization (cycle 64): surface the internal correlation
        # signal so the operator sees where the wheel sits vs the ¼-Kelly
        # 50% trigger and 95% hard cap from CENTRAL_PHILOSOPHY.
        'kelly_utilization': (lambda coll, cap: {
            'put_collateral': round(coll, 0),
            'capital': round(cap, 0),
            'utilization': round(coll / cap, 3) if cap > 0 else 0.0,
            'soft_trigger': 0.50,
            'hard_cap': 0.95,
            'over_kelly_mult': round(max(0.0, (coll / cap) - 0.50) * 4, 2)
                                if cap > 0 else 0.0,
        })(initial_state.get('baseline_put_coll', 0.0) or 0.0,
           float(capital_base) if capital_base else 1.0),
        # Cycle 143 — MARGIN-account correction: account is margin, not
        # cash-secured. Real put margin requirement is ~20% of strike + ITM
        # amount, NOT 100% of strike. The cycle-140 idle_cash formula
        # double-counted shares + collateral and showed $0 when in fact
        # there's substantial buying power. WS GraphQL doesn't expose
        # buyingPower cleanly (UNPROCESSABLE_ENTITY); user reports ~$51k
        # CAD ≈ $36k USD available. Switch to displaying portfolio asset
        # allocation honestly without claiming to know exact idle cash.
        'cash_park_suggestion': (lambda cap, put_coll_strike, sh, sp, other: {
            'capital_nlv': round(cap, 0),
            'account_type': 'margin',
            'put_collateral_notional': round(put_coll_strike, 0),
            'put_margin_est': round(put_coll_strike * 0.20, 0),  # ~20% margin req
            'share_value': round(sh * sp, 0) if sp > 0 else 0,
            'other_holdings': {k: {
                'qty': round(v.get('qty', 0), 2),
                'market_value': round(v.get('market_value', 0), 0),
            } for k, v in other.items()},
            'other_total': round(sum(v.get('market_value', 0) for v in other.values()), 0),
            'boxx_held': round((other.get('BOXX') or {}).get('market_value', 0), 0),
            # idle_cash is now an ESTIMATE: NLV minus shares/other/margin-used.
            # Won't match exact WS buying power but ballparks within ~$5k.
            'idle_cash_est': round(max(0, cap - sh * sp
                                       - sum(v.get('market_value', 0) for v in other.values())
                                       - put_coll_strike * 0.20), 0),
            'boxx_apr_pct': 5.0,
            'boxx_daily_yield_pct': round(5.0 / 252, 4),
            'note': ('Margin account: short put margin ≈ 20% of strike (NOT '
                     'cash-secured). Idle cash here is an ESTIMATE; check WS '
                     'directly for exact buying power.'),
        })(
            float(capital_base) if capital_base else 0.0,
            float(initial_state.get('baseline_put_coll', 0.0) or 0.0),
            int(SHARES) if SHARES else 0,
            float(spot) if spot else 0.0,
            globals().get('_OTHER_HOLDINGS') or {},
        ),
        'risk_by_expiry': (lambda eg, ed, et, ec: [
            {
                'expiry': _e,
                'gamma': round(eg.get(_e, 0), 0),
                'delta': round(ed.get(_e, 0), 0),
                'theta': round(et.get(_e, 0), 1),
                'contracts': int(ec.get(_e, 0)),
            }
            for _e in sorted(set(eg) | set(ed) | set(et),
                             key=lambda k: -abs(eg.get(k, 0)))
        ][:6])(
            initial_state.get('expiry_gamma', {}) or {},
            initial_state.get('expiry_delta', {}) or {},
            initial_state.get('expiry_theta', {}) or {},
            initial_state.get('expiry_contract_count', {}) or {},
        ),
        'predictions_updated_at': _model_predictions.get('updated_at'),
        'ic_weights': _model_predictions.get('ic_weights', {}),
        'income_bias': round(initial_state.get('income_bias', 0.5), 3),
        'growth_bias': round(initial_state.get('growth_bias', 0.5), 3),
        'bias_inputs': initial_state.get('bias_inputs', {}),
        'current': {
            'theta': round(initial_state['total_theta'], 1),
            'delta': round(initial_state['total_delta'], 0),
            'gamma': round(initial_state['total_gamma'], 0),
            'vega': round(initial_state['total_vega'], 0),
            'smoothness': round(initial_state['smoothness'] * 100, 0),
            # Cycle 158: avg_weekly_theta + % of $1500 target. Operator sees
            # current income progress alongside Greek state.
            'avg_weekly_theta': round(initial_state.get('avg_weekly_theta', 0), 1),
            'pct_of_target': round(
                initial_state.get('avg_weekly_theta', 0) /
                max(1, initial_state.get('target_weekly_income', 1500)) * 100, 1),
        },
        'after': {
            'theta': round(state['total_theta'], 1),
            'delta': round(state['total_delta'], 0),
            'gamma': round(state['total_gamma'], 0),
            'vega': round(state['total_vega'], 0),
            'smoothness': round(state['smoothness'] * 100, 0),
            # If operator executes all beam-selected trades, what does
            # their weekly income become? The strategic compass.
            'avg_weekly_theta': round(state.get('avg_weekly_theta', 0), 1),
            'pct_of_target': round(
                state.get('avg_weekly_theta', 0) /
                max(1, initial_state.get('target_weekly_income', 1500)) * 100, 1),
        },
        'stress': {
            'crash_5d': {
                'price': round(crash_price, 2),
                'delta_before': round(initial_state['total_delta'] + initial_state['total_gamma'] * (crash_price - spot), 0),
                'delta_after': round(state['total_delta'] + state['total_gamma'] * (crash_price - spot), 0),
            },
            'rally_5d': {
                'price': round(rally_price, 2),
                'delta_before': round(initial_state['total_delta'] + initial_state['total_gamma'] * (rally_price - spot), 0),
                'delta_after': round(state['total_delta'] + state['total_gamma'] * (rally_price - spot), 0),
            }
        }
    }

    # Daily snapshot for the progress tracker (cycle 52). Records once per
    # day; INSERT OR REPLACE so multiple visits intra-day refresh values.
    try:
        _qb_comp = _quality_before.get('components', {}) if isinstance(_quality_before, dict) else {}
        _ps = initial_state.get('pillar_scores', {}) or {}
        _progress_record({
            'date': today.isoformat() if hasattr(today, 'isoformat') else str(today),
            'avg_weekly_theta': float(avg_weekly_theta or 0.0),
            'quality_total': float(_quality_before.get('total', 0.0)) if isinstance(_quality_before, dict) else 0.0,
            'dd_penalty': float(_qb_comp.get('dd_penalty', 0.0)),
            'income_gap': float(_qb_comp.get('income_gap', 0.0)),
            'fund_score': float(_ps.get('fund') or 0.0),
            'yoy_score': float(_ps.get('yoy') or 0.0),
            'tech_score': float(_ps.get('tech') or 0.0),
            'supply_regime': str(initial_state.get('supply_regime', 'BALANCED')),
            'income_bias': float(initial_state.get('income_bias', 0.5)),
            'ung_price': float(spot),
            'shares': int(SHARES) if SHARES else 0,
            'options_count': len(initial_state.get('positions', []) or []),
        })
    except Exception as _se:
        print(f"[progress] snapshot at compute_recs end failed: {_se}")

    # Cycle 138 — Promote best OPEN candidate to the rec list when it has
    # positive qΔ but didn't make the beam (cheap_score/MIN_MARGINAL_SCORE
    # filtered it out). User wants the system to MOVE when conditions are
    # right; this surfaces the income-replenishment trade as an actionable
    # rec even if the beam's pre-filter missed it.
    # Cycle 145 — threshold lowered from $100 → $50 after cycle 144's
    # multi-strike menu showed qΔ rarely exceeds $100 in this account size
    # (live: best qΔ $74). $50 ≈ 3% of $1,500 weekly target — meaningful
    # income contribution, not noise. The beam's score gate still filters
    # bad strikes upstream; this just stops swallowing live qΔ wins
    # in the $50-100 band.
    # Cycle 146: changed `>` to `>=`. With qΔ values frequently landing
    # exactly at the threshold (e.g., live best_qdelta = 50.0 vs floor
    # of 50.0), strict `>` was a silent cliff that dropped recs at the
    # boundary. `>=` is the consistent inclusive intent.
    _OPEN_PROMOTE_QD_MIN = 50.0
    try:
        if (_open_commentary
                and not _open_commentary.get('in_beam')
                and _open_commentary.get('_qdelta_raw', 0) >= _OPEN_PROMOTE_QD_MIN):
            _c = _open_commentary.get('_candidate', {})
            if _c:
                _qd_disp = _open_commentary.get('best_qdelta', 0)
                recommendations.append({
                    'type': 'OPEN',
                    'urgency': 'medium',
                    'action': _c.get('action', ''),
                    'target_exp': _c.get('target_exp'),
                    'target_strike': _c.get('target_strike'),
                    'add_qty': _c.get('add_qty', 1),
                    'theta_impact': _c.get('theta_change', 0),
                    'delta_impact': _c.get('delta_change', 0),
                    'gamma_impact': _c.get('gamma_change', 0),
                    'vega_impact': _c.get('vega_change', 0),
                    # JS render code expects these fields — fill with safe defaults
                    # for the promoted rec (we don't have a beam score).
                    'score': float(_qd_disp),  # use qΔ in the score slot
                    'dollar_value': float(_qd_disp),
                    'smoothness_impact': 0,
                    'score_breakdown': {},
                    'liquidity': _c.get('liquidity', {}),
                    'detail': (_c.get('detail', '')
                               + f" | qΔ +${_qd_disp:.0f} (beam-bypass promotion)"),
                    'why': (f"Best OPEN candidate qΔ +${_qd_disp:.0f} but missed "
                            f"beam's score-based prefilter. dd_penalty wall is "
                            f"down; this is wheel-aligned income replenishment."),
                })
            # Strip the helper fields before serializing
            _open_commentary.pop('_candidate', None)
            _open_commentary.pop('_qdelta_raw', None)
    except Exception as _pe:
        print(f"[open promotion] failed: {_pe}")

    # Cycle 152: COVERED CALL beam-bypass promotion — mirrors OPEN above.
    # With income at 23% of target the beam tends to pick nothing because
    # all income trades score below MIN_MARGINAL_SCORE. Multi-strike CC
    # menu (cycle 151) needs the same promotion path the OPEN side has,
    # else the dashboard never shows the income lever even when there's
    # an obvious one available.
    _CC_PROMOTE_QD_MIN = 50.0
    try:
        if (_cc_commentary
                and not _cc_commentary.get('in_beam')
                and _cc_commentary.get('_qdelta_raw', 0) >= _CC_PROMOTE_QD_MIN):
            _c = _cc_commentary.get('_candidate', {})
            if _c:
                _qd_disp = _cc_commentary.get('best_qdelta', 0)
                recommendations.append({
                    'type': 'COVERED CALL',
                    'urgency': 'medium',
                    'action': _c.get('action', ''),
                    'target_exp': _c.get('target_exp'),
                    'target_strike': _c.get('target_strike'),
                    'add_qty': _c.get('add_qty', 1),
                    'theta_impact': _c.get('theta_change', 0),
                    'delta_impact': _c.get('delta_change', 0),
                    'gamma_impact': _c.get('gamma_change', 0),
                    'vega_impact': _c.get('vega_change', 0),
                    'score': float(_qd_disp),
                    'dollar_value': float(_qd_disp),
                    'smoothness_impact': 0,
                    'score_breakdown': {},
                    'liquidity': _c.get('liquidity', {}),
                    'detail': (_c.get('detail', '')
                               + f" | qΔ +${_qd_disp:.0f} (beam-bypass promotion)"),
                    'why': (f"Best COVERED CALL qΔ +${_qd_disp:.0f} but missed "
                            f"beam's score gate. Income at 23% of target + Δ over "
                            f"target — CC closes BOTH gaps."),
                })
            _cc_commentary.pop('_candidate', None)
            _cc_commentary.pop('_qdelta_raw', None)
    except Exception as _pe:
        print(f"[cc promotion] failed: {_pe}")

    # Cycle 165: always strip internal helper fields from commentaries
    # before serialization, even when promotion didn't fire. Previously
    # the pop only ran inside the promotion branch — when the best
    # candidate was sub-threshold (e.g., cc_commentary.best_qdelta=-2
    # vs $50 promote floor), `_candidate` (a full trade dict) leaked
    # into the API response. Cosmetic data-hygiene fix.
    for _cmt in (_open_commentary, _cc_commentary):
        if _cmt:
            _cmt.pop('_candidate', None)
            _cmt.pop('_qdelta_raw', None)

    # Cycle 147: stamp each rec with a stability count = # of the last 5
    # cycle windows in which the same signature appeared. Then append this
    # cycle's signature set to the rolling window. Note: counts BEFORE
    # appending — current cycle is the "now" frame, history is the past.
    try:
        _cur_sigs = {_rec_signature(_r) for _r in recommendations}
        _past = list(_RECS_HISTORY)  # snapshot, since deque is mutable
        _window_size = len(_past)
        for _r in recommendations:
            _sig = _rec_signature(_r)
            _count = sum(1 for _s in _past if _sig in _s)
            _r['stability_count'] = _count
            _r['stability_window'] = _window_size
        _RECS_HISTORY.append(_cur_sigs)
        _recs_history_persist()  # cycle 150: survive auto-reload
    except Exception as _se:
        print(f"[stability] annotate failed: {_se}")

    # Cycle 149: stability-gated urgency. Beam-picked recs that haven't
    # appeared in ≥3 of the last 5 cycles AND have qΔ < $200 get demoted
    # one urgency tier — so genuinely durable signal acts at full urgency,
    # but cycle-to-cycle flicker is presented as "watch this" instead of
    # "act now". Time-critical (LET EXPIRE / ASSIGNMENT) and structural
    # (TAIL HEDGE) recs are exempt — they earn their urgency from the
    # underlying state, not beam dynamics.
    _STABILITY_FLICKER_TYPES = {
        'OPEN', 'ROLL', 'TAKE PROFIT', 'BUY PUT', 'CLOSE',
        'COVERED CALL', 'BUY SHARES', 'SELL SHARES',
    }
    _STRONG_QD = 200.0   # qΔ above this — durability bypasses stability gate
    _STABLE_COUNT = 3    # cycles in window needed to be "durable"
    _URGENCY_DEMOTE = {'high': 'medium', 'medium': 'low', 'low': 'low'}
    # Cycle 157: skip demotion for OPEN / COVERED CALL in income-mode.
    # When we're below 60% of weekly income target, every actionable
    # income trade should surface at full urgency — the strategic
    # objective dominates the "wait for stability" prudence. The
    # cycle-156 multi-expiry OPEN chain had +$67-$85 qΔ recs being
    # demoted to "low" simply because they were new (stab=1/5), even
    # though they are the highest-conviction income trades produced in
    # days. Income-mode is exempt from flicker demotion for these types.
    _INCOME_MODE_EXEMPT = {'OPEN', 'COVERED CALL'}
    _income_mode_recs = False
    try:
        _income_mode_recs = (
            initial_state.get('avg_weekly_theta', 0) <
            initial_state.get('target_weekly_income', 1500) * 0.6
        )
    except Exception:
        _income_mode_recs = False
    try:
        for _r in recommendations:
            if _r.get('type') not in _STABILITY_FLICKER_TYPES:
                continue
            # Need ≥3 cycles of history to judge — else benefit of the doubt
            if _r.get('stability_window', 0) < 3:
                continue
            # Income-mode exemption for OPEN/CC
            if _income_mode_recs and _r.get('type') in _INCOME_MODE_EXEMPT:
                continue
            _qd = abs(float(_r.get('dollar_value') or _r.get('quality_delta') or 0))
            _sc = _r.get('stability_count', 0)
            if _sc < _STABLE_COUNT and _qd < _STRONG_QD:
                _orig = _r.get('urgency', 'low')
                _r['urgency'] = _URGENCY_DEMOTE.get(_orig, _orig)
                _r['urgency_original'] = _orig
                _r['urgency_demoted_for'] = (
                    f'flicker: {_sc}/{_r["stability_window"]} cycles, qΔ ${_qd:.0f}<{_STRONG_QD:.0f}')
    except Exception as _ue:
        print(f"[stability] urgency-gate failed: {_ue}")

    return recommendations, portfolio_metrics


def compute_timeline(price, iv, excluded_indices, thesis_tilt=0.0):
    """Compute expiration timeline and roll planner data."""
    r = 0.045
    today = date.today()
    spot = price

    # Group active options by expiration
    expiry_groups = {}
    for idx, (expiry_str, strike, right, qty, avg_cost) in enumerate(OPTIONS):
        if idx in excluded_indices:
            continue
        if expiry_str not in expiry_groups:
            expiry_groups[expiry_str] = []
        expiry = datetime.strptime(expiry_str, '%Y-%m-%d').date()
        dte = max((expiry - today).days, 0)
        T = dte / 365.0
        delta = bs_delta(price, strike, T, r, iv, right)
        theta = bs_theta(price, strike, T, r, iv, right)

        expiry_groups[expiry_str].append({
            'idx': idx,
            'strike': strike,
            'right': right,
            'qty': qty,
            'avg_cost': avg_cost,
            'delta': delta,
            'theta': theta,
            'dte': dte,
        })

    # Build expiration details
    expirations = []
    for exp_str in sorted(expiry_groups.keys()):
        positions = expiry_groups[exp_str]
        dte = positions[0]['dte']

        total_contracts = sum(abs(p['qty']) for p in positions)
        total_theta_at_risk = sum(p['qty'] * p['theta'] * 100 for p in positions)
        total_delta_change = 0.0  # net delta change when these positions expire/disappear

        contracts_detail = []
        assignment_detail = []
        position_recommendations = []

        for p in positions:
            label = f"{p['qty']}x ${p['strike']}{p['right']}"
            contracts_detail.append(label)

            # Current delta contribution of this position
            pos_delta = p['qty'] * p['delta'] * 100
            # When position expires, we LOSE this delta, so change = -pos_delta
            total_delta_change += -pos_delta

            # Compute moneyness for this position
            if p['right'] == 'P':
                itm = spot < p['strike']
                otm_pct = (p['strike'] - spot) / spot if not itm else 0
                itm_pct = (p['strike'] - spot) / spot if itm else 0
            else:  # Call
                itm = spot > p['strike']
                otm_pct = (p['strike'] - spot) / spot if not itm else 0
                itm_pct = (spot - p['strike']) / spot if itm else 0

            # Assignment impact
            if p['right'] == 'P' and spot < p['strike']:
                shares_impact = abs(p['qty']) * 100
                assignment_detail.append(
                    f"+{shares_impact} shares @ ${p['strike']:.1f}"
                )
            elif p['right'] == 'P' and spot >= p['strike']:
                assignment_detail.append(
                    f"${p['strike']:.1f}P expires OTM"
                )
            elif p['right'] == 'C' and spot > p['strike']:
                shares_impact = abs(p['qty']) * 100
                assignment_detail.append(
                    f"-{shares_impact} shares @ ${p['strike']:.1f}"
                )
            elif p['right'] == 'C' and spot <= p['strike']:
                assignment_detail.append(
                    f"${p['strike']:.1f}C expires OTM"
                )

            # Compute extrinsic % for this position
            T_cur = max(dte, 0) / 365.0
            per_share = abs(bs_price(spot, p['strike'], T_cur, 0.04, iv, p['right']))
            if p['right'] == 'P':
                pos_intrinsic = max(0, p['strike'] - spot)
            else:
                pos_intrinsic = max(0, spot - p['strike'])
            pos_extrinsic = max(0, per_share - pos_intrinsic)
            ext_pct = (pos_extrinsic / per_share * 100) if per_share > 0.01 else 0

            # Current theta per day for this position
            cur_theta = abs(bs_theta(spot, p['strike'], T_cur, 0.04, iv, p['right'])) * abs(p['qty']) * 100

            # Compare to theta from a fresh ATM position at 30 DTE and 45 DTE
            available = get_available_options()
            strike_pool = []
            for av_exp, av_chain in available.items():
                strike_pool.extend(av_chain['puts'] if p['right'] == 'P' else av_chain['calls'])
            strike_pool = sorted(set(strike_pool)) if strike_pool else [spot]
            atm_strike = find_nearest_strike(spot, strike_pool)
            fresh_30_theta = abs(bs_theta(spot, atm_strike, 30/365, 0.04, iv, p['right'])) * abs(p['qty']) * 100
            fresh_45_theta = abs(bs_theta(spot, atm_strike, 45/365, 0.04, iv, p['right'])) * abs(p['qty']) * 100

            # Opportunity cost: how much MORE theta/day from rolling
            opp_cost_30 = fresh_30_theta - cur_theta
            opp_cost_45 = fresh_45_theta - cur_theta

            # Smart recommendation per position
            if otm_pct > 0.10:
                position_recommendations.append(
                    f"${p['strike']}{p['right']}: EXPIRE WORTHLESS ({otm_pct*100:.0f}% OTM) - no action"
                )
            elif itm and ext_pct < 15:
                # Deep ITM with almost no extrinsic left — compare opportunity cost
                roll_strike = atm_strike
                if p['right'] == 'P':
                    position_recommendations.append(
                        f"${p['strike']}{p['right']}: THETA DRAINED ({ext_pct:.0f}% ext, ${cur_theta:.1f}/d) "
                        f"→ roll to ${roll_strike:.0f}P 30d: ${fresh_30_theta:.1f}/d (+${opp_cost_30:.1f}/d gain) "
                        f"or 45d: ${fresh_45_theta:.1f}/d (+${opp_cost_45:.1f}/d gain)"
                    )
                else:
                    position_recommendations.append(
                        f"${p['strike']}{p['right']}: THETA DRAINED ({ext_pct:.0f}% ext, ${cur_theta:.1f}/d) "
                        f"→ roll to ${roll_strike:.0f}C 30d: ${fresh_30_theta:.1f}/d or 45d: ${fresh_45_theta:.1f}/d"
                    )
            elif itm and ext_pct < 30:
                position_recommendations.append(
                    f"${p['strike']}{p['right']}: LOW THETA ({ext_pct:.0f}% ext, ${cur_theta:.1f}/d) "
                    f"— rolling to ATM 30d gains +${opp_cost_30:.1f}/d, 45d gains +${opp_cost_45:.1f}/d"
                )
            elif itm_pct > 0.05 and dte <= 7:
                if p['right'] == 'P':
                    position_recommendations.append(
                        f"${p['strike']}{p['right']}: ASSIGNMENT LIKELY (${cur_theta:.1f}/d left) "
                        f"— take {abs(p['qty'])*100} shares @ ${p['strike']} or roll to ATM 30d for ${fresh_30_theta:.1f}/d"
                    )
                else:
                    position_recommendations.append(
                        f"${p['strike']}{p['right']}: ASSIGNMENT LIKELY "
                        f"— {abs(p['qty'])*100} shares called @ ${p['strike']}"
                    )
            elif dte <= 5 and otm_pct <= 0.05:
                position_recommendations.append(
                    f"${p['strike']}{p['right']}: EXPIRING (${cur_theta:.1f}/d, {ext_pct:.0f}% ext) "
                    f"→ roll to 30d: ${fresh_30_theta:.1f}/d (+${opp_cost_30:.1f}/d) or 45d: ${fresh_45_theta:.1f}/d (+${opp_cost_45:.1f}/d)"
                )
            elif ext_pct > 50 and opp_cost_30 < cur_theta * 0.3:
                # Still good theta AND rolling wouldn't gain much
                position_recommendations.append(
                    f"${p['strike']}{p['right']}: HOLD ({ext_pct:.0f}% ext, ${cur_theta:.1f}/d — rolling only adds +${opp_cost_30:.1f}/d)"
                )
            elif ext_pct > 50 and opp_cost_30 >= cur_theta * 0.3:
                # Good extrinsic but rolling would be significantly better
                position_recommendations.append(
                    f"${p['strike']}{p['right']}: HOLD OK but rolling to 30d ATM gains +${opp_cost_30:.1f}/d ({ext_pct:.0f}% ext, ${cur_theta:.1f}/d now)"
                )
            else:
                position_recommendations.append(
                    f"${p['strike']}{p['right']}: MONITOR ({ext_pct:.0f}% extrinsic)"
                )

        # Urgency: only show ROLL NOW if near the money (within 5% of spot)
        any_near_money = False
        for p in positions:
            if p['right'] == 'P':
                otm_pct_check = (p['strike'] - spot) / spot if spot >= p['strike'] else 0
            else:
                otm_pct_check = (p['strike'] - spot) / spot if spot <= p['strike'] else 0
            if otm_pct_check <= 0.05:
                any_near_money = True
                break

        # Check if all positions are deeply OTM (>10%)
        all_deeply_otm = True
        for p in positions:
            if p['right'] == 'P':
                otm_pct_check = (p['strike'] - spot) / spot if spot >= p['strike'] else 0
            else:
                otm_pct_check = (p['strike'] - spot) / spot if spot <= p['strike'] else 0
            if otm_pct_check <= 0.10:
                all_deeply_otm = False
                break

        if all_deeply_otm and dte <= 7:
            action = "EXPIRE WORTHLESS"
            urgency = "ok"
        elif dte <= 5 and any_near_money:
            action = "ROLL NOW"
            urgency = "critical"
        elif dte <= 5 and not any_near_money:
            action = "EXPIRING - MONITOR"
            urgency = "caution"
        elif dte <= 7:
            action = "PLAN ROLL"
            urgency = "warning"
        elif dte <= 14:
            action = "WATCH"
            urgency = "caution"
        else:
            action = "MONITOR"
            urgency = "ok"

        # Suggest roll targets: find expiry at least 30 days out from today
        all_exp_dates = sorted(expiry_groups.keys())
        roll_target_exp = None
        for candidate_exp in all_exp_dates:
            candidate_date = datetime.strptime(candidate_exp, '%Y-%m-%d').date()
            candidate_dte = (candidate_date - today).days
            if candidate_dte >= 30 and candidate_exp != exp_str:
                roll_target_exp = candidate_exp
                break
        # If no existing expiry is 30+ DTE, compute a target date
        if roll_target_exp is None and dte < 30:
            target_date = today + timedelta(days=30)
            roll_target_exp = target_date.strftime('%Y-%m-%d') + " (or nearest)"

        roll_suggestions = []
        if dte <= 7 and any_near_money and roll_target_exp:
            for p in positions:
                # Only suggest rolling positions that are near the money
                if p['right'] == 'P':
                    p_otm = (p['strike'] - spot) / spot if spot >= p['strike'] else 0
                else:
                    p_otm = (p['strike'] - spot) / spot if spot <= p['strike'] else 0
                if p_otm <= 0.05:
                    roll_suggestions.append(
                        f"Roll ${p['strike']}{p['right']} -> {roll_target_exp} ${p['strike']}{p['right']}"
                    )

        expirations.append({
            'expiry': exp_str,
            'dte': dte,
            'total_contracts': total_contracts,
            'contracts_detail': contracts_detail,
            'total_theta_at_risk': round(total_theta_at_risk, 2),
            'total_delta_change': round(total_delta_change, 2),
            'assignment_detail': assignment_detail,
            'action': action,
            'urgency': urgency,
            'roll_suggestions': roll_suggestions,
            'position_recommendations': position_recommendations,
            'num_calls': sum(abs(p['qty']) for p in positions if p['right'] == 'C'),
            'num_puts': sum(abs(p['qty']) for p in positions if p['right'] == 'P'),
        })

    # Theta decay waterfall: how total theta changes after each expiry passes
    running_theta = sum(
        p['qty'] * p['theta'] * 100
        for positions in expiry_groups.values()
        for p in positions
    )
    theta_waterfall = [{'label': 'Today', 'theta': round(running_theta, 2)}]
    for exp in sorted(expiry_groups.keys()):
        exp_theta = sum(p['qty'] * p['theta'] * 100 for p in expiry_groups[exp])
        running_theta -= exp_theta
        theta_waterfall.append({
            'label': f'After {exp}',
            'theta': round(running_theta, 2),
        })

    # Rolling calendar grid: strikes x expiries with call/put breakdown
    all_strikes = sorted(set(o[1] for o in OPTIONS))
    all_expiries = sorted(expiry_groups.keys())
    calendar_grid = []
    for strike in all_strikes:
        row = []
        for exp in all_expiries:
            cell = {'calls': 0, 'puts': 0}
            if exp in expiry_groups:
                for p in expiry_groups[exp]:
                    if p['strike'] == strike:
                        if p['right'] == 'C':
                            cell['calls'] = p['qty']
                        else:
                            cell['puts'] = p['qty']
            row.append(cell)
        calendar_grid.append(row)

    # ── Weekly Theta Distribution & Smoothness Score ──
    # Find the last expiry date to limit our window
    all_expiry_dates = [datetime.strptime(e, '%Y-%m-%d').date() for e in expiry_groups.keys()]
    last_expiry = max(all_expiry_dates) if all_expiry_dates else today + timedelta(days=84)
    n_weeks = min(12, max(4, (last_expiry - today).days // 7 + 1))

    weekly_theta = {}
    for i in range(n_weeks):
        week_start = today + timedelta(days=(7 * i) - today.weekday())
        week_label = week_start.strftime('%b %d')

        week_theta = 0.0
        for exp_str, positions in expiry_groups.items():
            expiry_date = datetime.strptime(exp_str, '%Y-%m-%d').date()
            if expiry_date >= week_start:  # position is alive during this week
                for p in positions:
                    week_theta += abs(p['qty'] * p['theta'] * 100)
        # × 7 to convert daily theta sum → weekly $ income (see comment in
        # compute_portfolio_state.weekly_theta block above).
        weekly_theta[week_label] = round(week_theta * 7, 2)

    # Smoothness: first 4 weeks (cycle 188b). Matches compute_portfolio_state.
    thetas_list = list(weekly_theta.values())
    max_theta = max(thetas_list) if thetas_list else 0
    active_thetas = [t for t in thetas_list[:6] if t > 0]
    if len(active_thetas) > 2 and np.mean(active_thetas) > 0:
        smoothness = 1 - (float(np.std(active_thetas)) / float(np.mean(active_thetas)))
        smoothness = max(0.0, min(1.0, smoothness))
    else:
        smoothness = 0.0

    # Cycle 173: near 2-week average. User decision: "near 2 weeks average
    # because we all know this system will renew contracts." Replaces the
    # old all-active-weeks average that was dragged down by far-future
    # empty weeks, misrepresenting actual income capacity.
    _near_2 = [v for v in thetas_list[:2] if v > 0]
    avg_weekly_theta = float(sum(_near_2) / len(_near_2)) if _near_2 else 0.0

    # ── Generate Recommendations ──
    recommendations, portfolio_metrics = compute_recommendations(spot, iv, expiry_groups, weekly_theta, smoothness, avg_weekly_theta, today, thesis_tilt=thesis_tilt)

    # ── Nearest expiry info for daily status ──
    nearest_expiry = None
    nearest_dte = 999
    has_assignment_risk_today = False
    has_assignment_risk_tomorrow = False
    has_near_expiry_3_5 = False
    for exp in expirations:
        if exp['dte'] < nearest_dte:
            nearest_dte = exp['dte']
            nearest_expiry = exp
        if exp['dte'] <= 1:
            # Check assignment risk
            for detail in exp['assignment_detail']:
                if 'shares @' in detail and 'OTM' not in detail:
                    if exp['dte'] == 0:
                        has_assignment_risk_today = True
                    else:
                        has_assignment_risk_tomorrow = True
        if 3 <= exp['dte'] <= 5:
            has_near_expiry_3_5 = True

    daily_status = {
        'level': 'green',  # green, yellow, red
        'headline': 'ALL GOOD -- No Action Needed',
        'nearest_expiry': nearest_expiry['expiry'] if nearest_expiry else None,
        'nearest_dte': nearest_dte,
        'smoothness_pct': round(smoothness * 100),
    }

    if has_assignment_risk_today or has_assignment_risk_tomorrow:
        daily_status['level'] = 'red'
        if has_assignment_risk_today:
            daily_status['headline'] = 'ASSIGNMENT RISK -- Position Expiring TODAY'
        else:
            daily_status['headline'] = 'ASSIGNMENT RISK -- Position Expiring TOMORROW'
    elif has_near_expiry_3_5 or smoothness < 0.5:
        daily_status['level'] = 'yellow'
        reasons = []
        if has_near_expiry_3_5:
            reasons.append('expiry in 3-5 days needs rolling')
        if smoothness < 0.5:
            reasons.append(f'smoothness {round(smoothness*100)}% < 50%')
        daily_status['headline'] = 'ATTENTION NEEDED -- ' + ', '.join(reasons).capitalize()

    return {
        'expirations': expirations,
        'theta_waterfall': theta_waterfall,
        'calendar_grid': {
            'strikes': all_strikes,
            'expiries': all_expiries,
            'data': calendar_grid,
        },
        'today': today.strftime('%Y-%m-%d'),
        'weekly_theta_distribution': weekly_theta,
        'smoothness_score': round(smoothness, 4),
        'avg_weekly_theta': round(avg_weekly_theta, 2),
        'recommendations': recommendations,
        'portfolio_metrics': portfolio_metrics,
        'daily_status': daily_status,
    }


# ── HTML Page ────────────────────────────────────────────────────────────────

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>UNG Portfolio Visualizer</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
:root {
    --bg: #0d1117;
    --panel: #161b22;
    --border: #30363d;
    --text: #e6edf3;
    --text-dim: #8b949e;
    --green: #3fb950;
    --red: #f85149;
    --blue: #58a6ff;
    --purple: #bc8cff;
    --orange: #d29922;
    --cyan: #39d2c0;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.5;
}
.container { max-width: 1600px; margin: 0 auto; padding: 16px; }
h1 { font-size: 1.5rem; margin-bottom: 16px; }
h2 { font-size: 1.1rem; margin-bottom: 12px; color: var(--text-dim); font-weight: 500; }

/* Summary Cards */
.summary-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px;
    margin-bottom: 20px;
}
.card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
}
.card-label { font-size: 0.75rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.05em; }
.card-value { font-size: 1.5rem; font-weight: 600; margin-top: 4px; }
.card-value.positive { color: var(--green); }
.card-value.negative { color: var(--red); }

/* Controls */
.controls {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 20px;
    display: flex;
    gap: 32px;
    align-items: center;
    flex-wrap: wrap;
}
.control-group { display: flex; flex-direction: column; gap: 4px; }
.control-group label { font-size: 0.8rem; color: var(--text-dim); }
.control-group input[type="range"] { width: 200px; accent-color: var(--blue); }
.control-group .value-display {
    font-size: 1rem; font-weight: 600; color: var(--blue);
    min-width: 60px;
}

/* Charts */
.chart-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-bottom: 20px;
}
.chart-panel {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
}
.chart-panel.full-width { grid-column: 1 / -1; }

/* Position Table */
.table-container {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 20px;
    overflow-x: auto;
}
table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
}
th {
    text-align: left;
    padding: 8px 12px;
    border-bottom: 2px solid var(--border);
    color: var(--text-dim);
    font-weight: 600;
    cursor: pointer;
    user-select: none;
    white-space: nowrap;
}
th:hover { color: var(--blue); }
th.sorted-asc::after { content: ' \u25B2'; color: var(--blue); }
th.sorted-desc::after { content: ' \u25BC'; color: var(--blue); }
td {
    padding: 6px 12px;
    border-bottom: 1px solid var(--border);
    white-space: nowrap;
}
tr:hover { background: rgba(88,166,255,0.05); }
tr.excluded { opacity: 0.4; text-decoration: line-through; }
td.positive { color: var(--green); }
td.negative { color: var(--red); }
input[type="checkbox"] { accent-color: var(--blue); cursor: pointer; }

/* Heatmap */
.heatmap-container {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 20px;
}

/* What-If panel */
.whatif-summary {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 20px;
    display: none;
}
.whatif-summary.active { display: block; }
.whatif-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 12px;
}
.whatif-item {
    display: flex;
    justify-content: space-between;
    padding: 4px 0;
}
.whatif-item .label { color: var(--text-dim); }
.whatif-item .before { color: var(--text-dim); }
.whatif-item .arrow { color: var(--text-dim); margin: 0 6px; }

/* Loading */
.loading {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 40px;
    color: var(--text-dim);
}

/* Expiration Timeline */
.timeline-section {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 20px;
}
.timeline-section h2 {
    margin-bottom: 16px;
    color: var(--text);
    font-size: 1.2rem;
    font-weight: 600;
}
.expiry-cards {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(380px, 1fr));
    gap: 12px;
    margin-bottom: 20px;
}
.expiry-card {
    background: rgba(13, 17, 23, 0.6);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px 18px;
    border-left: 4px solid var(--green);
    min-height: auto;
}
.expiry-card.critical { border-left-color: var(--red); }
.expiry-card.warning { border-left-color: var(--orange); }
.expiry-card.caution { border-left-color: #e3b341; }
.expiry-card.ok { border-left-color: var(--green); }
.expiry-card h3 {
    font-size: 1.05rem;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
}
.expiry-card h3 .badge {
    font-size: 0.72rem;
    padding: 3px 10px;
    border-radius: 12px;
    font-weight: 600;
    text-transform: uppercase;
}
.badge-critical { background: var(--red); color: #fff; }
.badge-warning { background: var(--orange); color: #000; }
.badge-caution { background: #e3b341; color: #000; }
.badge-ok { background: rgba(63,185,80,0.2); color: var(--green); }
.expiry-card .detail-row {
    font-size: 0.9rem;
    color: var(--text-dim);
    padding: 4px 0;
    line-height: 1.7;
    word-wrap: break-word;
}
.expiry-card .detail-row strong { color: var(--text); }
.expiry-card .rec-item {
    display: block;
    padding: 3px 0 3px 12px;
    border-left: 2px solid var(--border);
    margin: 4px 0;
    font-size: 0.88rem;
    line-height: 1.6;
}
.expiry-card .rec-item.rec-expire { border-left-color: var(--green); }
.expiry-card .rec-item.rec-assign { border-left-color: var(--orange); }
.expiry-card .rec-item.rec-roll { border-left-color: var(--cyan); }
.expiry-card .rec-item.rec-monitor { border-left-color: var(--text-dim); }
.expiry-card .roll-suggestions {
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px solid var(--border);
    font-size: 0.85rem;
    color: var(--cyan);
    line-height: 1.7;
}

/* Theta waterfall */
.waterfall-container {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 20px;
}

/* Calendar Grid */
.calendar-grid-container {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 20px;
    overflow-x: auto;
}
.calendar-grid {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.8rem;
}
.calendar-grid th {
    padding: 6px 10px;
    border-bottom: 2px solid var(--border);
    color: var(--text-dim);
    font-weight: 600;
    cursor: default;
    white-space: nowrap;
}
.calendar-grid td {
    padding: 6px 10px;
    border-bottom: 1px solid var(--border);
    text-align: center;
    white-space: nowrap;
}
.calendar-grid td.has-calls { color: var(--cyan); }
.calendar-grid td.has-puts { color: var(--orange); }
.calendar-grid td.has-both { color: var(--purple); }
.calendar-grid td.empty-cell { color: var(--border); font-size: 0.7rem; }
.calendar-grid td .call-count { color: var(--cyan); }
.calendar-grid td .put-count { color: var(--orange); }

/* Delta Management Dashboard */
.delta-dashboard {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 20px;
}
.delta-dashboard h2 {
    font-size: 1.2rem;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 16px;
    border-bottom: 2px solid var(--blue);
    padding-bottom: 8px;
}
.delta-panels {
    display: grid;
    grid-template-columns: 60% 40%;
    gap: 16px;
    margin-bottom: 16px;
}
.metrics-grid {
    display: grid;
    grid-template-columns: 1fr;
    gap: 8px;
}
.metric-card {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 12px;
    background: rgba(13, 17, 23, 0.6);
    border-radius: 6px;
    border-left: 3px solid var(--border);
    font-size: 0.85rem;
}
.metric-card .metric-label { color: var(--text-dim); }
.metric-card .metric-value { font-weight: 600; font-family: monospace; font-size: 0.9rem; }
.metric-card.favorable { border-left-color: var(--green); }
.metric-card.favorable .metric-value { color: var(--green); }
.metric-card.risky { border-left-color: var(--red); }
.metric-card.risky .metric-value { color: var(--red); }
.metric-card.neutral { border-left-color: var(--blue); }
.metric-card.neutral .metric-value { color: var(--blue); }
.metric-card.warning { border-left-color: var(--orange); }
.metric-card.warning .metric-value { color: var(--orange); }
.metric-divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 4px 0;
}
.iv-section {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 20px;
}
.iv-section h2 {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 12px;
}
.iv-charts-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
}

/* Daily Status Banner */
.daily-banner {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 18px 24px;
    border-radius: 8px;
    margin-bottom: 20px;
    border: 1px solid var(--border);
}
.daily-banner.status-green {
    background: linear-gradient(135deg, rgba(63,185,80,0.12), rgba(63,185,80,0.04));
    border-color: rgba(63,185,80,0.4);
}
.daily-banner.status-yellow {
    background: linear-gradient(135deg, rgba(210,153,34,0.15), rgba(210,153,34,0.04));
    border-color: rgba(210,153,34,0.5);
}
.daily-banner.status-red {
    background: linear-gradient(135deg, rgba(248,81,73,0.15), rgba(248,81,73,0.04));
    border-color: rgba(248,81,73,0.5);
}
.status-icon { font-size: 2rem; flex-shrink: 0; }
.status-headline {
    font-size: 1.15rem;
    font-weight: 700;
    letter-spacing: 0.02em;
}
.status-green .status-headline { color: var(--green); }
.status-yellow .status-headline { color: var(--orange); }
.status-red .status-headline { color: var(--red); }
.status-detail {
    font-size: 0.88rem;
    color: var(--text-dim);
    margin-top: 4px;
}
.last-updated {
    font-size: 0.72rem;
    color: var(--text-dim);
    text-align: right;
    margin-bottom: 8px;
    opacity: 0.7;
}

/* Smoothness Gauge */
.smoothness-section {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 20px;
}
.smoothness-section h2 {
    font-size: 1.2rem;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 16px;
}
.smoothness-layout {
    display: grid;
    grid-template-columns: 280px 1fr;
    gap: 24px;
    align-items: start;
}
.gauge-container {
    display: flex;
    flex-direction: column;
    align-items: center;
}
.gauge-svg { width: 220px; height: 130px; }
.gauge-score {
    font-size: 2.2rem;
    font-weight: 700;
    margin-top: 4px;
}
.gauge-label {
    font-size: 0.8rem;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Recommendations Card */
.recommendations-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 20px;
}
.recommendations-card h2 {
    font-size: 1.2rem;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 12px;
}
/* Cycle 86: progress bar shown while /api/timeline computes (~15-25s) */
.rec-loading {
    padding: 14px 16px;
    background: rgba(88,166,255,0.08);
    border: 1px solid rgba(88,166,255,0.25);
    border-radius: 8px;
    margin-bottom: 12px;
}
.rec-loading-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.85rem;
    color: var(--text-dim);
    margin-bottom: 8px;
}
.rec-loading-bar {
    width: 100%;
    height: 6px;
    background: rgba(139,148,158,0.15);
    border-radius: 3px;
    overflow: hidden;
    position: relative;
}
.rec-loading-bar::after {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    height: 100%;
    width: 30%;
    background: linear-gradient(90deg,
        rgba(88,166,255,0.2),
        rgba(88,166,255,0.9),
        rgba(88,166,255,0.2));
    border-radius: 3px;
    animation: recLoadingSlide 1.4s ease-in-out infinite;
}
@keyframes recLoadingSlide {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(400%); }
}
.rec-loading-stage {
    color: var(--cyan, #58a6ff);
    font-weight: 600;
}
.rec-card {
    padding: 12px 14px;
    margin-bottom: 10px;
    background: rgba(13, 17, 23, 0.6);
    border-radius: 6px;
    border: 1px solid var(--border);
}
.rec-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 6px;
    flex-wrap: wrap;
}
.rec-rank {
    font-weight: 700;
    color: var(--text);
    font-size: 0.9rem;
}
.rec-type-badge {
    font-size: 0.7rem;
    padding: 2px 8px;
    border-radius: 10px;
    font-weight: 600;
}
.rec-urgency-badge {
    font-size: 0.65rem;
    padding: 2px 6px;
    border-radius: 10px;
    font-weight: 600;
    text-transform: uppercase;
}
.rec-urgency-badge.high { background: var(--red); color: #fff; }
.rec-urgency-badge.medium { background: var(--orange); color: #000; }
.rec-urgency-badge.low { background: rgba(139,148,158,0.2); color: #8b949e; }
.rec-stability-badge {
    font-size: 0.62rem;
    padding: 2px 5px;
    border-radius: 8px;
    font-family: monospace;
    font-weight: 600;
    border: 1px solid currentColor;
}
.rec-stability-badge.stable { color: var(--green); }
.rec-stability-badge.recent { color: var(--orange); }
.rec-stability-badge.flicker { color: var(--text-dim); }
.rec-theta {
    margin-left: auto;
    color: var(--green);
    font-weight: 600;
    font-size: 0.85rem;
}
.rec-action {
    font-weight: 600;
    color: var(--text);
    font-size: 0.9rem;
    margin-bottom: 4px;
    line-height: 1.5;
}
.rec-detail {
    font-size: 0.82rem;
    color: var(--text-dim);
    line-height: 1.6;
}
.rec-why {
    font-size: 0.78rem;
    color: var(--text-dim);
    opacity: 0.7;
    margin-top: 4px;
    line-height: 1.5;
}

/* Mobile Responsive */
@media (max-width: 768px) {
    .daily-banner { flex-direction: row; padding: 14px 16px; gap: 12px; }
    .status-headline { font-size: 1rem; }
    .status-detail { font-size: 0.8rem; }
    .smoothness-layout { grid-template-columns: 1fr; }
    .gauge-svg { width: 180px; height: 110px; }
    .rec-card { padding: 10px 12px; }
    .rec-action { font-size: 0.85rem; }
    .rec-detail { font-size: 0.8rem; }
    .rec-why { font-size: 0.75rem; }
    .container { padding: 8px; }
    h1 { font-size: 1.2rem; margin-bottom: 10px; }

    .summary-row {
        grid-template-columns: repeat(2, 1fr);
        gap: 8px;
    }
    .card { padding: 10px; }
    .card-label { font-size: 0.65rem; }
    .card-value { font-size: 1.1rem; }

    .controls {
        flex-direction: column;
        gap: 16px;
        align-items: stretch;
    }
    .control-group input[type="range"] { width: 100%; }

    .chart-grid {
        grid-template-columns: 1fr;
    }
    .chart-panel { padding: 10px; }

    .expiry-cards {
        grid-template-columns: 1fr;
        gap: 16px;
    }
    .expiry-card {
        width: 100%;
        padding: 14px 14px 18px;
    }
    .expiry-card h3 { font-size: 1rem; margin-bottom: 12px; }
    .expiry-card .detail-row { font-size: 0.88rem; padding: 5px 0; line-height: 1.8; }
    .expiry-card .rec-item { font-size: 0.85rem; padding: 4px 0 4px 10px; margin: 5px 0; line-height: 1.7; }
    .expiry-card .roll-suggestions { font-size: 0.83rem; line-height: 1.8; }

    .table-container {
        overflow-x: auto;
        -webkit-overflow-scrolling: touch;
    }

    .calendar-grid-container {
        overflow-x: auto;
        -webkit-overflow-scrolling: touch;
    }

    .whatif-row {
        grid-template-columns: 1fr;
    }
    .whatif-item { font-size: 0.8rem; }

    .timeline-section { padding: 10px; }
    .waterfall-container { padding: 10px; }
    .heatmap-container { padding: 10px; }

    .delta-panels {
        grid-template-columns: 1fr;
    }
    .delta-dashboard { padding: 10px; }
    .iv-charts-grid {
        grid-template-columns: 1fr;
    }
    .iv-section { padding: 10px; }
    #deltaPriceMap, #priceHistoryChart { min-height: 350px !important; }
    #ivTermChart, #ivSurfaceChart { min-height: 350px !important; }
}
</style>
</head>
<body>
<div class="container">
    <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
        <h1 style="margin:0;">UNG Portfolio Visualizer</h1>
        <button id="refreshBtn" onclick="refreshFromWS()"
            style="padding:6px 14px;border-radius:6px;border:1px solid #444;
                   background:#2d333b;color:#c9d1d9;cursor:pointer;font-size:0.85rem;
                   white-space:nowrap;">Refresh from WS</button>
    </div>
    <div class="last-updated" id="lastUpdated">Technicals: loading...</div>

    <!-- 1. Daily Status Banner -->
    <div id="dailyStatus" class="daily-banner status-green">
        <div class="status-icon">&#9989;</div>
        <div class="status-text">
            <div class="status-headline">Loading...</div>
            <div class="status-detail">Fetching portfolio data...</div>
        </div>
    </div>

    <!-- 2. Price Slider + Summary Cards -->
    <div class="controls">
        <div class="control-group">
            <label>UNG Price</label>
            <div style="display:flex;align-items:center;gap:8px;">
                <input type="range" id="priceSlider" min="7" max="16" step="0.01" value="10.74">
                <span class="value-display" id="priceDisplay">$10.74</span>
            </div>
        </div>
        <div class="control-group">
            <label>Implied Volatility</label>
            <div style="display:flex;align-items:center;gap:8px;">
                <input type="range" id="ivSlider" min="0.15" max="1.20" step="0.01" value="0.50">
                <span class="value-display" id="ivDisplay">50%</span>
            </div>
        </div>
        <div class="control-group">
            <label>Thesis Tilt <span style="font-size:0.7rem;color:var(--text-dim);">(−1 bear / 0 neutral / +1 bull)</span></label>
            <div style="display:flex;align-items:center;gap:8px;">
                <input type="range" id="thesisSlider" min="-1" max="1" step="0.1" value="0">
                <span class="value-display" id="thesisDisplay">0.0 (neutral)</span>
            </div>
        </div>
        <div class="control-group">
            <label style="color:var(--orange);">What-If Mode</label>
            <div style="font-size:0.8rem; color:var(--text-dim);">
                Uncheck positions in the table below to simulate closing them
            </div>
        </div>
    </div>

    <div class="summary-row" id="summaryCards">
        <div class="card"><div class="card-label">UNG Price</div><div class="card-value" id="cardPrice">--</div></div>
        <div class="card"><div class="card-label">Total P&L</div><div class="card-value" id="cardPnl">--</div></div>
        <div class="card"><div class="card-label">Share P&L</div><div class="card-value" id="cardSharePnl">--</div></div>
        <div class="card"><div class="card-label">Option P&L</div><div class="card-value" id="cardOptPnl">--</div></div>
        <div class="card"><div class="card-label">Net Delta</div><div class="card-value" id="cardDelta">--</div></div>
        <div class="card"><div class="card-label">Daily Theta</div><div class="card-value" id="cardTheta">--</div></div>
        <div class="card"><div class="card-label">Gamma</div><div class="card-value" id="cardGamma">--</div></div>
        <div class="card"><div class="card-label">Vega</div><div class="card-value" id="cardVega">--</div></div>
    </div>

    <!-- What-If Comparison -->
    <div class="whatif-summary" id="whatifSummary">
        <h2>What-If Comparison (before &rarr; after exclusions)</h2>
        <div class="whatif-row" id="whatifRow"></div>
    </div>

    <!-- 3. Expiration Timeline & Roll Planner -->
    <div class="timeline-section">
        <h2>Expiration Timeline &amp; Roll Planner</h2>
        <div id="timelineChart" style="height:400px;"></div>
    </div>
    <div id="expiryCards" class="expiry-cards"></div>

    <!-- 4. Theta Smoothness Chart + Gauge -->
    <div class="smoothness-section">
        <h2>Theta Smoothness Analysis</h2>
        <div class="smoothness-layout">
            <div class="gauge-container">
                <svg class="gauge-svg" viewBox="0 0 220 130">
                    <defs>
                        <linearGradient id="gaugeGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                            <stop offset="0%" style="stop-color:#f85149"/>
                            <stop offset="40%" style="stop-color:#d29922"/>
                            <stop offset="70%" style="stop-color:#3fb950"/>
                            <stop offset="100%" style="stop-color:#3fb950"/>
                        </linearGradient>
                    </defs>
                    <!-- Background arc -->
                    <path d="M 20 110 A 90 90 0 0 1 200 110" fill="none" stroke="#30363d" stroke-width="14" stroke-linecap="round"/>
                    <!-- Colored arc -->
                    <path d="M 20 110 A 90 90 0 0 1 200 110" fill="none" stroke="url(#gaugeGrad)" stroke-width="14" stroke-linecap="round" opacity="0.3"/>
                    <!-- Needle -->
                    <line id="gaugeNeedle" x1="110" y1="110" x2="110" y2="30" stroke="#e6edf3" stroke-width="2.5" stroke-linecap="round"/>
                    <circle cx="110" cy="110" r="5" fill="#e6edf3"/>
                    <!-- Labels -->
                    <text x="15" y="128" fill="#8b949e" font-size="10">0%</text>
                    <text x="190" y="128" fill="#8b949e" font-size="10">100%</text>
                </svg>
                <div class="gauge-score" id="smoothnessScore">--</div>
                <div class="gauge-label">Smoothness Score</div>
            </div>
            <div id="weeklyThetaChart" style="height:300px;"></div>
        </div>
    </div>

    <!-- Outlook (NG model bull/base/bear → UNG translation) -->
    <div class="recommendations-card" id="outlookCard" style="display:none;">
        <h2>Fundamentals Outlook (NG model)</h2>
        <div id="outlookContent" style="font-size:0.88rem;"></div>
    </div>

    <!-- 5. Recommendations -->
    <div class="recommendations-card" id="recommendationsCard">
        <h2>Recommendations</h2>
        <div id="recommendationsList"></div>
    </div>

    <!-- 6. Delta Management Dashboard -->
    <div class="delta-dashboard">
        <h2>Delta Management Dashboard</h2>
        <div class="delta-panels">
            <div>
                <div id="deltaPriceMap" style="height:500px;"></div>
            </div>
            <div class="metrics-grid" id="deltaMetrics">
                <div class="metric-card neutral"><span class="metric-label">Loading technicals...</span><span class="metric-value">--</span></div>
            </div>
        </div>
        <div id="priceHistoryChart" style="height:400px; margin-top:16px;"></div>
    </div>

    <!-- 7. Theta Decay Waterfall -->
    <div class="waterfall-container">
        <h2>Theta Decay Waterfall</h2>
        <div id="waterfallChart" style="height:420px;"></div>
    </div>

    <!-- 8. Price History + IV -->
    <div class="iv-section">
        <h2>Implied Volatility Analysis</h2>
        <div class="iv-charts-grid">
            <div id="ivTermChart" style="height:400px;"></div>
            <div id="ivSurfaceChart" style="height:400px;"></div>
        </div>
    </div>

    <!-- 9. Rolling Calendar Grid -->
    <div class="calendar-grid-container">
        <h2>Rolling Calendar Grid (Strike x Expiry)</h2>
        <table class="calendar-grid" id="calendarGrid"></table>
    </div>

    <!-- Charts -->
    <div class="chart-grid">
        <div class="chart-panel full-width">
            <h2>P&L Profile at Expiration</h2>
            <div id="pnlChart" style="height:400px;"></div>
        </div>
        <div class="chart-panel">
            <h2>Delta Exposure vs Price</h2>
            <div id="deltaChart" style="height:350px;"></div>
        </div>
        <div class="chart-panel">
            <h2>Daily Theta by Expiration</h2>
            <div id="thetaChart" style="height:350px;"></div>
        </div>
    </div>

    <!-- Heatmap -->
    <div class="heatmap-container">
        <h2>Liquidity Heat Map (Contracts by Strike x Expiry)</h2>
        <div id="heatmapChart" style="height:350px;"></div>
    </div>

    <!-- 10. Position Table -->
    <div class="table-container">
        <h2>Position Details</h2>
        <table id="positionTable">
            <thead>
                <tr>
                    <th data-col="active" style="cursor:default;">Active</th>
                    <th data-col="expiry">Expiry</th>
                    <th data-col="strike">Strike</th>
                    <th data-col="right">Type</th>
                    <th data-col="qty">Qty</th>
                    <th data-col="avg_cost">Avg Cost</th>
                    <th data-col="theo_price">Theo Value</th>
                    <th data-col="pnl">P&L</th>
                    <th data-col="delta">Delta</th>
                    <th data-col="gamma">Gamma</th>
                    <th data-col="theta">Theta</th>
                    <th data-col="vega">Vega</th>
                    <th data-col="dte">DTE</th>
                    <th data-col="extrinsic_pct">Ext%</th>
                </tr>
            </thead>
            <tbody id="positionBody"></tbody>
        </table>
    </div>
</div>

<script>
// State
let currentData = null;
let baselineData = null;  // data with no exclusions for what-if comparison
let technicalsData = null;  // cached technicals from yfinance
let excludedIndices = new Set();
let sortCol = null;
let sortAsc = true;
let debounceTimer = null;

const plotlyLayout = {
    paper_bgcolor: '#161b22',
    plot_bgcolor: '#161b22',
    font: { color: '#e6edf3', family: '-apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, Arial, sans-serif' },
    margin: { t: 30, r: 30, b: 90, l: 70 },
    xaxis: { gridcolor: '#30363d', zerolinecolor: '#30363d' },
    yaxis: { gridcolor: '#30363d', zerolinecolor: '#30363d' },
    hovermode: 'x unified',
};

function fmt(v) {
    if (v === undefined || v === null) return '--';
    return v >= 0 ? '$' + v.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})
                  : '-$' + Math.abs(v).toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2});
}
function fmtNum(v, dec=1) {
    if (v === undefined || v === null) return '--';
    return v.toLocaleString('en-US', {minimumFractionDigits: dec, maximumFractionDigits: dec});
}
function pnlClass(v) { return v >= 0 ? 'positive' : 'negative'; }

async function fetchData() {
    const price = document.getElementById('priceSlider').value;
    const iv = document.getElementById('ivSlider').value;
    const excluded = Array.from(excludedIndices).join(',');
    const url = `/api/data?price=${price}&iv=${iv}&excluded=${excluded}`;
    const resp = await fetch(url);
    return await resp.json();
}

async function fetchTimeline() {
    const price = document.getElementById('priceSlider').value;
    const iv = document.getElementById('ivSlider').value;
    const tEl = document.getElementById('thesisSlider');
    const tilt = tEl ? tEl.value : 0;
    const excluded = Array.from(excludedIndices).join(',');
    const url = `/api/timeline?price=${price}&iv=${iv}&excluded=${excluded}&thesis_tilt=${tilt}`;
    const resp = await fetch(url);
    return await resp.json();
}

async function fetchProgress(days = 30) {
    try {
        const r = await fetch(`/api/progress?days=${days}`);
        if (!r.ok) return null;
        return await r.json();
    } catch (e) {
        return null;
    }
}

async function fetchBaseline() {
    const price = document.getElementById('priceSlider').value;
    const iv = document.getElementById('ivSlider').value;
    const url = `/api/data?price=${price}&iv=${iv}&excluded=`;
    const resp = await fetch(url);
    return await resp.json();
}

function updateSummary(data) {
    const s = data.summary;
    document.getElementById('cardPrice').textContent = '$' + s.price.toFixed(2);
    document.getElementById('cardPrice').className = 'card-value';

    const pnlEl = document.getElementById('cardPnl');
    pnlEl.textContent = fmt(s.total_pnl);
    pnlEl.className = 'card-value ' + pnlClass(s.total_pnl);

    const spEl = document.getElementById('cardSharePnl');
    spEl.textContent = fmt(s.share_pnl);
    spEl.className = 'card-value ' + pnlClass(s.share_pnl);

    const opEl = document.getElementById('cardOptPnl');
    opEl.textContent = fmt(s.option_pnl);
    opEl.className = 'card-value ' + pnlClass(s.option_pnl);

    const dEl = document.getElementById('cardDelta');
    dEl.textContent = fmtNum(s.net_delta, 0);
    dEl.className = 'card-value ' + (s.net_delta >= 0 ? 'positive' : 'negative');

    const tEl = document.getElementById('cardTheta');
    tEl.textContent = fmt(s.total_theta);
    tEl.className = 'card-value ' + pnlClass(s.total_theta);

    const gEl = document.getElementById('cardGamma');
    gEl.textContent = fmtNum(s.total_gamma, 1);
    gEl.className = 'card-value';

    const vEl = document.getElementById('cardVega');
    vEl.textContent = fmtNum(s.total_vega, 1);
    vEl.className = 'card-value';
}

function updateOutlook(data) {
    const card = document.getElementById('outlookCard');
    const content = document.getElementById('outlookContent');
    const o = data.outlook;
    if (!o || !o.ng_current) {
        card.style.display = 'none';
        return;
    }
    card.style.display = '';
    const fmtPct = (target, base) => {
        if (target == null || !base) return '--';
        const pct = (target / base - 1) * 100;
        const sign = pct >= 0 ? '+' : '';
        return `${sign}${pct.toFixed(1)}%`;
    };
    const z = (o.z_score != null) ? o.z_score.toFixed(2) : '--';
    const zSign = (o.z_score != null && o.z_score >= 0) ? '+' : '';
    const zColor = (o.z_score == null) ? 'var(--text-dim)'
                  : o.z_score > 0.5 ? 'var(--green)'
                  : o.z_score < -0.5 ? 'var(--red)' : 'var(--text-dim)';
    const ngBaseColor = (o.ng_base && o.ng_current && o.ng_base > o.ng_current) ? 'var(--green)' : 'var(--red)';
    const ungBaseColor = (o.ung_base && o.ung_current && o.ung_base > o.ung_current) ? 'var(--green)' : 'var(--red)';
    content.innerHTML = `
        <div style="display:flex;gap:24px;flex-wrap:wrap;align-items:flex-start;">
          <div style="min-width:180px;">
            <div style="font-size:0.74rem;color:var(--text-dim);margin-bottom:3px;">Composite z-score</div>
            <div style="font-size:1.4rem;font-weight:600;color:${zColor};">${zSign}${z}</div>
            <div style="font-size:0.7rem;color:var(--text-dim);margin-top:2px;">
              ${o.z_score == null ? '' :
                o.z_score > 1.0 ? 'Strongly bullish' :
                o.z_score > 0.3 ? 'Mildly bullish' :
                o.z_score < -1.0 ? 'Strongly bearish' :
                o.z_score < -0.3 ? 'Mildly bearish' : 'Neutral'}
            </div>
          </div>
          <table style="font-size:0.85rem;border-collapse:collapse;">
            <tr>
              <th style="text-align:left;padding:4px 12px;color:var(--text-dim);">Scenario</th>
              <th style="text-align:right;padding:4px 12px;color:var(--text-dim);">NG ($/MMBtu)</th>
              <th style="text-align:right;padding:4px 12px;color:var(--text-dim);">% vs NG now</th>
              <th style="text-align:right;padding:4px 12px;color:var(--text-dim);">UNG target</th>
              <th style="text-align:right;padding:4px 12px;color:var(--text-dim);">% vs UNG now</th>
            </tr>
            <tr>
              <td style="padding:4px 12px;color:var(--text-dim);">Current</td>
              <td style="text-align:right;padding:4px 12px;">$${o.ng_current.toFixed(2)}</td>
              <td style="text-align:right;padding:4px 12px;color:var(--text-dim);">—</td>
              <td style="text-align:right;padding:4px 12px;">$${o.ung_current.toFixed(2)}</td>
              <td style="text-align:right;padding:4px 12px;color:var(--text-dim);">—</td>
            </tr>
            <tr style="background:rgba(63,185,80,0.06);">
              <td style="padding:4px 12px;color:var(--green);font-weight:600;">Bull (+1σ)</td>
              <td style="text-align:right;padding:4px 12px;">$${o.ng_bull ? o.ng_bull.toFixed(2) : '--'}</td>
              <td style="text-align:right;padding:4px 12px;color:var(--green);">${fmtPct(o.ng_bull, o.ng_current)}</td>
              <td style="text-align:right;padding:4px 12px;color:var(--green);font-weight:600;">$${o.ung_bull ? o.ung_bull.toFixed(2) : '--'}</td>
              <td style="text-align:right;padding:4px 12px;color:var(--green);">${fmtPct(o.ung_bull, o.ung_current)}</td>
            </tr>
            <tr>
              <td style="padding:4px 12px;">Base (model FV)</td>
              <td style="text-align:right;padding:4px 12px;">$${o.ng_base ? o.ng_base.toFixed(2) : '--'}</td>
              <td style="text-align:right;padding:4px 12px;color:${ngBaseColor};">${fmtPct(o.ng_base, o.ng_current)}</td>
              <td style="text-align:right;padding:4px 12px;">$${o.ung_base ? o.ung_base.toFixed(2) : '--'}</td>
              <td style="text-align:right;padding:4px 12px;color:${ungBaseColor};">${fmtPct(o.ung_base, o.ung_current)}</td>
            </tr>
            <tr style="background:rgba(248,81,73,0.06);">
              <td style="padding:4px 12px;color:var(--red);font-weight:600;">Bear (−1σ)</td>
              <td style="text-align:right;padding:4px 12px;">$${o.ng_bear ? o.ng_bear.toFixed(2) : '--'}</td>
              <td style="text-align:right;padding:4px 12px;color:var(--red);">${fmtPct(o.ng_bear, o.ng_current)}</td>
              <td style="text-align:right;padding:4px 12px;color:var(--red);font-weight:600;">$${o.ung_bear ? o.ung_bear.toFixed(2) : '--'}</td>
              <td style="text-align:right;padding:4px 12px;color:var(--red);">${fmtPct(o.ung_bear, o.ung_current)}</td>
            </tr>
          </table>
        </div>
        <div style="font-size:0.7rem;color:var(--text-dim);margin-top:8px;">
          UNG targets de-rated by ${o.contango_30d_pct.toFixed(1)}% 30-day contango.
          ${o.updated_at ? ' Updated: ' + o.updated_at.replace('T',' ') : ' (Click Refresh to compute)'}
        </div>
    `;
}

function updateWhatIf(data, baseline) {
    const panel = document.getElementById('whatifSummary');
    const row = document.getElementById('whatifRow');

    if (excludedIndices.size === 0) {
        panel.classList.remove('active');
        return;
    }
    panel.classList.add('active');

    const metrics = [
        { label: 'Total P&L', before: baseline.summary.total_pnl, after: data.summary.total_pnl, isCurrency: true },
        { label: 'Net Delta', before: baseline.summary.net_delta, after: data.summary.net_delta, isCurrency: false },
        { label: 'Daily Theta', before: baseline.summary.total_theta, after: data.summary.total_theta, isCurrency: true },
        { label: 'Gamma', before: baseline.summary.total_gamma, after: data.summary.total_gamma, isCurrency: false },
        { label: 'Options Active', before: baseline.summary.total_options, after: data.summary.total_options, isCurrency: false },
    ];

    row.innerHTML = metrics.map(m => {
        const bf = m.isCurrency ? fmt(m.before) : fmtNum(m.before, 1);
        const af = m.isCurrency ? fmt(m.after) : fmtNum(m.after, 1);
        const diff = m.after - m.before;
        const diffStr = m.isCurrency ? fmt(diff) : fmtNum(diff, 1);
        const diffColor = diff >= 0 ? 'var(--green)' : 'var(--red)';
        return `<div class="whatif-item">
            <span class="label">${m.label}</span>
            <span><span class="before">${bf}</span><span class="arrow">&rarr;</span><span style="color:${diffColor}">${af} (${diff >= 0 ? '+' : ''}${diffStr})</span></span>
        </div>`;
    }).join('');
}

function updatePnlChart(data) {
    const p = data.profile;
    const currentPrice = data.summary.price;

    // Find zero crossings for the total P&L line to color segments
    const traces = [
        {
            x: p.prices,
            y: p.pnl_total,
            name: 'Total P&L',
            type: 'scatter',
            mode: 'lines',
            line: { color: '#58a6ff', width: 2.5 },
        },
        {
            x: p.prices,
            y: p.pnl_shares,
            name: 'Share P&L',
            type: 'scatter',
            mode: 'lines',
            line: { color: '#3fb950', width: 1.5, dash: 'dot' },
        },
        {
            x: p.prices,
            y: p.pnl_options,
            name: 'Option P&L',
            type: 'scatter',
            mode: 'lines',
            line: { color: '#bc8cff', width: 1.5, dash: 'dot' },
        },
    ];

    // Green/red fill for total P&L
    const posY = p.pnl_total.map(v => v >= 0 ? v : 0);
    const negY = p.pnl_total.map(v => v < 0 ? v : 0);
    traces.unshift({
        x: p.prices, y: posY, type: 'scatter', mode: 'lines',
        fill: 'tozeroy', fillcolor: 'rgba(63,185,80,0.15)',
        line: { width: 0 }, showlegend: false, hoverinfo: 'skip',
    });
    traces.unshift({
        x: p.prices, y: negY, type: 'scatter', mode: 'lines',
        fill: 'tozeroy', fillcolor: 'rgba(248,81,73,0.15)',
        line: { width: 0 }, showlegend: false, hoverinfo: 'skip',
    });

    const layout = {
        ...plotlyLayout,
        shapes: [
            { type: 'line', x0: currentPrice, x1: currentPrice, y0: 0, y1: 1, yref: 'paper',
              line: { color: '#d29922', width: 2, dash: 'dash' } },
            { type: 'line', x0: 7, x1: 16, y0: 0, y1: 0, line: { color: '#30363d', width: 1 } },
        ],
        annotations: [
            { x: currentPrice, y: 1, yref: 'paper', text: 'Current: $' + currentPrice.toFixed(2),
              showarrow: false, font: { color: '#d29922', size: 11 }, yanchor: 'bottom' },
        ],
        xaxis: { ...plotlyLayout.xaxis, title: 'UNG Price' },
        yaxis: { ...plotlyLayout.yaxis, title: 'P&L ($)',
                 tickformat: '$,.0f' },
        legend: { x: 0.02, y: 0.98, bgcolor: 'rgba(0,0,0,0)' },
    };

    Plotly.react('pnlChart', traces, layout, { responsive: true, displayModeBar: false });
}

function updateDeltaChart(data) {
    const p = data.profile;
    const currentPrice = data.summary.price;

    // Compute dollar delta: for each price point, how many $/$ spot move
    // Dollar delta = share_delta × price (converts shares to $)
    const dollarDelta = p.delta_profile.map((d, i) => d * p.prices[i]);

    // Also compute $/1% move at each price for intuitive reading
    const dollarPerPct = p.delta_profile.map((d, i) => d * p.prices[i] * 0.01);

    const traces = [
        {
            x: p.prices,
            y: p.delta_profile,
            type: 'scatter',
            mode: 'lines',
            fill: 'tozeroy',
            fillcolor: 'rgba(88,166,255,0.08)',
            line: { color: '#58a6ff', width: 2 },
            name: 'Share Delta',
            yaxis: 'y',
        },
        {
            x: p.prices,
            y: dollarDelta,
            type: 'scatter',
            mode: 'lines',
            line: { color: '#f0883e', width: 2, dash: 'dot' },
            name: '$ Delta ($/$ move)',
            yaxis: 'y2',
        },
        {
            x: p.prices,
            y: dollarPerPct,
            type: 'scatter',
            mode: 'lines',
            line: { color: '#d2a8ff', width: 1.5 },
            name: '$/1% move',
            yaxis: 'y3',
            visible: 'legendonly',
        },
    ];

    // Annotate the asymmetry
    const curIdx = p.prices.findIndex(pr => pr >= currentPrice) || 0;
    const downIdx = Math.max(0, curIdx - 5);
    const upIdx = Math.min(p.prices.length - 1, curIdx + 5);

    const layout = {
        ...plotlyLayout,
        shapes: [
            { type: 'line', x0: currentPrice, x1: currentPrice, y0: 0, y1: 1, yref: 'paper',
              line: { color: '#d29922', width: 1.5, dash: 'dash' } },
        ],
        annotations: [
            { x: p.prices[downIdx], y: dollarDelta[downIdx], text: '$' + Math.round(dollarDelta[downIdx]).toLocaleString() + '/$ ↓',
              showarrow: true, arrowcolor: '#f85149', font: { color: '#f85149', size: 10 }, ax: -30, ay: -20 },
            { x: currentPrice, y: dollarDelta[curIdx], text: '$' + Math.round(dollarDelta[curIdx]).toLocaleString() + '/$ now',
              showarrow: true, arrowcolor: '#d29922', font: { color: '#d29922', size: 10 }, ax: 0, ay: -25 },
            { x: p.prices[upIdx], y: dollarDelta[upIdx], text: '$' + Math.round(dollarDelta[upIdx]).toLocaleString() + '/$ ↑',
              showarrow: true, arrowcolor: '#3fb950', font: { color: '#3fb950', size: 10 }, ax: 30, ay: -20 },
        ],
        xaxis: { ...plotlyLayout.xaxis, title: 'UNG Price' },
        yaxis: { ...plotlyLayout.yaxis, title: 'Share Delta', side: 'left',
                 titlefont: { color: '#58a6ff' }, tickfont: { color: '#58a6ff' } },
        yaxis2: { title: '$ Delta', overlaying: 'y', side: 'right',
                  titlefont: { color: '#f0883e' }, tickfont: { color: '#f0883e' },
                  tickformat: '$,.0f', showgrid: false },
        yaxis3: { overlaying: 'y', side: 'right', visible: false },
        legend: { x: 0.02, y: 0.98, bgcolor: 'rgba(0,0,0,0)', font: { size: 10 } },
        showlegend: true,
    };

    Plotly.react('deltaChart', traces, layout, { responsive: true, displayModeBar: false });
}

function updateThetaChart(data) {
    const tl = data.theta_timeline;

    const traces = [{
        x: tl.map(t => t.expiry),
        y: tl.map(t => t.daily_theta),
        type: 'bar',
        marker: {
            color: tl.map(t => t.daily_theta >= 0 ? '#3fb950' : '#f85149'),
        },
        text: tl.map(t => t.contracts + ' contracts'),
        hovertemplate: '%{x}<br>Daily Theta: $%{y:.2f}<br>%{text}<extra></extra>',
    }];

    const layout = {
        ...plotlyLayout,
        xaxis: { ...plotlyLayout.xaxis, title: 'Expiration', type: 'category' },
        yaxis: { ...plotlyLayout.yaxis, title: 'Daily Theta ($)', tickformat: '$,.2f' },
        showlegend: false,
    };

    Plotly.react('thetaChart', traces, layout, { responsive: true, displayModeBar: false });
}

function updateHeatmap(data) {
    const hm = data.heatmap;

    const traces = [{
        z: hm.data,
        x: hm.expiries,
        y: hm.strikes.map(s => '$' + s.toFixed(1)),
        type: 'heatmap',
        colorscale: [
            [0, '#161b22'],
            [0.2, '#1a3a2a'],
            [0.5, '#2ea043'],
            [0.75, '#56d364'],
            [1, '#7ee787'],
        ],
        showscale: true,
        colorbar: { title: 'Contracts', titlefont: { color: '#8b949e' }, tickfont: { color: '#8b949e' } },
        hovertemplate: 'Strike: %{y}<br>Expiry: %{x}<br>Contracts: %{z}<extra></extra>',
        text: hm.data.map(row => row.map(v => v > 0 ? String(v) : '')),
        texttemplate: '%{text}',
        textfont: { color: '#e6edf3', size: 11 },
    }];

    const layout = {
        ...plotlyLayout,
        xaxis: { ...plotlyLayout.xaxis, type: 'category', title: 'Expiration' },
        yaxis: { ...plotlyLayout.yaxis, title: 'Strike', autorange: true },
        margin: { ...plotlyLayout.margin, l: 60 },
    };

    Plotly.react('heatmapChart', traces, layout, { responsive: true, displayModeBar: false });
}

function updateTimelineChart(tl) {
    const exps = tl.expirations;
    if (exps.length === 0) return;

    const colors = exps.map(e => {
        if (e.dte <= 3) return '#f85149';
        if (e.dte <= 7) return '#d29922';
        if (e.dte <= 14) return '#e3b341';
        return '#3fb950';
    });

    const hoverText = exps.map(e => {
        return `<b>${e.expiry}</b><br>` +
               `DTE: ${e.dte}<br>` +
               `Contracts: ${e.total_contracts} (${e.num_calls}C / ${e.num_puts}P)<br>` +
               `Detail: ${e.contracts_detail.join(', ')}<br>` +
               `Delta impact: ${e.total_delta_change.toFixed(1)}<br>` +
               `Theta at risk: $${e.total_theta_at_risk.toFixed(2)}/day<br>` +
               `Action: ${e.action}`;
    });

    const traces = [{
        x: exps.map(e => e.expiry),
        y: exps.map(e => e.total_contracts),
        type: 'bar',
        marker: { color: colors },
        text: exps.map(e => e.total_contracts + ''),
        textposition: 'outside',
        textfont: { color: '#e6edf3', size: 11 },
        hovertext: hoverText,
        hoverinfo: 'text',
        name: 'Contracts',
    }];

    // Add TODAY marker - find position between category bars
    const todayStr = tl.today;
    const todayDate = new Date(todayStr + 'T12:00:00');
    const expiryDates = tl.expirations.map(e => new Date(e.expiry + 'T12:00:00'));

    // Find TODAY position as fractional index between expiry categories
    let todayPos = -0.5; // before first bar
    for (let i = 0; i < expiryDates.length; i++) {
        if (todayDate < expiryDates[i]) {
            if (i === 0) {
                todayPos = -0.5;
            } else {
                // Interpolate between bar i-1 and bar i
                const daysBetween = (expiryDates[i] - expiryDates[i-1]) / 86400000;
                const daysFromPrev = (todayDate - expiryDates[i-1]) / 86400000;
                todayPos = (i - 1) + daysFromPrev / daysBetween;
            }
            break;
        }
        todayPos = i + 0.5; // after last bar
    }

    const shapes = [{
        type: 'line', x0: todayPos, x1: todayPos, y0: 0, y1: 1, yref: 'paper',
        line: { color: '#58a6ff', width: 2, dash: 'dash' },
    }];
    const annotations = [{
        x: todayPos, y: 1, yref: 'paper',
        text: `TODAY (${todayStr})`,
        showarrow: false, font: { color: '#58a6ff', size: 11, weight: 'bold' },
        yanchor: 'bottom',
    }];

    const layout = {
        ...plotlyLayout,
        shapes: shapes,
        annotations: annotations,
        margin: { ...plotlyLayout.margin, b: 100 },
        xaxis: {
            ...plotlyLayout.xaxis,
            type: 'category',
            title: { text: 'Expiration Date', standoff: 20 },
            tickangle: -45,
            tickfont: { size: 11, color: '#e6edf3' },
        },
        yaxis: { ...plotlyLayout.yaxis, title: 'Contracts Expiring' },
        showlegend: false,
        bargap: 0.3,
    };

    Plotly.react('timelineChart', traces, layout, { responsive: true, displayModeBar: false });
}

function updateExpiryCards(tl) {
    const container = document.getElementById('expiryCards');
    const exps = tl.expirations;

    container.innerHTML = exps.map(e => {
        const badgeClass = 'badge-' + e.urgency;
        let actionLabel = e.action;
        let actionIcon = '';
        if (e.urgency === 'critical') actionIcon = '\u26a0\ufe0f';
        else if (e.urgency === 'warning') actionIcon = '\u23f0';

        // Format date nicely
        const d = new Date(e.expiry + 'T12:00:00');
        const dateStr = d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });

        return `<div class="expiry-card ${e.urgency}">
            <h3>${dateStr} (${e.dte} DTE) ${actionIcon} <span class="badge ${badgeClass}">${actionLabel}</span></h3>
            <div class="detail-row"><strong>Contracts:</strong> ${e.contracts_detail.join(', ')}</div>
            <div class="detail-row"><strong>Theta at risk:</strong> $${e.total_theta_at_risk.toFixed(1)}/day</div>
            <div class="detail-row"><strong>Delta change:</strong> ${e.total_delta_change > 0 ? '+' : ''}${e.total_delta_change.toFixed(0)} shares</div>
            <div class="detail-row"><strong>Assignment:</strong> ${e.assignment_detail.join(', ')}</div>
        </div>`;
    }).join('');
}

function updateWaterfallChart(tl) {
    const wf = tl.theta_waterfall;
    if (wf.length === 0) return;

    const colors = wf.map((w, i) => {
        if (i === 0) return '#58a6ff';
        return w.theta >= 0 ? '#3fb950' : '#f85149';
    });

    const traces = [{
        x: wf.map(w => w.label),
        y: wf.map(w => w.theta),
        type: 'bar',
        marker: { color: colors },
        text: wf.map(w => '$' + w.theta.toFixed(2)),
        textposition: 'outside',
        textfont: { color: '#e6edf3', size: 11 },
        hovertemplate: '%{x}<br>Daily Theta: $%{y:.2f}<extra></extra>',
    }];

    // Add annotation arrows showing the change between bars
    const annotations = [];
    for (let i = 1; i < wf.length; i++) {
        const diff = wf[i].theta - wf[i-1].theta;
        if (Math.abs(diff) > 0.01) {
            annotations.push({
                x: wf[i].label,
                y: Math.max(wf[i].theta, wf[i-1].theta) + 3,
                text: (diff >= 0 ? '+' : '') + '$' + diff.toFixed(2),
                showarrow: false,
                font: { color: diff >= 0 ? '#f85149' : '#3fb950', size: 9 },
            });
        }
    }

    const layout = {
        ...plotlyLayout,
        annotations: annotations,
        margin: { ...plotlyLayout.margin, b: 100 },
        xaxis: {
            ...plotlyLayout.xaxis,
            type: 'category',
            tickangle: -45,
            tickfont: { size: 10, color: '#e6edf3' },
        },
        yaxis: { ...plotlyLayout.yaxis, title: 'Daily Theta ($/day)', tickformat: '$,.2f' },
        showlegend: false,
        bargap: 0.25,
    };

    Plotly.react('waterfallChart', traces, layout, { responsive: true, displayModeBar: false });
}

function updateCalendarGrid(tl) {
    const grid = tl.calendar_grid;
    const table = document.getElementById('calendarGrid');

    // Header row
    let html = '<thead><tr><th>Strike</th>';
    for (const exp of grid.expiries) {
        const d = new Date(exp + 'T12:00:00');
        const label = d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
        html += `<th>${label}</th>`;
    }
    html += '</tr></thead><tbody>';

    // Data rows
    for (let i = 0; i < grid.strikes.length; i++) {
        const strike = grid.strikes[i];
        html += `<tr><td style="text-align:left;font-weight:600;color:var(--text);">$${strike.toFixed(1)}</td>`;
        for (let j = 0; j < grid.expiries.length; j++) {
            const cell = grid.data[i][j];
            if (cell.calls === 0 && cell.puts === 0) {
                html += '<td class="empty-cell">\u00b7</td>';
            } else {
                let parts = [];
                if (cell.calls !== 0) parts.push(`<span class="call-count">${cell.calls}C</span>`);
                if (cell.puts !== 0) parts.push(`<span class="put-count">${cell.puts}P</span>`);
                const cls = (cell.calls !== 0 && cell.puts !== 0) ? 'has-both' : (cell.calls !== 0 ? 'has-calls' : 'has-puts');
                html += `<td class="${cls}">${parts.join(' ')}</td>`;
            }
        }
        html += '</tr>';
    }
    html += '</tbody>';
    table.innerHTML = html;
}

function updateDailyStatus(tl, data) {
    const banner = document.getElementById('dailyStatus');
    const ds = tl.daily_status;
    if (!ds) return;

    const s = data.summary;
    const icons = { green: '\u2705', yellow: '\u26a0\ufe0f', red: '\ud83d\udd34' };

    banner.className = 'daily-banner status-' + ds.level;
    banner.innerHTML = `
        <div class="status-icon">${icons[ds.level] || '\u2705'}</div>
        <div class="status-text">
            <div class="status-headline">${ds.headline}</div>
            <div class="status-detail">Theta: ${fmt(s.total_theta)}/day | Delta: +${Math.round(s.net_delta).toLocaleString()} | Next expiry: ${ds.nearest_expiry || 'N/A'} (${ds.nearest_dte}d) | Smoothness: ${ds.smoothness_pct}%</div>
        </div>
    `;
}

function updateSmoothnessGauge(tl) {
    const score = tl.smoothness_score;
    const pct = Math.round(score * 100);

    // Update score display
    const scoreEl = document.getElementById('smoothnessScore');
    const color = pct < 40 ? 'var(--red)' : pct < 70 ? 'var(--orange)' : 'var(--green)';
    scoreEl.textContent = pct + '%';
    scoreEl.style.color = color;

    // Update needle angle: 0% = -90deg (left), 100% = 90deg (right)
    // Needle rotates from pointing left (-90) to pointing right (90)
    const angle = -90 + (pct / 100) * 180;
    const rad = angle * Math.PI / 180;
    const cx = 110, cy = 110, len = 80;
    const nx = cx + len * Math.sin(rad);
    const ny = cy - len * Math.cos(rad);
    const needle = document.getElementById('gaugeNeedle');
    if (needle) {
        needle.setAttribute('x2', nx.toFixed(1));
        needle.setAttribute('y2', ny.toFixed(1));
        needle.setAttribute('stroke', color);
    }
}

function updateWeeklyThetaChart(tl) {
    const dist = tl.weekly_theta_distribution;
    const avg = tl.avg_weekly_theta;
    if (!dist) return;

    const labels = Object.keys(dist);
    const values = Object.values(dist);

    // Color bars: green if >= 70% of avg, orange if 40-70%, red if < 40%
    const colors = values.map(v => {
        if (avg <= 0) return '#3fb950';
        const ratio = v / avg;
        if (ratio >= 0.7) return '#3fb950';
        if (ratio >= 0.4) return '#d29922';
        return '#f85149';
    });

    const traces = [{
        x: labels,
        y: values,
        type: 'bar',
        marker: { color: colors },
        text: values.map(v => '$' + v.toFixed(1)),
        textposition: 'outside',
        textfont: { color: '#e6edf3', size: 10 },
        hovertemplate: '%{x}<br>Theta: $%{y:.2f}/day<extra></extra>',
        name: 'Current',
    }];

    // Ideal line (average)
    traces.push({
        x: labels,
        y: labels.map(() => avg),
        type: 'scatter',
        mode: 'lines',
        line: { color: '#58a6ff', width: 2, dash: 'dash' },
        name: 'Avg ($' + avg.toFixed(1) + '/d)',
    });

    // Annotations for gaps
    const annotations = [];
    values.forEach((v, i) => {
        if (avg > 0 && v / avg < 0.5) {
            annotations.push({
                x: labels[i], y: v + avg * 0.25,
                text: '\u2191 GAP',
                showarrow: false,
                font: { color: '#f85149', size: 9 },
            });
        }
    });

    const layout = {
        ...plotlyLayout,
        title: { text: 'Theta Income per Week (Next 12 Weeks)', font: { color: '#8b949e', size: 13 } },
        annotations: annotations,
        margin: { ...plotlyLayout.margin, b: 80 },
        xaxis: {
            ...plotlyLayout.xaxis,
            type: 'category',
            tickangle: -45,
            tickfont: { size: 10, color: '#e6edf3' },
        },
        yaxis: { ...plotlyLayout.yaxis, title: 'Theta ($/day)', tickformat: '$,.1f' },
        legend: { x: 0.75, y: 0.98, bgcolor: 'rgba(0,0,0,0.5)', font: { size: 10 } },
        showlegend: true,
        bargap: 0.25,
    };

    Plotly.react('weeklyThetaChart', traces, layout, { responsive: true, displayModeBar: false });
}

function updateRecommendations(tl) {
    const container = document.getElementById('recommendationsList');
    const recs = tl.recommendations || [];

    if (recs.length === 0) {
        container.innerHTML = '<p style="color:var(--text-dim)">No recommendations</p>';
        return;
    }

    const typeColors = {
        'ROLL': 'var(--cyan)', 'ADD': 'var(--green)', 'OPEN': '#3fb950',
        'COVERED CALL': '#58a6ff', 'SELL SHARES': '#79c0ff', 'BUY SHARES': '#3fb950',
        'BUY PUT': '#d2a8ff',
        'STRANGLE': '#bc8cff', 'CLOSE': 'var(--orange)', 'TAKE PROFIT': '#f0883e',
        'LET EXPIRE': '#8b949e', 'ASSIGNMENT': '#f85149',
        'HOLD': '#8b949e'
    };

    // ── Time-sensitive items panel (assignments / expires within 7 days) ──
    // Surfaces near-deadline items at the very top — they're easy to miss
    // in the main rec list and the deadline is the most important fact.
    const timeSensitive = recs
        .filter(r => r.dte != null && r.dte <= 7 && r.dte >= 0
                  && ['ASSIGNMENT','LET EXPIRE','TAKE PROFIT','CLOSE'].includes(r.type))
        .sort((a, b) => a.dte - b.dte);
    let tsHtml = '';
    if (timeSensitive.length > 0) {
        const tsRows = timeSensitive.map(r => {
            const dteColor = r.dte <= 1 ? 'var(--red)' :
                              r.dte <= 3 ? 'var(--orange)' :
                              'var(--text-dim)';
            const dteLabel = r.dte === 0 ? 'today' :
                             r.dte === 1 ? 'tomorrow' :
                             `${r.dte}d`;
            const dv = r.dollar_value || 0;
            const dvColor = dv > 0 ? 'var(--green)' : dv < 0 ? 'var(--red)' : 'var(--text-dim)';
            const exp = r.source_exp || '';
            return `<div style="display:flex;align-items:center;gap:10px;padding:5px 0;border-bottom:1px dashed rgba(139,148,158,0.15);">
                <span style="font-weight:700;color:${dteColor};font-size:0.85rem;min-width:80px;">${dteLabel}</span>
                <span style="background:${typeColors[r.type]}30;color:${typeColors[r.type]};padding:2px 6px;border-radius:3px;font-size:0.7rem;font-weight:700;">${r.type}</span>
                <span style="flex:1;font-size:0.82rem;color:var(--text);">${r.action.replace(' — ', ' · ').substring(0, 90)}</span>
                ${dv ? `<span style="color:${dvColor};font-weight:700;font-size:0.82rem;">${dv>0?'+':''}$${Math.round(dv)}</span>` : ''}
            </div>`;
        }).join('');
        tsHtml = `<div style="margin-bottom:14px;padding:10px 14px;background:rgba(248,81,73,0.06);border-radius:8px;border:1px solid rgba(248,81,73,0.3);">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
                <span style="display:inline-block;padding:2px 10px;border-radius:4px;font-weight:700;font-size:0.85rem;color:#fff;background:#f85149;">⚠ TIME-SENSITIVE</span>
                <span style="font-size:0.74rem;color:var(--text-dim);">${timeSensitive.length} item${timeSensitive.length>1?'s':''} due within 7 days</span>
            </div>
            ${tsRows}
        </div>`;
    }

    // Portfolio metrics before/after
    const pm = tl.portfolio_metrics;
    let metricsHtml = '';
    if (pm) {
        const c = pm.current;
        const a = pm.after;
        const arrow = '\u2192';
        const chg = (val) => val > 0 ? `<span style="color:var(--green)">+${val}</span>` : val < 0 ? `<span style="color:var(--red)">${val}</span>` : `<span>${val}</span>`;
        const td = pm.target_delta || 7000;
        const regime = pm.regime || 'UNKNOWN';
        const maRatio = pm.ma_ratio || 1.0;
        const regimeColor = regime.includes('CHEAP') ? 'var(--green)' : (regime.includes('RICH') || regime.includes('EXPENSIVE')) ? 'var(--red)' : 'var(--orange)';
        const curExp = pm.current_exposure || 0;
        const tgtExp = pm.target_exposure || 0;
        const aftExp = pm.after_exposure || 0;
        const capBase = pm.capital_base || 1;
        const curLev = pm.current_leverage || 0;
        const tgtLev = pm.target_leverage || 0;
        const aftLev = pm.after_leverage || 0;
        const deltaVsTarget = c.delta - td;
        const deltaStatus = Math.abs(deltaVsTarget) < td * 0.15 ? 'ON TARGET' : deltaVsTarget > 0 ? 'OVER (reduce)' : 'UNDER (add)';
        const deltaStatusColor = Math.abs(deltaVsTarget) < td * 0.15 ? 'var(--green)' : 'var(--orange)';

        metricsHtml = `
        <div style="margin-bottom:12px;padding:10px 14px;background:rgba(88,166,255,0.08);border-radius:8px;border:1px solid var(--border);display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;align-items:center;">
            <div>
                <span style="font-size:0.7rem;color:var(--text-dim);text-transform:uppercase;">Regime</span><br>
                <span style="font-size:1.05rem;font-weight:700;color:${regimeColor}">${regime}</span>
                <span style="font-size:0.72rem;color:var(--text-dim);"> (z=${maRatio > 0 ? '+' : ''}${maRatio})</span>
            </div>
            <div>
                <span style="font-size:0.7rem;color:var(--text-dim);text-transform:uppercase;">$ Exposure</span><br>
                <span style="font-size:1.05rem;font-weight:700;color:var(--text)">$${(curExp/1000).toFixed(0)}k</span>
                <span style="font-size:0.72rem;color:var(--text-dim);"> → $${(tgtExp/1000).toFixed(0)}k target</span>
            </div>
            <div>
                <span style="font-size:0.7rem;color:var(--text-dim);text-transform:uppercase;">Leverage</span><br>
                <span style="font-size:1.05rem;font-weight:700;color:${curLev > tgtLev * 1.15 ? 'var(--red)' : curLev < tgtLev * 0.85 ? 'var(--orange)' : 'var(--green)'}">${curLev.toFixed(1)}x</span>
                <span style="font-size:0.72rem;color:var(--text-dim);"> → ${tgtLev.toFixed(1)}x target</span>
            </div>
            <div>
                <span style="font-size:0.7rem;color:var(--text-dim);text-transform:uppercase;">Delta (shares)</span><br>
                <span style="font-size:1.05rem;font-weight:700;">${c.delta.toLocaleString()}</span>
                <span style="font-size:0.72rem;color:${deltaStatusColor};"> ${deltaStatus}</span>
            </div>
        </div>
        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:8px;margin-bottom:16px;padding:12px;background:rgba(88,166,255,0.06);border-radius:8px;border:1px solid var(--border);">
            <div style="text-align:center;">
                <div style="font-size:0.7rem;color:var(--text-dim);text-transform:uppercase;">Theta</div>
                <div style="font-size:0.9rem;">$${c.theta.toFixed(0)}/d ${arrow} <strong>$${a.theta.toFixed(0)}/d</strong></div>
                <div style="font-size:0.75rem;">${chg((a.theta - c.theta).toFixed(1))}/d</div>
            </div>
            <div style="text-align:center;">
                <div style="font-size:0.7rem;color:var(--text-dim);text-transform:uppercase;">Delta (target: ${td.toLocaleString()})</div>
                <div style="font-size:0.9rem;">${c.delta.toFixed(0)} ${arrow} <strong>${a.delta.toFixed(0)}</strong></div>
                <div style="font-size:0.75rem;">${chg((a.delta - c.delta).toFixed(0))} (gap: ${(a.delta - td).toFixed(0)})</div>
            </div>
            <div style="text-align:center;">
                <div style="font-size:0.7rem;color:var(--text-dim);text-transform:uppercase;">Gamma</div>
                <div style="font-size:0.9rem;">${c.gamma.toFixed(0)} ${arrow} <strong>${a.gamma.toFixed(0)}</strong></div>
                <div style="font-size:0.75rem;">${chg((a.gamma - c.gamma).toFixed(0))}</div>
            </div>
            <div style="text-align:center;">
                <div style="font-size:0.7rem;color:var(--text-dim);text-transform:uppercase;">Smoothness</div>
                <div style="font-size:0.9rem;">${c.smoothness}% ${arrow} <strong>${a.smoothness}%</strong></div>
                <div style="font-size:0.75rem;">${chg((a.smoothness - c.smoothness).toFixed(0))}%</div>
            </div>
        </div>
        ${(() => {
            // Cycle 158: weekly income progress vs $1,500 target.
            // The strategic objective surface — operator sees where they
            // are and what executing the beam-picked trades would do.
            const cw = c.avg_weekly_theta;
            const aw = a.avg_weekly_theta;
            const cp = c.pct_of_target;
            const ap = a.pct_of_target;
            if (cw === undefined || aw === undefined) return '';
            const barCur = Math.min(100, cp);
            const barAft = Math.min(100, ap);
            const aftColor = ap >= 100 ? 'var(--green)' : ap >= 70 ? '#9bc' : ap >= 40 ? 'var(--orange)' : 'var(--red)';
            // Cycle 171: 7-day income trajectory sparkline from progress.db.
            // Operator sees whether income is trending up or down over the
            // past week, alongside the current → after comparison.
            // Target line ($1,500/wk) drawn dashed for reference.
            const trajHtml = (() => {
                const prog = window._progressData;
                if (!prog || !Array.isArray(prog.snapshots) || prog.snapshots.length < 2) return '';
                const snaps = prog.snapshots.slice().reverse();  // chronological
                const vals = snaps.map(s => typeof s.avg_weekly_theta === 'number' ? s.avg_weekly_theta : null);
                const finite = vals.filter(v => v !== null && isFinite(v));
                if (finite.length < 2) return '';
                const tgt = 1500;
                const mn = Math.min(...finite, 0);
                const mx = Math.max(...finite, tgt * 0.5);  // include some range for the trajectory
                const W = 300, H = 48, pad = 4;
                const n = vals.length;
                const xStep = (W - pad*2) / Math.max(1, n - 1);
                const pts = vals.map((v, i) => {
                    const x = pad + i * xStep;
                    const y = (v === null) ? (H - pad) : (H - pad) - ((v - mn) / (mx - mn)) * (H - pad*2);
                    return `${x.toFixed(1)},${y.toFixed(1)}`;
                }).join(' ');
                const tgtY = mx >= tgt ? (H - pad) - ((tgt - mn) / (mx - mn)) * (H - pad*2) : -10;
                const lastVal = finite[finite.length - 1];
                const firstVal = finite[0];
                const trendColor = lastVal > firstVal ? 'var(--green)' : lastVal < firstVal ? 'var(--red)' : '#8b949e';
                const trendChg = lastVal - firstVal;
                const trendLabel = `${trendChg >= 0 ? '+' : ''}$${Math.round(trendChg)}/wk vs ${n}d ago`;
                return `
                <div style="margin-top:8px;padding-top:6px;border-top:1px dashed rgba(139,148,158,0.15);">
                    <div style="display:flex;justify-content:space-between;align-items:baseline;font-size:0.7rem;color:var(--text-dim);margin-bottom:2px;">
                        <span>${n}-day trajectory</span>
                        <span style="color:${trendColor};font-weight:600;">${trendLabel}</span>
                    </div>
                    <svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}" preserveAspectRatio="none" style="display:block;">
                        ${tgtY >= 0 ? `<line x1="0" y1="${tgtY}" x2="${W}" y2="${tgtY}" stroke="rgba(63,185,80,0.4)" stroke-width="1" stroke-dasharray="3,3"/>` : ''}
                        <polyline points="${pts}" fill="none" stroke="${trendColor}" stroke-width="1.5"/>
                    </svg>
                </div>`;
            })();
            return `
            <div style="margin-bottom:16px;padding:12px 14px;background:linear-gradient(90deg,rgba(63,185,80,0.06),rgba(88,166,255,0.04));border-radius:8px;border:1px solid var(--border);">
                <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:6px;">
                    <span style="font-size:0.78rem;color:var(--text-dim);text-transform:uppercase;letter-spacing:0.04em;">Weekly Income Progress</span>
                    <span style="font-size:0.72rem;color:var(--text-dim);">target: $1,500/wk</span>
                </div>
                <div style="display:flex;align-items:baseline;gap:12px;font-size:0.95rem;">
                    <span><strong>$${cw.toFixed(0)}/wk</strong> <span style="color:var(--text-dim);font-size:0.78rem;">(${cp.toFixed(0)}% of target)</span></span>
                    <span style="color:var(--text-dim);">${arrow}</span>
                    <span style="color:${aftColor};"><strong>$${aw.toFixed(0)}/wk</strong> <span style="font-size:0.78rem;">(${ap.toFixed(0)}%)</span></span>
                    <span style="font-size:0.78rem;color:var(--text-dim);margin-left:auto;">${chg((aw - cw).toFixed(0))}/wk if you execute the beam-picked trades</span>
                </div>
                <div style="position:relative;height:8px;background:rgba(139,148,158,0.15);border-radius:4px;overflow:hidden;margin-top:8px;">
                    <div style="position:absolute;left:0;top:0;bottom:0;width:${barCur}%;background:#8b949e;"></div>
                    <div style="position:absolute;left:0;top:0;bottom:0;width:${barAft}%;background:${aftColor};opacity:0.6;border-right:2px solid ${aftColor};"></div>
                </div>
                ${trajHtml}
            </div>`;
        })()}`;

        // Stress test summary
        const stress = pm.stress;
        if (stress) {
            const crashDeltaColor = Math.abs(stress.crash_5d.delta_after) > Math.abs(stress.crash_5d.delta_before) * 1.2 ? 'var(--red)' : 'var(--green)';
            const rallyDeltaColor = Math.abs(stress.rally_5d.delta_after) > Math.abs(stress.rally_5d.delta_before) * 1.2 ? 'var(--orange)' : 'var(--green)';
            metricsHtml += `
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:16px;padding:8px 12px;background:rgba(248,81,73,0.06);border-radius:8px;border:1px solid rgba(248,81,73,0.2);font-size:0.8rem;">
                <div>
                    <span style="color:var(--red);">\u26A0</span> If UNG crashes to <strong>$${stress.crash_5d.price}</strong> (-11.7%):
                    Delta <span style="color:var(--text-dim)">${stress.crash_5d.delta_before}</span> ${arrow} <strong style="color:${crashDeltaColor}">${stress.crash_5d.delta_after}</strong>
                </div>
                <div>
                    <span style="color:var(--green);">\u25B2</span> If UNG spikes to <strong>$${stress.rally_5d.price}</strong> (+12.1%):
                    Delta <span style="color:var(--text-dim)">${stress.rally_5d.delta_before}</span> ${arrow} <strong style="color:${rallyDeltaColor}">${stress.rally_5d.delta_after}</strong>
                </div>
            </div>`;
        }

        // Gamma regime banner
        const gr = pm.gamma_regime;
        const gStance = pm.gamma_stance || '';
        const gReasons = pm.gamma_reasoning || [];
        if (gr) {
            const grColors = {'ACCUMULATE': 'var(--green)', 'HOLD': 'var(--blue)', 'HARVEST': 'var(--orange)', 'EXIT': 'var(--red)'};
            const grBgColors = {'ACCUMULATE': 'rgba(63,185,80,0.10)', 'HOLD': 'rgba(88,166,255,0.10)', 'HARVEST': 'rgba(210,153,34,0.10)', 'EXIT': 'rgba(248,81,73,0.10)'};
            const grBorderColors = {'ACCUMULATE': 'rgba(63,185,80,0.3)', 'HOLD': 'rgba(88,166,255,0.3)', 'HARVEST': 'rgba(210,153,34,0.3)', 'EXIT': 'rgba(248,81,73,0.3)'};
            const grColor = grColors[gr] || 'var(--text-dim)';
            const grBg = grBgColors[gr] || 'rgba(139,148,158,0.10)';
            const grBorder = grBorderColors[gr] || 'rgba(139,148,158,0.3)';
            const reasonBullets = gReasons.map(r => `<li style="margin:2px 0;">${r}</li>`).join('');
            metricsHtml += `
            <div style="margin-bottom:12px;padding:10px 14px;background:${grBg};border-radius:8px;border:1px solid ${grBorder};">
                <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">
                    <span style="display:inline-block;padding:2px 10px;border-radius:4px;font-weight:700;font-size:0.85rem;color:#fff;background:${grColor};">${gr}</span>
                    <span style="font-size:0.8rem;color:var(--text-dim);">Gamma Regime</span>
                </div>
                <div style="font-size:0.82rem;color:var(--text);margin-bottom:4px;">${gStance}</div>
                <ul style="font-size:0.75rem;color:var(--text-dim);margin:0;padding-left:18px;list-style:disc;">${reasonBullets}</ul>
            </div>`;
        }

        // ── Cyclical & Income card (Tech/Fund/YoY pillars + income target) ──
        const dm = pm.deployment_mode || 'ACTIVE';
        const dmColors = {'ACTIVE': 'var(--green)', 'TRANSITION': 'var(--orange)', 'WAITING': 'var(--text-dim)'};
        const dmColor = dmColors[dm] || 'var(--text-dim)';
        const sr = pm.supply_regime || 'BALANCED';
        const srColors = {'SURPLUS': 'var(--red)', 'BALANCED': 'var(--text-dim)', 'SHORTAGE': 'var(--green)'};
        const srColor = srColors[sr] || 'var(--text-dim)';
        const sz = pm.storage_z || 0;
        const ib = pm.income_bias || 0.5;
        const gb = pm.growth_bias || 0.5;
        const ps = pm.pillar_scores || {};
        const tech = (ps.tech || 0);
        const fund = (ps.fund || 0);
        const yoy  = (ps.yoy  || 0);
        const pdrift = (ps.drift_per_day || 0);
        const pillarBar = (v) => {
            const pct = Math.max(-1, Math.min(1, v)) * 100;
            const w = Math.abs(pct);
            const c = v > 0 ? 'var(--green)' : v < 0 ? 'var(--red)' : 'var(--text-dim)';
            const side = v >= 0 ? `left:50%;` : `right:50%;`;
            return `<div style="position:relative;height:6px;background:rgba(139,148,158,0.2);border-radius:3px;">
                <div style="position:absolute;${side}width:${w/2}%;height:100%;background:${c};border-radius:3px;"></div>
                <div style="position:absolute;left:50%;top:-2px;width:1px;height:10px;background:var(--text-dim);"></div>
            </div>`;
        };
        // Income target: avg_weekly_theta vs $1500/wk goal (CENTRAL_PHILOSOPHY).
        const awt = tl.avg_weekly_theta || 0;
        const targetWk = 1500;
        const incPct = Math.min(150, (awt / targetWk) * 100);
        const incColor = awt >= targetWk ? 'var(--green)' : awt >= targetWk * 0.66 ? 'var(--orange)' : 'var(--red)';
        const driftBps = (pdrift * 10000).toFixed(1);  // bps/day
        // Prediction freshness indicator
        const predUpdated = pm.predictions_updated_at;
        let freshnessTxt = 'predictions: never refreshed';
        let freshnessColor = 'var(--red)';
        if (predUpdated) {
            const ageMs = Date.now() - new Date(predUpdated).getTime();
            const ageMin = ageMs / 60000;
            const ageHr = ageMin / 60;
            if (ageMin < 60) {
                freshnessTxt = `predictions ${ageMin.toFixed(0)}m old`;
                freshnessColor = 'var(--green)';
            } else if (ageHr < 6) {
                freshnessTxt = `predictions ${ageHr.toFixed(1)}h old`;
                freshnessColor = 'var(--orange)';
            } else {
                freshnessTxt = `predictions ${ageHr.toFixed(0)}h old (stale)`;
                freshnessColor = 'var(--red)';
            }
        }
        // Quality score
        const qb = pm.quality_before || {total: 0, components: {}};
        const qa = pm.quality_after || {total: 0, components: {}};
        const qd = pm.quality_delta || 0;
        const qdColor = qd > 0 ? 'var(--green)' : qd < 0 ? 'var(--red)' : 'var(--text-dim)';
        const qFmt = (v) => `$${Math.round(v).toLocaleString()}`;
        metricsHtml += `
        <div style="margin-bottom:12px;padding:10px 14px;background:rgba(188,140,255,0.06);border-radius:8px;border:1px solid rgba(188,140,255,0.25);">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;flex-wrap:wrap;">
                <span style="display:inline-block;padding:2px 10px;border-radius:4px;font-weight:700;font-size:0.85rem;color:#fff;background:#bc8cff;">CYCLICAL & INCOME</span>
                <span style="font-size:0.78rem;color:var(--text-dim);">Tech/Fund/YoY drift: <strong style="color:${pdrift>=0?'var(--green)':'var(--red)'}">${pdrift>=0?'+':''}${driftBps} bps/d</strong></span>
                <span style="font-size:0.72rem;color:${freshnessColor};margin-left:auto;">${freshnessTxt}</span>
            </div>
            <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin-bottom:10px;padding:8px;background:rgba(88,166,255,0.04);border-radius:6px;font-size:0.78rem;">
                <div>
                    <div style="color:var(--text-dim);text-transform:uppercase;font-size:0.65rem;">Portfolio Quality (now)</div>
                    <div style="font-weight:700;font-size:0.95rem;color:${qb.total<0?'var(--red)':'var(--green)'};">${qFmt(qb.total)}</div>
                </div>
                <div>
                    <div style="color:var(--text-dim);text-transform:uppercase;font-size:0.65rem;">After all recs</div>
                    <div style="font-weight:700;font-size:0.95rem;color:${qa.total<0?'var(--red)':'var(--green)'};">${qFmt(qa.total)}</div>
                </div>
                <div>
                    <div style="color:var(--text-dim);text-transform:uppercase;font-size:0.65rem;">Δ if executed</div>
                    <div style="font-weight:700;font-size:0.95rem;color:${qdColor};">${qd>=0?'+':''}${qFmt(qd)}</div>
                </div>
                <div style="grid-column:1/-1;font-size:0.68rem;">
                    ${(() => {
                        // Quality components bar chart (cycle 55): each component
                        // gets a row with signed bar widthed by |magnitude| /
                        // max(|all magnitudes|). Two thin sub-bars stacked:
                        // current (top) and after-recs (bottom) so the operator
                        // sees both magnitude AND direction of the proposed move.
                        const compKeys = [
                            ['income_gap', 'Income gap'],
                            ['dd_penalty', 'DD penalty'],
                            ['delta_gap', 'Delta gap'],
                            ['smoothness', 'Smoothness'],
                            ['tail_hedge', 'Tail hedge'],
                            ['pillar_drift', 'Pillar drift'],
                        ];
                        const qbC = qb.components || {};
                        const qaC = qa.components || {};
                        const mags = compKeys.flatMap(([k]) => [Math.abs(qbC[k]||0), Math.abs(qaC[k]||0)]);
                        const maxMag = Math.max(1, ...mags);
                        const rowFor = (label, vBefore, vAfter) => {
                            const wB = Math.min(100, (Math.abs(vBefore) / maxMag) * 100);
                            const wA = Math.min(100, (Math.abs(vAfter) / maxMag) * 100);
                            const colorB = vBefore >= 0 ? 'rgba(63,185,80,0.55)' : 'rgba(248,81,73,0.55)';
                            const colorA = vAfter >= 0 ? 'rgba(63,185,80,0.85)' : 'rgba(248,81,73,0.85)';
                            const delta = vAfter - vBefore;
                            const dColor = delta > 0 ? 'var(--green)' : delta < 0 ? 'var(--red)' : 'var(--text-dim)';
                            const dStr = delta === 0 ? '·' : `${delta>0?'+':''}${qFmt(delta)}`;
                            return `<div style="display:grid;grid-template-columns:90px 1fr 80px 70px;align-items:center;gap:8px;padding:2px 0;">
                                <span style="color:var(--text);">${label}</span>
                                <div style="position:relative;height:14px;background:rgba(139,148,158,0.08);border-radius:3px;overflow:hidden;">
                                    <div style="position:absolute;left:0;top:0;height:7px;width:${wB}%;background:${colorB};"></div>
                                    <div style="position:absolute;left:0;bottom:0;height:7px;width:${wA}%;background:${colorA};"></div>
                                </div>
                                <span style="font-family:monospace;text-align:right;color:${vBefore<0?'var(--red)':'var(--green)'};">${qFmt(vBefore)}</span>
                                <span style="font-family:monospace;text-align:right;font-size:0.62rem;color:${dColor};">${dStr}</span>
                            </div>`;
                        };
                        const rows = compKeys.map(([k, label]) => rowFor(label, qbC[k]||0, qaC[k]||0)).join('');
                        return `
                            <div style="display:grid;grid-template-columns:90px 1fr 80px 70px;gap:8px;padding:0 0 4px;font-size:0.6rem;color:var(--text-dim);text-transform:uppercase;letter-spacing:0.04em;">
                                <span>component</span>
                                <span>magnitude (top=now, bottom=after)</span>
                                <span style="text-align:right;">now</span>
                                <span style="text-align:right;">Δ</span>
                            </div>
                            ${rows}`;
                    })()}
                </div>
                ${(() => {
                    // DD decomposition (cycle 56): when dd_penalty is the
                    // dominant negative component, show the operator what's
                    // driving the projected tail loss — delta exposure on a
                    // 5% CVaR price drop, gamma convexity, and theta offset.
                    const dd = qb.dd_diagnostics;
                    if (!dd || !dd.cvar_drop || dd.cvar_drop <= 0) return '';
                    const ddAfter = qa.dd_diagnostics || {};
                    const fmtUsdNoDollar = (v) => Math.round(v).toLocaleString();
                    const fmtUsd = (v) => `${v < 0 ? '-' : ''}$${fmtUsdNoDollar(Math.abs(v))}`;
                    const driverRow = (label, vBefore, vAfter, note) => {
                        const dlt = vAfter - vBefore;
                        const dColor = dlt > 0 ? 'var(--green)' : dlt < 0 ? 'var(--red)' : 'var(--text-dim)';
                        const vColor = vBefore < 0 ? 'var(--red)' : vBefore > 0 ? 'var(--green)' : 'var(--text-dim)';
                        return `<div style="display:grid;grid-template-columns:140px 90px 60px 1fr;align-items:center;gap:8px;padding:1px 0;font-size:0.7rem;">
                            <span style="color:var(--text);">${label}</span>
                            <span style="font-family:monospace;text-align:right;color:${vColor};">${fmtUsd(vBefore)}</span>
                            <span style="font-family:monospace;text-align:right;font-size:0.62rem;color:${dColor};">${dlt === 0 ? '·' : `${dlt>0?'+':''}${fmtUsd(dlt)}`}</span>
                            <span style="color:var(--text-dim);font-size:0.62rem;">${note || ''}</span>
                        </div>`;
                    };
                    const cvarPctOfSpot = pm.current && pm.current.theta ? '' : '';
                    return `
                    <div style="grid-column:1/-1;margin-top:8px;padding:8px 10px;background:rgba(248,81,73,0.06);border-radius:6px;border:1px solid rgba(248,81,73,0.2);">
                        <div style="display:flex;align-items:baseline;gap:8px;margin-bottom:6px;flex-wrap:wrap;">
                            <span style="display:inline-block;padding:1px 8px;border-radius:3px;font-weight:700;font-size:0.7rem;color:#fff;background:#f85149;">DD DRIVERS</span>
                            <span style="font-size:0.7rem;color:var(--text-dim);">5%-CVaR 30d price drop = <strong style="color:var(--text);">$${dd.cvar_drop}</strong>; tail P&amp;L projects <strong style="color:var(--red);">${fmtUsd(dd.tail_pnl)}</strong> = <strong>${(dd.dd_frac*100).toFixed(1)}%</strong> of capital (threshold ${(dd.dd_threshold*100).toFixed(0)}%)</span>
                        </div>
                        ${driverRow('Delta loss', dd.delta_loss || 0, ddAfter.delta_loss || 0,
                            `-Δ × CVaR drop · current Δ=${dd.total_delta}`)}
                        ${driverRow('Gamma convexity', dd.gamma_convexity || 0, ddAfter.gamma_convexity || 0,
                            `½·Γ·drop² · Γ=${dd.total_gamma}`)}
                        ${driverRow('Theta 30d offset', dd.theta_offset || 0, ddAfter.theta_offset || 0,
                            `expiry-aware accrual`)}
                        <div style="margin-top:4px;padding-top:4px;border-top:1px dashed rgba(248,81,73,0.25);">
                            ${driverRow('Net tail P&L', dd.tail_pnl || 0, ddAfter.tail_pnl || 0, '')}
                        </div>
                        ${(() => {
                            // Hedge math (cycle 57): convert the abstract
                            // -$X shortfall vs the -10% threshold into
                            // concrete levers — how many shares to trim, how
                            // much gamma to add, or how much extra income/wk
                            // would each close the gap independently.
                            const cap = dd.capital || 0;
                            if (cap <= 0) return '';
                            const thresholdPnl = dd.dd_threshold * cap;  // negative target
                            const shortfall = thresholdPnl - dd.tail_pnl; // positive $ needed
                            if (shortfall <= 0) {
                                return `<div style="margin-top:6px;padding:4px 6px;background:rgba(63,185,80,0.1);border-radius:4px;font-size:0.7rem;color:var(--green);">
                                    ✓ Within DD threshold — tail P&amp;L ${fmtUsd(dd.tail_pnl)} ≥ allowed ${fmtUsd(thresholdPnl)}
                                </div>`;
                            }
                            const drop = dd.cvar_drop || 0;
                            const sharesNeeded = drop > 0 ? Math.round(shortfall / drop) : null;
                            const gammaNeeded = drop > 0 ? Math.round(shortfall / (0.5 * drop * drop)) : null;
                            const wkNeeded = Math.round(shortfall * 7 / 30);
                            const baseRow = (label, val, unit, note) => `<div style="display:grid;grid-template-columns:140px 110px 1fr;align-items:center;gap:8px;padding:1px 0;font-size:0.7rem;">
                                <span style="color:var(--text);">${label}</span>
                                <span style="font-family:monospace;text-align:right;color:var(--orange);">${val !== null ? val.toLocaleString() : 'n/a'}${unit}</span>
                                <span style="color:var(--text-dim);font-size:0.62rem;">${note}</span>
                            </div>`;
                            return `
                            <div style="margin-top:6px;padding:6px 8px;background:rgba(240,136,62,0.08);border-radius:4px;border:1px solid rgba(240,136,62,0.2);">
                                <div style="font-size:0.68rem;color:var(--orange);text-transform:uppercase;letter-spacing:0.04em;margin-bottom:3px;">
                                    Hedge math · shortfall <strong>${fmtUsd(shortfall)}</strong> to reach ${(dd.dd_threshold*100).toFixed(0)}% threshold
                                </div>
                                ${baseRow('Trim shares', sharesNeeded, ' sh', `reduce Δ by ${sharesNeeded} (close ~${Math.min(100, sharesNeeded/(Math.abs(dd.total_delta)||1)*100).toFixed(0)}% of net Δ exposure)`)}
                                ${(() => {
                                    // Use REAL per-contract gamma (cycle 63 fix). Cycle 57
                                    // used a 2000-Γ heuristic that was 50-200× too high.
                                    const gpc = dd.atm_put_gamma_per_contract || {};
                                    const g30 = gpc['30d'] || 0;
                                    const g90 = gpc['90d'] || 0;
                                    const g365 = gpc['365d'] || 0;
                                    const n30 = g30 > 0 ? Math.ceil(gammaNeeded / g30) : null;
                                    const n90 = g90 > 0 ? Math.ceil(gammaNeeded / g90) : null;
                                    const n365 = g365 > 0 ? Math.ceil(gammaNeeded / g365) : null;
                                    return baseRow('Add long gamma', gammaNeeded, ' Γ',
                                        `~${n30}× 30d (${g30.toFixed(0)} Γ/ea) · ${n90}× 90d (${g90.toFixed(0)} Γ/ea) · ${n365}× LEAPS (${g365.toFixed(0)} Γ/ea) — long puts at ATM gamma per contract is small, so realistic for crisis hedge only`);
                                })()}
                                ${(() => {
                                    // "Close shorts" lever — actionable target derived
                                    // from cycle-62 risk_by_expiry. The biggest gamma
                                    // sink is the cheapest single close.
                                    const rbe = pm.risk_by_expiry || [];
                                    if (!rbe.length) return '';
                                    const top = rbe[0];
                                    if (!top || (top.gamma || 0) >= 0) return '';  // only short-gamma expiries
                                    const drop = dd.cvar_drop || 0;
                                    const gammaImp = 0.5 * Math.abs(top.gamma) * drop * drop;
                                    const deltaImp = Math.abs(top.delta || 0) * drop;
                                    const thetaLoss = (top.theta || 0) * 30;
                                    const netImp = gammaImp + deltaImp - thetaLoss;
                                    return `<div style="display:grid;grid-template-columns:140px 110px 1fr;align-items:center;gap:8px;padding:1px 0;font-size:0.7rem;">
                                        <span style="color:var(--text);">Close gamma sink</span>
                                        <span style="font-family:monospace;text-align:right;color:var(--orange);">${top.contracts}× ${top.expiry}</span>
                                        <span style="color:var(--text-dim);font-size:0.62rem;">+${fmtUsd(netImp)} tail (Γ +${Math.abs(top.gamma).toLocaleString()}, Δ +${Math.abs(top.delta||0)}, θ -${fmtUsd(thetaLoss)} forfeit)</span>
                                    </div>`;
                                })()}
                                ${baseRow('Lift income', wkNeeded, '$/wk', `would offset 30d shortfall — usually unrealistic vs ${pm.current && pm.current.theta ? Math.round((pm.current.theta||0)*7) : '?'} $/wk current`)}
                                <div style="margin-top:3px;font-size:0.62rem;color:var(--text-dim);font-style:italic;">
                                    Each row independently closes the gap. Long-put gamma is small per contract (5–40 Γ depending on DTE); reducing short gamma by closing existing positions is usually the more practical lever.
                                </div>
                            </div>`;
                        })()}
                    </div>`;
                })()}
            </div>
            <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:14px;font-size:0.78rem;">
                <div>
                    <div style="color:var(--text-dim);text-transform:uppercase;font-size:0.68rem;">Deployment Mode</div>
                    <div style="font-weight:700;font-size:1rem;color:${dmColor};">${dm}</div>
                </div>
                <div>
                    <div style="color:var(--text-dim);text-transform:uppercase;font-size:0.68rem;">Supply Regime (axis B)</div>
                    <div style="font-weight:700;font-size:1rem;color:${srColor};">${sr}</div>
                    <div style="color:var(--text-dim);font-size:0.72rem;">storage z=${sz >= 0 ? '+' : ''}${sz.toFixed(2)}</div>
                </div>
                <div>
                    <div style="color:var(--text-dim);text-transform:uppercase;font-size:0.68rem;">Income / Growth Bias</div>
                    <div style="display:flex;align-items:center;gap:6px;">
                        <span style="color:var(--orange);font-weight:700;">${(ib*100).toFixed(0)}%</span>
                        <div style="flex:1;height:6px;background:linear-gradient(to right, var(--orange) ${ib*100}%, var(--green) ${ib*100}%);border-radius:3px;"></div>
                        <span style="color:var(--green);font-weight:700;">${(gb*100).toFixed(0)}%</span>
                    </div>
                    <div style="color:var(--text-dim);font-size:0.68rem;">income ↔ growth</div>
                </div>
                <div>
                    <div style="color:var(--text-dim);text-transform:uppercase;font-size:0.68rem;">Weekly Income</div>
                    <div style="font-weight:700;font-size:1rem;color:${incColor};">$${awt.toFixed(0)}/wk</div>
                    <div style="height:5px;background:rgba(139,148,158,0.2);border-radius:3px;overflow:hidden;margin-top:3px;">
                        <div style="width:${Math.min(100,incPct)}%;height:100%;background:${incColor};"></div>
                    </div>
                    <div style="color:var(--text-dim);font-size:0.68rem;">target $${targetWk}/wk (${incPct.toFixed(0)}%)</div>
                </div>
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;margin-top:10px;font-size:0.72rem;">
                <div>
                    <div style="color:var(--text-dim);text-transform:uppercase;font-size:0.65rem;">Tech (price band + MA)</div>
                    ${pillarBar(tech)}
                    <div style="margin-top:2px;color:${tech>=0?'var(--green)':'var(--red)'}">${tech>=0?'+':''}${tech.toFixed(3)}</div>
                </div>
                <div>
                    <div style="color:var(--text-dim);text-transform:uppercase;font-size:0.65rem;">Fund (storage + days_supply)</div>
                    ${pillarBar(fund)}
                    <div style="margin-top:2px;color:${fund>=0?'var(--green)':'var(--red)'}">
                        ${fund>=0?'+':''}${fund.toFixed(3)}
                        ${(() => {
                            const fr = (ps && typeof ps.fund_raw === 'number') ? ps.fund_raw : fund;
                            return (Math.abs(fr) > 1.0 + 0.01)
                                ? `<span style="color:var(--orange);font-size:0.62rem;font-weight:600;margin-left:4px;">CLIPPED (raw ${fr>=0?'+':''}${fr.toFixed(3)})</span>`
                                : '';
                        })()}
                    </div>
                </div>
                <div>
                    <div style="color:var(--text-dim);text-transform:uppercase;font-size:0.65rem;">YoY (demand−supply growth)</div>
                    ${pillarBar(yoy)}
                    <div style="margin-top:2px;color:${yoy>=0?'var(--green)':'var(--red)'}">
                        ${yoy>=0?'+':''}${yoy.toFixed(3)}
                        ${(() => {
                            const yr = (ps && typeof ps.yoy_raw === 'number') ? ps.yoy_raw : yoy;
                            return (Math.abs(yr) > 1.0 + 0.01)
                                ? `<span style="color:var(--orange);font-size:0.62rem;font-weight:600;margin-left:4px;">CLIPPED (raw ${yr>=0?'+':''}${yr.toFixed(3)})</span>`
                                : '';
                        })()}
                    </div>
                </div>
            </div>
        </div>`;
    }

    const fmtImpact = (val, prefix, suffix) => {
        if (val === undefined || val === null || val === 0) return '';
        const color = val > 0 ? 'var(--green)' : 'var(--red)';
        const sign = val > 0 ? '+' : '';
        return `<span style="color:${color}">${sign}${prefix}${Math.round(val)}${suffix}</span>`;
    };

    // Model fundamentals (NG forecast charts) — embedded as PNGs served
    // by /api/factor_curves.png + siblings. Cache-buster forces fresh
    // pull whenever forecast re-runs.
    const cb = Date.now();
    // IC weights table — sorted descending so the top drivers are obvious
    const icw = pm.ic_weights || {};
    const icEntries = Object.entries(icw)
        .map(([col, v]) => [col, v.label || col, v.ic_weight || 0])
        .sort((a, b) => b[2] - a[2]);
    let icRows = '';
    const maxIc = icEntries.length > 0 ? Math.max(...icEntries.map(e => e[2])) : 1;
    for (const [col, label, w] of icEntries) {
        const widthPct = Math.min(100, (w / Math.max(0.02, maxIc)) * 100);
        const wColor = w >= 0.10 ? 'var(--green)' : w >= 0.05 ? 'var(--orange)' : 'var(--text-dim)';
        icRows += `<div style="display:flex;align-items:center;gap:8px;padding:2px 0;font-size:0.74rem;">
            <span style="min-width:160px;color:var(--text);">${label}</span>
            <div style="flex:1;height:5px;background:rgba(139,148,158,0.15);border-radius:3px;overflow:hidden;">
                <div style="width:${widthPct}%;height:100%;background:${wColor};"></div>
            </div>
            <span style="min-width:60px;text-align:right;color:${wColor};font-family:monospace;">${w.toFixed(3)}</span>
        </div>`;
    }

    // Progress card — sparklines from /api/progress (cycle 53)
    const _prog = window._progressData;
    const _snapsAll = (_prog && Array.isArray(_prog.snapshots)) ? _prog.snapshots.slice() : [];
    // /api/progress returns DESC by date; reverse to chronological
    const _snaps = _snapsAll.slice().reverse();
    let progressHtml = '';
    if (_snaps.length >= 1) {
        const _spark = (vals, color, w=220, h=44) => {
            if (!vals.length) return '';
            const finite = vals.filter(v => typeof v === 'number' && isFinite(v));
            if (!finite.length) return '';
            let mn = Math.min(...finite), mx = Math.max(...finite);
            if (mn === mx) { mn -= 1; mx += 1; }
            const pad = 4;
            const n = vals.length;
            const xStep = (w - pad*2) / Math.max(1, n - 1);
            const pts = vals.map((v, i) => {
                const x = pad + i * xStep;
                const y = (typeof v === 'number' && isFinite(v))
                    ? (h - pad) - ((v - mn) / (mx - mn)) * (h - pad*2)
                    : h - pad;
                return `${x.toFixed(1)},${y.toFixed(1)}`;
            }).join(' ');
            const areaPts = `${pad},${h-pad} ${pts} ${(pad + (n-1)*xStep).toFixed(1)},${h-pad}`;
            return `<svg viewBox="0 0 ${w} ${h}" width="100%" height="${h}" preserveAspectRatio="none" style="display:block;">
                <polygon points="${areaPts}" fill="${color}" opacity="0.18"/>
                <polyline points="${pts}" fill="none" stroke="${color}" stroke-width="1.6"/>
            </svg>`;
        };
        const _fmtUsd = (v) => {
            const a = Math.abs(v);
            if (a >= 1000) return (v < 0 ? '-' : '') + '$' + (a/1000).toFixed(1) + 'k';
            return (v < 0 ? '-' : '') + '$' + a.toFixed(0);
        };
        const _delta = (arr) => {
            if (arr.length < 2) return null;
            const a = arr[0], b = arr[arr.length - 1];
            if (typeof a !== 'number' || typeof b !== 'number') return null;
            return b - a;
        };
        const qVals = _snaps.map(s => typeof s.quality_total === 'number' ? s.quality_total : null).filter(v => v !== null);
        const wVals = _snaps.map(s => typeof s.avg_weekly_theta === 'number' ? s.avg_weekly_theta : null).filter(v => v !== null);
        const ddVals = _snaps.map(s => typeof s.dd_penalty === 'number' ? s.dd_penalty : null).filter(v => v !== null);
        const igVals = _snaps.map(s => typeof s.income_gap === 'number' ? s.income_gap : null).filter(v => v !== null);
        const qLast = qVals.length ? qVals[qVals.length - 1] : null;
        const wLast = wVals.length ? wVals[wVals.length - 1] : null;
        const ddLast = ddVals.length ? ddVals[ddVals.length - 1] : null;
        const igLast = igVals.length ? igVals[igVals.length - 1] : null;
        const qDelta = _delta(qVals);
        const wDelta = _delta(wVals);
        const ddDelta = _delta(ddVals);
        const igDelta = _delta(igVals);
        const _deltaLabel = (d, betterDown=false) => {
            if (d === null) return '';
            const arrow = d >= 0 ? '▲' : '▼';
            const good = betterDown ? (d < 0) : (d >= 0);
            const c = good ? 'var(--green)' : 'var(--red,#f85149)';
            return `<span style="color:${c};font-family:monospace;font-size:0.74rem;">${arrow} ${_fmtUsd(d)}</span>`;
        };
        const _card = (title, lastVal, deltaHtml, sparkSvg) => `
            <div style="background:rgba(88,166,255,0.04);border:1px solid rgba(88,166,255,0.18);border-radius:6px;padding:8px 10px;">
                <div style="display:flex;justify-content:space-between;align-items:baseline;gap:8px;">
                    <span style="font-size:0.72rem;color:var(--text-dim);text-transform:uppercase;letter-spacing:0.04em;">${title}</span>
                    ${deltaHtml}
                </div>
                <div style="font-family:monospace;font-size:1.05rem;color:var(--text);margin:2px 0 4px;">${lastVal}</div>
                ${sparkSvg}
            </div>`;
        const _n = _snaps.length;
        const dateFrom = _snaps[0]?.date || '';
        const dateTo = _snaps[_n-1]?.date || '';
        progressHtml = `
        <details style="margin-bottom:12px;padding:10px 14px;background:rgba(88,166,255,0.06);border-radius:8px;border:1px solid rgba(88,166,255,0.25);" open>
            <summary style="cursor:pointer;display:flex;align-items:center;gap:10px;list-style:none;">
                <span style="display:inline-block;padding:2px 10px;border-radius:4px;font-weight:700;font-size:0.85rem;color:#fff;background:#58a6ff;">PROGRESS</span>
                <span style="font-size:0.78rem;color:var(--text-dim);">${_n} daily snapshot${_n===1?'':'s'} (${dateFrom}${_n>1?` → ${dateTo}`:''}) — quality + income trend</span>
            </summary>
            <div style="margin-top:10px;display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:8px;">
                ${_card('Quality total', qLast !== null ? _fmtUsd(qLast) : 'n/a', _deltaLabel(qDelta, false), _spark(qVals, '#3fb950'))}
                ${_card('Avg weekly Θ', wLast !== null ? _fmtUsd(wLast) : 'n/a', _deltaLabel(wDelta, false), _spark(wVals, '#58a6ff'))}
                ${_card('DD penalty', ddLast !== null ? _fmtUsd(ddLast) : 'n/a', _deltaLabel(ddDelta, true), _spark(ddVals, '#f0883e'))}
                ${_card('Income gap', igLast !== null ? _fmtUsd(igLast) : 'n/a', _deltaLabel(igDelta, true), _spark(igVals, '#bc8cff'))}
            </div>
        </details>`;
    }

    const fundamentalsHtml = `
        <details style="margin-bottom:12px;padding:10px 14px;background:rgba(63,185,80,0.06);border-radius:8px;border:1px solid rgba(63,185,80,0.25);">
            <summary style="cursor:pointer;display:flex;align-items:center;gap:10px;list-style:none;">
                <span style="display:inline-block;padding:2px 10px;border-radius:4px;font-weight:700;font-size:0.85rem;color:#fff;background:#3fb950;">MODEL FUNDAMENTALS</span>
                <span style="font-size:0.78rem;color:var(--text-dim);">click to expand — 20 NG factor curves + IC weights</span>
            </summary>
            ${icEntries.length > 0 ? `
            <div style="margin-top:10px;padding:8px 10px;background:rgba(88,166,255,0.04);border-radius:6px;">
                <div style="font-size:0.78rem;color:var(--text-dim);margin-bottom:6px;">
                    <strong style="color:var(--text);">IC weights</strong> — factor influence on composite (Spearman corr with 3m fwd NG return; green ≥0.10, orange ≥0.05)
                </div>
                ${icRows}
            </div>` : ''}
            <div style="margin-top:10px;display:grid;grid-template-columns:1fr;gap:10px;">
                <div>
                    <div style="font-size:0.75rem;color:var(--text-dim);margin-bottom:4px;">Factor curves (each panel: blue=raw, orange=z; title color = freshness)</div>
                    <img src="/api/factor_curves.png?cb=${cb}" alt="NG factor curves"
                         style="width:100%;max-width:100%;border-radius:6px;border:1px solid var(--border);background:#1a1a2e;"
                         onerror="this.parentElement.innerHTML='<div style=\\'color:var(--text-dim);font-style:italic\\'>chart not generated yet — run ng_daily_forecast.py to produce ng_factor_curves.png</div>'" />
                </div>
                <div>
                    <div style="font-size:0.75rem;color:var(--text-dim);margin-bottom:4px;">Forecast dashboard (8-panel: composite, factors, weather, etc.)</div>
                    <img src="/api/forecast_chart.png?cb=${cb}" alt="NG forecast dashboard"
                         style="width:100%;max-width:100%;border-radius:6px;border:1px solid var(--border);background:#1a1a2e;" />
                </div>
                <div>
                    <div style="font-size:0.75rem;color:var(--text-dim);margin-bottom:4px;">Probability cone (forward NG price distribution)</div>
                    <img src="/api/probability_cone.png?cb=${cb}" alt="NG probability cone"
                         style="width:100%;max-width:100%;border-radius:6px;border:1px solid var(--border);background:#1a1a2e;" />
                </div>
            </div>
        </details>`;

    // Beam diagnostic panel (cycle 54): show chosen path + runners-up the
    // beam considered. Each runner-up notes the component dimension where
    // it lost to the winner.
    const _beam = (pm && Array.isArray(pm.beam_diagnostic)) ? pm.beam_diagnostic : [];
    let beamHtml = '';
    if (_beam.length > 0) {
        const _dimLabel = {
            income_gap: 'income',
            dd_penalty: 'drawdown',
            delta_gap: 'delta',
            smoothness: 'smoothness',
            tail_hedge: 'tail hedge',
            pillar_drift: 'pillar',
        };
        const _fmtUsd2 = (v) => {
            if (typeof v !== 'number' || !isFinite(v)) return 'n/a';
            const a = Math.abs(v);
            const s = v < 0 ? '-' : '';
            if (a >= 1000) return s + '$' + (a/1000).toFixed(1) + 'k';
            return s + '$' + a.toFixed(0);
        };
        const _pathRows = _beam.map(p => {
            const tradeList = (p.trades || []).map(t => {
                const tagBg = t.type === 'ROLL' ? '#58a6ff' :
                              t.type === 'CLOSE' ? '#f0883e' :
                              t.type === 'TAKE PROFIT' ? '#3fb950' :
                              t.type === 'LET EXPIRE' ? '#8b949e' :
                              t.type === 'COVERED CALL' ? '#bc8cff' :
                              t.type === 'BUY PUT' ? '#d2a8ff' :
                              t.type === 'ASSIGNMENT' ? '#79c0ff' : '#8b949e';
                const dv = typeof t.dollar_value === 'number' ? t.dollar_value : 0;
                const dvColor = dv >= 0 ? 'var(--green)' : 'var(--red,#f85149)';
                const tk = (t.target_strike !== null && t.target_strike !== undefined) ? `$${t.target_strike}` : '';
                const te = t.target_exp || '';
                return `<div style="display:flex;align-items:center;gap:6px;font-size:0.72rem;padding:1px 0;">
                    <span style="display:inline-block;padding:1px 6px;border-radius:3px;background:${tagBg};color:#fff;font-size:0.65rem;min-width:80px;text-align:center;">${t.type || '?'}</span>
                    <span style="color:var(--text-dim);min-width:140px;">${te} ${tk}</span>
                    <span style="font-family:monospace;color:${dvColor};">${_fmtUsd2(dv)}</span>
                </div>`;
            }).join('');
            const qd = p.quality_delta || 0;
            const qdColor = qd >= 0 ? 'var(--green)' : 'var(--red,#f85149)';
            const headerBg = p.is_winner ? 'rgba(63,185,80,0.12)' : 'rgba(139,148,158,0.06)';
            const headerBorder = p.is_winner ? 'rgba(63,185,80,0.4)' : 'rgba(139,148,158,0.2)';
            const winnerTag = p.is_winner
                ? `<span style="display:inline-block;padding:1px 8px;border-radius:3px;background:#3fb950;color:#fff;font-size:0.7rem;font-weight:600;">CHOSEN</span>`
                : `<span style="display:inline-block;padding:1px 8px;border-radius:3px;background:rgba(139,148,158,0.3);color:var(--text-dim);font-size:0.7rem;">RANK #${(p.rank||0)+1}</span>`;
            const lossNote = (!p.is_winner && p.losing_dim)
                ? `<span style="font-size:0.72rem;color:var(--text-dim);">lost on <strong style="color:var(--orange,#f0883e);">${_dimLabel[p.losing_dim] || p.losing_dim}</strong> (-${_fmtUsd2(p.losing_gap || 0)} vs chosen)</span>`
                : '';
            return `
            <div style="margin-top:8px;padding:8px 10px;background:${headerBg};border:1px solid ${headerBorder};border-radius:6px;">
                <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
                    ${winnerTag}
                    <span style="font-size:0.78rem;color:var(--text-dim);">${p.trade_count || 0} trade${(p.trade_count||0)===1?'':'s'} ·</span>
                    <span style="font-family:monospace;color:${qdColor};font-size:0.85rem;">quality Δ ${_fmtUsd2(qd)}</span>
                    ${lossNote}
                </div>
                ${tradeList ? `<div style="margin-top:6px;padding-top:6px;border-top:1px dashed ${headerBorder};">${tradeList}</div>` : '<div style="margin-top:4px;font-size:0.7rem;color:var(--text-dim);font-style:italic;">no trades (stand pat)</div>'}
            </div>`;
        }).join('');
        beamHtml = `
        <details style="margin-bottom:12px;padding:10px 14px;background:rgba(188,140,255,0.05);border-radius:8px;border:1px solid rgba(188,140,255,0.22);">
            <summary style="cursor:pointer;display:flex;align-items:center;gap:10px;list-style:none;">
                <span style="display:inline-block;padding:2px 10px;border-radius:4px;font-weight:700;font-size:0.85rem;color:#fff;background:#bc8cff;">BEAM DIAGNOSTIC</span>
                <span style="font-size:0.78rem;color:var(--text-dim);">${_beam.length} path${_beam.length===1?'':'s'} considered — chosen + runners-up with losing dimension</span>
            </summary>
            ${_pathRows}
        </details>`;
    }

    // Near-misses panel (cycle 59): when the beam returned no trades,
    // show the operator the top candidates that were considered but
    // didn't make the cut, along with each candidate's reject reason.
    // Helps decide whether to wait or override.
    const _near = (pm && Array.isArray(pm.near_misses)) ? pm.near_misses : [];
    let nearMissHtml = '';
    if (_near.length > 0) {
        const rows = _near.map(n => {
            const tagBg = n.type === 'ROLL' ? '#58a6ff' :
                          n.type === 'CLOSE' ? '#f0883e' :
                          n.type === 'TAKE PROFIT' ? '#3fb950' :
                          n.type === 'LET EXPIRE' ? '#8b949e' :
                          n.type === 'COVERED CALL' ? '#bc8cff' :
                          n.type === 'BUY PUT' ? '#d2a8ff' :
                          n.type === 'ASSIGNMENT' ? '#79c0ff' :
                          n.type === 'OPEN' ? '#3fb950' :
                          n.type === 'ADD' ? '#3fb950' : '#8b949e';
            const tk = (n.target_strike !== null && n.target_strike !== undefined) ? `$${n.target_strike}` : '';
            const qd = (typeof n.quality_delta === 'number') ? n.quality_delta : null;
            const qdColor = qd === null ? 'var(--text-dim)' :
                            qd > 0 ? 'var(--green)' :
                            qd < 0 ? 'var(--red)' : 'var(--text-dim)';
            const qdStr = qd === null ? '—' :
                          qd >= 0 ? `+$${Math.abs(qd).toLocaleString()}` : `-$${Math.abs(qd).toLocaleString()}`;
            return `<div style="display:grid;grid-template-columns:90px 1fr 60px 80px 1fr;align-items:center;gap:8px;padding:2px 0;font-size:0.72rem;">
                <span style="display:inline-block;padding:1px 6px;border-radius:3px;background:${tagBg};color:#fff;font-size:0.65rem;text-align:center;">${n.type || '?'}</span>
                <span style="color:var(--text-dim);">${n.target_exp || ''} ${tk}</span>
                <span style="font-family:monospace;text-align:right;color:var(--text);">${(n.score || 0).toFixed(1)}</span>
                <span style="font-family:monospace;text-align:right;color:${qdColor};">${qdStr}</span>
                <span style="color:var(--text-dim);font-size:0.66rem;font-style:italic;">${n.reject_reason || ''}</span>
            </div>`;
        }).join('');
        nearMissHtml = `
        <details style="margin-bottom:12px;padding:10px 14px;background:rgba(139,148,158,0.06);border-radius:8px;border:1px solid rgba(139,148,158,0.25);" open>
            <summary style="cursor:pointer;display:flex;align-items:center;gap:10px;list-style:none;">
                <span style="display:inline-block;padding:2px 10px;border-radius:4px;font-weight:700;font-size:0.85rem;color:#fff;background:#8b949e;">NO-ACTION CYCLE</span>
                <span style="font-size:0.78rem;color:var(--text-dim);">optimizer found nothing — top ${_near.length} near-misses with reject reasons + actual quality delta</span>
            </summary>
            <div style="margin-top:8px;padding-top:6px;border-top:1px dashed rgba(139,148,158,0.2);">
                <div style="display:grid;grid-template-columns:90px 1fr 60px 80px 1fr;gap:8px;padding:0 0 4px;font-size:0.6rem;color:var(--text-dim);text-transform:uppercase;letter-spacing:0.04em;">
                    <span>type</span>
                    <span>target</span>
                    <span style="text-align:right;">score</span>
                    <span style="text-align:right;">qΔ if taken</span>
                    <span>reject reason</span>
                </div>
                ${rows}
            </div>
        </details>`;
    }

    // Risk concentration by expiry (cycle 62): when gamma is the DD
    // driver (cycle 56-57), operator wants to know which expiries hold
    // the most negative gamma so they can plan targeted closes/rolls.
    const _rbe = (pm && Array.isArray(pm.risk_by_expiry)) ? pm.risk_by_expiry : [];
    let riskByExpiryHtml = '';
    if (_rbe.length > 0) {
        const maxAbsG = Math.max(1, ...(_rbe.map(r => Math.abs(r.gamma || 0))));
        const rows = _rbe.map(r => {
            const wG = Math.min(100, (Math.abs(r.gamma || 0) / maxAbsG) * 100);
            const gColor = (r.gamma || 0) < 0 ? 'rgba(248,81,73,0.7)' : 'rgba(63,185,80,0.7)';
            const dStr = `${r.delta>=0?'+':''}${(r.delta||0).toLocaleString()}`;
            const dColor = (r.delta||0) >= 0 ? 'var(--green)' : 'var(--red)';
            return `<div style="display:grid;grid-template-columns:90px 60px 1fr 90px 80px;align-items:center;gap:8px;padding:2px 0;font-size:0.72rem;">
                <span style="color:var(--text);font-family:monospace;">${r.expiry || ''}</span>
                <span style="color:var(--text-dim);text-align:right;">${r.contracts}×</span>
                <div style="position:relative;height:10px;background:rgba(139,148,158,0.08);border-radius:3px;overflow:hidden;">
                    <div style="position:absolute;left:0;top:0;height:100%;width:${wG}%;background:${gColor};"></div>
                </div>
                <span style="font-family:monospace;text-align:right;color:${(r.gamma||0)<0?'var(--red)':'var(--green)'};">Γ ${(r.gamma||0).toLocaleString()}</span>
                <span style="font-family:monospace;text-align:right;color:${dColor};font-size:0.66rem;">Δ ${dStr}</span>
            </div>`;
        }).join('');
        riskByExpiryHtml = `
        <details style="margin-bottom:12px;padding:10px 14px;background:rgba(240,136,62,0.05);border-radius:8px;border:1px solid rgba(240,136,62,0.22);">
            <summary style="cursor:pointer;display:flex;align-items:center;gap:10px;list-style:none;">
                <span style="display:inline-block;padding:2px 10px;border-radius:4px;font-weight:700;font-size:0.85rem;color:#fff;background:#f0883e;">RISK BY EXPIRY</span>
                <span style="font-size:0.78rem;color:var(--text-dim);">top ${_rbe.length} expiries by |Γ| — close the worst to shrink gamma_convexity DD driver</span>
            </summary>
            <div style="margin-top:8px;padding-top:6px;border-top:1px dashed rgba(240,136,62,0.2);">
                <div style="display:grid;grid-template-columns:90px 60px 1fr 90px 80px;gap:8px;padding:0 0 4px;font-size:0.6rem;color:var(--text-dim);text-transform:uppercase;letter-spacing:0.04em;">
                    <span>expiry</span>
                    <span style="text-align:right;">qty</span>
                    <span>|Γ| relative</span>
                    <span style="text-align:right;">gamma</span>
                    <span style="text-align:right;">delta</span>
                </div>
                ${rows}
            </div>
        </details>`;
    }

    // Kelly utilization gauge (cycle 64): show where the wheel sits
    // relative to the ¼-Kelly 50% soft trigger and the 95% hard cap.
    const _kelly = (pm && pm.kelly_utilization) ? pm.kelly_utilization : null;
    let kellyHtml = '';
    if (_kelly && _kelly.capital > 0) {
        const u = _kelly.utilization || 0;
        const pct = u * 100;
        const soft = (_kelly.soft_trigger || 0.5) * 100;
        const hard = (_kelly.hard_cap || 0.95) * 100;
        const overK = _kelly.over_kelly_mult || 0;
        const barColor = u >= 0.95 ? '#f85149' :
                          u >= 0.75 ? '#f0883e' :
                          u >= 0.50 ? '#d29922' : '#3fb950';
        const statusLabel = u >= 0.95 ? 'AT HARD CAP — new short-put adds vetoed'
                          : u >= 0.75 ? 'WELL OVER soft trigger — adds heavily penalized'
                          : u >= 0.50 ? 'Over soft trigger — incremental adds penalized'
                          : 'Within ¼-Kelly budget — full deployment OK';
        const widthPct = Math.min(100, pct);
        kellyHtml = `
        <details style="margin-bottom:12px;padding:10px 14px;background:rgba(210,153,34,0.05);border-radius:8px;border:1px solid rgba(210,153,34,0.22);" open>
            <summary style="cursor:pointer;display:flex;align-items:center;gap:10px;list-style:none;">
                <span style="display:inline-block;padding:2px 10px;border-radius:4px;font-weight:700;font-size:0.85rem;color:#fff;background:#d29922;">KELLY UTILIZATION</span>
                <span style="font-size:0.78rem;color:var(--text-dim);">put collateral / capital — ¼-Kelly sizing principle</span>
                <span style="margin-left:auto;font-family:monospace;font-size:0.85rem;font-weight:700;color:${barColor};">${pct.toFixed(1)}%</span>
            </summary>
            <div style="margin-top:8px;padding-top:6px;border-top:1px dashed rgba(210,153,34,0.2);">
                <div style="position:relative;height:18px;background:rgba(139,148,158,0.1);border-radius:4px;overflow:hidden;margin-bottom:6px;">
                    <div style="position:absolute;left:0;top:0;height:100%;width:${widthPct}%;background:${barColor};opacity:0.7;"></div>
                    <div style="position:absolute;left:${soft}%;top:0;bottom:0;width:1px;background:#d29922;"></div>
                    <div style="position:absolute;left:${hard}%;top:0;bottom:0;width:2px;background:#f85149;"></div>
                    <div style="position:absolute;left:${soft}%;top:-1px;font-size:0.55rem;color:#d29922;padding-left:3px;">soft ${soft.toFixed(0)}%</div>
                    <div style="position:absolute;left:${hard}%;top:-1px;font-size:0.55rem;color:#f85149;padding-left:3px;transform:translateX(-100%);">hard ${hard.toFixed(0)}%</div>
                </div>
                <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;font-size:0.7rem;margin-bottom:4px;">
                    <div>
                        <div style="color:var(--text-dim);font-size:0.62rem;text-transform:uppercase;">Put collateral</div>
                        <div style="font-family:monospace;color:var(--text);">$${(_kelly.put_collateral || 0).toLocaleString()}</div>
                    </div>
                    <div>
                        <div style="color:var(--text-dim);font-size:0.62rem;text-transform:uppercase;">Capital (margin NLV)</div>
                        <div style="font-family:monospace;color:var(--text);">$${(_kelly.capital || 0).toLocaleString()}</div>
                    </div>
                    <div>
                        <div style="color:var(--text-dim);font-size:0.62rem;text-transform:uppercase;">Over-Kelly mult</div>
                        <div style="font-family:monospace;color:${overK > 0 ? 'var(--orange)' : 'var(--green)'};">${overK.toFixed(2)}×</div>
                    </div>
                </div>
                <div style="font-size:0.7rem;color:${barColor};font-style:italic;">${statusLabel}</div>
            </div>
        </details>`;
    }

    // Hidden wins (cycle 66): seed candidates with positive quality_delta
    // that the heuristic score-ranking may have left out of the beam's
    // top-N. Only renders when the list is non-empty.
    const _hw = (pm && Array.isArray(pm.hidden_wins)) ? pm.hidden_wins : [];
    let hiddenWinsHtml = '';
    if (_hw.length > 0) {
        const rows = _hw.map(h => {
            const tagBg = h.type === 'ROLL' ? '#58a6ff' :
                          h.type === 'CLOSE' ? '#f0883e' :
                          h.type === 'TAKE PROFIT' ? '#3fb950' :
                          h.type === 'LET EXPIRE' ? '#8b949e' :
                          h.type === 'COVERED CALL' ? '#bc8cff' :
                          h.type === 'BUY PUT' ? '#d2a8ff' :
                          h.type === 'ASSIGNMENT' ? '#79c0ff' :
                          h.type === 'OPEN' ? '#3fb950' :
                          h.type === 'ADD' ? '#3fb950' : '#8b949e';
            const tk = (h.target_strike !== null && h.target_strike !== undefined) ? `$${h.target_strike}` : '';
            const qd = h.quality_delta || 0;
            const vsBeam = (typeof h.vs_beam_chain === 'number') ? h.vs_beam_chain : qd;
            const vsBeamColor = vsBeam > 0 ? 'var(--green)' : 'var(--text-dim)';
            return `<div style="display:grid;grid-template-columns:90px 1fr 100px 100px 1fr;align-items:center;gap:8px;padding:2px 0;font-size:0.72rem;">
                <span style="display:inline-block;padding:1px 6px;border-radius:3px;background:${tagBg};color:#fff;font-size:0.65rem;text-align:center;">${h.type || '?'}</span>
                <span style="color:var(--text-dim);">${h.target_exp || ''} ${tk}</span>
                <span style="font-family:monospace;text-align:right;color:var(--green);font-weight:700;">+$${qd.toLocaleString()}</span>
                <span style="font-family:monospace;text-align:right;color:${vsBeamColor};font-size:0.66rem;">${vsBeam>=0?'+':''}$${Math.abs(vsBeam).toLocaleString()}</span>
                <span style="color:var(--text-dim);font-size:0.66rem;font-style:italic;">${h.action || ''}</span>
            </div>`;
        }).join('');
        hiddenWinsHtml = `
        <details style="margin-bottom:12px;padding:10px 14px;background:rgba(63,185,80,0.05);border-radius:8px;border:1px solid rgba(63,185,80,0.3);" open>
            <summary style="cursor:pointer;display:flex;align-items:center;gap:10px;list-style:none;">
                <span style="display:inline-block;padding:2px 10px;border-radius:4px;font-weight:700;font-size:0.85rem;color:#fff;background:#3fb950;">HIDDEN WINS</span>
                <span style="font-size:0.78rem;color:var(--text-dim);">${_hw.length} single trade${_hw.length===1?'':'s'} with +qΔ exceeding the beam chain — operator override candidates</span>
            </summary>
            <div style="margin-top:8px;padding-top:6px;border-top:1px dashed rgba(63,185,80,0.25);">
                <div style="display:grid;grid-template-columns:90px 1fr 100px 100px 1fr;gap:8px;padding:0 0 4px;font-size:0.6rem;color:var(--text-dim);text-transform:uppercase;letter-spacing:0.04em;">
                    <span>type</span>
                    <span>target</span>
                    <span style="text-align:right;">qΔ standalone</span>
                    <span style="text-align:right;">vs beam chain</span>
                    <span>action</span>
                </div>
                ${rows}
            </div>
        </details>`;
    }

    // Cycle 91: OPEN-rejected commentary. Explain why no new short put
    // is proposed when income is below target and Δ is under target —
    // because adding short gamma would worsen dd_penalty more than the
    // income gain helps.
    const _oc = (pm && pm.open_commentary) ? pm.open_commentary : null;
    let openCommentaryHtml = '';
    if (_oc && !_oc.in_beam) {
        const cd = _oc.components_delta || {};
        const qd = _oc.best_qdelta || 0;
        const dollar = (v) => `${v < 0 ? '-' : '+'}$${Math.abs(Math.round(v)).toLocaleString()}`;
        const verdict = qd < 0
            ? `Net quality <strong style="color:var(--red,#f85149);">${dollar(qd)}</strong> — adding a short put would <strong>worsen</strong> portfolio quality despite higher income.`
            : `Best OPEN scores ${dollar(qd)}, below other actions the beam preferred.`;
        // Identify the dominant negative component so the message is specific
        const negComps = Object.entries(cd).filter(([k,v]) => v < 0).sort((a,b) => a[1]-b[1]);
        let reasonLine = '';
        if (negComps.length > 0) {
            const [worstK, worstV] = negComps[0];
            const labels = {
                income_gap: 'income gap',
                dd_penalty: 'drawdown penalty',
                delta_gap: 'delta gap',
                tail_hedge: 'tail-hedge floor',
                pillar_drift: 'pillar drift',
            };
            reasonLine = `Worst component impact: <strong style="color:var(--red,#f85149);">${labels[worstK] || worstK} ${dollar(worstV)}</strong>.`;
        }
        openCommentaryHtml = `
        <details style="margin-bottom:12px;padding:10px 14px;background:rgba(210,153,34,0.06);border-radius:8px;border:1px solid rgba(210,153,34,0.25);" open>
            <summary style="cursor:pointer;display:flex;align-items:center;gap:10px;list-style:none;">
                <span style="display:inline-block;padding:2px 10px;border-radius:4px;font-weight:700;font-size:0.85rem;color:#fff;background:#d29922;">WHY NO NEW PUTS</span>
                <span style="font-size:0.78rem;color:var(--text-dim);">${_oc.open_candidate_count} OPEN candidate${_oc.open_candidate_count===1?'':'s'} considered, none selected</span>
            </summary>
            <div style="margin-top:8px;padding-top:6px;border-top:1px dashed rgba(210,153,34,0.2);font-size:0.78rem;line-height:1.5;">
                <div style="margin-bottom:4px;"><strong>Best candidate</strong>: ${_oc.best_action || '(none)'}.</div>
                <div style="margin-bottom:4px;">${verdict}</div>
                <div style="margin-bottom:4px;">${reasonLine}</div>
                <div style="font-size:0.7rem;color:var(--text-dim);font-style:italic;">
                    Component deltas if executed:
                    income_gap ${dollar(cd.income_gap||0)},
                    dd_penalty ${dollar(cd.dd_penalty||0)},
                    delta_gap ${dollar(cd.delta_gap||0)},
                    pillar ${dollar(cd.pillar_drift||0)}.
                </div>
                <div style="margin-top:6px;font-size:0.72rem;color:var(--text);">
                    Income is under target, but the system gates new short puts when
                    the gamma exposure they add worsens projected drawdown more than
                    the premium helps. Address dd_penalty first (close more shorts,
                    add long puts, or accept higher DD risk) before reloading income.
                </div>
            </div>
        </details>`;
    }

    // Cycle 140 BOXX cash-park card
    const _cp = pm && pm.cash_park_suggestion;
    let cashParkHtml = '';
    if (_cp && _cp.capital_nlv > 0) {
        const idle = _cp.idle_cash || 0;
        const cap = _cp.capital_nlv || 1;
        const idle_pct = (idle / cap * 100);
        const yieldPerDay = (idle * (_cp.boxx_daily_yield_pct || 0) / 100);
        const yieldPerYear = (idle * (_cp.boxx_apr_pct || 0) / 100);
        const usd = (v) => `$${Math.round(v).toLocaleString()}`;
        cashParkHtml = `
        <details style="margin-bottom:12px;padding:10px 14px;background:rgba(63,185,80,0.05);border-radius:8px;border:1px solid rgba(63,185,80,0.22);" open>
            <summary style="cursor:pointer;display:flex;align-items:center;gap:10px;list-style:none;">
                <span style="display:inline-block;padding:2px 10px;border-radius:4px;font-weight:700;font-size:0.85rem;color:#fff;background:#3fb950;">CASH PARK · BOXX</span>
                <span style="font-size:0.78rem;color:var(--text-dim);">park idle cash at ~${_cp.boxx_apr_pct}% APR — better than 0%</span>
                <span style="margin-left:auto;font-family:monospace;font-size:0.85rem;font-weight:700;color:#3fb950;">${usd(idle)} idle</span>
            </summary>
            <div style="margin-top:8px;padding-top:6px;border-top:1px dashed rgba(63,185,80,0.2);">
                <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:8px;font-size:0.72rem;">
                    <div>
                        <div style="color:var(--text-dim);font-size:0.62rem;text-transform:uppercase;">Capital (NLV)</div>
                        <div style="font-family:monospace;color:var(--text);">${usd(cap)}</div>
                    </div>
                    <div>
                        <div style="color:var(--text-dim);font-size:0.62rem;text-transform:uppercase;">Put Collateral</div>
                        <div style="font-family:monospace;color:var(--text);">${usd(_cp.put_collateral || 0)}</div>
                    </div>
                    <div>
                        <div style="color:var(--text-dim);font-size:0.62rem;text-transform:uppercase;">UNG Shares</div>
                        <div style="font-family:monospace;color:var(--text);">${usd(_cp.share_value || 0)}</div>
                    </div>
                    <div>
                        <div style="color:var(--text-dim);font-size:0.62rem;text-transform:uppercase;">BOXX Held ✓</div>
                        <div style="font-family:monospace;color:#3fb950;font-weight:600;">${usd(_cp.boxx_held || 0)}</div>
                    </div>
                    <div>
                        <div style="color:var(--text-dim);font-size:0.62rem;text-transform:uppercase;">Other Cash-Park</div>
                        <div style="font-family:monospace;color:var(--text);">${usd((_cp.other_total || 0) - (_cp.boxx_held || 0))}</div>
                    </div>
                    <div>
                        <div style="color:var(--text-dim);font-size:0.62rem;text-transform:uppercase;">Remaining Idle</div>
                        <div style="font-family:monospace;color:${idle > 1000 ? '#3fb950' : 'var(--text-dim)'};font-weight:600;">${usd(idle)}</div>
                    </div>
                    <div>
                        <div style="color:var(--text-dim);font-size:0.62rem;text-transform:uppercase;">BOXX yield (annual on idle)</div>
                        <div style="font-family:monospace;color:#3fb950;">${usd(yieldPerYear)}/yr · ${usd(yieldPerDay)}/d</div>
                    </div>
                </div>
                <div style="font-size:0.7rem;color:var(--text-dim);margin-top:8px;font-style:italic;">
                    ${_cp.note || ''} Margin: short put = cash-secured (100% strike), BOXX = 50% requirement (cash-only here).
                </div>
                ${Object.keys(_cp.other_holdings || {}).length > 0 ? `
                <div style="font-size:0.66rem;color:var(--text-dim);margin-top:6px;padding-top:4px;border-top:1px dashed rgba(63,185,80,0.15);">
                    Non-UNG holdings: ${Object.entries(_cp.other_holdings).map(([sym,v]) => `${sym} ${v.qty.toLocaleString()} sh · ${usd(v.market_value)}`).join(' · ')}
                </div>` : ''}
            </div>
        </details>`;
    }

    container.innerHTML = tsHtml + metricsHtml + kellyHtml + cashParkHtml + progressHtml + fundamentalsHtml + beamHtml + nearMissHtml + hiddenWinsHtml + openCommentaryHtml + riskByExpiryHtml + recs.map((r, i) => {
        const impacts = [
            r.theta_impact ? `\u0398 ${fmtImpact(r.theta_impact, '$', '/d')}` : '',
            r.delta_impact ? `\u0394 ${fmtImpact(r.delta_impact, '', '')}` : '',
            r.gamma_impact ? `\u0393 ${fmtImpact(r.gamma_impact, '', '')}` : '',
            r.vega_impact ? `V ${fmtImpact(r.vega_impact, '', '')}` : '',
            r.smoothness_impact ? `Sm ${fmtImpact(r.smoothness_impact, '', '%')}` : '',
        ].filter(x => x).join(' \u00a0 ');

        // Stress test display
        const stressHtml = (r.stress_crash_delta !== undefined)
            ? `<div style="font-size:0.75rem;color:var(--text-dim);margin-top:2px;">
                 Stress: crash\u2192\u0394${fmtImpact(r.stress_crash_change, '', '')}
                 rally\u2192\u0394${fmtImpact(r.stress_rally_change, '', '')}
               </div>`
            : '';

        // Score breakdown bar
        const bd = r.score_breakdown || {};
        const scoreKeys = [
            ['delta', '\u0394 target', 'var(--cyan)'],
            ['economic', 'Econ', 'var(--green)'],
            ['theta_rate', '\u0398 rate', '#58a6ff'],
            ['drain', 'Drain', '#f0883e'],
            ['concentration', 'Conc', '#bc8cff'],
            ['type_bonus', 'Type', '#8b949e'],
            ['recovery', 'Recovery', '#d2a8ff'],
            ['waterfall', 'Waterfall', '#3fb950'],
            ['assignment_sim', 'Assign sim', '#79c0ff'],
            ['gamma_regime', 'Gamma', '#d29922'],
            ['kelly', 'Kelly', '#58a6ff'],
            ['correlation', 'Corr', '#ff7b72'],
            ['thesis', 'Thesis', '#a5d6ff'],
            ['scenario', 'E[P/L]', '#ffa657'],
            ['crash_recovery', 'Crash rec.', 'var(--red)'],
            ['crash_penalty', 'Crash', '#f85149'],
        ];
        const bdParts = scoreKeys
            .filter(([k]) => bd[k] !== undefined && Math.abs(bd[k]) >= 0.1)
            .map(([k, label, color]) => {
                const v = bd[k];
                const sign = v > 0 ? '+' : '';
                const c = v >= 0 ? color : 'var(--red)';
                return `<span style="color:${c};font-size:0.72rem;white-space:nowrap;">${label} ${sign}${v.toFixed(1)}</span>`;
            }).join(' <span style="color:var(--border)">|</span> ');

        const pItmHtml = r.p_itm !== undefined
            ? `<span style="font-size:0.72rem;color:var(--text-dim);margin-left:8px;">P(ITM)=${r.p_itm}%</span>`
            : '';

        const breakdownHtml = bdParts
            ? `<div style="margin-top:3px;display:flex;gap:4px;flex-wrap:wrap;align-items:center;">${bdParts}${pItmHtml}</div>`
            : '';

        // Cycle 167: render per-rec quality-component breakdown alongside
        // the heuristic score breakdown. Shows WHICH component(s) drove
        // the qΔ — operator can spot when a high qΔ is dominated by a
        // smoothness/income_gap CLIFF (from active-bucket averaging in
        // avg_weekly_theta) vs real income contribution.
        // Highlight (orange tinted) any component with |delta| > $100
        // since those typically indicate a metric cliff was crossed.
        const cdHtml = (() => {
            const cd = r.components_delta;
            if (!cd || typeof cd !== 'object') return '';
            const nonzero = Object.entries(cd)
                .filter(([k, v]) => v && Math.abs(v) >= 1)
                .sort(([, a], [, b]) => Math.abs(b) - Math.abs(a));
            if (!nonzero.length) return '';
            const parts = nonzero.map(([k, v]) => {
                const sign = v > 0 ? '+' : '';
                const abs = Math.abs(v);
                const cliff = abs >= 100;
                const color = cliff
                    ? 'var(--orange)'
                    : v > 0 ? 'var(--green)' : 'var(--red)';
                const weight = cliff ? '700' : '400';
                const tag = cliff ? ' ⚠️' : '';
                return `<span style="color:${color};font-weight:${weight};font-size:0.7rem;">${k}: ${sign}${v}${tag}</span>`;
            }).join(' · ');
            return `<div style="margin-top:2px;padding:3px 6px;background:rgba(139,148,158,0.05);border-radius:4px;font-family:monospace;">${parts}</div>`;
        })();

        return `
        <div class="rec-card" style="border-left: 3px solid ${typeColors[r.type] || '#8b949e'}">
            <div class="rec-header">
                <span class="rec-rank">#${i+1}</span>
                <span class="rec-type-badge" style="background:${typeColors[r.type]}20;color:${typeColors[r.type]}">${r.type}</span>
                <span class="rec-urgency-badge ${r.urgency}">${r.urgency.toUpperCase()}</span>
                ${(() => {
                    // Cycle 148: stability badge — count/window from cycle 147 backend.
                    const sc = r.stability_count;
                    const sw = r.stability_window;
                    if (sc === undefined || sw === undefined || sw === 0) return '';
                    // Class: stable=≥3 of available history; recent=1-2; flicker=0 (first sighting)
                    const cls = sc >= 3 ? 'stable' : sc >= 1 ? 'recent' : 'flicker';
                    const title = `Appeared in ${sc} of last ${sw} cycle${sw===1?'':'s'}`;
                    return `<span class="rec-stability-badge ${cls}" title="${title}">${sc}/${sw}</span>`;
                })()}
                <span style="font-size:0.75rem;color:var(--text-dim);margin-left:auto;">
                    ${(r.dollar_value !== undefined && r.dollar_value !== 0) ?
                        `<strong style="color:${r.dollar_value > 0 ? 'var(--green)' : 'var(--red)'}">${r.dollar_value > 0 ? '+' : ''}$${Math.round(r.dollar_value)}</strong> · ` : ''}
                    score: ${r.score.toFixed(1)}
                </span>
            </div>
            <div class="rec-action">${r.action}</div>
            <div style="font-size:0.82rem;margin:4px 0;display:flex;gap:12px;flex-wrap:wrap;">${impacts}</div>
            ${breakdownHtml}
            ${cdHtml}
            ${stressHtml}
            ${(() => {
                const lq = r.liquidity;
                if (!lq) return '';
                const cdLabel = lq.credit_debit >= 0 ? 'credit' : 'debit';
                const cdColor = lq.credit_debit >= 0 ? 'var(--green)' : 'var(--red)';
                const fmtSpread = (bid, ask, pct) => {
                    const color = pct > 20 ? 'var(--red)' : pct > 10 ? 'var(--orange)' : 'var(--green)';
                    return bid > 0 ? '<span style="color:' + color + '">$' + bid + '/$' + ask + ' (' + pct + '%)</span>' : '<span style="color:var(--text-dim)">n/a</span>';
                };
                const oiColor = lq.oi_usage_pct > 30 ? 'var(--red)' : lq.oi_usage_pct > 15 ? 'var(--orange)' : 'var(--green)';
                const isRoll = lq.source_oi !== undefined;
                let html = '<div style="font-size:0.75rem;margin-top:3px;padding:4px 8px;background:rgba(139,148,158,0.08);border-radius:4px;">';
                html += '<div style="display:flex;gap:10px;flex-wrap:wrap;align-items:center;">';
                html += '<span style="color:var(--text-dim);">$' + Math.abs(lq.credit_debit) + ' <span style="color:' + cdColor + '">' + cdLabel + '</span></span>';
                html += '<span style="color:var(--text-dim);">notional $' + lq.notional + '</span>';
                html += '<span style="color:var(--text-dim);">~$' + lq.friction_est + ' total fric</span>';
                html += '</div>';
                if (isRoll) {
                    // BS fair vs market mid coloring
                    const fmtFair = (mid, fair, dev) => {
                        const devPct = fair > 0 ? (dev / fair * 100) : 0;
                        const color = Math.abs(devPct) < 5 ? 'var(--green)' : Math.abs(devPct) < 15 ? 'var(--orange)' : 'var(--red)';
                        const sign = dev >= 0 ? '+' : '';
                        return '<span style="color:var(--text-dim);">mid $' + mid + ' vs fair $' + fair + ' <span style="color:' + color + '">(' + sign + (devPct).toFixed(0) + '%)</span></span>';
                    };
                    html += '<div style="display:grid;grid-template-columns:auto 1fr;gap:6px 12px;margin-top:4px;font-size:0.7rem;">';
                    html += '<span style="color:var(--text-dim);font-weight:600;">FROM (BTC):</span>';
                    html += '<span>OI=' + lq.source_oi + ' | spread ' + fmtSpread(lq.source_bid, lq.source_ask, lq.source_spread_pct) + ' | ' + fmtFair(lq.source_mid, lq.source_fair, lq.source_mid_vs_fair) + ' | ~$' + lq.source_friction + '</span>';
                    html += '<span style="color:var(--text-dim);font-weight:600;">TO (STO):</span>';
                    html += '<span>OI=' + lq.oi + ' <span style="color:' + oiColor + '">(' + lq.oi_usage_pct + '% used)</span> | spread ' + fmtSpread(lq.bid, lq.ask, lq.spread_pct) + ' | ' + fmtFair(lq.target_mid, lq.target_fair, lq.target_mid_vs_fair) + ' | ~$' + (lq.friction_est - lq.source_friction) + '</span>';
                    html += '</div>';
                } else {
                    // Single leg
                    html += '<div style="display:flex;gap:10px;flex-wrap:wrap;align-items:center;font-size:0.7rem;margin-top:3px;">';
                    html += '<span>OI=' + lq.oi + ' <span style="color:' + oiColor + '">(' + lq.oi_usage_pct + '% used)</span></span>';
                    html += '<span>spread ' + fmtSpread(lq.bid, lq.ask, lq.spread_pct) + '</span>';
                    html += '</div>';
                }
                html += '</div>';
                return html;
            })()}
            ${(() => {
                const k = r.kelly;
                if (!k) return '';
                const kColor = k.ev_total < 0 ? 'var(--red)' : k.kelly > 0.3 ? 'var(--green)' : k.kelly > 0.1 ? 'var(--cyan)' : 'var(--orange)';
                const evColor = k.ev_total >= 0 ? 'var(--green)' : 'var(--red)';
                const sign = k.ev_total >= 0 ? '+' : '';
                return '<div style="font-size:0.75rem;margin-top:3px;padding:4px 8px;background:rgba(88,166,255,0.06);border-radius:4px;display:flex;gap:10px;flex-wrap:wrap;align-items:center;">' +
                    '<span style="color:var(--text-dim);font-weight:600;">Kelly:</span>' +
                    '<span style="color:' + kColor + ';font-weight:600;">' + (k.kelly * 100).toFixed(1) + '% full</span>' +
                    '<span style="color:var(--text-dim);">(half=' + (k.half_kelly * 100).toFixed(1) + '%)</span>' +
                    '<span style="color:var(--text-dim);">EV ' + sign + '<span style="color:' + evColor + '">$' + k.ev_total + '</span></span>' +
                    '<span style="color:var(--text-dim);">P(OTM)=' + k.p_otm + '%</span>' +
                    '<span style="color:var(--text-dim);">win $' + k.win_per + ' vs loss $' + k.loss_per + '</span>' +
                '</div>';
            })()}
        </div>`;
    }).join('');
}

function updateTable(data) {
    const tbody = document.getElementById('positionBody');
    let rows = [...data.options];

    // Sort
    if (sortCol) {
        rows.sort((a, b) => {
            let av = a[sortCol], bv = b[sortCol];
            if (typeof av === 'string') return sortAsc ? av.localeCompare(bv) : bv.localeCompare(av);
            return sortAsc ? av - bv : bv - av;
        });
    }

    tbody.innerHTML = rows.map(o => {
        const excl = excludedIndices.has(o.idx);
        return `<tr class="${excl ? 'excluded' : ''}">
            <td><input type="checkbox" ${excl ? '' : 'checked'} onchange="togglePosition(${o.idx}, this.checked)"></td>
            <td>${o.expiry}</td>
            <td>$${o.strike.toFixed(1)}</td>
            <td style="color:${o.right === 'C' ? 'var(--cyan)' : 'var(--orange)'}">${o.right === 'C' ? 'Call' : 'Put'}</td>
            <td>${o.qty}</td>
            <td>$${o.avg_cost.toFixed(2)}</td>
            <td>$${o.theo_price.toFixed(2)}</td>
            <td class="${pnlClass(o.pnl)}">${fmt(o.pnl)}</td>
            <td class="${pnlClass(o.delta)}">${fmtNum(o.delta, 1)}</td>
            <td>${fmtNum(o.gamma, 2)}</td>
            <td class="${pnlClass(o.theta)}">${fmt(o.theta)}</td>
            <td>${fmtNum(o.vega, 2)}</td>
            <td>${o.dte}d</td>
            <td style="color:${o.extrinsic_pct > 50 ? 'var(--green)' : o.extrinsic_pct > 20 ? 'var(--orange)' : 'var(--red)'}">${o.extrinsic_pct.toFixed(0)}%</td>
        </tr>`;
    }).join('');
}

async function fetchTechnicals() {
    if (technicalsData !== null) return technicalsData;
    try {
        const resp = await fetch('/api/technicals');
        technicalsData = await resp.json();
        return technicalsData;
    } catch (e) {
        console.error('Failed to fetch technicals:', e);
        return null;
    }
}

function updateDeltaPriceMap(data, tech) {
    if (!tech || !data) return;
    const p = data.profile;
    const currentPrice = data.summary.price;

    // Filter to $8-$15 range
    const mask = p.prices.map(pr => pr >= 8 && pr <= 15);
    const prices = p.prices.filter((_, i) => mask[i]);
    const deltas = p.delta_profile.filter((_, i) => mask[i]);
    const pnls = p.pnl_total.filter((_, i) => mask[i]);

    const traces = [
        {
            x: prices, y: deltas, name: 'Net Delta',
            type: 'scatter', mode: 'lines',
            line: { color: '#58a6ff', width: 2.5 },
            yaxis: 'y',
        },
        {
            x: prices, y: pnls, name: 'Total P&L',
            type: 'scatter', mode: 'lines',
            line: { color: '#e6edf3', width: 2, dash: 'dot' },
            yaxis: 'y2',
        },
    ];

    // Build vertical reference lines
    const refLines = [
        { val: tech.low_52w, color: '#f85149', dash: 'dash', label: '52w Low' },
        { val: tech.high_52w, color: '#3fb950', dash: 'dash', label: '52w High' },
        { val: tech.ma_200, color: '#d29922', dash: 'solid', label: '200d MA' },
        { val: tech.ma_100, color: '#d29922', dash: 'dash', label: '100d MA' },
        { val: tech.ma_50, color: '#e3b341', dash: 'solid', label: '50d MA' },
        { val: tech.ma_20, color: '#e3b341', dash: 'dash', label: '20d MA' },
        { val: tech.vwap, color: '#39d2c0', dash: 'dot', label: 'VWAP' },
        { val: currentPrice, color: '#ffffff', dash: 'solid', label: 'Current', width: 2.5 },
        { val: tech.share_avg, color: '#ff69b4', dash: 'dash', label: 'Share Avg' },
        { val: tech.contango_30d, color: '#555555', dash: 'dot', label: 'Ctgo 30d' },
        { val: tech.contango_60d, color: '#555555', dash: 'dot', label: 'Ctgo 60d' },
        { val: tech.contango_90d, color: '#555555', dash: 'dot', label: 'Ctgo 90d' },
        { val: tech.gex_call_wall, color: '#bc8cff', dash: 'solid', label: 'GEX Call Wall' },
        { val: tech.gex_put_wall, color: '#3fb950', dash: 'dashdot', label: 'GEX Put Wall' },
    ];

    const shapes = [];
    const annotations = [];
    let annY = 1.0;
    const annStep = 0.055;

    for (const ref of refLines) {
        if (ref.val < 8 || ref.val > 15) continue;
        shapes.push({
            type: 'line', x0: ref.val, x1: ref.val, y0: 0, y1: 1, yref: 'paper',
            line: { color: ref.color, width: ref.width || 1.5, dash: ref.dash },
        });
        annotations.push({
            x: ref.val, y: annY, yref: 'paper',
            text: `${ref.label} $${ref.val.toFixed(2)}`,
            showarrow: false,
            font: { color: ref.color, size: 9 },
            yanchor: 'bottom', xanchor: 'left',
            textangle: -90,
        });
        annY -= annStep;
        if (annY < 0.05) annY = 1.0;
    }

    const layout = {
        ...plotlyLayout,
        title: { text: 'Price Map with Technical Levels', font: { color: '#8b949e', size: 13 } },
        shapes: shapes,
        annotations: annotations,
        xaxis: { ...plotlyLayout.xaxis, title: 'UNG Price ($)', range: [8, 15] },
        yaxis: { ...plotlyLayout.yaxis, title: 'Net Delta (shares)', side: 'left',
                 titlefont: { color: '#58a6ff' }, tickfont: { color: '#58a6ff' } },
        yaxis2: {
            title: 'P&L ($)', side: 'right', overlaying: 'y',
            gridcolor: 'rgba(48,54,61,0.3)', zerolinecolor: '#30363d',
            titlefont: { color: '#e6edf3' }, tickfont: { color: '#e6edf3' },
            tickformat: '$,.0f',
        },
        legend: { x: 0.02, y: 0.98, bgcolor: 'rgba(0,0,0,0.5)', font: { size: 11 } },
        margin: { t: 40, r: 70, b: 60, l: 70 },
    };

    Plotly.react('deltaPriceMap', traces, layout, { responsive: true, displayModeBar: false });
}

function updateDeltaMetrics(data, tech) {
    if (!tech || !data) return;
    const container = document.getElementById('deltaMetrics');
    const p = data.profile;
    const s = data.summary;

    // Interpolate delta at a specific price from profile data
    function deltaAtPrice(targetPrice) {
        const prices = p.prices;
        const deltas = p.delta_profile;
        if (targetPrice <= prices[0]) return deltas[0];
        if (targetPrice >= prices[prices.length - 1]) return deltas[deltas.length - 1];
        for (let i = 0; i < prices.length - 1; i++) {
            if (prices[i] <= targetPrice && prices[i+1] >= targetPrice) {
                const frac = (targetPrice - prices[i]) / (prices[i+1] - prices[i]);
                return deltas[i] + frac * (deltas[i+1] - deltas[i]);
            }
        }
        return deltas[0];
    }

    const deltaAt50 = deltaAtPrice(tech.ma_50);
    const deltaAt200 = deltaAtPrice(tech.ma_200);
    const deltaAtLow = deltaAtPrice(tech.low_52w);
    const deltaAtAvg = deltaAtPrice(tech.share_avg);

    const pctVs50 = ((tech.spot - tech.ma_50) / tech.ma_50 * 100);
    const pctVs200 = ((tech.spot - tech.ma_200) / tech.ma_200 * 100);

    const metrics = [
        { label: 'Current Delta', value: '+' + Math.round(s.net_delta).toLocaleString() + ' shares', cls: 'neutral' },
        { label: `Delta at 50d MA ($${tech.ma_50})`, value: '+' + Math.round(deltaAt50).toLocaleString(), cls: 'neutral' },
        { label: `Delta at 200d MA ($${tech.ma_200})`, value: '+' + Math.round(deltaAt200).toLocaleString(), cls: 'neutral' },
        { label: `Delta at 52w Low ($${tech.low_52w})`, value: '+' + Math.round(deltaAtLow).toLocaleString(), cls: 'risky' },
        { label: `Delta at Share Avg ($${tech.share_avg})`, value: '+' + Math.round(deltaAtAvg).toLocaleString(), cls: 'neutral' },
        { divider: true },
        { label: 'Gamma', value: fmtNum(s.total_gamma, 1) + (s.total_gamma < 0 ? ' (short gamma)' : ' (long gamma)'),
          cls: s.total_gamma < 0 ? 'risky' : 'favorable' },
        { label: 'Daily Theta', value: fmt(s.total_theta), cls: s.total_theta > 0 ? 'favorable' : 'risky' },
        { divider: true },
        { label: 'Regime', value: tech.rv_21 > 40 ? `HIGH VOL (RV 21d: ${tech.rv_21}%)` : tech.rv_21 > 25 ? `MED VOL (RV 21d: ${tech.rv_21}%)` : `LOW VOL (RV 21d: ${tech.rv_21}%)`,
          cls: tech.rv_21 > 40 ? 'warning' : tech.rv_21 > 25 ? 'neutral' : 'favorable' },
        { label: 'RV 63-day', value: tech.rv_63 + '%', cls: 'neutral' },
        { divider: true },
        { label: 'Contango Drag', value: '-3%/month', cls: 'risky' },
        { label: 'UNG vs 50d MA', value: (pctVs50 >= 0 ? '+' : '') + pctVs50.toFixed(1) + '%',
          cls: pctVs50 >= 0 ? 'favorable' : 'risky' },
        { label: 'UNG vs 200d MA', value: (pctVs200 >= 0 ? '+' : '') + pctVs200.toFixed(1) + '%',
          cls: pctVs200 >= 0 ? 'favorable' : 'risky' },
        { divider: true },
        { label: '52-Week Range', value: `$${tech.low_52w} - $${tech.high_52w}`, cls: 'neutral' },
        { label: 'VWAP (1Y)', value: '$' + tech.vwap.toFixed(2), cls: 'neutral' },
    ];

    container.innerHTML = metrics.map(m => {
        if (m.divider) return '<hr class="metric-divider">';
        return `<div class="metric-card ${m.cls}">
            <span class="metric-label">${m.label}</span>
            <span class="metric-value">${m.value}</span>
        </div>`;
    }).join('');
}

function updatePriceHistoryChart(tech) {
    if (!tech || !tech.price_history || tech.price_history.length === 0) return;

    const ph = tech.price_history;
    const mh = tech.ma_history;

    const traces = [
        {
            x: ph.map(p => p.date),
            open: ph.map(p => p.open),
            high: ph.map(p => p.high),
            low: ph.map(p => p.low),
            close: ph.map(p => p.close),
            type: 'candlestick',
            name: 'UNG',
            increasing: { line: { color: '#3fb950' } },
            decreasing: { line: { color: '#f85149' } },
        },
    ];

    // Add MA overlays
    const maConfigs = [
        { data: mh.ma_20, name: '20d MA', color: '#e3b341', dash: 'dash' },
        { data: mh.ma_50, name: '50d MA', color: '#e3b341', dash: 'solid' },
        { data: mh.ma_100, name: '100d MA', color: '#d29922', dash: 'dash' },
        { data: mh.ma_200, name: '200d MA', color: '#d29922', dash: 'solid' },
    ];

    for (const ma of maConfigs) {
        const validDates = [];
        const validVals = [];
        for (let i = 0; i < mh.dates.length; i++) {
            if (ma.data[i] !== null) {
                validDates.push(mh.dates[i]);
                validVals.push(ma.data[i]);
            }
        }
        if (validVals.length > 0) {
            traces.push({
                x: validDates, y: validVals,
                type: 'scatter', mode: 'lines',
                name: ma.name,
                line: { color: ma.color, width: 1.5, dash: ma.dash },
            });
        }
    }

    // Add horizontal lines at key strikes
    const strikeLines = [10.0, 10.5, 11.0, 11.5, 12.0];
    const shapes = strikeLines.map(k => ({
        type: 'line', x0: 0, x1: 1, xref: 'paper', y0: k, y1: k,
        line: { color: 'rgba(188,140,255,0.3)', width: 1, dash: 'dot' },
    }));

    const annotations = strikeLines.map(k => ({
        x: 1, xref: 'paper', y: k,
        text: '$' + k.toFixed(1) + ' K',
        showarrow: false,
        font: { color: 'rgba(188,140,255,0.6)', size: 9 },
        xanchor: 'left',
    }));

    const layout = {
        ...plotlyLayout,
        title: { text: 'UNG 60-Day Price History with Moving Averages', font: { color: '#8b949e', size: 13 } },
        shapes: shapes,
        annotations: annotations,
        xaxis: { ...plotlyLayout.xaxis, rangeslider: { visible: false } },
        yaxis: { ...plotlyLayout.yaxis, title: 'Price ($)' },
        legend: { x: 0.02, y: 0.98, bgcolor: 'rgba(0,0,0,0.5)', font: { size: 10 } },
        margin: { t: 40, r: 60, b: 60, l: 60 },
    };

    Plotly.react('priceHistoryChart', traces, layout, { responsive: true, displayModeBar: false });
}

function updateIVTermChart(tech) {
    if (!tech || !tech.iv_term || tech.iv_term.length === 0) return;
    const iv = tech.iv_term;

    const traces = [
        {
            x: iv.map(v => v.dte), y: iv.map(v => v.call_iv),
            name: 'Call IV', type: 'scatter', mode: 'lines+markers',
            line: { color: '#39d2c0', width: 2 },
            marker: { size: 6 },
        },
        {
            x: iv.map(v => v.dte), y: iv.map(v => v.put_iv),
            name: 'Put IV', type: 'scatter', mode: 'lines+markers',
            line: { color: '#d29922', width: 2 },
            marker: { size: 6 },
        },
        {
            x: iv.map(v => v.dte), y: iv.map(v => v.avg_iv),
            name: 'Avg IV', type: 'scatter', mode: 'lines+markers',
            line: { color: '#e6edf3', width: 2, dash: 'dot' },
            marker: { size: 5 },
        },
    ];

    // Add RV reference lines
    const shapes = [];
    const annotations = [];
    if (tech.rv_21) {
        shapes.push({
            type: 'line', x0: 0, x1: 1, xref: 'paper', y0: tech.rv_21, y1: tech.rv_21,
            line: { color: '#f85149', width: 1, dash: 'dash' },
        });
        annotations.push({
            x: 1, xref: 'paper', y: tech.rv_21,
            text: 'RV 21d: ' + tech.rv_21 + '%',
            showarrow: false, font: { color: '#f85149', size: 9 }, xanchor: 'left',
        });
    }
    if (tech.rv_63) {
        shapes.push({
            type: 'line', x0: 0, x1: 1, xref: 'paper', y0: tech.rv_63, y1: tech.rv_63,
            line: { color: '#bc8cff', width: 1, dash: 'dash' },
        });
        annotations.push({
            x: 1, xref: 'paper', y: tech.rv_63,
            text: 'RV 63d: ' + tech.rv_63 + '%',
            showarrow: false, font: { color: '#bc8cff', size: 9 }, xanchor: 'left',
        });
    }

    const layout = {
        ...plotlyLayout,
        title: { text: 'IV Term Structure (ATM)', font: { color: '#8b949e', size: 13 } },
        shapes: shapes,
        annotations: annotations,
        xaxis: { ...plotlyLayout.xaxis, title: 'Days to Expiry (DTE)' },
        yaxis: { ...plotlyLayout.yaxis, title: 'Implied Volatility (%)', ticksuffix: '%' },
        legend: { x: 0.02, y: 0.98, bgcolor: 'rgba(0,0,0,0.5)', font: { size: 10 } },
        margin: { t: 40, r: 60, b: 60, l: 60 },
    };

    Plotly.react('ivTermChart', traces, layout, { responsive: true, displayModeBar: false });
}

function updateIVSurfaceChart(tech) {
    if (!tech || !tech.iv_surface || tech.iv_surface.length === 0) return;

    // Build grid for heatmap
    const strikes = [...new Set(tech.iv_surface.map(p => p.strike))].sort((a, b) => a - b);
    const dtes = [...new Set(tech.iv_surface.map(p => p.dte))].sort((a, b) => a - b);

    const z = [];
    for (const strike of strikes) {
        const row = [];
        for (const dte of dtes) {
            const point = tech.iv_surface.find(p => p.strike === strike && p.dte === dte);
            row.push(point ? point.iv : null);
        }
        z.push(row);
    }

    const traces = [{
        z: z,
        x: dtes.map(d => d + 'd'),
        y: strikes.map(s => '$' + s.toFixed(1)),
        type: 'heatmap',
        colorscale: [
            [0, '#0d1117'],
            [0.2, '#1a1a4e'],
            [0.4, '#2d1b69'],
            [0.6, '#8b2fc9'],
            [0.8, '#d63384'],
            [1.0, '#ff6b6b'],
        ],
        colorbar: { title: 'IV%', titlefont: { color: '#8b949e' }, tickfont: { color: '#8b949e' }, ticksuffix: '%' },
        hovertemplate: 'Strike: %{y}<br>DTE: %{x}<br>IV: %{z}%<extra></extra>',
        connectgaps: false,
    }];

    const layout = {
        ...plotlyLayout,
        title: { text: 'IV Surface Heatmap', font: { color: '#8b949e', size: 13 } },
        xaxis: { ...plotlyLayout.xaxis, title: 'Days to Expiry', type: 'category' },
        yaxis: { ...plotlyLayout.yaxis, title: 'Strike Price', type: 'category' },
        margin: { t: 40, r: 30, b: 60, l: 60 },
    };

    Plotly.react('ivSurfaceChart', traces, layout, { responsive: true, displayModeBar: false });
}

// Cycle 86: indeterminate progress indicator for the long-running
// /api/timeline call (15-25s). Operator was unsure if recommendations
// were loading or hung. Shows an animated bar with elapsed seconds
// and stage hints. Returns a cleanup function the refresh() caller
// invokes when results arrive.
function _showRecommendationsProgress() {
    const container = document.getElementById('recommendationsList');
    if (!container) return () => {};
    const startedAt = Date.now();
    const stages = [
        { at: 0,  label: 'Generating candidate trades…' },
        { at: 3,  label: 'Running beam expansion (3 paths × 8 steps)…' },
        { at: 8,  label: 'Evaluating portfolio quality per path…' },
        { at: 14, label: 'Scoring hidden-win alternatives…' },
        { at: 22, label: 'Still working — large book takes longer…' },
    ];
    container.innerHTML = `
        <div class="rec-loading" id="recLoadingCard">
            <div class="rec-loading-header">
                <span class="rec-loading-stage" id="recLoadingStage">${stages[0].label}</span>
                <span><span id="recLoadingElapsed">0.0</span>s elapsed</span>
            </div>
            <div class="rec-loading-bar"></div>
            <div style="font-size:0.7rem;color:var(--text-dim);margin-top:6px;">
                Each request runs ~480 trade evaluations across the beam.
                Typical: 15–25s. Cached caches warm up after first hit.
            </div>
        </div>`;
    const tick = () => {
        const el = document.getElementById('recLoadingElapsed');
        const stageEl = document.getElementById('recLoadingStage');
        if (!el) return;
        const elapsed = (Date.now() - startedAt) / 1000;
        el.textContent = elapsed.toFixed(1);
        // Pick the latest stage whose `at` threshold has passed
        let label = stages[0].label;
        for (const s of stages) {
            if (elapsed >= s.at) label = s.label;
        }
        if (stageEl && stageEl.textContent !== label) stageEl.textContent = label;
    };
    const iv = setInterval(tick, 200);
    // Cleanup also REMOVES the loading card from the DOM. Without this, if
    // an upstream render function throws before updateRecommendations runs,
    // the bar would freeze in place at the elapsed-time it hit (user
    // reported "stuck at 18.7s"). Now the bar always disappears, and any
    // subsequent updateRecommendations call will populate the empty
    // container normally.
    return () => {
        clearInterval(iv);
        const card = document.getElementById('recLoadingCard');
        if (card && card.parentNode) card.parentNode.removeChild(card);
    };
}

async function refresh() {
    const _stopProgress = _showRecommendationsProgress();
    let data, baseline, timeline, progress;
    try {
        [data, baseline, timeline, progress] = await Promise.all([
            fetchData(), fetchBaseline(), fetchTimeline(), fetchProgress(30)
        ]);
    } catch (e) {
        _stopProgress();
        const c = document.getElementById('recommendationsList');
        if (c) c.innerHTML = `<div class="rec-loading" style="background:rgba(248,81,73,0.08);border-color:rgba(248,81,73,0.3);">
            <div style="color:var(--red,#f85149);font-weight:600;">Refresh failed</div>
            <div style="font-size:0.8rem;color:var(--text-dim);margin-top:4px;">${(e && e.message) || e}</div>
        </div>`;
        throw e;
    }
    _stopProgress();
    currentData = data;
    baselineData = baseline;
    window._progressData = progress;

    // 1. Daily status banner
    updateDailyStatus(timeline, data);

    // 2. Summary cards
    updateSummary(data);
    updateWhatIf(data, baseline);
    updateOutlook(data);

    // 3. Timeline
    updateTimelineChart(timeline);
    updateExpiryCards(timeline);

    // 4. Theta smoothness
    updateSmoothnessGauge(timeline);
    updateWeeklyThetaChart(timeline);

    // 5. Recommendations
    updateRecommendations(timeline);

    // 6. Delta dashboard
    const tech = technicalsData;
    if (tech) {
        updateDeltaPriceMap(data, tech);
        updateDeltaMetrics(data, tech);
    }

    // 7-10. Remaining sections
    updateWaterfallChart(timeline);
    updateCalendarGrid(timeline);
    updatePnlChart(data);
    updateDeltaChart(data);
    updateThetaChart(data);
    updateHeatmap(data);
    updateTable(data);
}

async function refreshFromWS() {
    const btn = document.getElementById('refreshBtn');
    btn.textContent = 'Refreshing...';
    btn.disabled = true;
    try {
        const resp = await fetch('/api/refresh');
        const data = await resp.json();
        if (data.success) {
            btn.textContent = 'Updated!';
            // Update slider to new live price
            const slider = document.getElementById('priceSlider');
            slider.value = data.ung_price.toFixed(2);
            document.getElementById('priceDisplay').textContent = '$' + data.ung_price.toFixed(2);
            // Reset technicals cache so it re-fetches
            technicalsData = null;
            const tech = await fetchTechnicals();
            if (tech) {
                updatePriceHistoryChart(tech);
                updateIVTermChart(tech);
                updateIVSurfaceChart(tech);
            }
            // Clear excluded indices since position indices may have changed
            excludedIndices.clear();
            await refresh();
        } else {
            btn.textContent = 'Failed - check cookies';
        }
    } catch(e) {
        btn.textContent = 'Error';
        console.error('WS refresh error:', e);
    }
    setTimeout(() => {
        const ts = new Date().toLocaleTimeString('en-US', {hour:'numeric', minute:'2-digit', hour12:true});
        btn.textContent = 'Refresh from WS (last: ' + ts + ')';
        btn.disabled = false;
    }, 3000);
}

function debouncedRefresh() {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(refresh, 80);
}

// Event Listeners
document.getElementById('priceSlider').addEventListener('input', function() {
    document.getElementById('priceDisplay').textContent = '$' + parseFloat(this.value).toFixed(2);
    debouncedRefresh();
});

document.getElementById('ivSlider').addEventListener('input', function() {
    document.getElementById('ivDisplay').textContent = Math.round(parseFloat(this.value) * 100) + '%';
    debouncedRefresh();
});

(function(){
    const tEl = document.getElementById('thesisSlider');
    if (!tEl) return;
    tEl.addEventListener('input', function() {
        const v = parseFloat(this.value);
        const label = v > 0.6 ? 'strong bull'
                    : v > 0.2 ? 'bull'
                    : v > -0.2 ? 'neutral'
                    : v > -0.6 ? 'bear'
                    : 'strong bear';
        document.getElementById('thesisDisplay').textContent = `${v >= 0 ? '+' : ''}${v.toFixed(1)} (${label})`;
        debouncedRefresh();
    });
})();

// Table sorting
document.querySelectorAll('th[data-col]').forEach(th => {
    th.addEventListener('click', function() {
        const col = this.dataset.col;
        if (col === 'active') return;
        if (sortCol === col) {
            sortAsc = !sortAsc;
        } else {
            sortCol = col;
            sortAsc = true;
        }
        document.querySelectorAll('th').forEach(h => h.classList.remove('sorted-asc', 'sorted-desc'));
        this.classList.add(sortAsc ? 'sorted-asc' : 'sorted-desc');
        updateTable(currentData);
    });
});

// Toggle position for what-if
window.togglePosition = function(idx, checked) {
    if (checked) {
        excludedIndices.delete(idx);
    } else {
        excludedIndices.add(idx);
    }
    refresh();
};

// Initial load: auto-refresh from WS, then fetch technicals + render.
// Cycle 85 bug fix: refreshFromWS catches its own network/cookie errors
// silently (sets btn text but doesn't throw), so the try/catch IIFE
// missed the failure-but-no-throw paths — technicalsData stayed null,
// refresh() never ran, and the Delta Management Dashboard sat at
// "Loading technicals…" forever. Now ALWAYS ensure technicals + render
// run regardless of whether refreshFromWS reports success.
(async function() {
    document.getElementById('lastUpdated').textContent = 'Refreshing from WS...';
    try {
        await refreshFromWS();
    } catch(e) {
        console.error('Auto-refresh on load failed:', e);
    }
    // Always reach a rendered state, even if refreshFromWS hit a non-
    // throwing failure path ("Failed - check cookies", "Error").
    if (technicalsData === null) {
        const tech = await fetchTechnicals();
        if (tech) {
            updatePriceHistoryChart(tech);
            updateIVTermChart(tech);
            updateIVSurfaceChart(tech);
        }
        await refresh();
    }
    const now = new Date();
    document.getElementById('lastUpdated').textContent =
        'Last refresh: ' + now.toLocaleString('en-US', {
            month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit', hour12: true
        });

    // Periodic auto-refresh every 5 minutes
    setInterval(() => {
        console.log('Auto-refresh triggered');
        refreshFromWS();
    }, 5 * 60 * 1000);
})();
</script>
</body>
</html>
"""


# ── HTTP Server ──────────────────────────────────────────────────────────────

def _handle_refresh():
    """Re-fetch positions from WS, update globals, trigger z-score update."""
    global SHARES, SHARE_AVG, OPTIONS, UNG_PRICE
    result = fetch_ws_positions()
    if result:
        SHARES, SHARE_AVG, OPTIONS, UNG_PRICE = result
        # Invalidate technicals cache
        _technicals_cache['data'] = None
        _technicals_cache['timestamp'] = 0
        # Trigger model z-score update in background (non-blocking)
        refresh_model_zscore()
        print(f"Refreshed: {SHARES} shares @ ${SHARE_AVG:.2f}, "
              f"{len(OPTIONS)} options, UNG ${UNG_PRICE:.2f}, z-score updating...")
        return {
            'success': True,
            'shares': SHARES,
            'share_avg': SHARE_AVG,
            'n_options': len(OPTIONS),
            'ung_price': UNG_PRICE,
            'zscore': _model_zscore,
        }
    return {'success': False, 'error': 'WS fetch failed'}


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Quiet logging - only show errors
        if args and '404' in str(args[0]):
            super().log_message(format, *args)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path == '/' or parsed.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            # Inject current UNG_PRICE into the slider defaults
            page = HTML_PAGE.replace('value="10.74"', f'value="{UNG_PRICE:.2f}"')
            page = page.replace('$10.74</span>', f'${UNG_PRICE:.2f}</span>')
            self.wfile.write(page.encode('utf-8'))

        elif parsed.path == '/api/data':
            params = urllib.parse.parse_qs(parsed.query)
            price = float(params.get('price', [str(UNG_PRICE)])[0])
            iv_raw = float(params.get('iv', ['0.50'])[0])
            iv = iv_raw if iv_raw <= 2.0 else iv_raw / 100.0  # handle both 0.50 and 50
            excluded_str = params.get('excluded', [''])[0]
            excluded = set()
            if excluded_str:
                for x in excluded_str.split(','):
                    x = x.strip()
                    if x:
                        excluded.add(int(x))

            data = compute_data(price, iv, excluded)

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(data).encode('utf-8'))

        elif parsed.path == '/api/technicals':
            try:
                data = compute_technicals()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(data).encode('utf-8'))
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': str(e)}).encode('utf-8'))

        elif parsed.path == '/api/timeline':
            params = urllib.parse.parse_qs(parsed.query)
            price = float(params.get('price', [str(UNG_PRICE)])[0])
            iv_raw = float(params.get('iv', ['0.50'])[0])
            iv = iv_raw if iv_raw <= 2.0 else iv_raw / 100.0
            excluded_str = params.get('excluded', [''])[0]
            excluded = set()
            if excluded_str:
                for x in excluded_str.split(','):
                    x = x.strip()
                    if x:
                        excluded.add(int(x))
            try:
                thesis_tilt = float(params.get('thesis_tilt', ['0'])[0])
            except ValueError:
                thesis_tilt = 0.0

            data = compute_timeline(price, iv, excluded, thesis_tilt=thesis_tilt)

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(data).encode('utf-8'))

        elif parsed.path == '/api/refresh':
            resp_data = _handle_refresh()
            if resp_data['success']:
                self.send_response(200)
            else:
                self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(resp_data).encode())

        elif parsed.path == '/api/health':
            # Lightweight health probe (cycle 46). Returns 200 if all data
            # sources are fresh enough; 503 if any are stale. Cron / uptime
            # checks should hit this rather than /api/timeline (which is
            # expensive ~30s with the beam).
            import datetime as _dt
            _now = time.time()
            uptime_s = _now - _server_startup_time
            tech_age_s = _now - _technicals_cache.get('timestamp', 0) if _technicals_cache.get('timestamp') else None
            opts_age_s = _now - _available_options_time if _available_options_time else None
            pred_updated_at = _model_predictions.get('updated_at')
            pred_age_s = None
            if pred_updated_at:
                try:
                    pred_dt = _dt.datetime.fromisoformat(pred_updated_at)
                    pred_age_s = (_dt.datetime.now() - pred_dt).total_seconds()
                except Exception:
                    pred_age_s = None
            # Thresholds: predictions stale > 2h, technicals/options > 15m
            # Per-factor freshness (cycle 49): forecasts publish on
            # different cadences (weekly EIA storage = 7d; monthly EIA
            # power burn = 30-60d incl pub lag). Flag any factor >95d
            # since last update as a warning in the checks dict.
            factor_freshness = _model_predictions.get('factor_freshness', {}) or {}
            stale_factors = [
                k for k, v in factor_freshness.items()
                if isinstance(v, dict) and v.get('age_days', 0) > 95
            ]
            checks = {
                'server_up': True,
                'predictions_fresh': pred_age_s is not None and pred_age_s < 7200,
                'technicals_fresh': tech_age_s is not None and tech_age_s < 900,
                'options_fresh': opts_age_s is not None and opts_age_s < 900,
                'factors_fresh': len(stale_factors) == 0,
            }
            all_ok = all(checks.values())
            payload = {
                'status': 'ok' if all_ok else 'degraded',
                'uptime_seconds': round(uptime_s, 1),
                'predictions_age_seconds': round(pred_age_s, 1) if pred_age_s is not None else None,
                'technicals_age_seconds': round(tech_age_s, 1) if tech_age_s is not None else None,
                'options_age_seconds': round(opts_age_s, 1) if opts_age_s is not None else None,
                'predictions_updated_at': pred_updated_at,
                'shares': SHARES,
                'options_count': len(OPTIONS) if OPTIONS else 0,
                'ung_price': UNG_PRICE,
                'factor_freshness': factor_freshness,
                'stale_factors': stale_factors,
                'checks': checks,
            }
            self.send_response(200 if all_ok else 503)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode())

        elif parsed.path == '/api/progress':
            # Daily-snapshot time series for the income progress tracker
            # (cycle 52). Query: ?days=30 (default), ?days=365 etc.
            q = urllib.parse.parse_qs(parsed.query)
            days = int(q.get('days', ['30'])[0])
            days = max(1, min(days, 730))
            rows = _progress_load(days=days)
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({'days': days, 'snapshots': rows}).encode())

        elif parsed.path in ('/api/factor_curves.png', '/api/forecast_chart.png',
                             '/api/probability_cone.png'):
            # Serve the latest NG forecast charts as static images (cycle 48).
            # Lets the dashboard embed them as <img> tags without a separate
            # static-file server. mtime-based no-cache so the browser pulls
            # the fresh PNG whenever the forecast re-runs.
            file_map = {
                '/api/factor_curves.png': '/home/wyatt/weather/ng_factor_curves.png',
                '/api/forecast_chart.png': '/home/wyatt/weather/ng_daily_forecast.png',
                '/api/probability_cone.png': '/home/wyatt/weather/ng_probability_cone.png',
            }
            file_path = file_map[parsed.path]
            try:
                with open(file_path, 'rb') as f:
                    data = f.read()
                self.send_response(200)
                self.send_header('Content-Type', 'image/png')
                self.send_header('Content-Length', str(len(data)))
                self.send_header('Cache-Control', 'no-cache')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(data)
            except FileNotFoundError:
                self.send_response(404)
                self.send_header('Content-Type', 'text/plain')
                self.end_headers()
                self.wfile.write(f'Chart not generated yet: {file_path}'.encode())

        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'Not found')


class ReusableTCPServer(http.server.HTTPServer):
    allow_reuse_address = True


def main():
    # Margin capital already fetched during fetch_ws_positions() at import time

    # Refresh model predictions (PREDICTION_*) at startup so the dashboard
    # never serves with pillars at default 0. Background thread, non-blocking.
    # Also schedule a periodic refresh (every hour) to keep predictions fresh.
    refresh_model_zscore()
    import threading
    def _periodic_pred_refresh():
        import time
        while True:
            time.sleep(3600)  # 1 hour
            try:
                refresh_model_zscore()
            except Exception as e:
                print(f"periodic refresh_model_zscore failed: {e}")
    threading.Thread(target=_periodic_pred_refresh, daemon=True).start()

    # Cycle 95: also refresh technicals + positions periodically so the
    # cache doesn't go stale when no browser session is open. Without this,
    # /api/health reports "degraded" (technicals/options >15min) after the
    # operator closes the tab, and the next open serves recommendations
    # computed against stale market state. Warm immediately on startup,
    # then refresh every 10min (under the 15min freshness threshold).
    def _tech_refresh_once():
        global SHARES, SHARE_AVG, OPTIONS, UNG_PRICE, _available_options, _available_options_time
        # Technicals
        _technicals_cache['data'] = None
        _technicals_cache['timestamp'] = 0
        _ = get_technicals_cached()
        # WS positions + margin NLV
        _result = fetch_ws_positions()
        if _result:
            SHARES, SHARE_AVG, OPTIONS, UNG_PRICE = _result
        # Option chain — get_available_options is what /api/health probes
        # for options_age_seconds. Invalidate then re-fetch.
        _available_options = None
        _available_options_time = 0
        _ = get_available_options()

    # Initial warm — don't block startup if it fails
    try:
        _tech_refresh_once()
    except Exception as e:
        print(f"initial technicals warm failed: {e}")

    def _periodic_tech_refresh():
        import time
        while True:
            time.sleep(600)  # 10 min
            try:
                _tech_refresh_once()
            except Exception as e:
                print(f"periodic technicals refresh failed: {e}")
    threading.Thread(target=_periodic_tech_refresh, daemon=True).start()

    # Auto-restart on source file change (closes cycle-39 stale-process bug).
    # Watches mtime of ung_visualizer.py; on detected change, waits 2s for the
    # file to stabilize, then os.execv re-exec to pick up the new code.
    # Disable by setting env UNG_VIZ_NO_AUTO_RELOAD=1.
    import os as _os
    if not _os.environ.get('UNG_VIZ_NO_AUTO_RELOAD'):
        def _watch_for_reload():
            import os
            import sys
            import time
            path = os.path.abspath(__file__)
            last_mtime = os.path.getmtime(path)
            while True:
                time.sleep(3)
                try:
                    mtime = os.path.getmtime(path)
                    if mtime != last_mtime:
                        time.sleep(2)  # stability wait
                        mtime2 = os.path.getmtime(path)
                        if mtime2 == mtime:
                            print(f"\n[auto-reload] {path} changed; re-execing...\n")
                            sys.stdout.flush()
                            os.execv(sys.executable, [sys.executable, path])
                        last_mtime = mtime
                except Exception as e:
                    print(f"[auto-reload] watcher error: {e}")
        threading.Thread(target=_watch_for_reload, daemon=True).start()

    port = 9999
    server = ReusableTCPServer(('0.0.0.0', port), Handler)
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = "0.0.0.0"
    print("Server running at:")
    print(f"  Local:   http://localhost:{port}")
    print(f"  Network: http://{local_ip}:{port}")
    print("  (accessible from phone/laptop on same WiFi)")
    print("Press Ctrl+C to stop")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.server_close()


if __name__ == '__main__':
    main()
