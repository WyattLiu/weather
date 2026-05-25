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
        return (new_q_total - initial_quality, new_state, new_q_dict, c)
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
                    liquidity[(s, 'P')] = {
                        'oi': _safe_int(row.get('openInterest', 0)),
                        'vol': _safe_int(row.get('volume', 0)),
                        'bid': _safe_float(row.get('bid', 0)),
                        'ask': _safe_float(row.get('ask', 0)),
                        'iv': _iv if 0.05 < _iv < 3.0 else 0.0,
                    }
            for _, row in chain.calls.iterrows():
                s = float(row['strike'])
                if 5 <= s <= 25:
                    _iv = _safe_float(row.get('impliedVolatility', 0))
                    liquidity[(s, 'C')] = {
                        'oi': _safe_int(row.get('openInterest', 0)),
                        'vol': _safe_int(row.get('volume', 0)),
                        'bid': _safe_float(row.get('bid', 0)),
                        'ask': _safe_float(row.get('ask', 0)),
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
        hist.index = hist.index.tz_localize(None)

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

        data = graphql_query(session, "FetchIdentityPositions", QUERY_FETCH_POSITIONS, {
            "identityId": identity_id,
            "currency": "CAD",
            "first": 50,
            "aggregated": True,
            "currencyOverride": "MARKET",
            "sort": "TODAY_GAIN",
            "includeSecurity": True,
            "includeAccountData": True,
            "includeOneDayReturnsBaseline": True,
        })

        shares = 0
        share_avg = 0.0
        options = []
        ung_price = None
        # Cycle 142: also capture cash-park holdings (BOXX, ADA, etc.) so
        # the idle-cash math accounts for what's already deployed in
        # cash-equivalents. Previously the visualizer was UNG-only and
        # the "park in BOXX" suggestion ignored existing BOXX positions.
        global _OTHER_HOLDINGS
        _OTHER_HOLDINGS = {}  # symbol -> {qty, market_value}

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

        if ung_price is None:
            # Fallback: get spot from yfinance
            ung_price = float(yf.Ticker('UNG').history(period='1d')['Close'].iloc[-1])

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
        delta_profile.append(round(SHARES + o_delta, 2))

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
            'net_delta': round(net_delta, 2),
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
    active = [v for v in weekly_theta.values() if v > max_wk * 0.05]
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

        # Generate CLOSE candidates for near-worthless positions
        if extrinsic * abs(qty) * 100 < 50:  # close when < $50 total extrinsic remaining
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
                'detail': f"Only ${extrinsic * abs(qty) * 100:.2f} extrinsic left. Consider rolling to ATM ${nearest_atm_strike}{right} at 30-40 DTE.",
                'why': "Theta exhausted. Free up margin.",
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

    # Generate ADD candidates using REAL expiries, strikes, and liquidity
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

        # STRANGLE REMOVED (cycle 180, codex P0 audit). The call leg was
        # sold without checking share coverage — naked short call risk.
        # Wheel strategy handles puts (OPEN) and calls (COVERED CALL)
        # separately; strangles are not wheel-aligned.

    # ── BUY/SELL shares (once, outside expiry loop) ──
    # Only for large gaps. Qty = match the gap exactly, not overshoot.
    if delta_gap > 500:
        share_qty = min(SHARES, max(100, int(round(delta_gap / 100) * 100)))
        loss_per_share = SHARE_AVG - spot  # realized loss if selling below avg cost
        candidates.append({
            'type': 'SELL SHARES',
            'action': f"Sell {share_qty} UNG shares",
            'add_qty': share_qty,
            'theta_change': 0,
            'delta_change': -share_qty,
            'gamma_change': 0,
            'vega_change': 0,
            'new_extrinsic_total': 0,
            'n_legs': 1,
            'detail': f"Δ-{share_qty} | realized {'loss' if loss_per_share > 0 else 'gain'}: ${abs(loss_per_share)*share_qty:,.0f} | ~${share_qty * 0.01:.0f} friction",
            'why': f"Delta emergency. Selling at ${spot:.2f} vs avg ${SHARE_AVG:.2f}.",
        })

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
    # already had its own ladder generator; LET EXPIRE/ASSIGNMENT
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
    _LEAPS_MIN_DTE = 180
    _tail_qty = int(portfolio_state.get('tail_hedge_qty', 0) or 0)
    _tail_floor = int(portfolio_state.get('tail_hedge_floor', 2) or 2)
    if _tail_qty < _tail_floor:
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
                _need = _tail_floor - _tail_qty
                for _lp_qty in sorted(set([1, _need])):
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
                        'why': f"Tail-hedge floor requires {_tail_floor} LEAPS puts, have {_tail_qty}. This is catastrophe protection per CENTRAL_PHILOSOPHY.",
                    })

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


@_functools.lru_cache(maxsize=128)
def _compute_target_delta_cached(spot, z, capital_base):
    """Inner cached body — args are explicit so cache invalidates naturally
    when z or capital changes across requests. Cycle 81."""
    if z >= 1.0:
        leverage = 1.5
    elif z >= 0.5:
        leverage = 0.8 + (z - 0.5) / 0.5 * 0.7
    elif z >= 0:
        leverage = 0.6 + z / 0.5 * 0.2
    elif z >= -0.5:
        leverage = 0.4 + (z + 0.5) / 0.5 * 0.2
    else:
        leverage = 0.3
    target_dollar = capital_base * leverage
    target = target_dollar / spot if spot > 0 else 7000
    target = round(target / 100) * 100
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


def compute_target_delta(spot):
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
    return _compute_target_delta_cached(spot, z, _margin_capital_usd)


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
                expected_move_past = (e_intr / p_itm) if (p_itm or 0) > 0.001 else 0.0
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
            expected_assignment_cost = float(p_itm) * (expected_move_past + expected_upside) * trade_qty * 100
            # Scale: $100 expected cost = 1 point penalty
            assignment_score = delta_sim_score - expected_assignment_cost / 100
        elif target_right == 'P':
            # Short put assignment = gain shares.
            # In bullish regime: buying shares cheap is GOOD (expected recovery)
            expected_assignment_benefit = float(p_itm) * max(0, expected_spot - target_strike_val + expected_move_past) * trade_qty * 100
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

    # _recovery_days REMOVED — dead code (codex audit, never called)

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
    # Cycle 180: trigger at 25% (¼ Kelly per CENTRAL_PHILOSOPHY, was 50%).
    # Income mode is "conservative" — ¼ Kelly is the strategic default.
    over_kelly_mult = max(0, kelly_pct - 0.25) * 4

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


    elif trade['type'] == 'OPEN':
        target_exp = trade.get('target_exp')
        target_strike = trade.get('target_strike')
        add_qty = trade.get('add_qty', 3)
        # Determine right from action string
        right = 'C' if 'C ' in trade.get('action', '') or trade.get('action', '').endswith('C') else 'P'
        new_positions.append((target_exp, target_strike, right, -add_qty, 0))

    elif trade['type'] == 'SELL SHARES':
        # Cycle 180 P1 fix: update modeled share count so downstream
        # covered-call capacity checks reflect shares already "sold."
        _sell_qty = abs(trade.get('qty', trade.get('add_qty', 0)))
        new_state['shares'] = max(0, new_state.get('shares', SHARES) - _sell_qty)

    elif trade['type'] == 'BUY SHARES':
        _buy_qty = abs(trade.get('qty', trade.get('add_qty', 0)))
        new_state['shares'] = new_state.get('shares', SHARES) + _buy_qty

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
    _hard_dd_veto = dd_frac < -0.15
    theta_30d = theta_horizon * (30.0 / HORIZON_D)  # legacy field for diag display

    # Delta gap (quadratic, mild)
    target_delta, _, _ = compute_target_delta(spot) if spot > 0 else (total_delta, '', 0.0)
    delta_gap_shares = total_delta - target_delta
    delta_gap = -(delta_gap_shares ** 2) * 0.0001  # 1000-share gap = -$100

    # Smoothness bonus
    smoothness_bonus = smoothness * 500.0

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
    _missing = max(0, tail_floor - tail_qty)
    if _missing > 0:
        try:
            _ALPHA = 0.05
            _HORIZON = 7
            # CVaR from ScenarioDistribution if available, else estimate
            # from IV (annualized vol → 7-day 5%-CVaR ≈ iv × sqrt(7/252) × 2.06)
            _spot = float(state.get('spot', 11.0) or 11.0)
            _leaps_iv = float(state.get('iv_est', 0.45) or 0.45)
            # CVaR as a PRICE DROP (not fraction) — matches dd_penalty units
            if sd is not None:
                _cvar_price = float(sd.cvar_loss(_HORIZON, alpha=_ALPHA))
            else:
                _daily_vol = _leaps_iv / (252 ** 0.5)
                _cvar_frac = _daily_vol * (_HORIZON ** 0.5) * 2.06
                _cvar_price = _cvar_frac * _spot

            # LEAPS put Greeks (200-DTE ATM, per contract = 100 shares)
            _leaps_T = 200.0 / 365.0
            _leaps_delta = bs_delta(_spot, _spot, _leaps_T, 0.045, _leaps_iv, 'P') * 100
            _leaps_gamma = bs_gamma(_spot, _spot, _leaps_T, 0.045, _leaps_iv) * 100
            # In a crash, LEAPS gain: delta offset + gamma convexity
            _benefit_per = max(0,
                -_leaps_delta * _cvar_price
                + 0.5 * _leaps_gamma * _cvar_price ** 2
            )
            # Probability-weighted
            _expected_benefit = _ALPHA * _benefit_per
            # Scale by short gamma exposure (more shorts = hedge more urgent)
            _short_gamma = abs(min(0, total_gamma))
            _exposure_mult = max(1.0, _short_gamma / 500.0)
            tail_hedge_penalty = -_expected_benefit * _missing * _exposure_mult
        except Exception:
            tail_hedge_penalty = -500.0 * _missing
    else:
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

    total = (income_gap + dd_penalty + delta_gap + smoothness_bonus
             + tail_hedge_penalty + pillar_bonus + _friction_penalty)

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
    MAX_RECS = 6
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
