"""Historical replay engine — walks day-by-day applying strategy variants.

Uses the master dataset produced by historical_data_pipeline.py.
Outputs per-strategy equity curves, trade logs, attribution.

WS = zero commission. Only friction is bid-ask spread (~$0.05/contract typical).

Run:
  python backtest/replay_engine.py --start 2021-06-01 --strategy regime_aware
  python backtest/replay_engine.py --compare  # compare all strategies
"""
import os
import sys
import math
import json
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from seasonal_z import add_seasonal_factors  # type: ignore
from iv_model import precompute_realized_vol, iv_for_quote  # type: ignore
from attribution import attribute_trades, print_attribution  # type: ignore
from kelly_sizing import kelly_qty_short_put, kelly_qty_covered_call  # type: ignore

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# WS = zero commission
COMMISSION = 0.0
SPREAD_OPTION = 0.03  # $0.03/share bid-ask half-spread typical UNG
SPREAD_SHARE = 0.005


def bs_put(S, K, T, sig, r=0.045):
    if T <= 0.001 or sig <= 0:
        return max(0, K - S)
    d1 = (math.log(S/K) + (r + 0.5*sig**2)*T) / (sig*math.sqrt(T))
    return K*math.exp(-r*T)*norm.cdf(-(d1 - sig*math.sqrt(T))) - S*norm.cdf(-d1)


def bs_call(S, K, T, sig, r=0.045):
    if T <= 0.001 or sig <= 0:
        return max(0, S - K)
    d1 = (math.log(S/K) + (r + 0.5*sig**2)*T) / (sig*math.sqrt(T))
    return S*norm.cdf(d1) - K*math.exp(-r*T)*norm.cdf(d1 - sig*math.sqrt(T))


def compute_historical_z(row, use_surprise=False):
    """Approximate composite z-score from available factors.

    use_surprise=True uses seasonal-detrended storage/days_supply
    (storage_surprise_z) — removes the dominant ~annual sine cycle,
    so signal reflects deviation from SEASONAL expectation, not raw level.
    """
    z_components = []
    weights = []

    # Storage deviation (negative when storage low = bullish)
    storage_col = 'storage_surprise_z' if use_surprise else 'storage_z'
    if storage_col in row and not pd.isna(row[storage_col]):
        z_components.append(-row[storage_col])
        weights.append(0.30)

    # Days supply
    ds_col = 'days_supply_surprise_z' if use_surprise else 'days_supply_z'
    if ds_col in row and not pd.isna(row[ds_col]):
        z_components.append(-row[ds_col])
        weights.append(0.25)

    # NG trend
    if 'ng_trend' in row and not pd.isna(row['ng_trend']):
        z_components.append(-row['ng_trend'] * 3)  # high trend (above MA) = bearish (mean revert)
        weights.append(0.20)

    # VIX (market fear) — mildly bearish for NG (demand fear)
    if 'VIX' in row and not pd.isna(row['VIX']):
        vix_normed = (row['VIX'] - 20) / 10  # rough z
        z_components.append(-vix_normed * 0.5)
        weights.append(0.10)

    # CL/NG ratio (oil/gas)
    if 'CL' in row and 'NG' in row and not pd.isna(row['NG']) and row['NG'] > 0:
        ratio = row['CL'] / row['NG']
        # Typical ratio ~25; > 30 = NG cheap relative
        ratio_z = (ratio - 25) / 10
        z_components.append(ratio_z * 0.5)
        weights.append(0.15)

    if not z_components:
        return 0.0
    return float(np.average(z_components, weights=weights))


def precompute_factor_z(df):
    """Add z-score normalized columns: naive (252d) AND seasonal surprise.
    Also adds price-spike indicator (UNG % change vs 60d ago).
    BUG FIX (cycle 20260531_143753): drop NaN UNG rows FIRST so rolling
    windows count trading days, not calendar days. Previously rolling(50)
    on weekend-padded data could never accumulate 50 valid samples,
    silently NaN-ing all trend signals (50d MA, 200d MA, uptrend)."""
    if 'UNG' in df.columns:
        df = df[df['UNG'].notna()].copy()
    if 'eia_storage_weekly' in df.columns:
        s = df['eia_storage_weekly']
        df['storage_z'] = ((s - s.rolling(252).mean()) / (s.rolling(252).std() + 1e-9))
    if 'days_supply' in df.columns:
        s = df['days_supply']
        df['days_supply_z'] = ((s - s.rolling(252).mean()) / (s.rolling(252).std() + 1e-9))
    # Seasonal-detrended (removes annual sine cycle that dominates raw z)
    df = add_seasonal_factors(df)
    # Price spike indicator (60d % change) — catches demand spikes (Russia)
    # that storage-based z misses.
    if 'UNG' in df.columns:
        df['ung_spike_60d'] = df['UNG'].pct_change(60)
        # Falling-knife signals: 20d low + 5d momentum
        df['ung_20d_low'] = df['UNG'].rolling(20).min()
        df['ung_at_20d_low'] = df['UNG'] <= df['ung_20d_low'] * 1.005
        df['ung_5d_mom'] = df['UNG'].pct_change(5)
        # 60d high (for called-away cycle peak detection)
        df['ung_60d_high'] = df['UNG'].rolling(60).max()
        # 252d (1yr) range — for anomaly detection AND trend
        df['ung_252d_mean'] = df['UNG'].rolling(252).mean()
        df['ung_252d_std'] = df['UNG'].rolling(252).std()
        df['ung_200d_ma'] = df['UNG'].rolling(200).mean()
        df['ung_50d_ma'] = df['UNG'].rolling(50).mean()
        # Trend: 50d above 200d AND price above 50d = uptrend confirmed
        df['ung_uptrend'] = (df['UNG'] > df['ung_50d_ma']) & (df['ung_50d_ma'] > df['ung_200d_ma'])
        # Downtrend: price below 200d AND 50d below 200d
        df['ung_downtrend'] = (df['UNG'] < df['ung_200d_ma']) & (df['ung_50d_ma'] < df['ung_200d_ma'])
    df = precompute_realized_vol(df, col='UNG')
    return df


def detect_anomaly(row) -> str:
    """Return 'ANOMALY_UP' / 'ANOMALY_DOWN' / 'NORMAL'.
    Per [[feedback_no_falling_knife_anomaly]]: stand down in extreme regimes
    rather than fight them with normal wheel kernels."""
    spike = row.get('ung_spike_60d', 0) or 0
    if spike > 0.50:
        return 'ANOMALY_UP'
    if spike < -0.50:
        return 'ANOMALY_DOWN'
    rv = row.get('rv_30', 0) or 0
    if rv > 1.00:
        return 'ANOMALY_UP'  # extreme vol — treat as caution
    return 'NORMAL'


def model_conviction(row, z: float, anomaly: str) -> float:
    """Returns p_otm adjustment in [-0.20, +0.20]. Small effect on Kelly."""
    conv = 0.0
    z_surp = float(row.get('storage_surprise_z') or 0)
    if z_surp > 1.0:    conv += 0.08
    elif z_surp > 0.5:  conv += 0.04
    elif z_surp < -1.0: conv -= 0.08
    elif z_surp < -0.5: conv -= 0.04
    if z > 0.5:   conv += 0.04
    elif z < -0.5: conv -= 0.04
    mom5 = float(row.get('ung_5d_mom') or 0)
    if mom5 > 0.05:    conv += 0.04
    elif mom5 > 0.02:  conv += 0.02
    elif mom5 < -0.05: conv -= 0.04
    if falling_knife(row): conv -= 0.05
    if anomaly != 'NORMAL':
        conv *= 0.4
    return max(-0.20, min(0.20, conv))


def firmness_multiplier(row, z: float, anomaly: str) -> float:
    """Per user: 'when extreme volatility happens, based on history and
    modeling, the firmness of confidence will get a higher [sizing]'.

    Returns a MULTIPLIER on Kelly qty in [0.5, 2.5]. Activates strongest
    when HIGH VOL + multiple model signals align. The intuition: rich
    premium AND high conviction is rare — when both happen, size up hard.
    Default 1.0 (neutral). 0.5 (de-risk) when anomaly conflicts with model.

    Signal stack:
      vol regime (rv_30):  amplifies confidence interpretation
      z direction:         is model bullish or bearish?
      momentum direction:  confirms or contradicts?
      seasonal:            (encoded via z_surprise already)
      anomaly:             damp during true 2022-style events
    """
    rv30 = float(row.get('rv_30') or 0.5)
    mom5 = float(row.get('ung_5d_mom') or 0)
    z_surp = float(row.get('storage_surprise_z') or 0)
    # Count aligned bullish signals
    bullish_signals = 0
    if z > 0.5:        bullish_signals += 1
    if z_surp > 0.5:   bullish_signals += 1
    if mom5 > 0.02 and not falling_knife(row): bullish_signals += 1
    # Count aligned bearish signals
    bearish_signals = 0
    if z < -0.5:        bearish_signals += 1
    if z_surp < -0.5:   bearish_signals += 1
    if mom5 < -0.03 or falling_knife(row): bearish_signals += 1

    # Base multiplier
    mult = 1.0
    if bullish_signals >= 2:
        # Multi-signal bullish alignment → size UP
        if rv30 > 0.80:  mult = 2.0   # HIGH VOL + confirmed → max conviction (rare gold)
        elif rv30 > 0.60: mult = 1.6
        else:             mult = 1.3
    elif bearish_signals >= 2:
        # Multi-signal bearish → size DOWN
        if rv30 > 0.80:  mult = 0.5
        elif rv30 > 0.60: mult = 0.7
        else:             mult = 0.9

    # Anomaly damp: even if model says go big, anomaly says be careful.
    # ANOMALY_UP (parabolic up) is the dangerous one — don't sell more
    # puts into a melting rally. ANOMALY_DOWN — can be opportunity if
    # model agrees, but capped.
    if anomaly == 'ANOMALY_UP':
        mult = min(mult, 0.7)   # never size up in parabolic up
    elif anomaly == 'ANOMALY_DOWN':
        mult = min(mult, 1.3)   # cap during true panic

    return max(0.5, min(2.5, mult))


def falling_knife(row) -> bool:
    """True if UNG is in active downtrend — don't accumulate here.
    Per [[feedback_no_falling_knife_anomaly]]."""
    at_low = bool(row.get('ung_at_20d_low', False))
    mom5 = float(row.get('ung_5d_mom') or 0)
    return at_low and mom5 < -0.03


def detect_divergence(row, z: float) -> str:
    """Return ALIGNED / PANIC_BUY_OPP / EUPHORIC_SELL_OPP.
    Per [[project_fundamental_divergence_alpha]]: trade fundamental vs crowd
    divergence. TIGHTLY GATED — false positives are catastrophic (deep ITM
    CC sold into a bounce = massive locked loss). Require ALL conditions:

    PANIC_BUY_OPP needs:
      - z > +1.0 (extreme cheap, not just slightly)
      - 5d momentum < -7% (real panic, not noise)
      - NOT falling-knife
      - 60d spike < 0 (sustained weakness, not a pullback in uptrend)

    EUPHORIC_SELL_OPP needs:
      - z < -1.0 (extreme rich)
      - 60d return > +30% (real euphoria, not a normal rally)
      - rv_30 > 0.80 (IV/vol pricing the euphoria)
    """
    mom5 = float(row.get('ung_5d_mom') or 0)
    spike = float(row.get('ung_spike_60d') or 0)
    rv = float(row.get('rv_30') or 0)
    if (z > 1.0 and mom5 < -0.07 and not falling_knife(row)
            and spike < 0):
        return 'PANIC_BUY_OPP'
    if z < -1.0 and spike > 0.30 and rv > 0.80:
        return 'EUPHORIC_SELL_OPP'
    return 'ALIGNED'


def regime(z_val):
    if z_val > 1.0: return 'EXTREME_CHEAP'
    if z_val > 0.5: return 'CHEAP'
    if z_val > -0.5: return 'NEUTRAL'
    if z_val > -1.0: return 'RICH'
    return 'EXTREME_RICH'


def run_strategy_simple(df, strategy_params, initial_cash=48000, initial_shares=6200):
    """Simpler procedural runner with state dict."""
    s = {
        'cash': initial_cash, 'shares': initial_shares, 'boxx': 0, 'kold': 0,
        'short_puts': [], 'short_calls': [], 'long_puts': [],
    }
    history = []
    trades = []

    p = strategy_params
    use_surprise = p.get('use_surprise_z', False)
    # Income-mode tracking — rolling 4-week premium income vs target.
    # Per production's _tp_income_mode pattern. Adjusts strike aggressiveness.
    target_weekly_income = p.get('target_weekly_income', 1500.0)
    recent_premium = []  # last N weeks of put+CC credit

    for i in range(len(df) - 30):
        idx = df.index[i]
        row = df.iloc[i]
        spot_u = row.get('UNG', 0)
        if spot_u <= 0:
            continue
        spot_k = row.get('KOLD', 0) or 0
        if isinstance(spot_k, float) and math.isnan(spot_k):
            spot_k = 0
        # Time-varying IV: per-strike via calibrated model (realized vol +
        # VIX regime + skew + term structure). Falls back to 0.55 only if
        # no realized vol available (first 30 days).
        def iv_at(K, dte, right='C'):
            return iv_for_quote(row, K, spot_u, dte, right)
        z = compute_historical_z(row, use_surprise=use_surprise)
        r = regime(z)

        # Expire short puts
        keep = []
        for sp in s['short_puts']:
            days = (idx - sp['entry']).days
            T_left = max(1, sp['dte'] - days) / 365

            # GAMMA-MANAGEMENT FORCE-CLOSE — classic tastytrade 45/21 rule.
            # Close all positions when remaining DTE drops below threshold,
            # to avoid the high-gamma zone where small price moves cause
            # large P&L swings.
            fcd = p.get('force_close_dte', 0)
            dte_left = T_left * 365
            if fcd > 0 and 0 < dte_left <= fcd:
                cv = bs_put(spot_u, sp['K'], T_left, iv_at(sp['K'], int(dte_left), 'P'))
                pnl = (sp['entry_prem'] - cv) * 100 * sp['qty'] - sp['qty'] * SPREAD_OPTION * 100
                s['cash'] += pnl
                trades.append({'date': idx, 'type': 'PUT_GAMMA_CLOSE',
                               'pnl': pnl, 'dte_left': dte_left, 'K': sp['K']})
                # IMMEDIATE RE-OPEN at fresh open_dte (continuous-roll variant
                # of the tastytrade rule). Maintains theta capture while
                # preserving gamma protection. Re-opens at same OTM% as
                # original, fresh strike vs current spot.
                if p.get('roll_on_gamma_close'):
                    new_dte = p.get('open_dte', 45)
                    otm_orig = p.get('otm_put', 0.10)
                    new_K = round(spot_u * (1 - otm_orig))
                    new_prem = bs_put(spot_u, new_K, new_dte/365,
                                      iv_at(new_K, new_dte, 'P'))
                    if new_prem > 0.05:
                        new_qty = sp['qty']
                        # Quick margin check
                        existing_coll = sum(spx['K']*100*spx['qty'] for spx in s['short_puts'])
                        if s['cash'] + new_prem*100*new_qty >= existing_coll + new_K*100*new_qty:
                            s['cash'] += new_prem*100*new_qty - new_qty*SPREAD_OPTION*100
                            s['short_puts'].append({
                                'entry': idx, 'K': new_K, 'dte': new_dte,
                                'qty': new_qty, 'entry_prem': new_prem,
                            })
                            trades.append({'date': idx, 'type': 'PUT_GAMMA_REOPEN',
                                           'pnl': 0.0, 'K': new_K, 'qty': new_qty,
                                           'dte': new_dte})
                continue

            # Take profit — configurable threshold (default 50% drop).
            # If tp_dynamic enabled, vary by vol regime per
            # [[feedback_fast_tp_in_high_vol]]: TP=70% in high vol, 50% mid,
            # 30% low (when premium is too meager for fast capture).
            tp_thresh = None
            if p.get('tp_50'):
                tp_thresh = p.get('tp_threshold', 0.5)
                if p.get('tp_dynamic'):
                    rv30 = float(row.get('rv_30') or 0.5)
                    if rv30 > 0.80:   tp_thresh = 0.7
                    elif rv30 < 0.40: tp_thresh = 0.3
                    else:             tp_thresh = 0.5
            if tp_thresh is not None and T_left > 1/365:
                cv = bs_put(spot_u, sp['K'], T_left, iv_at(sp['K'], int(T_left*365), 'P'))
                if cv < sp['entry_prem'] * tp_thresh:
                    pnl = (sp['entry_prem'] - cv) * 100 * sp['qty'] - sp['qty'] * SPREAD_OPTION * 100
                    s['cash'] += pnl
                    trades.append({'date': idx, 'type': 'PUT_TP', 'pnl': pnl})
                    continue

            # Roll down — only if remaining DTE is above min_roll_dte
            # threshold. Per [[feedback_dte_diversification]] (refined cycle
            # 20260531_140253): "let near-DTE OTM expire vs roll". Short-DTE
            # puts have little extrinsic to capture by rolling.
            # ALSO trend-aware: in uptrend, let ITM puts ride (price may
            # recover); in downtrend, rolling is protective.
            min_roll_dte = p.get('min_roll_dte', 5)  # default = old behavior
            dte_left = T_left * 365
            roll_eligible = (p.get('roll_down') and spot_u < sp['K'] * 0.98
                             and dte_left > min_roll_dte)
            # Trend-aware skip — only if flag enabled
            if roll_eligible and p.get('trend_aware_roll'):
                if bool(row.get('ung_uptrend', False)):
                    # Uptrend → let it recover, skip the roll
                    roll_eligible = False
            if roll_eligible:
                cv = bs_put(spot_u, sp['K'], T_left, iv_at(sp['K'], int(T_left*365), 'P'))
                close_pnl = (sp['entry_prem'] - cv) * 100 * sp['qty']
                s['cash'] -= cv * 100 * sp['qty']
                nk = round(spot_u * (1 - p.get('otm_put', 0.10)))
                npr = bs_put(spot_u, nk, 30/365, iv_at(nk, 30, 'P'))
                s['cash'] += npr * 100 * sp['qty'] - sp['qty'] * SPREAD_OPTION * 100
                keep.append({'entry': idx, 'K': nk, 'dte': 30, 'qty': sp['qty'], 'entry_prem': npr})
                # Roll P&L = closed leg's gain (premium collected may be future credit)
                trades.append({'date': idx, 'type': 'PUT_ROLL_DOWN', 'pnl': close_pnl,
                               'from_K': sp['K'], 'to_K': nk, 'qty': sp['qty']})
                continue

            if days >= sp['dte']:
                if spot_u < sp['K']:
                    # Assigned: P&L = premium kept - assignment loss
                    loss = (sp['K'] - spot_u) * 100 * sp['qty']
                    pnl = sp['entry_prem'] * 100 * sp['qty'] - loss
                    s['cash'] -= (sp['K'] - spot_u) * 100 * sp['qty']
                    s['shares'] += sp['qty'] * 100
                    s['cash'] -= sp['qty'] * 100 * sp['K']
                    trades.append({'date': idx, 'type': 'PUT_ASSIGN', 'qty': sp['qty'],
                                   'pnl': pnl, 'K': sp['K']})
                else:
                    # OTM expiry — full premium kept
                    pnl = sp['entry_prem'] * 100 * sp['qty']
                    trades.append({'date': idx, 'type': 'PUT_EXPIRE_OTM', 'qty': sp['qty'],
                                   'pnl': pnl})
                continue
            keep.append(sp)
        s['short_puts'] = keep

        # Expire short calls — with roll_up_call + elevator close support
        keep = []
        for sc in s['short_calls']:
            days = (idx - sc['entry']).days
            T_left = max(1, sc['dte'] - days) / 365
            # Gamma force-close for calls (45/21 rule)
            fcd_c = p.get('force_close_dte', 0)
            dte_left_c = T_left * 365
            if fcd_c > 0 and 0 < dte_left_c <= fcd_c:
                cv = bs_call(spot_u, sc['K'], T_left, iv_at(sc['K'], int(dte_left_c), 'C'))
                pnl = (sc['entry_prem'] - cv) * 100 * sc['qty'] - sc['qty'] * SPREAD_OPTION * 100
                s['cash'] += pnl
                trades.append({'date': idx, 'type': 'CALL_GAMMA_CLOSE',
                               'pnl': pnl, 'dte_left': dte_left_c, 'K': sc['K']})
                continue
            tp_thresh = None
            if p.get('tp_50'):
                tp_thresh = p.get('tp_threshold', 0.5)
                if p.get('tp_dynamic'):
                    rv30 = float(row.get('rv_30') or 0.5)
                    if rv30 > 0.80:   tp_thresh = 0.7
                    elif rv30 < 0.40: tp_thresh = 0.3
                    else:             tp_thresh = 0.5
            if tp_thresh is not None and T_left > 1/365:
                cv = bs_call(spot_u, sc['K'], T_left, iv_at(sc['K'], int(T_left*365), 'C'))
                if cv < sc['entry_prem'] * tp_thresh:
                    pnl = (sc['entry_prem'] - cv) * 100 * sc['qty']
                    s['cash'] += pnl
                    trades.append({'date': idx, 'type': 'CALL_TP', 'pnl': pnl,
                                   'K': sc['K'], 'qty': sc['qty']})
                    continue

            # ELEVATOR CLOSE (user's "Russia-spike" pattern):
            # At peak — EITHER (regime EXTREME_RICH on surprise_z)
            # OR (UNG up >30% in 60d → price-momentum spike, catches demand-
            # driven peaks that storage z misses, e.g. Russia 2022) —
            # AND short call is deep ITM with low extrinsic →
            # buy-to-close call + sell underlying shares to lock the rally
            # gain before mean reversion. Avoids waiting for assignment.
            if p.get('elevator_close') and T_left > 1/365:
                deep_itm_thresh = p.get('elevator_itm_pct', 0.05)
                is_deep_itm = spot_u > sc['K'] * (1 + deep_itm_thresh)
                cv = bs_call(spot_u, sc['K'], T_left, iv_at(sc['K'], int(T_left*365), 'C'))
                intrinsic = max(0, spot_u - sc['K'])
                extrinsic = cv - intrinsic
                ext_max = p.get('elevator_extrinsic_max', 0.15)
                low_extrinsic = extrinsic < ext_max
                # Peak triggers: tighter — require BOTH price spike AND
                # near-peak detection. "Near peak" = current spot within 5%
                # of trailing 60d high (avoids dumping early in the up-leg).
                spike_pct = row.get('ung_spike_60d', 0) or 0
                spike_thresh = p.get('elevator_spike_pct', 0.30)
                price_spike = spike_pct > spike_thresh
                # Trailing 60d high
                if i >= 60:
                    win = df['UNG'].iloc[max(0, i-60):i+1]
                    h60 = win.max()
                    near_peak_top = spot_u >= h60 * 0.95
                else:
                    near_peak_top = False
                # Mode selector: 'strict' = both, 'or' = either
                mode = p.get('elevator_mode', 'strict')
                if mode == 'strict':
                    at_peak = price_spike and near_peak_top
                else:
                    storage_peak = r == 'EXTREME_RICH'
                    at_peak = storage_peak or (price_spike and near_peak_top)
                # CORE-SHARES FLOOR: elevator can only sell shares above the
                # core floor. Prevents the wheel from spinning out of base.
                core_floor = p.get('core_shares', 0)
                available_above_core = max(0, s['shares'] - core_floor)
                core_ok = available_above_core >= sc['qty'] * 100
                if (at_peak and is_deep_itm and low_extrinsic
                        and s['shares'] >= sc['qty'] * 100
                        and core_ok):
                    s['cash'] -= cv * 100 * sc['qty'] + sc['qty'] * SPREAD_OPTION * 100
                    n_shares = sc['qty'] * 100
                    s['cash'] += n_shares * spot_u - n_shares * SPREAD_SHARE
                    s['shares'] -= n_shares
                    locked = (spot_u - sc['K']) * 100 * sc['qty']
                    # P&L vs letting it assign: spot - K per share (assignment
                    # would have given only K), minus extrinsic cost to close
                    pnl = (spot_u - sc['K'] - extrinsic) * 100 * sc['qty']
                    trades.append({
                        'date': idx, 'type': 'ELEVATOR_CLOSE',
                        'trigger': 'spike' if price_spike else 'storage_peak',
                        'K': sc['K'], 'spot': spot_u, 'qty': sc['qty'],
                        'locked_gain': locked, 'pnl': pnl,
                        'z': z, 'spike_60d': spike_pct,
                    })
                    continue

            # ROLL UP + OUT when ITM call expiring soon
            # User: 'if we are in CHEAP/NEUTRAL region, sell 30 DTE 11.5C
            # or 12C on Monday to roll the 11 expiring 12C'
            # Trigger: call is ITM (or near), <=7 days, regime CHEAP/NEUTRAL
            is_itm = spot_u > sc['K']
            near_expiry = T_left * 365 <= 7
            in_cheap_neutral = z > -0.25  # CHEAP/NEUTRAL/upward
            if (p.get('roll_up_calls') and is_itm and near_expiry
                    and in_cheap_neutral and T_left > 1/365):
                cv = bs_call(spot_u, sc['K'], T_left, iv_at(sc['K'], int(T_left*365), 'C'))
                close_pnl = (sc['entry_prem'] - cv) * 100 * sc['qty']
                s['cash'] -= cv * 100 * sc['qty']
                new_K = round(spot_u * 1.05)
                new_prem = bs_call(spot_u, new_K, 30/365, iv_at(new_K, 30, 'C'))
                if new_prem > 0.05:
                    s['cash'] += (new_prem * 100 * sc['qty']
                                  - sc['qty'] * SPREAD_OPTION * 100 * 2)
                    keep.append({
                        'entry': idx, 'K': new_K, 'dte': 30,
                        'qty': sc['qty'], 'entry_prem': new_prem,
                    })
                    trades.append({'date': idx, 'type': 'CALL_ROLL_UP',
                                   'pnl': close_pnl,
                                   'from_K': sc['K'], 'to_K': new_K, 'qty': sc['qty']})
                continue

            if days >= sc['dte']:
                if spot_u > sc['K']:
                    # Premium kept, but shares called away at K (lost spot-K)
                    lost = (spot_u - sc['K']) * 100 * sc['qty']
                    pnl = sc['entry_prem'] * 100 * sc['qty'] - lost
                    s['shares'] -= sc['qty'] * 100
                    s['cash'] += sc['qty'] * 100 * sc['K']
                    trades.append({'date': idx, 'type': 'CALL_ASSIGN',
                                   'qty': sc['qty'], 'pnl': pnl, 'K': sc['K']})
                else:
                    pnl = sc['entry_prem'] * 100 * sc['qty']
                    trades.append({'date': idx, 'type': 'CALL_EXPIRE_OTM',
                                   'qty': sc['qty'], 'pnl': pnl})
                continue
            keep.append(sc)
        s['short_calls'] = keep

        # Expire long puts
        keep = []
        for lp in s['long_puts']:
            if (idx - lp['entry']).days >= lp['dte']:
                payout = max(0, lp['K'] - spot_u) * 100 * lp['qty']
                cost = lp.get('cost', 0) * 100 * lp['qty']
                pnl = payout - cost
                s['cash'] += payout
                trades.append({'date': idx, 'type': 'LONG_PUT_EXPIRE',
                               'pnl': pnl, 'qty': lp['qty'], 'K': lp['K']})
                continue
            keep.append(lp)
        s['long_puts'] = keep

        # BOXX yield
        if s['boxx'] > 0:
            s['cash'] += s['boxx'] * 117 * 0.04 / 365

        # KOLD exit
        if s['kold'] > 0 and z > -0.3:
            s['cash'] += s['kold'] * spot_k - s['kold'] * SPREAD_SHARE
            s['kold'] = 0

        # Entry cadence — default weekly; configurable for smoother layering
        entry_cadence = p.get('entry_cadence', 7)
        if i % entry_cadence == 0:
            # Scale per-entry size so total weekly exposure stays constant
            size_scale = entry_cadence / 7.0
            otm_put = p.get('otm_put', 0.10)
            otm_call = p.get('otm_call', 0.05)
            put_qty = max(1, int(p.get('put_qty', 3) * size_scale))
            call_qty = max(1, int(p.get('call_qty', 3) * size_scale))

            # VOL-REGIME SIZING: when IV (proxied by rv_30) is high, sell
            # more premium; when depressed, back off. Premium is the alpha
            # source — chase it when it's rich.
            vol_mode = p.get('vol_aware_sizing')
            if vol_mode:
                rv30 = float(row.get('rv_30') or 0.5)
                if vol_mode == 'aggressive':
                    # Finer-grained ladder, more aggressive at extremes
                    if rv30 > 1.00:   mult = 2.0
                    elif rv30 > 0.80: mult = 1.5
                    elif rv30 > 0.60: mult = 1.2
                    elif rv30 < 0.40: mult = 0.6
                    elif rv30 < 0.30: mult = 0.4
                    else:             mult = 1.0
                else:
                    # Default 2-step (the proven winner)
                    if rv30 > 0.80:   mult = 1.5
                    elif rv30 < 0.40: mult = 0.6
                    else:             mult = 1.0
                put_qty = max(1, int(put_qty * mult))
                call_qty = max(1, int(call_qty * mult))

            # Anomaly gate — stand down entirely if 2022-style spike
            anomaly = detect_anomaly(row)
            if p.get('anomaly_standdown') and anomaly != 'NORMAL':
                trades.append({'date': idx, 'type': 'STAND_DOWN_ANOMALY',
                               'pnl': 0.0, 'regime': anomaly})
            # Sustained downtrend gate — skip ALL put-selling when UNG
            # is in confirmed downtrend (price < 200d MA AND 50d < 200d).
            # Catches the slow multi-year grind that anomaly detector misses.
            in_sustained_down = bool(row.get('ung_downtrend', False))
            if p.get('downtrend_standdown') and in_sustained_down:
                trades.append({'date': idx, 'type': 'STAND_DOWN_DOWNTREND',
                               'pnl': 0.0, 'spot': spot_u})
            # PRICE-LEVEL-AWARE DOWNTREND gate (refines simple downtrend
            # gate per user's "low UNG accumulate" rule). Only standdown
            # when downtrend STARTS FROM HIGH (UNG > high_floor); allow
            # wheel at low prices regardless of trend.
            high_floor = p.get('downtrend_high_floor', 0)
            if (p.get('downtrend_from_high_standdown')
                    and in_sustained_down
                    and spot_u > high_floor):
                trades.append({'date': idx, 'type': 'STAND_DOWN_HIGH_DT',
                               'pnl': 0.0, 'spot': spot_u})

            # DIRECT ACCUMULATION KERNEL — per user "in low UNG time we
            # accumulate". REQUIRES uptrend confirmation OR (deep cheap z
            # AND price near 252d low). UNG can bleed down for months in
            # calm regime — accumulating against a sustained downtrend is
            # exactly the falling knife the user told us never to catch.
            target_shares = p.get('target_shares', 0)
            in_downtrend = bool(row.get('ung_downtrend', False))
            in_uptrend = bool(row.get('ung_uptrend', False))
            # Only accumulate if (uptrend confirmed) OR (EXTREME cheap z + not
            # falling). Skip in downtrend even if z says cheap.
            trend_ok = in_uptrend or (z > 1.0 and not in_downtrend)
            if (target_shares > 0
                    and s['shares'] < target_shares
                    and z > -0.25
                    and not falling_knife(row)
                    and anomaly == 'NORMAL'
                    and trend_ok):
                gap = target_shares - s['shares']
                # Buy in tranches sized to z conviction: full gap at z>+0.5,
                # half at +0.25, quarter near neutral
                if z > 0.5:    tranche = gap
                elif z > 0.25: tranche = gap // 2
                else:          tranche = gap // 4
                tranche = (tranche // 100) * 100  # round to whole lots
                cost = tranche * (spot_u + SPREAD_SHARE)
                if tranche >= 100 and s['cash'] > cost + 5000:
                    s['cash'] -= cost
                    s['shares'] += tranche
                    trades.append({'date': idx, 'type': 'BUY_SHARES_ACCUMULATE',
                                   'pnl': 0.0, 'qty': tranche, 'spot': spot_u,
                                   'z': z, 'cost': cost})
            # Skip puts based on regime AND falling-knife filter
            skip_put = p.get('regime_skip_puts_z') is not None and z < p['regime_skip_puts_z']
            if p.get('anomaly_standdown') and anomaly != 'NORMAL':
                skip_put = True
            if p.get('downtrend_standdown') and in_sustained_down:
                skip_put = True
            if (p.get('downtrend_from_high_standdown')
                    and in_sustained_down
                    and spot_u > p.get('downtrend_high_floor', 0)):
                skip_put = True
            if p.get('falling_knife_filter') and falling_knife(row):
                skip_put = True
                trades.append({'date': idx, 'type': 'SKIP_PUT_FALLING_KNIFE',
                               'pnl': 0.0, 'spot': spot_u})
            if not skip_put:
                divergence = detect_divergence(row, z)
                # REGIME-AWARE STRIKE: when z says cheap (bullish), lean
                # CLOSER to ATM puts (higher delta = more income, accept
                # higher assignment prob since we want shares anyway).
                # When z says rich (bearish), go FARTHER OTM (safer).
                # Continuous adjustment by z, replaces fixed otm_put.
                if p.get('regime_aware_strike'):
                    if z > 1.0:   otm_put = 0.02
                    elif z > 0.5: otm_put = 0.05
                    elif z > 0.0: otm_put = 0.08
                    elif z > -0.5: otm_put = 0.10
                    elif z > -1.0: otm_put = 0.13
                    else:          otm_put = 0.18
                # INCOME-MODE strike push: if rolling weekly income < 60% of
                # target, push strike CLOSER to ATM (more income, accept
                # higher assignment prob). If above 120% of target, push
                # FARTHER OTM (we have buffer, prefer safety).
                if p.get('income_mode_strike') and len(recent_premium) >= 4:
                    avg_weekly = sum(recent_premium[-4:]) / 4
                    ratio = avg_weekly / target_weekly_income
                    if ratio < 0.6:    otm_put = max(0.02, otm_put - 0.04)
                    elif ratio > 1.2:  otm_put = min(0.25, otm_put + 0.04)
                # Z-scaled sizing — bigger when more cheap.
                if p.get('z_scaled_sizing'):
                    if z > 0.75:   put_qty = int(p.get('put_qty', 3) * 3)
                    elif z > 0.25: put_qty = int(p.get('put_qty', 3) * 2)
                    elif z > -0.25: put_qty = int(p.get('put_qty', 3) * 1)
                    else: put_qty = max(1, int(p.get('put_qty', 3) * 0.5))
                # DIVERGENCE ALPHA: panic-selling through cheap signals =
                # high-conviction entry. Bump size + tighten strike.
                # Per [[project_fundamental_divergence_alpha]]
                effective_otm = otm_put
                if p.get('divergence_trading') and divergence == 'PANIC_BUY_OPP':
                    put_qty = int(put_qty * 1.5)
                    effective_otm = max(0.02, otm_put / 2)  # closer to money
                    trades.append({'date': idx, 'type': 'PANIC_BUY_DETECTED',
                                   'pnl': 0.0, 'spot': spot_u, 'z': z})
                # CALM INCOME BOOST: when sitting in cash with 0 shares in
                # NEUTRAL-CHEAP regime + not falling-knife + not anomaly,
                # write closer-to-money puts. Acceptable assignment risk
                # because we WANT shares back if cheap, and premium income
                # ~3x higher than 10% OTM.
                elif (p.get('calm_boost')
                        and s['shares'] == 0
                        and -0.25 < z < 0.75
                        and not falling_knife(row)
                        and anomaly == 'NORMAL'):
                    effective_otm = 0.03  # 3% OTM, ~30 delta
                    put_qty = max(2, int(put_qty * 0.6))  # smaller qty
                    trades.append({'date': idx, 'type': 'CALM_BOOST_PUT',
                                   'pnl': 0.0, 'z': z, 'spot': spot_u})
                K = round(spot_u * (1 - effective_otm))
                # Tunable open-DTE (default 30; tastytrade rule uses 45).
                open_dte = p.get('open_dte', 30)
                if p.get('vol_aware_dte'):
                    rv30 = float(row.get('rv_30') or 0.5)
                    if rv30 > 0.80:   open_dte = 60
                    else:             open_dte = 45
                prem = bs_put(spot_u, K, open_dte/365, iv_at(K, open_dte, 'P'))
                # KELLY SIZING with conviction-aware "firmness" multiplier.
                # Per user: high vol + high model conviction → SIZE UP, not down.
                if p.get('kelly_sizing') and prem > 0.05:
                    iv_use = iv_at(K, open_dte, 'P')
                    conv_adj = model_conviction(row, z, anomaly) if p.get('kelly_conviction') else 0.0
                    kelly_q = kelly_qty_short_put(
                        spot_u, K, open_dte, iv_use,
                        cash_available=s['cash'],
                        premium=prem,
                        model_conviction=conv_adj,
                    )
                    if p.get('kelly_firmness'):
                        firm = firmness_multiplier(row, z, anomaly)
                        kelly_q = int(kelly_q * firm)
                    put_qty = max(0, min(kelly_q, int(p.get('kelly_max_qty', 20))))
                if prem > 0.05:
                    # MARGIN CHECK: cash-secured put requires K*100*qty collateral.
                    # Available collateral = current cash + premium received -
                    # existing short-put obligations. Refuse trade if would push
                    # cash + collateral_used past available.
                    if p.get('margin_check', True):
                        existing_put_collateral = sum(
                            sp['K'] * 100 * sp['qty'] for sp in s['short_puts']
                        )
                        new_collateral = K * 100 * put_qty
                        credit = prem * 100 * put_qty - put_qty * SPREAD_OPTION * 100
                        # Cash floor after this trade: cash + credit must cover
                        # total put obligations
                        if s['cash'] + credit < existing_put_collateral + new_collateral:
                            trades.append({'date': idx, 'type': 'OPEN_PUT_REJECTED_MARGIN',
                                           'pnl': 0.0, 'K': K, 'qty': put_qty,
                                           'reason': 'insufficient_buying_power'})
                            put_qty = 0
                    if put_qty > 0:
                        # DTE diversification — per [[feedback_dte_diversification]]:
                        # spread across DTEs not pile at 30. Each DTE gets a slice
                        # sized by 1/N.
                        dte_ladder = p.get('dte_ladder', [p.get('open_dte', 30)])
                        n_dtes = len(dte_ladder)
                        per_dte_qty = max(1, put_qty // n_dtes)
                        # Re-margin-check the FULL allocation
                        for dte_choice in dte_ladder:
                            # Re-price for this specific DTE
                            iv_dte = iv_at(K, dte_choice, 'P')
                            prem_dte = bs_put(spot_u, K, dte_choice/365, iv_dte)
                            if prem_dte < 0.05:
                                continue
                            credit_dte = prem_dte * 100 * per_dte_qty - per_dte_qty * SPREAD_OPTION * 100
                            s['cash'] += credit_dte
                            s['short_puts'].append({'entry': idx, 'K': K, 'dte': dte_choice,
                                                    'qty': per_dte_qty, 'entry_prem': prem_dte})
                            trades.append({'date': idx, 'type': 'OPEN_PUT',
                                           'pnl': 0.0, 'credit': credit_dte,
                                           'K': K, 'qty': per_dte_qty, 'dte': dte_choice})

            # CCs (only if have UNCOVERED shares ABOVE core — covered-call
            # ONLY [[feedback_covered_calls_only]]; core shares are
            # protected from CC writing to avoid bleed-out via assignment.)
            existing_cc_qty = sum(sc['qty'] for sc in s['short_calls'])
            core_floor = p.get('core_shares', 0)
            uncovered_shares = max(0, s['shares'] - core_floor - existing_cc_qty * 100)
            if uncovered_shares >= 100:
                use_itm = (p.get('aggressive_itm_cc_z') is not None
                           and z < p['aggressive_itm_cc_z'])
                # REGIME-AWARE CC STRIKE: when z is BEARISH/rich, push CC
                # strike closer or ITM (force assignment, divest shares).
                # When z is BULLISH/cheap, push CC FARTHER OTM (preserve
                # upside, don't sacrifice shares cheaply).
                if p.get('regime_aware_strike'):
                    if z > 1.0:    otm_call = 0.15   # extreme cheap → far OTM
                    elif z > 0.5:  otm_call = 0.10   # cheap → 10% OTM
                    elif z > 0.0:  otm_call = 0.07
                    elif z > -0.5: otm_call = 0.05   # neutral → standard 5%
                    elif z > -1.0: otm_call = 0.00   # rich → ATM
                    else:          otm_call = -0.05  # extreme rich → 5% ITM
                # Divergence: when crowd is euphorically buying through rich
                # signals, sell DEEPER ITM CCs to lock the move via assignment.
                # Per [[project_fundamental_divergence_alpha]]
                divergence_cc = detect_divergence(row, z)
                if p.get('divergence_trading') and divergence_cc == 'EUPHORIC_SELL_OPP':
                    use_itm = True
                    effective_otm = -0.08  # 8% ITM, near-certain assignment
                    trades.append({'date': idx, 'type': 'EUPHORIC_SELL_DETECTED',
                                   'pnl': 0.0, 'spot': spot_u, 'z': z})
                else:
                    effective_otm = p.get('itm_cc_pct', otm_call) if use_itm else otm_call
                K = round(spot_u * (1 + effective_otm))
                qty = min(call_qty, uncovered_shares // 100)
                cc_dte = p.get('open_dte', 30)
                if p.get('vol_aware_dte'):
                    rv30 = float(row.get('rv_30') or 0.5)
                    if rv30 > 0.80:   cc_dte = 60
                    else:             cc_dte = 45
                prem = bs_call(spot_u, K, cc_dte/365, iv_at(K, cc_dte, 'C'))
                # KELLY SIZING for CCs (with conviction + firmness)
                if p.get('kelly_sizing') and prem > 0.05:
                    iv_use = iv_at(K, cc_dte, 'C')
                    conv_adj = model_conviction(row, z, anomaly) if p.get('kelly_conviction') else 0.0
                    kelly_q = kelly_qty_covered_call(
                        spot_u, K, cc_dte, iv_use,
                        uncovered_shares=uncovered_shares,
                        premium=prem,
                        model_conviction=conv_adj,
                    )
                    if p.get('kelly_firmness'):
                        # For CCs, firmness is inverted — bearish alignment
                        # = sell MORE CCs, bullish = fewer.
                        firm = firmness_multiplier(row, z, anomaly)
                        # Invert: 2.0 (bullish) → 0.5 (sell few CCs)
                        cc_firm = 1.0 / max(0.5, min(2.5, firm))
                        kelly_q = int(kelly_q * cc_firm)
                    qty = max(0, min(kelly_q, int(p.get('kelly_max_qty', 20))))
                if prem > 0.05:
                    credit = prem * 100 * qty - qty * SPREAD_OPTION * 100
                    s['cash'] += credit
                    s['short_calls'].append({'entry': idx, 'K': K, 'dte': cc_dte,
                                             'qty': qty, 'entry_prem': prem,
                                             'is_itm_aggressive': use_itm})
                    open_kind = 'OPEN_ITM_CC' if use_itm else 'OPEN_CC'
                    trades.append({'date': idx, 'type': open_kind,
                                   'pnl': 0.0, 'credit': credit,
                                   'K': K, 'qty': qty, 'z': z})

            # EXTREME_RICH bearish stack
            if p.get('bearish_stack') and r == 'EXTREME_RICH':
                if not s['long_puts']:
                    Kp = round(spot_u * 0.95)
                    cost = bs_put(spot_u, Kp, 90/365, iv_at(Kp, 90, 'P'))
                    qty = 3
                    debit = cost * 100 * qty + qty * SPREAD_OPTION * 100
                    s['cash'] -= debit
                    s['long_puts'].append({'entry': idx, 'K': Kp, 'dte': 90, 'qty': qty, 'cost': cost})
                    trades.append({'date': idx, 'type': 'OPEN_LONG_PUT',
                                   'pnl': -debit, 'K': Kp, 'qty': qty})

            # TAIL-HEDGE FLOOR (production-port #5) — maintain minimum long
            # put count regardless of regime. Production default = 2 LEAPS.
            # Backtest uses 90 DTE puts as proxy.
            tail_floor = p.get('tail_hedge_floor', 0)
            existing_long_qty = sum(lp.get('qty', 0) for lp in s['long_puts'])
            if tail_floor > 0 and existing_long_qty < tail_floor:
                # Add to floor — pick OTM strike, 90 DTE
                need = tail_floor - existing_long_qty
                Kp = round(spot_u * 0.92)
                cost = bs_put(spot_u, Kp, 90/365, iv_at(Kp, 90, 'P'))
                debit = cost * 100 * need + need * SPREAD_OPTION * 100
                if s['cash'] > debit + 1000:  # leave $1K buffer
                    s['cash'] -= debit
                    s['long_puts'].append({'entry': idx, 'K': Kp, 'dte': 90,
                                           'qty': need, 'cost': cost})
                    trades.append({'date': idx, 'type': 'OPEN_LONG_PUT_FLOOR',
                                   'pnl': -debit, 'K': Kp, 'qty': need})

                # Guard against NaN spot_k or NaN nav
                if s['kold'] == 0 and spot_k > 0 and pd.notna(spot_k):
                    nav = s['cash'] + s['shares'] * spot_u + s['boxx'] * 117
                    if pd.isna(nav) or nav <= 0:
                        tq = 0
                    else:
                        try:
                            tq = int(nav * 0.03 / spot_k)
                        except (ValueError, OverflowError):
                            tq = 0
                    if tq > 5 and s['cash'] > tq * spot_k + 200:
                        s['kold'] += tq
                        s['cash'] -= tq * spot_k + tq * SPREAD_SHARE

            # BOXX management
            if p.get('boxx'):
                excess = s['cash'] - 20000
                if excess > 5000:
                    nb = int(excess * 0.6 / 117)
                    if nb >= 10:
                        s['boxx'] += nb
                        s['cash'] -= nb * 117 + nb * SPREAD_SHARE

        # Track weekly premium for income-mode awareness (every Monday-ish)
        if i % 7 == 0 and i > 0:
            # Sum credits in last 7 trades from any opens that fired this cycle
            recent = trades[-10:]  # rough window
            wk_credit = sum(t.get('credit', 0) for t in recent
                           if t.get('type', '').startswith('OPEN_')
                           and (idx - t.get('date', idx)).days <= 7)
            recent_premium.append(wk_credit)
            if len(recent_premium) > 12:
                recent_premium.pop(0)
        nav = s['cash'] + s['shares'] * spot_u + s['boxx'] * 117 + s['kold'] * spot_k
        history.append({
            'date': idx, 'spot': spot_u, 'z': z, 'regime': r,
            'cash': s['cash'], 'shares': s['shares'], 'boxx': s['boxx'], 'kold': s['kold'],
            'nav': nav, 'short_puts': len(s['short_puts']), 'short_calls': len(s['short_calls']),
        })

    return pd.DataFrame(history), pd.DataFrame(trades)


STRATEGIES = {
    'naive_atm': {
        'otm_put': 0.0, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
    },
    'otm_managed': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'roll_down': True,
    },
    'regime_aware': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'roll_down': True,
        'regime_skip_puts_z': -0.5, 'bearish_stack': True, 'boxx': True,
    },
    'deep_otm_passive': {
        'otm_put': 0.20, 'otm_call': 0.10, 'put_qty': 3, 'call_qty': 3,
        'tp_50': True,
    },
    # User's Monday-roll-up strategy: when CHEAP/NEUTRAL and have ITM calls
    # expiring within a week, roll up + out to higher strike 30 DTE.
    # Lets the position stay alive in bullish regime without giving up shares.
    'regime_aware_roll_up': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'roll_down': True, 'roll_up_calls': True,
        'regime_skip_puts_z': -0.5, 'bearish_stack': True, 'boxx': True,
    },
    # User: 'sell ITM CCs to take off shares when Z says so'
    # When regime is RICH or worse, write 5% ITM CCs to FORCE share assignment
    # Aggressive divestment of share exposure when model says expensive
    'aggressive_unload_on_rich': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'roll_down': True,
        'regime_skip_puts_z': -0.5, 'bearish_stack': True, 'boxx': True,
        'aggressive_itm_cc_z': -0.25,  # if z < -0.25, sell 5% ITM CCs
        'itm_cc_pct': -0.05,           # 5% ITM strike
    },
    # NEW: use seasonal-detrended z (removes annual sine contamination)
    # Same machinery as regime_aware but z reflects deviation from
    # SEASONAL expectation, not raw level.
    'regime_aware_surprise': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'roll_down': True,
        'regime_skip_puts_z': -0.5, 'bearish_stack': True, 'boxx': True,
        'use_surprise_z': True,
    },
    # NEW: elevator close — at peak spike (EXTREME_RICH surprise_z), if short
    # call is deep ITM with no extrinsic, buy back + sell shares to lock the
    # rally gain before mean reversion.
    'elevator_close_surprise': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'roll_down': True,
        'regime_skip_puts_z': -0.5, 'bearish_stack': True, 'boxx': True,
        'use_surprise_z': True,
        'elevator_close': True, 'elevator_itm_pct': 0.05,
        'elevator_extrinsic_max': 0.15,
        'elevator_mode': 'strict',   # require spike + near-peak top
    },
    # Looser: ANY rich signal triggers elevator
    'elevator_or_mode': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'roll_down': True,
        'regime_skip_puts_z': -0.5, 'bearish_stack': True, 'boxx': True,
        'use_surprise_z': True,
        'elevator_close': True, 'elevator_itm_pct': 0.05,
        'elevator_extrinsic_max': 0.15,
        'elevator_mode': 'or',
    },
    # Attribution finding (cycle 205c): PUT_ROLL_DOWN had 0% win rate,
    # losing $755-924/fire across all strategies. Test disabling it.
    'no_rolldown': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'roll_down': False,
        'regime_skip_puts_z': -0.5, 'bearish_stack': False, 'boxx': True,
        'use_surprise_z': True,
    },
    # Best-of: surprise_z + no rolldown + no tail hedge + elevator close
    'best_of_attribution': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'roll_down': False,
        'regime_skip_puts_z': -0.5, 'bearish_stack': False, 'boxx': True,
        'use_surprise_z': True,
        'elevator_close': True, 'elevator_itm_pct': 0.05,
        'elevator_extrinsic_max': 0.15,
        'elevator_mode': 'strict',
    },
    # Aims for 27% annualized in NORMAL regime. Encodes user rules:
    # - covered calls only (already global)
    # - never falling knife (require momentum confirmation)
    # - z-scaled sizing (bigger when conviction stacks)
    # - 2022-style anomaly: stand down, don't fight
    # - elevator close at peaks
    'smooth_27': {
        'otm_put': 0.08, 'otm_call': 0.05, 'put_qty': 4, 'call_qty': 5,
        'tp_50': True, 'roll_down': True, 'roll_up_calls': True,
        'regime_skip_puts_z': -0.5, 'boxx': True,
        'use_surprise_z': True,
        'falling_knife_filter': True,
        'anomaly_standdown': True,
        'z_scaled_sizing': True,
        'elevator_close': True, 'elevator_itm_pct': 0.05,
        'elevator_extrinsic_max': 0.15,
        'elevator_mode': 'strict',
    },
    # v2 adds: direct accumulation when z cheap + low shares — fixes
    # smooth_27's structural problem of losing entire share base over
    # 5 years (6200 → 0 by 2025) because passive puts at 8% OTM in calm
    # regime rarely assign. target_shares=4000 maintains active wheel base.
    'smooth_27_v2': {
        'otm_put': 0.08, 'otm_call': 0.05, 'put_qty': 4, 'call_qty': 5,
        'tp_50': True, 'roll_down': True, 'roll_up_calls': True,
        'regime_skip_puts_z': -0.5, 'boxx': True,
        'use_surprise_z': True,
        'falling_knife_filter': True,
        'anomaly_standdown': True,
        'z_scaled_sizing': True,
        'elevator_close': True, 'elevator_itm_pct': 0.05,
        'elevator_extrinsic_max': 0.15,
        'elevator_mode': 'strict',
        'target_shares': 4000,
    },
    # v3 adds core_shares floor — elevator + CC won't drain shares below
    # core. Keeps wheel base alive through spikes for future income.
    # Same chassis as regime_aware_roll_up (the empirical winner) plus
    # divergence trading — bigger size + ATM strikes when fundamentals
    # diverge from crowd panic/euphoria.
    'roll_up_plus_divergence': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'roll_down': True, 'roll_up_calls': True,
        'regime_skip_puts_z': -0.5, 'bearish_stack': True, 'boxx': True,
        'divergence_trading': True,
    },
    # Champion chassis + calm_boost: write closer-to-money puts when
    # sitting in cash in NEUTRAL regime, harvesting premium that 10% OTM
    # puts can't capture at sub-$15 UNG. Tested only in calm regime (no
    # shares, neutral z, not falling, not anomaly).
    'roll_up_calm_boost': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'roll_down': True, 'roll_up_calls': True,
        'regime_skip_puts_z': -0.5, 'bearish_stack': True, 'boxx': True,
        'calm_boost': True,
    },
    # DTE diversification per [[feedback_dte_diversification]]: spread
    # weekly put writes across 7/14/30/45 DTEs. Each DTE gets put_qty/4
    # contracts. Smoother theta curve, less concentration at one expiry.
    'roll_up_dte_ladder': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 8, 'call_qty': 5,
        'tp_50': True, 'roll_down': True, 'roll_up_calls': True,
        'regime_skip_puts_z': -0.5, 'bearish_stack': True, 'boxx': True,
        'dte_ladder': [7, 14, 30, 45],
    },
    # DTE ladder + DTE-aware rolls: don't roll puts with <14 DTE left
    # (let them expire OTM or assign). Per [[feedback_dte_diversification]]
    # "let near-DTE OTM expire vs roll".
    'roll_up_dte_smart': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 8, 'call_qty': 5,
        'tp_50': True, 'roll_down': True, 'roll_up_calls': True,
        'regime_skip_puts_z': -0.5, 'bearish_stack': True, 'boxx': True,
        'dte_ladder': [7, 14, 30, 45],
        'min_roll_dte': 14,
    },
    # Trend-aware rolling: roll only when ITM AND in downtrend; in uptrend
    # let ITM puts ride for recovery. Per refined rule in
    # [[feedback_dte_diversification]] cycle 20260531_140253.
    'roll_up_trend_aware': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'roll_down': True, 'roll_up_calls': True,
        'regime_skip_puts_z': -0.5, 'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
    },
    # Hybrid: spike-period DTE-smart + calm-period aggressive ITM CC.
    # Idea is to capture both regime advantages.
    'champion_hybrid': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 8, 'call_qty': 5,
        'tp_50': True, 'roll_down': True, 'roll_up_calls': True,
        'regime_skip_puts_z': -0.5, 'bearish_stack': True, 'boxx': True,
        'dte_ladder': [7, 14, 30, 45],
        'min_roll_dte': 14,
        'trend_aware_roll': True,
        # Calm-regime ITM CC (the aggressive_unload edge)
        'aggressive_itm_cc_z': -0.25,
        'itm_cc_pct': -0.05,
    },
    # Strip DTE ladder — test if hybrid's Sharpe is from itm_cc + trend_aware
    # alone (without ladder churn that ate spike returns)
    'champion_no_ladder': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'roll_down': True, 'roll_up_calls': True,
        'regime_skip_puts_z': -0.5, 'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'aggressive_itm_cc_z': -0.25,
        'itm_cc_pct': -0.05,
    },
    # Hybrid + elevator_close — should catch peak gains during spikes
    'champion_with_elevator': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'roll_down': True, 'roll_up_calls': True,
        'regime_skip_puts_z': -0.5, 'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'aggressive_itm_cc_z': -0.25,
        'itm_cc_pct': -0.05,
        'elevator_close': True, 'elevator_itm_pct': 0.05,
        'elevator_extrinsic_max': 0.15,
        'elevator_mode': 'strict',
    },
    # Ablation revealed aggressive_itm_cc costs $89K in champion_with_elevator
    # AND elevator_close adds ~$0. Strip both, keep the rest of the chassis.
    'champion_clean': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'roll_down': True, 'roll_up_calls': True,
        'regime_skip_puts_z': -0.5, 'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
    },
    # Tightened: only fire aggressive_itm_cc at z<-1.0 (EXTREME rich) not
    # z<-0.25. Hypothesis: keep some Sharpe boost without giving up as
    # much return.
    'champion_tight_itm': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'roll_down': True, 'roll_up_calls': True,
        'regime_skip_puts_z': -0.5, 'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'aggressive_itm_cc_z': -1.0,  # extreme rich only
        'itm_cc_pct': -0.05,
    },
    # tight_itm + elevator — test if elevator adds value when combined
    # with the tighter ITM trigger (different fire-context than full champion)
    'champion_tight_plus_elev': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'roll_down': True, 'roll_up_calls': True,
        'regime_skip_puts_z': -0.5, 'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'aggressive_itm_cc_z': -1.0,
        'itm_cc_pct': -0.05,
        'elevator_close': True, 'elevator_itm_pct': 0.05,
        'elevator_extrinsic_max': 0.15, 'elevator_mode': 'strict',
    },
    # Same chassis, but enter twice-weekly (i%3) with half size — smoother
    # cadence, same total exposure. Tests if entry-day concentration matters.
    'champion_biweekly': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'roll_down': True, 'roll_up_calls': True,
        'regime_skip_puts_z': -0.5, 'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'aggressive_itm_cc_z': -1.0,
        'itm_cc_pct': -0.05,
        'elevator_close': True, 'elevator_itm_pct': 0.05,
        'elevator_extrinsic_max': 0.15, 'elevator_mode': 'strict',
        'entry_cadence': 3,
    },
    # Vol-aware sizing: scale put/call qty by realized vol regime.
    # High vol (>80%) → 1.5x, Low vol (<40%) → 0.6x. Chase premium when
    # rich (matches user "make sure we use options well").
    'champion_vol_aware': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'roll_down': True, 'roll_up_calls': True,
        'regime_skip_puts_z': -0.5, 'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'aggressive_itm_cc_z': -1.0,
        'itm_cc_pct': -0.05,
        'elevator_close': True, 'elevator_itm_pct': 0.05,
        'elevator_extrinsic_max': 0.15, 'elevator_mode': 'strict',
        'vol_aware_sizing': True,
    },
    # Drop regime_skip_puts (ablation showed it costs $20K when combined
    # with vol_aware — gates fight each other). Keep everything else.
    'champion_vol_unleashed': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'roll_down': True, 'roll_up_calls': True,
        'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'aggressive_itm_cc_z': -1.0,
        'itm_cc_pct': -0.05,
        'elevator_close': True, 'elevator_itm_pct': 0.05,
        'elevator_extrinsic_max': 0.15, 'elevator_mode': 'strict',
        'vol_aware_sizing': True,
    },
    # Strip elevator_close + boxx + roll_down (ablation flagged $12K drag).
    # Keep tp_50, roll_up_calls (the proven big winners) + vol_aware.
    'champion_vol_pure': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'roll_down': False, 'roll_up_calls': True,
        'bearish_stack': True,
        'trend_aware_roll': True,
        'vol_aware_sizing': True,
    },
    # Vol_unleashed with TP at 30% (take profits faster)
    'champion_vol_tp30': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'tp_threshold': 0.3,
        'roll_down': True, 'roll_up_calls': True,
        'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'aggressive_itm_cc_z': -1.0, 'itm_cc_pct': -0.05,
        'elevator_close': True, 'elevator_itm_pct': 0.05,
        'elevator_extrinsic_max': 0.15, 'elevator_mode': 'strict',
        'vol_aware_sizing': True,
    },
    # Vol_unleashed with TP at 70% (let winners run)
    'champion_vol_tp70': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'tp_threshold': 0.7,
        'roll_down': True, 'roll_up_calls': True,
        'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'aggressive_itm_cc_z': -1.0, 'itm_cc_pct': -0.05,
        'elevator_close': True, 'elevator_itm_pct': 0.05,
        'elevator_extrinsic_max': 0.15, 'elevator_mode': 'strict',
        'vol_aware_sizing': True,
    },
    # Push the TP-fast hypothesis further — TP at 80% and 90%
    'champion_vol_tp80': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'tp_threshold': 0.8,
        'roll_down': True, 'roll_up_calls': True,
        'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'aggressive_itm_cc_z': -1.0, 'itm_cc_pct': -0.05,
        'elevator_close': True, 'elevator_itm_pct': 0.05,
        'elevator_extrinsic_max': 0.15, 'elevator_mode': 'strict',
        'vol_aware_sizing': True,
    },
    'champion_vol_tp90': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'tp_threshold': 0.9,
        'roll_down': True, 'roll_up_calls': True,
        'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'aggressive_itm_cc_z': -1.0, 'itm_cc_pct': -0.05,
        'elevator_close': True, 'elevator_itm_pct': 0.05,
        'elevator_extrinsic_max': 0.15, 'elevator_mode': 'strict',
        'vol_aware_sizing': True,
    },
    # Dynamic TP: 70% in high vol, 50% mid, 30% low vol
    'champion_vol_tp_dyn': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'tp_dynamic': True,
        'roll_down': True, 'roll_up_calls': True,
        'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'aggressive_itm_cc_z': -1.0, 'itm_cc_pct': -0.05,
        'elevator_close': True, 'elevator_itm_pct': 0.05,
        'elevator_extrinsic_max': 0.15, 'elevator_mode': 'strict',
        'vol_aware_sizing': True,
    },
    # tp_dyn + frequent ITM CC (z<-0.25) — try to push Sharpe up
    'champion_dyn_freq_itm': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'tp_dynamic': True,
        'roll_down': True, 'roll_up_calls': True,
        'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'aggressive_itm_cc_z': -0.25, 'itm_cc_pct': -0.05,  # frequent
        'elevator_close': True, 'elevator_itm_pct': 0.05,
        'elevator_extrinsic_max': 0.15, 'elevator_mode': 'strict',
        'vol_aware_sizing': True,
    },
    # Probe itm_cc_pct depth in dyn_freq_itm chassis
    'champion_dyn_itm_3pct': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'tp_dynamic': True,
        'roll_down': True, 'roll_up_calls': True,
        'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'aggressive_itm_cc_z': -0.25, 'itm_cc_pct': -0.03,  # closer to ATM
        'elevator_close': True, 'elevator_itm_pct': 0.05,
        'elevator_extrinsic_max': 0.15, 'elevator_mode': 'strict',
        'vol_aware_sizing': True,
    },
    'champion_dyn_itm_10pct': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'tp_dynamic': True,
        'roll_down': True, 'roll_up_calls': True,
        'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'aggressive_itm_cc_z': -0.25, 'itm_cc_pct': -0.10,  # deeper ITM
        'elevator_close': True, 'elevator_itm_pct': 0.05,
        'elevator_extrinsic_max': 0.15, 'elevator_mode': 'strict',
        'vol_aware_sizing': True,
    },
    'champion_dyn_itm_15pct': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'tp_dynamic': True,
        'roll_down': True, 'roll_up_calls': True,
        'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'aggressive_itm_cc_z': -0.25, 'itm_cc_pct': -0.15,  # very deep
        'elevator_close': True, 'elevator_itm_pct': 0.05,
        'elevator_extrinsic_max': 0.15, 'elevator_mode': 'strict',
        'vol_aware_sizing': True,
    },
    'champion_dyn_itm_20pct': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'tp_dynamic': True,
        'roll_down': True, 'roll_up_calls': True,
        'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'aggressive_itm_cc_z': -0.25, 'itm_cc_pct': -0.20,  # 20% ITM
        'elevator_close': True, 'elevator_itm_pct': 0.05,
        'elevator_extrinsic_max': 0.15, 'elevator_mode': 'strict',
        'vol_aware_sizing': True,
    },
    'champion_dyn_itm_30pct': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'tp_dynamic': True,
        'roll_down': True, 'roll_up_calls': True,
        'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'aggressive_itm_cc_z': -0.25, 'itm_cc_pct': -0.30,  # 30% ITM
        'elevator_close': True, 'elevator_itm_pct': 0.05,
        'elevator_extrinsic_max': 0.15, 'elevator_mode': 'strict',
        'vol_aware_sizing': True,
    },
    # Hybrid: fixed TP=70% (return leader's lever) + deep ITM (Sharpe
    # leader's lever). Test if combining captures both.
    'champion_tp70_itm20': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'tp_threshold': 0.7,
        'roll_down': True, 'roll_up_calls': True,
        'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'aggressive_itm_cc_z': -0.25, 'itm_cc_pct': -0.20,
        'elevator_close': True, 'elevator_itm_pct': 0.05,
        'elevator_extrinsic_max': 0.15, 'elevator_mode': 'strict',
        'vol_aware_sizing': True,
    },
    # Robustness-target: dyn_itm_30pct (Sharpe leader) + anomaly standdown
    # + falling-knife filter. Goal: minimize worst-window drawdown.
    'champion_robust': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'tp_dynamic': True,
        'roll_down': True, 'roll_up_calls': True,
        'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'aggressive_itm_cc_z': -0.25, 'itm_cc_pct': -0.30,
        'elevator_close': True, 'elevator_itm_pct': 0.05,
        'elevator_extrinsic_max': 0.15, 'elevator_mode': 'strict',
        'vol_aware_sizing': True,
        'anomaly_standdown': True,
        'falling_knife_filter': True,
    },
    # Add sustained-downtrend gate: skip put-selling when UNG < 200d MA
    # AND 50d < 200d (catches the slow grind regimes).
    'champion_robust_dt': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'tp_dynamic': True,
        'roll_down': True, 'roll_up_calls': True,
        'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'aggressive_itm_cc_z': -0.25, 'itm_cc_pct': -0.30,
        'elevator_close': True, 'elevator_itm_pct': 0.05,
        'elevator_extrinsic_max': 0.15, 'elevator_mode': 'strict',
        'vol_aware_sizing': True,
        'downtrend_standdown': True,
    },
    # Price-level-aware standdown: only stop wheel when downtrend STARTS
    # from high prices (UNG > $30). Allows continued operation at low
    # prices per user's "low UNG accumulate" rule.
    # Classic tastytrade 45/21 rule
    'tastytrade_45_21': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'tp_threshold': 0.5,
        'roll_down': False, 'roll_up_calls': False,
        'bearish_stack': False, 'boxx': True,
        'open_dte': 45, 'force_close_dte': 21,
    },
    # Variant: keep good kernels + 45/21 management
    'tastytrade_full': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'tp_threshold': 0.5,
        'roll_down': True, 'roll_up_calls': True,
        'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'open_dte': 45, 'force_close_dte': 21,
        'vol_aware_sizing': True,
    },
    # Open at 45 only (no force-close — control)
    'open_45_only': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'tp_threshold': 0.5,
        'roll_down': True, 'roll_up_calls': True,
        'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'open_dte': 45,
        'vol_aware_sizing': True,
    },
    # 45/21 with immediate re-roll (user-suggested fix)
    'tt_45_21_reroll': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'tp_threshold': 0.5,
        'roll_down': True, 'roll_up_calls': True,
        'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'open_dte': 45, 'force_close_dte': 21,
        'roll_on_gamma_close': True,
        'vol_aware_sizing': True,
    },
    # User's 37/14 suggestion
    'tt_37_14_reroll': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'tp_threshold': 0.5,
        'roll_down': True, 'roll_up_calls': True,
        'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'open_dte': 37, 'force_close_dte': 14,
        'roll_on_gamma_close': True,
        'vol_aware_sizing': True,
    },
    # Longer pair — 60/30
    'tt_60_30_reroll': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'tp_threshold': 0.5,
        'roll_down': True, 'roll_up_calls': True,
        'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'open_dte': 60, 'force_close_dte': 30,
        'roll_on_gamma_close': True,
        'vol_aware_sizing': True,
    },
    # Shorter pair — 30/14
    'tt_30_14_reroll': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'tp_threshold': 0.5,
        'roll_down': True, 'roll_up_calls': True,
        'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'open_dte': 30, 'force_close_dte': 14,
        'roll_on_gamma_close': True,
        'vol_aware_sizing': True,
    },
    # Very short — 21/7
    'tt_21_7_reroll': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'tp_threshold': 0.5,
        'roll_down': True, 'roll_up_calls': True,
        'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'open_dte': 21, 'force_close_dte': 7,
        'roll_on_gamma_close': True,
        'vol_aware_sizing': True,
    },
    # REGIME-AWARE DTE — sharp move → longer DTE, calm → shorter
    # rv > 80% → 60d, rv 60-80% → 45d, rv 40-60% → 30d, rv <40% → 21d
    'regime_dte': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'tp_threshold': 0.5,
        'roll_down': True, 'roll_up_calls': True,
        'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'vol_aware_dte': True,
        'vol_aware_sizing': True,
    },
    # PRODUCTION-PORT: Kelly position sizing (replaces fixed qty)
    'kelly_sized': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'tp_threshold': 0.5,
        'roll_down': True, 'roll_up_calls': True,
        'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'kelly_sizing': True,
        'kelly_max_qty': 15,
    },
    # Kelly + best other features (closer-to-production chassis)
    'kelly_champion': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'tp_dynamic': True,
        'roll_down': True, 'roll_up_calls': True,
        'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'kelly_sizing': True,
        'kelly_max_qty': 15,
        'open_dte': 45,
        'vol_aware_dte': True,
    },
    # Per user (cycle 20260602): "when extreme volatility happens, based on
    # history and modeling, the firmness of confidence will get a higher".
    # Kelly + conviction adjustment to BS p_otm based on aligned signals.
    'kelly_firm': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'tp_dynamic': True,
        'roll_down': True, 'roll_up_calls': True,
        'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'kelly_sizing': True,
        'kelly_conviction': True,
        'kelly_max_qty': 20,
        'open_dte': 45,
        'vol_aware_dte': True,
    },
    # Full firmness — multiplier-based, scales kelly qty up to 2x at high
    # conviction + high vol, down to 0.5x when signals conflict.
    'kelly_firmness': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'tp_dynamic': True,
        'roll_down': True, 'roll_up_calls': True,
        'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'kelly_sizing': True,
        'kelly_conviction': True,
        'kelly_firmness': True,
        'kelly_max_qty': 25,
        'open_dte': 45,
        'vol_aware_dte': True,
    },
    # Regime-aware STRIKE (continuous, by z) — flex between OTM/ATM/ITM
    'regime_aware_strike': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'tp_dynamic': True,
        'roll_down': True, 'roll_up_calls': True,
        'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'regime_aware_strike': True,
        'open_dte': 45,
        'vol_aware_dte': True,
    },
    # Stack: regime_aware_strike + Kelly + firmness — production-style
    'fully_dynamic': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'tp_dynamic': True,
        'roll_down': True, 'roll_up_calls': True,
        'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'regime_aware_strike': True,
        'kelly_sizing': True,
        'kelly_conviction': True,
        'kelly_firmness': True,
        'kelly_max_qty': 25,
        'open_dte': 45,
        'vol_aware_dte': True,
    },
    # Income-mode strike adjustment + regime-aware (production-style loop)
    'income_mode_strike': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'tp_dynamic': True,
        'roll_down': True, 'roll_up_calls': True,
        'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'regime_aware_strike': True,
        'income_mode_strike': True,
        'target_weekly_income': 1500.0,
        'open_dte': 45,
        'vol_aware_dte': True,
    },
    # Best kernel + tail-hedge floor (production-port item #5)
    'best_plus_tail_floor': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'tp_dynamic': True,
        'roll_down': True, 'roll_up_calls': True,
        'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'regime_aware_strike': True,
        'income_mode_strike': True,
        'target_weekly_income': 1500.0,
        'open_dte': 45,
        'vol_aware_dte': True,
        'tail_hedge_floor': 2,  # always keep 2 long puts
    },
    'champion_robust_pricedt': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'tp_dynamic': True,
        'roll_down': True, 'roll_up_calls': True,
        'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'aggressive_itm_cc_z': -0.25, 'itm_cc_pct': -0.30,
        'elevator_close': True, 'elevator_itm_pct': 0.05,
        'elevator_extrinsic_max': 0.15, 'elevator_mode': 'strict',
        'vol_aware_sizing': True,
        'downtrend_from_high_standdown': True,
        'downtrend_high_floor': 30.0,  # UNG > $30 + downtrend = wait
    },
    # Aggressive vol ladder: 5-step instead of 2-step.
    'champion_vol_aggressive': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'roll_down': True, 'roll_up_calls': True,
        'regime_skip_puts_z': -0.5, 'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'aggressive_itm_cc_z': -1.0,
        'itm_cc_pct': -0.05,
        'elevator_close': True, 'elevator_itm_pct': 0.05,
        'elevator_extrinsic_max': 0.15, 'elevator_mode': 'strict',
        'vol_aware_sizing': 'aggressive',
    },
    # Even tighter — z<-1.5 (only deepest rich extremes)
    'champion_extreme_itm': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'roll_down': True, 'roll_up_calls': True,
        'regime_skip_puts_z': -0.5, 'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'aggressive_itm_cc_z': -1.5,  # only deepest rich
        'itm_cc_pct': -0.05,
    },
    'smooth_27_v3_core': {
        'otm_put': 0.08, 'otm_call': 0.05, 'put_qty': 4, 'call_qty': 5,
        'tp_50': True, 'roll_down': True, 'roll_up_calls': True,
        'regime_skip_puts_z': -0.5, 'boxx': True,
        'use_surprise_z': True,
        'falling_knife_filter': True,
        'anomaly_standdown': True,
        'z_scaled_sizing': True,
        'elevator_close': True, 'elevator_itm_pct': 0.05,
        'elevator_extrinsic_max': 0.15,
        'elevator_mode': 'strict',
        'target_shares': 4000,
        'core_shares': 2000,  # never drain below 2000 via elevator
    },
}


def compare_strategies():
    print(f"=== Loading dataset ===")
    df_path = os.path.join(CACHE_DIR, 'master_dataset.csv')
    if not os.path.exists(df_path):
        print("Master dataset missing — run historical_data_pipeline.py first")
        return
    df = pd.read_csv(df_path, parse_dates=['Date'] if 'Date' in pd.read_csv(df_path, nrows=1).columns else [0], index_col=0)
    df = precompute_factor_z(df)
    df = df.dropna(subset=['UNG'])
    print(f"Loaded {len(df)} days: {df.index[0].date()} → {df.index[-1].date()}")
    print()

    initial_cash = 48000
    initial_shares = 6200
    initial_nav = initial_cash + initial_shares * df['UNG'].iloc[0]

    years_held = (df.index[-1] - df.index[0]).days / 365
    results: dict = {}
    for name, params in STRATEGIES.items():
        print(f"Running {name}...")
        hist, trades = run_strategy_simple(df, params, initial_cash, initial_shares)
        if hist.empty:
            continue
        final = hist.iloc[-1]['nav']
        ret = (final / initial_nav - 1) * 100
        max_dd_pct = ((hist['nav'].min() - hist['nav'].cummax().max()) /
                      hist['nav'].cummax().max() * 100)
        daily_ret = hist['nav'].pct_change().dropna()
        sharpe = daily_ret.mean() / (daily_ret.std() + 1e-9) * math.sqrt(252)
        ann = ((final / initial_nav) ** (1/years_held) - 1) * 100 if years_held > 0 else 0
        results[name] = {
            'final': final, 'return_pct': ret, 'annual_pct': ann,
            'max_dd_pct': max_dd_pct, 'sharpe': sharpe, 'history': hist, 'trades': trades,
        }

    ung_ret = (df['UNG'].iloc[-30] / df['UNG'].iloc[0] - 1) * 100

    print()
    print(f"=== RESULTS (initial ${initial_nav:,.0f}, period {years_held:.1f} years) ===")
    print(f"{'Strategy':<25} {'Final NAV':>12} {'Return':>10} {'Annual':>9} {'MaxDD':>9} {'Sharpe':>8}")
    print("-" * 80)
    for name, r in results.items():
        print(f"{name:<25} ${r['final']:>11,.0f} {r['return_pct']:>+9.1f}% {r['annual_pct']:>+7.1f}% "
              f"{r['max_dd_pct']:>+7.1f}% {r['sharpe']:>+7.2f}")
    print(f"{'UNG buy-and-hold':<25} {ung_ret:>+9.1f}% over {years_held:.1f}y")

    # Save histories
    for name, r in results.items():
        out_path = os.path.join(RESULTS_DIR, f'{name}_history.csv')
        r['history'].to_csv(out_path, index=False)
        out_path = os.path.join(RESULTS_DIR, f'{name}_trades.csv')
        r['trades'].to_csv(out_path, index=False)
    print(f"\nResults saved to {RESULTS_DIR}/")

    # Per-kernel attribution
    for name, r in results.items():
        nav_delta = r['final'] - initial_nav
        attr = attribute_trades(r['trades'].to_dict('records'))
        print_attribution(name, attr, nav_delta)
        attr.to_csv(os.path.join(RESULTS_DIR, f'{name}_attribution.csv'), index=False)

    # Save summary JSON for web UI
    summary = {
        name: {k: v for k, v in r.items() if k not in ('history', 'trades')}
        for name, r in results.items()
    }
    summary['ung_return_pct'] = ung_ret  # type: ignore
    summary['initial_nav'] = initial_nav  # type: ignore
    summary['years'] = years_held  # type: ignore
    summary['updated_at'] = datetime.now().isoformat()  # type: ignore
    with open(os.path.join(RESULTS_DIR, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--compare', action='store_true', default=True)
    parser.parse_args()
    compare_strategies()
