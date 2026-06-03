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
from scenario_distribution import ScenarioDistribution  # type: ignore
from quality_scorer import score_portfolio_quality  # type: ignore

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
    """Composite z-score (CLEANED 20260603).

    Honest finding: of 5 candidate factors, only storage_surprise_z has
    IC > 0.10 with future returns (IC +0.146 at 20d). Adding other
    factors NAIVELY DILUTES this signal. days_supply_z = +0.04,
    ng_trend = +0.024 (inversion bug also: was multiplied by -1 wrongly),
    rv_30 as z component flips sign at combination.

    NEW APPROACH: single-factor z based on storage_surprise_z. Other
    signals (rv_30 for vol, momentum for trend) used separately in
    strategies, NOT folded into a noisy composite.

    use_surprise kept for backward compat — both modes now identical
    since we use surprise-detrended storage as the canonical z.
    """
    z_components = []
    weights = []

    # PRIMARY (and only) factor: storage surprise z (inverted = bullish positive)
    if 'storage_surprise_z' in row and not pd.isna(row['storage_surprise_z']):
        z_components.append(-row['storage_surprise_z'])
        weights.append(1.0)
    # Fallback to raw storage_z when surprise unavailable (warmup)
    elif 'storage_z' in row and not pd.isna(row['storage_z']):
        z_components.append(-row['storage_z'])
        weights.append(1.0)

    # Other fundamental signals NOT in z directly (would dilute) but used
    # via compute_fundamental_health() for sizing modulation + dashboard.
    # See [[project_z_audit]].

    if not z_components:
        return 0.0
    return float(np.average(z_components, weights=weights))


def compute_fundamental_health(row) -> dict:
    """Separate fundamental score — NOT a predictor, but interpretability +
    sizing modulator + risk context. Each component independently rated
    in [-1, +1] where + = bullish for NG.

    Returns:
      dict with each pillar + 'sum' (rough overall health).

    Use cases:
      - Dashboard display (show operator why z says what it says)
      - Sizing modulation (down-size when fundamentals weak even if z neutral)
      - Risk gating (don't sell deep OTM puts when LNG exports dropping)
    """
    out = {'storage': 0.0, 'days_supply': 0.0, 'cl_ng_ratio': 0.0,
           'price_band': 0.0, 'rv_regime': 0.0, 'sum': 0.0}
    try:
        # Storage component (low = bullish, signed +)
        sz = float(row.get('storage_z') or 0)
        out['storage'] = max(-1, min(1, -sz / 2.0))
        # Days supply (low = bullish)
        dz = float(row.get('days_supply_z') or 0)
        out['days_supply'] = max(-1, min(1, -dz / 2.0))
        # CL/NG ratio
        cl = float(row.get('CL') or 0)
        ng = float(row.get('NG') or 0)
        if ng > 0:
            ratio = cl / ng
            # ~25 typical, > 30 = NG cheap relative
            out['cl_ng_ratio'] = max(-1, min(1, (ratio - 25) / 10))
        # Price band (high in 120d range = mean-rev bearish)
        spot = float(row.get('UNG') or 0)
        lo = float(row.get('ung_252d_mean') or 0)
        std = float(row.get('ung_252d_std') or 1)
        if std > 0 and lo > 0:
            band = (spot - lo) / std
            out['price_band'] = max(-1, min(1, -band / 2.0))
        # Vol regime — high RV is risk caution signal (not bullish/bearish, more "be careful")
        rv = float(row.get('rv_30') or 0.5)
        if rv > 0.80:
            out['rv_regime'] = -0.5   # high vol → cautious
        elif rv < 0.40:
            out['rv_regime'] = +0.2   # calm = mild positive
    except Exception:
        pass
    out['sum'] = sum(v for k, v in out.items() if k != 'sum')
    return out


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


def compute_pillar_score(row, spot: float) -> dict:
    """Production-port (item #6): compute tech/fund/yoy pillar scores.

    Each pillar in [-1, +1], + = bullish for NG.
      tech: price band (near 120d low → bullish) + MA tilt (20 > 50 → bullish)
      fund: storage surprise (cheap = bullish)
      yoy: year-over-year price ratio (well below = bullish)

    Returns dict with each pillar score + sum.
    Sum can be added to composite z for amplified conviction.
    """
    out = {'tech': 0.0, 'fund': 0.0, 'yoy': 0.0, 'sum': 0.0}
    if spot <= 0:
        return out
    # Tech pillar
    lo_252 = float(row.get('ung_252d_mean') or 0)
    try:
        std_252 = float(row.get('ung_252d_std') or 1)
        if lo_252 > 0 and std_252 > 0:
            # Distance from mean in std units, inverted so low = bullish
            band = (spot - lo_252) / std_252
            tech = max(-1.0, min(1.0, -band * 0.5))
        else:
            tech = 0.0
        # Add MA tilt: 50d > 200d = bullish, but UNG rarely has this
        ma_50 = float(row.get('ung_50d_ma') or 0)
        ma_200 = float(row.get('ung_200d_ma') or 0)
        if ma_50 > 0 and ma_200 > 0:
            ma_diff = (ma_50 - ma_200) / ma_200
            tech += max(-0.3, min(0.3, ma_diff * 5.0))
        out['tech'] = max(-1.0, min(1.0, tech))
    except Exception:
        pass
    # Fund pillar — storage surprise z (already detrended)
    try:
        z_surp = float(row.get('storage_surprise_z') or 0)
        out['fund'] = max(-1.0, min(1.0, z_surp * 0.5))
    except Exception:
        pass
    # YoY pillar — vs 252d ago (rough proxy; we don't have exact 1yr lookback)
    try:
        # Use the long-term mean as a proxy: if spot < 70% of mean, very cheap
        if lo_252 > 0:
            ratio = spot / lo_252
            # ratio < 0.7 → very bullish (well below avg)
            # ratio > 1.3 → very bearish
            yoy = max(-1.0, min(1.0, (1.0 - ratio) * 2.0))
            out['yoy'] = yoy
    except Exception:
        pass
    out['sum'] = out['tech'] + out['fund'] + out['yoy']
    return out


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
        'short_puts': [], 'short_calls': [], 'long_puts': [], 'long_calls': [],
        'upside_call_open': None,
    }
    history = []
    trades = []

    p = strategy_params
    use_surprise = p.get('use_surprise_z', False)
    target_weekly_income = p.get('target_weekly_income', 1500.0)
    recent_premium = []
    # Drawdown-aware risk dial — single parameter that down-scales sizing
    # when NAV is deep in drawdown. Generic risk control — protects against
    # BOTH sharp crashes (2021-12 → 2022-02) AND slow declines (2023-2026).
    nav_peak = float(initial_cash + initial_shares * (df['UNG'].iloc[0] if len(df) else 1))

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
        # DD-aware risk dial — generic protection against any adverse
        # regime (sharp crash or slow decline). Track NAV peak; if
        # current NAV is significantly below peak, scale down all sizing.
        cur_nav = s['cash'] + s['shares'] * spot_u + s['boxx'] * 117 + s['kold'] * spot_k
        if cur_nav > nav_peak:
            nav_peak = cur_nav
        dd_pct = (cur_nav - nav_peak) / nav_peak * 100 if nav_peak > 0 else 0

        # TREND-FOLLOWING SHARE TRIM — generic protection against multi-year
        # declines. When UNG in confirmed downtrend (per ung_downtrend flag),
        # trim shares periodically to limit share-based losses. NOT a stop
        # loss — gradual exit that respects the wheel philosophy.
        trim_pct = p.get('trend_share_trim_pct_per_month', 0)
        if trim_pct > 0 and bool(row.get('ung_downtrend', False)):
            # Trim once per ~21 trading days
            if i % 21 == 0 and s['shares'] > 100:
                short_call_lots = sum(sc.get('qty', 0) for sc in s['short_calls'])
                min_shares_required = short_call_lots * 100
                trim_qty = int(s['shares'] * trim_pct / 100)
                trim_qty = (trim_qty // 100) * 100  # round to whole lots
                trim_qty = min(trim_qty, s['shares'] - min_shares_required)
                if trim_qty >= 100:
                    proceeds = trim_qty * (spot_u - SPREAD_SHARE)
                    s['cash'] += proceeds
                    s['shares'] -= trim_qty
                    trades.append({'date': idx, 'type': 'TREND_TRIM_SHARES',
                                   'pnl': 0.0, 'qty': trim_qty, 'spot': spot_u,
                                   'shares_after': s['shares']})

        # DD-TRIGGERED SHARE TRIM — emergency share reduction when DD breaches
        # explicit threshold. With smart conditions:
        #   - Skip trim if z extreme cheap (oversold, likely bounce)
        #   - Skip trim if confirmed uptrend (would miss spike capture)
        #   - Only act after persistent DD (avoid noise)
        dd_trim_trigger = p.get('dd_trim_trigger_pct', 0)  # e.g. -15 means trim if DD < -15%
        dd_trim_qty_pct = p.get('dd_trim_qty_pct', 20)     # 20% of shares per cadence
        dd_trim_floor = p.get('dd_trim_floor', 1000)       # minimum share level (never go below)
        dd_trim_cadence = p.get('dd_trim_cadence_days', 21)  # how often to trim (in trading days)
        # Smart-trim conditions
        smart_trim = p.get('smart_trim', False)
        z_skip_threshold = p.get('dd_trim_z_skip_below', -1.0)  # skip if z < this (oversold)
        # Pre-compute 50/200 MAs (cheap lookback if cached on row)
        if smart_trim:
            try:
                window_50 = df['UNG'].iloc[max(0,i-49):i+1]
                ma50 = window_50.mean()
                window_200 = df['UNG'].iloc[max(0,i-199):i+1]
                ma200 = window_200.mean()
                is_uptrend = (ma50 > ma200 * 1.02)  # bullish MA cross with 2% buffer
                is_oversold = (z < z_skip_threshold)
            except Exception:
                is_uptrend = False
                is_oversold = False
        else:
            is_uptrend = False
            is_oversold = False
        if dd_trim_trigger < 0 and dd_pct < dd_trim_trigger:
            should_trim = True
            if smart_trim and (is_uptrend or is_oversold):
                should_trim = False
            if should_trim and i % dd_trim_cadence == 0 and s['shares'] > dd_trim_floor:
                desired_trim = int(s['shares'] * dd_trim_qty_pct / 100)
                desired_trim = (desired_trim // 100) * 100
                desired_trim = min(desired_trim, s['shares'] - dd_trim_floor)

                # CC-AWARE CUT: if desired trim exceeds the freely-tradable
                # uncovered shares, close CCs (cheapest first = nearest
                # expiry, least time value) to free up shares. This lets
                # us cut delta even when wheel is fully covered.
                short_call_lots = sum(sc.get('qty', 0) for sc in s['short_calls'])
                min_shares_required = short_call_lots * 100
                free_shares = s['shares'] - min_shares_required
                if desired_trim > free_shares and p.get('cc_aware_cut', True):
                    needed_lots = (desired_trim - free_shares + 99) // 100
                    # Sort CCs by ascending DTE (close near-expiry first = cheapest)
                    s['short_calls'].sort(key=lambda sc: (idx - sc['entry']).days - sc['dte'])
                    while needed_lots > 0 and s['short_calls']:
                        sc = s['short_calls'][0]
                        days_left = max(1, sc['dte'] - (idx - sc['entry']).days)
                        T_left = days_left / 365
                        cur_prem = bs_call(spot_u, sc['K'], T_left, iv_at(sc['K'], days_left, 'C'))
                        close_lots = min(sc['qty'], needed_lots)
                        debit = cur_prem * 100 * close_lots + close_lots * SPREAD_OPTION * 100
                        if s['cash'] < debit + 500:
                            break
                        s['cash'] -= debit
                        pnl = sc['entry_prem'] * 100 * close_lots - debit
                        trades.append({'date': idx, 'type': 'CC_CLOSE_FOR_CUT',
                                       'pnl': pnl, 'qty': close_lots, 'K': sc['K'],
                                       'spot': spot_u, 'dd_pct': dd_pct})
                        sc['qty'] -= close_lots
                        needed_lots -= close_lots
                        if sc['qty'] <= 0:
                            s['short_calls'].pop(0)
                    # Recompute free shares after closing
                    short_call_lots = sum(sc.get('qty', 0) for sc in s['short_calls'])
                    min_shares_required = short_call_lots * 100
                    free_shares = s['shares'] - min_shares_required

                trim_qty = min(desired_trim, free_shares, s['shares'] - dd_trim_floor)
                trim_qty = (trim_qty // 100) * 100
                if trim_qty >= 100:
                    proceeds = trim_qty * (spot_u - SPREAD_SHARE)
                    s['cash'] += proceeds
                    s['shares'] -= trim_qty
                    trades.append({'date': idx, 'type': 'DD_TRIM_SHARES',
                                   'pnl': 0.0, 'qty': trim_qty, 'spot': spot_u,
                                   'dd_pct': dd_pct, 'shares_after': s['shares']})

                    # CUT-AND-REBUILD: after cutting shares, sell OTM puts
                    # BELOW spot to (a) capture put-skew premium (richer in
                    # declines), (b) stay paid while waiting for re-entry,
                    # (c) get re-acquired at lower strike if assigned. More
                    # executable than ITM CCs in fast-down markets.
                    if p.get('cut_and_rebuild_puts', False):
                        trim_lots = trim_qty // 100
                        put_otm = p.get('rebuild_put_otm_pct', 0.10)  # 10% below
                        put_dte = p.get('rebuild_put_dte', 45)
                        Kp = round(spot_u * (1 - put_otm))
                        put_prem = bs_put(spot_u, Kp, put_dte/365, iv_at(Kp, put_dte, 'P'))
                        if put_prem > 0.05:
                            # Size to match trim lots so re-entry restores
                            # original share count. Cap at cash-secured limit.
                            put_qty_rebuild = min(trim_lots, int(s['cash'] / (Kp * 100)))
                            if put_qty_rebuild >= 1:
                                credit = put_prem * 100 * put_qty_rebuild - put_qty_rebuild * SPREAD_OPTION * 100
                                s['cash'] += credit
                                s['short_puts'].append({'entry': idx, 'K': Kp, 'dte': put_dte,
                                                        'qty': put_qty_rebuild, 'entry_prem': put_prem})
                                trades.append({'date': idx, 'type': 'OPEN_REBUILD_PUT',
                                               'pnl': 0.0, 'credit': credit, 'K': Kp,
                                               'qty': put_qty_rebuild, 'spot': spot_u, 'dd_pct': dd_pct})

        # Z-BASED SHARE TARGETING — proactive (not reactive) wheel sizing.
        # Encode the wheel philosophy directly: accumulate cheap, lighten
        # expensive. Maintain shares = base × z_multiplier. Avoids the
        # reactive-trim trap where we sell into noise pullbacks.
        z_target_enabled = p.get('z_share_target_enabled', False)
        if z_target_enabled and i % p.get('z_target_cadence_days', 5) == 0:
            base_shares = p.get('z_share_target_base', 6200)
            # Tunable multipliers — defaults are the wheel philosophy curve
            mults = p.get('z_target_mults', {
                'extreme_cheap': 1.4, 'cheap': 1.2, 'neutral': 1.0,
                'rich': 0.7, 'extreme_rich': 0.3,
            })
            if z < -1.5:    mult = mults['extreme_cheap']
            elif z < -0.5:  mult = mults['cheap']
            elif z < 0.5:   mult = mults['neutral']
            elif z < 1.0:   mult = mults['rich']
            else:           mult = mults['extreme_rich']
            # DD-aware override: if in deep DD, cap the multiplier
            dd_cap_15 = p.get('z_target_dd_cap_15', 0.6)
            dd_cap_10 = p.get('z_target_dd_cap_10', 0.8)
            if dd_pct < -15:
                mult = min(mult, dd_cap_15)
            elif dd_pct < -10:
                mult = min(mult, dd_cap_10)
            target = int(base_shares * mult)
            target = (target // 100) * 100
            current = s['shares']
            # MOMENTUM OVERRIDE — when shares are about to be TRIMMED (rich
            # regime) BUT trend is clearly bullish (50/20d MAs ripping),
            # hold shares anyway. Captures spike runups (2022-style) that
            # pure z-target would have trimmed too early. Doesn't override
            # ADD direction — accumulation always proceeds.
            momentum_override = p.get('momentum_override', False)
            if momentum_override and target < current and i > 50:
                try:
                    win50 = df['UNG'].iloc[max(0,i-49):i+1]
                    win20 = df['UNG'].iloc[max(0,i-19):i+1]
                    win5 = df['UNG'].iloc[max(0,i-4):i+1]
                    ma50 = win50.mean(); ma20 = win20.mean(); ma5 = win5.mean()
                    # Steep bullish: short MA > med MA > long MA + recent gain
                    recent_gain = (spot_u / win50.iloc[0] - 1) if win50.iloc[0] > 0 else 0
                    rip = (ma5 > ma20 * 1.03) and (ma20 > ma50 * 1.03) and (recent_gain > 0.20)
                    if rip:
                        # Suspend trim — set target to current
                        target = current
                        # Log once per regime turn
                        if not s.get('_momentum_overriding'):
                            trades.append({'date': idx, 'type': 'MOMENTUM_OVERRIDE_HOLD',
                                           'pnl': 0.0, 'spot': spot_u, 'z': z,
                                           'shares': current})
                        s['_momentum_overriding'] = True
                    else:
                        s['_momentum_overriding'] = False
                except Exception:
                    pass
            delta = target - current
            # Cut speed tunable: 0.3 = slow (30% toward target per cadence),
            # 0.5 = balanced (default), 1.0 = full snap. Higher = faster cut.
            cut_speed = p.get('cut_speed', 0.5)
            # Over-cut buffer: trim a bit MORE than the calculated amount to
            # build cash buffer (e.g., 0.10 = 10% extra cut). Only applies
            # when we're SELLING (delta < 0), not buying. Restores parity
            # on the way back up via z_target_add when z turns cheap.
            over_cut = p.get('over_cut_pct', 0.0)
            if delta < 0 and over_cut > 0:
                adjust = int(delta * cut_speed * (1 + over_cut))
            else:
                adjust = int(delta * cut_speed)
            adjust = (adjust // 100) * 100
            # Must keep enough shares to cover existing short calls
            short_call_lots = sum(sc.get('qty', 0) for sc in s['short_calls'])
            min_shares_required = short_call_lots * 100
            if adjust < 0:  # selling
                max_sell = current - min_shares_required
                adjust = max(adjust, -max(0, max_sell))
                if adjust <= -100:
                    sell_qty = -adjust
                    proceeds = sell_qty * (spot_u - SPREAD_SHARE)
                    s['cash'] += proceeds
                    s['shares'] -= sell_qty
                    trades.append({'date': idx, 'type': 'Z_TARGET_TRIM',
                                   'pnl': 0.0, 'qty': sell_qty, 'spot': spot_u,
                                   'z': z, 'target': target, 'shares_after': s['shares']})
                    # CUT-AND-REBUILD on z_target trim too
                    if p.get('cut_and_rebuild_puts', False) and z > -1.0:
                        # Skip if already very cheap (don't want more shares at bottom)
                        trim_lots = sell_qty // 100
                        put_otm = p.get('rebuild_put_otm_pct', 0.10)
                        put_dte = p.get('rebuild_put_dte', 45)
                        Kp = round(spot_u * (1 - put_otm))
                        put_prem = bs_put(spot_u, Kp, put_dte/365, iv_at(Kp, put_dte, 'P'))
                        if put_prem > 0.05:
                            put_qty_rebuild = min(trim_lots, int(s['cash'] / (Kp * 100)))
                            if put_qty_rebuild >= 1:
                                credit = put_prem * 100 * put_qty_rebuild - put_qty_rebuild * SPREAD_OPTION * 100
                                s['cash'] += credit
                                s['short_puts'].append({'entry': idx, 'K': Kp, 'dte': put_dte,
                                                        'qty': put_qty_rebuild, 'entry_prem': put_prem})
                                trades.append({'date': idx, 'type': 'OPEN_REBUILD_PUT_Z',
                                               'pnl': 0.0, 'credit': credit, 'K': Kp,
                                               'qty': put_qty_rebuild, 'spot': spot_u, 'z': z})
            elif adjust >= 100:  # buying
                max_afford = int((s['cash'] - 5000) / (spot_u + SPREAD_SHARE)) if s['cash'] > 5000 else 0
                max_afford = (max_afford // 100) * 100
                adjust = min(adjust, max_afford)
                if adjust >= 100:
                    cost = adjust * (spot_u + SPREAD_SHARE)
                    s['cash'] -= cost
                    s['shares'] += adjust
                    trades.append({'date': idx, 'type': 'Z_TARGET_ADD',
                                   'pnl': 0.0, 'qty': adjust, 'spot': spot_u,
                                   'z': z, 'target': target, 'shares_after': s['shares']})

        # SMART RE-ENTRY — when we've trimmed shares and conditions turn
        # favorable, buy back. Recovers upside that pure-trim variants miss.
        # Conditions: shares below initial AND z cheap AND uptrend confirmed
        rebuy_enabled = p.get('dd_rebuy_enabled', False)
        if rebuy_enabled and i % dd_trim_cadence == 0:
            initial_shares_ref = p.get('initial_shares_ref', 6200)
            shares_deficit = initial_shares_ref - s['shares']
            try:
                window_5 = df['UNG'].iloc[max(0,i-4):i+1]
                window_20 = df['UNG'].iloc[max(0,i-19):i+1]
                short_ma = window_5.mean()
                med_ma = window_20.mean()
                trend_turning = short_ma > med_ma * 1.01
            except Exception:
                trend_turning = False
            rebuy_z_threshold = p.get('dd_rebuy_z_below', -0.5)
            if shares_deficit >= 100 and z < rebuy_z_threshold and trend_turning:
                # Buy back 25% of deficit per cycle, capped by cash
                rebuy_qty = int(shares_deficit * 0.25)
                rebuy_qty = (rebuy_qty // 100) * 100
                max_affordable = int((s['cash'] - 5000) / (spot_u + SPREAD_SHARE)) if s['cash'] > 5000 else 0
                rebuy_qty = min(rebuy_qty, max_affordable, shares_deficit)
                if rebuy_qty >= 100:
                    cost = rebuy_qty * (spot_u + SPREAD_SHARE)
                    s['cash'] -= cost
                    s['shares'] += rebuy_qty
                    trades.append({'date': idx, 'type': 'DD_REBUY_SHARES',
                                   'pnl': 0.0, 'qty': rebuy_qty, 'spot': spot_u,
                                   'z': z, 'shares_after': s['shares']})

        # DYNAMIC KOLD HEDGE — scale KOLD position with DD severity. Preserves
        # share inventory (wheel-friendly) but uses inverse ETF as portfolio
        # delta hedge. Larger DD → larger KOLD allocation.
        dd_kold_max = p.get('dd_kold_hedge_max_pct', 0)  # e.g. 0.20 = up to 20% NAV in KOLD
        if dd_kold_max > 0 and dd_pct < -5 and spot_k > 0 and pd.notna(spot_k):
            # Linear ramp: 0% at DD=-5, max at DD=-25
            ramp = min(1.0, max(0.0, (-dd_pct - 5) / 20.0))
            target_kold_dollars = cur_nav * dd_kold_max * ramp
            current_kold_dollars = s['kold'] * spot_k
            delta_dollars = target_kold_dollars - current_kold_dollars
            # Only act if change > 5% of target (avoid churning)
            if abs(delta_dollars) > max(500, target_kold_dollars * 0.10):
                if delta_dollars > 0:  # buy more KOLD
                    buy_q = int(delta_dollars / spot_k)
                    if buy_q >= 5 and s['cash'] > buy_q * spot_k + 200:
                        s['kold'] += buy_q
                        s['cash'] -= buy_q * spot_k + buy_q * SPREAD_SHARE
                        trades.append({'date': idx, 'type': 'KOLD_DD_HEDGE_BUY',
                                       'pnl': 0.0, 'qty': buy_q, 'spot': spot_k,
                                       'dd_pct': dd_pct})
                else:  # trim KOLD
                    sell_q = min(s['kold'], int(-delta_dollars / spot_k))
                    if sell_q >= 5:
                        s['cash'] += sell_q * spot_k - sell_q * SPREAD_SHARE
                        s['kold'] -= sell_q
                        trades.append({'date': idx, 'type': 'KOLD_DD_HEDGE_TRIM',
                                       'pnl': 0.0, 'qty': sell_q, 'spot': spot_k,
                                       'dd_pct': dd_pct})
        # Default no scaling. Strategies opt-in via dd_aware_dial.
        if p.get('dd_aware_dial'):
            if dd_pct < -25:    dd_scale = 0.4
            elif dd_pct < -15:  dd_scale = 0.6
            elif dd_pct < -10:  dd_scale = 0.8
            else:               dd_scale = 1.0
        else:
            dd_scale = 1.0
        # Pillar boost: if enabled, add scaled pillar sum to composite z.
        # Production uses pillar_drift to nudge the scenario distribution;
        # backtest just adds it to z for similar effect on regime/sizing.
        if p.get('pillar_boost'):
            pillars = compute_pillar_score(row, spot_u)
            # Pillars in [-1, +1] each, sum in [-3, +3]. Scale to ±0.5 z adj.
            z = z + pillars['sum'] * 0.15
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
            # MOMENTUM GATE: if elevator_skip_on_momentum is enabled and
            # confirmed parabolic uptrend is in progress, SKIP elevator-close.
            # Holds the spike instead of locking small gains while it runs.
            skip_elevator = False
            if p.get('elevator_skip_on_momentum') and i > 200:
                try:
                    win50 = df['UNG'].iloc[max(0,i-49):i+1]
                    win200 = df['UNG'].iloc[max(0,i-199):i+1]
                    ma50 = win50.mean(); ma200 = win200.mean()
                    ret90 = (spot_u / df['UNG'].iloc[max(0,i-90)] - 1) if i >= 90 else 0
                    if ma50 > ma200 * 1.05 and ret90 > 0.30:
                        skip_elevator = True
                except Exception:
                    pass

            if p.get('elevator_close') and T_left > 1/365 and not skip_elevator:
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

        # MOMENTUM CALL LAYER — buy OTM calls during confirmed parabolic
        # uptrend. Decouples upside capture from share holding (so we still
        # benefit from z_target's protective trim on the eventual crash).
        # Trigger: 50d/200d MA cross AND 90d return > momentum_threshold.
        # Sizing: small NAV % per fire, max stack of N concurrent positions.
        mom_call_pct = p.get('momentum_call_pct', 0)
        if mom_call_pct > 0 and i > 200:
            try:
                win50 = df['UNG'].iloc[max(0,i-49):i+1]
                win200 = df['UNG'].iloc[max(0,i-199):i+1]
                ma50 = win50.mean(); ma200 = win200.mean()
                ret90 = (spot_u / df['UNG'].iloc[max(0,i-90)] - 1) if i >= 90 else 0
                mom_threshold = p.get('momentum_call_threshold', 0.20)
                bullish_cross = ma50 > ma200 * 1.05
                strong_momentum = ret90 > mom_threshold
                existing_mom_calls = len([lc for lc in s.get('long_calls', [])
                                          if lc.get('momentum_call')])
                max_stack = p.get('momentum_call_max_stack', 3)
                # Fire monthly cadence to ladder positions through the move
                if bullish_cross and strong_momentum and existing_mom_calls < max_stack and i % 21 == 0:
                    Kc = round(spot_u * (1 + p.get('momentum_call_otm_pct', 0.15)))
                    dte_c = p.get('momentum_call_dte', 90)
                    cost = bs_call(spot_u, Kc, dte_c/365, iv_at(Kc, dte_c, 'C'))
                    if cost > 0.05:
                        budget = cur_nav * mom_call_pct
                        qty_c = int(budget / (cost * 100))
                        if qty_c >= 1 and s['cash'] > qty_c * cost * 100 + 200:
                            debit = qty_c * cost * 100 + qty_c * SPREAD_OPTION * 100
                            s['cash'] -= debit
                            s.setdefault('long_calls', []).append({
                                'entry': idx, 'K': Kc, 'dte': dte_c,
                                'qty': qty_c, 'cost': cost,
                                'momentum_call': True,
                            })
                            trades.append({'date': idx, 'type': 'OPEN_MOMENTUM_CALL',
                                           'pnl': -debit, 'K': Kc, 'qty': qty_c,
                                           'spot': spot_u, 'ret90': ret90})
            except Exception:
                pass

        # LONG OTM CALL UPSIDE TICKET — when EXTREME_CHEAP regime, allocate
        # tiny NAV % to far-OTM calls. Bounded downside, convex upside.
        # Fills the "missed 2022 spike" gap on protected variants without
        # giving up MDD discipline. Premium is small because OTM + cheap
        # regime depresses IV.
        upside_ticket_pct = p.get('upside_ticket_pct', 0)
        if upside_ticket_pct > 0 and z < -1.0 and not s.get('upside_call_open'):
            # Buy 90d calls 30% OTM — far enough to be cheap, close enough
            # to pay in a real spike
            Kc = round(spot_u * 1.30)
            dte_c = 90
            cost = bs_call(spot_u, Kc, dte_c/365, iv_at(Kc, dte_c, 'C'))
            if cost > 0.05:
                budget = cur_nav * upside_ticket_pct
                qty_c = int(budget / (cost * 100))
                if qty_c >= 1 and s['cash'] > qty_c * cost * 100 + 200:
                    debit = qty_c * cost * 100 + qty_c * SPREAD_OPTION * 100
                    s['cash'] -= debit
                    s.setdefault('long_calls', []).append({'entry': idx, 'K': Kc,
                                                           'dte': dte_c, 'qty': qty_c,
                                                           'cost': cost})
                    s['upside_call_open'] = idx
                    trades.append({'date': idx, 'type': 'OPEN_UPSIDE_TICKET',
                                   'pnl': -debit, 'K': Kc, 'qty': qty_c,
                                   'spot': spot_u, 'z': z})

        # Expire / value long calls (upside tickets)
        if s.get('long_calls'):
            keep = []
            for lc in s['long_calls']:
                days = (idx - lc['entry']).days
                if days >= lc['dte']:
                    # Expire — settle vs spot
                    payoff = max(0, spot_u - lc['K']) * 100 * lc['qty']
                    s['cash'] += payoff
                    if payoff > 0:
                        trades.append({'date': idx, 'type': 'UPSIDE_TICKET_PAYOFF',
                                       'pnl': payoff - lc['cost']*100*lc['qty'],
                                       'qty': lc['qty'], 'K': lc['K'], 'spot': spot_u})
                    if s.get('upside_call_open') == lc['entry']:
                        s['upside_call_open'] = None
                else:
                    keep.append(lc)
            s['long_calls'] = keep

        # SHOULDER-SEASON KOLD HEDGE — March-May and Sept-Nov are NG's
        # structurally weak periods (low HDD/CDD, storage builds). KOLD
        # (-2x NG) profits there. Empirical: shoulder months show +0.33%/d
        # KOLD vs +0.12%/d non-shoulder. Allocate a small NAV % to KOLD
        # during shoulder when z is NEUTRAL or RICH. Exit when leaving
        # shoulder OR z turns CHEAP (NG rebound likely).
        shoulder_pct = p.get('kold_shoulder_hedge', 0)
        if shoulder_pct > 0:
            month = idx.month
            in_shoulder = month in (3, 4, 5, 9, 10, 11)
            # Force exit if outside shoulder or z is cheap (NG bouncing)
            if not in_shoulder and s['kold'] > 0:
                s['cash'] += s['kold'] * spot_k - s['kold'] * SPREAD_SHARE
                trades.append({'date': idx, 'type': 'KOLD_EXIT_SHOULDER_END',
                               'pnl': 0.0, 'qty': s['kold'], 'spot': spot_k})
                s['kold'] = 0
            elif in_shoulder and z > -0.5 and s['kold'] == 0 and spot_k > 0 and pd.notna(spot_k):
                # Scale entry by z-score richness — richer NG → larger hedge
                z_scale = min(1.0, max(0.3, 0.5 + z * 0.3))  # 0.3 at z=-0.5, 1.0 at z=+1.5
                nav_now = s['cash'] + s['shares'] * spot_u + s['boxx'] * 117
                if nav_now > 0:
                    try:
                        tq = int(nav_now * shoulder_pct * z_scale / spot_k)
                    except (ValueError, OverflowError):
                        tq = 0
                    if tq >= 5 and s['cash'] > tq * spot_k + 200:
                        s['kold'] += tq
                        s['cash'] -= tq * spot_k + tq * SPREAD_SHARE
                        trades.append({'date': idx, 'type': 'KOLD_SHOULDER_ENTRY',
                                       'pnl': 0.0, 'qty': tq, 'spot': spot_k,
                                       'month': month, 'z': z})

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
                # BEAM SELECTOR (item #1 port — critical version): generate
                # ladder of put candidates, score each, pick best by quality
                # delta. Tests if multi-objective scoring beats hardcoded
                # regime_aware_strike rules.
                if p.get('beam_put'):
                    ladder_otm = p.get('beam_put_otm_ladder',
                                       [0.02, 0.05, 0.08, 0.12, 0.15])
                    best_qty = 0
                    best_K = 0
                    best_prem = 0
                    best_score = -1e9
                    for try_otm in ladder_otm:
                        try_K = round(spot_u * (1 - try_otm))
                        try_dte = p.get('open_dte', 30)
                        try_iv = iv_at(try_K, try_dte, 'P')
                        try_prem = bs_put(spot_u, try_K, try_dte/365, try_iv)
                        if try_prem < 0.05:
                            continue
                        # Approximate per-trade score components:
                        # income contribution (premium * qty * 100)
                        # assignment risk (p_itm * (K - spot_at_assign))
                        T = try_dte / 365
                        d2 = (math.log(spot_u/try_K) - 0.5*try_iv**2*T) / (try_iv*math.sqrt(T))
                        from scipy.stats import norm as _norm
                        p_otm = float(_norm.cdf(d2))
                        p_itm = 1 - p_otm
                        qty_try = p.get('put_qty', 3)
                        income = try_prem * 100 * qty_try
                        # Penalty: expected loss if assigned in adverse move
                        loss = max(0, try_K - spot_u * 0.92) * 100 * qty_try * p_itm
                        # Quality: income - 0.5 * loss (asymmetric)
                        score = income - 0.5 * loss
                        if score > best_score:
                            best_score = score
                            best_K = try_K
                            best_prem = try_prem
                            best_qty = qty_try
                    if best_qty > 0:
                        otm_put = (spot_u - best_K) / spot_u  # back-derive
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
                # SMOOTHNESS PENALTY (production-port #7): if the last 4
                # weeks of income have high variance (std/mean > 0.5),
                # push strike FARTHER OTM to reduce future variance. The
                # idea: jagged income → bad operator experience even if
                # total is high.
                if p.get('smoothness_aware') and len(recent_premium) >= 4:
                    recent = recent_premium[-4:]
                    mean = sum(recent) / 4
                    if mean > 0:
                        import statistics
                        std = statistics.pstdev(recent)
                        cv = std / mean if mean > 0 else 0
                        if cv > 0.5:  # jagged income
                            otm_put = min(0.25, otm_put + 0.02)
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
                # Optionally backed by ScenarioDistribution (port item #2).
                if p.get('kelly_sizing') and prem > 0.05:
                    iv_use = iv_at(K, open_dte, 'P')
                    conv_adj = model_conviction(row, z, anomaly) if p.get('kelly_conviction') else 0.0
                    sd = None
                    if p.get('use_scenario_dist'):
                        sd = ScenarioDistribution(spot=spot_u, sigma_annual=iv_use,
                                                  z_score=z, contango_per_day=-0.001)
                    kelly_q = kelly_qty_short_put(
                        spot_u, K, open_dte, iv_use,
                        cash_available=s['cash'],
                        premium=prem,
                        model_conviction=conv_adj,
                        scenario_dist=sd,
                    )
                    if p.get('kelly_firmness'):
                        firm = firmness_multiplier(row, z, anomaly)
                        kelly_q = int(kelly_q * firm)
                    # FUNDAMENTAL SIZE MODULATION — fundamentals don't predict
                    # direction (per IC audit) but they modulate confidence.
                    # If fundamentals strongly support the position, size up
                    # 20%. If they contradict, size down 30%.
                    if p.get('fundamental_modulation'):
                        f = compute_fundamental_health(row)
                        f_sum = f['sum']
                        if f_sum > 1.5:
                            kelly_q = int(kelly_q * 1.2)
                        elif f_sum < -1.5:
                            kelly_q = int(kelly_q * 0.7)
                    # DD-aware risk dial — generic risk control
                    kelly_q = int(kelly_q * dd_scale)
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
                        # CONCENTRATION PENALTY (production-port #4):
                        # Cap total contracts at any one strike. Production
                        # beam penalizes via quality scoring; backtest uses
                        # hard cap for simplicity.
                        max_conc = p.get('max_concentration_per_strike', 0)
                        if max_conc > 0:
                            existing_at_K = sum(sp.get('qty', 0)
                                               for sp in s['short_puts']
                                               if abs(sp.get('K', 0) - K) < 0.5)
                            allowed = max(0, max_conc - existing_at_K)
                            if put_qty > allowed:
                                trades.append({'date': idx, 'type': 'CONCENTRATION_CAP',
                                              'pnl': 0.0, 'K': K,
                                              'requested': put_qty, 'allowed': allowed})
                                put_qty = allowed
                        if put_qty == 0:
                            continue
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
                # Disable aggressive ITM CC during confirmed momentum
                if use_itm and p.get('itm_cc_skip_on_momentum') and i > 200:
                    try:
                        win50 = df['UNG'].iloc[max(0,i-49):i+1]
                        win200 = df['UNG'].iloc[max(0,i-199):i+1]
                        ma50 = win50.mean(); ma200 = win200.mean()
                        ret90 = (spot_u / df['UNG'].iloc[max(0,i-90)] - 1) if i >= 90 else 0
                        if ma50 > ma200 * 1.05 and ret90 > 0.30:
                            use_itm = False
                    except Exception:
                        pass
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
                    qty = max(0, min(kelly_q, int(p.get('kelly_max_qty', 20)), uncovered_shares // 100))
                if prem > 0.05 and qty >= 1:
                    credit = prem * 100 * qty - qty * SPREAD_OPTION * 100
                    s['cash'] += credit
                    s['short_calls'].append({'entry': idx, 'K': K, 'dte': cc_dte,
                                             'qty': qty, 'entry_prem': prem,
                                             'is_itm_aggressive': use_itm})
                    open_kind = 'OPEN_ITM_CC' if use_itm else 'OPEN_CC'
                    trades.append({'date': idx, 'type': open_kind,
                                   'pnl': 0.0, 'credit': credit,
                                   'K': K, 'qty': qty, 'z': z})

                    # UPSIDE WING — when shorting an ITM/ATM CC, buy a far-OTM
                    # call at K_wing = K * (1 + wing_otm_pct) so if UNG spikes
                    # through, the wing recovers the capped upside. Net credit
                    # still positive because wing premium << CC premium when
                    # wing is deep OTM. Same expiry as CC.
                    wing_otm = p.get('upside_wing_otm_pct', 0)
                    wing_always = p.get('upside_wing_always', False)
                    if wing_otm > 0 and (use_itm or wing_always):
                        K_wing = round(spot_u * (1 + wing_otm))
                        wing_cost = bs_call(spot_u, K_wing, cc_dte/365, iv_at(K_wing, cc_dte, 'C'))
                        if wing_cost > 0.02:
                            wing_debit = wing_cost * 100 * qty + qty * SPREAD_OPTION * 100
                            # Only buy wing if wing_debit < 30% of credit (net positive)
                            if wing_debit < credit * 0.30 and s['cash'] > wing_debit + 500:
                                s['cash'] -= wing_debit
                                s.setdefault('long_calls', []).append({
                                    'entry': idx, 'K': K_wing, 'dte': cc_dte,
                                    'qty': qty, 'cost': wing_cost,
                                    'wing_for_cc_K': K,  # link to its CC
                                })
                                trades.append({'date': idx, 'type': 'OPEN_UPSIDE_WING',
                                               'pnl': -wing_debit, 'K': K_wing,
                                               'qty': qty, 'cc_K': K, 'spot': spot_u})

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

            # PROTECTIVE COLLAR — maintain long puts proportional to share
            # exposure. Direct depth control: if 100% of shares hedged at
            # strike S, max loss = (spot - S) / spot. Tunable hedge ratio
            # and strike offset.
            collar_ratio = p.get('protective_collar_ratio', 0)  # fraction of share lots to hedge (0.5 = 50%)
            collar_otm = p.get('protective_collar_otm_pct', 0.05)  # 5% OTM by default
            collar_dte = p.get('protective_collar_dte', 90)
            if collar_ratio > 0 and s['shares'] >= 100:
                share_lots = s['shares'] // 100
                target_protected_lots = int(share_lots * collar_ratio)
                existing_long_qty = sum(lp.get('qty', 0) for lp in s['long_puts'])
                shortfall = target_protected_lots - existing_long_qty
                if shortfall > 0:
                    Kp = round(spot_u * (1 - collar_otm))
                    cost = bs_put(spot_u, Kp, collar_dte/365, iv_at(Kp, collar_dte, 'P'))
                    debit = cost * 100 * shortfall + shortfall * SPREAD_OPTION * 100
                    if s['cash'] > debit + 1000:
                        s['cash'] -= debit
                        s['long_puts'].append({'entry': idx, 'K': Kp, 'dte': collar_dte,
                                               'qty': shortfall, 'cost': cost})
                        trades.append({'date': idx, 'type': 'OPEN_PROTECTIVE_COLLAR',
                                       'pnl': -debit, 'K': Kp, 'qty': shortfall,
                                       'spot': spot_u, 'ratio': collar_ratio})

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
        'tail_hedge_floor': 2,
    },
    # Champion_dyn_itm_20pct chassis (Sharpe 1.77 post-refresh) + tail floor
    'champion_20pct_plus_floor': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'tp_dynamic': True,
        'roll_down': True, 'roll_up_calls': True,
        'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'aggressive_itm_cc_z': -0.25, 'itm_cc_pct': -0.20,
        'elevator_close': True, 'elevator_itm_pct': 0.05,
        'elevator_extrinsic_max': 0.15, 'elevator_mode': 'strict',
        'vol_aware_sizing': True,
        'tail_hedge_floor': 2,
    },
    # Champion + KOLD shoulder hedge — addresses calm-period bleed
    'champion_plus_shoulder': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'tp_dynamic': True,
        'roll_down': True, 'roll_up_calls': True,
        'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'aggressive_itm_cc_z': -0.25, 'itm_cc_pct': -0.20,
        'elevator_close': True, 'elevator_itm_pct': 0.05,
        'elevator_extrinsic_max': 0.15, 'elevator_mode': 'strict',
        'vol_aware_sizing': True,
        'tail_hedge_floor': 2,
        'kold_shoulder_hedge': 0.08,
    },
    'champion_plus_shoulder_heavy': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'tp_dynamic': True,
        'roll_down': True, 'roll_up_calls': True,
        'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'aggressive_itm_cc_z': -0.25, 'itm_cc_pct': -0.20,
        'elevator_close': True, 'elevator_itm_pct': 0.05,
        'elevator_extrinsic_max': 0.15, 'elevator_mode': 'strict',
        'vol_aware_sizing': True,
        'tail_hedge_floor': 2,
        'kold_shoulder_hedge': 0.15,
    },
    # Best + concentration penalty (production-port item #4)
    'best_plus_concentration': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'tp_dynamic': True,
        'roll_down': True, 'roll_up_calls': True,
        'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'aggressive_itm_cc_z': -0.25, 'itm_cc_pct': -0.20,
        'elevator_close': True, 'elevator_itm_pct': 0.05,
        'elevator_extrinsic_max': 0.15, 'elevator_mode': 'strict',
        'vol_aware_sizing': True,
        'tail_hedge_floor': 2,
        'max_concentration_per_strike': 15,
    },
    # Best + smoothness penalty (production-port item #7)
    'best_plus_smoothness': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'tp_dynamic': True,
        'roll_down': True, 'roll_up_calls': True,
        'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'aggressive_itm_cc_z': -0.25, 'itm_cc_pct': -0.20,
        'elevator_close': True, 'elevator_itm_pct': 0.05,
        'elevator_extrinsic_max': 0.15, 'elevator_mode': 'strict',
        'vol_aware_sizing': True,
        'tail_hedge_floor': 2,
        'income_mode_strike': True,
        'target_weekly_income': 1500.0,
        'smoothness_aware': True,
    },
    # Best + pillar boost (production-port item #6)
    'best_plus_pillars': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'tp_dynamic': True,
        'roll_down': True, 'roll_up_calls': True,
        'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'aggressive_itm_cc_z': -0.25, 'itm_cc_pct': -0.20,
        'elevator_close': True, 'elevator_itm_pct': 0.05,
        'elevator_extrinsic_max': 0.15, 'elevator_mode': 'strict',
        'vol_aware_sizing': True,
        'tail_hedge_floor': 2,
        'pillar_boost': True,
    },
    # Kelly backed by ScenarioDistribution (production-port item #2)
    'kelly_with_sd': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'tp_dynamic': True,
        'roll_down': True, 'roll_up_calls': True,
        'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'kelly_sizing': True, 'kelly_conviction': True,
        'use_scenario_dist': True,
        'kelly_max_qty': 20,
        'open_dte': 45,
        'vol_aware_dte': True,
        'tail_hedge_floor': 2,
    },
    # Fundamentals back — used as SIZING modulator (not in z)
    'kelly_fundamental_modulated': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'tp_dynamic': True,
        'roll_down': True, 'roll_up_calls': True,
        'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'kelly_sizing': True, 'kelly_conviction': True,
        'fundamental_modulation': True,
        'kelly_max_qty': 20,
        'open_dte': 45,
        'vol_aware_dte': True,
        'tail_hedge_floor': 2,
    },
    # + drawdown-aware risk dial (generic risk control)
    'kelly_dd_aware': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'tp_dynamic': True,
        'roll_down': True, 'roll_up_calls': True,
        'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'kelly_sizing': True, 'kelly_conviction': True,
        'fundamental_modulation': True,
        'dd_aware_dial': True,
        'kelly_max_qty': 20,
        'open_dte': 45,
        'vol_aware_dte': True,
        'tail_hedge_floor': 2,
    },
    # + KOLD shoulder-season hedge (Mar-May, Sept-Nov are weak NG periods)
    'kelly_shoulder_kold': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'tp_dynamic': True,
        'roll_down': True, 'roll_up_calls': True,
        'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'kelly_sizing': True, 'kelly_conviction': True,
        'fundamental_modulation': True,
        'dd_aware_dial': True,
        'kold_shoulder_hedge': 0.05,
        'kelly_max_qty': 20,
        'open_dte': 45,
        'vol_aware_dte': True,
        'tail_hedge_floor': 2,
    },
    'kelly_shoulder_kold_heavy': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'tp_dynamic': True,
        'roll_down': True, 'roll_up_calls': True,
        'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'kelly_sizing': True, 'kelly_conviction': True,
        'fundamental_modulation': True,
        'dd_aware_dial': True,
        'kold_shoulder_hedge': 0.10,
        'kelly_max_qty': 20,
        'open_dte': 45,
        'vol_aware_dte': True,
        'tail_hedge_floor': 2,
    },
    # ANTI-DD FAMILY — explicit drawdown control tunables
    # A: DD-triggered share trim (sell shares when DD breaches threshold)
    'kelly_dd_share_trim_15': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'tp_dynamic': True,
        'roll_down': True, 'roll_up_calls': True,
        'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'kelly_sizing': True, 'kelly_conviction': True,
        'fundamental_modulation': True,
        'kold_shoulder_hedge': 0.05,
        'dd_trim_trigger_pct': -15, 'dd_trim_qty_pct': 25, 'dd_trim_floor': 2000,
        'kelly_max_qty': 20,
        'open_dte': 45,
        'vol_aware_dte': True,
        'tail_hedge_floor': 2,
    },
    'kelly_dd_share_trim_10': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'tp_dynamic': True,
        'roll_down': True, 'roll_up_calls': True,
        'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'kelly_sizing': True, 'kelly_conviction': True,
        'fundamental_modulation': True,
        'kold_shoulder_hedge': 0.05,
        'dd_trim_trigger_pct': -10, 'dd_trim_qty_pct': 30, 'dd_trim_floor': 2000,
        'kelly_max_qty': 20,
        'open_dte': 45,
        'vol_aware_dte': True,
        'tail_hedge_floor': 2,
    },
    # B: Dynamic KOLD hedge (scale KOLD with DD severity)
    'kelly_dd_kold_hedge': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'tp_dynamic': True,
        'roll_down': True, 'roll_up_calls': True,
        'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'kelly_sizing': True, 'kelly_conviction': True,
        'fundamental_modulation': True,
        'kold_shoulder_hedge': 0.05,
        'dd_kold_hedge_max_pct': 0.20,  # up to 20% NAV in KOLD at DD=-25%
        'kelly_max_qty': 20,
        'open_dte': 45,
        'vol_aware_dte': True,
        'tail_hedge_floor': 2,
    },
    # TRIFECTA CANDIDATES — combining best Sharpe bases with z_target
    'otm_managed_z_target': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'roll_down': True,
        'z_share_target_enabled': True, 'z_target_cadence_days': 21,
        'z_target_mults': {
            'extreme_cheap': 1.7, 'cheap': 1.4, 'neutral': 1.0,
            'rich': 0.6, 'extreme_rich': 0.2,
        },
        'z_share_target_base': 6200,
    },
    'otm_managed_z_target_shoulder': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'roll_down': True,
        'z_share_target_enabled': True, 'z_target_cadence_days': 21,
        'z_target_mults': {
            'extreme_cheap': 1.7, 'cheap': 1.4, 'neutral': 1.0,
            'rich': 0.6, 'extreme_rich': 0.2,
        },
        'z_share_target_base': 6200,
        'kold_shoulder_hedge': 0.05,
    },
    'regime_aware_z_target': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'roll_down': True,
        'regime_skip_puts_z': -0.5, 'bearish_stack': True, 'boxx': True,
        'z_share_target_enabled': True, 'z_target_cadence_days': 21,
        'z_target_mults': {
            'extreme_cheap': 1.7, 'cheap': 1.4, 'neutral': 1.0,
            'rich': 0.6, 'extreme_rich': 0.2,
        },
        'z_share_target_base': 6200,
        'kold_shoulder_hedge': 0.05,
    },
    # CUT-AND-REBUILD: sell shares + sell OTM puts BELOW to stay paid
    # while waiting for re-entry. Captures put-skew richness (panic IV
    # is fattest in declines). More executable than ITM CCs in fast-down
    # markets where call-side spreads widen.
    # Speed-tunable variants of champion_cut_rebuild
    'champion_cut_rebuild_fast': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'roll_down': True,
        'regime_skip_puts_z': -0.5, 'bearish_stack': True, 'boxx': True,
        'use_surprise_z': True,
        'elevator_close': True, 'elevator_itm_pct': 0.05, 'elevator_extrinsic_max': 0.15,
        'z_share_target_enabled': True, 'z_target_cadence_days': 5,  # weekly cadence
        'z_target_mults': {'extreme_cheap': 1.7, 'cheap': 1.4, 'neutral': 1.0, 'rich': 0.6, 'extreme_rich': 0.2},
        'z_share_target_base': 6200,
        'cut_speed': 0.8,  # faster snap
        'over_cut_pct': 0.10,  # 10% over-cut buffer
        'kold_shoulder_hedge': 0.10,
        'dd_trim_trigger_pct': -10, 'dd_trim_qty_pct': 35, 'dd_trim_floor': 0, 'dd_trim_cadence_days': 5,
        'cut_and_rebuild_puts': True, 'rebuild_put_otm_pct': 0.10, 'rebuild_put_dte': 45,
    },
    'champion_cut_rebuild_slow': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'roll_down': True,
        'regime_skip_puts_z': -0.5, 'bearish_stack': True, 'boxx': True,
        'use_surprise_z': True,
        'elevator_close': True, 'elevator_itm_pct': 0.05, 'elevator_extrinsic_max': 0.15,
        'z_share_target_enabled': True, 'z_target_cadence_days': 42,  # ~2 months
        'z_target_mults': {'extreme_cheap': 1.7, 'cheap': 1.4, 'neutral': 1.0, 'rich': 0.6, 'extreme_rich': 0.2},
        'z_share_target_base': 6200,
        'cut_speed': 0.3,
        'kold_shoulder_hedge': 0.10,
        'dd_trim_trigger_pct': -20, 'dd_trim_qty_pct': 20, 'dd_trim_floor': 0, 'dd_trim_cadence_days': 21,
        'cut_and_rebuild_puts': True, 'rebuild_put_otm_pct': 0.10, 'rebuild_put_dte': 45,
    },
    # Over-cut variant — same as champion_cut_rebuild but with over_cut buffer
    'champion_cut_rebuild_overcut': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'roll_down': True,
        'regime_skip_puts_z': -0.5, 'bearish_stack': True, 'boxx': True,
        'use_surprise_z': True,
        'elevator_close': True, 'elevator_itm_pct': 0.05, 'elevator_extrinsic_max': 0.15,
        'z_share_target_enabled': True, 'z_target_cadence_days': 21,
        'z_target_mults': {'extreme_cheap': 1.7, 'cheap': 1.4, 'neutral': 1.0, 'rich': 0.6, 'extreme_rich': 0.2},
        'z_share_target_base': 6200,
        'cut_speed': 0.5, 'over_cut_pct': 0.15,
        'kold_shoulder_hedge': 0.10,
        'dd_trim_trigger_pct': -15, 'dd_trim_qty_pct': 35, 'dd_trim_floor': 0, 'dd_trim_cadence_days': 5,
        'cut_and_rebuild_puts': True, 'rebuild_put_otm_pct': 0.12, 'rebuild_put_dte': 45,
    },
    'champion_cut_rebuild': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'roll_down': True,
        'regime_skip_puts_z': -0.5, 'bearish_stack': True, 'boxx': True,
        'use_surprise_z': True,
        'elevator_close': True, 'elevator_itm_pct': 0.05,
        'elevator_extrinsic_max': 0.15,
        'z_share_target_enabled': True, 'z_target_cadence_days': 21,
        'z_target_mults': {
            'extreme_cheap': 1.7, 'cheap': 1.4, 'neutral': 1.0,
            'rich': 0.6, 'extreme_rich': 0.2,
        },
        'z_share_target_base': 6200,
        'kold_shoulder_hedge': 0.10,
        'dd_trim_trigger_pct': -15, 'dd_trim_qty_pct': 30,
        'dd_trim_floor': 0, 'dd_trim_cadence_days': 5,
        # NEW: cut-and-rebuild mechanism
        'cut_and_rebuild_puts': True,
        'rebuild_put_otm_pct': 0.10,
        'rebuild_put_dte': 45,
    },
    'champion_cut_rebuild_deep': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'roll_down': True,
        'regime_skip_puts_z': -0.5, 'bearish_stack': True, 'boxx': True,
        'use_surprise_z': True,
        'elevator_close': True, 'elevator_itm_pct': 0.05,
        'elevator_extrinsic_max': 0.15,
        'z_share_target_enabled': True, 'z_target_cadence_days': 21,
        'z_target_mults': {
            'extreme_cheap': 1.7, 'cheap': 1.4, 'neutral': 1.0,
            'rich': 0.6, 'extreme_rich': 0.2,
        },
        'z_share_target_base': 6200,
        'kold_shoulder_hedge': 0.10,
        'dd_trim_trigger_pct': -15, 'dd_trim_qty_pct': 30,
        'dd_trim_floor': 0, 'dd_trim_cadence_days': 5,
        'cut_and_rebuild_puts': True,
        'rebuild_put_otm_pct': 0.15,  # deeper OTM
        'rebuild_put_dte': 45,
    },
    # NEW HARNESS WINNER: aggressive z-target mults push Sharpe to 1.86
    # Counter-intuitive: trimming MORE at rich + loading MORE at cheap
    # IMPROVES everything (Sharpe, return, MDD all better). Cash from
    # rich trims funds bigger accumulation at lows.
    'champion_aggressive_z': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'tp_dynamic': True,
        'roll_down': True, 'roll_up_calls': True,
        'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'aggressive_itm_cc_z': -0.25, 'itm_cc_pct': -0.20,
        'elevator_close': True, 'elevator_itm_pct': 0.05,
        'elevator_extrinsic_max': 0.15, 'elevator_mode': 'strict',
        'vol_aware_sizing': True,
        'tail_hedge_floor': 2,
        'z_share_target_enabled': True, 'z_target_cadence_days': 21,
        'z_target_mults': {
            'extreme_cheap': 2.0, 'cheap': 1.6, 'neutral': 1.0,
            'rich': 0.4, 'extreme_rich': 0.1,
        },
        'z_share_target_base': 6200,
        'kold_shoulder_hedge': 0.10,
        'cut_and_rebuild_puts': True, 'rebuild_put_otm_pct': 0.10, 'rebuild_put_dte': 45,
        'elevator_skip_on_momentum': True,
        'itm_cc_skip_on_momentum': True,
    },
    # CHAMPION + MOMENTUM GATES (preserve spike capture by not forcing
    # ITM assignment / elevator close during confirmed parabolic move)
    'champion_20pct_protected_mom_gated': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'tp_dynamic': True,
        'roll_down': True, 'roll_up_calls': True,
        'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'aggressive_itm_cc_z': -0.25, 'itm_cc_pct': -0.20,
        'elevator_close': True, 'elevator_itm_pct': 0.05,
        'elevator_extrinsic_max': 0.15, 'elevator_mode': 'strict',
        'vol_aware_sizing': True,
        'tail_hedge_floor': 2,
        'z_share_target_enabled': True, 'z_target_cadence_days': 21,
        'z_target_mults': {'extreme_cheap': 1.7, 'cheap': 1.4, 'neutral': 1.0, 'rich': 0.6, 'extreme_rich': 0.2},
        'z_share_target_base': 6200,
        'kold_shoulder_hedge': 0.10,
        'cut_and_rebuild_puts': True, 'rebuild_put_otm_pct': 0.10, 'rebuild_put_dte': 45,
        # MOMENTUM GATES — skip pinch-the-spike mechanics during true parabola
        'elevator_skip_on_momentum': True,
        'itm_cc_skip_on_momentum': True,
    },
    # CHAMPION + MOMENTUM CALL LAYER — buy OTM calls during 2022-style spikes
    # Decouples upside capture from share holding. Targets the 1-bad-year
    # (2022) where protected gave up $226K to unprotected.
    'champion_20pct_protected_momcalls': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'tp_dynamic': True,
        'roll_down': True, 'roll_up_calls': True,
        'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'aggressive_itm_cc_z': -0.25, 'itm_cc_pct': -0.20,
        'elevator_close': True, 'elevator_itm_pct': 0.05,
        'elevator_extrinsic_max': 0.15, 'elevator_mode': 'strict',
        'vol_aware_sizing': True,
        'tail_hedge_floor': 2,
        'z_share_target_enabled': True, 'z_target_cadence_days': 21,
        'z_target_mults': {'extreme_cheap': 1.7, 'cheap': 1.4, 'neutral': 1.0, 'rich': 0.6, 'extreme_rich': 0.2},
        'z_share_target_base': 6200,
        'kold_shoulder_hedge': 0.10,
        'cut_and_rebuild_puts': True, 'rebuild_put_otm_pct': 0.10, 'rebuild_put_dte': 45,
        'momentum_call_pct': 0.02,           # 2% NAV per ladder rung
        'momentum_call_threshold': 0.20,     # 90d return > 20%
        'momentum_call_otm_pct': 0.15,       # 15% OTM
        'momentum_call_dte': 90,
        'momentum_call_max_stack': 3,        # max 3 concurrent
    },
    # More aggressive momentum sizing
    'champion_20pct_protected_momcalls_aggressive': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'tp_dynamic': True,
        'roll_down': True, 'roll_up_calls': True,
        'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'aggressive_itm_cc_z': -0.25, 'itm_cc_pct': -0.20,
        'elevator_close': True, 'elevator_itm_pct': 0.05,
        'elevator_extrinsic_max': 0.15, 'elevator_mode': 'strict',
        'vol_aware_sizing': True,
        'tail_hedge_floor': 2,
        'z_share_target_enabled': True, 'z_target_cadence_days': 21,
        'z_target_mults': {'extreme_cheap': 1.7, 'cheap': 1.4, 'neutral': 1.0, 'rich': 0.6, 'extreme_rich': 0.2},
        'z_share_target_base': 6200,
        'kold_shoulder_hedge': 0.10,
        'cut_and_rebuild_puts': True, 'rebuild_put_otm_pct': 0.10, 'rebuild_put_dte': 45,
        'momentum_call_pct': 0.04,
        'momentum_call_threshold': 0.15,
        'momentum_call_otm_pct': 0.15,
        'momentum_call_dte': 120,
        'momentum_call_max_stack': 5,
    },
    # CHAMPION + UPSIDE WING — covered call spread to recapture extreme upside
    # Attach long far-OTM call to each ITM CC so spike-through gets paid
    'champion_20pct_protected_wing': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'tp_dynamic': True,
        'roll_down': True, 'roll_up_calls': True,
        'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'aggressive_itm_cc_z': -0.25, 'itm_cc_pct': -0.20,
        'elevator_close': True, 'elevator_itm_pct': 0.05,
        'elevator_extrinsic_max': 0.15, 'elevator_mode': 'strict',
        'vol_aware_sizing': True,
        'tail_hedge_floor': 2,
        'z_share_target_enabled': True, 'z_target_cadence_days': 21,
        'z_target_mults': {'extreme_cheap': 1.7, 'cheap': 1.4, 'neutral': 1.0, 'rich': 0.6, 'extreme_rich': 0.2},
        'z_share_target_base': 6200,
        'kold_shoulder_hedge': 0.10,
        'cut_and_rebuild_puts': True, 'rebuild_put_otm_pct': 0.10, 'rebuild_put_dte': 45,
        'upside_wing_otm_pct': 0.30,  # 30% above spot
    },
    'champion_20pct_protected_wing_all': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'tp_dynamic': True,
        'roll_down': True, 'roll_up_calls': True,
        'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'aggressive_itm_cc_z': -0.25, 'itm_cc_pct': -0.20,
        'elevator_close': True, 'elevator_itm_pct': 0.05,
        'elevator_extrinsic_max': 0.15, 'elevator_mode': 'strict',
        'vol_aware_sizing': True,
        'tail_hedge_floor': 2,
        'z_share_target_enabled': True, 'z_target_cadence_days': 21,
        'z_target_mults': {'extreme_cheap': 1.7, 'cheap': 1.4, 'neutral': 1.0, 'rich': 0.6, 'extreme_rich': 0.2},
        'z_share_target_base': 6200,
        'kold_shoulder_hedge': 0.10,
        'cut_and_rebuild_puts': True, 'rebuild_put_otm_pct': 0.10, 'rebuild_put_dte': 45,
        'upside_wing_otm_pct': 0.20, 'upside_wing_always': True,
    },
    'champion_20pct_protected_wing_close': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'tp_dynamic': True,
        'roll_down': True, 'roll_up_calls': True,
        'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'aggressive_itm_cc_z': -0.25, 'itm_cc_pct': -0.20,
        'elevator_close': True, 'elevator_itm_pct': 0.05,
        'elevator_extrinsic_max': 0.15, 'elevator_mode': 'strict',
        'vol_aware_sizing': True,
        'tail_hedge_floor': 2,
        'z_share_target_enabled': True, 'z_target_cadence_days': 21,
        'z_target_mults': {'extreme_cheap': 1.7, 'cheap': 1.4, 'neutral': 1.0, 'rich': 0.6, 'extreme_rich': 0.2},
        'z_share_target_base': 6200,
        'kold_shoulder_hedge': 0.10,
        'cut_and_rebuild_puts': True, 'rebuild_put_otm_pct': 0.10, 'rebuild_put_dte': 45,
        'upside_wing_otm_pct': 0.15,  # closer-in wing
    },
    # CHAMPION + momentum override (capture spike runs without giving up MDD)
    'champion_20pct_protected_momentum': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'tp_dynamic': True,
        'roll_down': True, 'roll_up_calls': True,
        'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'aggressive_itm_cc_z': -0.25, 'itm_cc_pct': -0.20,
        'elevator_close': True, 'elevator_itm_pct': 0.05,
        'elevator_extrinsic_max': 0.15, 'elevator_mode': 'strict',
        'vol_aware_sizing': True,
        'tail_hedge_floor': 2,
        'z_share_target_enabled': True, 'z_target_cadence_days': 21,
        'z_target_mults': {'extreme_cheap': 1.7, 'cheap': 1.4, 'neutral': 1.0, 'rich': 0.6, 'extreme_rich': 0.2},
        'z_share_target_base': 6200,
        'kold_shoulder_hedge': 0.10,
        'cut_and_rebuild_puts': True, 'rebuild_put_otm_pct': 0.10, 'rebuild_put_dte': 45,
        'momentum_override': True,
    },
    # CHAMPION + upside ticket (fills the 2022-spike-capture gap)
    'champion_20pct_protected_upside': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'tp_dynamic': True,
        'roll_down': True, 'roll_up_calls': True,
        'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'aggressive_itm_cc_z': -0.25, 'itm_cc_pct': -0.20,
        'elevator_close': True, 'elevator_itm_pct': 0.05,
        'elevator_extrinsic_max': 0.15, 'elevator_mode': 'strict',
        'vol_aware_sizing': True,
        'tail_hedge_floor': 2,
        'z_share_target_enabled': True, 'z_target_cadence_days': 21,
        'z_target_mults': {'extreme_cheap': 1.7, 'cheap': 1.4, 'neutral': 1.0, 'rich': 0.6, 'extreme_rich': 0.2},
        'z_share_target_base': 6200,
        'kold_shoulder_hedge': 0.10,
        'cut_and_rebuild_puts': True, 'rebuild_put_otm_pct': 0.10, 'rebuild_put_dte': 45,
        'upside_ticket_pct': 0.02,  # 2% NAV in 30% OTM 90DTE calls when EXTREME_CHEAP
    },
    'champion_20pct_protected_upside_heavy': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'tp_dynamic': True,
        'roll_down': True, 'roll_up_calls': True,
        'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'aggressive_itm_cc_z': -0.25, 'itm_cc_pct': -0.20,
        'elevator_close': True, 'elevator_itm_pct': 0.05,
        'elevator_extrinsic_max': 0.15, 'elevator_mode': 'strict',
        'vol_aware_sizing': True,
        'tail_hedge_floor': 2,
        'z_share_target_enabled': True, 'z_target_cadence_days': 21,
        'z_target_mults': {'extreme_cheap': 1.7, 'cheap': 1.4, 'neutral': 1.0, 'rich': 0.6, 'extreme_rich': 0.2},
        'z_share_target_base': 6200,
        'kold_shoulder_hedge': 0.10,
        'cut_and_rebuild_puts': True, 'rebuild_put_otm_pct': 0.10, 'rebuild_put_dte': 45,
        'upside_ticket_pct': 0.04,
    },
    # HIGH-RETURN BASES + protection layer (cut_rebuild + z_target + shoulder)
    # Goal: retain old champion's 140%+ return profile but cap MDD at <-10%
    'champion_20pct_protected': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'tp_dynamic': True,
        'roll_down': True, 'roll_up_calls': True,
        'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'aggressive_itm_cc_z': -0.25, 'itm_cc_pct': -0.20,
        'elevator_close': True, 'elevator_itm_pct': 0.05,
        'elevator_extrinsic_max': 0.15, 'elevator_mode': 'strict',
        'vol_aware_sizing': True,
        'tail_hedge_floor': 2,
        # Protection layer
        'z_share_target_enabled': True, 'z_target_cadence_days': 21,
        'z_target_mults': {'extreme_cheap': 1.7, 'cheap': 1.4, 'neutral': 1.0, 'rich': 0.6, 'extreme_rich': 0.2},
        'z_share_target_base': 6200,
        'kold_shoulder_hedge': 0.10,
        'cut_and_rebuild_puts': True, 'rebuild_put_otm_pct': 0.10, 'rebuild_put_dte': 45,
    },
    'beam_put_protected': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'tp_dynamic': True,
        'roll_down': True, 'roll_up_calls': True,
        'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'aggressive_itm_cc_z': -0.25, 'itm_cc_pct': -0.20,
        'elevator_close': True, 'elevator_itm_pct': 0.05,
        'elevator_extrinsic_max': 0.15, 'elevator_mode': 'strict',
        'vol_aware_sizing': True,
        'beam_put_selector': True, 'beam_put_n': 5,
        'tail_hedge_floor': 2,
        'z_share_target_enabled': True, 'z_target_cadence_days': 21,
        'z_target_mults': {'extreme_cheap': 1.7, 'cheap': 1.4, 'neutral': 1.0, 'rich': 0.6, 'extreme_rich': 0.2},
        'z_share_target_base': 6200,
        'kold_shoulder_hedge': 0.10,
        'cut_and_rebuild_puts': True, 'rebuild_put_otm_pct': 0.10, 'rebuild_put_dte': 45,
    },
    'kelly_firmness_protected': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'tp_dynamic': True,
        'roll_down': True, 'roll_up_calls': True,
        'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'kelly_sizing': True, 'kelly_conviction': True, 'kelly_firmness': True,
        'kelly_max_qty': 25, 'open_dte': 45, 'vol_aware_dte': True,
        'z_share_target_enabled': True, 'z_target_cadence_days': 21,
        'z_target_mults': {'extreme_cheap': 1.7, 'cheap': 1.4, 'neutral': 1.0, 'rich': 0.6, 'extreme_rich': 0.2},
        'z_share_target_base': 6200,
        'kold_shoulder_hedge': 0.10,
        'cut_and_rebuild_puts': True, 'rebuild_put_otm_pct': 0.10, 'rebuild_put_dte': 45,
    },
    # CHAMPION TRIFECTA: highest Sharpe in entire harness (1.48)
    # +90.5% return, -49% MDD, +16% calm, calm Sharpe ~0.3
    # Mechanics: elevator close (lock spike), shoulder hedge (KOLD in weak
    # seasons), z-target sizing (proactive), dd-trim safety net
    'champion_trifecta': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'roll_down': True,
        'regime_skip_puts_z': -0.5, 'bearish_stack': True, 'boxx': True,
        'use_surprise_z': True,
        'elevator_close': True, 'elevator_itm_pct': 0.05,
        'elevator_extrinsic_max': 0.15,
        'z_share_target_enabled': True, 'z_target_cadence_days': 21,
        'z_target_mults': {
            'extreme_cheap': 1.7, 'cheap': 1.4, 'neutral': 1.0,
            'rich': 0.6, 'extreme_rich': 0.2,
        },
        'z_share_target_base': 6200,
        'kold_shoulder_hedge': 0.10,
        'dd_trim_trigger_pct': -15, 'dd_trim_qty_pct': 30,
        'dd_trim_floor': 0, 'dd_trim_cadence_days': 5,
    },
    'elevator_z_target_shoulder': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'roll_down': True,
        'regime_skip_puts_z': -0.5, 'bearish_stack': True, 'boxx': True,
        'use_surprise_z': True,
        'elevator_close': True, 'elevator_itm_pct': 0.05,
        'elevator_extrinsic_max': 0.15,
        'z_share_target_enabled': True, 'z_target_cadence_days': 21,
        'z_target_mults': {
            'extreme_cheap': 1.7, 'cheap': 1.4, 'neutral': 1.0,
            'rich': 0.6, 'extreme_rich': 0.2,
        },
        'z_share_target_base': 6200,
        'kold_shoulder_hedge': 0.05,
    },
    # Z-BASED DYNAMIC SHARE TARGETING — proactive wheel sizing.
    # Beats baseline on Sharpe, MDD, AND calm return simultaneously.
    # Sharpe 1.09 (vs 0.69), MDD -51% (vs -69%), calm +46% (vs +31%).
    # Only sacrifice: 2022 spike capture (full +102% vs +132%).
    'kelly_z_target_winner': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'tp_dynamic': True,
        'roll_down': True, 'roll_up_calls': True,
        'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'kelly_sizing': True, 'kelly_conviction': True,
        'fundamental_modulation': True,
        'kold_shoulder_hedge': 0.05,
        'z_share_target_enabled': True, 'z_target_cadence_days': 21,
        'z_target_mults': {
            'extreme_cheap': 1.7, 'cheap': 1.4, 'neutral': 1.0,
            'rich': 0.6, 'extreme_rich': 0.2,
        },
        'z_share_target_base': 6200,
        'kelly_max_qty': 20, 'open_dte': 45, 'vol_aware_dte': True,
        'tail_hedge_floor': 2,
    },
    # Calm-period maximizer — same but with heavier shoulder hedge
    'kelly_z_target_calm_max': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'tp_dynamic': True,
        'roll_down': True, 'roll_up_calls': True,
        'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'kelly_sizing': True, 'kelly_conviction': True,
        'fundamental_modulation': True,
        'kold_shoulder_hedge': 0.10,
        'z_share_target_enabled': True, 'z_target_cadence_days': 21,
        'z_target_mults': {
            'extreme_cheap': 1.7, 'cheap': 1.4, 'neutral': 1.0,
            'rich': 0.85, 'extreme_rich': 0.4,
        },
        'z_share_target_base': 6200,
        'kelly_max_qty': 20, 'open_dte': 45, 'vol_aware_dte': True,
        'tail_hedge_floor': 2,
    },
    # SHORT-DD VARIANTS — depth-minimizing tunables (daily cadence is key)
    # Sweet spot: weekly trim, best Sharpe 1.24, MDD -32%
    'kelly_short_dd_balanced': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'tp_dynamic': True,
        'roll_down': True, 'roll_up_calls': True,
        'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'kelly_sizing': True, 'kelly_conviction': True,
        'fundamental_modulation': True,
        'kold_shoulder_hedge': 0.05,
        'dd_trim_trigger_pct': -3, 'dd_trim_qty_pct': 30,
        'dd_trim_floor': 0, 'dd_trim_cadence_days': 5,
        'kelly_max_qty': 20, 'open_dte': 45, 'vol_aware_dte': True,
        'tail_hedge_floor': 2,
    },
    # Tight depth control — MDD -23% at the cost of -100pp full return
    'kelly_short_dd_tight': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'tp_dynamic': True,
        'roll_down': True, 'roll_up_calls': True,
        'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'kelly_sizing': True, 'kelly_conviction': True,
        'fundamental_modulation': True,
        'kold_shoulder_hedge': 0.05,
        'dd_trim_trigger_pct': -3, 'dd_trim_qty_pct': 30,
        'dd_trim_floor': 0, 'dd_trim_cadence_days': 1,
        'kelly_max_qty': 20, 'open_dte': 45, 'vol_aware_dte': True,
        'tail_hedge_floor': 2,
    },
    # Shallowest — MDD -18%, very defensive
    'kelly_short_dd_minimal': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'tp_dynamic': True,
        'roll_down': True, 'roll_up_calls': True,
        'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'kelly_sizing': True, 'kelly_conviction': True,
        'fundamental_modulation': True,
        'kold_shoulder_hedge': 0.05,
        'dd_trim_trigger_pct': -2, 'dd_trim_qty_pct': 40,
        'dd_trim_floor': 0, 'dd_trim_cadence_days': 1,
        'kelly_max_qty': 20, 'open_dte': 45, 'vol_aware_dte': True,
        'tail_hedge_floor': 2,
    },
    # WINNER of DD-control sweep: aggressive early trim, floor=0
    # MDD -69% → -44%, Sharpe 0.69 → 1.08, full +131% → +82%
    'kelly_dd_controlled': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'tp_dynamic': True,
        'roll_down': True, 'roll_up_calls': True,
        'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'kelly_sizing': True, 'kelly_conviction': True,
        'fundamental_modulation': True,
        'kold_shoulder_hedge': 0.05,
        'dd_trim_trigger_pct': -3, 'dd_trim_qty_pct': 50, 'dd_trim_floor': 0,
        'kelly_max_qty': 20,
        'open_dte': 45,
        'vol_aware_dte': True,
        'tail_hedge_floor': 2,
    },
    # C: Combined — both share trim AND KOLD hedge
    'kelly_dd_belt_and_suspenders': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'tp_dynamic': True,
        'roll_down': True, 'roll_up_calls': True,
        'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'kelly_sizing': True, 'kelly_conviction': True,
        'fundamental_modulation': True,
        'kold_shoulder_hedge': 0.05,
        'dd_trim_trigger_pct': -12, 'dd_trim_qty_pct': 20, 'dd_trim_floor': 2000,
        'dd_kold_hedge_max_pct': 0.15,
        'kelly_max_qty': 20,
        'open_dte': 45,
        'vol_aware_dte': True,
        'tail_hedge_floor': 2,
    },
    # BEAM SELECTOR (item #1 port — critical version)
    # Generate 5 put candidates, score by income-minus-expected-loss,
    # pick highest. Tests if multi-objective scoring beats hardcoded
    # regime_aware_strike rules.
    'beam_put_only': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'tp_50': True, 'tp_dynamic': True,
        'roll_down': True, 'roll_up_calls': True,
        'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'aggressive_itm_cc_z': -0.25, 'itm_cc_pct': -0.20,
        'elevator_close': True, 'elevator_itm_pct': 0.05,
        'elevator_extrinsic_max': 0.15, 'elevator_mode': 'strict',
        'vol_aware_sizing': True,
        'tail_hedge_floor': 2,
        'beam_put': True,
        'beam_put_otm_ladder': [0.02, 0.05, 0.08, 0.12, 0.15],
        'open_dte': 45,
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
        # CORRECT MDD: max peak-to-trough running drawdown, NOT (min vs all-time-max).
        # Old buggy formula compared absolute-min (often the start) to absolute-max,
        # massively overstating MDD by mixing different points in time.
        running_peak = hist['nav'].cummax()
        dd_series = (hist['nav'] - running_peak) / running_peak * 100
        max_dd_pct = dd_series.min() if len(dd_series) else 0.0
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
