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
from datetime import datetime, timedelta
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from seasonal_z import add_seasonal_factors  # type: ignore
from iv_model import precompute_realized_vol, iv_for_quote  # type: ignore
from attribution import attribute_trades, print_attribution  # type: ignore
from kelly_sizing import kelly_qty_short_put, kelly_qty_covered_call  # type: ignore
from assignment_model import assignment_probability, expected_value_wait_vs_btc  # type: ignore
from scenario_distribution import ScenarioDistribution  # type: ignore

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# WS = zero commission
COMMISSION = 0.0
# Fill realism: fraction-of-mid optimism for the spread-width/DTE fill model
# (real_chain_price). 1.0 = the calibrated P(mid) model; >1 fills nearer mid
# (more optimistic), <1 nearer the touch (more pessimistic). Honest middle ground
# between always-mid (too good) and always-touch (too harsh).
FILL_MID_OPTIMISM = 1.0
SPREAD_OPTION = 0.07  # $0.07/share half-spread, calibrated to MEASURED UNG NBBO
# (2026-06-12): near-term legs ~$0.04 wide (~$0.02 half), but the 45-60d legs the
# kernel actually trades run ~$0.14-0.19 wide (~$0.07-0.09 half). Modal DTE ~45d →
# $0.07. The old $0.05 flat was ~$0.04 too low on the legs that dominate, inflating
# returns ~1.7pp/yr (see roll-friction audit 2026-06-15). DTE-aware spread is a TODO.
                      # on ATM is typically $0.04-0.10 wide → half ≈ 0.02-0.05
                      # Bumped from 0.03 for realism; honest_walkforward also
                      # adds 5% slippage on opens — together = full-cycle realism.
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


# Z-conditional, DTE-aware P(short put assigns within its own DTE) for the gamma-aware concentration
# cap. drift μ(z)=a+b·z (UNG daily; fit in research/spy_vol/ung_scenario_delta.py), vol scales √dte —
# so time is first-class: a short-DTE OTM put barely counts, a long-DTE near-money put counts a lot.
def p_assign(K, S, dte_days, z, a=-0.001205, b=-0.000112, sig=0.04066):
    if dte_days <= 0 or S <= 0:
        return 1.0 if S < K else 0.0
    d = (math.log(K / S) - (a + b * z) * dte_days) / (sig * math.sqrt(dte_days))
    return norm.cdf(d)


def bs_greeks_pt(S, K, T, sig, right, r=0.045):
    """Per-share (delta, gamma) for one option contract. delta∈[-1,1]."""
    if T <= 0.001 or sig <= 0 or S <= 0:
        intr = (S > K) if right == 'C' else (S < K)
        return ((1.0 if right == 'C' else -1.0) if intr else 0.0), 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sig ** 2) * T) / (sig * math.sqrt(T))
    delta = norm.cdf(d1) if right == 'C' else norm.cdf(d1) - 1.0
    gamma = norm.pdf(d1) / (S * sig * math.sqrt(T))
    return delta, gamma


def long_calls_mtm(long_calls, spot, idx, iv_fn):
    """Mark-to-market value of long calls. The NAV is otherwise cash-basis (cash+shares+boxx+kold)
    and would NOT count a held long-call asset until it settles — penalizing any long-call strategy
    (premium leaves cash now, value realized only at expiry). Mark it so the calls-accumulation arm
    is valued fairly against the shares arm."""
    v = 0.0
    for lc in (long_calls or []):
        T = max(1, lc['dte'] - (idx - lc['entry']).days) / 365.0
        v += bs_call(spot, lc['K'], T, iv_fn(lc['K'], int(T * 365), 'C')) * lc['qty'] * 100
    return v


def book_greeks(s, spot, iv_at):
    """Aggregate book net DELTA and GAMMA (in share-equivalents) across shares + all
    short/long puts & calls. Long delta = bullish exposure. Short put adds +delta,
    short call adds −delta, long put adds −delta (the bearish hedge)."""
    nd = float(s.get('shares', 0)); ng = 0.0
    for sp in s.get('short_puts', []):
        d, g = bs_greeks_pt(spot, sp['K'], max(sp.get('dte', 30), 1) / 365,
                            iv_at(sp['K'], sp.get('dte', 30), 'P'), 'P')
        nd += -sp['qty'] * d * 100; ng += -sp['qty'] * g * 100
    for sc in s.get('short_calls', []):
        d, g = bs_greeks_pt(spot, sc['K'], max(sc.get('dte', 30), 1) / 365,
                            iv_at(sc['K'], sc.get('dte', 30), 'C'), 'C')
        nd += -sc['qty'] * d * 100; ng += -sc['qty'] * g * 100
    for lp in s.get('long_puts', []):
        d, g = bs_greeks_pt(spot, lp['K'], max(lp.get('dte', 30), 1) / 365,
                            iv_at(lp['K'], lp.get('dte', 30), 'P'), 'P')
        nd += lp['qty'] * d * 100; ng += lp['qty'] * g * 100
    return nd, ng


def book_greeks_stat(s, spot, z, a=-0.000797, b=-0.000009, sig=0.0390):
    """GAMMA → DELTA via the statistical model. The expected SHARE position after every current
    short option resolves at its OWN expiry, under the REAL-WORLD drift μ(z)=a+b·z (not risk-neutral
    BS). A short put assigns into a decline → +p_assign·qty·100; a short call is called away →
    −P(called)·qty·100 = −(1−p_assign)·qty·100. This projects the book's GAMMA (short-option
    curvature) into its expected DELTA impact, DTE-weighted — so the hedge targets where the book
    actually drifts, not the instantaneous risk-neutral snapshot. Same fn in backtest and live."""
    nd = float(s.get('shares', 0))
    for sp in s.get('short_puts', []):
        nd += sp['qty'] * p_assign(sp['K'], spot, sp.get('dte', 30), z, a, b, sig) * 100
    for sc in s.get('short_calls', []):
        nd -= sc['qty'] * (1.0 - p_assign(sc['K'], spot, sc.get('dte', 30), z, a, b, sig)) * 100
    for lp in s.get('long_puts', []):
        nd -= lp['qty'] * p_assign(lp['K'], spot, lp.get('dte', 30), z, a, b, sig) * 100
    return nd


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
        # Surge-z for assignment model: spot vs 20d MA/sd (mean-reversion signal)
        _ma20 = df['UNG'].rolling(20).mean()
        _sd20 = df['UNG'].rolling(20).std()
        df['ung_surge_z'] = ((df['UNG'] - _ma20) / _sd20.replace(0, float('nan'))).fillna(0.0)
    # GEX call wall (real OI history, split-adjusted) — for cc_gex_floor
    try:
        _gw = pd.read_csv(os.path.join(CACHE_DIR, 'ung_gex_wall_daily.csv'),
                          parse_dates=['date']).set_index('date')
        df['gex_call_wall'] = _gw['gex_call_wall_adj'].reindex(
            df.index, method='ffill', limit=5)
    except Exception:
        df['gex_call_wall'] = float('nan')
    # IV-rank (252d pct of real ATM IV) — top quintile → -23% fwd-63d
    # (p=.002, [[project_ung_iv_rank_alpha]]) — for iv_rank_z_scale
    try:
        _ivr = pd.read_csv(os.path.join(CACHE_DIR, 'ung_iv_rank_daily.csv'),
                           index_col=0, parse_dates=True)
        df['iv_rank'] = _ivr['iv_rank'].reindex(df.index, method='ffill', limit=10)
    except Exception:
        df['iv_rank'] = float('nan')
    # BUGFIX 2026-06-16: the features below were mis-indented INSIDE the except
    # above, so they only ran when the iv_rank CSV FAILED to load — i.e. almost
    # never. That silently left hh_basis / MAs / trend flags absent in every
    # normal backtest. Dedented to function level so they're ALWAYS computed.
    # HH basis storm: spot-futures backwardation > +$0.40 → defensive (validated:
    # top-5% basis → UNG -3.7% fwd-5d, see timing_signals_eval).
    if 'eia_hh_spot_daily' in df.columns:
        # ffill BOTH legs: NG futures is only ~50% populated, so a raw spot−NG
        # zeros out most days. Forward-fill (≤5d) for a continuous basis.
        df['hh_basis'] = (df['eia_hh_spot_daily'].ffill(limit=5)
                          - df['NG'].ffill(limit=5)).fillna(0.0)
        df['hh_basis_storm'] = (df['hh_basis'] > 0.40).astype(int)
    else:
        df['hh_basis'] = 0.0
        df['hh_basis_storm'] = 0
    df['ung_5d_mom'] = df['UNG'].pct_change(5)
    df['ung_30d_return'] = df['UNG'].pct_change(30)  # for grind-down detection
    df['ung_60d_high'] = df['UNG'].rolling(60).max()
    df['ung_252d_mean'] = df['UNG'].rolling(252).mean()
    df['ung_252d_std'] = df['UNG'].rolling(252).std()
    df['ung_200d_ma'] = df['UNG'].rolling(200).mean()
    df['ung_50d_ma'] = df['UNG'].rolling(50).mean()
    df['ung_uptrend'] = (df['UNG'] > df['ung_50d_ma']) & (df['ung_50d_ma'] > df['ung_200d_ma'])
    df['ung_downtrend'] = (df['UNG'] < df['ung_200d_ma']) & (df['ung_50d_ma'] < df['ung_200d_ma'])
    df = precompute_realized_vol(df, col='UNG')
    # ── NaN GUARD (user requirement: no kernel decision is ever made on NaN).
    # Two policies, applied here at the single source so all downstream logic
    # is safe regardless of the (NaN-leaky) `float(x or D)` idiom:
    #   (1) DIRECTIONAL signals → fill NaN with their NEUTRAL default, so a
    #       missing value reads as "no signal", never as an accidental NaN.
    #   (2) AGGRESSIVE-GATE signals (iv_rank, gex_call_wall) are intentionally
    #       LEFT NaN — their gates already require a present value, so an
    #       unknown reading correctly DISABLES the aggressive action rather
    #       than firing it on a substituted guess.
    _neutral0 = ['storage_z', 'days_supply_z', 'storage_surprise_z',
                 'ung_5d_mom', 'ung_spike_60d', 'ung_surge_z', 'ung_5d_mom']
    for _c in _neutral0:
        if _c in df.columns:
            df[_c] = df[_c].fillna(0.0)
    if 'rv_30' in df.columns:
        df['rv_30'] = df['rv_30'].fillna(0.5)
    # QUANTIFIED REGIME STRENGTH s∈[-1,+1] — causal sticky Markov FILTER (not a fitted
    # HMM: avoids spurious regimes on a short sample; the persistence justified by the
    # 0.94-0.97 stay-probs measured via statsmodels MarkovRegression on storage_surprise_z).
    # sign = direction (+distribute / -accumulate), magnitude = STRENGTH (strong vs weak).
    # s_t = 0.8·s_{t-1} + 0.2·tanh(0.9·ssz − 0.3·min(0,surge))  ≡ EWMA(alpha=0.2).
    _ssz = df['storage_surprise_z'] if 'storage_surprise_z' in df.columns else 0.0 * df['UNG']
    _surge = df['ung_surge_z'] if 'ung_surge_z' in df.columns else 0.0 * df['UNG']
    _ev = np.tanh(0.9 * _ssz.fillna(0.0) - 0.3 * _surge.fillna(0.0).clip(upper=0.0))
    df['regime_strength'] = _ev.ewm(alpha=0.2, adjust=False).mean()
    # SIGNAL-NOISE (walk-forward early-alpha finding: ssz_vol corr -0.83 with edge) —
    # rolling std of the storage-surprise signal; high = the regime is unreliable → gate it.
    df['ssz_vol'] = _ssz.fillna(0.0).rolling(63, min_periods=10).std().fillna(0.0)
    # PRICE-CRASH signal (drawdown from the 60d high) — drives the crash-fallback meta-regime
    # that catches PRICE crashes the fundamental storage signal misses (the 2022 blind spot).
    df['ung_dd_60'] = (df['UNG'] / df['UNG'].rolling(60, min_periods=20).max() - 1.0).fillna(0.0)
    # NOAA DEGREE-DAY anomaly (persisted: cache/noaa_degree_days_daily.csv) — daily
    # gas-weighted HDD+CDD vs day-of-year seasonal normal, z-scored. Orthogonal to storage
    # and LEADS the storage print ~1wk. Used to nowcast the storage regime earlier.
    try:
        _dd = pd.read_csv(os.path.join(CACHE_DIR, 'noaa_degree_days_daily.csv'),
                          index_col=0, parse_dates=True).iloc[:, 0]
        _dd = _dd[~_dd.index.duplicated()].reindex(df.index, method='ffill')
        _ddn = _dd - _dd.groupby(_dd.index.dayofyear).transform('mean')
        df['dd_anom_z'] = ((_ddn - _ddn.rolling(252, min_periods=30).mean())
                           / (_ddn.rolling(252, min_periods=30).std() + 1e-9)).fillna(0.0)
    except Exception:
        df['dd_anom_z'] = 0.0
    # OU MEAN-REVERSION z (short-scale buy-low/sell-high) — rolling AR(1) fit on log-UNG.
    # Negative = cheap vs local mean (bounce expected → good to sell puts); positive = rich
    # (fade expected → tilt to calls). |z| sizes the tactical tilt. Orthogonal to storage regime.
    try:
        from ou_model import ou_z_series
        df['ou_z'] = ou_z_series(df['UNG'], 90).clip(-4, 4).reindex(df.index).fillna(0.0)
    except Exception:
        df['ou_z'] = 0.0
    return df


def detect_grind_down(row) -> bool:
    """True when UNG is in a slow grind-down (chronic chronic):
    - 30-day return < -8% AND
    - 5-day return < 0 (still falling)
    Catches the Dec 2023-style multi-week declines that anomaly misses.
    """
    r30 = row.get('ung_30d_return', None)
    r5 = row.get('ung_5d_mom', None)
    if r30 is None or r5 is None: return False
    try:
        return float(r30) < -0.08 and float(r5) < 0
    except (ValueError, TypeError):
        return False


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


_IV_SURFACE_CACHE = None  # lazy-loaded once per process
_CAPTURE_STATES = False   # test-only: snapshot start-of-day book (live==backtest equivalence)
_STATE_SNAPSHOTS = {}
_REAL_STRIKE_GRID = None  # {(date_str, right): sorted_list_of_strikes_adj}
_REAL_STRIKE_GRID_BY_EXP = None  # {(date_str, exp_str, right): sorted_strikes}


def _is_third_friday(d):
    """True if date d is the 3rd Friday of its month (monthly expiry)."""
    return d.weekday() == 4 and 15 <= d.day <= 21


def _heuristic_strike_grid(spot, expiration_str):
    """Fallback strike grid when PG doesn't cover (date, expiration).

    UNG empirical rules:
    - Monthly expirations (3rd Friday): integer strikes only ($1 increments
      when spot ≤ $20; $2.50/$5 increments above that).
    - Weekly expirations: half-strikes ($0.50 increments) when spot ≤ $20,
      else $1 increments.
    """
    from datetime import date as _date
    try:
        exp_d = _date.fromisoformat(expiration_str)
    except Exception:
        exp_d = None
    is_monthly = exp_d is not None and _is_third_friday(exp_d)
    if spot <= 0:
        spot = 12.0
    # Generate a reasonable window around spot
    low = max(1.0, spot * 0.5)
    high = spot * 1.5
    if is_monthly:
        inc = 1.0 if spot <= 20 else 2.5
    else:
        inc = 0.5 if spot <= 20 else 1.0
    # Snap low to inc grid
    n0 = int(low / inc)
    strikes = []
    for i in range(n0, int(high / inc) + 2):
        strikes.append(round(i * inc, 2))
    return strikes


def _load_real_strike_grid():
    """Load real listed UNG strikes per (date, right) and per (date, exp, right)
    from PG ung_iv_surface. Cached once per process.
    """
    global _REAL_STRIKE_GRID, _REAL_STRIKE_GRID_BY_EXP
    if _REAL_STRIKE_GRID is not None:
        return _REAL_STRIKE_GRID
    try:
        import psycopg2
        conn = psycopg2.connect(
            host='192.168.1.172', port=5432, database='market_scanner',
            user='postgres', password='shinobi2025', connect_timeout=5,
        )
        cur = conn.cursor()
        cur.execute(
            'SELECT date, expiration, option_right, strike_adj FROM ung_iv_surface '
            'ORDER BY date, expiration, option_right, strike_adj'
        )
        grid = {}
        grid_by_exp = {}
        for d, exp, r, k in cur.fetchall():
            d_str = d.isoformat(); exp_str = exp.isoformat(); k_f = float(k)
            grid.setdefault((d_str, r), []).append(k_f)
            grid_by_exp.setdefault((d_str, exp_str, r), []).append(k_f)
        for key in grid:
            grid[key] = sorted(set(grid[key]))
        for key in grid_by_exp:
            grid_by_exp[key] = sorted(set(grid_by_exp[key]))
        conn.close()
        _REAL_STRIKE_GRID = grid
        _REAL_STRIKE_GRID_BY_EXP = grid_by_exp
        return grid
    except Exception:
        _REAL_STRIKE_GRID = {}
        _REAL_STRIKE_GRID_BY_EXP = {}
        return {}


def snap_to_real_strike(K, date_str, right='P', expiration=None, spot=None):
    """Snap a computed strike to the nearest REAL listed strike.

    Lookup order:
    1. PG (date, expiration, right) — exact match if backfill covered this contract
    2. Heuristic for that expiration — monthly=integer, weekly=half-strike (UNG-specific)
    3. PG (date, right) — fallback to date-pooled strikes
    4. Original K — no data at all
    """
    _load_real_strike_grid()
    grid = _REAL_STRIKE_GRID or {}
    grid_exp = _REAL_STRIKE_GRID_BY_EXP or {}

    # (1) Exact (date, exp, right) match
    if expiration is not None:
        exp_str = expiration if isinstance(expiration, str) else expiration.isoformat()
        strikes = grid_exp.get((date_str, exp_str, right))
        if strikes:
            return min(strikes, key=lambda s: abs(s - K))

        # (2) Heuristic for this expiration (when spot known)
        if spot is not None and spot > 0:
            strikes = _heuristic_strike_grid(spot, exp_str)
            if strikes:
                return min(strikes, key=lambda s: abs(s - K))

    # (3) Fallback to date-pooled
    strikes = grid.get((date_str, right))
    if not strikes:
        all_dates = sorted({d for (d, r) in grid.keys() if r == right})
        if not all_dates:
            return K
        prior = [d for d in all_dates if d <= date_str]
        nearest = prior[-1] if prior else all_dates[0]
        strikes = grid.get((nearest, right), [])
        if not strikes:
            return K
    return min(strikes, key=lambda s: abs(s - K))


def _load_iv_surface():
    """Load full UNG IV surface from Postgres into a fast in-memory lookup.

    Returns dict keyed by date_str → list of (dte, strike_adj, right, iv).
    Returns None if PG unavailable; caller falls back to parametric model.
    """
    global _IV_SURFACE_CACHE
    if _IV_SURFACE_CACHE is not None:
        return _IV_SURFACE_CACHE
    try:
        import psycopg2
        conn = psycopg2.connect(
            host='192.168.1.172', port=5432, database='market_scanner',
            user='postgres', password='shinobi2025', connect_timeout=5,
        )
        cur = conn.cursor()
        cur.execute('SELECT date, dte, strike_adj, option_right, iv FROM ung_iv_surface')
        rows = cur.fetchall()
        conn.close()
        surface = {}
        for d, dte, K, right, iv in rows:
            d_str = d.isoformat()
            surface.setdefault(d_str, []).append((int(dte), float(K), right, float(iv)))
        _IV_SURFACE_CACHE = surface
        return surface
    except Exception:
        _IV_SURFACE_CACHE = {}
        return {}


_SURFACE_DATES_SORTED = None


def surface_latest_date(surface, on_or_before=None):
    """Most recent surface date (optionally ≤ on_or_before). Used to CARRY FORWARD the last
    real smile when today's date is missing (the options feed lags the price feed) instead of
    dropping to the crude realized-vol proxy. Cached sorted-date list for O(log n) lookup."""
    global _SURFACE_DATES_SORTED
    if not surface:
        return None
    if _SURFACE_DATES_SORTED is None:
        _SURFACE_DATES_SORTED = sorted(surface.keys())
    dates = _SURFACE_DATES_SORTED
    if not dates:
        return None
    if on_or_before is None:
        return dates[-1]
    import bisect
    i = bisect.bisect_right(dates, on_or_before)
    return dates[i - 1] if i > 0 else None


def iv_from_surface(surface, date_str, K, dte, right):
    """Nearest-neighbor IV lookup. Returns None if no data within tolerance."""
    rows = surface.get(date_str)
    if not rows:
        return None
    # Filter by right; find closest (K, dte) jointly
    best = None
    best_dist = float('inf')
    for r_dte, r_K, r_right, r_iv in rows:
        if r_right != right:
            continue
        dist = abs(r_K - K) / max(K, 1) + abs(r_dte - dte) * 0.001
        if dist < best_dist:
            best_dist = dist
            best = r_iv
    if best is None or best_dist > 0.4:
        return None
    return best


def iv_shape_features(surface, date_str, spot_adj):
    """Extract IV term-structure & skew features from PG surface.

    Returns dict with:
      atm_iv      — IV at ~ATM strike (proxy for IV30)
      put_skew    — IV(0.9 OTM put) - IV(ATM)  (positive = put skew rich)
      call_skew   — IV(1.1 OTM call) - IV(ATM)
      pc_skew     — put_skew - call_skew (smile asymmetry)
      term_slope  — IV at long DTE - IV at short DTE (positive = contango)
    Returns None if surface lacks data for this date.
    """
    rows = surface.get(date_str)
    if not rows or len(rows) < 4:
        return None
    # ATM = nearest strike to spot, prefer call
    atm = sorted(rows, key=lambda r: abs(r[1]-spot_adj))[:6]
    atm_iv = sum(r[3] for r in atm) / len(atm)
    # OTM put = strike ~0.9 × spot, right='P'
    Kp_target = spot_adj * 0.90
    put_candidates = [r for r in rows if r[2] == 'P']
    if put_candidates:
        put_otm = min(put_candidates, key=lambda r: abs(r[1] - Kp_target))
        put_skew = put_otm[3] - atm_iv
    else:
        put_skew = 0.0
    # OTM call = strike ~1.1 × spot
    Kc_target = spot_adj * 1.10
    call_candidates = [r for r in rows if r[2] == 'C']
    if call_candidates:
        call_otm = min(call_candidates, key=lambda r: abs(r[1] - Kc_target))
        call_skew = call_otm[3] - atm_iv
    else:
        call_skew = 0.0
    # Term slope: average IV at high-dte rows - average at low-dte rows
    if rows:
        dtes = [r[0] for r in rows]
        if max(dtes) - min(dtes) > 14:
            short_dte = [r[3] for r in rows if r[0] <= min(dtes) + 7]
            long_dte = [r[3] for r in rows if r[0] >= max(dtes) - 7]
            if short_dte and long_dte:
                term_slope = sum(long_dte)/len(long_dte) - sum(short_dte)/len(short_dte)
            else:
                term_slope = 0.0
        else:
            term_slope = 0.0
    else:
        term_slope = 0.0
    return {
        'atm_iv': atm_iv,
        'put_skew': put_skew,
        'call_skew': call_skew,
        'pc_skew': put_skew - call_skew,
        'term_slope': term_slope,
    }


_FILL_GRID = None


def fill_factor(right, dte_days, otm_pct):
    """Empirical fill ratio (real BID / BSM-est) from 8y of actual UNG
    quotes (backtest/cache/ung_fill_grid.csv). Applied at entry credits
    when p['real_fill_model']. Fallback 0.92 (median OTM haircut)."""
    global _FILL_GRID
    if _FILL_GRID is None:
        try:
            _g = pd.read_csv(os.path.join(CACHE_DIR, 'ung_fill_grid.csv'))
            _FILL_GRID = {(r['right'], int(r['dte_b']), round(float(r['otm_b']), 1)):
                          float(r['median']) for _, r in _g.iterrows()}
        except Exception:
            _FILL_GRID = {}
    if not _FILL_GRID:
        return 1.0
    dte_b = int(dte_days // 30) * 30
    otm_b = round(otm_pct * 10) / 10
    for k in ((right, dte_b, otm_b), (right, dte_b, 0.0), (right, 30, otm_b)):
        if k in _FILL_GRID:
            return _FILL_GRID[k]
    return 0.92


def _p_mid_fill(rel_spread, dte):
    """Probability you get a MID fill (vs having to cross to the touch), as a
    function of spread WIDTH and DTE. Tight spread → easy mid fill; wide spread →
    you usually must cross. More DTE → more time to work the limit → nearer mid.
    Neither extreme (always-mid = too optimistic, always-touch = too pessimistic)."""
    base = 1.0 - min(1.0, rel_spread / 0.20)          # ~0% wide→1.0, ≥20% wide→0.0
    dte_adj = 0.6 + 0.4 * min(1.0, (dte or 30) / 45.0)  # short DTE shaves patience
    return max(0.0, min(1.0, base * dte_adj))


_REAL_CHAIN = None
def real_chain_price(date, K, dte, right, spot, side='sell'):
    """REAL historical fill (tier-3, ThetaData bid/ask) priced at the EXPECTED fill
    between mid and the touch — modeled via P(mid fill | spread width, DTE):
      sell → mid − (1−P)·half_spread   (you receive between mid and bid)
      buy  → mid + (1−P)·half_spread   (you pay between mid and ask)
    Tight/longer-DTE ≈ mid; wide/short-DTE ≈ touch. Returns None off-grid (model fallback).
    Tunable via FILL_MID_OPTIMISM (1.0 = this model; >1 more mid, <1 more touch)."""
    global _REAL_CHAIN
    if _REAL_CHAIN is None:
        try:
            import real_chain as _rc
            _REAL_CHAIN = _rc
        except Exception:
            _REAL_CHAIN = False
    if not _REAL_CHAIN:
        return None
    try:
        b, a, m, real = _REAL_CHAIN.price(date, K, dte, right, spot_adj=spot)
        if not real:
            return None
        b, a = float(b), float(a)
        if a <= b:                       # locked/crossed/one-sided → just use it
            return a if side == 'buy' else b
        mid = (a + b) / 2.0
        half = (a - b) / 2.0
        rel = (a - b) / mid if mid > 0 else 1.0
        p_mid = min(1.0, _p_mid_fill(rel, dte) * FILL_MID_OPTIMISM)
        give = (1.0 - p_mid) * half      # expected slippage from mid toward the touch
        return mid + give if side == 'buy' else mid - give
    except Exception:
        return None


def _is_tom(idx):
    """NG futures expiry window: month-end ±2 calendar days."""
    return idx.day >= 27 or idx.day <= 2


def exec_fill(idx, K, dte, right, side, spot, p, model_price):
    """Unified, AUDITABLE fill. Returns (price, audit). We NEVER execute at EOD, so the
    EOD-real fill model is RETIRED — priority is intraday (real minute path + microstructure
    timing) → BS model (only as a gap fallback when a contract lacks minute data)."""
    eod_str = (idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx)[:10])
    if p.get('intraday_exec'):
        try:
            from intraday_fill import execute_audit
            a = execute_audit(idx, K, dte, right, side,
                              exec_window=p.get('exec_window', 15),
                              avoid_print=p.get('avoid_eia_print', True),
                              patience=p.get('exec_patience', 0.6))
            if a:
                # fold the verifiable fill mode into source: intraday:passive_mid
                # (market traded through your resting mid) vs intraday:crossed_touch
                # (you crossed at the real touch). Lets every trade record HOW it filled.
                if a.get('how'):
                    a['source'] = f"intraday:{a['how']}"
                return a['price'], a
        except Exception:
            pass
    # EOD-real fill RETIRED (we never execute EOD). Gap fallback = BS model only.
    return model_price, {'price': round(model_price, 4), 'exec_time': eod_str,
                         'bid': None, 'ask': None, 'spread_pct': None, 'source': 'model'}


def run_strategy_simple(df, strategy_params, initial_cash=48000, initial_shares=6200,
                        seed_state=None, live_decision=False):
    """Simpler procedural runner with state dict.

    SINGLE SOURCE OF TRUTH for live recs: with live_decision=True + seed_state
    (your real positions), the engine runs its NORMAL loop, then on the FINAL row
    overwrites state with your live portfolio and emits the engine's actual orders
    for today. Those captured trades ARE the recommendation — no re-implementation,
    so live == backtest by construction."""
    s = {
        'cash': initial_cash, 'shares': initial_shares, 'boxx': 0, 'kold': 0,
        'short_puts': [], 'short_calls': [], 'long_puts': [], 'long_calls': [],
        'upside_call_open': None,
    }
    history = []
    trades = []

    p = strategy_params
    use_surprise = p.get('use_surprise_z', False)
    # Real IV from PG ung_iv_surface is now the DEFAULT for all strategies.
    # Opt out only with use_real_iv_surface=False (e.g., proxy-only diagnostics).
    use_real_iv = p.get('use_real_iv_surface', True)
    iv_surface = _load_iv_surface() if use_real_iv else None
    target_weekly_income = p.get('target_weekly_income', 1500.0)
    recent_premium = []
    # Drawdown-aware risk dial — single parameter that down-scales sizing
    # when NAV is deep in drawdown. Generic risk control — protects against
    # BOTH sharp crashes (2021-12 → 2022-02) AND slow declines (2023-2026).
    nav_peak = float(initial_cash + initial_shares * (df['UNG'].iloc[0] if len(df) else 1))

    _last_i = len(df) - 1
    for i in range(len(df) if live_decision else len(df) - 30):
        idx = df.index[i]
        row = df.iloc[i]
        spot_u = row.get('UNG', 0)
        if spot_u <= 0:
            continue
        # TEST-ONLY (default off): snapshot the start-of-day book so a live-decision run can be
        # seeded with the continuous backtest's own state and proven to reproduce that day's
        # trades (live == backtest equivalence test). Zero impact when _CAPTURE_STATES is False.
        if _CAPTURE_STATES:
            import copy as _copy
            _STATE_SNAPSHOTS[str(pd.Timestamp(idx).date())] = {
                # book-only fields (what the live path can reconstruct from real positions)
                'cash': s['cash'], 'shares': s['shares'],
                'short_puts': [dict(x) for x in s['short_puts']],
                'short_calls': [dict(x) for x in s['short_calls']],
                'long_puts': [dict(x) for x in s.get('long_puts', [])],
                'long_calls': [dict(x) for x in s.get('long_calls', [])],
                'boxx': s.get('boxx', 0), 'kold': s.get('kold', 0),
                # FULL internal state + nav_peak (path-dependent; for the determinism test only)
                '_full_s': _copy.deepcopy(s), '_full_nav_peak': nav_peak,
            }
        # LIVE-DECISION SEED: on the final row, replace the engine's accumulated
        # paper portfolio with the operator's REAL positions, then let the normal
        # body emit today's orders. nav_peak reset so dd_scale doesn't see a false
        # drawdown from the paper run.
        if live_decision and i == _last_i and seed_state is not None and seed_state.get('_full_s') is not None:
            # DETERMINISM-TEST PATH: restore the COMPLETE engine state (incl path-dependent
            # trackers + nav_peak). Proves seeded single-day == continuous run, bit-for-bit.
            import copy as _copy
            s = _copy.deepcopy(seed_state['_full_s'])
            nav_peak = seed_state['_full_nav_peak']
            _live_trade_mark = len(trades)
        elif live_decision and i == _last_i and seed_state is not None:
            s['cash'] = float(seed_state.get('cash', s['cash']))
            s['shares'] = int(seed_state.get('shares', s['shares']))
            s['short_puts'] = list(seed_state.get('short_puts', []))
            s['short_calls'] = list(seed_state.get('short_calls', []))
            s['long_puts'] = list(seed_state.get('long_puts', []))
            s['long_calls'] = list(seed_state.get('long_calls', []))
            # BOXX + KOLD are part of NAV (cur_nav above) and drive NAV-based sizing (KOLD
            # hedge, share target, call_qty_nav_pct, margin). Omitting them understated NAV by
            # the BOXX value → live sizes diverged from the backtest. Restore them too.
            s['boxx'] = float(seed_state.get('boxx', s['boxx']))
            s['kold'] = int(seed_state.get('kold', s['kold']))
            _boxx_px = float(row.get('BOXX') if (row.get('BOXX') == row.get('BOXX')
                                                 and row.get('BOXX') is not None) else 117.0)
            _kold_px = float(row.get('KOLD') if (row.get('KOLD') == row.get('KOLD')
                                                 and row.get('KOLD') is not None) else 0.0)
            _cur_nav = s['cash'] + s['shares'] * spot_u + s['boxx'] * _boxx_px + s['kold'] * _kold_px
            # nav_peak = the operator's REAL high-water mark when supplied (so drawdown-scaling
            # and DD_TRIM reflect the operator's actual path, matching the backtest's defensive
            # behavior) — not a fresh reset that hides real drawdown. Floor at current NAV.
            nav_peak = max(float(seed_state.get('nav_peak') or 0.0), _cur_nav)
            _live_trade_mark = len(trades)   # capture trades emitted from here
        spot_k = row.get('KOLD', 0) or 0
        if isinstance(spot_k, float) and math.isnan(spot_k):
            spot_k = 0
        # Time-varying IV: per-strike via calibrated model (realized vol +
        # VIX regime + skew + term structure). Falls back to 0.55 only if
        # no realized vol available (first 30 days).
        d_str = idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx)[:10]
        def iv_at(K, dte, right='C'):
            if iv_surface is not None and iv_surface:
                real = iv_from_surface(iv_surface, d_str, K, dte, right)
                if real is not None:
                    return real
                # CARRY-FORWARD: the options feed (ung_iv_surface) lags the price feed, so on a
                # fresh day d_str has no entry. Use the most recent REAL smile (≤ d_str) before
                # dropping to the crude realized-vol proxy — real implied vol shape beats a
                # realized-vol guess that overshoots after volatile moves. (Root-caused 2026-06-18:
                # stale surface → proxy IV 0.55 vs real 0.44 → call mispriced $0.40 vs $0.30.)
                cf = surface_latest_date(iv_surface, d_str)
                if cf is not None and cf != d_str:
                    real = iv_from_surface(iv_surface, cf, K, dte, right)
                    if real is not None:
                        return real
            return iv_for_quote(row, K, spot_u, dte, right)
        # Per-day IV shape features (term + skew). None if no surface coverage.
        iv_shape = iv_shape_features(iv_surface, d_str, spot_u) if iv_surface else None
        z = compute_historical_z(row, use_surprise=use_surprise)
        # DD-aware risk dial — generic protection against any adverse
        # regime (sharp crash or slow decline). Track NAV peak; if
        # current NAV is significantly below peak, scale down all sizing.
        cur_nav = s['cash'] + s['shares'] * spot_u + s['boxx'] * float((row.get('BOXX') if (row.get('BOXX') == row.get('BOXX') and row.get('BOXX') is not None) else 117.0)) + s['kold'] * spot_k + long_calls_mtm(s.get('long_calls'), spot_u, idx, iv_at)
        if cur_nav > nav_peak:
            nav_peak = cur_nav
        dd_pct = (cur_nav - nav_peak) / nav_peak * 100 if nav_peak > 0 else 0
        # GEN-8 CONTROLLED TEST: when hedge_sizing_neutral, the share-sizing
        # drawdown signal EXCLUDES the KOLD hedge P&L, so the share book
        # follows the SAME path as an unhedged baseline. This isolates the
        # hedge overlay's effect from the share-count confound (the hedge
        # otherwise dampens DD → fewer trims → more shares = exposure
        # confound). Used only for the matched-share controlled comparison.
        if p.get('hedge_sizing_neutral'):
            _nav_exhedge = cur_nav - s['kold'] * spot_k
            _peak_exh = s.get('_peak_exhedge', _nav_exhedge)
            if _nav_exhedge > _peak_exh:
                _peak_exh = _nav_exhedge
            s['_peak_exhedge'] = _peak_exh
            dd_pct = (_nav_exhedge - _peak_exh) / _peak_exh * 100 if _peak_exh > 0 else 0

        # BACKWARDATION-SPIKE DE-RISK (validated 2026-06-16 timing_signals_eval:
        # HH spot−futures basis in its top decile (≥$0.18) / top-5% (≥$0.33)
        # precedes UNG −2.7% / −3.7% over the next 5 days, ~70% down — a sharp,
        # rare BEARISH event, NOT a continuous signal (continuous basis IC≈0).
        # The SHARE BOOK is what bleeds, so trim it ahead of the drop and let the
        # normal z-target re-accumulate after. Respects CC coverage + a floor;
        # cooldown prevents churn within one event window.
        if p.get('backwardation_derisk'):
            _basis = float(row.get('hh_basis') or 0.0)
            _last_bwd = s.get('_last_bwd_trim_i', -999)
            if (_basis >= p.get('bwd_derisk_thresh', 0.33)
                    and (i - _last_bwd) >= p.get('bwd_derisk_cooldown', 5)
                    and s['shares'] > 100):
                _cc_lots = sum(sc.get('qty', 0) for sc in s['short_calls'])
                _bfloor = max(p.get('bwd_derisk_floor', 1000), _cc_lots * 100)
                _btrim = int(s['shares'] * p.get('bwd_derisk_trim_pct', 25) / 100)
                _btrim = (_btrim // 100) * 100
                _btrim = min(_btrim, s['shares'] - _bfloor)
                if _btrim >= 100:
                    s['cash'] += _btrim * (spot_u - SPREAD_SHARE)
                    s['shares'] -= _btrim
                    s['_last_bwd_trim_i'] = i
                    trades.append({'date': idx, 'type': 'BACKWARDATION_DERISK_TRIM',
                                   'pnl': 0.0, 'qty': _btrim, 'spot': spot_u,
                                   'basis': round(_basis, 3), 'shares_after': s['shares']})

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
        # GEN-4 dd_trim_iv_gate (KERNEL_LAB #5: long DDs start at rich-vol
        # tops, the -23% fwd-63d zone): trim sooner when iv_rank is high
        if p.get('dd_trim_iv_gate') and dd_trim_trigger < 0:
            _ivr_dd = row.get('iv_rank') if 'row' in dir() else None
            try:
                _ivr_dd = df['iv_rank'].iloc[i] if 'iv_rank' in df.columns else None
            except Exception:
                _ivr_dd = None
            if _ivr_dd is not None and _ivr_dd == _ivr_dd and _ivr_dd > 0.8:
                dd_trim_trigger = dd_trim_trigger / 2  # e.g. -4 → -2: act sooner
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

                    # ASSIGNMENT-AWARE FILTER (statistical, not heuristic):
                    # Use BSM+extrinsic model on each leg. Calls with
                    # p_assign ≥ 0.55 are "likely+" — defer to assignment.
                    pending_assign_lots = 0
                    leg_p_assign = {}  # idx → p_assign
                    _surge_z = float(row.get('ung_surge_z') or 0.0)
                    for ci, sc in enumerate(s['short_calls']):
                        days_left = max(1, sc['dte'] - (idx - sc['entry']).days)
                        leg_iv = iv_at(sc['K'], days_left, 'C')
                        leg_prem = bs_call(spot_u, sc['K'], days_left/365, leg_iv)
                        a = assignment_probability(K=sc['K'], spot=spot_u, dte=days_left,
                                                    iv=leg_iv, right='CALL',
                                                    premium_market=leg_prem,
                                                    mean_reversion_z=_surge_z)
                        leg_p_assign[ci] = a['p_assign']
                        if a['p_assign'] >= 0.55:
                            pending_assign_lots += sc['qty']

                    deferred_lots = min(pending_assign_lots, needed_lots)
                    needed_lots -= deferred_lots
                    if deferred_lots > 0:
                        trades.append({'date': idx, 'type': 'CC_DEFER_TO_ASSIGN',
                                       'pnl': 0.0, 'qty': deferred_lots,
                                       'spot': spot_u, 'dd_pct': dd_pct,
                                       'note': 'statistical p_assign≥0.55 → wait for natural assignment'})

                    # For RESIDUAL lots, rank by EV(wait) - EV(btc) and BTC
                    # only legs where btc is genuinely cheaper.
                    if needed_lots > 0:
                        candidates = []
                        for ci, sc in enumerate(s['short_calls']):
                            if leg_p_assign.get(ci, 0) >= 0.55:
                                continue  # already deferred
                            days_left = max(1, sc['dte'] - (idx - sc['entry']).days)
                            leg_iv = iv_at(sc['K'], days_left, 'C')
                            leg_prem = bs_call(spot_u, sc['K'], days_left/365, leg_iv)
                            ev = expected_value_wait_vs_btc(
                                K=sc['K'], spot=spot_u, dte=days_left, iv=leg_iv,
                                right='CALL', entry_prem=sc['entry_prem'],
                                premium_market=leg_prem, contracts=sc['qty'])
                            # Prefer BTC where ev_btc > ev_wait (i.e. ev_diff < 0)
                            candidates.append((ev['ev_diff_wait_minus_btc'], ci, sc, leg_prem))
                        # Sort ascending diff → BTC the leg where wait is WORST
                        candidates.sort(key=lambda x: x[0])
                        for _, ci, sc, cur_prem in candidates:
                            if needed_lots <= 0:
                                break
                            close_lots = min(sc['qty'], needed_lots)
                            debit = cur_prem * 100 * close_lots + close_lots * SPREAD_OPTION * 100
                            if s['cash'] < debit + 500:
                                continue
                            s['cash'] -= debit
                            pnl = sc['entry_prem'] * 100 * close_lots - debit
                            trades.append({'date': idx, 'type': 'CC_CLOSE_FOR_CUT',
                                           'pnl': pnl, 'qty': close_lots, 'K': sc['K'],
                                           'spot': spot_u, 'dd_pct': dd_pct})
                            sc['qty'] -= close_lots
                            needed_lots -= close_lots
                        s['short_calls'] = [c for c in s['short_calls'] if c['qty'] > 0]
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
        # EVENT-DRIVEN RE-ACCUM: after a large called-away strips shares well below target, re-evaluate
        # the share-target off-cadence (for a short window) so the book glides back to the desired delta
        # instead of waiting up to a full cadence period. Armed in the CALL_ASSIGN handler below.
        _reaccum_now = bool(p.get('reaccum_on_called_away')) and i <= s.get('_reaccum_until', -1)
        if z_target_enabled and (i % p.get('z_target_cadence_days', 5) == 0 or _reaccum_now):
            # SCALE-INVARIANT base: if z_share_target_pct_nav set, compute base
            # shares as (NAV * pct / spot) so base scales with account size.
            # Falls back to hardcoded z_share_target_base for legacy strategies.
            pct_base = p.get('z_share_target_pct_nav')
            if pct_base is not None and pct_base > 0 and spot_u > 0 and cur_nav == cur_nav:
                # NaN guard: cur_nav can be NaN if intermediate state went bad
                base_shares = int((cur_nav * pct_base / spot_u) / 100) * 100
                base_shares = max(100, base_shares)
            else:
                base_shares = p.get('z_share_target_base', 6200)
            # Tunable multipliers — defaults are the wheel philosophy curve
            mults = p.get('z_target_mults', {
                'extreme_cheap': 1.4, 'cheap': 1.2, 'neutral': 1.0,
                'rich': 0.7, 'extreme_rich': 0.3,
            })
            # SMOOTH variant: continuous mult via tanh(z) interpolation between
            # extreme_cheap and extreme_rich endpoints. Eliminates the bucket
            # discontinuities (e.g. z=0.49 → 1.0× but z=0.51 → 0.4×) and
            # produces much smoother NAV path (lower daily volatility).
            if p.get('smooth_z_target'):
                lo = mults.get('extreme_rich', 0.1)
                hi = mults.get('extreme_cheap', 2.0)
                # tanh(z) maps [-3, +3] roughly to [-0.99, 0.99]; convert to [lo,hi]
                t = math.tanh(z * -0.5)  # negative because cheap = high mult
                mult = lo + (hi - lo) * (t + 1) / 2
            elif z < -1.5:    mult = mults['extreme_cheap']
            elif z < -0.5:  mult = mults['cheap']
            elif z < 0.5:   mult = mults['neutral']
            elif z < 1.0:   mult = mults['rich']
            else:           mult = mults['extreme_rich']
            # IV-RANK SCALE ([[project_ung_iv_rank_alpha]]: top-quintile real
            # ATM IV → -23% fwd-63d, p=.002). Rich-vol regime trims the
            # share target; cheap-vol regime boosts accumulation. Same
            # philosophy as the z mult curve, driven by the option market.
            if p.get('iv_rank_z_scale'):
                _ivr = row.get('iv_rank')
                if _ivr == _ivr and _ivr is not None:
                    if _ivr > 0.8:
                        mult *= 0.5
                    elif _ivr > 0.6:
                        mult *= 0.8
                    elif _ivr < 0.2:
                        mult *= 1.3
            # GEN-9 CONVICTION AMPLIFIER (return lever): when BOTH signals
            # scream — extreme-cheap z AND bottom-quintile IV-rank (the
            # validated +23% fwd-63d zone) — size the share book UP harder.
            # Asymmetric: amplifies only at max conviction, never trims.
            # The return-seeking mirror of the drawdown-cutting delta band.
            if p.get('conviction_amplify'):
                _ivr2 = row.get('iv_rank')
                if (z < -1.0 and _ivr2 == _ivr2 and _ivr2 is not None
                        and _ivr2 < 0.2 and not falling_knife(row)):
                    mult *= p.get('conviction_amplify_mult', 1.4)
            # DD-aware override: if in deep DD, cap the multiplier
            dd_cap_15 = p.get('z_target_dd_cap_15', 0.6)
            dd_cap_10 = p.get('z_target_dd_cap_10', 0.8)
            if dd_pct < -15:
                mult = min(mult, dd_cap_15)
            elif dd_pct < -10:
                mult = min(mult, dd_cap_10)
            # SEASONAL REGIME (user design): NG bottoms in mild SHOULDER season
            # (injection) and firms into PEAK-demand season. ACCUMULATE in shoulder —
            # but ONLY when not a falling knife (never build into a confirmed decline);
            # DUMP into the peak ('soon enough'). Tilts the valuation share-target curve.
            if p.get('seasonal_regime'):
                _m = idx.month
                if _m in p.get('accumulate_months', (3, 4, 5, 9)):
                    mult *= (p.get('accumulate_boost', 1.4) if not falling_knife(row)
                             else p.get('knife_accum_cut', 0.7))
                elif _m in p.get('distribute_months', (11, 12, 1, 6, 7)):
                    mult *= p.get('distribute_cut', 0.45)        # dump into the peak season
            # STATE REGIME (user design, drift-aware): classify ACCUMULATE / NEUTRAL /
            # DISTRIBUTE from WHAT ACTUALLY HAPPENED — storage_surprise_z (storage vs its
            # SEASONAL expectation, so it self-adjusts to each year's drift) + momentum.
            # Tight supply (bullish surprise) & not crashing → accumulate; oversupply
            # (bearish surprise) or momentum breakdown → DUMP soon. No fixed calendar.
            if p.get('state_regime'):
                _ssz = row.get('storage_surprise_z')
                _surge = row.get('ung_surge_z')
                _surge = _surge if (_surge == _surge and _surge is not None) else 0.0
                # NOAA DEGREE-DAY NOWCAST: the DD anomaly leads the storage print ~1wk
                # (high demand → tighter storage). dd_w<0 = nowcast (accumulate earlier on a
                # tightening); dd_w>0 = contrarian (fade demand spikes). Walk-forward decides.
                _ddw = p.get('dd_w', 0.0)
                if _ddw and _ssz == _ssz and _ssz is not None:
                    _ddz = row.get('dd_anom_z')
                    if _ddz == _ddz and _ddz is not None:
                        _ssz = _ssz + _ddw * _ddz
                # FIX 1 — confidence gate: don't ACCUMULATE when the storage signal is
                # noisy (ssz_vol high → regime unreliable; walk-forward: corr -0.83).
                _noisy = False
                if p.get('regime_confidence_gate'):
                    _sv = row.get('ssz_vol')
                    _noisy = (_sv == _sv and _sv is not None and _sv > p.get('ssz_vol_gate', 1.2))
                # FIX 2 — price-breakdown distribute trigger: a confirmed DOWNTREND forces
                # distribute even if storage is neutral (catches the 2022-style price crash
                # the fundamental signal misses → the -56pp walk-forward window).
                _breakdown = bool(p.get('regime_downtrend_distribute') and row.get('ung_downtrend'))
                # CRASH FALLBACK: a deep PRICE crash (drawdown from 60d high) forces DISTRIBUTE
                # regardless of the fundamental — the 2022 blind spot where storage was neutral.
                _dd60 = row.get('ung_dd_60')
                _crash = bool(p.get('crash_fallback') and _dd60 == _dd60 and _dd60 is not None
                              and _dd60 < p.get('crash_dd', -0.18))
                if _ssz == _ssz and _ssz is not None:
                    if (_ssz < p.get('accumulate_ssz', -0.5) and not falling_knife(row)
                            and _surge > -1.0 and not _noisy and not _breakdown and not _crash):
                        mult *= p.get('accumulate_boost', 1.5)        # tight + stable → build
                    elif (_ssz > p.get('distribute_ssz', 0.5)
                          or _surge < p.get('distribute_surge', -1.2) or _breakdown or _crash):
                        mult *= p.get('distribute_cut', 0.4) * (p.get('crash_distribute_extra', 0.6)
                                                                if _crash else 1.0)  # dump harder in a crash
            # CONTINUOUS REGIME STRENGTH (quantified): scale the tilt by |s| — strong
            # regime → full tilt, WEAK/uncertain → ~neutral (don't churn on noise).
            if p.get('regime_continuous'):
                rs = row.get('regime_strength')
                if rs == rs and rs is not None:
                    if rs > 0:                                        # distribute strength
                        mult *= 1.0 - min(1.0, rs) * p.get('distribute_strength_max', 0.6)
                    elif not falling_knife(row):                      # accumulate strength (gated)
                        mult *= 1.0 + min(1.0, -rs) * p.get('accumulate_strength_max', 0.5)
            target = int(base_shares * mult)
            target = (target // 100) * 100
            current = s['shares']
            if p.get('stat_share_target'):
                # GAMMA→DELTA on the ACCUMULATION engine: measure `current` as the statistical FORWARD
                # share count — deep-ITM calls (delta→1) count as forward-GONE, short puts as forward-
                # ACQUIRED (p_assign-weighted). So the wheel re-accumulates a few days EARLY as calls go
                # deep-ITM (capturing the theta the same-bar backtest banks, incl. weekends), and SELF-
                # LIMITS once the replacement puts are pending (no double-sell). Same fn backtest & live.
                current = int(round(book_greeks_stat(
                    s, spot_u, z, p.get('scenario_mu_a', -0.000797),
                    p.get('scenario_mu_b', -0.000009), p.get('scenario_sigma', 0.0390))))
            # LONG-CALL ACCUMULATION arm: count long-call delta toward the share target so the calls
            # self-limit the gap (otherwise the engine re-buys calls every bar, never "seeing" the
            # delta it already added). Only when the arm is on, to leave baseline behaviour untouched.
            if p.get('reaccum_via_calls') and s.get('long_calls'):
                for _lc in s['long_calls']:
                    _Tlc = max(1, _lc['dte'] - (idx - _lc['entry']).days) / 365.0
                    _d_lc = bs_greeks_pt(spot_u, _lc['K'], _Tlc, iv_at(_lc['K'], int(_Tlc * 365), 'C'), 'C')[0]
                    current += int(_d_lc * _lc['qty'] * 100)
            # PUT-ACCUMULATION arm: target NET delta. Count the FULL book's option delta toward `current`
            # (short puts add +|delta|, short calls subtract) so the arm self-limits AND accounts for the
            # operator's ALREADY-OPEN positions in the live seed — not just this engine's own sells.
            # Matches the delta_compass framing (net_delta vs target) the operator steers by; without it
            # the live engine ignores the puts you've already sold and over-accumulates.
            # Count ALL short-put delta toward `current` (the synthetic long from puts) so the arm
            # self-limits AND sees the operator's already-open puts. Do NOT subtract the covered calls:
            # they're sold against the shares (an income overlay), they don't reduce share OWNERSHIP, so
            # the accumulation target shouldn't sell extra puts to offset them (that over-piles short vol
            # — it dropped the arm to 13.9% vs 16.7%). Net book delta naturally stays under the ownership
            # target by the covered-call drag, which is intentional.
            if p.get('reaccum_via_puts') or p.get('reaccum_delta_gamma'):
                for _sp in s.get('short_puts', []):
                    _Tsp = max(1, _sp['dte'] - (idx - _sp['entry']).days) / 365.0
                    current += int(-bs_greeks_pt(spot_u, _sp['K'], _Tsp, iv_at(_sp['K'], int(_Tsp * 365), 'P'), 'P')[0] * _sp['qty'] * 100)
                # delta+gamma mode targets TRUE net delta → also subtract the short calls' (negative) delta.
                if p.get('reaccum_delta_gamma'):
                    for _scl in s.get('short_calls', []):
                        _Tsc = max(1, _scl['dte'] - (idx - _scl['entry']).days) / 365.0
                        current -= int(bs_greeks_pt(spot_u, _scl['K'], _Tsc, iv_at(_scl['K'], int(_Tsc * 365), 'C'), 'C')[0] * _scl['qty'] * 100)
            # ── DISTRIBUTIONAL DELTA BAND (gen-5: the rigidity fix) ──────
            # The point target is μ. σ comes from SIGNAL DISAGREEMENT: when
            # z, iv_rank, and momentum point the same way → tight band (act
            # decisively); when they conflict → wide band (the future is
            # genuinely uncertain, so DON'T churn the book chasing a noisy
            # point estimate). Only act when |current-μ| > k·σ, then trade
            # toward μ, not onto it. Hysteresis kills whipsaw.
            band_skip = False
            if p.get('delta_band_sizing'):
                # three directional votes in [-1,+1] (bullish=+, want more shares)
                _vz = max(-1.0, min(1.0, -z / 1.5))          # cheap z → bullish
                _ivr_b = row.get('iv_rank')
                _viv = 0.0
                if _ivr_b == _ivr_b and _ivr_b is not None:
                    _viv = max(-1.0, min(1.0, (0.5 - _ivr_b) * 2))  # low IV → bullish
                _vmom = 0.0
                if i > 20:
                    try:
                        _m20 = df['UNG'].iloc[max(0, i-19):i+1].mean()
                        _vmom = max(-1.0, min(1.0, (spot_u / _m20 - 1) * 10))
                    except Exception:
                        pass
                _votes = [_vz, _viv, _vmom]
                _disagree = float(pd.Series(_votes).std())   # 0=aligned, ~1=conflict
                _k = p.get('delta_band_k', 1.0)
                if p.get('conviction_band'):
                    # CONVICTION-SCALED width (gen-6): narrow when the
                    # consensus is EXTREME (high edge → act, load up), wide
                    # when NEUTRAL (no edge → hold base, don't churn) OR
                    # CONFLICTED. |consensus| = strength of aligned vote.
                    # Recovers return at extremes (where it's made) while
                    # damping the wasteful neutral churn.
                    _consensus = abs(float(pd.Series(_votes).mean()))  # 0=neutral,1=extreme
                    _a = p.get('conviction_a', 0.30)   # neutral-widening weight
                    _b = p.get('conviction_b', 0.20)   # disagreement weight
                    _floor = p.get('conviction_floor', 0.05)
                    _width = _floor + _a * (1 - _consensus) + _b * _disagree
                    _sigma_sh = base_shares * _width
                else:
                    # gen-5 disagreement-only width
                    _sigma_sh = base_shares * (0.10 + 0.35 * _disagree)
                if abs(current - target) <= _k * _sigma_sh:
                    band_skip = True                          # inside band → hold
                    if not s.get('_in_band'):
                        trades.append({'date': idx, 'type': 'DELTA_BAND_HOLD',
                                       'pnl': 0.0, 'target': target, 'current': current,
                                       'sigma_sh': int(_sigma_sh),
                                       'disagree': round(_disagree, 2)})
                    s['_in_band'] = True
                else:
                    s['_in_band'] = False
                    # trade toward μ, not onto it: move halfway across the band edge
                    _edge = target + (_k * _sigma_sh if current > target else -_k * _sigma_sh)
                    target = int(((current + _edge) / 2) // 100) * 100
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
            delta = 0 if band_skip else (target - current)
            # GAP-DRIVEN WHEEL (user design): store the share gap so the put/call
            # sections can size + strike-pick to STEER the book via assignment
            # instead of churning shares directly. Positive = need shares (acquire
            # via puts), negative = excess (divest via calls).
            s['_share_gap_lots'] = int(round((target - current) / 100))
            # MICROSTRUCTURE TIMING for ADDS only (trims fire anytime —
            # risk control never waits): Tuesday adds (open bleeds -40bps
            # overnight → cheaper entries) and skip the NG-expiry
            # turn-of-month window (-43bps/day churn).
            if delta > 0:
                if p.get('avoid_tom_adds') and _is_tom(idx):
                    delta = 0
                elif (p.get('share_add_dow') is not None
                        and idx.dayofweek != p['share_add_dow']):
                    delta = 0
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
            if adjust < 0 and not p.get('gap_to_wheel'):  # selling (direct; OFF when gap_to_wheel → divest via CCs)
                # clamp to the REAL share balance — `current` may be a forward count (stat_share_target)
                # that exceeds s['shares']; selling against it could drive shares negative (naked short).
                max_sell = min(current, s['shares']) - min_shares_required
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
            elif adjust >= 100 and not p.get('gap_to_wheel'):  # buying (direct; OFF when gap_to_wheel → acquire via puts)
                _ivr_now = row.get('iv_rank')
                _use_calls = (p.get('reaccum_via_calls')
                              and _ivr_now == _ivr_now and _ivr_now is not None
                              and _ivr_now < p.get('reaccum_calls_iv_max', 0.30))
                if _use_calls:
                    # LOW-IV ACCUMULATION VIA LONG CALLS: add the target delta with a fraction of the
                    # capital (keep BOXX rather than liquidate it for shares) AND go long cheap vol.
                    # Buy ~ATM calls for the delta-equivalent of `adjust` shares.
                    _dte_c = int(p.get('reaccum_call_dte', 90))
                    _Kc = round(spot_u * (1 + p.get('reaccum_call_moneyness', 0.0)))
                    _ivc = iv_at(_Kc, _dte_c, 'C')
                    _cdelta = bs_greeks_pt(spot_u, _Kc, _dte_c / 365, _ivc, 'C')[0]
                    _cost_c = bs_call(spot_u, _Kc, _dte_c / 365, _ivc)
                    if _cdelta > 0.05 and _cost_c > 0.01:
                        _qty_c = int(round(adjust / (_cdelta * 100)))   # delta-equivalent contracts
                        _afford_c = int((s['cash'] - 2000) / (_cost_c * 100 + SPREAD_OPTION * 100)) if s['cash'] > 2000 else 0
                        _qty_c = min(_qty_c, _afford_c)
                        if _qty_c >= 1:
                            _debit = _qty_c * _cost_c * 100 + _qty_c * SPREAD_OPTION * 100
                            s['cash'] -= _debit
                            s.setdefault('long_calls', []).append({'entry': idx, 'K': _Kc, 'dte': _dte_c,
                                                                   'qty': _qty_c, 'cost': _cost_c})
                            trades.append({'date': idx, 'type': 'Z_TARGET_ADD_CALLS', 'pnl': -_debit,
                                           'K': _Kc, 'qty': _qty_c, 'spot': spot_u, 'z': z,
                                           'cdelta': round(_cdelta, 2)})
                elif p.get('reaccum_via_puts'):
                    # ACCUMULATE VIA SELLING ATM PUTS: add delta (~+0.5/put) + collect premium, and
                    # acquire shares at strike IF assigned. Short vol — pays thin in low IV and the
                    # accumulation is contingent on the put going ITM.
                    _Kp = round(spot_u * (1 + p.get('reaccum_put_moneyness', 0.0)))   # 0.0 = ATM
                    _dte_p = int(p.get('reaccum_put_dte', 45))
                    _ivp = iv_at(_Kp, _dte_p, 'P')
                    _pdelta = -bs_greeks_pt(spot_u, _Kp, _dte_p / 365, _ivp, 'P')[0]   # short-put delta
                    _prem = bs_put(spot_u, _Kp, _dte_p / 365, _ivp)
                    if p.get('real_fill_model'):
                        _prem *= fill_factor('P', _dte_p, 1 - _Kp / spot_u)
                    if _pdelta > 0.05 and _prem > 0.02:
                        _qty_p = max(0, min(int(round(adjust / (_pdelta * 100))), 60))
                        if _qty_p >= 1:
                            _credit = _qty_p * _prem * 100 - _qty_p * SPREAD_OPTION * 100
                            s['cash'] += _credit
                            s['short_puts'].append({'entry': idx, 'K': _Kp, 'dte': _dte_p, 'qty': _qty_p,
                                                    'entry_prem': _prem, 'src': 'reaccum'})
                            trades.append({'date': idx, 'type': 'Z_TARGET_ADD_PUTS', 'pnl': 0.0,
                                           'credit': _credit, 'K': _Kp, 'qty': _qty_p, 'spot': spot_u, 'z': z})
                elif p.get('reaccum_delta_gamma'):
                    # DELTA+GAMMA TARGETED: close the delta gap `adjust` while steering book gamma toward a
                    # target (target_gamma_per_nav × NAV/spot, scale-invariant). Short puts carry the
                    # negative-gamma BUDGET (+premium); shares carry the rest (delta 1, gamma 0). With the
                    # book already past the gamma target it's ALL SHARES — which is the whole point: when
                    # the put book is over-gamma'd (like now, −1,553), the gamma target dilutes it with flat
                    # share delta instead of piling on more puts. target_gamma=0 ⇒ pure shares.
                    _Kp = round(spot_u * (1 + p.get('reaccum_put_moneyness', 0.05)))
                    _dte_p = int(p.get('reaccum_put_dte', 30))
                    _ivp = iv_at(_Kp, _dte_p, 'P')
                    _pd, _pg = bs_greeks_pt(spot_u, _Kp, _dte_p / 365, _ivp, 'P')
                    _put_d = -_pd * 100              # short-put delta / contract (+)
                    _put_g = -_pg * 100              # short-put gamma / contract (−)
                    _cg = 0.0
                    for _x in s.get('short_puts', []):
                        _T = max(1, _x['dte'] - (idx - _x['entry']).days) / 365.0
                        _cg -= bs_greeks_pt(spot_u, _x['K'], _T, iv_at(_x['K'], int(_T * 365), 'P'), 'P')[1] * _x['qty'] * 100
                    for _x in s.get('short_calls', []):
                        _T = max(1, _x['dte'] - (idx - _x['entry']).days) / 365.0
                        _cg -= bs_greeks_pt(spot_u, _x['K'], _T, iv_at(_x['K'], int(_T * 365), 'C'), 'C')[1] * _x['qty'] * 100
                    _tgt_g = p.get('target_gamma_per_nav', 0.0) * cur_nav / spot_u
                    _n_puts = 0
                    if _put_g < 0 and _cg > _tgt_g:                       # room for more negative gamma
                        _n_puts = int((_tgt_g - _cg) / _put_g)
                    if _put_d > 0:
                        _n_puts = max(0, min(_n_puts, int(adjust / _put_d)))
                    else:
                        _n_puts = 0
                    if _n_puts >= 1:
                        _prem = bs_put(spot_u, _Kp, _dte_p / 365, _ivp)
                        # PERFECT THE FILL: same empirical fill_factor (8yr real UNG NBBO) the engine's
                        # normal puts use — not a fixed haircut — so the ITM credit is modelled exactly
                        # like every other short put in the book. Keep SPREAD_OPTION as a small extra buffer.
                        if p.get('real_fill_model'):
                            _prem *= fill_factor('P', _dte_p, 1 - _Kp / spot_u)
                        if _prem > 0.02:
                            _credit = _n_puts * _prem * 100 - _n_puts * SPREAD_OPTION * 100
                            s['cash'] += _credit
                            s['short_puts'].append({'entry': idx, 'K': _Kp, 'dte': _dte_p, 'qty': _n_puts,
                                                    'entry_prem': _prem, 'src': 'reaccum'})
                            trades.append({'date': idx, 'type': 'Z_TARGET_ADD_PUTS', 'pnl': 0.0, 'credit': _credit,
                                           'K': _Kp, 'qty': _n_puts, 'spot': spot_u, 'z': z})
                        adjust -= int(_n_puts * _put_d)
                    _n_sh = (max(0, int(adjust)) // 100) * 100
                    _maff = int((s['cash'] - 5000) / (spot_u + SPREAD_SHARE)) if s['cash'] > 5000 else 0
                    _n_sh = min(_n_sh, (_maff // 100) * 100)
                    if _n_sh >= 100:
                        s['cash'] -= _n_sh * (spot_u + SPREAD_SHARE)
                        s['shares'] += _n_sh
                        trades.append({'date': idx, 'type': 'Z_TARGET_ADD', 'pnl': 0.0, 'qty': _n_sh,
                                       'spot': spot_u, 'z': z, 'target': target, 'shares_after': s['shares']})
                else:
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
        # REGIME-SCALED KOLD HEDGE (OOS missing-mechanism fix) — size the inverse-ETF
        # hedge by the DISTRIBUTE regime STRENGTH (regime_strength > 0). Strong distribute
        # → hold more KOLD to MONETIZE the decline the regime is calling; accumulate/weak
        # → none. Covered-calls-only can't short, so this is the bearish-capture vehicle.
        reg_kold_max = p.get('regime_kold_max_pct', 0)
        _rs = row.get('regime_strength')
        if (reg_kold_max > 0 and _rs == _rs and _rs is not None and spot_k > 0
                and pd.notna(spot_k)):
            target_kold_dollars = cur_nav * reg_kold_max * max(0.0, min(1.0, _rs))
            current_kold_dollars = s['kold'] * spot_k
            delta_dollars = target_kold_dollars - current_kold_dollars
            if abs(delta_dollars) > max(500, target_kold_dollars * 0.10 + 1):
                if delta_dollars > 0:
                    buy_q = int(delta_dollars / spot_k)
                    if buy_q >= 5 and s['cash'] > buy_q * spot_k + 200:
                        s['kold'] += buy_q
                        s['cash'] -= buy_q * spot_k + buy_q * SPREAD_SHARE
                        trades.append({'date': idx, 'type': 'KOLD_REGIME_BUY', 'pnl': 0.0,
                                       'qty': buy_q, 'spot': spot_k, 'rs': round(_rs, 2)})
                else:
                    sell_q = min(s['kold'], int(-delta_dollars / spot_k))
                    if sell_q >= 5:
                        s['cash'] += sell_q * spot_k - sell_q * SPREAD_SHARE
                        s['kold'] -= sell_q
                        trades.append({'date': idx, 'type': 'KOLD_REGIME_TRIM', 'pnl': 0.0,
                                       'qty': sell_q, 'spot': spot_k, 'rs': round(_rs, 2)})
        # ── DELTA-BAND HEDGE (greeks-based bookwise risk management) ──────────────────
        # When the regime turns bearish, buy LONG PUTS to pull the book's NET DELTA down
        # toward a regime-scaled target — forming effective BEAR PUT SPREADS with the
        # existing short puts (we do NOT close the premium-collecting shorts). Sizing and
        # DTE are guided by the book greeks (book_greeks/bs_greeks_pt → calibrated IV).
        if p.get('delta_hedge'):
            _nd, _ng = book_greeks(s, spot_u, iv_at)
            if p.get('stat_delta_hedge'):
                # GAMMA→DELTA: hedge the statistical FORWARD delta (real drift), not BS instantaneous.
                _z_sd = compute_historical_z(row, use_surprise=p.get('use_surprise', False))
                _nd = book_greeks_stat(s, spot_u, _z_sd, p.get('scenario_mu_a', -0.000797),
                                       p.get('scenario_mu_b', -0.000009), p.get('scenario_sigma', 0.0390))
            _rsd = row.get('regime_strength'); _rsd = _rsd if (_rsd == _rsd and _rsd is not None) else 0.0
            _base = p.get('delta_target_nav', 0.5) * cur_nav / max(spot_u, 0.5)   # ~normal delta
            _target = _base * (1.0 - p.get('delta_bearish_cut', 0.9) * max(0.0, min(1.0, _rsd)))
            _band = 0.15 * abs(_base) + p.get('delta_band_abs', 0.0)
            if _nd > _target + _band and _rsd > p.get('delta_hedge_rs_min', 0.25):
                _hd = p.get('delta_hedge_dte', 30)
                _hK = round(spot_u * (1 + p.get('delta_hedge_otm', 0.0)))   # ~ATM long put
                _pd, _ = bs_greeks_pt(spot_u, _hK, _hd / 365.0, iv_at(_hK, _hd, 'P'), 'P')
                _per = abs(_pd) * 100
                if _per > 1:
                    _n = max(0, min(int((_nd - _target) / _per), p.get('delta_hedge_max', 15)))
                    _cost = bs_put(spot_u, _hK, _hd / 365.0, iv_at(_hK, _hd, 'P'))
                    _debit = _cost * 100 * _n + _n * SPREAD_OPTION * 100
                    if _n >= 1 and s['cash'] > _debit + 1000:
                        s['cash'] -= _debit
                        s['long_puts'].append({'entry': idx, 'K': _hK, 'dte': _hd,
                                               'qty': _n, 'cost': _cost, 'expiry': None})
                        trades.append({'date': idx, 'type': 'DELTA_HEDGE_LONG_PUT',
                                       'pnl': -_debit, 'K': _hK, 'qty': _n, 'dte': _hd,
                                       'net_delta': round(_nd, 0), 'target': round(_target, 0)})
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
            if p.get('tp_50') and not p.get('ablate_put_tp'):
                tp_thresh = p.get('tp_threshold', 0.5)
                if p.get('tp_dynamic'):
                    rv30 = float(row.get('rv_30') or 0.5)
                    if rv30 > 0.80:   tp_thresh = 0.7
                    elif rv30 < 0.40: tp_thresh = 0.3
                    else:             tp_thresh = 0.5
                # GRIND-AWARE TP: during slow chronic declines, exit puts
                # earlier (lower tp_thresh = exit at higher premium-capture %).
                if p.get('grind_tp_accelerate') and detect_grind_down(row):
                    tp_thresh = min(tp_thresh, 0.3)
                # GEN-4 tp_by_iv_rank (KERNEL_LAB #6: put TPs avg only $167):
                # rich vol → capture fast before reversion; cheap calm vol →
                # let decay run to 70% capture
                if p.get('tp_by_iv_rank'):
                    _ivr = row.get('iv_rank')
                    if _ivr == _ivr and _ivr is not None:
                        if _ivr > 0.6:
                            tp_thresh = 0.5
                        elif _ivr < 0.4:
                            tp_thresh = 0.7
            if tp_thresh is not None and T_left > 1/365:
                _cv_model = bs_put(spot_u, sp['K'], T_left, iv_at(sp['K'], int(T_left*365), 'P'))
                cv, _aud = exec_fill(idx, sp['K'], int(T_left*365), 'P', 'buy',
                                     spot_u, p, _cv_model)
                if cv < sp['entry_prem'] * tp_thresh:
                    pnl = (sp['entry_prem'] - cv) * 100 * sp['qty'] - sp['qty'] * SPREAD_OPTION * 100
                    s['cash'] += pnl
                    trades.append({'date': idx, 'type': 'PUT_TP', 'pnl': pnl,
                                   'K': sp['K'], 'qty': sp['qty'],
                                   'dte': int(round(T_left * 365)),
                                   'expiry': sp.get('expiry'), 'buyback': round(cv, 2),
                                   'exec_time': _aud['exec_time'], 'bid': _aud['bid'],
                                   'ask': _aud['ask'], 'spread_pct': _aud['spread_pct'],
                                   'fill_source': _aud['source']})
                    continue

            # Roll down — only if remaining DTE is above min_roll_dte
            # threshold. Per [[feedback_dte_diversification]] (refined cycle
            # 20260531_140253): "let near-DTE OTM expire vs roll". Short-DTE
            # puts have little extrinsic to capture by rolling.
            # ALSO trend-aware: in uptrend, let ITM puts ride (price may
            # recover); in downtrend, rolling is protective.
            min_roll_dte = p.get('min_roll_dte', 5)  # default = old behavior
            dte_left = T_left * 365
            # CRASH FALLBACK: in a deep price crash, re-enable roll-down protection even if
            # the kernel lets puts assign normally (don't take assignment into a falling knife).
            _dd60r = row.get('ung_dd_60')
            _crash_roll = bool(p.get('crash_fallback') and _dd60r == _dd60r and _dd60r is not None
                               and _dd60r < p.get('crash_dd', -0.18))
            roll_eligible = ((p.get('roll_down') or _crash_roll) and spot_u < sp['K'] * 0.98
                             and dte_left > min_roll_dte)
            # Trend-aware skip — only if flag enabled
            if roll_eligible and p.get('trend_aware_roll'):
                if bool(row.get('ung_uptrend', False)):
                    # Uptrend → let it recover, skip the roll
                    roll_eligible = False
            # SURGE-Z GATE: when UNG has dumped hard (surge_z < -1.5),
            # rolling down LOCKS IN the loss right before mean reversion.
            # Per session analysis: 2022-02-03 PUT_ROLL_DOWN locked -$8.6K
            # right before spot rallied -19.4% in 5d. PUT_ROLL_DOWN is the
            # single biggest losing trade type (-$41K cumulative on
            # premium_harvest_scale_invariant). Skip rolls during outsized
            # down-moves; let position ride to recovery or assignment.
            if roll_eligible and p.get('surge_gate_roll', True):
                _sz = float(row.get('ung_surge_z') or 0.0)
                if _sz < -1.5:
                    roll_eligible = False
                    trades.append({'date': idx, 'type': 'PUT_ROLL_SKIP_SURGE',
                                   'pnl': 0.0, 'qty': sp['qty'], 'K': sp['K'],
                                   'spot': spot_u, 'surge_z': round(_sz, 2),
                                   'note': 'spot dumped hard; mean reversion likely → skip roll'})
            # ASSIGNMENT-AWARE roll skip: if put is deep ITM AND we WANT shares
            # at this strike (e.g., extreme cheap regime), let it assign rather
            # than roll. Wheel mechanic: assignment = "accumulate at the strike
            # we picked". Only skip when kernel wants accumulation (regime cheap).
            if roll_eligible and p.get('assignment_aware_put_skip', True):
                leg_iv = iv_at(sp['K'], int(dte_left), 'P')
                leg_prem = bs_put(spot_u, sp['K'], T_left, leg_iv)
                a = assignment_probability(K=sp['K'], spot=spot_u, dte=int(dte_left),
                                            iv=leg_iv, right='PUT',
                                            premium_market=leg_prem,
                                            mean_reversion_z=float(row.get('ung_surge_z') or 0.0))
                # If assignment is certain/very-likely AND extrinsic is gone,
                # rolling pays the loss now instead of letting wheel mechanic
                # complete. Skip the roll in those cases.
                if a['p_assign'] >= 0.85 and a['extrinsic'] < 0.10:
                    roll_eligible = False
                    trades.append({'date': idx, 'type': 'PUT_DEFER_TO_ASSIGN',
                                   'pnl': 0.0, 'qty': sp['qty'], 'K': sp['K'],
                                   'spot': spot_u, 'p_assign': a['p_assign'],
                                   'note': 'deep-ITM low-extrinsic → assignment cheaper than roll'})
            # GEN-4 ROLL GUARDS (KERNEL_LAB findings #2/#4: -$218k roll cost,
            # 28% futile, cascade days rolled the whole book at once):
            # (a) roll_accept_cheap_z — in accumulation regime (z<-0.5, no
            #     falling knife) take assignment; we WANT shares there
            if (roll_eligible and p.get('roll_accept_cheap_z')
                    and z < -0.5 and not falling_knife(row)):
                roll_eligible = False
                trades.append({'date': idx, 'type': 'PUT_ROLL_SKIP_CHEAPZ',
                               'pnl': 0.0, 'K': sp['K'], 'z': round(z, 2)})
            # (b) max_rolls_per_chain — paying to delay >N times is futile
            if (roll_eligible and p.get('max_rolls_per_chain') is not None
                    and sp.get('rolls', 0) >= p['max_rolls_per_chain']):
                roll_eligible = False
                trades.append({'date': idx, 'type': 'PUT_ROLL_SKIP_CHAIN',
                               'pnl': 0.0, 'K': sp['K'],
                               'rolls': sp.get('rolls', 0)})
            # (c) roll_stagger_max_per_day — never the whole book into one
            #     vol-spike's spreads
            if roll_eligible and p.get('roll_stagger_max_per_day') is not None:
                _rd = s.get('_rolls_day')
                _n_today = _rd[1] if (_rd and _rd[0] == idx) else 0
                if _n_today >= p['roll_stagger_max_per_day']:
                    roll_eligible = False
                    trades.append({'date': idx, 'type': 'PUT_ROLL_SKIP_STAGGER',
                                   'pnl': 0.0, 'K': sp['K']})
            if roll_eligible:
                nk = round(spot_u * (1 - p.get('otm_put', 0.10)))
                # close leg: buy back old at AUDITED ask; open leg: sell new at AUDITED bid
                _cv_model = bs_put(spot_u, sp['K'], T_left, iv_at(sp['K'], int(T_left*365), 'P'))
                cv, _aud_c = exec_fill(idx, sp['K'], int(T_left*365), 'P', 'buy',
                                       spot_u, p, _cv_model)
                _npr_model = bs_put(spot_u, nk, 30/365, iv_at(nk, 30, 'P'))
                npr, _aud_o = exec_fill(idx, nk, 30, 'P', 'sell', spot_u, p, _npr_model)
                close_pnl = (sp['entry_prem'] - cv) * 100 * sp['qty']
                s['cash'] -= cv * 100 * sp['qty']
                s['cash'] += npr * 100 * sp['qty'] - sp['qty'] * SPREAD_OPTION * 100
                keep.append({'entry': idx, 'K': nk, 'dte': 30, 'qty': sp['qty'],
                             'entry_prem': npr, 'rolls': sp.get('rolls', 0) + 1})
                _rd = s.get('_rolls_day')
                s['_rolls_day'] = (idx, (_rd[1] + 1) if (_rd and _rd[0] == idx) else 1)
                # Roll P&L = closed leg's gain (premium collected may be future credit)
                trades.append({'date': idx, 'type': 'PUT_ROLL_DOWN', 'pnl': close_pnl,
                               'from_K': sp['K'], 'to_K': nk, 'qty': sp['qty'],
                               'exec_time': _aud_o['exec_time'], 'bid': _aud_o['bid'],
                               'ask': _aud_o['ask'], 'spread_pct': _aud_o['spread_pct'],
                               'fill_source': _aud_o['source'],
                               'close_exec_time': _aud_c['exec_time'],
                               'close_spread_pct': _aud_c['spread_pct'],
                               'close_fill_source': _aud_c['source']})
                continue

            # EARLY ASSIGNMENT: a deep-ITM short put with ~zero extrinsic (|delta|>0.99) is near-certain
            # to be exercised EARLY by the long holder (no time value left to keep). Model it rather than
            # waiting for expiry — realistic and broker-independent. [[feedback_synthetic_early_assignment]]
            _early_p = False
            if p.get('model_early_assign', True) and days < sp['dte'] and spot_u < sp['K']:
                _ivp = iv_at(sp['K'], max(1, sp['dte'] - days), 'P')
                _dlt, _ = bs_greeks_pt(spot_u, sp['K'], T_left, _ivp, 'P')
                _extr = bs_put(spot_u, sp['K'], T_left, _ivp) - max(0.0, sp['K'] - spot_u)
                _early_p = (_dlt < -0.99) and (_extr < p.get('early_assign_extr', 0.02))
            if days >= sp['dte'] or _early_p:
                if spot_u < sp['K']:
                    # Assigned: buy 100*qty shares at the strike (worth spot now).
                    # BUGFIX 2026-06-16: was double-deducting the intrinsic loss
                    # (extra `cash -= (K-spot)*...` on top of paying the strike) —
                    # asymmetric vs CALL_ASSIGN. Correct = pay strike, receive shares;
                    # NAV then drops by exactly (K-spot)*100*qty = the real loss.
                    loss = (sp['K'] - spot_u) * 100 * sp['qty']
                    pnl = sp['entry_prem'] * 100 * sp['qty'] - loss   # trade-record P&L
                    s['shares'] += sp['qty'] * 100
                    s['cash'] -= sp['qty'] * 100 * sp['K']
                    trades.append({'date': idx, 'type': 'PUT_EARLY_ASSIGN' if _early_p else 'PUT_ASSIGN',
                                   'qty': sp['qty'], 'pnl': pnl, 'K': sp['K']})
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
            if p.get('tp_50') and not p.get('ablate_call_tp'):
                tp_thresh = p.get('tp_threshold', 0.5)
                if p.get('tp_dynamic'):
                    rv30 = float(row.get('rv_30') or 0.5)
                    if rv30 > 0.80:   tp_thresh = 0.7
                    elif rv30 < 0.40: tp_thresh = 0.3
                    else:             tp_thresh = 0.5
                # GRIND-AWARE TP: during slow chronic declines, exit puts
                # earlier (lower tp_thresh = exit at higher premium-capture %).
                if p.get('grind_tp_accelerate') and detect_grind_down(row):
                    tp_thresh = min(tp_thresh, 0.3)
            if tp_thresh is not None and T_left > 1/365:
                _cv_model = bs_call(spot_u, sc['K'], T_left, iv_at(sc['K'], int(T_left*365), 'C'))
                cv, _aud = exec_fill(idx, sc['K'], int(T_left*365), 'C', 'buy',
                                     spot_u, p, _cv_model)
                if cv < sc['entry_prem'] * tp_thresh:
                    pnl = (sc['entry_prem'] - cv) * 100 * sc['qty']
                    s['cash'] += pnl
                    trades.append({'date': idx, 'type': 'CALL_TP', 'pnl': pnl,
                                   'K': sc['K'], 'qty': sc['qty'],
                                   'dte': int(round(T_left * 365)),
                                   'expiry': sc.get('expiry'), 'buyback': round(cv, 2),
                                   'exec_time': _aud['exec_time'], 'bid': _aud['bid'],
                                   'ask': _aud['ask'], 'spread_pct': _aud['spread_pct'],
                                   'fill_source': _aud['source']})
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
            # GEN-4 spike-day patience (KERNEL_LAB #3: worst day was call-
            # side on +3.9% spot; knee-jerks reverse — 5s study r=-.19):
            # defer roll-ups for N days after a >3% up-move
            _spike_defer = False
            if p.get('roll_up_spike_defer_days') and i >= 2:
                try:
                    _r1 = df['UNG'].iloc[i] / df['UNG'].iloc[i-1] - 1
                    _r2 = df['UNG'].iloc[i-1] / df['UNG'].iloc[i-2] - 1
                    if max(_r1, _r2) > 0.03:
                        _spike_defer = True
                except Exception:
                    pass
            if (p.get('roll_up_calls') and is_itm and near_expiry
                    and in_cheap_neutral and T_left > 1/365
                    and not _spike_defer):
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

            # EARLY ASSIGNMENT (calls): deep-ITM short call with ~zero extrinsic (delta>0.99) → shares
            # called away EARLY (holder exercises rather than hold a no-time-value option). Same model
            # as puts; covered 1:1 so it just moves the called-away forward in time.
            # CALLS gated behind early_assign_calls (default OFF): early exercise of a call is only
            # rational to capture a DIVIDEND. UNG pays none → early call assignment ≈ never happens;
            # modeling it (252 events) was unrealistic and inflated Sharpe. Enable only for dividend names.
            _early_c = False
            if (p.get('model_early_assign', True) and p.get('early_assign_calls', False)
                    and days < sc['dte'] and spot_u > sc['K']):
                _ivc = iv_at(sc['K'], max(1, sc['dte'] - days), 'C')
                _dltc, _ = bs_greeks_pt(spot_u, sc['K'], T_left, _ivc, 'C')
                _extrc = bs_call(spot_u, sc['K'], T_left, _ivc) - max(0.0, spot_u - sc['K'])
                _early_c = (_dltc > 0.99) and (_extrc < p.get('early_assign_extr', 0.02))
            if days >= sc['dte'] or _early_c:
                if spot_u > sc['K']:
                    # Premium kept, but shares called away at K (lost spot-K)
                    lost = (spot_u - sc['K']) * 100 * sc['qty']
                    pnl = sc['entry_prem'] * 100 * sc['qty'] - lost
                    s['shares'] -= sc['qty'] * 100
                    s['cash'] += sc['qty'] * 100 * sc['K']
                    # arm event-driven re-accumulation when a LARGE block is called away (the share
                    # book just dropped well below target — glide it back over the next few bars).
                    if p.get('reaccum_on_called_away') and sc['qty'] >= p.get('reaccum_lots_threshold', 5):
                        s['_reaccum_until'] = i + int(p.get('reaccum_window', 5))
                    trades.append({'date': idx, 'type': 'CALL_EARLY_ASSIGN' if _early_c else 'CALL_ASSIGN',
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
        # (BOXX yield accrued once, in the BOXX-management block below — was double-counted here)

        # KOLD exit (skip when regime_kold manages the hedge by regime strength)
        if s['kold'] > 0 and z > -0.3 and not p.get('regime_kold_max_pct'):
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
                nav_now = s['cash'] + s['shares'] * spot_u + s['boxx'] * float((row.get('BOXX') if (row.get('BOXX') == row.get('BOXX') and row.get('BOXX') is not None) else 117.0))
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

            # REGIME-AWARE PUT QTY — smaller in NEUTRAL noise, bigger in
            # CHEAP/RICH where conviction is clearer. Independent of regime_aware_strike.
            if p.get('regime_aware_put_qty'):
                if abs(z) < 0.5:
                    put_qty = max(1, int(put_qty * 0.6))
                elif abs(z) > 1.5:
                    put_qty = int(put_qty * 1.3)

            # NAV-RELATIVE SIZING — scale put_qty by current NAV so the
            # strategy works at any capital level. Default off (use fixed
            # put_qty); when on, put_qty = floor(nav * pct / strike_estimate).
            # This prevents over-leverage at small NAV and under-utilization
            # at large NAV.
            put_nav_pct = p.get('put_qty_nav_pct', 0)
            if put_nav_pct > 0:
                # Estimate strike at otm_put OTM
                K_est = spot_u * (1 - otm_put)
                if K_est > 0:
                    target_put = int(cur_nav * put_nav_pct / (K_est * 100))
                    put_qty = max(1, min(target_put, int(p.get('put_qty_max', 50))))
                    put_qty = int(put_qty * size_scale)
            call_nav_pct = p.get('call_qty_nav_pct', 0)
            if call_nav_pct > 0:
                K_est = spot_u * (1 + otm_call)
                if K_est > 0:
                    target_call = int(cur_nav * call_nav_pct / (K_est * 100))
                    call_qty = max(1, min(target_call, int(p.get('call_qty_max', 50))))
                    call_qty = int(call_qty * size_scale)

            # IV-SHAPE SIZING — react to real surface term & skew.
            # put_skew rich → sell MORE puts (premium is fat)
            # call_skew rich → sell MORE calls
            # term contango → prefer longer DTE (premium decays slower)
            # term backwardation (negative slope) → vol spike imminent, reduce
            if p.get('iv_shape_sizing') and iv_shape:
                # Skew-based qty scaling
                ps = iv_shape['put_skew']
                cs = iv_shape['call_skew']
                # Reference levels: typical UNG put_skew ~0.02-0.05, rich >0.08
                if ps > 0.08:
                    put_qty = int(put_qty * 1.5)
                elif ps > 0.05:
                    put_qty = int(put_qty * 1.2)
                elif ps < 0.01:
                    put_qty = max(1, int(put_qty * 0.7))
                if cs > 0.05:
                    call_qty = int(call_qty * 1.3)
                elif cs < 0:
                    call_qty = max(1, int(call_qty * 0.7))
                # Backwardation = imminent vol spike → cut size
                ts = iv_shape['term_slope']
                if ts < -0.05:
                    put_qty = max(1, int(put_qty * 0.6))
                    call_qty = max(1, int(call_qty * 0.6))

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
            # NEW: Grind-down filter — no new PUTS during slow chronic declines
            # (catches Dec 2023-style multi-week bleeds the anomaly misses)
            if p.get('skip_puts_on_grind_down') and detect_grind_down(row):
                trades.append({'date': idx, 'type': 'STAND_DOWN_GRIND',
                               'pnl': 0.0, 'r30': float(row.get('ung_30d_return') or 0)})
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
            # SELECTIVE: skip puts ONLY on ANOMALY_DOWN (allow UP for spike premium)
            if p.get('anomaly_standdown_down_only') and anomaly == 'ANOMALY_DOWN':
                skip_put = True
            if p.get('downtrend_standdown') and in_sustained_down:
                skip_put = True
            if (p.get('downtrend_from_high_standdown')
                    and in_sustained_down
                    and spot_u > p.get('downtrend_high_floor', 0)):
                skip_put = True
            if p.get('falling_knife_filter') and falling_knife(row):
                skip_put = True
            if p.get('skip_puts_on_grind_down') and detect_grind_down(row):
                skip_put = True  # multi-week chronic decline — don't add to share-acq risk
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
                # AGGRESSIVE ITM PUT GATE — when z is RICH (overvalued),
                # sell ITM puts to harvest fat premium AND acquire shares
                # at discount via assignment. Only fires when we want
                # MORE shares (under target). Pairs with smaller share base
                # to keep total inventory controlled.
                aip_z = p.get('aggressive_itm_put_z')
                if aip_z is not None and z > aip_z:
                    # Allow ITM only if we're below target shares (room to acquire)
                    pct_b = p.get('z_share_target_pct_nav')
                    if pct_b and spot_u > 0:
                        target_check = int(cur_nav * pct_b / spot_u)
                    else:
                        target_check = p.get('z_share_target_base', 6200)
                    if s['shares'] < target_check * 1.2:  # not over-acquired
                        otm_put = p.get('itm_put_pct', -0.05)  # 5% ITM by default
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
                # GEN-11 ANGLE A — CONVICTION ITM PUT (bullish expression).
                # When DEEP-CHEAP z + LOW iv-rank + momentum-confirm (not a
                # falling knife, regime NORMAL) and we have room to acquire,
                # sell ITM puts to accumulate at a CUSHIONED effective basis.
                # Return source = basis discount + intrinsic premium, NOT extra
                # net delta: at the SAME target share count we arrive cheaper
                # and with less time-at-risk. Depth scales with conviction
                # (deeper-cheap z → deeper ITM). [[project_ung_iv_rank_alpha]]
                # says LOW real IV-rank is the bullish-entry condition.
                if p.get('conviction_itm_put'):
                    _civ = row.get('iv_rank')
                    # NaN-SAFE: unknown IV-rank → do NOT take the IVR-gated
                    # aggressive action (never decide on missing data).
                    _ivr_ok = (_civ == _civ and _civ is not None
                               and _civ < p.get('conviction_itm_ivr_max', 0.4))
                    _zg = p.get('conviction_itm_z', -1.0)
                    _pctb = p.get('z_share_target_pct_nav')
                    if _pctb and spot_u > 0:
                        _tgt = int(cur_nav * _pctb / spot_u)
                    else:
                        _tgt = p.get('z_share_target_base', 6200)
                    if (z < _zg and _ivr_ok and not falling_knife(row)
                            and anomaly == 'NORMAL'
                            and s['shares'] < _tgt * 1.2):
                        # scale depth: 1 full z-unit below gate → full depth
                        _md = p.get('conviction_itm_depth', 0.06)
                        _sc = min(1.0, max(0.0, (_zg - z)))
                        effective_otm = -(_md * (0.5 + 0.5 * _sc))  # ITM (K>spot)
                        put_qty = int(put_qty
                                      * p.get('conviction_itm_qty_mult', 1.3))
                        trades.append({'date': idx, 'type': 'CONVICTION_ITM_PUT',
                                       'pnl': 0.0, 'z': z, 'spot': spot_u,
                                       'depth': round(effective_otm, 4),
                                       'iv_rank': _civ})
                # GAP-DRIVEN WHEEL (user design): the share gap drives put SIZING,
                # STRIKE-depth (and DTE below) so the book steers to target via
                # ASSIGNMENT — get PAID to acquire, never direct-buy. Deeper gap →
                # bigger size + closer/ITM strike (high Δ, fast paid acquisition).
                _gap_dte = None
                if p.get('gap_to_wheel'):
                    _gap = s.get('_share_gap_lots', 0)
                    if _gap > 0 and z < p.get('gap_wheel_z_max', 0.75):
                        _per = max(1, int(math.ceil(_gap * p.get('gap_wheel_fill_frac', 0.34))))
                        put_qty = max(put_qty, min(_per, int(p.get('gap_wheel_max_lots', 25))))
                        if _gap >= p.get('gap_wheel_itm_lots', 12):
                            effective_otm = min(effective_otm, p.get('gap_wheel_itm_otm', -0.03))
                            _gap_dte = p.get('gap_wheel_itm_dte', 30)   # urgent → shorter DTE
                        elif _gap >= p.get('gap_wheel_atm_lots', 5):
                            effective_otm = min(effective_otm, 0.01)
                    elif _gap < 0:
                        put_qty = 0   # above target → acquire nothing; CCs divest
                K = round(spot_u * (1 - effective_otm))
                # Tunable open-DTE (default 30; tastytrade rule uses 45).
                # GAP-DRIVEN: urgent acquisition (deep gap) → shorter DTE to assign sooner.
                open_dte = _gap_dte if _gap_dte else p.get('open_dte', 30)
                if p.get('vol_aware_dte'):
                    rv30 = float(row.get('rv_30') or 0.5)
                    if rv30 > 0.80:   open_dte = 60
                    else:             open_dte = 45
                # REAL STRIKE SNAP — now per-expiry aware. Compute the actual
                # expiration date for this contract, then snap to strikes that
                # exist on THAT expiry (monthly=integer, weekly=half).
                if p.get('use_real_strikes'):
                    _exp_d = idx.date() + timedelta(days=int(open_dte))
                    while _exp_d.weekday() != 4:
                        _exp_d += timedelta(days=1)
                    K = snap_to_real_strike(K, d_str, 'P',
                                             expiration=_exp_d.isoformat(),
                                             spot=spot_u)
                prem = bs_put(spot_u, K, open_dte/365, iv_at(K, open_dte, 'P'))
                # REAL-FILL MODEL: scale entry credit by the empirical
                # bid/BSM grid from 8y of actual UNG quotes (30-45d OTM
                # puts really fill at 0.67-0.95x the model estimate)
                if p.get('real_fill_model', True):  # gen-9: real fills are now DEFAULT
                    prem *= fill_factor('P', open_dte, 1 - K / spot_u)
                # TIER-3 REAL CHAIN: when on, replace the model premium with the
                # ACTUAL historical bid you'd sell into (incl. ~$0 in illiquid
                # low-price regimes). The `prem > 0.05` gate below then naturally
                # SKIPS puts that aren't really bid — no fictional premium.
                if p.get('real_chain_pricing'):
                    _rb = real_chain_price(idx, K, open_dte, 'P', spot_u)
                    if _rb is not None:
                        prem = _rb
                # MICROSTRUCTURE TIMING: Thursday put entries (post-print
                # day bleeds -40bps intraday → cheaper strikes + the
                # executor's validated 14:30-15:30 window)
                if (p.get('put_entry_dow') is not None
                        and idx.dayofweek != p['put_entry_dow']):
                    prem = 0  # not the entry day — skip this cycle
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
                # HH BASIS STORM SKIP: backwardation > +\$0.40 historically
                # precedes UNG -4.5% in 5d. Don't sell new puts → don't get
                # assigned into a falling market. Param: hh_storm_skip_puts.
                if p.get('hh_storm_skip_puts', False) and int(row.get('hh_basis_storm', 0)):
                    trades.append({'date': idx, 'type': 'PUT_SKIP_HH_STORM',
                                   'pnl': 0.0, 'hh_basis': float(row.get('hh_basis', 0)),
                                   'note': 'backwardation storm → defensive'})
                    prem = 0  # neutralize subsequent put-write conditional
                # GEN-11 C3 — CASH-SECURED PUT RATIO (bullish, defined-risk accum).
                # On deep-cheap z + momentum-confirm, sell MORE cash-secured puts
                # (the "2") to accumulate aggressively, then buy 1 long put per 2
                # shorts at a lower strike (the "1") = a hard downside FLOOR so the
                # extra accumulation is DEFINED-risk. Both shorts cash-secured
                # (margin-checked below); the long put bounds the tail. Boost runs
                # BEFORE the margin check so the larger size is collateral-tested.
                _c3_active = False
                if (p.get('put_ratio') and prem > 0.05
                        and z < p.get('put_ratio_z', -1.0)
                        and float(row.get('ung_surge_z') or 0.0)
                            > p.get('put_ratio_surge', 0.0)
                        and not falling_knife(row)
                        # ROUTER safety: optionally only accumulate aggressively in
                        # NORMAL regime — skips spike-crash periods where the 2x
                        # accumulation bloated TRAIN DD to -16.9%.
                        and (not p.get('put_ratio_normal_only')
                             or anomaly == 'NORMAL')):
                    put_qty = int(put_qty * p.get('put_ratio_qty_mult', 1.5))
                    _c3_active = True
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
                        # GAMMA-WEIGHTED ladder: allocate contracts ∝ 1/gamma per DTE so each expiry
                        # carries ~EQUAL gamma (short DTE = HIGH gamma → FEWER contracts; gamma∝1/√T).
                        # Even-split (put_qty//n) dumps gamma into the short bucket — that's WHY the
                        # even ladder tested LESS smooth. Skip very-short DTEs in the ladder config.
                        if p.get('gamma_weighted_ladder') and n_dtes > 1:
                            _gw = []
                            for _d in dte_ladder:
                                _, _g = bs_greeks_pt(spot_u, K, _d / 365, iv_at(K, _d, 'P'), 'P')
                                _gw.append(1.0 / max(_g, 1e-9))
                            _gws = sum(_gw) or 1.0
                            dte_qtys = [int(round(put_qty * w / _gws)) for w in _gw]
                            # clamp the rounding residual so Σ NEVER exceeds the margin-approved put_qty
                            # (3+ buckets can round up to put_qty+1, bypassing the collateral check).
                            while sum(dte_qtys) > put_qty and max(dte_qtys) > 0:
                                dte_qtys[dte_qtys.index(max(dte_qtys))] -= 1
                        else:
                            dte_qtys = [max(1, put_qty // n_dtes)] * n_dtes
                        # OU TILT (fine-scale buy-low timing): OU-cheap (z<0, bounce) → sell MORE;
                        # OU-rich (z>0, fade) → sell FEWER (assignment risk). |z| sizes it.
                        if p.get('ou_tilt'):
                            _ou = row.get('ou_z'); _ou = _ou if (_ou == _ou and _ou is not None) else 0.0
                            _f = 1.0 - max(-1.0, min(1.0, _ou)) * p.get('ou_tilt_k', 0.5)
                            dte_qtys = [max(0, int(round(q * _f))) for q in dte_qtys]
                        # Re-margin-check the FULL allocation
                        for _di, dte_choice in enumerate(dte_ladder):
                            per_dte_qty = dte_qtys[_di]
                            if per_dte_qty < 1:
                                continue
                            # Re-price for this specific DTE — model, then AUDITED fill
                            # (intraday minute path → EOD real → model), stamping the
                            # trade with exec_time / bid / ask / spread / source.
                            iv_dte = iv_at(K, dte_choice, 'P')
                            model_dte = bs_put(spot_u, K, dte_choice/365, iv_dte)
                            if p.get('real_fill_model', True):
                                model_dte *= fill_factor('P', dte_choice, 1 - K / spot_u)
                            prem_dte, _aud = exec_fill(idx, K, dte_choice, 'P', 'sell',
                                                       spot_u, p, model_dte)
                            if prem_dte < 0.05:
                                continue
                            # GAMMA-CAP (anti-concentration): limit short contracts per
                            # strike×expiry so the book doesn't stack short gamma at one
                            # point (pin/whipsaw + mass-assignment risk — the $11/23-lot).
                            q_dte = per_dte_qty
                            if p.get('scenario_delta_target'):
                                # GAMMA-AWARE cap: cap the BOOK's EXPECTED put-assignment delta
                                # (Σ P(assign|z,dte)·qty·100) to a fraction of NAV-equiv shares.
                                # Probability-weighted + DTE-aware → tightens in bearish-z / long-dte,
                                # loosens in calm-z / short-dte. Replaces the flat notional cap.
                                _z = compute_historical_z(row, use_surprise=p.get('use_surprise', False))
                                _A = p.get('scenario_mu_a', -0.001205)
                                _B = p.get('scenario_mu_b', -0.000112)
                                _Sg = p.get('scenario_sigma', 0.04066)
                                _exp_d = sum(p_assign(_sp['K'], spot_u, _sp.get('dte', 30), _z, _A, _B, _Sg)
                                             * _sp['qty'] * 100 for _sp in s['short_puts'])
                                _target = p['scenario_delta_target'] * (cur_nav / spot_u)
                                _pn = p_assign(K, spot_u, dte_choice, _z, _A, _B, _Sg)
                                _cap_sc = int(max(0.0, _target - _exp_d) / (_pn * 100)) if _pn > 1e-6 else q_dte
                                q_dte = min(q_dte, max(0, _cap_sc))
                                if q_dte < 1:
                                    continue
                            elif p.get('gamma_cap'):
                                _ex = sum(_sp['qty'] for _sp in s['short_puts']
                                          if abs(_sp['K'] - K) < 0.01
                                          and abs(_sp.get('dte', 30) - dte_choice) <= 7)
                                # SCALE-INVARIANT cap: when max_short_pct_nav is set, the per-
                                # strike×expiry cap is a % of NAV notional (proportional to the
                                # account), not a fixed count — so a $50k and a $500k book are
                                # capped at the SAME fraction of risk. Falls back to the fixed
                                # max_short_per_strike for legacy kernels. cap = pct·NAV/(K·100).
                                if p.get('max_short_pct_nav') and K > 0:
                                    _cap_ct = max(1, int(p['max_short_pct_nav'] * cur_nav / (K * 100)))
                                else:
                                    _cap_ct = p.get('max_short_per_strike', 10)
                                q_dte = min(q_dte, max(0, _cap_ct - _ex))
                                if q_dte < 1:
                                    continue   # strike×expiry full → ladder tries other DTEs
                            credit_dte = prem_dte * 100 * q_dte - q_dte * SPREAD_OPTION * 100
                            s['cash'] += credit_dte
                            s['short_puts'].append({'entry': idx, 'K': K, 'dte': dte_choice,
                                                    'qty': q_dte, 'entry_prem': prem_dte})
                            trades.append({'date': idx, 'type': 'OPEN_PUT',
                                           'pnl': 0.0, 'credit': credit_dte,
                                           'K': K, 'qty': q_dte, 'dte': dte_choice,
                                           'exec_time': _aud['exec_time'], 'bid': _aud['bid'],
                                           'ask': _aud['ask'], 'spread_pct': _aud['spread_pct'],
                                           'fill_source': _aud['source']})
                        # GEN-11 C3 — buy the long-put FLOOR (1 per 2 short puts)
                        # at a lower strike = defined downside for the aggressive
                        # accumulation. Settles vs spot at expiry (pays if UNG dumps).
                        if _c3_active and put_qty >= 2:
                            _nfloor = put_qty // 2
                            K_floor = round(spot_u * (1 - p.get('put_ratio_floor_otm', 0.12)))
                            fcost = bs_put(spot_u, K_floor, open_dte/365,
                                           iv_at(K_floor, open_dte, 'P'))
                            if fcost > 0.02:
                                f_debit = fcost * 100 * _nfloor + _nfloor * SPREAD_OPTION * 100
                                if s['cash'] > f_debit + 500:
                                    s['cash'] -= f_debit
                                    s['long_puts'].append({'entry': idx, 'K': K_floor,
                                                           'dte': open_dte, 'qty': _nfloor,
                                                           'cost': fcost})
                                    trades.append({'date': idx, 'type': 'OPEN_PUT_RATIO_FLOOR',
                                                   'pnl': -f_debit, 'K': K_floor,
                                                   'qty': _nfloor, 'spot': spot_u, 'z': z})

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
                # GEN-11 ANGLE B — ITM COVERED-CALL DIVEST (bearish expression).
                # When z is RICH (z>0 = expensive) AND price is HOT (surge_z up)
                # AND we're at/over the share target, sell DEEP-ITM covered calls
                # to monetize the rich price and pre-commit a high-probability
                # called-away exit — divest INTO strength. Pure income on shares
                # already held, fully covered 1:1 (qty capped to uncovered//100
                # below). [[feedback_synthetic_early_assignment]]
                # [[feedback_hot_shares_rocket_divest]]
                if p.get('itm_cc_divest'):
                    _dz = p.get('itm_cc_divest_z', 0.75)   # rich threshold
                    _sz = float(row.get('ung_surge_z') or 0.0)
                    _hot = _sz > p.get('itm_cc_divest_surge', 0.5)
                    # No over-target gate: the outer uncovered_shares>=100 already
                    # ensures we hold shares to divest; when rich the book is
                    # intentionally below full target, so divest what remains.
                    if z > _dz and _hot:
                        use_itm = True
                        effective_otm = p.get('itm_cc_divest_pct', -0.08)  # ~8% ITM
                        trades.append({'date': idx, 'type': 'ITM_CC_DIVEST',
                                       'pnl': 0.0, 'spot': spot_u, 'z': z,
                                       'surge_z': round(_sz, 2)})
                # GEN-11 C2 — COVERED UPSIDE-TAIL RATIO (neutral-income).
                # When z is NEUTRAL and IV-rank is ELEVATED (rich premium), push
                # the CC strike CLOSER to money to harvest MORE premium, then buy
                # 1 long call per 2 shorts (further OTM) as a tail cap so the
                # extra assignment/upside give-up is bounded. All shorts stay
                # covered 1:1 by shares (qty<=uncovered//100); the long tail is
                # the "1" in the short-2/long-1 ratio. Income-POSITIVE on net
                # (sell 2 collect, buy 1 pay) — unlike C1 which is a net debit.
                _c2_active = False
                if p.get('cc_tail_ratio') and not use_itm:
                    _c2ivr = row.get('iv_rank')
                    if (abs(z) < p.get('cc_tail_z', 0.5)
                            and _c2ivr == _c2ivr and _c2ivr is not None
                            and _c2ivr > p.get('cc_tail_ivr', 0.5)):
                        effective_otm = min(effective_otm,
                                            p.get('cc_tail_call_otm', 0.04))
                        _c2_active = True
                K = round(spot_u * (1 + effective_otm))
                # GEX WALL FLOOR (live-validated: 74% vs 69% final-week hold):
                # never sell OTM CCs below the dealer call wall. ITM CCs
                # (intentional divestment) are exempt — they WANT assignment.
                if p.get('cc_gex_floor') and effective_otm >= 0:
                    _gwall = row.get('gex_call_wall')
                    if _gwall == _gwall and _gwall and _gwall > K:
                        K = round(float(_gwall))
                qty = min(call_qty, uncovered_shares // 100)
                cc_dte = p.get('open_dte', 30)
                if p.get('vol_aware_dte'):
                    rv30 = float(row.get('rv_30') or 0.5)
                    if rv30 > 0.80:   cc_dte = 60
                    else:             cc_dte = 45
                # REAL STRIKE SNAP — per-expiry aware (monthly=integer, weekly=half)
                if p.get('use_real_strikes'):
                    _exp_d = idx.date() + timedelta(days=int(cc_dte))
                    while _exp_d.weekday() != 4:
                        _exp_d += timedelta(days=1)
                    K = snap_to_real_strike(K, d_str, 'C',
                                             expiration=_exp_d.isoformat(),
                                             spot=spot_u)
                prem = bs_call(spot_u, K, cc_dte/365, iv_at(K, cc_dte, 'C'))
                if p.get('real_fill_model', True):  # gen-9: real fills are now DEFAULT
                    prem *= fill_factor('C', cc_dte, K / spot_u - 1)
                # AUDITED fill (intraday minute → EOD real → model)
                prem, _aud_cc = exec_fill(idx, K, cc_dte, 'C', 'sell', spot_u, p, prem)
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
                # OU TILT (fine-scale sell-high timing): when OU-rich (z>0, fade expected)
                # sell MORE calls; when OU-cheap (z<0, bounce expected) sell FEWER (let it run).
                if p.get('ou_tilt'):
                    _ouc = row.get('ou_z'); _ouc = _ouc if (_ouc == _ouc and _ouc is not None) else 0.0
                    _fc = 1.0 + max(-1.0, min(1.0, _ouc)) * p.get('ou_tilt_k', 0.5)
                    qty = max(0, min(int(round(qty * _fc)), uncovered_shares // 100))
                if prem > 0.05 and qty >= 1:
                    credit = prem * 100 * qty - qty * SPREAD_OPTION * 100
                    s['cash'] += credit
                    # decaying tracker of recent CC premium → funds the G7-1 collar
                    s['_cc_prem_recent'] = s.get('_cc_prem_recent', 0) * 0.7 + credit
                    s['short_calls'].append({'entry': idx, 'K': K, 'dte': cc_dte,
                                             'qty': qty, 'entry_prem': prem,
                                             'is_itm_aggressive': use_itm})
                    open_kind = 'OPEN_ITM_CC' if use_itm else 'OPEN_CC'
                    trades.append({'date': idx, 'type': open_kind,
                                   'pnl': 0.0, 'credit': credit,
                                   'K': K, 'qty': qty, 'z': z,
                                   'exec_time': _aud_cc['exec_time'], 'bid': _aud_cc['bid'],
                                   'ask': _aud_cc['ask'], 'spread_pct': _aud_cc['spread_pct'],
                                   'fill_source': _aud_cc['source']})

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

                    # GEN-11 C1 — COVERED CALL BACKSPREAD (bullish-convex).
                    # The short CC just opened is SHARE-covered (qty<=uncovered//100).
                    # On top, buy backspread_long_ratio x qty long calls further OTM
                    # → NET LONG convexity (long minus the 1 short). Pays in
                    # vol-EXPANSION / trend-up — the regime the grinder misses.
                    # Short stays covered 1:1 by shares; longs are pure debit
                    # convexity. Fire only z-cheap + momentum-up (surge>0), where
                    # an up-move is plausible, and fund within the CC credit.
                    if (p.get('call_backspread')
                            and z < p.get('backspread_z_max', -0.5)
                            and float(row.get('ung_surge_z') or 0.0)
                                > p.get('backspread_surge_min', 0.0)):
                        _lr = p.get('backspread_long_ratio', 2)
                        _lq = int(qty * _lr)
                        K_long = round(spot_u * (1 + p.get('backspread_long_otm', 0.15)))
                        lcost = bs_call(spot_u, K_long, cc_dte/365,
                                        iv_at(K_long, cc_dte, 'C'))
                        if lcost > 0.02 and _lq > 0:
                            bs_debit = lcost * 100 * _lq + _lq * SPREAD_OPTION * 100
                            _budget = credit * p.get('backspread_budget_frac', 1.0)
                            if bs_debit < _budget and s['cash'] > bs_debit + 500:
                                s['cash'] -= bs_debit
                                s.setdefault('long_calls', []).append({
                                    'entry': idx, 'K': K_long, 'dte': cc_dte,
                                    'qty': _lq, 'cost': lcost,
                                    'wing_for_cc_K': K})
                                trades.append({'date': idx, 'type': 'OPEN_CALL_BACKSPREAD',
                                               'pnl': -bs_debit, 'K': K_long,
                                               'qty': _lq, 'short_K': K, 'short_qty': qty,
                                               'spot': spot_u, 'z': z})

                    # GEN-11 C2 — buy the upside-tail (1 long per 2 short CCs),
                    # further OTM, to cap the give-up from the closer CC strike.
                    if _c2_active and qty >= 2:
                        _ntail = qty // 2
                        K_tail = round(spot_u * (1 + p.get('cc_tail_long_otm', 0.12)))
                        tcost = bs_call(spot_u, K_tail, cc_dte/365,
                                        iv_at(K_tail, cc_dte, 'C'))
                        if tcost > 0.02:
                            t_debit = tcost * 100 * _ntail + _ntail * SPREAD_OPTION * 100
                            if t_debit < credit * 0.5 and s['cash'] > t_debit + 500:
                                s['cash'] -= t_debit
                                s.setdefault('long_calls', []).append({
                                    'entry': idx, 'K': K_tail, 'dte': cc_dte,
                                    'qty': _ntail, 'cost': tcost,
                                    'wing_for_cc_K': K})
                                trades.append({'date': idx, 'type': 'OPEN_CC_TAIL_RATIO',
                                               'pnl': -t_debit, 'K': K_tail,
                                               'qty': _ntail, 'short_K': K,
                                               'short_qty': qty, 'spot': spot_u, 'z': z})

            # EXTREME_RICH bearish stack
            # SUPER-OTM PUT TAIL HEDGE (user exploration): the gap-wheel is SHORT
            # ATM premium, so buy cheap super-OTM puts as crash insurance, sized to
            # the SHORT-PUT exposure (the real risk). Super-OTM = cheap drag, big
            # payoff in a -40% gap — caps the tail. Refresh when none outstanding.
            if p.get('put_tail_hedge'):
                _sp_lots = sum(sp.get('qty', 0) for sp in s['short_puts'])
                _have = sum(lp.get('qty', 0) for lp in s['long_puts'] if lp.get('tail'))
                _want = int(_sp_lots * p.get('put_tail_hedge_ratio', 0.5))
                if _want > _have and _sp_lots >= 2:
                    Kp = round(spot_u * (1 - p.get('put_tail_hedge_otm', 0.20)), 1)
                    hdte = p.get('put_tail_hedge_dte', 60)
                    cost = bs_put(spot_u, Kp, hdte/365, iv_at(Kp, hdte, 'P'))
                    need = _want - _have
                    debit = cost * 100 * need + need * SPREAD_OPTION * 100
                    if cost > 0.005 and s['cash'] > debit + 2000:
                        s['cash'] -= debit
                        s['long_puts'].append({'entry': idx, 'K': Kp, 'dte': hdte,
                                               'qty': need, 'cost': cost, 'tail': True})
                        trades.append({'date': idx, 'type': 'OPEN_TAIL_HEDGE_PUT',
                                       'pnl': -debit, 'K': Kp, 'qty': need, 'spot': spot_u})

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

            # ── GEN-7 BOOK HEDGES (diagnosis: drawdowns are 84%-uncovered
            # share-book beta, NOT over-exposure). Hedge the book, keep the
            # shares. uncov_lots = uncovered share lots (the unhedged risk).
            _cc_lots = sum(sc.get('qty', 0) for sc in s['short_calls'])
            _lp_lots = sum(lp.get('qty', 0) for lp in s['long_puts'])
            uncov_lots = max(0, s['shares'] // 100 - _cc_lots - _lp_lots)

            # G7-1 FUNDED COLLAR: buy a 10%-OTM put-spread on a fraction of
            # the uncovered book, funded by recent CC premium → ~0 net cost,
            # caps catastrophic share-book drawdown without cutting shares.
            if p.get('funded_collar') and uncov_lots >= 5:
                cover = int(uncov_lots * p.get('collar_cover_frac', 0.5))
                if cover >= 1:
                    Kp_long = round(spot_u * 0.90)
                    Kp_short = round(spot_u * 0.80)   # spread floor (cheapens it)
                    long_c = bs_put(spot_u, Kp_long, 60/365, iv_at(Kp_long, 60, 'P'))
                    short_c = bs_put(spot_u, Kp_short, 60/365, iv_at(Kp_short, 60, 'P'))
                    net = (long_c - short_c) * 100 * cover + cover * SPREAD_OPTION * 100
                    # only if recent CC premium roughly funds it (within budget)
                    budget = s.get('_cc_prem_recent', 0)
                    if net < budget * p.get('collar_fund_ratio', 1.0) + 50 and s['cash'] > net + 1000:
                        s['cash'] -= net
                        s['long_puts'].append({'entry': idx, 'K': Kp_long, 'dte': 60,
                                               'qty': cover, 'cost': long_c - short_c,
                                               'spread_floor': Kp_short})
                        trades.append({'date': idx, 'type': 'FUNDED_COLLAR',
                                       'pnl': -net, 'K': Kp_long, 'floor': Kp_short,
                                       'qty': cover, 'uncov_lots': uncov_lots})

            # G7-2 SCALED PUT FLOOR: protective puts sized to the UNCOVERED
            # book (not a token count). Cheapest at low IV-rank.
            if p.get('scaled_put_floor') and uncov_lots >= 5:
                target_hedge = int(uncov_lots * p.get('floor_cover_frac', 0.3))
                if _lp_lots < target_hedge:
                    need = target_hedge - _lp_lots
                    Kp = round(spot_u * 0.92)
                    cost = bs_put(spot_u, Kp, 90/365, iv_at(Kp, 90, 'P'))
                    debit = cost * 100 * need + need * SPREAD_OPTION * 100
                    if s['cash'] > debit + 1000:
                        s['cash'] -= debit
                        s['long_puts'].append({'entry': idx, 'K': Kp, 'dte': 90,
                                               'qty': need, 'cost': cost})
                        trades.append({'date': idx, 'type': 'SCALED_PUT_FLOOR',
                                       'pnl': -debit, 'K': Kp, 'qty': need,
                                       'uncov_lots': uncov_lots})

            # G7-3 KOLD BOOK HEDGE: scale KOLD shares to offset the uncovered
            # UNG book year-round (not shoulder-only). 2x inverse → ~half the
            # KOLD notional offsets the book.
            if p.get('kold_book_hedge') and spot_k > 0 and pd.notna(spot_k) and uncov_lots >= 5:
                target_kold = int(uncov_lots * 100 * spot_u * 0.5
                                  * p.get('kold_book_frac', 0.5) / spot_k)
                if target_kold > s['kold'] and s['cash'] > (target_kold - s['kold']) * spot_k + 500:
                    add = target_kold - s['kold']
                    s['kold'] += add
                    s['cash'] -= add * spot_k + add * SPREAD_SHARE
                    trades.append({'date': idx, 'type': 'KOLD_BOOK_HEDGE',
                                   'pnl': 0.0, 'qty': add, 'uncov_lots': uncov_lots})

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
                    nav = s['cash'] + s['shares'] * spot_u + s['boxx'] * float((row.get('BOXX') if (row.get('BOXX') == row.get('BOXX') and row.get('BOXX') is not None) else 117.0)) + long_calls_mtm(s.get('long_calls'), spot_u, idx, iv_at)
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

            # BOXX management — use REAL price from master_dataset, not hardcoded.
            # 50% margin requirement → can hold 2x free cash in BOXX with margin.
            # Decision: hold BOXX up to (excess_cash * margin_factor) where
            # excess = cash - put_collateral_required - cash_buffer.
            spot_boxx = float((row.get('BOXX') if (row.get('BOXX') == row.get('BOXX') and row.get('BOXX') is not None) else 117.0))
            if p.get('boxx'):
                cash_buffer = p.get('boxx_cash_buffer', 20000)  # keep this much liquid
                if p.get('boxx_sweep_full'):
                    # OOS fix: T-bills/BOXX ARE the put collateral (marginable), so don't
                    # double-reserve cash for puts — sweep idle cash above the buffer to
                    # yield. Recovers the ~54%-NAV idle-cash drag (~+2.6%/yr risk-free).
                    excess = s['cash'] - cash_buffer
                    margin_factor = p.get('boxx_margin_factor', 0.95)
                else:
                    put_collat = sum(sp['K'] * 100 * sp['qty'] for sp in s['short_puts'])
                    excess = s['cash'] - put_collat - cash_buffer
                    margin_factor = p.get('boxx_margin_factor', 0.6)  # default conservative
                if excess > 5000:
                    if p.get('boxx_sweep_full') and p.get('boxx_sweep_direct', True):
                        # SWEEP ALL idle cash above the buffer into BOXX every bar. Buy the EXCESS
                        # DIRECTLY — the old `delta = target - s['boxx']` netted the cash-derived target
                        # against the existing holding, so once BOXX was large it UNDER-SWEPT every cash
                        # inflow (assignments, premium) and left it sitting idle (no yield) until cash
                        # re-accumulated past the whole BOXX value. BOXX is marginable put collateral, so
                        # proactively convert idle cash; margin_factor (0.95) keeps headroom.
                        buy = int(excess * margin_factor / spot_boxx)
                        if buy >= 10:
                            s['boxx'] += buy
                            s['cash'] -= buy * spot_boxx + buy * SPREAD_SHARE
                    else:
                        target_boxx_dollars = excess * margin_factor
                        target_boxx_shares = int(target_boxx_dollars / spot_boxx)
                        delta = target_boxx_shares - s['boxx']
                        if delta >= 10:
                            s['boxx'] += delta
                            s['cash'] -= delta * spot_boxx + delta * SPREAD_SHARE
                elif excess < -1000 and s['boxx'] > 0:
                    # Need cash for puts — sell BOXX
                    needed = min(s['boxx'], int(abs(excess) / spot_boxx) + 10)
                    s['boxx'] -= needed
                    s['cash'] += needed * spot_boxx - needed * SPREAD_SHARE

            # BOXX return comes ENTIRELY from PRICE appreciation (box-spread ETF, no
            # distributions — the master_dataset BOXX series appreciates ~4.6%/yr and is
            # marked to market in NAV). The old synthetic 4.74% cash yield here DOUBLE-COUNTED
            # it (~+2.4%/yr fake on a 50%-BOXX book). Removed 2026-06-17 — price MTM is the return.

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
        # NAV uses real BOXX price now
        nav = s['cash'] + s['shares'] * spot_u + s['boxx'] * float((row.get('BOXX') if (row.get('BOXX') == row.get('BOXX') and row.get('BOXX') is not None) else 117.0)) + s['kold'] * spot_k + long_calls_mtm(s.get('long_calls'), spot_u, idx, iv_at)
        def _book(_legs):   # real strike→contracts tally (no reconstruction leak)
            _b = {}
            for _x in _legs:
                _k = round(float(_x.get('K', 0)), 1)
                _b[_k] = _b.get(_k, 0) + int(_x.get('qty', 0))
            return _b
        _ndh, _ngh = book_greeks(s, spot_u, iv_at)   # book net delta/gamma (share-equiv)
        history.append({
            'date': idx, 'spot': spot_u, 'z': z, 'regime': r,
            'cash': s['cash'], 'shares': s['shares'], 'boxx': s['boxx'], 'kold': s['kold'],
            'nav': nav, 'short_puts': len(s['short_puts']), 'short_calls': len(s['short_calls']),
            'put_book': _book(s['short_puts']), 'call_book': _book(s['short_calls']),
            'net_delta': round(_ndh, 0), 'net_gamma': round(_ngh, 1),
        })

    if live_decision and seed_state is not None:
        # Return ONLY the orders the engine decided for today (today's seeded state),
        # and stash the POST-decision book so callers can compute after-action theta.
        _mark = locals().get('_live_trade_mark', len(trades))
        today_orders = trades[_mark:]
        globals()['_LIVE_FINAL'] = {
            'short_puts': list(s['short_puts']), 'short_calls': list(s['short_calls']),
            'long_puts': list(s.get('long_puts', [])), 'long_calls': list(s.get('long_calls', [])),
            'shares': s['shares'], 'cash': s['cash'], 'kold': s['kold'],
        }
        return pd.DataFrame(history), pd.DataFrame(today_orders)
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
        'use_real_strikes': True,
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
    # Real-IV variant of aggressive_z — uses PG ung_iv_surface table
    'champion_aggressive_z_real_iv': {
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
        'use_real_iv_surface': True,  # NEW: pull IV from PG when available
    },
    # NEW HARNESS WINNER: aggressive z-target mults push Sharpe to 1.86
    # Counter-intuitive: trimming MORE at rich + loading MORE at cheap
    # IMPROVES everything (Sharpe, return, MDD all better). Cash from
    # rich trims funds bigger accumulation at lows.
    'champion_aggressive_z': {
        'use_real_strikes': True,
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
        'use_real_strikes': True,
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
        'use_real_strikes': True,
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
        'use_real_strikes': True,
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
        'use_real_strikes': True,
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
    # NEW CHAMPION: hits all 3 targets (ann ≥25%, Sharpe ≥2.0, MDD ≤10%)
    # daily entry cadence + larger size buffer
    'champion_target_25': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 18, 'call_qty': 15,
        'entry_cadence': 1,  # daily layering instead of weekly
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
    # SMOOTH variant — continuous tanh-based z-mult eliminates bucket jumps.
    # Lower daily NAV vol, easier to manage psychologically + operationally.
    'champion_target_25_smooth': {
        'use_real_strikes': True,
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 18, 'call_qty': 15,
        'entry_cadence': 1,
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
        'smooth_z_target': True,  # tanh continuous interpolation
        'z_share_target_base': 6200,
        'kold_shoulder_hedge': 0.10,
        'cut_and_rebuild_puts': True, 'rebuild_put_otm_pct': 0.10, 'rebuild_put_dte': 45,
        'elevator_skip_on_momentum': True,
        'itm_cc_skip_on_momentum': True,
        'dd_trim_trigger_pct': -8, 'dd_trim_qty_pct': 30,
        'dd_trim_floor': 0, 'dd_trim_cadence_days': 5,
    },
    # ITM-PUT PREMIUM HARVEST — high premium, less shares.
    # EVOLUTIONS BAKED IN (day-by-day analyzer iterations):
    # - V1: put_qty 18→12 (less NEUTRAL noise-zone leverage)
    # - V2: z_target_cadence_days 14→21 (slower regime response)
    # - V3 (this): put_qty 12→21 (was flooring to 1 at daily cadence) +
    #   regime_aware_put_qty=True (smaller in NEUTRAL noise, bigger at extreme z)
    #   WF worst Sharpe +2.13→+2.35, range 72→54pp, avg ann -1.4pp tradeoff
    'champion_premium_harvest': {
        'otm_put': 0.05, 'otm_call': 0.05, 'put_qty': 21, 'call_qty': 12,
        'regime_aware_put_qty': True,
        'entry_cadence': 1,
        'tp_50': True, 'tp_dynamic': True,
        'roll_down': True, 'roll_up_calls': True,
        'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        # Aggressive ITM on BOTH sides
        'aggressive_itm_cc_z': -0.25, 'itm_cc_pct': -0.20,
        'aggressive_itm_put_z': 0.3,   # NEW: sell ITM put when z > +0.3 (RICH-ish)
        'itm_put_pct': -0.05,           # NEW: 5% ITM by default
        'elevator_close': True, 'elevator_itm_pct': 0.05,
        'elevator_extrinsic_max': 0.15, 'elevator_mode': 'strict',
        'vol_aware_sizing': True,
        'tail_hedge_floor': 2,
        # Smaller share inventory
        'z_share_target_enabled': True, 'z_target_cadence_days': 21,  # V2: 14→21
        'z_target_mults': {
            'extreme_cheap': 1.7, 'cheap': 1.4, 'neutral': 1.0,
            'rich': 0.4, 'extreme_rich': 0.1,
        },
        'z_share_target_base': 3000,    # smaller (was 6200)
        'kold_shoulder_hedge': 0.10,
        'cut_and_rebuild_puts': True, 'rebuild_put_otm_pct': 0.08, 'rebuild_put_dte': 30,
        'elevator_skip_on_momentum': True,
        'itm_cc_skip_on_momentum': True,
        'dd_trim_trigger_pct': -5, 'dd_trim_qty_pct': 30,
        'dd_trim_floor': 0, 'dd_trim_cadence_days': 5,
        'cc_aware_cut': True,
        'skip_puts_on_grind_down': True,
    },
    # SCALE-INVARIANT — same annualized return regardless of starting NAV.
    # All sizing as % of NAV: put_qty via put_qty_nav_pct, share base via
    # z_share_target_pct_nav. Tests user's principle: a good algo's return
    # shouldn't depend heavily on starting capital.
    'champion_premium_harvest_scale_invariant': {
        'use_real_strikes': True,
        'otm_put': 0.05, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,  # floor only
        'entry_cadence': 1,
        'tp_50': True, 'tp_dynamic': True,
        'roll_down': True, 'roll_up_calls': True,
        'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'aggressive_itm_cc_z': -0.25, 'itm_cc_pct': -0.20,
        'aggressive_itm_put_z': 0.3, 'itm_put_pct': -0.05,
        'elevator_close': True, 'elevator_itm_pct': 0.05,
        'elevator_extrinsic_max': 0.15, 'elevator_mode': 'strict',
        'vol_aware_sizing': True,
        'tail_hedge_floor': 2,
        'z_share_target_enabled': True, 'z_target_cadence_days': 21,
        'z_target_mults': {
            'extreme_cheap': 1.7, 'cheap': 1.4, 'neutral': 1.0,
            'rich': 0.4, 'extreme_rich': 0.1,
        },
        'z_share_target_pct_nav': 0.35,  # 35% of NAV in shares at NEUTRAL z
        'kold_shoulder_hedge': 0.10,
        'cut_and_rebuild_puts': True, 'rebuild_put_otm_pct': 0.08, 'rebuild_put_dte': 30,
        'elevator_skip_on_momentum': True, 'itm_cc_skip_on_momentum': True,
        'dd_trim_trigger_pct': -5, 'dd_trim_qty_pct': 30,
        'dd_trim_floor': 0, 'dd_trim_cadence_days': 5,
        'cc_aware_cut': True, 'skip_puts_on_grind_down': True,
        'regime_aware_put_qty': True,
        # NAV-relative put/call sizing
        'put_qty_nav_pct': 0.06,   # 6% of NAV in put collateral per cycle
        'call_qty_nav_pct': 0.04,
        'put_qty_max': 100, 'call_qty_max': 50,
    },
    # SAME AS scale_invariant + HH backwardation storm filter.
    # When hh_basis > +$0.40, skip new puts (don't get assigned into the
    # mean-reversion crash). Tested edge: 41 events × -4.5% UNG 5d.
    'champion_premium_harvest_scale_invariant_hh_storm': {
        'use_real_strikes': True,
        'otm_put': 0.05, 'otm_call': 0.05, 'put_qty': 5, 'call_qty': 5,
        'entry_cadence': 1,
        'tp_50': True, 'tp_dynamic': True,
        'roll_down': True, 'roll_up_calls': True,
        'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'aggressive_itm_cc_z': -0.25, 'itm_cc_pct': -0.20,
        'aggressive_itm_put_z': 0.3, 'itm_put_pct': -0.05,
        'elevator_close': True, 'elevator_itm_pct': 0.05,
        'elevator_extrinsic_max': 0.15, 'elevator_mode': 'strict',
        'vol_aware_sizing': True,
        'tail_hedge_floor': 2,
        'z_share_target_enabled': True, 'z_target_cadence_days': 21,
        'z_target_mults': {
            'extreme_cheap': 1.7, 'cheap': 1.4, 'neutral': 1.0,
            'rich': 0.4, 'extreme_rich': 0.1,
        },
        'z_share_target_pct_nav': 0.35,
        'kold_shoulder_hedge': 0.10,
        'cut_and_rebuild_puts': True, 'rebuild_put_otm_pct': 0.08, 'rebuild_put_dte': 30,
        'elevator_skip_on_momentum': True, 'itm_cc_skip_on_momentum': True,
        'dd_trim_trigger_pct': -5, 'dd_trim_qty_pct': 30,
        'dd_trim_floor': 0, 'dd_trim_cadence_days': 5,
        'cc_aware_cut': True, 'skip_puts_on_grind_down': True,
        'regime_aware_put_qty': True,
        'put_qty_nav_pct': 0.06,
        'call_qty_nav_pct': 0.04,
        'put_qty_max': 100, 'call_qty_max': 50,
        # NEW: backwardation storm defensive
        'hh_storm_skip_puts': True,
    },
    # ULTRA-CONSERVATIVE — tightest WF range ever (51pp). Trades return
    # for ultimate predictability across rolling windows.
    'champion_premium_harvest_ultra': {
        'use_real_strikes': True,
        'otm_put': 0.05, 'otm_call': 0.05, 'put_qty': 12, 'call_qty': 12,
        'entry_cadence': 2,  # bi-daily (less noise)
        'tp_50': True, 'tp_dynamic': False, 'tp_threshold': 0.4,  # faster exit
        'roll_down': True, 'roll_up_calls': True,
        'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'aggressive_itm_cc_z': -0.25, 'itm_cc_pct': -0.20,
        'aggressive_itm_put_z': 0.3, 'itm_put_pct': -0.05,
        'elevator_close': True, 'elevator_itm_pct': 0.05,
        'elevator_extrinsic_max': 0.15, 'elevator_mode': 'strict',
        'vol_aware_sizing': True,
        'tail_hedge_floor': 2,
        'z_share_target_enabled': True, 'z_target_cadence_days': 21,
        'z_target_mults': {
            'extreme_cheap': 1.7, 'cheap': 1.4, 'neutral': 1.0,
            'rich': 0.4, 'extreme_rich': 0.1,
        },
        'z_share_target_base': 3000,
        'kold_shoulder_hedge': 0.10,
        'cut_and_rebuild_puts': True, 'rebuild_put_otm_pct': 0.08, 'rebuild_put_dte': 30,
        'elevator_skip_on_momentum': True, 'itm_cc_skip_on_momentum': True,
        'dd_trim_trigger_pct': -5, 'dd_trim_qty_pct': 30,
        'dd_trim_floor': 0, 'dd_trim_cadence_days': 5,
        'cc_aware_cut': True,
        'skip_puts_on_grind_down': True,
        'grind_tp_accelerate': True,
    },
    # WALK-FORWARD HARDENED — tightest dd_trim + cc_aware_cut + smaller put size
    # Designed specifically to survive 12mo rolling windows (target worst MDD > -15%)
    'champion_target_25_walkforward_safe': {
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 12, 'call_qty': 10,  # smaller
        'entry_cadence': 1,
        'tp_50': True, 'tp_dynamic': True,
        'roll_down': True, 'roll_up_calls': True,
        'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'aggressive_itm_cc_z': -0.25, 'itm_cc_pct': -0.15,  # less ITM
        'elevator_close': True, 'elevator_itm_pct': 0.05,
        'elevator_extrinsic_max': 0.15, 'elevator_mode': 'strict',
        'vol_aware_sizing': True,
        'tail_hedge_floor': 2,
        'z_share_target_enabled': True, 'z_target_cadence_days': 14,  # faster rebal
        'z_target_mults': {
            'extreme_cheap': 1.8, 'cheap': 1.4, 'neutral': 1.0,
            'rich': 0.5, 'extreme_rich': 0.2,  # slightly less aggressive
        },
        'z_share_target_base': 6200,
        'kold_shoulder_hedge': 0.12,  # slightly heavier seasonal hedge
        'cut_and_rebuild_puts': True, 'rebuild_put_otm_pct': 0.10, 'rebuild_put_dte': 45,
        'elevator_skip_on_momentum': True,
        'itm_cc_skip_on_momentum': True,
        # Tighter DD-trim: catch DD earlier (-5% vs -8%) + cc_aware_cut implicit
        'dd_trim_trigger_pct': -5, 'dd_trim_qty_pct': 25,
        'dd_trim_floor': 0, 'dd_trim_cadence_days': 5,
        'cc_aware_cut': True,  # close near-DTE CCs first to free shares for cut
        # NEW: stop adding put exposure during slow chronic declines (Dec 2023-style)
        # — catches the bad windows that detect_anomaly misses (12 of 15 worst-DD
        # days were NORMAL anomaly). Marginal: tighter range, -0.7pp avg ann.
        'skip_puts_on_grind_down': True,
    },
    # DD-TRIM variant — best Sharpe variant from walk-forward (2.73 on full sample)
    # Cuts worst 12mo MDD from -24% to -17% with almost no return cost.
    'champion_target_25_dd_trim': {
        'use_real_strikes': True,
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 18, 'call_qty': 15,
        'entry_cadence': 1,
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
        # NEW: reactive DD-trim catches the 2023-2024 bad windows
        'dd_trim_trigger_pct': -8, 'dd_trim_qty_pct': 30,
        'dd_trim_floor': 0, 'dd_trim_cadence_days': 5,
    },
    # MAX-PROTECTED variant — sacrifices ~10pp ann for worst-window MDD -15%
    # Walk-forward worst 12mo MDD: -15% (vs -24% baseline)
    'champion_target_25_max_protected': {
        'use_real_strikes': True,
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 18, 'call_qty': 15,
        'entry_cadence': 1,
        'tp_50': True, 'tp_dynamic': True,
        'roll_down': True, 'roll_up_calls': False,  # disable based on bad-window finding
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
        'dd_trim_trigger_pct': -6, 'dd_trim_qty_pct': 35,
        'dd_trim_floor': 0, 'dd_trim_cadence_days': 5,
    },
    # NAV-AWARE variant — sizes puts/calls as % of NAV (capital-scale invariant)
    # Use this if your account is ~$100K-200K (puts sized to your actual cash)
    'champion_target_25_nav_aware': {
        'use_real_strikes': True,
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 8, 'call_qty': 7,
        'entry_cadence': 1,
        'tp_50': True, 'tp_dynamic': True,
        'roll_down': True,
        # roll_up_calls intentionally DROPPED — ablation showed it hurts Sharpe
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
        # NEW: NAV-relative sizing
        'put_qty_nav_pct': 0.06,   # 6% of NAV in put notional per cycle
        'call_qty_nav_pct': 0.045, # 4.5% of NAV in call notional per cycle
        'put_qty_max': 50,
        'call_qty_max': 50,
    },
    # CASH-START variant — best Sharpe (2.96) + best MDD (-4.4%) per init sweep
    # Use this if starting fresh with all cash; no initial share exposure
    'champion_target_25_cash_start': {
        'use_real_strikes': True,
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 8, 'call_qty': 7,
        'entry_cadence': 1,
        'tp_50': True, 'tp_dynamic': True,
        'roll_down': True,
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
        'put_qty_nav_pct': 0.06,
        'call_qty_nav_pct': 0.045,
        'put_qty_max': 50,
        'call_qty_max': 50,
    },
    # WINDOW-SAFE variant — tighter risk controls for rolling-window MDD safety
    # Trades some return for limiting 12mo MDD < -10% in walk-forward
    'champion_target_25_window_safe': {
        'use_real_strikes': True,
        'otm_put': 0.10, 'otm_call': 0.05, 'put_qty': 12, 'call_qty': 10,
        'entry_cadence': 2,  # bi-daily (less leverage than daily=1)
        'tp_50': True, 'tp_dynamic': True,
        'roll_down': True,
        'bearish_stack': True, 'boxx': True,
        'trend_aware_roll': True,
        'aggressive_itm_cc_z': -0.25, 'itm_cc_pct': -0.15,  # softer ITM
        'elevator_close': True, 'elevator_itm_pct': 0.05,
        'elevator_extrinsic_max': 0.15, 'elevator_mode': 'strict',
        'vol_aware_sizing': True,
        'tail_hedge_floor': 3,  # MORE long puts as insurance
        'z_share_target_enabled': True, 'z_target_cadence_days': 14,  # faster rebal
        'z_target_mults': {
            'extreme_cheap': 1.7, 'cheap': 1.4, 'neutral': 1.0,
            'rich': 0.5, 'extreme_rich': 0.2,
        },
        'z_share_target_base': 6200,
        'kold_shoulder_hedge': 0.12,  # slightly more KOLD
        'cut_and_rebuild_puts': True, 'rebuild_put_otm_pct': 0.12, 'rebuild_put_dte': 45,
        'elevator_skip_on_momentum': True,
        'itm_cc_skip_on_momentum': True,
        'put_qty_nav_pct': 0.04,  # smaller put sizing
        'call_qty_nav_pct': 0.035,
        'put_qty_max': 40,
        'call_qty_max': 40,
    },
    # NEW: IV-SHAPE-AWARE — react to real surface term & skew (default real IV)
    'champion_aggressive_z_iv_shape': {
        'use_real_strikes': True,
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
        'iv_shape_sizing': True,  # NEW: react to PG surface skew + term
    },
}


# RETIRE old format. Keep only strategies that use real strikes (per-expiry
# snap to actually-listed strikes via PG ung_iv_surface + heuristic fallback).
# All remaining strategies use real IV surface (default) AND real strikes.
# This makes backtest decisions match live constraints exactly.
#
# Retired 2026-06-05 (15 strategies dropped — see git history for definitions):
#   naive_atm, otm_managed, beam_put_only, elevator_close_surprise (baselines)
#   kelly_firmness, kelly_short_dd_balanced, kelly_short_dd_minimal, kelly_z_target_winner
#   champion_20pct_plus_floor (no real strikes)
#   champion_aggressive_z_real_iv (superseded by champion_aggressive_z which has real strikes)
#   champion_cut_rebuild_fast, champion_cut_rebuild_slow
#   champion_premium_harvest (superseded by _scale_invariant which has real strikes)
#   champion_target_25 (superseded by _smooth / _dd_trim / _window_safe)
#   champion_target_25_walkforward_safe (no real strikes; superseded by _window_safe)
# ── NEW CANDIDATES 2026-06-11 ("sweet kernels") — clones of validated
# champions with one new knob each, so attribution is clean.
_psi = STRATEGIES['champion_premium_harvest_scale_invariant']
_smooth = STRATEGIES.get('champion_target_25_smooth', _psi)
STRATEGIES['champion_psi_gex'] = {**_psi, 'cc_gex_floor': True}
STRATEGIES['champion_psi_fasttp'] = {**_psi, 'tp_dynamic': False,
                                     'tp_threshold': 0.30}
STRATEGIES['champion_psi_kold15'] = {**_psi, 'kold_shoulder_hedge': 0.15}
STRATEGIES['champion_smooth_ddtrim'] = {**_smooth,
                                        'dd_trim_trigger_pct': -4,
                                        'dd_trim_qty_pct': 40,
                                        'dd_trim_cadence_days': 5}
STRATEGIES['champion_smooth_gex'] = {**_smooth, 'cc_gex_floor': True}
STRATEGIES['champion_psi_ivrank'] = {**_psi, 'iv_rank_z_scale': True}
STRATEGIES['champion_kold15_ivrank'] = {**_psi, 'kold_shoulder_hedge': 0.15,
                                        'iv_rank_z_scale': True}
# PROMOTED 2026-06-14 (gen-8): champion + KOLD book hedge (all 4 rigor
# criteria passed: confound-free +0.32 OOS Sharpe, -1.9pp MDD, -0.3pp
# return, bootstrap-significant). Standing overlay on the uncovered book.
STRATEGIES['champion_kold15_ivrank_kbh'] = {**_psi, 'kold_shoulder_hedge': 0.15,
                                            'iv_rank_z_scale': True,
                                            'kold_book_hedge': True,
                                            'kold_book_frac': 0.5}
STRATEGIES['champion_smooth_ddtrim_ivrank'] = {**_smooth,
                                               'dd_trim_trigger_pct': -4,
                                               'dd_trim_qty_pct': 40,
                                               'dd_trim_cadence_days': 5,
                                               'iv_rank_z_scale': True}

# ── GEN-3 (2026-06-12): ALL session learnings, REAL-FILL model on every
# entrant (same fill basis = fair comparison). Knobs: empirical fill grid
# (real bid/BSM by dte/otm), Thursday put entries (-40bps print-day bleed),
# Tuesday share adds (-40bps overnight), TOM add-skip (NG expiry churn),
# kold15, iv_rank scaling, tight dd_trim.
_RF = {'real_fill_model': True}
_TIMING = {'put_entry_dow': 3, 'share_add_dow': 1, 'avoid_tom_adds': True}
STRATEGIES['g3_psi_rf'] = {**_psi, **_RF}                       # baseline w/ real fills
STRATEGIES['g3_kold15_ivrank_rf'] = {**_psi, 'kold_shoulder_hedge': 0.15,
                                     'iv_rank_z_scale': True, **_RF}
STRATEGIES['g3_timing_rf'] = {**_psi, **_RF, **_TIMING}         # timing attribution
STRATEGIES['g3_full_stack'] = {**_psi, 'kold_shoulder_hedge': 0.15,
                               'iv_rank_z_scale': True, **_RF, **_TIMING}
STRATEGIES['g3_smooth_ddtrim_rf'] = {**_smooth, 'dd_trim_trigger_pct': -4,
                                     'dd_trim_qty_pct': 40,
                                     'dd_trim_cadence_days': 5,
                                     'iv_rank_z_scale': True, **_RF, **_TIMING}

# ── GEN-4: forensic knobs from KERNEL_LAB.md, one per clone, on the
# g3_full_stack base (attribution ladder against it)
_G4B = STRATEGIES['g3_full_stack']
STRATEGIES['g4_rollguards'] = {**_G4B, 'roll_accept_cheap_z': True,
                               'max_rolls_per_chain': 1,
                               'roll_stagger_max_per_day': 3}
STRATEGIES['g4_elevator25'] = {**_G4B, 'elevator_extrinsic_max': 0.25}
STRATEGIES['g4_tp_ivrank'] = {**_G4B, 'tp_by_iv_rank': True}
STRATEGIES['g4_dd_ivgate'] = {**_G4B, 'dd_trim_iv_gate': True}
STRATEGIES['g4_spike_defer'] = {**_G4B, 'roll_up_spike_defer_days': 2}
STRATEGIES['g4_everything'] = {**_G4B, 'roll_accept_cheap_z': True,
                               'max_rolls_per_chain': 1,
                               'roll_stagger_max_per_day': 3,
                               'elevator_extrinsic_max': 0.25,
                               'tp_by_iv_rank': True,
                               'dd_trim_iv_gate': True,
                               'roll_up_spike_defer_days': 2}

# ── GEN-5 (2026-06-13): on the PROMOTED kernel (champion_kold15_ivrank).
# (1) distributional delta band — the rigidity fix, k in {0.5,1.0,1.5};
# (2) g4 forensic knobs re-tested on the UNCRIPPLED promoted base (gen-4
#     ran them on the timing-crippled stack); (3) fair timing test
#     (weekly cadence any-day vs Thursday — the gen-3 test was
#     frequency-confounded). Real fills on every entrant.
_PROMO = STRATEGIES['champion_kold15_ivrank']
STRATEGIES['g5_band_k05'] = {**_PROMO, 'real_fill_model': True,
                             'delta_band_sizing': True, 'delta_band_k': 0.5}
STRATEGIES['g5_band_k10'] = {**_PROMO, 'real_fill_model': True,
                             'delta_band_sizing': True, 'delta_band_k': 1.0}
STRATEGIES['g5_band_k15'] = {**_PROMO, 'real_fill_model': True,
                             'delta_band_sizing': True, 'delta_band_k': 1.5}
STRATEGIES['g5_promo_rf'] = {**_PROMO, 'real_fill_model': True}   # baseline
STRATEGIES['g5_rollguards'] = {**_PROMO, 'real_fill_model': True,
                               'roll_accept_cheap_z': True,
                               'max_rolls_per_chain': 1,
                               'roll_stagger_max_per_day': 3}
STRATEGIES['g5_dd_ivgate'] = {**_PROMO, 'real_fill_model': True,
                              'dd_trim_iv_gate': True}
STRATEGIES['g5_tp_ivrank'] = {**_PROMO, 'real_fill_model': True,
                              'tp_by_iv_rank': True}
STRATEGIES['g5_best_combo'] = {**_PROMO, 'real_fill_model': True,
                               'delta_band_sizing': True, 'delta_band_k': 1.0,
                               'roll_accept_cheap_z': True,
                               'max_rolls_per_chain': 1,
                               'dd_trim_iv_gate': True}
STRATEGIES['g5_timing_weekly'] = {**_PROMO, 'real_fill_model': True,
                                  'entry_cadence': 5}          # any-day weekly
STRATEGIES['g5_timing_thu'] = {**_PROMO, 'real_fill_model': True,
                               'entry_cadence': 5, 'put_entry_dow': 3}  # Thursday weekly

# ── GEN-6 (2026-06-13): CONVICTION-SCALED delta band — narrow at
# extremes (recover return), wide at neutral (keep MDD benefit). Coarse
# 3x3 (a=neutral-widen, b=disagreement) sweep + carry the g5 baselines
# for direct comparison. real fills. OOS gate mandatory (2 free params).
_CB = {**STRATEGIES['champion_kold15_ivrank'], 'real_fill_model': True,
       'delta_band_sizing': True, 'delta_band_k': 1.0, 'conviction_band': True}
for _a in (0.20, 0.35, 0.50):
    for _b in (0.10, 0.20, 0.30):
        STRATEGIES[f'g6_cb_a{int(_a*100)}_b{int(_b*100)}'] = {
            **_CB, 'conviction_a': _a, 'conviction_b': _b}
# tighter floor variant (act even harder at extremes)
STRATEGIES['g6_cb_tightfloor'] = {**_CB, 'conviction_a': 0.35,
                                  'conviction_b': 0.20, 'conviction_floor': 0.02}

# ── GEN-7 (2026-06-13): BOOK HEDGES — keep the shares, hedge the 84%-
# uncovered book that causes the drawdowns. One knob per clone on the
# PROMOTED kernel + real fills, so each hedge's effect is clean. Goal:
# the band's low MDD WITHOUT the band's return cost.
_H = {**STRATEGIES['champion_kold15_ivrank'], 'real_fill_model': True}
STRATEGIES['g7_funded_collar']  = {**_H, 'funded_collar': True,
                                   'collar_cover_frac': 0.5, 'collar_fund_ratio': 1.0}
STRATEGIES['g7_collar_aggr']    = {**_H, 'funded_collar': True,
                                   'collar_cover_frac': 0.7, 'collar_fund_ratio': 1.5}
STRATEGIES['g7_scaled_floor']   = {**_H, 'scaled_put_floor': True, 'floor_cover_frac': 0.3}
STRATEGIES['g7_scaled_floor_hi']= {**_H, 'scaled_put_floor': True, 'floor_cover_frac': 0.5}
STRATEGIES['g7_kold_bookhedge'] = {**_H, 'kold_book_hedge': True, 'kold_book_frac': 0.5}
STRATEGIES['g7_combo_collar_kold'] = {**_H, 'funded_collar': True, 'collar_cover_frac': 0.5,
                                      'kold_book_hedge': True, 'kold_book_frac': 0.3}
STRATEGIES['g7_baseline_rf']    = {**_H}   # promoted kernel, real fills, no hedge

# ── GEN-8 (2026-06-14): CONTROLLED hedge test — matched share book.
# hedge_sizing_neutral makes share-sizing ignore KOLD P&L so the hedge
# and baseline run near-identical share paths; the ONLY difference is the
# KOLD overlay → isolates hedge effect from the exposure confound.
STRATEGIES['g8_kold_matched']   = {**_H, 'kold_book_hedge': True,
                                   'kold_book_frac': 0.5, 'hedge_sizing_neutral': True}
STRATEGIES['g8_baseline_matched'] = {**_H, 'hedge_sizing_neutral': True}  # same sizing, no hedge
STRATEGIES['g8_kold_light']     = {**_H, 'kold_book_hedge': True,
                                   'kold_book_frac': 0.25, 'hedge_sizing_neutral': True}

# ── GEN-9 (2026-06-15): SMOOTH RETURN-ENGINE RECOVERY. User priority =
# high return + Sharpe, MDD secondary. smooth posted +32.7%/1.79 model
# fills but only +19.0% real (fill-fragile: leans on premium volume the
# 35-45 DTE haircut punishes). LEVER: 60-DTE fills are neutral-positive
# (1.0-1.08x) → move smooth's premium there to recover the haircut. Plus
# conviction amplifier (size up at extreme-cheap-z + low IV = return) and
# IV-rank scaling. Real fills + real strikes on every entrant. The audit
# gate (audit.py) is the HARD promotion gate.
_SM = {**STRATEGIES['champion_target_25_smooth'],
       'use_real_strikes': True, 'real_fill_model': True}
STRATEGIES['g9_smooth_rf']      = {**_SM}                       # baseline: shows fill fragility
STRATEGIES['g9_smooth_60d']     = {**_SM, 'open_dte': 60}       # recover the haircut
STRATEGIES['g9_smooth_60d_iv']  = {**_SM, 'open_dte': 60, 'iv_rank_z_scale': True}
STRATEGIES['g9_smooth_60d_conv']= {**_SM, 'open_dte': 60, 'iv_rank_z_scale': True,
                                   'conviction_amplify': True, 'conviction_amplify_mult': 1.4}
STRATEGIES['g9_smooth_full']    = {**_SM, 'open_dte': 60, 'iv_rank_z_scale': True,
                                   'conviction_amplify': True, 'conviction_amplify_mult': 1.5,
                                   'kold_shoulder_hedge': 0.15}

# ── GEN-10 (2026-06-15): the GEN-8 ANGLE to return. smooth got return
# from UNHEDGED aggressive sizing → Sharpe collapsed OOS. Gen-8 proved
# hedging the book lets you hold MORE safely (Sharpe UP, DD down). So push
# return via a LARGER book + smooth's continuous engine, but PROTECTED by
# the KOLD hedge AND the champion's risk controls (which smooth lacked).
# All real fills (default), 60-DTE. Criterion: max return at Sharpe>=2.0 OOS.
_KBH = STRATEGIES['champion_kold15_ivrank_kbh']
STRATEGIES['g10_base']         = {**_KBH, 'open_dte': 60}        # frontier base, real fills
STRATEGIES['g10_book45']       = {**_KBH, 'open_dte': 60, 'z_share_target_pct_nav': 0.45}
STRATEGIES['g10_book55']       = {**_KBH, 'open_dte': 60, 'z_share_target_pct_nav': 0.55}
STRATEGIES['g10_book45_h6']    = {**_KBH, 'open_dte': 60, 'z_share_target_pct_nav': 0.45,
                                  'kold_book_frac': 0.6}        # more hedge for more book
STRATEGIES['g10_conv']         = {**_KBH, 'open_dte': 60, 'conviction_amplify': True,
                                  'conviction_amplify_mult': 1.4}
STRATEGIES['g10_smoothz']      = {**_KBH, 'open_dte': 60, 'smooth_z_target': True}  # smooth engine, hedged
STRATEGIES['g10_smoothz_book45']= {**_KBH, 'open_dte': 60, 'smooth_z_target': True,
                                   'z_share_target_pct_nav': 0.45, 'kold_book_frac': 0.6}
STRATEGIES['g10_full']         = {**_KBH, 'open_dte': 60, 'smooth_z_target': True,
                                  'z_share_target_pct_nav': 0.45, 'kold_book_frac': 0.6,
                                  'conviction_amplify': True, 'conviction_amplify_mult': 1.4}

# GEN-11 ANGLE A — conviction ITM put (bullish expression; return from cushioned
# basis, not extra delta). Built on the champion to isolate the structure knob,
# and on g10_base (60-DTE) to stack the proven-clean fill edge.
STRATEGIES['g11_itmput_conv']  = {**_KBH, 'conviction_itm_put': True}   # default: z<-1, depth 6%, ivr<0.4
STRATEGIES['g11_itmput_deep']  = {**_KBH, 'conviction_itm_put': True,
                                  'conviction_itm_depth': 0.10}          # deeper ITM cushion
STRATEGIES['g11_itmput_wide']  = {**_KBH, 'conviction_itm_put': True,
                                  'conviction_itm_z': -0.5}              # fires more often (milder cheap)
STRATEGIES['g11_itmput_60d']   = {**_KBH, 'open_dte': 60,
                                  'conviction_itm_put': True}            # + proven 60-DTE fill edge

# GEN-11 ANGLE B — ITM covered-call DIVEST (bearish; monetize rich price, exit
# into strength). Covered 1:1 by construction. Built on champion to isolate.
STRATEGIES['g11_itmcc_divest'] = {**_KBH, 'itm_cc_divest': True}        # default z>0.75, surge>0.5, 8% ITM
STRATEGIES['g11_itmcc_eager']  = {**_KBH, 'itm_cc_divest': True,
                                  'itm_cc_divest_z': 0.5}                # divest earlier (milder rich)
STRATEGIES['g11_itmcc_deep']   = {**_KBH, 'itm_cc_divest': True,
                                  'itm_cc_divest_pct': -0.12}            # deeper ITM, near-certain assign

# GEN-11 C1 — COVERED call backspread (bullish-convex). Short CC share-covered;
# 2x long calls on top = net long convexity for vol-expansion / trend-up.
STRATEGIES['g11_backspread']      = {**_KBH, 'call_backspread': True}   # ratio2, longs15%OTM, z<-0.5+surge>0
STRATEGIES['g11_backspread_wide'] = {**_KBH, 'call_backspread': True,
                                     'backspread_long_otm': 0.20}        # cheaper, more convex longs
STRATEGIES['g11_backspread_3x']   = {**_KBH, 'call_backspread': True,
                                     'backspread_long_ratio': 3}         # more convexity per short
STRATEGIES['g11_backspread_deep'] = {**_KBH, 'call_backspread': True,
                                     'backspread_z_max': -1.0}           # only deep-cheap (higher conviction)

# GEN-11 C2 — COVERED upside-tail ratio (neutral-income). Closer CC strike for
# more premium + 1 long per 2 shorts as tail cap; all shorts share-covered.
STRATEGIES['g11_covratio']      = {**_KBH, 'cc_tail_ratio': True}        # z neutral<0.5, IVR>0.5, CC@4%, tail@12%
STRATEGIES['g11_covratio_rich'] = {**_KBH, 'cc_tail_ratio': True,
                                   'cc_tail_ivr': 0.6, 'cc_tail_call_otm': 0.03}  # only high IVR, sell closer
STRATEGIES['g11_covratio_wide'] = {**_KBH, 'cc_tail_ratio': True,
                                   'cc_tail_z': 0.75}                     # fire on a wider neutral band

# GEN-11 C3 — cash-secured PUT ratio (bullish, defined-risk accumulation). Sell
# more cash-secured puts on deep-cheap+momentum, buy 1 long put per 2 as a floor.
STRATEGIES['g11_putratio']      = {**_KBH, 'put_ratio': True}            # z<-1, surge>0, qty*1.5, floor@12%
STRATEGIES['g11_putratio_tight']= {**_KBH, 'put_ratio': True,
                                   'put_ratio_floor_otm': 0.08}           # tighter (more expensive) floor
STRATEGIES['g11_putratio_big']  = {**_KBH, 'put_ratio': True,
                                   'put_ratio_qty_mult': 2.0}             # full short-2 (more accumulation)

# GEN-11 ROUTER — signal->structure brain: stack the compliant winners (A,B,C2,C3),
# each fires ONLY in its own regime (gates non-overlapping by design). C1 backspread
# EXCLUDED (rejected: convexity redundant on a share book). C3 is the return engine
# (+2.6pp OOS); router_safe gates its 2x to NORMAL regime to tame spike-crash DD.
_ROUTER = {**_KBH, 'conviction_itm_put': True,   # A: deep-cheap + low-IVR
                   'itm_cc_divest': True, 'itm_cc_divest_z': 0.5,  # B: rich + hot
                   'cc_tail_ratio': True, 'cc_tail_z': 0.75,       # C2: neutral + high-IVR
                   'put_ratio': True}                              # C3: deep-cheap + momentum
STRATEGIES['g11_router']       = {**_ROUTER, 'put_ratio_qty_mult': 1.5}  # conservative C3
STRATEGIES['g11_router_big']   = {**_ROUTER, 'put_ratio_qty_mult': 2.0}  # aggressive C3 (chase +2.6pp)
STRATEGIES['g11_router_safe']  = {**_ROUTER, 'put_ratio_qty_mult': 2.0,
                                  'put_ratio_normal_only': True}    # 2x C3 only in NORMAL regime

# GEN-12 — BACKWARDATION-SPIKE DE-RISK (validated directional timing overlay):
# trim the share book on top-decile/5% HH-basis spikes (-3.7% fwd-5d, 70% down).
STRATEGIES['g12_bwd_derisk']     = {**_KBH, 'backwardation_derisk': True}                    # thr 0.33, trim 25%
STRATEGIES['g12_bwd_derisk_deep']= {**_KBH, 'backwardation_derisk': True,
                                    'bwd_derisk_trim_pct': 40}                               # trim harder
STRATEGIES['g12_bwd_derisk_d10'] = {**_KBH, 'backwardation_derisk': True,
                                    'bwd_derisk_thresh': 0.18}                               # top-decile (fires more)
STRATEGIES['g12_bwd_on_router']  = {**STRATEGIES['g11_router_safe'], 'backwardation_derisk': True}  # stack on best

# GEN-13 — WHEEL-ONLY share-book management (user insight: don't churn shares by
# direct buy/sell; let put/call ASSIGNMENT move the book — get PAID to enter/exit).
# Direct z-targeting round-trips on z noise (buy-high/sell-low + spread). Disabling
# it cuts churn ~99% AND improves return/Sharpe/DD in-sample.
STRATEGIES['g13_wheel_only']     = {**STRATEGIES['g11_router_safe'], 'z_share_target_enabled': False}
STRATEGIES['g13_wheel_ddtrim']   = {**STRATEGIES['g11_router_safe'], 'z_share_target_enabled': False,
                                    'dd_trim_trigger_pct': -25}   # + emergency-only direct de-risk
STRATEGIES['g13_wheel_bwd']      = {**STRATEGIES['g11_router_safe'], 'z_share_target_enabled': False,
                                    'backwardation_derisk': True}  # wheel + validated bwd de-risk

# GEN-14 — GAP-DRIVEN WHEEL (user design): z-target DELTA drives put sizing+strike+DTE
# (and CC divest) so the book steers to target via ASSIGNMENT, never direct churn.
STRATEGIES['g14_gap_wheel']      = {**STRATEGIES['g11_router_safe'], 'gap_to_wheel': True}
STRATEGIES['g14_gap_wheel_bwd']  = {**STRATEGIES['g11_router_safe'], 'gap_to_wheel': True,
                                    'backwardation_derisk': True}
# TIER-3 real-chain pricing (real historical bid/ask) to settle the fidelity question.
STRATEGIES['g14_gap_wheel_real'] = {**STRATEGIES['g11_router_safe'], 'gap_to_wheel': True,
                                    'real_chain_pricing': True}
# INTRADAY-EXECUTION champion: real minute fills + microstructure timing (15:00 window, avoid Thu pre-print)
STRATEGIES['champion_intraday'] = {**STRATEGIES['champion_kold15_ivrank_kbh'], 'intraday_exec': True, 'exec_window': 15, 'avoid_eia_print': True}
# Tail-hedge exploration: gap-wheel + super-OTM long-put crash insurance.
STRATEGIES['g15_gap_wheel_hedge']= {**STRATEGIES['g14_gap_wheel_real'], 'put_tail_hedge': True}
STRATEGIES['g15_hedge_deep']     = {**STRATEGIES['g14_gap_wheel_real'], 'put_tail_hedge': True,
                                    'put_tail_hedge_otm': 0.25, 'put_tail_hedge_ratio': 0.75}
# GEN-16 — ASSIGN instead of ROLL-DOWN (plug the -$992k PUT_ROLL_DOWN leak): when
# puts go ITM, let them ASSIGN (acquire shares at the chosen strike), then sell CCs.
STRATEGIES['g16_assign']         = {**STRATEGIES['g14_gap_wheel_real'], 'roll_down': False}
STRATEGIES['g16_assign_bwd']     = {**STRATEGIES['g14_gap_wheel_real'], 'roll_down': False,
                                    'backwardation_derisk': True}

# ── USER'S ACCUMULATE / DISTRIBUTE WHEEL (2026-06-17) ──────────────────────────
# The classical wheel the user actually designed (vs the champion's premium-scalping
# churn): STEADY-LADDER valuation accumulation (smooth z-target), FULL short puts that
# are LET ASSIGN low (no roll-away), CONTROLLED valuation-scaled CC that lets it run when
# cheap and invites call-away at the top, and MINIMAL churn — no TP scalping, no elevator,
# no upside-capping ITM-CC-when-cheap. Low turnover → fill-robust under honest minute fills.
STRATEGIES['accum_wheel'] = {**_psi,
    'smooth_z_target': True, 'z_target_cadence_days': 5,
    'z_target_mults': {'extreme_cheap': 1.6, 'cheap': 1.3, 'neutral': 1.0,
                       'rich': 0.6, 'extreme_rich': 0.3},
    'otm_put': 0.05, 'roll_down': False, 'cut_and_rebuild_puts': False,
    'tp_50': False, 'elevator_close': False,
    'aggressive_itm_cc_z': -99.0, 'itm_cc_pct': 0.0,   # no ITM-CC-when-cheap (let it run)
    'otm_call': 0.07, 'iv_rank_z_scale': True,
}
STRATEGIES['accum_wheel_tp'] = {**STRATEGIES['accum_wheel'], 'tp_50': True,
                                'tp_threshold': 0.5}  # variant: keep light TP for comparison
# REGIME SWITCH (user design): keep the champion's premium-harvest survival engine, but
# SEASONALLY tilt the valuation share-target — ACCUMULATE in shoulder (gated on not-falling-
# knife), DUMP into peak-demand season. Honors 'accumulate shoulder, dump during peak'
# without abandoning what survives UNG's secular grind.
STRATEGIES['seasonal_wheel'] = {**STRATEGIES['champion_kold15_ivrank_kbh'],
    'seasonal_regime': True,
    'accumulate_months': (3, 4, 5, 9), 'distribute_months': (11, 12, 1, 6, 7),
    'accumulate_boost': 1.5, 'knife_accum_cut': 0.6, 'distribute_cut': 0.4}
STRATEGIES['seasonal_wheel_lowchurn'] = {**STRATEGIES['seasonal_wheel'],
    'tp_threshold': 0.7, 'roll_down': False}   # lighter churn variant
# STATE-REGIME WHEEL (winner) — drift-aware accumulate/neutral/distribute from
# storage_surprise_z (storage vs SEASONAL norm, weekly) + momentum, on the lowchurn
# (let-assign + lighter-TP) base. OOS Sharpe 0.21→1.03, MDD -17→-11 vs champion.
STRATEGIES['regime_wheel'] = {**STRATEGIES['champion_kold15_ivrank_kbh'],
    'tp_threshold': 0.7, 'roll_down': False,
    'state_regime': True, 'accumulate_ssz': -0.5, 'distribute_ssz': 0.5,
    'accumulate_boost': 1.5, 'distribute_cut': 0.4, 'distribute_surge': -1.2}
# CONTINUOUS regime strength (quantified, Markov-filter): scale tilt by |s|.
STRATEGIES['regime_wheel_cont'] = {**STRATEGIES['champion_kold15_ivrank_kbh'],
    'tp_threshold': 0.7, 'roll_down': False, 'regime_continuous': True,
    'distribute_strength_max': 0.6, 'accumulate_strength_max': 0.5}
# WINNER: regime_wheel + BOXX cash-sweep (idle cash -> T-bill yield). OOS +11.3%/
# Sharpe 1.20/-9% MDD. (regime-KOLD tested + dropped: inverse-ETF decay > decline capture.)
STRATEGIES['regime_wheel_boxx'] = {**STRATEGIES['regime_wheel'],
    'boxx': True, 'boxx_sweep_full': True, 'boxx_cash_buffer': 15000,
    'intraday_exec': True, 'exec_window': 15, 'avoid_eia_print': True, 'tail_hedge_floor': 0}  # minute-fill by default (EOD-real retired)
# FAST live-decision variant for the dashboard (model fills → today's orders in ~15s; the
# execution advisor supplies the real minute pricing/ladder for the operator to act on).
STRATEGIES['regime_wheel_boxx_live'] = {**STRATEGIES['regime_wheel_boxx'],
    'intraday_exec': False, 'real_chain_pricing': False}
# DELTA-HEDGE variant: greeks-based book risk mgmt — buy long puts to pull net delta
# down in bearish regimes (effective bear spreads w/ existing shorts). +0.8pp OOS, Δ -22%.
STRATEGIES['regime_wheel_boxx_dh'] = {**STRATEGIES['regime_wheel_boxx'],
    'delta_hedge': True, 'delta_target_nav': 0.5, 'delta_bearish_cut': 0.9,
    'delta_hedge_rs_min': 0.25, 'delta_hedge_dte': 30, 'delta_hedge_max': 15}
# FULL greeks-managed: delta-band hedge + gamma-cap (anti-concentration). The complete
# bookwise risk layer — delta hedged when bearish, short-gamma capped per strike/expiry.
# KOLD KEPT ON (inherited from base regime_wheel): the KOLD-free A/B walk-forward (2026-06-18,
# verifiable fills, 6 windows) showed removing KOLD COSTS ~0.4 median Sharpe (1.63 vs 2.03)
# and ~3pp ann — KOLD earns its keep as crash/bear insurance (2022 crash + the two most recent
# windows). It was a greeks blind spot, so instead of dropping it we make the greeks engine
# KOLD-AWARE (live_kernel _book_greeks + delta compass count KOLD's ~-2x UNG-equiv delta).
STRATEGIES['regime_wheel_boxx_greeks'] = {**STRATEGIES['regime_wheel_boxx_dh'],
    'gamma_cap': True,
    # SCALE-INVARIANT per-strike concentration cap: limit single-strike short-option ASSIGNMENT
    # notional to ~8.5% of NAV → cap_contracts = 0.085·NAV/(K·100). Proportional to the account
    # (≈10 lots @ $11 on $133k, ≈77 @ $1M) instead of a fixed 10. max_short_per_strike kept as the
    # legacy fallback / floor when pct is unset. Forward-only (grandfathers existing legs).
    'max_short_pct_nav': 0.085, 'max_short_per_strike': 10}
# NOTE (2026-06-24): gamma_weighted_ladder=[14,30] was promoted then REVERTED. The audit showed the
# "+0.02 Sharpe" was TEST-SET-SELECTED (I picked the variant by its sealed-test score = invalid), a
# statistical tie, and gamma-weighting is near-nil at UNG's ~$11 scale (the only real effect was
# dropping the 7-DTE bucket). Single-[30] re-accumulation is the parsimonious champion. The
# gamma_weighted_ladder engine code remains, param-gated OFF, for any future wider-strike asset.
# FAST live variant of the promoted champion (model fills; advisor supplies real exec).
STRATEGIES['regime_wheel_boxx_greeks_live'] = {**STRATEGIES['regime_wheel_boxx_greeks'],
    'intraday_exec': False, 'real_chain_pricing': False,
    # LIVE multi-currency reality: the operator parks CAD as the collateral/margin reserve (for
    # selling puts + no-FX spending). BOXX is filled with USD CASH ONLY — never by borrowing USD
    # OPERATOR CHOICE: ALL-BOXX (buffer:0) + ITM put accumulation, CAD margin funds assignments.
    # The low buffer:0 backtest Sharpe (1.37) is a USD-MODEL ARTIFACT: with no CAD in the model, the
    # engine fakes assignment funding by CHURNING BOXX (sell→rebuy, paying the spread each cycle) — and
    # THAT churn, not the BOXX holding, is the ~0.6 Sharpe drag. In the real account CAD margin funds
    # assignments, BOXX is never sold, so there is no churn. The buffer:15k run (hold USD instead of
    # churning BOXX) = Sharpe 2.0 — the correct proxy for the CAD-funded reality. So real all-BOXX Sharpe
    # ≈ 2.0. Puts at +15% ITM (more share-like → less short-vol → higher Sharpe; buf15k proxy = 2.04).
    'boxx_cash_buffer': 0,
    'reaccum_via_puts': True, 'reaccum_put_dte': 30, 'reaccum_put_moneyness': 0.15}
# reaccum_via_puts (accumulate to target via slightly-ITM puts) was trialled here: standalone it is
# ~parity with shares (+5% ITM 30d ≈ 16-17%) BUT in the live buffer:0 config it backtests ~13% (vs
# shares 16.7%) — a USD-MODEL ARTIFACT, since the backtest can't represent CAD-financed puts and the
# all-cash-to-BOXX sweep churns against the (simulated USD) assignments. Engine default stays SHARES
# (the only cleanly-validated method); the operator fills the gap with CAD-financed ITM puts by hand
# (~parity, BOXX-preserving) using the dashboard's put-ladder execution aid. Code kept, param-gated off.
# v3: + confidence gate (skip accumulate when storage signal noisy) + price-breakdown
# distribute trigger (downtrend forces dump) — targets the 2022 walk-forward blind spot.
STRATEGIES['regime_wheel_boxx_v3'] = {**STRATEGIES['regime_wheel_boxx'],
    'regime_confidence_gate': True, 'ssz_vol_gate': 1.2, 'regime_downtrend_distribute': True}
# v4: + CRASH FALLBACK — deep price drawdown forces distribute + re-enables roll-down
# protection (champion-mode in crashes). Targets the 2022 -56pp blind spot directly.
STRATEGIES['regime_wheel_boxx_v4'] = {**STRATEGIES['regime_wheel_boxx_v3'],
    'crash_fallback': True, 'crash_dd': -0.18, 'crash_distribute_extra': 0.6}

_KEEP_STRATEGIES = {
    'accum_wheel', 'accum_wheel_tp', 'seasonal_wheel', 'seasonal_wheel_lowchurn', 'regime_wheel', 'regime_wheel_cont', 'regime_wheel_boxx', 'regime_wheel_boxx_live', 'regime_wheel_boxx_dh', 'regime_wheel_boxx_greeks', 'regime_wheel_boxx_greeks_live', 'regime_wheel_boxx_v3', 'regime_wheel_boxx_v4',
    'g11_itmput_conv', 'g11_itmput_deep', 'g11_itmput_wide', 'g11_itmput_60d',
    'g11_itmcc_divest', 'g11_itmcc_eager', 'g11_itmcc_deep',
    'g11_backspread', 'g11_backspread_wide', 'g11_backspread_3x', 'g11_backspread_deep',
    'g11_covratio', 'g11_covratio_rich', 'g11_covratio_wide',
    'g11_putratio', 'g11_putratio_tight', 'g11_putratio_big',
    'g11_router', 'g11_router_big', 'g11_router_safe',
    'g12_bwd_derisk', 'g12_bwd_derisk_deep', 'g12_bwd_derisk_d10', 'g12_bwd_on_router',
    'g13_wheel_only', 'g13_wheel_ddtrim', 'g13_wheel_bwd',
    'g14_gap_wheel', 'g14_gap_wheel_bwd', 'g14_gap_wheel_real', 'champion_intraday',
    'g15_gap_wheel_hedge', 'g15_hedge_deep',
    'g16_assign', 'g16_assign_bwd',
    'g10_base', 'g10_book45', 'g10_book55', 'g10_book45_h6', 'g10_conv',
    'g10_smoothz', 'g10_smoothz_book45', 'g10_full',
    'champion_kold15_ivrank_kbh', 'champion_kold15_ivrank',
    'g9_smooth_rf', 'g9_smooth_60d', 'g9_smooth_60d_iv',
    'g9_smooth_60d_conv', 'g9_smooth_full',
    'champion_target_25_smooth', 'champion_kold15_ivrank_kbh',
    'g8_kold_matched', 'g8_baseline_matched',
    'g6_cb_a35_b10', 'g6_cb_a35_b20', 'g6_cb_a35_b30',
    'g6_cb_a50_b10', 'g6_cb_a50_b20', 'g6_cb_a50_b30',
    'g6_cb_tightfloor',
    'g5_band_k10', 'g5_band_k15', 'g5_promo_rf',  # baselines for comparison
    'champion_kold15_ivrank',
    'champion_psi_gex', 'champion_psi_fasttp', 'champion_psi_kold15',
    'champion_smooth_ddtrim', 'champion_smooth_gex',
    'champion_psi_ivrank', 'champion_kold15_ivrank',
    'champion_smooth_ddtrim_ivrank',
    'g3_psi_rf', 'g3_kold15_ivrank_rf', 'g3_timing_rf',
    'g3_full_stack', 'g3_smooth_ddtrim_rf',
    'g4_rollguards', 'g4_elevator25', 'g4_tp_ivrank',
    'g4_dd_ivgate', 'g4_spike_defer', 'g4_everything',
    # Pareto-frontier protected family (real strikes ✓)
    'champion_20pct_protected', 'champion_20pct_protected_mom_gated',
    'champion_cut_rebuild',
    # Winner family (real strikes ✓)
    'champion_aggressive_z',
    'champion_aggressive_z_iv_shape',
    'champion_target_25_nav_aware',
    'champion_target_25_cash_start',
    'champion_target_25_window_safe',
    'champion_target_25_dd_trim',
    'champion_target_25_max_protected',
    'champion_premium_harvest_ultra',
    'champion_premium_harvest_scale_invariant',  # production kernel
    'champion_premium_harvest_scale_invariant_hh_storm',  # HH backwardation defensive
    'champion_target_25_smooth',
    'champion_trifecta',
    'champion_20pct_protected_wing_all',
}
# GEN-9: REAL FILLS EVERYWHERE — every kept strategy trades at empirical
# bid/BS-haircut fills unless it EXPLICITLY opts out (real_fill_model=False).
# Makes the whole frontier honest + cross-comparable (kills the model-vs-
# real conflation the audit kept catching).
for _k, _v in STRATEGIES.items():
    _v.setdefault('real_fill_model', True)
    # HONEST FILLS DEFAULT (2026-06-16): price opens at the real BID and closes/rolls
    # at the real ASK (tier-3 real_chain), falling back to model only off-grid. Closes
    # were silently at model mid before — that inflated every churn-heavy strategy.
    _v.setdefault('real_chain_pricing', True)

# Defense-in-depth: also filter out any strategy missing use_real_strikes
STRATEGIES = {k: v for k, v in STRATEGIES.items()
              if k in _KEEP_STRATEGIES and v.get('use_real_strikes')}


# ─── STALE FILTER (lifecycle) ────────────────────────────────────────────────
# Stale strategies remain in code/git but are excluded from cycle runs and
# dashboard. Managed by backtest/retire_stale.py; state in strategy_lifecycle.json.
# Each retirement is logged with date + reason — strategies are NEVER deleted,
# only marked, so we can re-activate or compare retroactively.
try:
    import json as _json
    _LIFECYCLE_PATH = os.path.join(CACHE_DIR, '..', 'strategy_lifecycle.json')
    if os.path.exists(_LIFECYCLE_PATH):
        _STALE = set(_json.loads(open(_LIFECYCLE_PATH).read()).get('stale', []))
        STRATEGIES = {k: v for k, v in STRATEGIES.items() if k not in _STALE}
except Exception:
    pass


def compare_strategies():
    print("=== Loading dataset ===")
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
