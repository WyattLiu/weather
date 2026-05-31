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
    Also adds price-spike indicator (UNG % change vs 60d ago)."""
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
    return df


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

    for i in range(len(df) - 30):
        idx = df.index[i]
        row = df.iloc[i]
        spot_u = row.get('UNG', 0)
        if spot_u <= 0:
            continue
        spot_k = row.get('KOLD', 0) or 0
        if isinstance(spot_k, float) and math.isnan(spot_k):
            spot_k = 0
        iv_u = row.get('iv_30d', 0.55)
        if iv_u is None or (isinstance(iv_u, float) and math.isnan(iv_u)) or iv_u <= 0:
            iv_u = 0.55
        z = compute_historical_z(row, use_surprise=use_surprise)
        r = regime(z)

        # Expire short puts
        keep = []
        for sp in s['short_puts']:
            days = (idx - sp['entry']).days
            T_left = max(1, sp['dte'] - days) / 365

            # Take profit
            if p.get('tp_50') and T_left > 1/365:
                cv = bs_put(spot_u, sp['K'], T_left, iv_u)
                if cv < sp['entry_prem'] * 0.5:
                    s['cash'] += (sp['entry_prem'] - cv) * 100 * sp['qty'] - sp['qty'] * SPREAD_OPTION * 100
                    trades.append({'date': idx, 'type': 'PUT_TP', 'pnl': (sp['entry_prem']-cv)*100*sp['qty']})
                    continue

            # Roll down
            if p.get('roll_down') and spot_u < sp['K'] * 0.98 and T_left > 5/365:
                cv = bs_put(spot_u, sp['K'], T_left, iv_u)
                s['cash'] -= cv * 100 * sp['qty']
                # New strike
                nk = round(spot_u * (1 - p.get('otm_put', 0.10)))
                npr = bs_put(spot_u, nk, 30/365, iv_u)
                s['cash'] += npr * 100 * sp['qty'] - sp['qty'] * SPREAD_OPTION * 100
                keep.append({'entry': idx, 'K': nk, 'dte': 30, 'qty': sp['qty'], 'entry_prem': npr})
                continue

            if days >= sp['dte']:
                if spot_u < sp['K']:
                    s['cash'] -= (sp['K'] - spot_u) * 100 * sp['qty']
                    s['shares'] += sp['qty'] * 100
                    s['cash'] -= sp['qty'] * 100 * sp['K']
                    trades.append({'date': idx, 'type': 'PUT_ASSIGN', 'qty': sp['qty']})
                continue
            keep.append(sp)
        s['short_puts'] = keep

        # Expire short calls — with roll_up_call + elevator close support
        keep = []
        for sc in s['short_calls']:
            days = (idx - sc['entry']).days
            T_left = max(1, sc['dte'] - days) / 365
            if p.get('tp_50') and T_left > 1/365:
                cv = bs_call(spot_u, sc['K'], T_left, iv_u)
                if cv < sc['entry_prem'] * 0.5:
                    s['cash'] += (sc['entry_prem'] - cv) * 100 * sc['qty']
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
                cv = bs_call(spot_u, sc['K'], T_left, iv_u)
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
                if (at_peak and is_deep_itm and low_extrinsic
                        and s['shares'] >= sc['qty'] * 100):
                    s['cash'] -= cv * 100 * sc['qty'] + sc['qty'] * SPREAD_OPTION * 100
                    n_shares = sc['qty'] * 100
                    s['cash'] += n_shares * spot_u - n_shares * SPREAD_SHARE
                    s['shares'] -= n_shares
                    locked = (spot_u - sc['K']) * 100 * sc['qty']
                    trades.append({
                        'date': idx, 'type': 'ELEVATOR_CLOSE',
                        'trigger': 'spike' if price_spike else 'storage_peak',
                        'K': sc['K'], 'spot': spot_u, 'qty': sc['qty'],
                        'locked_gain': locked, 'z': z, 'spike_60d': spike_pct,
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
                # Close current (pay intrinsic + small extrinsic)
                cv = bs_call(spot_u, sc['K'], T_left, iv_u)
                s['cash'] -= cv * 100 * sc['qty']
                # Open new at higher strike (5% OTM from current spot) + 30 DTE
                new_K = round(spot_u * 1.05)
                new_prem = bs_call(spot_u, new_K, 30/365, iv_u)
                if new_prem > 0.05:
                    s['cash'] += (new_prem * 100 * sc['qty']
                                  - sc['qty'] * SPREAD_OPTION * 100 * 2)  # 2 spreads
                    keep.append({
                        'entry': idx, 'K': new_K, 'dte': 30,
                        'qty': sc['qty'], 'entry_prem': new_prem,
                    })
                    trades.append({'date': idx, 'type': 'CALL_ROLL_UP',
                                   'from_K': sc['K'], 'to_K': new_K, 'qty': sc['qty']})
                continue

            if days >= sc['dte']:
                if spot_u > sc['K']:
                    s['shares'] -= sc['qty'] * 100
                    s['cash'] += sc['qty'] * 100 * sc['K']
                    trades.append({'date': idx, 'type': 'CALL_ASSIGN', 'qty': sc['qty']})
                continue
            keep.append(sc)
        s['short_calls'] = keep

        # Expire long puts
        keep = []
        for lp in s['long_puts']:
            if (idx - lp['entry']).days >= lp['dte']:
                s['cash'] += max(0, lp['K'] - spot_u) * 100 * lp['qty']
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

        # Weekly entries
        if i % 7 == 0:
            otm_put = p.get('otm_put', 0.10)
            otm_call = p.get('otm_call', 0.05)
            put_qty = p.get('put_qty', 3)
            call_qty = p.get('call_qty', 3)

            # Skip puts based on regime
            skip_put = p.get('regime_skip_puts_z') is not None and z < p['regime_skip_puts_z']
            if not skip_put:
                K = round(spot_u * (1 - otm_put))
                prem = bs_put(spot_u, K, 30/365, iv_u)
                if prem > 0.05:
                    s['cash'] += prem * 100 * put_qty - put_qty * SPREAD_OPTION * 100
                    s['short_puts'].append({'entry': idx, 'K': K, 'dte': 30,
                                            'qty': put_qty, 'entry_prem': prem})

            # CCs (only if have shares)
            if s['shares'] >= 300:
                # Aggressive ITM CC: when z is rich, write deeper ITM to force assignment
                use_itm = (p.get('aggressive_itm_cc_z') is not None
                           and z < p['aggressive_itm_cc_z'])
                effective_otm = p.get('itm_cc_pct', otm_call) if use_itm else otm_call
                K = round(spot_u * (1 + effective_otm))
                qty = min(call_qty, s['shares'] // 100)
                prem = bs_call(spot_u, K, 30/365, iv_u)
                if prem > 0.05:
                    s['cash'] += prem * 100 * qty - qty * SPREAD_OPTION * 100
                    s['short_calls'].append({'entry': idx, 'K': K, 'dte': 30,
                                             'qty': qty, 'entry_prem': prem,
                                             'is_itm_aggressive': use_itm})
                    if use_itm:
                        trades.append({'date': idx, 'type': 'AGGRESSIVE_ITM_CC',
                                       'K': K, 'qty': qty, 'z': z})

            # EXTREME_RICH bearish stack
            if p.get('bearish_stack') and r == 'EXTREME_RICH':
                if not s['long_puts']:
                    Kp = round(spot_u * 0.95)
                    cost = bs_put(spot_u, Kp, 90/365, iv_u)
                    qty = 3
                    s['cash'] -= cost * 100 * qty + qty * SPREAD_OPTION * 100
                    s['long_puts'].append({'entry': idx, 'K': Kp, 'dte': 90, 'qty': qty, 'cost': cost})

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
