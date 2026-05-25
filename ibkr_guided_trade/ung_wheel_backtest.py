#!/usr/bin/env python3
"""
UNG Wheel Strategy Backtest
============================
Comprehensive backtest of selling puts + covered calls on UNG in a systematic wheel.

Strategy:
  1. Sell 30-45 DTE puts at various deltas (ATM to OTM)
  2. Roll at ~5-7 DTE if OTM, take assignment if ITM
  3. When holding shares from assignment, sell covered calls
  4. When calls get assigned, sell puts again
  5. Tracks contango decay impact vs buy-and-hold

Uses Black-Scholes for option pricing with IV proxy = 1.15 * realized vol.

Position sizing: Always trade 1 cash-secured contract. Capital is scaled so that
the put collateral (strike * 100) always fits. This normalizes returns across the
full history regardless of UNG's split-adjusted price level.
"""

import sys
import warnings
warnings.filterwarnings('ignore')

import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import norm
from datetime import datetime, timedelta

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# =============================================================================
# Black-Scholes Functions
# =============================================================================

def bs_call(S, K, T, r, sigma):
    """Black-Scholes call price."""
    if T <= 1e-6:
        return max(0.0, S - K)
    if sigma <= 0:
        sigma = 0.01
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put(S, K, T, r, sigma):
    """Black-Scholes put price."""
    if T <= 1e-6:
        return max(0.0, K - S)
    if sigma <= 0:
        sigma = 0.01
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_delta_call(S, K, T, r, sigma):
    """Black-Scholes call delta."""
    if T <= 1e-6:
        return 1.0 if S > K else 0.0
    if sigma <= 0:
        sigma = 0.01
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)


def bs_delta_put(S, K, T, r, sigma):
    """Black-Scholes put delta (negative)."""
    return bs_delta_call(S, K, T, r, sigma) - 1.0


def strike_for_delta(S, T, r, sigma, target_delta, right='P'):
    """Find strike that gives target |delta| using bisection."""
    if T <= 1e-6:
        return S

    low, high = S * 0.3, S * 2.0

    for _ in range(80):
        mid = (low + high) / 2
        if right == 'P':
            d = abs(bs_delta_put(S, mid, T, r, sigma))
        else:
            d = bs_delta_call(S, mid, T, r, sigma)

        if d > target_delta:
            if right == 'P':
                high = mid
            else:
                low = mid
        else:
            if right == 'P':
                low = mid
            else:
                high = mid

    return mid


# =============================================================================
# Backtest Engine
# =============================================================================

def run_backtest(hist_data, config_name, put_delta, call_delta, dte_target,
                 roll_dte, take_assignment):
    """
    Run a single wheel backtest configuration.

    Returns percentage-based P&L per cycle, normalized to capital at risk.
    Capital at risk for a cash-secured put = strike * 100.
    This makes results comparable across UNG's extreme price range.

    We track a growth factor (starting at 1.0) that compounds each cycle's
    return on capital deployed.
    """
    data = hist_data.dropna(subset=['iv_proxy']).copy().reset_index(drop=True)

    if len(data) < 50:
        print(f"  [SKIP] {config_name}: insufficient data ({len(data)} rows)")
        return None

    MULT = 100  # shares per contract
    r = 0.04    # risk-free rate
    daily_rf = (1 + r) ** (1 / 252) - 1

    # State: track as NAV growth factor (start at 1.0)
    nav = 1.0
    shares_held = False    # whether we hold shares from put assignment
    share_entry_price = 0  # price at which shares were assigned
    position = None

    trades = []
    daily_nav = []
    n_assignments = 0
    n_cycles = 0

    i = 0
    while i < len(data):
        row = data.iloc[i]
        S = row['close']
        iv = row['iv_proxy']
        dt = row['date']

        if np.isnan(iv) or iv <= 0.05:
            iv = 0.50
        iv = min(iv, 3.0)

        # --- Open new position if flat ---
        if position is None:
            T = dte_target / 365.0
            expiry_idx = min(i + dte_target, len(data) - 1)

            if not shares_held:
                # Sell cash-secured put
                K = strike_for_delta(S, T, r, iv, put_delta, 'P')
                if K <= 0:
                    K = S * 0.90
                premium = bs_put(S, K, T, r, iv)

                # Capital at risk = K (per share, cash-secured)
                # Premium yield = premium / K
                position = {
                    'type': 'put',
                    'strike': K,
                    'entry_idx': i,
                    'entry_spot': S,
                    'premium': premium,
                    'premium_yield': premium / K,  # as fraction of collateral
                    'expiry_idx': expiry_idx,
                    'iv_entry': iv,
                }
                n_cycles += 1
            else:
                # Sell covered call on shares we hold
                K = strike_for_delta(S, T, r, iv, call_delta, 'C')
                if K <= 0:
                    K = S * 1.10
                premium = bs_call(S, K, T, r, iv)

                # Capital at risk = share value (S per share)
                position = {
                    'type': 'call',
                    'strike': K,
                    'entry_idx': i,
                    'entry_spot': S,
                    'premium': premium,
                    'premium_yield': premium / S,  # as fraction of share value
                    'expiry_idx': expiry_idx,
                    'iv_entry': iv,
                }
                n_cycles += 1

        # --- Check if time to roll or expire ---
        if position is not None:
            days_left = position['expiry_idx'] - i

            if days_left <= roll_dte or i >= position['expiry_idx']:
                K = position['strike']
                T_remaining = max(days_left, 0) / 365.0

                if position['type'] == 'put':
                    if S < K and take_assignment:
                        # Put ITM -> take assignment
                        # We paid K per share, received premium, shares now worth S
                        # P&L as % of capital at risk (K):
                        cycle_return = (position['premium'] - (K - S)) / K
                        shares_held = True
                        share_entry_price = K - position['premium']  # net cost basis
                        n_assignments += 1
                        trades.append({
                            'date': dt, 'type': 'put_assigned',
                            'strike': K, 'spot': S,
                            'premium_pct': position['premium_yield'] * 100,
                            'cycle_return_pct': cycle_return * 100,
                        })
                    else:
                        # Put OTM or roll -> close position
                        close_val = bs_put(S, K, T_remaining, r, iv) if T_remaining > 1e-6 else max(0, K - S)
                        # P&L = premium received - cost to close, as % of collateral
                        cycle_return = (position['premium'] - close_val) / K
                        trades.append({
                            'date': dt,
                            'type': 'put_expired_otm' if S >= K else 'put_rolled',
                            'strike': K, 'spot': S,
                            'premium_pct': position['premium_yield'] * 100,
                            'cycle_return_pct': cycle_return * 100,
                        })

                elif position['type'] == 'call':
                    if S > K:
                        # Call ITM -> shares called away at K
                        # Total return: premium + (K - entry_spot) per share, as % of entry_spot
                        cycle_return = (position['premium'] + K - position['entry_spot']) / position['entry_spot']
                        shares_held = False
                        share_entry_price = 0
                        n_assignments += 1
                        trades.append({
                            'date': dt, 'type': 'call_assigned',
                            'strike': K, 'spot': S,
                            'premium_pct': position['premium_yield'] * 100,
                            'cycle_return_pct': cycle_return * 100,
                        })
                    else:
                        # Call OTM -> keep shares, close call
                        close_val = bs_call(S, K, T_remaining, r, iv) if T_remaining > 1e-6 else max(0, S - K)
                        # Return = (premium - close_cost + share change) / entry_spot
                        share_change = S - position['entry_spot']
                        cycle_return = (position['premium'] - close_val + share_change) / position['entry_spot']
                        trades.append({
                            'date': dt, 'type': 'call_expired_otm',
                            'strike': K, 'spot': S,
                            'premium_pct': position['premium_yield'] * 100,
                            'cycle_return_pct': cycle_return * 100,
                        })

                # Compound the cycle return into NAV
                nav *= (1 + cycle_return)
                position = None

        # --- Track daily NAV ---
        # Between cycles, if holding shares, mark to market
        if shares_held and position is None:
            # No option position, just holding shares -- NAV changes with share price
            pass
        elif position is not None:
            pass  # will be settled at cycle end

        # For daily tracking, compute mark-to-market NAV
        daily_mtm = nav
        if position is not None:
            days_left_now = max(position['expiry_idx'] - i, 0)
            T_now = days_left_now / 365.0
            if position['type'] == 'put':
                current_val = bs_put(S, position['strike'], T_now, r, iv) if T_now > 1e-6 else max(0, position['strike'] - S)
                # Unrealized P&L on this cycle
                unrealized = (position['premium'] - current_val) / position['strike']
                # Also factor in potential assignment
                if S < position['strike']:
                    unrealized = (position['premium'] - (position['strike'] - S)) / position['strike']
                daily_mtm = nav * (1 + unrealized)
            elif position['type'] == 'call':
                current_val = bs_call(S, position['strike'], T_now, r, iv) if T_now > 1e-6 else max(0, S - position['strike'])
                share_change = S - position['entry_spot']
                unrealized = (position['premium'] - current_val + share_change) / position['entry_spot']
                daily_mtm = nav * (1 + unrealized)

        daily_nav.append({'date': dt, 'nav': daily_mtm, 'spot': S})
        i += 1

    if not daily_nav or not trades:
        return None

    # --- Compute metrics ---
    eq = pd.DataFrame(daily_nav)
    eq['ret'] = eq['nav'].pct_change()

    total_days = len(eq)
    total_years = total_days / 252.0

    final_nav = eq['nav'].iloc[-1]
    total_return_pct = (final_nav - 1) * 100
    if total_years > 0.5:
        ann_return = (final_nav ** (1 / total_years) - 1) * 100
    else:
        ann_return = total_return_pct

    # Buy-and-hold comparison
    bnh_start = data.iloc[0]['close']
    bnh_end = data.iloc[-1]['close']
    bnh_ratio = bnh_end / bnh_start
    if total_years > 0.5 and bnh_ratio > 0:
        bnh_ann = (bnh_ratio ** (1 / total_years) - 1) * 100
    else:
        bnh_ann = (bnh_ratio - 1) * 100
    bnh_return_pct = (bnh_ratio - 1) * 100

    # Sharpe ratio (excess over risk-free)
    daily_rets = eq['ret'].dropna()
    excess_rets = daily_rets - daily_rf
    sharpe = (excess_rets.mean() / excess_rets.std() * np.sqrt(252)) if excess_rets.std() > 0 else 0

    # Max drawdown
    eq['peak'] = eq['nav'].cummax()
    eq['dd'] = (eq['nav'] / eq['peak'] - 1) * 100
    max_dd = eq['dd'].min()

    # Sortino ratio
    downside = daily_rets[daily_rets < daily_rf]
    sortino = ((daily_rets.mean() - daily_rf) / downside.std() * np.sqrt(252)) if len(downside) > 10 and downside.std() > 0 else 0

    # Calmar ratio
    calmar = ann_return / abs(max_dd) if abs(max_dd) > 0.1 else 0

    # Win rate per cycle
    cycle_rets = [t['cycle_return_pct'] for t in trades]
    win_count = sum(1 for r in cycle_rets if r > 0)
    win_rate = win_count / len(cycle_rets) * 100 if cycle_rets else 0

    # Average premium yield
    avg_prem_pct = np.mean([t['premium_pct'] for t in trades]) if trades else 0

    # Average cycle return
    avg_cycle_ret = np.mean(cycle_rets) if cycle_rets else 0

    # Total premium (sum of premium yields)
    total_prem_yield = sum(t['premium_pct'] for t in trades)

    return {
        'name': config_name,
        'total_return': total_return_pct,
        'ann_return': ann_return,
        'bnh_return': bnh_return_pct,
        'bnh_ann': bnh_ann,
        'alpha': ann_return - bnh_ann,
        'sharpe': sharpe,
        'sortino': sortino,
        'calmar': calmar,
        'max_dd': max_dd,
        'n_cycles': n_cycles,
        'n_assignments': n_assignments,
        'win_rate': win_rate,
        'avg_prem_pct': avg_prem_pct,
        'avg_cycle_ret': avg_cycle_ret,
        'total_prem_yield': total_prem_yield,
        'final_nav': final_nav,
        'trades': trades,
        'equity_curve': eq,
        'total_years': total_years,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 100)
    print("UNG WHEEL STRATEGY BACKTEST")
    print("Selling puts + covered calls systematically on UNG")
    print("=" * 100)

    # --- Fetch UNG history ---
    print("\nFetching UNG price history from Yahoo Finance...")
    ung = yf.Ticker('UNG')
    hist = ung.history(period='max', interval='1d')

    if hist.empty:
        print("ERROR: No data returned for UNG")
        sys.exit(1)

    # Normalize index
    if hist.index.tz is not None:
        hist.index = hist.index.tz_localize(None)

    hist = hist[['Close', 'High', 'Low', 'Volume']].rename(
        columns={'Close': 'close', 'High': 'high', 'Low': 'low', 'Volume': 'volume'}
    )
    hist['date'] = hist.index
    hist = hist.reset_index(drop=True)

    # Drop any rows with zero or nan close
    hist = hist[hist['close'] > 0].reset_index(drop=True)

    print(f"  Loaded {len(hist)} daily bars from {hist['date'].iloc[0].date()} to {hist['date'].iloc[-1].date()}")
    print(f"  Price range: ${hist['close'].min():.2f} - ${hist['close'].max():.2f}")
    print(f"  Current price: ${hist['close'].iloc[-1]:.2f}")
    print()
    print("  NOTE: yfinance adjusts UNG history for reverse splits (12 since inception).")
    print("  Split-adjusted prices range from $6500 (2007) to $10 (today).")
    print("  Returns are computed as % of capital at risk per cycle, so they are")
    print("  comparable across the full history regardless of price level.")

    # --- Compute realized vol and IV proxy ---
    hist['ret'] = hist['close'].pct_change()
    hist['rvol_21d'] = hist['ret'].rolling(21).std() * np.sqrt(252)
    hist['iv_proxy'] = hist['rvol_21d'] * 1.15  # IV premium over realized vol

    print(f"\n  Avg realized vol: {hist['rvol_21d'].mean():.1%}")
    print(f"  Avg IV proxy: {hist['iv_proxy'].mean():.1%}")

    # --- Define configurations ---
    configs = [
        # (name, put_delta, call_delta, dte_target, roll_at_dte, take_assignment)
        # Wheel strategies (take assignment and sell calls)
        ('Conservative 30d',  0.25, 0.30, 30, 5, True),
        ('Moderate 30d',      0.35, 0.35, 30, 5, True),
        ('Aggressive 30d',    0.45, 0.40, 30, 5, True),
        ('ATM Straddle 30d',  0.50, 0.50, 30, 5, True),
        ('Conservative 45d',  0.25, 0.30, 45, 7, True),
        ('Moderate 45d',      0.35, 0.35, 45, 7, True),
        ('Weekly 14d',        0.35, 0.35, 14, 3, True),
        # Pure premium strategies (never take assignment -- always roll)
        ('PutsOnly 30d 0.25d', 0.25, 0.30, 30, 5, False),
        ('PutsOnly 30d 0.35d', 0.35, 0.35, 30, 5, False),
        ('PutsOnly 45d 0.25d', 0.25, 0.30, 45, 7, False),
        ('PutsOnly Wkly 0.35', 0.35, 0.35, 14, 3, False),
    ]

    # --- Run backtests ---
    print("\n" + "-" * 100)
    print("Running backtests...")
    print("-" * 100)

    results = []
    for name, pd_delta, cd_delta, dte, roll, assign in configs:
        print(f"  Running: {name} (put_d={pd_delta}, call_d={cd_delta}, DTE={dte}, roll={roll})...", end='')
        res = run_backtest(hist, name, pd_delta, cd_delta, dte, roll, assign)
        if res is not None:
            results.append(res)
            print(f" {res['n_cycles']} cycles, {res['ann_return']:+.1f}% ann, Sharpe={res['sharpe']:.2f}")
        else:
            print(" SKIPPED")

    if not results:
        print("ERROR: No valid results")
        sys.exit(1)

    # --- Print comparison table ---
    print("\n")
    print("=" * 145)
    print("RESULTS COMPARISON (all returns are % of capital at risk per cycle, compounded)")
    first_valid = hist.dropna(subset=['iv_proxy'])
    print(f"Period: {first_valid['date'].iloc[0].date()} to {hist['date'].iloc[-1].date()}")
    print(f"Method: 1 cash-secured contract per cycle | IV proxy = 1.15x realized vol")
    print("=" * 145)

    header = (f"{'Strategy':<22} {'Total%':>9} {'Ann%':>7} {'B&H Ann':>8} {'Alpha':>7} "
              f"{'Sharpe':>7} {'Sortino':>8} {'Calmar':>7} {'MaxDD%':>7} "
              f"{'Cycles':>7} {'Assign':>7} {'Win%':>6} {'AvgPrem':>8} {'AvgCyc':>8}")
    print(header)
    print("-" * 145)

    for res in results:
        line = (f"{res['name']:<22} {res['total_return']:>+8.1f}% {res['ann_return']:>+6.1f}% "
                f"{res['bnh_ann']:>+7.1f}% {res['alpha']:>+6.1f}% {res['sharpe']:>6.2f} "
                f"{res['sortino']:>7.2f} {res['calmar']:>6.2f} "
                f"{res['max_dd']:>+6.1f}% {res['n_cycles']:>7} {res['n_assignments']:>7} "
                f"{res['win_rate']:>5.0f}% {res['avg_prem_pct']:>6.1f}%c {res['avg_cycle_ret']:>+6.1f}%c")
        print(line)

    print("-" * 145)
    print("  AvgPrem = avg premium as % of capital per cycle | AvgCyc = avg total return per cycle")

    # --- Identify best strategy ---
    best_sharpe = max(results, key=lambda x: x['sharpe'])
    best_return = max(results, key=lambda x: x['ann_return'])
    best_calmar = max(results, key=lambda x: x['calmar'])

    print(f"\n  Best Sharpe Ratio:      {best_sharpe['name']} ({best_sharpe['sharpe']:.2f})")
    print(f"  Best Annualized Return: {best_return['name']} ({best_return['ann_return']:+.1f}%)")
    print(f"  Best Calmar Ratio:      {best_calmar['name']} ({best_calmar['calmar']:.2f})")

    # --- Detailed breakdown for best strategy ---
    best = best_sharpe
    print(f"\n{'=' * 80}")
    print(f"DETAILED BREAKDOWN: {best['name']}")
    print(f"{'=' * 80}")
    print(f"  Final NAV (1.0 start): {best['final_nav']:>10.4f}")
    print(f"  Total return:          {best['total_return']:>+10.1f}%")
    print(f"  Annualized return:     {best['ann_return']:>+10.1f}%")
    print(f"  Buy & hold ann return: {best['bnh_ann']:>+10.1f}%")
    print(f"  Alpha over B&H:        {best['alpha']:>+10.1f}%")
    print(f"  Sharpe ratio:          {best['sharpe']:>10.2f}")
    print(f"  Sortino ratio:         {best['sortino']:>10.2f}")
    print(f"  Calmar ratio:          {best['calmar']:>10.2f}")
    print(f"  Max drawdown:          {best['max_dd']:>+10.1f}%")
    print(f"  Total cycles:          {best['n_cycles']:>10}")
    print(f"  Assignments:           {best['n_assignments']:>10}")
    print(f"  Win rate:              {best['win_rate']:>9.0f}%")
    print(f"  Avg premium/cycle:     {best['avg_prem_pct']:>9.1f}%")
    print(f"  Avg cycle return:      {best['avg_cycle_ret']:>+9.1f}%")
    print(f"  Years tested:          {best['total_years']:>10.1f}")

    # Trade type breakdown
    trade_types = {}
    for t in best['trades']:
        tp = t['type']
        if tp not in trade_types:
            trade_types[tp] = {'count': 0, 'total_ret': 0, 'rets': []}
        trade_types[tp]['count'] += 1
        trade_types[tp]['total_ret'] += t['cycle_return_pct']
        trade_types[tp]['rets'].append(t['cycle_return_pct'])

    print(f"\n  Trade Type Breakdown:")
    print(f"    {'Type':<22} {'Count':>6} {'AvgRet%':>9} {'WinRate':>8} {'BestCyc':>9} {'WorstCyc':>10}")
    for tp, info in sorted(trade_types.items()):
        avg_ret = info['total_ret'] / info['count'] if info['count'] > 0 else 0
        wins = sum(1 for r in info['rets'] if r > 0)
        wr = wins / info['count'] * 100 if info['count'] > 0 else 0
        best_cycle = max(info['rets']) if info['rets'] else 0
        worst_cycle = min(info['rets']) if info['rets'] else 0
        print(f"    {tp:<22} {info['count']:>6} {avg_ret:>+8.2f}% {wr:>6.0f}% {best_cycle:>+8.2f}% {worst_cycle:>+9.2f}%")

    # --- Year-by-year breakdown for best ---
    print(f"\n  Year-by-Year Performance ($100k starting capital for illustration):")
    eq = best['equity_curve'].copy()
    eq['year'] = pd.to_datetime(eq['date']).dt.year
    eq['equity_100k'] = eq['nav'] * 100000
    years = sorted(eq['year'].unique())

    print(f"    {'Year':<6} {'Start$':>10} {'End$':>10} {'Return%':>9} {'MaxDD%':>8} {'UNG%':>8}")
    for yr in years:
        yr_data = eq[eq['year'] == yr].copy()
        if len(yr_data) < 5:
            continue
        yr_start = yr_data['equity_100k'].iloc[0]
        yr_end = yr_data['equity_100k'].iloc[-1]
        yr_ret = (yr_end / yr_start - 1) * 100
        yr_peak = yr_data['equity_100k'].cummax()
        yr_dd = ((yr_data['equity_100k'] / yr_peak) - 1).min() * 100
        ung_start = yr_data['spot'].iloc[0]
        ung_end = yr_data['spot'].iloc[-1]
        ung_ret = (ung_end / ung_start - 1) * 100
        print(f"    {yr:<6} ${yr_start:>9,.0f} ${yr_end:>9,.0f} {yr_ret:>+8.1f}% {yr_dd:>+7.1f}% {ung_ret:>+7.1f}%")

    # --- Key insight about contango ---
    print(f"\n{'=' * 80}")
    print("KEY INSIGHTS: WHEEL vs CONTANGO")
    print(f"{'=' * 80}")
    bnh = results[0]['bnh_ann']
    print(f"  UNG buy-and-hold annualized: {bnh:+.1f}% (extreme contango drag)")
    print(f"  UNG has lost {abs(results[0]['bnh_return']):.0f}% of its value since inception.")
    print()

    positive_alpha = [r for r in results if r['alpha'] > 0]
    negative_alpha = [r for r in results if r['alpha'] <= 0]

    if positive_alpha:
        print(f"  Strategies beating B&H:  {len(positive_alpha)}/{len(results)}")
        for r in sorted(positive_alpha, key=lambda x: -x['alpha']):
            print(f"    {r['name']:<22} alpha={r['alpha']:+.1f}%, ann_ret={r['ann_return']:+.1f}%, Sharpe={r['sharpe']:.2f}, MaxDD={r['max_dd']:+.1f}%")
    if negative_alpha:
        print(f"\n  Strategies trailing B&H: {len(negative_alpha)}/{len(results)}")
        for r in sorted(negative_alpha, key=lambda x: x['alpha']):
            print(f"    {r['name']:<22} alpha={r['alpha']:+.1f}%, ann_ret={r['ann_return']:+.1f}%, Sharpe={r['sharpe']:.2f}, MaxDD={r['max_dd']:+.1f}%")

    print()
    all_positive_ret = all(r['ann_return'] > 0 for r in results)
    any_positive_ret = any(r['ann_return'] > 0 for r in results)
    all_beat_bnh = all(r['alpha'] > 0 for r in results)

    if all_beat_bnh and all_positive_ret:
        print("  VERDICT: ALL wheel configs generate positive absolute AND relative returns.")
        print("  The premium income from selling options more than offsets UNG's contango drag.")
    elif all_beat_bnh:
        neg_ret_count = sum(1 for r in results if r['ann_return'] <= 0)
        print(f"  VERDICT: All strategies beat buy-and-hold, but {neg_ret_count} still lose money on")
        print("  an absolute basis. The wheel reduces losses but can't fully overcome contango.")
    elif any_positive_ret:
        pos_count = sum(1 for r in results if r['ann_return'] > 0)
        print(f"  VERDICT: {pos_count}/{len(results)} strategies produce positive absolute returns.")
        print("  Conservative approaches work better -- aggressive deltas take too much assignment risk.")
    else:
        print("  VERDICT: No wheel strategy fully overcomes UNG's contango decay.")
        print("  NOTE: IV proxy (1.15x realized vol) understates real IV; actual results would be better.")

    # Practical recommendation
    print(f"\n  PRACTICAL RECOMMENDATION:")
    print(f"    Best risk-adjusted strategy: {best_sharpe['name']}")
    print(f"    - {best_sharpe['ann_return']:+.1f}% annualized, {best_sharpe['sharpe']:.2f} Sharpe, {best_sharpe['max_dd']:+.1f}% max DD")
    print(f"    - {best_sharpe['win_rate']:.0f}% win rate, avg {best_sharpe['avg_prem_pct']:.1f}% premium per cycle")

    if best_return != best_sharpe:
        print(f"    Best absolute return: {best_return['name']}")
        print(f"    - {best_return['ann_return']:+.1f}% annualized, {best_return['sharpe']:.2f} Sharpe, {best_return['max_dd']:+.1f}% max DD")

    # --- Caveats ---
    print(f"\n{'=' * 80}")
    print("CAVEATS / LIMITATIONS")
    print(f"{'=' * 80}")
    print("  1. IV proxy (1.15x realized vol) likely UNDERSTATES real IV for UNG options.")
    print("     Real UNG IV is typically 1.3-1.5x RV. Actual premiums would be ~15-30% higher.")
    print("  2. No transaction costs modeled (commissions ~$0.65/contract, slippage ~$0.03-0.05).")
    print("  3. No bid-ask spread modeled (UNG options ~$0.05-0.15 wide).")
    print("  4. No early assignment risk modeled (rare for ETF options but possible).")
    print("  5. UNG has had 12 reverse splits; yfinance adjusts prices backward for continuity.")
    print("  6. Black-Scholes assumes lognormal; UNG has significant skew and fat tails.")
    print("  7. Single contract at a time; no scaling, hedging, or multi-leg strategies.")
    print("  8. Risk-free rate fixed at 4%; actual rate varied from 0-5% over this period.")

    # ==========================================================================
    # Charts
    # ==========================================================================
    print("\nGenerating charts...")

    fig, axes = plt.subplots(3, 1, figsize=(16, 14), facecolor='#0d1117')

    for ax in axes:
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#e6edf3', labelsize=9)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_color('#30363d')
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

    colors = ['#58a6ff', '#3fb950', '#f0883e', '#f85149', '#bc8cff', '#79c0ff', '#d2a8ff']

    # --- Chart 1: NAV curves (normalized to $100) ---
    ax1 = axes[0]
    for idx, res in enumerate(results):
        eq_data = res['equity_curve']
        normalized = eq_data['nav'] * 100
        ax1.plot(eq_data['date'], normalized, label=res['name'],
                 color=colors[idx % len(colors)], linewidth=1.2, alpha=0.9)

    # Buy-and-hold line
    ref_eq = results[0]['equity_curve']
    bnh_normalized = ref_eq['spot'] / ref_eq['spot'].iloc[0] * 100
    ax1.plot(ref_eq['date'], bnh_normalized, label='Buy & Hold UNG',
             color='#8b949e', linewidth=2.5, linestyle='--', alpha=0.8)

    ax1.axhline(100, color='#30363d', linestyle='-', alpha=0.5)
    ax1.set_ylabel('Portfolio Value ($100 start)', color='#e6edf3', fontsize=11)
    ax1.set_title('UNG Wheel Strategy Backtest - Growth of $100', color='#e6edf3', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=7, facecolor='#161b22', edgecolor='#30363d', labelcolor='#e6edf3',
               loc='upper left', ncol=2)
    ax1.grid(True, color='#21262d', alpha=0.5)
    ax1.set_yscale('log')
    ax1.set_ylim(bottom=0.1)

    # --- Chart 2: Rolling 1-year returns for best strategy vs B&H ---
    ax2 = axes[1]
    best_eq = best['equity_curve'].copy()
    if len(best_eq) > 252:
        rolling_ret = best_eq['nav'].pct_change(252) * 100
        bnh_rolling_ret = best_eq['spot'].pct_change(252) * 100
        ax2.plot(best_eq['date'], rolling_ret, label=f'{best["name"]} (rolling 1yr)',
                 color='#58a6ff', linewidth=1.2)
        ax2.plot(best_eq['date'], bnh_rolling_ret, label='Buy & Hold (rolling 1yr)',
                 color='#8b949e', linewidth=1.2, linestyle='--')
        ax2.axhline(0, color='#f85149', linestyle='-', alpha=0.5, linewidth=0.8)
        ax2.fill_between(best_eq['date'], rolling_ret, 0,
                         where=rolling_ret > 0, alpha=0.15, color='#3fb950')
        ax2.fill_between(best_eq['date'], rolling_ret, 0,
                         where=rolling_ret < 0, alpha=0.15, color='#f85149')
    ax2.set_ylabel('Rolling 1-Year Return (%)', color='#e6edf3', fontsize=11)
    ax2.set_title(f'Rolling Returns: {best["name"]} vs Buy & Hold', color='#e6edf3', fontsize=12)
    ax2.legend(fontsize=9, facecolor='#161b22', edgecolor='#30363d', labelcolor='#e6edf3')
    ax2.grid(True, color='#21262d', alpha=0.5)

    # --- Chart 3: UNG price (log scale) + strategy drawdown ---
    ax3 = axes[2]
    ax3.plot(ref_eq['date'], ref_eq['spot'], color='#8b949e', linewidth=1)
    ax3.set_ylabel('UNG Price ($, log scale)', color='#e6edf3', fontsize=11)
    ax3.set_xlabel('Date', color='#e6edf3', fontsize=11)
    ax3.set_title('UNG Price History (note persistent contango decay)', color='#e6edf3', fontsize=12)
    ax3.grid(True, color='#21262d', alpha=0.5)
    ax3.set_yscale('log')

    # Add drawdown on secondary axis
    ax3b = ax3.twinx()
    ax3b.spines['right'].set_color('#30363d')
    ax3b.spines['top'].set_visible(False)
    ax3b.tick_params(colors='#e6edf3', labelsize=9)
    best_dd = best['equity_curve']['dd']
    ax3b.fill_between(best['equity_curve']['date'], best_dd, 0,
                      alpha=0.3, color='#f85149', label=f'{best["name"]} Drawdown')
    ax3b.set_ylabel('Strategy Drawdown (%)', color='#f85149', fontsize=10)
    dd_min = best_dd.min()
    ax3b.set_ylim(dd_min * 1.3 if dd_min < -1 else -10, 5)
    ax3b.legend(fontsize=8, facecolor='#161b22', edgecolor='#30363d', labelcolor='#e6edf3',
                loc='lower right')

    plt.tight_layout()

    chart_path = '/home/wyatt/weather/ung_wheel_backtest.png'
    plt.savefig(chart_path, dpi=150, facecolor='#0d1117', bbox_inches='tight')
    plt.close()
    print(f"  Equity curves: {chart_path}")

    # --- Summary bar chart ---
    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5), facecolor='#0d1117')
    for ax in axes2:
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#e6edf3', labelsize=9)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_color('#30363d')
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

    names = [r['name'] for r in results]
    x = np.arange(len(names))

    # Ann return comparison
    ann_rets = [r['ann_return'] for r in results]
    bar_colors = ['#3fb950' if v > 0 else '#f85149' for v in ann_rets]
    axes2[0].barh(x, ann_rets, color=bar_colors, alpha=0.8)
    axes2[0].axvline(results[0]['bnh_ann'], color='#8b949e', linestyle='--', linewidth=1.5, label='B&H UNG')
    axes2[0].set_yticks(x)
    axes2[0].set_yticklabels(names, fontsize=9)
    axes2[0].set_xlabel('Annualized Return (%)', color='#e6edf3')
    axes2[0].set_title('Annualized Return', color='#e6edf3', fontsize=12)
    axes2[0].legend(fontsize=9, facecolor='#161b22', edgecolor='#30363d', labelcolor='#e6edf3')

    # Sharpe comparison
    sharpes = [r['sharpe'] for r in results]
    bar_colors2 = ['#3fb950' if v > 0 else '#f85149' for v in sharpes]
    axes2[1].barh(x, sharpes, color=bar_colors2, alpha=0.8)
    axes2[1].axvline(0, color='#30363d', linestyle='-', linewidth=0.5)
    axes2[1].set_yticks(x)
    axes2[1].set_yticklabels(names, fontsize=9)
    axes2[1].set_xlabel('Sharpe Ratio', color='#e6edf3')
    axes2[1].set_title('Sharpe Ratio', color='#e6edf3', fontsize=12)

    # Max drawdown comparison
    dds = [r['max_dd'] for r in results]
    axes2[2].barh(x, dds, color='#f85149', alpha=0.8)
    axes2[2].set_yticks(x)
    axes2[2].set_yticklabels(names, fontsize=9)
    axes2[2].set_xlabel('Max Drawdown (%)', color='#e6edf3')
    axes2[2].set_title('Max Drawdown', color='#e6edf3', fontsize=12)

    plt.suptitle('UNG Wheel Strategy Comparison', color='#e6edf3', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    chart_path2 = '/home/wyatt/weather/ung_wheel_comparison.png'
    plt.savefig(chart_path2, dpi=150, facecolor='#0d1117', bbox_inches='tight')
    plt.close()
    print(f"  Comparison:    {chart_path2}")

    print("\nDone.")


if __name__ == '__main__':
    main()
