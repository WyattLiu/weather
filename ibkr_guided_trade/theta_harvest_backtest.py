#!/usr/bin/env python3
"""
Theta Harvest Backtest — Delta-Neutral Put Selling on UNG + KOLD

Sells OTM puts on both UNG (long NG) and KOLD (2x inverse NG) during
shoulder seasons, keeping delta-neutral. These two ETFs are ~-0.98
correlated, so selling OTM puts on both creates a theta-harvesting
portfolio that profits regardless of NG direction.

Edge: volatility risk premium (IV > RV). Since no historical option
chains exist, IV is estimated as RV × configurable premium (default 1.30).
"""

import argparse
from datetime import timedelta

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm


# ── Black-Scholes Functions ─────────────────────────────────────────

def bs_price(S, K, T, r, sigma, option_type='put'):
    """Black-Scholes option price."""
    if T <= 0.001:
        return max(0, S - K) if option_type == 'call' else max(0, K - S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_delta(S, K, T, r, sigma):
    """Black-Scholes put delta."""
    if T <= 0.001:
        return -1.0 if K > S else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) - 1.0


def bs_greeks(S, K, T, r, sigma):
    """Return dict of put greeks: delta, gamma, theta, vega."""
    if T <= 0.001:
        intrinsic = max(0, K - S)
        return {'delta': -1.0 if K > S else 0.0,
                'gamma': 0.0, 'theta': 0.0, 'vega': 0.0,
                'price': intrinsic}
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    sqrt_T = np.sqrt(T)
    pdf_d1 = norm.pdf(d1)

    delta = norm.cdf(d1) - 1.0
    gamma = pdf_d1 / (S * sigma * sqrt_T)
    theta = (-(S * pdf_d1 * sigma) / (2 * sqrt_T)
             + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 252  # per day
    vega = S * pdf_d1 * sqrt_T / 100  # per 1% IV move
    price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return {'delta': delta, 'gamma': gamma, 'theta': theta,
            'vega': vega, 'price': price}


def find_strike_for_delta(S, target_delta, T, r, sigma, precision=0.001):
    """Binary search for put strike that gives target_delta (negative)."""
    target = -abs(target_delta)  # ensure negative
    lo, hi = S * 0.50, S * 1.00  # OTM puts: strike < spot
    mid = (lo + hi) / 2
    for _ in range(100):
        mid = (lo + hi) / 2
        d = bs_delta(S, mid, T, r, sigma)
        if abs(d - target) < precision * 0.1:
            break
        if d < target:  # delta too negative → strike too high
            hi = mid
        else:
            lo = mid
    # Round to nearest 0.50 for UNG-like ETFs
    return round(mid * 2) / 2


# ── Data Fetch ───────────────────────────────────────────────────────

def fetch_data(start_date=None):
    """Fetch daily bars for UNG, KOLD, NG=F via yfinance."""
    tickers = {'UNG': 'ung_close', 'KOLD': 'kold_close', 'NG=F': 'ng_close'}
    frames = {}

    for sym, col in tickers.items():
        print(f"  Fetching {sym}...", end=' ')
        t = yf.Ticker(sym)
        hist = t.history(period="max", interval="1d")
        if hist.index.tz is not None:
            hist.index = hist.index.tz_localize(None)
        hist = hist[['Close']].rename(columns={'Close': col})
        hist.index.name = 'date'
        hist = hist.reset_index()
        if start_date:
            hist = hist[hist['date'] >= pd.Timestamp(start_date)]
        frames[col] = hist
        print(f"{len(hist)} bars")

    # Merge on date
    df = frames['ung_close']
    for col in ['kold_close', 'ng_close']:
        df = df.merge(frames[col], on='date', how='inner')

    df = df.sort_values('date').reset_index(drop=True)
    df = df.dropna()
    print(f"  Merged: {len(df)} trading days "
          f"({df['date'].iloc[0].strftime('%Y-%m-%d')} to "
          f"{df['date'].iloc[-1].strftime('%Y-%m-%d')})")
    return df


# ── IV Estimation ────────────────────────────────────────────────────

def compute_rv_series(prices, window=30):
    """Compute rolling annualized realized vol from close prices."""
    log_ret = np.log(prices / prices.shift(1))
    rv = log_ret.rolling(window).std() * np.sqrt(252)
    return rv


def estimate_iv(rv, rv_premium=1.30):
    """Estimate IV from realized vol with risk premium multiplier."""
    return rv * rv_premium


def otm_skew_adjustment(iv_atm, K, S):
    """Add skew premium for OTM puts: further OTM → higher IV."""
    moneyness = K / S  # < 1 for OTM puts
    skew = 1.0 + 0.10 * (1.0 - moneyness)  # ~2-3% for 15% OTM
    return iv_atm * skew


# ── Shoulder Season ──────────────────────────────────────────────────

def is_shoulder(date):
    """Mar-May, Sep-Nov are shoulder seasons for nat gas."""
    return date.month in (3, 4, 5, 9, 10, 11)


# ── Backtest Engine ──────────────────────────────────────────────────

class Position:
    """A single short put position."""
    def __init__(self, symbol, entry_date, expiry_date, spot, strike,
                 iv, premium, contracts, dte):
        self.symbol = symbol
        self.entry_date = entry_date
        self.expiry_date = expiry_date
        self.entry_spot = spot
        self.strike = strike
        self.entry_iv = iv
        self.premium = premium          # per-share premium received
        self.contracts = contracts
        self.entry_dte = dte
        self.closed = False
        self.close_date = None
        self.close_pnl = 0.0
        self.close_reason = ''
        self.current_value = premium    # current mark of the option
        self.current_delta = 0.0

    def total_premium(self):
        """Total premium received (per share × 100 shares × contracts)."""
        return self.premium * 100 * self.contracts

    def unrealized_pnl(self, mark_price):
        """P&L = premium received - current cost to close."""
        return (self.premium - mark_price) * 100 * self.contracts

    def mark_to_market(self, spot, iv, dte_remaining, r=0.045):
        """Update current value and delta."""
        T = max(dte_remaining / 365.0, 0.001)
        iv_adj = otm_skew_adjustment(iv, self.strike, spot)
        self.current_value = bs_price(spot, self.strike, T, r, iv_adj, 'put')
        self.current_delta = bs_delta(spot, self.strike, T, r, iv_adj)
        return self.unrealized_pnl(self.current_value)


def run_backtest(df, rv_premium=1.30, target_delta=0.20,
                 ung_contracts=2, entry_interval=14, r=0.045,
                 slippage=0.05):
    """
    Run the theta harvest backtest.

    Returns: positions list, daily_log DataFrame
    """
    # Compute rolling RV and estimated IV
    df = df.copy()
    df['ung_rv'] = compute_rv_series(df['ung_close'], 30)
    df['kold_rv'] = compute_rv_series(df['kold_close'], 30)
    df['ung_iv'] = estimate_iv(df['ung_rv'], rv_premium)
    df['kold_iv'] = estimate_iv(df['kold_rv'], rv_premium)
    df['ung_vrp'] = df['ung_iv'] - df['ung_rv']
    df['kold_vrp'] = df['kold_iv'] - df['kold_rv']

    # Drop rows without enough history for RV
    df = df.dropna(subset=['ung_rv', 'kold_rv']).reset_index(drop=True)

    positions = []
    daily_log = []
    days_since_entry = entry_interval  # allow immediate first entry
    cumulative_pnl = 0.0

    for i in range(len(df)):
        row = df.iloc[i]
        date = row['date']
        ung_spot = row['ung_close']
        kold_spot = row['kold_close']
        ung_iv = row['ung_iv']
        kold_iv = row['kold_iv']

        # ── Check exits on open positions ──
        for pos in positions:
            if pos.closed:
                continue

            spot = ung_spot if pos.symbol == 'UNG' else kold_spot
            iv = ung_iv if pos.symbol == 'UNG' else kold_iv
            dte_remaining = (pos.expiry_date - date).days

            if dte_remaining <= 0:
                # Expired: settle at intrinsic
                intrinsic = max(0, pos.strike - spot)
                close_cost = intrinsic * 100 * pos.contracts
                pos.close_pnl = pos.total_premium() - close_cost
                pos.closed = True
                pos.close_date = date
                pos.close_reason = 'expired'
                cumulative_pnl += pos.close_pnl
                continue

            unrealized = pos.mark_to_market(spot, iv, dte_remaining, r)

            # Exit rule 1: DTE ≤ 7
            if dte_remaining <= 7:
                close_price = pos.current_value * (1 + slippage)  # buy at ask
                pos.close_pnl = (pos.premium - close_price) * 100 * pos.contracts
                pos.closed = True
                pos.close_date = date
                pos.close_reason = 'dte_exit'
                cumulative_pnl += pos.close_pnl
                continue

            # Exit rule 2: 50% profit
            if unrealized >= 0.50 * pos.total_premium():
                close_price = pos.current_value * (1 + slippage)
                pos.close_pnl = (pos.premium - close_price) * 100 * pos.contracts
                pos.closed = True
                pos.close_date = date
                pos.close_reason = 'profit_target'
                cumulative_pnl += pos.close_pnl
                continue

            # Exit rule 3: 200% loss
            if unrealized <= -2.0 * pos.total_premium():
                close_price = pos.current_value * (1 + slippage)
                pos.close_pnl = (pos.premium - close_price) * 100 * pos.contracts
                pos.closed = True
                pos.close_date = date
                pos.close_reason = 'stop_loss'
                cumulative_pnl += pos.close_pnl
                continue

        # ── Check for new entry ──
        days_since_entry += 1
        if days_since_entry >= entry_interval:
            # Only enter if we have valid IV
            if np.isfinite(ung_iv) and np.isfinite(kold_iv) and ung_iv > 0 and kold_iv > 0:
                dte = 37  # midpoint of 30-45 DTE
                T = dte / 365.0
                expiry = date + timedelta(days=dte)

                # UNG put
                ung_strike = find_strike_for_delta(
                    ung_spot, target_delta, T, r, ung_iv)
                ung_iv_otm = otm_skew_adjustment(ung_iv, ung_strike, ung_spot)
                ung_mid = bs_price(ung_spot, ung_strike, T, r, ung_iv_otm, 'put')
                ung_prem = ung_mid * (1 - slippage)  # sell at bid

                # KOLD put — size for dollar-delta balance
                kold_cts = max(1, round(
                    ung_contracts * ung_spot / (kold_spot * 2)))
                kold_strike = find_strike_for_delta(
                    kold_spot, target_delta, T, r, kold_iv)
                kold_iv_otm = otm_skew_adjustment(kold_iv, kold_strike, kold_spot)
                kold_mid = bs_price(kold_spot, kold_strike, T, r, kold_iv_otm, 'put')
                kold_prem = kold_mid * (1 - slippage)

                # Only enter if premium is meaningful (> $0.05)
                if ung_prem > 0.05 and kold_prem > 0.05:
                    positions.append(Position(
                        'UNG', date, expiry, ung_spot, ung_strike,
                        ung_iv_otm, ung_prem, ung_contracts, dte))
                    positions.append(Position(
                        'KOLD', date, expiry, kold_spot, kold_strike,
                        kold_iv_otm, kold_prem, kold_cts, dte))
                    days_since_entry = 0

        # ── Daily log ──
        open_positions = [p for p in positions if not p.closed]
        net_delta_ung = sum(
            p.current_delta * p.contracts * 100
            for p in open_positions if p.symbol == 'UNG')
        net_delta_kold = sum(
            p.current_delta * p.contracts * 100
            for p in open_positions if p.symbol == 'KOLD')
        net_delta = net_delta_ung + net_delta_kold

        unrealized_total = sum(
            p.unrealized_pnl(p.current_value) for p in open_positions)

        daily_log.append({
            'date': date,
            'ung_spot': ung_spot,
            'kold_spot': kold_spot,
            'ung_rv': row['ung_rv'],
            'kold_rv': row['kold_rv'],
            'ung_iv': ung_iv,
            'kold_iv': kold_iv,
            'ung_vrp': row['ung_vrp'],
            'kold_vrp': row['kold_vrp'],
            'net_delta': net_delta,
            'net_delta_ung': net_delta_ung,
            'net_delta_kold': net_delta_kold,
            'unrealized_pnl': unrealized_total,
            'cumulative_pnl': cumulative_pnl,
            'realized_pnl': cumulative_pnl,
            'total_pnl': cumulative_pnl + unrealized_total,
            'shoulder': is_shoulder(date),
            'open_count': len(open_positions),
        })

    log_df = pd.DataFrame(daily_log)
    return positions, log_df, df


# ── Statistics ───────────────────────────────────────────────────────

def compute_stats(positions, log_df):
    """Compute backtest statistics."""
    closed = [p for p in positions if p.closed]
    if not closed:
        print("  No closed positions!")
        return {}

    # Pair trades (UNG + KOLD entered same day)
    from collections import defaultdict
    by_entry = defaultdict(list)
    for p in closed:
        by_entry[p.entry_date].append(p)

    trade_pairs = []
    for dt, plist in sorted(by_entry.items()):
        pair_pnl = sum(p.close_pnl for p in plist)
        pair_prem = sum(p.total_premium() for p in plist)
        reasons = [p.close_reason for p in plist]
        trade_pairs.append({
            'entry_date': dt,
            'pnl': pair_pnl,
            'premium': pair_prem,
            'reasons': reasons,
            'shoulder': is_shoulder(dt),
        })

    wins = [t for t in trade_pairs if t['pnl'] > 0]
    losses = [t for t in trade_pairs if t['pnl'] <= 0]

    total_premium = sum(p.total_premium() for p in closed)
    total_pnl = sum(p.close_pnl for p in closed)
    win_rate = len(wins) / len(trade_pairs) * 100 if trade_pairs else 0

    avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
    avg_loss = np.mean([t['pnl'] for t in losses]) if losses else 0
    gross_profit = sum(t['pnl'] for t in wins)
    gross_loss = abs(sum(t['pnl'] for t in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Sharpe from daily P&L
    if len(log_df) > 1:
        daily_returns = log_df['total_pnl'].diff().dropna()
        sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)
                  if daily_returns.std() > 0 else 0)
    else:
        sharpe = 0

    # Max drawdown
    peak = log_df['total_pnl'].cummax()
    drawdown = log_df['total_pnl'] - peak
    max_dd = drawdown.min()

    # Shoulder vs non-shoulder
    shoulder_pairs = [t for t in trade_pairs if t['shoulder']]
    non_shoulder_pairs = [t for t in trade_pairs if not t['shoulder']]
    shoulder_wr = (len([t for t in shoulder_pairs if t['pnl'] > 0])
                   / len(shoulder_pairs) * 100 if shoulder_pairs else 0)
    non_shoulder_wr = (len([t for t in non_shoulder_pairs if t['pnl'] > 0])
                       / len(non_shoulder_pairs) * 100 if non_shoulder_pairs else 0)

    # Monthly win rate
    monthly_wr = {}
    for t in trade_pairs:
        m = t['entry_date'].month
        if m not in monthly_wr:
            monthly_wr[m] = {'wins': 0, 'total': 0}
        monthly_wr[m]['total'] += 1
        if t['pnl'] > 0:
            monthly_wr[m]['wins'] += 1

    # Delta stats
    if len(log_df) > 0:
        avg_delta = log_df['net_delta'].mean()
        max_delta = log_df['net_delta'].abs().max()
    else:
        avg_delta = max_delta = 0

    # Average IV-RV spread
    avg_ung_vrp = log_df['ung_vrp'].mean() * 100 if len(log_df) > 0 else 0
    avg_kold_vrp = log_df['kold_vrp'].mean() * 100 if len(log_df) > 0 else 0

    # Exit reason breakdown
    reason_counts = defaultdict(int)
    for p in closed:
        reason_counts[p.close_reason] += 1

    stats = {
        'total_trades': len(trade_pairs),
        'total_premium': total_premium,
        'total_pnl': total_pnl,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'shoulder_trades': len(shoulder_pairs),
        'shoulder_wr': shoulder_wr,
        'non_shoulder_trades': len(non_shoulder_pairs),
        'non_shoulder_wr': non_shoulder_wr,
        'monthly_wr': monthly_wr,
        'avg_delta': avg_delta,
        'max_delta': max_delta,
        'avg_ung_vrp': avg_ung_vrp,
        'avg_kold_vrp': avg_kold_vrp,
        'reason_counts': dict(reason_counts),
        'trade_pairs': trade_pairs,
    }
    return stats


def print_stats(stats):
    """Print backtest statistics to console."""
    if not stats:
        return
    print("\n" + "=" * 65)
    print("  THETA HARVEST BACKTEST — UNG + KOLD PUT SELLING")
    print("=" * 65)
    print(f"  Trade pairs:       {stats['total_trades']}")
    print(f"  Total premium:    ${stats['total_premium']:>10,.0f}")
    print(f"  Total P&L:        ${stats['total_pnl']:>10,.0f}")
    print(f"  Win rate:          {stats['win_rate']:.1f}%")
    print(f"  Avg win:          ${stats['avg_win']:>10,.0f}")
    print(f"  Avg loss:         ${stats['avg_loss']:>10,.0f}")
    pf = stats['profit_factor']
    pf_str = f"{pf:.2f}" if pf < 100 else "inf"
    print(f"  Profit factor:     {pf_str}")
    print(f"  Sharpe ratio:      {stats['sharpe']:.2f}")
    print(f"  Max drawdown:     ${stats['max_drawdown']:>10,.0f}")
    print()
    print(f"  Shoulder season:   {stats['shoulder_trades']} trades, "
          f"{stats['shoulder_wr']:.1f}% win rate")
    print(f"  Non-shoulder:      {stats['non_shoulder_trades']} trades, "
          f"{stats['non_shoulder_wr']:.1f}% win rate")
    print()
    print(f"  Avg net delta:     {stats['avg_delta']:.1f}")
    print(f"  Max |delta|:       {stats['max_delta']:.1f}")
    print(f"  Avg UNG IV-RV:     {stats['avg_ung_vrp']:.1f}pp")
    print(f"  Avg KOLD IV-RV:    {stats['avg_kold_vrp']:.1f}pp")
    print()
    print("  Exit reasons:")
    for reason, count in sorted(stats['reason_counts'].items()):
        print(f"    {reason:20s} {count:>4d}")
    print()
    print("  Monthly win rate:")
    month_names = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    for m in range(1, 13):
        if m in stats['monthly_wr']:
            mw = stats['monthly_wr'][m]
            wr = mw['wins'] / mw['total'] * 100 if mw['total'] > 0 else 0
            bar = '█' * int(wr / 5)
            shoulder_flag = ' ◄' if m in (3, 4, 5, 9, 10, 11) else ''
            print(f"    {month_names[m]:>3s}: {wr:5.1f}% "
                  f"({mw['wins']}/{mw['total']}) {bar}{shoulder_flag}")
    print("=" * 65)


# ── Chart ────────────────────────────────────────────────────────────

def plot_results(log_df, stats, save_path='theta_harvest_backtest.png'):
    """Create 4×2 GridSpec chart with dark theme."""
    C_BG = '#0d1117'
    C_PANEL = '#161b22'
    C_TEXT = '#e6edf3'
    C_GRID = '#21262d'
    C_GREEN = '#3fb950'
    C_RED = '#f85149'
    C_ORANGE = '#f0883e'
    C_GOLD = '#ffd700'
    C_BLUE = '#58a6ff'
    C_PURPLE = '#bc8cff'
    fig = plt.figure(figsize=(22, 28), facecolor=C_BG)
    gs = gridspec.GridSpec(4, 2, hspace=0.30, wspace=0.22,
                           left=0.06, right=0.96, top=0.95, bottom=0.03)

    def style_ax(ax, title=''):
        ax.set_facecolor(C_PANEL)
        ax.tick_params(colors=C_TEXT, labelsize=9)
        ax.set_title(title, color=C_TEXT, fontsize=12, fontweight='bold',
                     pad=8)
        for spine in ax.spines.values():
            spine.set_color(C_GRID)
        ax.grid(True, color=C_GRID, alpha=0.4, linewidth=0.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    def shade_shoulder(ax, dates):
        """Shade shoulder season regions."""
        in_shoulder = False
        start = None
        for d in dates:
            if is_shoulder(d) and not in_shoulder:
                start = d
                in_shoulder = True
            elif not is_shoulder(d) and in_shoulder:
                ax.axvspan(start, d, alpha=0.15, color=C_BLUE, zorder=0)
                in_shoulder = False
        if in_shoulder:
            ax.axvspan(start, dates.iloc[-1], alpha=0.15, color=C_BLUE,
                       zorder=0)

    dates = log_df['date']

    # [0,0] Cumulative P&L + premium line
    ax = fig.add_subplot(gs[0, 0])
    style_ax(ax, 'Cumulative P&L (Total) + Realized')
    shade_shoulder(ax, dates)
    ax.plot(dates, log_df['total_pnl'], color=C_GREEN, linewidth=1.5,
            label='Total P&L (realized + unrealized)')
    ax.plot(dates, log_df['realized_pnl'], color=C_GOLD, linewidth=1.0,
            alpha=0.7, label='Realized P&L')
    ax.axhline(0, color=C_TEXT, alpha=0.3, linewidth=0.5)
    ax.fill_between(dates, 0, log_df['total_pnl'],
                     where=log_df['total_pnl'] > 0,
                     alpha=0.15, color=C_GREEN)
    ax.fill_between(dates, 0, log_df['total_pnl'],
                     where=log_df['total_pnl'] < 0,
                     alpha=0.15, color=C_RED)
    ax.legend(fontsize=8, loc='upper left',
              facecolor=C_PANEL, edgecolor=C_GRID, labelcolor=C_TEXT)
    ax.set_ylabel('P&L ($)', color=C_TEXT, fontsize=10)

    # [0,1] Drawdown from peak
    ax = fig.add_subplot(gs[0, 1])
    style_ax(ax, 'Drawdown from Peak')
    shade_shoulder(ax, dates)
    peak = log_df['total_pnl'].cummax()
    drawdown = log_df['total_pnl'] - peak
    ax.fill_between(dates, 0, drawdown, color=C_RED, alpha=0.5)
    ax.plot(dates, drawdown, color=C_RED, linewidth=0.8)
    ax.set_ylabel('Drawdown ($)', color=C_TEXT, fontsize=10)

    # [1,0] Vol regime: RV vs est-IV + edge
    ax = fig.add_subplot(gs[1, 0])
    style_ax(ax, 'UNG Volatility: RV vs Estimated IV')
    shade_shoulder(ax, dates)
    ax.plot(dates, log_df['ung_rv'] * 100, color=C_ORANGE, linewidth=1.0,
            alpha=0.8, label='30d RV')
    ax.plot(dates, log_df['ung_iv'] * 100, color=C_PURPLE, linewidth=1.0,
            alpha=0.8, label='Est. IV')
    ax.fill_between(dates, log_df['ung_rv'] * 100, log_df['ung_iv'] * 100,
                     alpha=0.15, color=C_GREEN, label='VRP (edge)')
    ax.legend(fontsize=8, loc='upper right',
              facecolor=C_PANEL, edgecolor=C_GRID, labelcolor=C_TEXT)
    ax.set_ylabel('Volatility (%)', color=C_TEXT, fontsize=10)

    # [1,1] Trade scatter (win/loss by date)
    ax = fig.add_subplot(gs[1, 1])
    style_ax(ax, 'Trade P&L Scatter')
    if stats and stats.get('trade_pairs'):
        tp = stats['trade_pairs']
        win_dates = [t['entry_date'] for t in tp if t['pnl'] > 0]
        win_pnl = [t['pnl'] for t in tp if t['pnl'] > 0]
        loss_dates = [t['entry_date'] for t in tp if t['pnl'] <= 0]
        loss_pnl = [t['pnl'] for t in tp if t['pnl'] <= 0]
        ax.scatter(win_dates, win_pnl, color=C_GREEN, s=25, alpha=0.7,
                   label=f'Wins ({len(win_dates)})', zorder=3)
        ax.scatter(loss_dates, loss_pnl, color=C_RED, s=25, alpha=0.7,
                   label=f'Losses ({len(loss_dates)})', zorder=3)
        ax.axhline(0, color=C_TEXT, alpha=0.3, linewidth=0.5)
        ax.legend(fontsize=8, loc='upper left',
                  facecolor=C_PANEL, edgecolor=C_GRID, labelcolor=C_TEXT)
    ax.set_ylabel('Trade P&L ($)', color=C_TEXT, fontsize=10)

    # [2,0] Net delta exposure over time
    ax = fig.add_subplot(gs[2, 0])
    style_ax(ax, 'Net Delta Exposure')
    shade_shoulder(ax, dates)
    ax.plot(dates, log_df['net_delta'], color=C_BLUE, linewidth=0.8,
            alpha=0.8)
    ax.fill_between(dates, 0, log_df['net_delta'],
                     where=log_df['net_delta'] > 0,
                     alpha=0.2, color=C_GREEN)
    ax.fill_between(dates, 0, log_df['net_delta'],
                     where=log_df['net_delta'] < 0,
                     alpha=0.2, color=C_RED)
    ax.axhline(0, color=C_TEXT, alpha=0.3, linewidth=0.5)
    ax.set_ylabel('Delta (shares)', color=C_TEXT, fontsize=10)

    # [2,1] Monthly win rate bars
    ax = fig.add_subplot(gs[2, 1])
    ax.set_facecolor(C_PANEL)
    ax.tick_params(colors=C_TEXT, labelsize=9)
    ax.set_title('Monthly Win Rate', color=C_TEXT, fontsize=12,
                 fontweight='bold', pad=8)
    for spine in ax.spines.values():
        spine.set_color(C_GRID)
    ax.grid(True, color=C_GRID, alpha=0.4, linewidth=0.5, axis='y')

    if stats and stats.get('monthly_wr'):
        months = list(range(1, 13))
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        wr_vals = []
        colors = []
        for m in months:
            if m in stats['monthly_wr']:
                mw = stats['monthly_wr'][m]
                wr = mw['wins'] / mw['total'] * 100 if mw['total'] > 0 else 0
            else:
                wr = 0
            wr_vals.append(wr)
            # Shoulder months in blue, others in orange
            colors.append(C_BLUE if m in (3, 4, 5, 9, 10, 11) else C_ORANGE)
        ax.bar(month_names, wr_vals, color=colors, alpha=0.8,
               edgecolor=C_GRID)
        ax.axhline(50, color=C_TEXT, alpha=0.3, linewidth=0.5,
                   linestyle='--')
        ax.set_ylabel('Win Rate (%)', color=C_TEXT, fontsize=10)
        ax.set_ylim(0, 100)
        # Add count labels
        for idx, m in enumerate(months):
            if m in stats['monthly_wr']:
                mw = stats['monthly_wr'][m]
                ax.text(idx, wr_vals[idx] + 2,
                        f"{mw['total']}", ha='center', va='bottom',
                        color=C_TEXT, fontsize=7)

    # [3,0] Rolling 6mo IV-RV spread
    ax = fig.add_subplot(gs[3, 0])
    style_ax(ax, 'Rolling 6-Month Avg IV-RV Spread (VRP)')
    shade_shoulder(ax, dates)
    ung_roll = (log_df['ung_vrp'] * 100).rolling(126, min_periods=30).mean()
    kold_roll = (log_df['kold_vrp'] * 100).rolling(126, min_periods=30).mean()
    ax.plot(dates, ung_roll, color=C_ORANGE, linewidth=1.2,
            label='UNG VRP (6mo avg)')
    ax.plot(dates, kold_roll, color=C_PURPLE, linewidth=1.2,
            label='KOLD VRP (6mo avg)')
    ax.axhline(0, color=C_TEXT, alpha=0.3, linewidth=0.5)
    ax.legend(fontsize=8, loc='upper right',
              facecolor=C_PANEL, edgecolor=C_GRID, labelcolor=C_TEXT)
    ax.set_ylabel('IV - RV (pp)', color=C_TEXT, fontsize=10)

    # [3,1] Stats summary text
    ax = fig.add_subplot(gs[3, 1])
    ax.set_facecolor(C_PANEL)
    ax.axis('off')
    ax.set_title('Summary Statistics', color=C_TEXT, fontsize=12,
                 fontweight='bold', pad=8)

    if stats:
        pf = stats['profit_factor']
        pf_str = f"{pf:.2f}" if pf < 100 else "inf"
        lines = [
            f"Trade Pairs:        {stats['total_trades']}",
            f"Total Premium:     ${stats['total_premium']:,.0f}",
            f"Total P&L:         ${stats['total_pnl']:,.0f}",
            f"Win Rate:           {stats['win_rate']:.1f}%",
            f"Avg Win:           ${stats['avg_win']:,.0f}",
            f"Avg Loss:          ${stats['avg_loss']:,.0f}",
            f"Profit Factor:      {pf_str}",
            f"Sharpe Ratio:       {stats['sharpe']:.2f}",
            f"Max Drawdown:      ${stats['max_drawdown']:,.0f}",
            "",
            f"Shoulder WR:        {stats['shoulder_wr']:.1f}% "
            f"({stats['shoulder_trades']} trades)",
            f"Non-Shoulder WR:    {stats['non_shoulder_wr']:.1f}% "
            f"({stats['non_shoulder_trades']} trades)",
            "",
            f"Avg Net Delta:      {stats['avg_delta']:.1f}",
            f"Max |Delta|:        {stats['max_delta']:.1f}",
            f"Avg UNG VRP:        {stats['avg_ung_vrp']:.1f}pp",
            f"Avg KOLD VRP:       {stats['avg_kold_vrp']:.1f}pp",
        ]
        # Exit reasons
        lines.append("")
        lines.append("Exit Reasons:")
        for reason, count in sorted(stats['reason_counts'].items()):
            lines.append(f"  {reason:18s} {count:>4d}")

        text = '\n'.join(lines)
        ax.text(0.05, 0.95, text, transform=ax.transAxes,
                fontfamily='monospace', fontsize=9, color=C_TEXT,
                verticalalignment='top')

    # Title
    fig.suptitle('Theta Harvest Backtest: Delta-Neutral Put Selling on UNG + KOLD',
                 color=C_GOLD, fontsize=16, fontweight='bold', y=0.98)

    # Legend note for shoulder shading
    fig.text(0.5, 0.005,
             'Blue shading = shoulder season (Mar-May, Sep-Nov)  |  '
             'Blue bars = shoulder months  |  Orange bars = non-shoulder',
             ha='center', color=C_TEXT, fontsize=8, alpha=0.6)

    plt.savefig(save_path, dpi=150, facecolor=C_BG,
                bbox_inches='tight')
    print(f"\n  Chart saved: {save_path}")
    plt.close()


# ── Calibration Check ────────────────────────────────────────────────

def calibration_check(df, rv_premium):
    """Print calibration: compare estimated IV to known current IV."""
    ung_rv = df['ung_rv'].iloc[-1]
    kold_rv = df['kold_rv'].iloc[-1]
    ung_rv_med = df['ung_rv'].median()
    kold_rv_med = df['kold_rv'].median()

    print("\n  Calibration Check (RV × {:.2f} = est. IV):".format(rv_premium))
    print(f"    UNG  30d RV median: {ung_rv_med*100:.0f}%, "
          f"current: {ung_rv*100:.0f}%, "
          f"est. IV: {ung_rv*rv_premium*100:.0f}% "
          f"(actual ~75-100%)")
    print(f"    KOLD 30d RV median: {kold_rv_med*100:.0f}%, "
          f"current: {kold_rv*100:.0f}%, "
          f"est. IV: {kold_rv*rv_premium*100:.0f}% "
          f"(actual ~145-170%)")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Theta Harvest Backtest: Delta-neutral put selling '
                    'on UNG + KOLD')
    parser.add_argument('--start', type=str, default=None,
                        help='Start date YYYY-MM-DD (default: all available)')
    parser.add_argument('--rv-premium', type=float, default=1.30,
                        help='IV/RV risk premium multiplier (default: 1.30)')
    parser.add_argument('--delta', type=float, default=0.20,
                        help='Target put delta (default: 0.20)')
    parser.add_argument('--ung-contracts', type=int, default=2,
                        help='Number of UNG put contracts (default: 2)')
    parser.add_argument('--entry-interval', type=int, default=14,
                        help='Days between new entries (default: 14)')
    parser.add_argument('--slippage', type=float, default=0.05,
                        help='Bid/ask slippage fraction (default: 0.05)')
    args = parser.parse_args()

    print("\n  Theta Harvest Backtest — UNG + KOLD Delta-Neutral Put Selling")
    print("  " + "─" * 55)
    print(f"  RV premium: {args.rv_premium:.2f}x  |  Target delta: "
          f"{args.delta:.2f}  |  UNG contracts: {args.ung_contracts}")
    print(f"  Entry interval: {args.entry_interval}d  |  Slippage: "
          f"{args.slippage*100:.0f}%")
    print()

    # Fetch data
    print("  Fetching market data...")
    df = fetch_data(args.start)

    # Run backtest
    print("\n  Running backtest...")
    positions, log_df, df_full = run_backtest(
        df,
        rv_premium=args.rv_premium,
        target_delta=args.delta,
        ung_contracts=args.ung_contracts,
        entry_interval=args.entry_interval,
        slippage=args.slippage,
    )

    # Calibration
    calibration_check(df_full, args.rv_premium)

    # Stats
    stats = compute_stats(positions, log_df)
    print_stats(stats)

    # Chart
    print("  Generating chart...")
    save_path = '/home/wyatt/ibkr_guided_trade/theta_harvest_backtest.png'
    plot_results(log_df, stats, save_path)

    print("  Done.\n")


if __name__ == '__main__':
    main()
