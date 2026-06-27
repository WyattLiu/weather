#!/usr/bin/env python3
"""
UNG Wheel Optimizer Backtest
==============================
Tests different scoring weight configurations and delta targeting strategies
over 5+ years of UNG history to find optimal wheel parameters.

Tests:
  1. Optimal scoring weights for the recommendation engine
  2. Whether dynamic delta targeting adds value vs static
  3. How much smoothness (concentration) matters vs theta maximization
  4. Best DTE range and roll timing

Uses Black-Scholes for all option pricing. IV estimated as 1.20 * realized vol.
Starts from $100k cash. Sells cash-secured puts to enter, covered calls when
assigned shares. Strictly enforces cash-secured constraint.
"""

import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from copy import deepcopy
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =============================================================================
# Black-Scholes
# =============================================================================

def bs_price(S, K, T, r, sigma, right='P'):
    if T <= 1e-6:
        return max(0.0, K - S) if right == 'P' else max(0.0, S - K)
    sigma = max(sigma, 0.01)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if right == 'C':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_delta(S, K, T, r, sigma, right='P'):
    if T <= 1e-6:
        if right == 'P':
            return -1.0 if K > S else 0.0
        else:
            return 1.0 if S > K else 0.0
    sigma = max(sigma, 0.01)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) if right == 'C' else norm.cdf(d1) - 1.0


def bs_gamma(S, K, T, r, sigma):
    if T <= 1e-6:
        return 0.0
    sigma = max(sigma, 0.01)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def bs_theta(S, K, T, r, sigma, right='P'):
    if T <= 1e-6:
        return 0.0
    sigma = max(sigma, 0.01)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    term1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    if right == 'C':
        theta = term1 - r * K * np.exp(-r * T) * norm.cdf(d2)
    else:
        theta = term1 + r * K * np.exp(-r * T) * norm.cdf(-d2)
    return theta / 252


def snap_strike(price):
    return round(price * 2) / 2


# =============================================================================
# Wheel Backtest
# =============================================================================

class WheelBacktest:
    """
    Cash-secured wheel on UNG. Position sizing strictly enforced:
    - Total put collateral (strike * contracts * 100) <= free_cash
    - Covered calls only against shares we hold
    - free_cash = cash (actual cash balance, always >= 0 after trades)

    When a put is SOLD: premium goes into cash, but the collateral (strike*100*qty)
    is "reserved" from the free cash pool.

    When a put is ASSIGNED: cash goes down by strike*shares, shares go up.
    This is a wash against the reserved collateral.

    Key insight: at any time, free_cash_for_new_puts =
        cash - sum(strike * qty * 100 for active short puts)
    This must stay >= 0. If it would go negative, reject the trade.
    """

    def __init__(self, config):
        self.config = config
        self.cash = 100_000.0
        self.shares = 0
        self.positions = []
        self.daily_records = []
        self.theta_collected = 0.0
        self.premium_collected = 0.0
        self.put_assign_count = 0
        self.call_assign_count = 0
        self.roll_count = 0
        self.n_trades = 0
        self.r = 0.04
        self.ma_100 = 0.0

    def run(self, prices, ivs, ma100):
        n = len(prices)
        for day_idx in range(n):
            price = prices[day_idx]
            iv = ivs[day_idx]
            self.ma_100 = ma100[day_idx] if not np.isnan(ma100[day_idx]) else price

            self._handle_expirations(day_idx, price)
            if day_idx % 5 == 0:
                self._evaluate_trades(day_idx, price, iv, n)

            equity = self._equity(day_idx, price, iv)
            pdelta = self._port_delta(day_idx, price, iv)
            self.daily_records.append({
                'day': day_idx, 'price': price, 'equity': equity,
                'shares': self.shares, 'cash': self.cash,
                'n_positions': len(self.positions), 'delta': pdelta,
            })
            if day_idx > 0 and day_idx % 500 == 0:
                fc = self._free_cash()
                print(f"    Day {day_idx}/{n}: eq=${equity:,.0f}, "
                      f"sh={self.shares}, pos={len(self.positions)}, "
                      f"d={pdelta:.0f}, cash=${self.cash:,.0f}, free=${fc:,.0f}")

        return self._results(prices)

    # ---- Free cash: cash not reserved for put collateral ----
    def _free_cash(self):
        """Cash available for new put collateral."""
        reserved = sum(p['strike'] * p['qty'] * 100
                       for p in self.positions if p['right'] == 'P')
        return self.cash - reserved

    # ---- Expirations ----
    def _handle_expirations(self, day_idx, price):
        surviving = []
        for pos in self.positions:
            if day_idx < pos['expiry_day']:
                surviving.append(pos)
                continue
            ns = pos['qty'] * 100
            if pos['right'] == 'P':
                if price < pos['strike']:
                    # Put assigned: cash was already reserved as collateral.
                    # Cash decreases by strike*shares, shares increase.
                    self.cash -= pos['strike'] * ns
                    self.shares += ns
                    self.put_assign_count += 1
                # OTM: collateral released, premium kept
            else:
                if price > pos['strike']:
                    called = min(ns, self.shares)
                    self.shares -= called
                    self.cash += pos['strike'] * called
                    self.call_assign_count += 1
        self.positions = surviving

    # ---- Trade Evaluation ----
    def _evaluate_trades(self, day_idx, price, iv, max_days):
        target_dte = self.config['target_dte']
        T_new = target_dte / 252.0
        expiry_day = min(day_idx + target_dte, max_days - 1)
        if expiry_day <= day_idx + 5:
            return

        pdelta = self._port_delta(day_idx, price, iv)
        tdelta = self._target_delta(price)
        free = self._free_cash()
        candidates = []

        # ---- ROLL candidates ----
        for i, pos in enumerate(self.positions):
            days_left = pos['expiry_day'] - day_idx
            if days_left <= 0:
                continue
            T_old = days_left / 252.0

            old_val = bs_price(price, pos['strike'], T_old, self.r, iv, pos['right'])
            intrinsic = (max(0, pos['strike'] - price) if pos['right'] == 'P'
                         else max(0, price - pos['strike']))
            extrinsic = max(0, old_val - intrinsic)
            ext_pct = (extrinsic / max(old_val, 0.01)) * 100

            # Only roll when near expiry, low extrinsic, or significantly ITM
            if not (days_left <= 7 or ext_pct < 25 or
                    abs(pos['strike'] - price) / max(price, 0.01) > 0.12):
                continue

            for offset in [-0.50, 0, 0.50]:
                ns = snap_strike(price + offset)
                if ns <= 0:
                    continue
                new_val = bs_price(price, ns, T_new, self.r, iv, pos['right'])
                old_theta = abs(bs_theta(price, pos['strike'], T_old, self.r, iv, pos['right'])) * pos['qty'] * 100
                new_theta = abs(bs_theta(price, ns, T_new, self.r, iv, pos['right'])) * pos['qty'] * 100
                old_d = bs_delta(price, pos['strike'], T_old, self.r, iv, pos['right']) * pos['qty'] * 100
                new_d = bs_delta(price, ns, T_new, self.r, iv, pos['right']) * pos['qty'] * 100
                old_g = bs_gamma(price, pos['strike'], T_old, self.r, iv) * pos['qty'] * 100
                new_g = bs_gamma(price, ns, T_new, self.r, iv) * pos['qty'] * 100

                candidates.append({
                    'type': 'ROLL', 'pos_idx': i, 'right': pos['right'],
                    'target_strike': ns, 'expiry_day': expiry_day,
                    'theta_change': new_theta - old_theta,
                    'delta_change': -(new_d - old_d),
                    'gamma_change': new_g - old_g,
                    'qty': pos['qty'], 'n_legs': 2,
                    'new_premium': new_val, 'old_val': old_val,
                })

        # ---- ADD PUT ----
        if pdelta < tdelta and free > 0:
            for spct in [0.95, 1.0, 1.05]:
                strike = snap_strike(price * spct)
                if strike <= 0:
                    continue
                collateral_per = strike * 100
                max_qty = min(int(free / max(collateral_per, 1)), 5)
                if max_qty < 1:
                    continue
                put_d_per = abs(bs_delta(price, strike, T_new, self.r, iv, 'P')) * 100
                gap = tdelta - pdelta
                qty = max(1, min(max_qty, int(gap / max(put_d_per, 1))))

                prem = bs_price(price, strike, T_new, self.r, iv, 'P')
                candidates.append({
                    'type': 'ADD_PUT', 'right': 'P',
                    'target_strike': strike, 'expiry_day': expiry_day,
                    'theta_change': abs(bs_theta(price, strike, T_new, self.r, iv, 'P')) * qty * 100,
                    'delta_change': put_d_per * qty,
                    'gamma_change': bs_gamma(price, strike, T_new, self.r, iv) * qty * 100,
                    'qty': qty, 'n_legs': 1, 'new_premium': prem,
                })

        # ---- ADD CALL (covered) ----
        covered = self.shares - sum(p['qty'] * 100 for p in self.positions if p['right'] == 'C')
        if covered >= 100 and pdelta > tdelta * 0.7:
            for spct in [1.0, 1.05, 1.10]:
                strike = snap_strike(price * spct)
                if strike <= 0:
                    continue
                qty = min(covered // 100, 3)
                prem = bs_price(price, strike, T_new, self.r, iv, 'C')
                cd = bs_delta(price, strike, T_new, self.r, iv, 'C') * qty * 100
                candidates.append({
                    'type': 'ADD_CALL', 'right': 'C',
                    'target_strike': strike, 'expiry_day': expiry_day,
                    'theta_change': abs(bs_theta(price, strike, T_new, self.r, iv, 'C')) * qty * 100,
                    'delta_change': -cd,
                    'gamma_change': bs_gamma(price, strike, T_new, self.r, iv) * qty * 100,
                    'qty': qty, 'n_legs': 1, 'new_premium': prem,
                })

        if not candidates:
            return

        scored = [(self._score(c, price, iv, pdelta, tdelta), c) for c in candidates]
        scored.sort(key=lambda x: -x[0])

        budget = self.config.get('max_trades_per_week', 3)
        min_score = self.config.get('min_score', 0.0)
        rolled = set()

        for score, trade in scored:
            if budget <= 0:
                break
            if score < min_score:
                continue
            if trade['type'] == 'ROLL' and trade['pos_idx'] in rolled:
                continue
            if self._execute(trade, day_idx, price, iv):
                budget -= 1
                if trade['type'] == 'ROLL':
                    rolled.add(trade['pos_idx'])

    # ---- Scoring ----
    def _score(self, trade, price, iv, pdelta, tdelta):
        w = self.config['weights']
        theta_s = trade['theta_change'] * w['theta']
        conc = self._concentration()
        conc_s = max(0, conc - 0.30) * w['concentration'] * 50
        gap_b = abs(pdelta - tdelta)
        gap_a = abs(pdelta + trade.get('delta_change', 0) - tdelta)
        delta_s = (gap_b - gap_a) / max(abs(tdelta), 100) * w['delta_targeting'] * 100
        gamma_s = -abs(trade.get('gamma_change', 0)) * w['gamma'] * 0.01
        spread = trade.get('n_legs', 1) * 0.03 * trade.get('qty', 1) * price * 10
        spread_s = -spread / 10000 * w['spread_cost']
        return theta_s + conc_s + delta_s + gamma_s + spread_s

    # ---- Execution ----
    def _execute(self, trade, day_idx, price, iv):
        spread_pct = 0.03

        if trade['type'] == 'ROLL':
            idx = trade['pos_idx']
            if idx >= len(self.positions):
                return False
            old = self.positions[idx]
            days_left = max(old['expiry_day'] - day_idx, 1)
            T_old = days_left / 252.0
            close_val = bs_price(price, old['strike'], T_old, self.r, iv, old['right'])

            # Net cash effect of roll
            # Close: pay close_val * (1+spread)
            # Open: receive new_premium * (1-spread)
            close_cost = close_val * (1 + spread_pct) * old['qty'] * 100
            open_credit = trade['new_premium'] * (1 - spread_pct) * trade['qty'] * 100

            # For put rolls: check if new collateral fits
            if trade['right'] == 'P':
                # Old collateral released, new collateral reserved
                old_collateral = old['strike'] * old['qty'] * 100
                new_collateral = trade['target_strike'] * trade['qty'] * 100
                # Net change in free cash: +old_collateral - new_collateral + open_credit - close_cost
                free_after = self._free_cash() + old_collateral - new_collateral + open_credit - close_cost
                if free_after < 0:
                    return False

            self.cash -= close_cost
            self.cash += open_credit
            self.premium_collected += open_credit
            self.theta_collected += open_credit

            self.positions[idx] = {
                'entry_day': day_idx, 'expiry_day': trade['expiry_day'],
                'strike': trade['target_strike'], 'right': trade['right'],
                'qty': trade['qty'], 'entry_price': price,
                'premium': trade['new_premium'] * (1 - spread_pct),
            }
            self.roll_count += 1
            self.n_trades += 1
            return True

        elif trade['type'] in ('ADD_PUT', 'ADD_CALL'):
            qty = trade['qty']
            new_prem = trade['new_premium'] * (1 - spread_pct)

            if trade['right'] == 'P':
                new_collateral = trade['target_strike'] * qty * 100
                # Free cash after: current_free + premium_received - new_collateral
                free_after = self._free_cash() + new_prem * qty * 100 - new_collateral
                if free_after < 0:
                    return False
                if len(self.positions) >= 10:
                    return False
            elif trade['right'] == 'C':
                covered = self.shares - sum(
                    p['qty'] * 100 for p in self.positions if p['right'] == 'C')
                if qty * 100 > covered:
                    return False
                if len(self.positions) >= 10:
                    return False

            self.cash += new_prem * qty * 100
            self.premium_collected += new_prem * qty * 100
            self.theta_collected += new_prem * qty * 100

            self.positions.append({
                'entry_day': day_idx, 'expiry_day': trade['expiry_day'],
                'strike': trade['target_strike'], 'right': trade['right'],
                'qty': qty, 'entry_price': price, 'premium': new_prem,
            })
            self.n_trades += 1
            return True
        return False

    # ---- Metrics ----
    def _equity(self, day_idx, price, iv):
        eq = self.cash + self.shares * price
        for pos in self.positions:
            dl = max(pos['expiry_day'] - day_idx, 1)
            T = dl / 252.0
            eq -= bs_price(price, pos['strike'], T, self.r, iv, pos['right']) * pos['qty'] * 100
        return eq

    def _port_delta(self, day_idx, price, iv):
        total = float(self.shares)
        for pos in self.positions:
            dl = max(pos['expiry_day'] - day_idx, 1)
            T = dl / 252.0
            d = bs_delta(price, pos['strike'], T, self.r, iv, pos['right'])
            total -= d * pos['qty'] * 100  # short: negate
        return total

    def _target_delta(self, price):
        if self.config.get('dynamic_delta', False):
            ma = self.ma_100
            if ma <= 0:
                return self.config.get('static_delta', 3000)
            ratio = price / ma
            mn = self.config.get('delta_min', 1500)
            mx = self.config.get('delta_max', 5000)
            t = mx - (ratio - 0.75) / (1.25 - 0.75) * (mx - mn)
            return max(mn, min(mx, t))
        return self.config.get('static_delta', 3000)

    def _concentration(self):
        if not self.positions:
            return 0.0
        buckets = defaultdict(int)
        for p in self.positions:
            buckets[p['expiry_day'] // 5] += p['qty']
        total = sum(buckets.values())
        return max(buckets.values()) / max(total, 1) if buckets else 0.0

    # ---- Results ----
    def _results(self, prices):
        df = pd.DataFrame(self.daily_records)
        if len(df) < 50:
            return None
        equity = df['equity'].values.astype(float)
        n_days = len(equity)
        n_years = n_days / 252.0

        # Use log returns for stability
        eq_pos = np.maximum(equity, 1.0)  # floor for log
        total_ret = equity[-1] / equity[0] - 1
        ann_ret = (equity[-1] / equity[0]) ** (1 / max(n_years, 0.1)) - 1 if equity[-1] > 0 and equity[0] > 0 else -1.0

        dr = np.diff(eq_pos) / eq_pos[:-1]
        dr = dr[np.isfinite(dr)]
        sharpe = np.mean(dr) / max(np.std(dr), 1e-8) * np.sqrt(252) if len(dr) > 20 else 0

        peak = np.maximum.accumulate(eq_pos)
        dd = (eq_pos - peak) / peak
        max_dd = dd.min()

        prices[-1] / prices[0] - 1
        bnh_ann = (prices[-1] / prices[0]) ** (1 / max(n_years, 0.1)) - 1 if prices[-1] > 0 else -1
        alpha = ann_ret - bnh_ann

        meq = equity[::21]
        win_rate = (np.mean(np.diff(meq) / np.maximum(meq[:-1], 1) > 0) * 100
                    if len(meq) > 2 else 0)

        avg_theta = self.theta_collected / max(n_years * 12, 1)
        avg_npos = np.mean(df['n_positions'].values)

        return {
            'name': self.config['name'],
            'total_return': total_ret * 100,
            'ann_return': ann_ret * 100,
            'alpha': alpha * 100,
            'sharpe': sharpe,
            'max_drawdown': max_dd * 100,
            'avg_theta_monthly': avg_theta,
            'premium_collected': self.premium_collected,
            'n_trades': self.n_trades,
            'n_put_assigns': self.put_assign_count,
            'n_call_assigns': self.call_assign_count,
            'n_rolls': self.roll_count,
            'win_rate': win_rate,
            'avg_positions': avg_npos,
            'equity_curve': equity,
            'daily_df': df,
            'bnh_ann_return': bnh_ann * 100,
        }


# =============================================================================
# Data
# =============================================================================

def fetch_ung_data():
    print("Fetching UNG data from yfinance...")
    ung = yf.Ticker('UNG')
    hist = ung.history(start='2015-01-01', end='2026-12-31', auto_adjust=True)
    if hist.empty:
        print("ERROR: No data")
        sys.exit(1)

    df = pd.DataFrame({
        'date': hist.index,
        'close': hist['Close'].values,
    }).reset_index(drop=True)

    print(f"  {len(df)} bars: {df['date'].iloc[0].strftime('%Y-%m-%d')} "
          f"to {df['date'].iloc[-1].strftime('%Y-%m-%d')}")

    lr = np.log(df['close'] / df['close'].shift(1))
    df['rv_20'] = lr.rolling(20).std() * np.sqrt(252)
    df['iv_proxy'] = (df['rv_20'] * 1.20).clip(0.15, 2.5).fillna(0.50)
    df['ma_100'] = df['close'].rolling(100).mean()
    df['ma_200'] = df['close'].rolling(200).mean()
    return df


# =============================================================================
# Configurations
# =============================================================================

CONFIGS = [
    {
        'name': 'Baseline (equal weights)',
        'weights': {'theta': 1.0, 'concentration': 1.0, 'delta_targeting': 0.0,
                    'gamma': 1.0, 'spread_cost': 1.0},
        'dynamic_delta': False, 'static_delta': 3000,
        'target_dte': 30, 'delta_min': 2000, 'delta_max': 5000,
        'max_trades_per_week': 2, 'min_score': 0.0,
    },
    {
        'name': 'Theta-heavy',
        'weights': {'theta': 3.0, 'concentration': 0.3, 'delta_targeting': 0.0,
                    'gamma': 0.3, 'spread_cost': 0.5},
        'dynamic_delta': False, 'static_delta': 4000,
        'target_dte': 30, 'delta_min': 2000, 'delta_max': 5000,
        'max_trades_per_week': 3, 'min_score': 0.0,
    },
    {
        'name': 'Smoothness-heavy',
        'weights': {'theta': 0.5, 'concentration': 3.0, 'delta_targeting': 0.0,
                    'gamma': 1.5, 'spread_cost': 1.0},
        'dynamic_delta': False, 'static_delta': 2500,
        'target_dte': 30, 'delta_min': 2000, 'delta_max': 5000,
        'max_trades_per_week': 2, 'min_score': 0.0,
    },
    {
        'name': 'Dynamic delta (moderate)',
        'weights': {'theta': 1.5, 'concentration': 1.0, 'delta_targeting': 2.0,
                    'gamma': 0.5, 'spread_cost': 1.0},
        'dynamic_delta': True, 'static_delta': 3000,
        'target_dte': 30, 'delta_min': 1500, 'delta_max': 5000,
        'max_trades_per_week': 3, 'min_score': 0.0,
    },
    {
        'name': 'Dynamic delta (aggressive)',
        'weights': {'theta': 1.0, 'concentration': 0.5, 'delta_targeting': 3.0,
                    'gamma': 0.3, 'spread_cost': 0.3},
        'dynamic_delta': True, 'static_delta': 3000,
        'target_dte': 30, 'delta_min': 1000, 'delta_max': 6000,
        'max_trades_per_week': 4, 'min_score': 0.0,
    },
    {
        'name': '45 DTE target',
        'weights': {'theta': 1.5, 'concentration': 1.0, 'delta_targeting': 1.5,
                    'gamma': 0.5, 'spread_cost': 1.0},
        'dynamic_delta': True, 'static_delta': 3000,
        'target_dte': 45, 'delta_min': 1500, 'delta_max': 5000,
        'max_trades_per_week': 3, 'min_score': 0.0,
    },
    {
        'name': 'Minimal smoothness',
        'weights': {'theta': 2.5, 'concentration': 0.1, 'delta_targeting': 1.0,
                    'gamma': 0.3, 'spread_cost': 0.5},
        'dynamic_delta': True, 'static_delta': 3500,
        'target_dte': 30, 'delta_min': 1500, 'delta_max': 5000,
        'max_trades_per_week': 3, 'min_score': 0.0,
    },
    {
        'name': 'Max smoothness',
        'weights': {'theta': 0.5, 'concentration': 3.0, 'delta_targeting': 1.5,
                    'gamma': 1.5, 'spread_cost': 0.5},
        'dynamic_delta': True, 'static_delta': 2500,
        'target_dte': 30, 'delta_min': 1500, 'delta_max': 4000,
        'max_trades_per_week': 2, 'min_score': 0.0,
    },
]


# =============================================================================
# Reporting
# =============================================================================

def print_results_table(results):
    print("\n" + "=" * 145)
    print("OPTIMIZER BACKTEST RESULTS")
    print("=" * 145)

    hdr = (f"{'Config':<30s} {'AnnRet%':>8s} {'Alpha%':>8s} {'Sharpe':>7s} "
           f"{'MaxDD%':>8s} {'Theta/mo':>10s} {'Premium':>12s} {'Trades':>7s} "
           f"{'PutA':>6s} {'CallA':>6s} {'Rolls':>6s} {'WinRate':>8s} {'AvgPos':>7s}")
    print(hdr)
    print("-" * 145)

    for r in results:
        line = (f"{r['name']:<30s} {r['ann_return']:>8.1f} {r['alpha']:>8.1f} "
                f"{r['sharpe']:>7.2f} {r['max_drawdown']:>8.1f} "
                f"${r['avg_theta_monthly']:>9,.0f} ${r['premium_collected']:>11,.0f} "
                f"{r['n_trades']:>7d} "
                f"{r['n_put_assigns']:>6d} {r['n_call_assigns']:>6d} "
                f"{r['n_rolls']:>6d} {r['win_rate']:>7.1f}% {r['avg_positions']:>7.1f}")
        print(line)

    print("-" * 145)
    bnh = results[0]['bnh_ann_return']
    print(f"{'Buy & Hold UNG':<30s} {bnh:>8.1f} {'0.0':>8s}")
    print("=" * 145)

    print("\nBEST PERFORMERS:")
    for label, key in [('Best Ann. Return', 'ann_return'), ('Best Alpha', 'alpha'),
                       ('Best Sharpe', 'sharpe'), ('Smallest MaxDD', 'max_drawdown'),
                       ('Most Theta/mo', 'avg_theta_monthly'), ('Best Win Rate', 'win_rate')]:
        best = max(results, key=lambda x: x[key])
        v = best[key]
        if 'Theta' in label:
            print(f"  {label:20s} {best['name']} (${v:,.0f})")
        else:
            print(f"  {label:20s} {best['name']} ({v:.1f}{'%' if key != 'sharpe' else ''})")


def plot_results(results, prices, save_path):
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(22, 18))
    colors = ['#00d4ff', '#ff6b35', '#7ddf64', '#ffd166', '#ef476f',
              '#a78bfa', '#06d6a0', '#f72585']

    # P1: Equity curves
    ax1 = fig.add_subplot(3, 2, 1)
    for i, r in enumerate(results):
        eq = r['equity_curve'] / r['equity_curve'][0] * 100
        ax1.plot(eq, color=colors[i % 8], alpha=0.85, lw=1.2, label=r['name'][:25])
    ax1.plot(prices / prices[0] * 100, 'w--', alpha=0.4, lw=1.5, label='Buy & Hold')
    ax1.set_title('Equity Curves (Normalized)', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=6, loc='best', framealpha=0.7)
    ax1.grid(True, alpha=0.2)

    # P2: Drawdowns
    ax2 = fig.add_subplot(3, 2, 2)
    for i, r in enumerate(results):
        eq = np.maximum(r['equity_curve'], 1)
        pk = np.maximum.accumulate(eq)
        ax2.plot((eq - pk) / pk * 100, color=colors[i % 8], alpha=0.7, lw=1.0,
                 label=r['name'][:20])
    ax2.set_title('Drawdowns (%)', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=6, loc='best', framealpha=0.7)
    ax2.grid(True, alpha=0.2)

    # P3: Annualized returns
    ax3 = fig.add_subplot(3, 2, 3)
    names = [r['name'][:22] for r in results]
    yp = list(range(len(names)))
    ax3.barh(yp, [r['ann_return'] for r in results], color=colors[:len(results)], alpha=0.85)
    ax3.set_yticks(yp); ax3.set_yticklabels(names, fontsize=8)
    ax3.axvline(x=results[0]['bnh_ann_return'], color='w', ls='--', alpha=0.5,
                label=f"B&H: {results[0]['bnh_ann_return']:.1f}%")
    ax3.set_title('Annualized Return (%)', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=9); ax3.grid(True, alpha=0.2, axis='x')

    # P4: Sharpe
    ax4 = fig.add_subplot(3, 2, 4)
    ax4.barh(yp, [r['sharpe'] for r in results], color=colors[:len(results)], alpha=0.85)
    ax4.set_yticks(yp); ax4.set_yticklabels(names, fontsize=8)
    ax4.set_title('Sharpe Ratio', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.2, axis='x')

    # P5: Delta over time (first 4)
    ax5 = fig.add_subplot(3, 2, 5)
    for i, r in enumerate(results[:4]):
        d = pd.Series(r['daily_df']['delta'].values).rolling(20, min_periods=1).mean()
        ax5.plot(d.values, color=colors[i % 8], alpha=0.7, lw=1.0, label=r['name'][:25])
    ax5.set_title('Portfolio Delta (20d smoothed)', fontsize=13, fontweight='bold')
    ax5.legend(fontsize=7, loc='best', framealpha=0.7)
    ax5.grid(True, alpha=0.2)

    # P6: Alpha vs MaxDD scatter
    ax6 = fig.add_subplot(3, 2, 6)
    for i, r in enumerate(results):
        ax6.scatter(r['max_drawdown'], r['alpha'], color=colors[i % 8],
                    s=140, zorder=5, edgecolors='w', lw=0.5)
        ax6.annotate(r['name'][:18], (r['max_drawdown'], r['alpha']),
                     fontsize=7, color=colors[i % 8], xytext=(5, 5), textcoords='offset points')
    ax6.set_title('Alpha vs Max Drawdown', fontsize=13, fontweight='bold')
    ax6.set_xlabel('Max Drawdown (%)'); ax6.set_ylabel('Alpha vs B&H (%)')
    ax6.grid(True, alpha=0.2)

    fig.suptitle('UNG Wheel Optimizer Backtest', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"\nChart saved to {save_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("UNG WHEEL OPTIMIZER BACKTEST")
    print("=" * 70)

    df = fetch_ung_data()
    valid = df['ma_200'].notna() & (df['date'] >= '2018-01-01')
    if valid.sum() < 100:
        print("ERROR: Insufficient data"); sys.exit(1)

    start_idx = valid.idxmax()
    df_bt = df.iloc[start_idx:].reset_index(drop=True)
    print(f"\nBacktest: {df_bt['date'].iloc[0].strftime('%Y-%m-%d')} to "
          f"{df_bt['date'].iloc[-1].strftime('%Y-%m-%d')} "
          f"({len(df_bt)} days, {len(df_bt)/252:.1f} yr)")
    print(f"UNG: ${df_bt['close'].min():.2f} - ${df_bt['close'].max():.2f}")

    prices = df_bt['close'].values.astype(float)
    ivs = df_bt['iv_proxy'].values.astype(float)
    ma100 = df_bt['ma_100'].values.astype(float)

    results = []
    for i, config in enumerate(CONFIGS):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(CONFIGS)}] {config['name']}")
        print(f"{'='*60}")
        print(f"  Wts: theta={config['weights']['theta']}, conc={config['weights']['concentration']}, "
              f"delta_tgt={config['weights']['delta_targeting']}, gamma={config['weights']['gamma']}, "
              f"spread={config['weights']['spread_cost']}")
        print(f"  Dynamic={config.get('dynamic_delta')}, DTE={config['target_dte']}, "
              f"StaticD={config.get('static_delta')}, MaxTrades={config.get('max_trades_per_week')}")

        bt = WheelBacktest(deepcopy(config))
        result = bt.run(prices.copy(), ivs.copy(), ma100.copy())

        if result:
            results.append(result)
            print(f"  -> Ann={result['ann_return']:.1f}%, Alpha={result['alpha']:.1f}%, "
                  f"Sharpe={result['sharpe']:.2f}, MaxDD={result['max_drawdown']:.1f}%")
            print(f"     PutA={result['n_put_assigns']}, CallA={result['n_call_assigns']}, "
                  f"Rolls={result['n_rolls']}, Trades={result['n_trades']}")
        else:
            print("  -> FAILED")

    if not results:
        print("ERROR: No configs completed"); sys.exit(1)

    print_results_table(results)
    chart_path = '/home/wyatt/ibkr_guided_trade/ung_optimizer_backtest.png'
    plot_results(results, prices, chart_path)

    # ---- Analysis ----
    print("\n" + "=" * 70)
    print("WEIGHT SENSITIVITY ANALYSIS")
    print("=" * 70)

    # Static vs dynamic
    for i, r in enumerate(results):
        r['_cfg_idx'] = i
    static_r = [r for r in results if not CONFIGS[r['_cfg_idx']].get('dynamic_delta', False)]
    dynamic_r = [r for r in results if CONFIGS[r['_cfg_idx']].get('dynamic_delta', False)]
    if static_r and dynamic_r:
        sa = np.mean([r['alpha'] for r in static_r])
        da = np.mean([r['alpha'] for r in dynamic_r])
        ss = np.mean([r['sharpe'] for r in static_r])
        ds = np.mean([r['sharpe'] for r in dynamic_r])
        print(f"\n  Static delta:  avg alpha={sa:.1f}%, avg sharpe={ss:.2f} (n={len(static_r)})")
        print(f"  Dynamic delta: avg alpha={da:.1f}%, avg sharpe={ds:.2f} (n={len(dynamic_r)})")
        b = da - sa
        print(f"  -> Dynamic delta {'adds' if b > 0 else 'costs'} {abs(b):.1f}% alpha")

    # DTE
    dte30 = [r for r in results if '45 DTE' not in r['name']]
    dte45 = [r for r in results if '45 DTE' in r['name']]
    if dte30 and dte45:
        print(f"\n  30 DTE avg alpha: {np.mean([r['alpha'] for r in dte30]):.1f}%")
        print(f"  45 DTE avg alpha: {np.mean([r['alpha'] for r in dte45]):.1f}%")

    # Theta vs smoothness emphasis
    for name_match, label in [('Theta', 'Theta-heavy'), ('Smoothness', 'Smoothness-heavy'),
                               ('Minimal', 'Min smoothness'), ('Max smooth', 'Max smoothness')]:
        matches = [r for r in results if name_match in r['name']]
        for m in matches:
            print(f"\n  {label:22s} ann={m['ann_return']:.1f}%, alpha={m['alpha']:.1f}%, "
                  f"sharpe={m['sharpe']:.2f}, maxDD={m['max_drawdown']:.1f}%")

    # ---- Recommendation ----
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    def norm_m(vals):
        mn, mx = min(vals), max(vals)
        r = mx - mn
        return [(v - mn) / r if r > 1e-8 else 0.5 for v in vals]

    composite = np.zeros(len(results))
    for key in ['ann_return', 'alpha', 'sharpe', 'win_rate', 'max_drawdown']:
        composite += np.array(norm_m([r[key] for r in results]))

    best_i = int(np.argmax(composite))
    best = results[best_i]
    cfg = CONFIGS[best['_cfg_idx']]

    print(f"\n  Overall best: {best['name']}")
    print(f"    Ann. Return:    {best['ann_return']:.1f}%")
    print(f"    Alpha vs B&H:   {best['alpha']:.1f}%")
    print(f"    Sharpe:         {best['sharpe']:.2f}")
    print(f"    Max Drawdown:   {best['max_drawdown']:.1f}%")
    print(f"    Win Rate:       {best['win_rate']:.1f}%")
    print(f"    Theta/month:    ${best['avg_theta_monthly']:,.0f}")
    print(f"    Premium total:  ${best['premium_collected']:,.0f}")
    print(f"    Trades: {best['n_trades']}  PutA: {best['n_put_assigns']}  "
          f"CallA: {best['n_call_assigns']}  Rolls: {best['n_rolls']}")
    print(f"    Weights: {cfg['weights']}")
    print(f"    Dynamic delta: {cfg.get('dynamic_delta')}, DTE: {cfg['target_dte']}")

    print("\n  Composite ranking:")
    for rank, (idx, sc) in enumerate(sorted(enumerate(composite), key=lambda x: -x[1]), 1):
        r = results[idx]
        print(f"    {rank}. {r['name']:<30s}  score={sc:.2f}  "
              f"ann={r['ann_return']:.1f}%  alpha={r['alpha']:.1f}%  "
              f"sharpe={r['sharpe']:.2f}  maxDD={r['max_drawdown']:.1f}%")

    print("\nDone.")


if __name__ == '__main__':
    main()
