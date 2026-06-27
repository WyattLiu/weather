#!/usr/bin/env python3
"""
UNG Wheel Strategy Backtest v2
================================
Comprehensive backtest of the UNG wheel strategy with regime-based delta targeting.

Strategy:
  1. Start with $100k capital
  2. Regime model based on trailing 60-day return z-score:
       z > 0.5   (cheap):     0.8x leverage
       z 0-0.5   (fair):      0.6x leverage
       z -0.5-0  (rich):      0.3x leverage
       z < -0.5  (expensive): 0.1x leverage
     Target delta = leverage × NAV / spot (shares equivalent)
  3. Every 5 trading days (weekly rebalance):
     - Sell CSP puts at 0.25-0.30 delta, 22-30 DTE
     - If holding shares, sell covered calls at 0.30 delta
     - Roll puts when extrinsic < 30% of original extrinsic
     - Assignments: take them (acquire shares / sell shares)
  4. Black-Scholes pricing, fixed 50% IV (UNG historical average)
  5. Friction: $0.03/share per leg, cash earns 4% risk-free

Position sizing:
  - STRICT cash-secured puts: we reserve strike × contracts × 100 in cash
  - free_cash = total_cash - put_collateral_reserved - shares_cost
  - New puts only opened if free_cash can cover the full collateral
  - Covered calls require owning the underlying shares (no naked calls)

Benchmarks: buy-and-hold UNG, short UNG, risk-free only
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from datetime import timedelta
from collections import defaultdict


# ============================================================================
# Black-Scholes Engine
# ============================================================================

def bs_price(S, K, T, r, sigma, right='P'):
    """Black-Scholes option price. T in years."""
    if T <= 1e-6:
        return max(0.0, K - S) if right == 'P' else max(0.0, S - K)
    sigma = max(sigma, 0.005)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if right == 'C':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_delta(S, K, T, r, sigma, right='P'):
    """Black-Scholes delta. Put delta is negative."""
    if T <= 1e-6:
        if right == 'P':
            return -1.0 if K > S else 0.0
        else:
            return 1.0 if S > K else 0.0
    sigma = max(sigma, 0.005)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) if right == 'C' else norm.cdf(d1) - 1.0


def strike_for_delta(S, T, r, sigma, target_abs_delta, right='P'):
    """
    Bisection search for the strike that produces a given |delta|.
    target_abs_delta should be positive (e.g. 0.25 for a 0.25-delta put).
    """
    if T <= 1e-6:
        return S
    low, high = S * 0.1, S * 3.0
    for _ in range(60):
        mid = (low + high) / 2.0
        d = abs(bs_delta(S, mid, T, r, sigma, right))
        if right == 'P':
            # Lower strike → lower |put delta|
            if d > target_abs_delta:
                high = mid
            else:
                low = mid
        else:
            # Higher strike → lower call delta
            if d > target_abs_delta:
                low = mid
            else:
                high = mid
    return (low + high) / 2.0


def snap_strike(price, tick=0.50):
    """Snap to nearest option strike grid ($0.50 ticks)."""
    return round(price / tick) * tick


def extrinsic_value(price, S, K, right='P'):
    """Compute extrinsic (time) value of an option."""
    intrinsic = max(0.0, K - S) if right == 'P' else max(0.0, S - K)
    return max(0.0, price - intrinsic)


# ============================================================================
# Regime Model
# ============================================================================

def compute_regime(prices_series, lookback=60, window=252):
    """
    Compute rolling z-score of 60-day returns.
    Returns series of z-scores and regime labels.

    A high z-score means UNG has been going up (momentum), which in the
    context of a mean-reverting commodity means it may be 'rich' (expensive),
    but for trend-following it might be cheap.

    Here we use the simple convention from the spec:
      z > 0.5 → 'cheap' (high momentum, ride it with 0.8x)
      z 0-0.5 → 'fair'
      z -0.5-0 → 'rich' (slight downtrend, cautious)
      z < -0.5 → 'expensive' (strong downtrend, minimal exposure)

    Note: 'cheap/expensive' here refers to the trade opportunity, not
    the absolute price level.
    """
    ret60 = prices_series.pct_change(lookback)
    roll_mean = ret60.rolling(window, min_periods=90).mean()
    roll_std  = ret60.rolling(window, min_periods=90).std()
    z = (ret60 - roll_mean) / roll_std.clip(lower=1e-6)

    def label(z_val):
        if np.isnan(z_val):
            return 'fair'
        if z_val > 0.5:
            return 'cheap'
        elif z_val > 0.0:
            return 'fair'
        elif z_val > -0.5:
            return 'rich'
        else:
            return 'expensive'

    regime = z.apply(label)
    leverage_map = {'cheap': 0.8, 'fair': 0.6, 'rich': 0.3, 'expensive': 0.1}
    leverage = regime.map(leverage_map)

    return z, regime, leverage


# ============================================================================
# Main Backtest Engine
# ============================================================================

class WheelBacktest:
    """
    Simulates the UNG wheel strategy over a historical price series.

    Cash accounting:
      - `cash` = total cash (actual dollars in account)
      - put collateral is RESERVED from cash when we sell puts
      - `free_cash` = cash minus total put collateral reserved
      - When put expires OTM: collateral is released, net gain = premium
      - When put assigned: collateral used to buy shares (already reserved)
      - Covered calls: no extra cash needed (shares are collateral)

    The NAV accounts for:
      - Cash balance (already includes received premiums)
      - Mark-to-market of shares
      - Mark-to-market of short options (liability)
    """

    IV   = 0.50   # Fixed 50% IV for all pricing
    RFIR = 0.04   # 4% annual risk-free rate
    FRIC = 0.03   # $0.03/share friction per leg
    TARGET_DTE     = 26    # DTE target (midpoint of 22-30)
    TARGET_DELTA_P = 0.275 # Target put |delta| (midpoint of 0.25-0.30)
    TARGET_DELTA_C = 0.30  # Target call delta
    ROLL_PCT       = 0.30  # Roll when extrinsic < 30% of initial
    REBAL_DAYS     = 5     # Rebalance every 5 trading days

    def __init__(self, prices, regimes, leverages, start_capital=100_000):
        self.prices    = prices       # pd.Series of adjusted close
        self.regimes   = regimes      # dict: date -> regime label
        self.leverages = leverages    # dict: date -> leverage float

        self.cash      = float(start_capital)  # total cash
        self.shares    = 0            # shares of UNG held
        self.options   = []           # list of open short option dicts

        # Analytics
        self.nav_history     = []               # [(date, nav)]
        self.monthly_premium = defaultdict(float)   # (yr,mo) -> premium $
        self.regime_pnl      = defaultdict(float)   # regime -> realized pnl
        self.trade_log       = []

        self.total_premium     = 0.0
        self.trade_count       = 0
        self._last_rebal_idx   = -999
        self._current_date     = None

    # ------------------------------------------------------------------
    # Cash management helpers
    # ------------------------------------------------------------------

    def _put_collateral_reserved(self):
        """Total cash reserved as collateral for short puts."""
        return sum(o['strike'] * o['contracts'] * 100
                   for o in self.options if o['right'] == 'P')

    def _free_cash(self):
        """Cash available for new put collateral."""
        return self.cash - self._put_collateral_reserved()

    # ------------------------------------------------------------------
    # NAV computation
    # ------------------------------------------------------------------

    def _option_mtm_liability(self, S):
        """Current mark-to-market liability of short option book."""
        r, iv = self.RFIR, self.IV
        total = 0.0
        for opt in self.options:
            T = max(0.0, (opt['expiry'] - self._current_date).days / 365.0)
            curr_price = bs_price(S, opt['strike'], T, r, iv, opt['right'])
            # We received open_price, current cost to close is curr_price
            # Unrealized P&L per share = open_price - curr_price
            # Positive if option has decayed (we profit)
            total += (opt['open_price'] - curr_price) * opt['contracts'] * 100
        return total

    def nav(self, S):
        """Compute current portfolio NAV given spot price S."""
        return self.cash + self.shares * S + self._option_mtm_liability(S)

    # ------------------------------------------------------------------
    # Target delta based on regime
    # ------------------------------------------------------------------

    def _target_delta_shares(self, S):
        """
        Target net delta in shares equivalent.
        = leverage × NAV / spot
        Capped at NAV / spot (max 100% long).
        """
        lev = self.leverages.get(self._current_date, 0.6)
        n   = self.nav(S)
        if n <= 0:
            return 0.0
        return lev * n / S

    def _current_delta_shares(self, S):
        """
        Current net delta in shares equivalent.
        Shares have delta +1 each.
        Short puts add positive delta (we profit when UNG goes up).
        Short calls add negative delta.
        """
        r, iv = self.RFIR, self.IV
        delta = float(self.shares)
        for opt in self.options:
            T = max(0.0, (opt['expiry'] - self._current_date).days / 365.0)
            d_per_share = bs_delta(S, opt['strike'], T, r, iv, opt['right'])
            # Short option: flip sign (we are short)
            delta += -d_per_share * opt['contracts'] * 100
        return delta

    # ------------------------------------------------------------------
    # Open / close options
    # ------------------------------------------------------------------

    def _open_short(self, S, right, strike, expiry, contracts, date):
        """
        Open a short option position.
        Returns premium received (per-account net).
        Checks and reserves collateral for puts.
        """
        r, iv = self.RFIR, self.IV
        T = max(1.0 / 365.0, (expiry - date).days / 365.0)
        theo = bs_price(S, strike, T, r, iv, right)
        extrin = extrinsic_value(theo, S, strike, right)

        # Receive premium minus friction
        net_per_share = max(0.01, theo - self.FRIC)
        premium_total = net_per_share * contracts * 100

        self.cash += premium_total
        self.total_premium += premium_total
        ym = (date.year, date.month)
        self.monthly_premium[ym] += premium_total

        opt = {
            'right':             right,
            'strike':            strike,
            'expiry':            expiry,
            'contracts':         contracts,    # always positive (quantity)
            'open_price':        theo,         # theoretical price at open
            'initial_extrinsic': extrin,
            'open_date':         date,
        }
        self.options.append(opt)
        self.trade_count += 1

        self.trade_log.append({
            'date': date, 'action': f'SELL_{right}',
            'strike': strike, 'expiry': expiry, 'qty': contracts,
            'price': theo, 'premium': premium_total, 'S': S,
            'regime': self.regimes.get(date, 'fair'),
        })
        return premium_total

    def _close_short(self, opt, S, date, reason=''):
        """
        Close (buy back) a short option.
        Returns cost paid (positive = cash outflow).
        """
        r, iv = self.RFIR, self.IV
        T = max(0.0, (opt['expiry'] - date).days / 365.0)
        curr_price = bs_price(S, opt['strike'], T, r, iv, opt['right'])

        # Pay friction to close
        net_per_share = curr_price + self.FRIC
        cost_total = net_per_share * opt['contracts'] * 100

        self.cash -= cost_total

        # Realized P&L = premium received minus cost to close
        realized = (opt['open_price'] - curr_price) * opt['contracts'] * 100
        self.regime_pnl[self.regimes.get(date, 'fair')] += realized

        self.trade_log.append({
            'date': date, 'action': f'CLOSE_{opt["right"]}',
            'strike': opt['strike'], 'expiry': opt['expiry'], 'qty': opt['contracts'],
            'price': curr_price, 'premium': -cost_total, 'S': S,
            'reason': reason, 'regime': self.regimes.get(date, 'fair'),
        })
        return cost_total

    # ------------------------------------------------------------------
    # Expiry processing
    # ------------------------------------------------------------------

    def _process_expirations(self, S, date):
        """Handle options that reached or passed expiry."""
        expired   = [o for o in self.options if o['expiry'].date() <= date.date()]
        remaining = [o for o in self.options if o['expiry'].date() >  date.date()]

        for opt in expired:
            K, right, qty = opt['strike'], opt['right'], opt['contracts']

            if right == 'P':
                if S < K:
                    # Put assigned: buy shares at strike
                    # Collateral was already reserved from cash when we sold
                    cost = K * qty * 100
                    self.cash -= cost       # deduct from cash
                    self.shares += qty * 100
                    # Realized: we received premium upfront, pay intrinsic on assignment
                    realized = (opt['open_price'] - (K - S)) * qty * 100
                    self.regime_pnl[self.regimes.get(date, 'fair')] += realized
                    self.trade_log.append({
                        'date': date, 'action': 'ASSIGNED_PUT',
                        'strike': K, 'expiry': opt['expiry'], 'qty': qty,
                        'price': K, 'S': S,
                        'regime': self.regimes.get(date, 'fair'),
                    })
                else:
                    # Put expires OTM: collateral released, keep premium
                    realized = opt['open_price'] * qty * 100
                    self.regime_pnl[self.regimes.get(date, 'fair')] += realized
                    self.trade_log.append({
                        'date': date, 'action': 'EXPIRED_PUT_OTM',
                        'strike': K, 'expiry': opt['expiry'], 'qty': qty,
                        'price': 0.0, 'S': S,
                        'regime': self.regimes.get(date, 'fair'),
                    })

            elif right == 'C':
                if S > K:
                    # Call assigned: shares called away at strike
                    shares_called = qty * 100
                    if self.shares >= shares_called:
                        self.cash += K * shares_called
                        self.shares -= shares_called
                        realized = (opt['open_price'] - (S - K)) * qty * 100
                        self.regime_pnl[self.regimes.get(date, 'fair')] += realized
                        self.trade_log.append({
                            'date': date, 'action': 'ASSIGNED_CALL',
                            'strike': K, 'expiry': opt['expiry'], 'qty': qty,
                            'price': K, 'S': S,
                            'regime': self.regimes.get(date, 'fair'),
                        })
                    else:
                        # Shouldn't happen in properly managed book; just mark OTM
                        pass
                else:
                    realized = opt['open_price'] * qty * 100
                    self.regime_pnl[self.regimes.get(date, 'fair')] += realized
                    self.trade_log.append({
                        'date': date, 'action': 'EXPIRED_CALL_OTM',
                        'strike': K, 'expiry': opt['expiry'], 'qty': qty,
                        'price': 0.0, 'S': S,
                        'regime': self.regimes.get(date, 'fair'),
                    })

        self.options = remaining

    # ------------------------------------------------------------------
    # Roll logic
    # ------------------------------------------------------------------

    def _check_rolls(self, S, date):
        """
        Roll short puts when extrinsic < 30% of initial.
        Close old, open new 26 DTE at 0.275 delta.
        Only roll if free cash can support new collateral.
        """
        r, iv = self.RFIR, self.IV
        to_roll = []
        keep    = []

        for opt in self.options:
            if opt['right'] != 'P':
                keep.append(opt)
                continue
            T = max(0.0, (opt['expiry'] - date).days / 365.0)
            curr_price = bs_price(S, opt['strike'], T, r, iv, 'P')
            curr_extrin = extrinsic_value(curr_price, S, opt['strike'], 'P')
            threshold = self.ROLL_PCT * opt['initial_extrinsic']

            if curr_extrin < threshold and T < 10/365.0:
                # Only roll when close to expiry AND extrinsic is low
                to_roll.append(opt)
            else:
                keep.append(opt)

        self.options = keep

        for opt in to_roll:
            # Close the old put (releases its collateral)
            self._close_short(opt, S, date, reason='roll_extrinsic')

            # Open new put: check free cash after closing old one
            new_expiry = date + timedelta(days=self.TARGET_DTE)
            T_new = self.TARGET_DTE / 365.0
            new_K = snap_strike(
                strike_for_delta(S, T_new, r, iv, self.TARGET_DELTA_P, 'P')
            )
            new_K = max(new_K, 0.50)
            new_contracts = opt['contracts']

            collateral_needed = new_K * new_contracts * 100
            if self._free_cash() >= collateral_needed:
                self._open_short(S, 'P', new_K, new_expiry, new_contracts, date)

    # ------------------------------------------------------------------
    # Weekly rebalance
    # ------------------------------------------------------------------

    def _weekly_rebalance(self, S, date):
        """
        Manage position every 5 trading days:
        1. Sell covered calls on unhedged shares (up to 0.30 delta call)
        2. Compute target vs current delta gap
        3. Add or trim put positions to close the gap
        """
        r, iv = self.RFIR, self.IV

        # ---- 1. Covered calls ----
        # Count shares already covered by short calls
        call_contracts_out = sum(o['contracts'] for o in self.options if o['right'] == 'C')
        shares_covered = call_contracts_out * 100
        unhedged_shares = max(0, self.shares - shares_covered)

        if unhedged_shares >= 100:
            new_expiry = date + timedelta(days=self.TARGET_DTE)
            T_new = self.TARGET_DTE / 365.0
            call_K = snap_strike(
                strike_for_delta(S, T_new, r, iv, self.TARGET_DELTA_C, 'C')
            )
            cc_qty = unhedged_shares // 100
            self._open_short(S, 'C', call_K, new_expiry, int(cc_qty), date)

        # ---- 2. Delta targeting via puts ----
        tgt = self._target_delta_shares(S)
        cur = self._current_delta_shares(S)
        gap = tgt - cur

        # Each short 0.275-delta put contributes +27.5 delta equivalent per contract
        # (Short put has negative delta; shorting it gives positive exposure)
        delta_per_contract = self.TARGET_DELTA_P * 100  # ~27.5 per contract

        if gap > delta_per_contract * 0.5:
            # Need more delta → sell more puts
            contracts_needed = max(1, int(gap / delta_per_contract))
            contracts_needed = min(contracts_needed, 10)  # cap per cycle

            new_expiry = date + timedelta(days=self.TARGET_DTE)
            T_new = self.TARGET_DTE / 365.0
            put_K = snap_strike(
                strike_for_delta(S, T_new, r, iv, self.TARGET_DELTA_P, 'P')
            )
            put_K = max(put_K, 0.50)

            # Strict collateral check
            collateral_per = put_K * 100
            free = self._free_cash()
            max_by_cash = int(free / collateral_per) if collateral_per > 0 else 0
            qty = min(contracts_needed, max_by_cash)

            if qty >= 1:
                self._open_short(S, 'P', put_K, new_expiry, qty, date)

        elif gap < -delta_per_contract * 1.5:
            # Too much delta → close the most-decayed put to reduce exposure
            put_opts = sorted(
                [o for o in self.options if o['right'] == 'P'],
                key=lambda o: o['open_date']
            )
            if put_opts:
                oldest = put_opts[0]
                self._close_short(oldest, S, date, reason='delta_reduce')
                self.options = [o for o in self.options if o is not oldest]

    # ------------------------------------------------------------------
    # Cash interest accrual
    # ------------------------------------------------------------------

    def _accrue_interest(self, days):
        """Accrue risk-free interest on free cash (not collateral-locked)."""
        if days <= 0:
            return
        # Interest applies to free cash only
        daily_rate = (1 + self.RFIR) ** (1 / 365.0) - 1
        free = self._free_cash()
        if free > 0:
            interest = free * ((1 + daily_rate) ** days - 1)
            self.cash += interest

    # ------------------------------------------------------------------
    # Main simulation loop
    # ------------------------------------------------------------------

    def run(self):
        dates = list(self.prices.index)
        prev_date = dates[0]

        for i, date in enumerate(dates):
            self._current_date = date
            S = float(self.prices.iloc[i])

            # Accrue interest on free cash
            days_elapsed = (date - prev_date).days
            self._accrue_interest(days_elapsed)

            # Process expirations
            self._process_expirations(S, date)

            # Weekly management
            if (i - self._last_rebal_idx) >= self.REBAL_DAYS:
                self._check_rolls(S, date)
                self._weekly_rebalance(S, date)
                self._last_rebal_idx = i

            # Record NAV
            n = self.nav(S)
            self.nav_history.append((date, n))

            prev_date = date

        return self


# ============================================================================
# Analytics
# ============================================================================

def compute_metrics(nav_data, start_capital=100_000):
    """
    Compute performance metrics from a NAV time series.
    nav_data: list of (date, nav) tuples or pd.Series indexed by date.
    """
    if isinstance(nav_data, list):
        dates = [x[0] for x in nav_data]
        navs  = np.array([x[1] for x in nav_data])
        s = pd.Series(navs, index=pd.DatetimeIndex(dates))
    else:
        s = nav_data.copy()

    s = s.dropna()
    daily_ret = s.pct_change().dropna()

    total_return = (s.iloc[-1] / start_capital) - 1
    n_years = (s.index[-1] - s.index[0]).days / 365.25

    if n_years > 0 and s.iloc[-1] > 0:
        cagr = (s.iloc[-1] / start_capital) ** (1 / n_years) - 1
    else:
        cagr = float('nan')

    rf_daily = (1 + 0.04) ** (1 / 252) - 1
    excess = daily_ret - rf_daily
    sharpe = (excess.mean() / excess.std() * np.sqrt(252)
              if len(excess) > 1 and excess.std() > 0 else float('nan'))

    downside = daily_ret[daily_ret < rf_daily]
    if len(downside) > 1 and downside.std() > 0:
        sortino = (daily_ret.mean() - rf_daily) / downside.std() * np.sqrt(252)
    else:
        sortino = float('nan')

    rolling_max = s.cummax()
    drawdown = (s - rolling_max) / rolling_max
    max_dd = drawdown.min()

    return {
        'total_return': total_return,
        'cagr':         cagr,
        'sharpe':       sharpe,
        'sortino':      sortino,
        'max_drawdown': max_dd,
        'n_years':      n_years,
        'final_nav':    s.iloc[-1],
        'start_nav':    s.iloc[0],
        'series':       s,
        'daily_ret':    daily_ret,
    }


def monthly_returns(nav_series):
    """Compute month-end returns from daily NAV series."""
    s = nav_series.resample('ME').last()
    return s.pct_change().dropna()


def annual_returns(nav_series):
    """Compute year-end returns."""
    nav_series.resample('YE').last()
    # Compute from start of each year
    pct = []
    years = []
    prev = nav_series.iloc[0]
    for yr in sorted(set(nav_series.index.year)):
        yr_data = nav_series[nav_series.index.year == yr]
        if len(yr_data) == 0:
            continue
        yr_end = yr_data.iloc[-1]
        ret = yr_end / prev - 1
        pct.append(ret)
        years.append(yr)
        prev = yr_end
    return pd.Series(pct, index=pd.Index(years, name='year'))


def rolling_3month_windows(nav_series):
    """Find best and worst rolling 3-month periods (calendar month boundaries)."""
    monthly = nav_series.resample('ME').last()
    results = []
    for i in range(2, len(monthly)):
        start_val = monthly.iloc[i - 2]
        end_val   = monthly.iloc[i]
        if start_val > 0:
            window_ret = end_val / start_val - 1
            results.append((monthly.index[i], window_ret))
    if not results:
        return None, None
    results.sort(key=lambda x: x[1])
    return results[0], results[-1]


# ============================================================================
# Text-based histogram
# ============================================================================

def text_histogram(data, title, bins=14, width=50):
    """Print a text-based histogram."""
    data = np.array([x for x in data if not np.isnan(x)])
    if len(data) == 0:
        print(f"  {title}: no data")
        return
    counts, edges = np.histogram(data, bins=bins)
    max_count = max(counts) if max(counts) > 0 else 1

    print(f"\n  {title}")
    print(f"  {'─' * (width + 34)}")
    for cnt, edge_l, edge_r in zip(counts, edges[:-1], edges[1:]):
        bar_len = int(cnt / max_count * width)
        bar = '█' * bar_len
        label = f"[{edge_l*100:+7.1f}%, {edge_r*100:+7.1f}%)"
        print(f"  {label}  {bar:<{width}} {cnt}")
    print(f"  {'─' * (width + 34)}")
    print(f"  n={len(data)}  Mean={np.mean(data)*100:.2f}%  "
          f"Std={np.std(data)*100:.2f}%  "
          f"Min={np.min(data)*100:.2f}%  Max={np.max(data)*100:.2f}%")


# ============================================================================
# Main
# ============================================================================

def fmt_pct(v, digits=1):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "   N/A"
    return f"{v*100:.{digits}f}%"


def fmt_f(v, digits=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "   N/A"
    return f"{v:.{digits}f}"


def main():
    print("=" * 70)
    print("  UNG Wheel Strategy Backtest v2")
    print("  Regime-based delta targeting | 50% fixed IV | 10-year window")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # 1. Fetch UNG data
    # -----------------------------------------------------------------------
    print("\n[1] Fetching UNG price data from yfinance...")
    ticker = yf.Ticker('UNG')
    hist = ticker.history(period='max')
    hist.index = hist.index.tz_localize(None) if hist.index.tz is not None else hist.index

    prices = hist['Close'].sort_index()
    prices = prices[prices > 0].dropna()

    # Limit to last 10 years
    cutoff = prices.index[-1] - pd.DateOffset(years=10)
    prices = prices[prices.index >= cutoff]
    prices.index = prices.index.normalize()

    print(f"    Period: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"    Trading days: {len(prices)}")
    print(f"    Price range: ${prices.min():.2f} - ${prices.max():.2f}")
    print(f"    UNG total return over period: {(prices.iloc[-1]/prices.iloc[0]-1)*100:.1f}%")

    # -----------------------------------------------------------------------
    # 2. Compute regime
    # -----------------------------------------------------------------------
    print("\n[2] Computing regime labels (60-day return z-score)...")
    z_scores, regimes, leverages = compute_regime(prices, lookback=60, window=252)

    regime_counts = regimes.value_counts()
    print("    Regime distribution over backtest period:")
    for r_label in ['cheap', 'fair', 'rich', 'expensive']:
        cnt = regime_counts.get(r_label, 0)
        pct = cnt / len(regimes) * 100
        lev_map = {'cheap': 0.8, 'fair': 0.6, 'rich': 0.3, 'expensive': 0.1}
        print(f"      {r_label:12s}: {cnt:4d} days ({pct:5.1f}%)  "
              f"→ {lev_map[r_label]:.0%} leverage")

    # -----------------------------------------------------------------------
    # 3. Run the wheel backtest
    # -----------------------------------------------------------------------
    print("\n[3] Running wheel backtest simulation...")

    regimes_dict   = regimes.reindex(prices.index, method='ffill').fillna('fair').to_dict()
    leverages_dict = leverages.reindex(prices.index, method='ffill').fillna(0.6).to_dict()

    bt = WheelBacktest(prices, regimes_dict, leverages_dict, start_capital=100_000)
    bt.run()

    print(f"    Trades executed: {bt.trade_count}")
    print(f"    Total gross premium collected: ${bt.total_premium:,.0f}")
    print(f"    Final cash: ${bt.cash:,.0f}")
    print(f"    Final shares: {bt.shares}")
    print(f"    Final UNG price: ${float(prices.iloc[-1]):.2f}")

    # -----------------------------------------------------------------------
    # 4. Build series for all strategies
    # -----------------------------------------------------------------------
    nav_dates  = [x[0] for x in bt.nav_history]
    nav_vals   = [x[1] for x in bt.nav_history]
    wheel_nav  = pd.Series(nav_vals, index=pd.DatetimeIndex(nav_dates))
    wheel_nav.index = wheel_nav.index.normalize()

    # Buy & hold UNG
    bah_nav = prices / prices.iloc[0] * 100_000

    # Short UNG (cash-settled, no borrow cost model)
    daily_ung = prices.pct_change().fillna(0)
    short_nav  = 100_000 * (1 - daily_ung).cumprod()

    # Risk-free only
    rf_daily = (1 + 0.04) ** (1 / 252) - 1
    rf_nav   = pd.Series(
        100_000 * (1 + rf_daily) ** np.arange(len(prices)),
        index=prices.index
    )

    # -----------------------------------------------------------------------
    # 5. Compute metrics
    # -----------------------------------------------------------------------
    wheel_m = compute_metrics(bt.nav_history,     start_capital=100_000)
    bah_m   = compute_metrics(bah_nav,            start_capital=100_000)
    short_m = compute_metrics(short_nav,          start_capital=100_000)
    rf_m    = compute_metrics(rf_nav,             start_capital=100_000)

    # -----------------------------------------------------------------------
    # 6. Annual / monthly returns
    # -----------------------------------------------------------------------
    wheel_monthly = monthly_returns(wheel_nav)
    bah_monthly   = monthly_returns(bah_nav)

    wheel_annual  = annual_returns(wheel_nav)
    bah_annual    = annual_returns(bah_nav)

    # -----------------------------------------------------------------------
    # Print: Summary table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  PERFORMANCE SUMMARY  (starting capital: $100,000)")
    print("=" * 70)
    col_w = 12
    header = (f"  {'Metric':<26} {'Wheel':>{col_w}} {'B&H UNG':>{col_w}} "
              f"{'Short UNG':>{col_w}} {'Risk-Free':>{col_w}}")
    print(f"\n{header}")
    print(f"  {'─'*26} {'─'*col_w} {'─'*col_w} {'─'*col_w} {'─'*col_w}")

    rows = [
        ('Total Return',  wheel_m['total_return'],  bah_m['total_return'],
         short_m['total_return'],  rf_m['total_return'],  'pct'),
        ('CAGR',          wheel_m['cagr'],           bah_m['cagr'],
         short_m['cagr'],           rf_m['cagr'],           'pct'),
        ('Sharpe Ratio',  wheel_m['sharpe'],         bah_m['sharpe'],
         short_m['sharpe'],         rf_m['sharpe'],         'f'),
        ('Sortino Ratio', wheel_m['sortino'],        bah_m['sortino'],
         short_m['sortino'],        rf_m['sortino'],        'f'),
        ('Max Drawdown',  wheel_m['max_drawdown'],   bah_m['max_drawdown'],
         short_m['max_drawdown'],   rf_m['max_drawdown'],   'pct'),
        ('Final NAV ($k)', wheel_m['final_nav']/1e3, bah_m['final_nav']/1e3,
         short_m['final_nav']/1e3,  rf_m['final_nav']/1e3,  'fk'),
    ]
    for label, w, b, s, rf, fmt in rows:
        if fmt == 'pct':
            vals = [fmt_pct(x) for x in (w, b, s, rf)]
        elif fmt == 'f':
            vals = [fmt_f(x) for x in (w, b, s, rf)]
        else:
            vals = [f"${x:.1f}k" for x in (w, b, s, rf)]
        print(f"  {label:<26} {vals[0]:>{col_w}} {vals[1]:>{col_w}} "
              f"{vals[2]:>{col_w}} {vals[3]:>{col_w}}")

    # -----------------------------------------------------------------------
    # Print: Annual returns
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  ANNUAL RETURNS BY YEAR")
    print("=" * 70)
    print(f"\n  {'Year':<8} {'Wheel':>10} {'B&H UNG':>10} {'Alpha':>10}")
    print(f"  {'─'*8} {'─'*10} {'─'*10} {'─'*10}")

    all_years = sorted(set(list(wheel_annual.index) + list(bah_annual.index)))
    for yr in all_years:
        w_ret = wheel_annual.get(yr, float('nan'))
        b_ret = bah_annual.get(yr, float('nan'))
        alpha = w_ret - b_ret if not (np.isnan(w_ret) or np.isnan(b_ret)) else float('nan')
        print(f"  {yr:<8} {fmt_pct(w_ret):>10} {fmt_pct(b_ret):>10} "
              f"{('+' if not np.isnan(alpha) and alpha>0 else '')}{fmt_pct(alpha):>9}")

    # -----------------------------------------------------------------------
    # Print: Regime analysis
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  RETURN ATTRIBUTION BY REGIME")
    print("=" * 70)
    lev_map = {'cheap': 0.8, 'fair': 0.6, 'rich': 0.3, 'expensive': 0.1}
    print(f"\n  {'Regime':<12} {'Days':>6} {'Leverage':>10} {'Realized P&L':>15}")
    print(f"  {'─'*12} {'─'*6} {'─'*10} {'─'*15}")
    for r_label in ['cheap', 'fair', 'rich', 'expensive']:
        cnt = regime_counts.get(r_label, 0)
        pnl = bt.regime_pnl.get(r_label, 0.0)
        lev = lev_map[r_label]
        sign = '+' if pnl >= 0 else ''
        print(f"  {r_label:<12} {cnt:>6} {lev:>10.0%} {sign}${pnl:>13,.0f}")

    # -----------------------------------------------------------------------
    # Print: Premium stats
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  PREMIUM COLLECTION STATISTICS")
    print("=" * 70)
    monthly_prems = [v for v in bt.monthly_premium.values() if v > 0]
    if monthly_prems:
        avg_prem  = np.mean(monthly_prems)
        med_prem  = np.median(monthly_prems)
        ann_prem  = bt.total_premium / wheel_m['n_years']
        prem_yield = ann_prem / 100_000
        print(f"\n  Total premium collected:   ${bt.total_premium:>12,.0f}")
        print(f"  Annualized premium:        ${ann_prem:>12,.0f}")
        print(f"  Premium yield (on $100k):  {prem_yield:>12.1%}/yr")
        print(f"  Avg premium / month:       ${avg_prem:>12,.0f}")
        print(f"  Median premium / month:    ${med_prem:>12,.0f}")
        print(f"  Months tracked:            {len(bt.monthly_premium):>12d}")

    # -----------------------------------------------------------------------
    # Print: Win rate
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  MONTHLY WIN RATE ANALYSIS")
    print("=" * 70)
    if len(wheel_monthly) > 0:
        w_wins   = (wheel_monthly > 0).sum()
        w_total  = len(wheel_monthly)
        w_rate   = w_wins / w_total
        b_wins   = (bah_monthly > 0).sum()
        b_total  = len(bah_monthly)
        b_rate   = b_wins / b_total
        print(f"\n  Wheel monthly win rate: {w_rate:.1%}  ({w_wins}/{w_total} positive months)")
        print(f"  B&H UNG win rate:       {b_rate:.1%}  ({b_wins}/{b_total} positive months)")
        print(f"  Avg winning month:      {wheel_monthly[wheel_monthly>0].mean()*100:.2f}%")
        print(f"  Avg losing month:       {wheel_monthly[wheel_monthly<0].mean()*100:.2f}%")

    # -----------------------------------------------------------------------
    # Print: Best / worst 3-month periods
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  BEST & WORST 3-MONTH ROLLING PERIODS")
    print("=" * 70)
    worst_3, best_3 = rolling_3month_windows(wheel_nav)
    w3_bah, b3_bah = rolling_3month_windows(bah_nav)
    if worst_3:
        print(f"\n  Wheel worst 3-month: ending {worst_3[0].strftime('%Y-%m')}  "
              f"return {worst_3[1]*100:.2f}%")
    if best_3:
        print(f"  Wheel best  3-month: ending {best_3[0].strftime('%Y-%m')}  "
              f"return {best_3[1]*100:.2f}%")
    if w3_bah:
        print(f"\n  B&H   worst 3-month: ending {w3_bah[0].strftime('%Y-%m')}  "
              f"return {w3_bah[1]*100:.2f}%")
    if b3_bah:
        print(f"  B&H   best  3-month: ending {b3_bah[0].strftime('%Y-%m')}  "
              f"return {b3_bah[1]*100:.2f}%")

    # -----------------------------------------------------------------------
    # Print: Equity curve (quarterly snapshots)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  EQUITY CURVE — QUARTERLY NAV SNAPSHOTS")
    print("=" * 70)
    print(f"\n  {'Date':<12} {'Wheel NAV':>12} {'B&H NAV':>12} {'RF NAV':>12} {'vs BH':>10}")
    print(f"  {'─'*12} {'─'*12} {'─'*12} {'─'*12} {'─'*10}")

    wheel_q = wheel_nav.resample('QE').last()
    bah_q   = bah_nav.resample('QE').last()
    rf_q    = rf_nav.resample('QE').last()

    for dt in wheel_q.index:
        w = wheel_q.get(dt, float('nan'))
        b = bah_q.reindex([dt]).iloc[0] if dt in bah_q.index else float('nan')
        r = rf_q.reindex([dt]).iloc[0] if dt in rf_q.index else float('nan')
        diff_vs_bah = (w - b) if not (np.isnan(w) or np.isnan(b)) else float('nan')
        diff_str = f"${diff_vs_bah:>+,.0f}" if not np.isnan(diff_vs_bah) else "  N/A"
        print(f"  {str(dt.date()):<12} ${w:>10,.0f}  ${b:>10,.0f}  ${r:>10,.0f}  {diff_str:>10}")

    # -----------------------------------------------------------------------
    # Print: Histograms
    # -----------------------------------------------------------------------
    text_histogram(
        wheel_monthly.values,
        "WHEEL STRATEGY — Monthly Return Distribution",
        bins=14, width=45
    )
    text_histogram(
        bah_monthly.values,
        "BUY & HOLD UNG — Monthly Return Distribution",
        bins=14, width=45
    )

    # -----------------------------------------------------------------------
    # Print: Trade summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  TRADE LOG SUMMARY")
    print("=" * 70)
    tl = pd.DataFrame(bt.trade_log)
    if len(tl) > 0:
        action_counts = tl['action'].value_counts()
        print(f"\n  Total recorded events: {len(tl)}")
        for action, cnt in sorted(action_counts.items()):
            print(f"    {action:<25}: {cnt:5d}")

        put_assign  = action_counts.get('ASSIGNED_PUT', 0)
        put_expire  = action_counts.get('EXPIRED_PUT_OTM', 0)
        call_assign = action_counts.get('ASSIGNED_CALL', 0)
        call_expire = action_counts.get('EXPIRED_CALL_OTM', 0)
        print()
        if (put_assign + put_expire) > 0:
            pa_rate = put_assign / (put_assign + put_expire)
            print(f"  Put  assignment rate: {pa_rate:.1%}  "
                  f"({put_assign} assigned / {put_assign+put_expire} settled)")
        if (call_assign + call_expire) > 0:
            ca_rate = call_assign / (call_assign + call_expire)
            print(f"  Call assignment rate: {ca_rate:.1%}  "
                  f"({call_assign} assigned / {call_assign+call_expire} settled)")

    # -----------------------------------------------------------------------
    # Print: Final state
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  FINAL PORTFOLIO STATE")
    print("=" * 70)
    final_S = float(prices.iloc[-1])
    final_nav = bt.nav(final_S)
    print(f"\n  Cash balance:        ${bt.cash:>14,.2f}")
    print(f"  Put collateral rsv:  ${bt._put_collateral_reserved():>14,.2f}")
    print(f"  Free cash:           ${bt._free_cash():>14,.2f}")
    print(f"  Shares held:         {bt.shares:>14,d}")
    print(f"  Share value:         ${bt.shares * final_S:>14,.2f}")
    print(f"  Open option pos:     {len(bt.options):>14,d}")
    opt_mtm = bt._option_mtm_liability(final_S)
    print(f"  Option MTM P&L:      ${opt_mtm:>+14,.2f}")
    print("  ─────────────────────────────────────────")
    print(f"  FINAL NAV:           ${final_nav:>14,.2f}")
    print(f"  UNG final price:     ${final_S:>14.2f}")
    print()

    return bt, wheel_m, bah_m


if __name__ == '__main__':
    bt, wheel_m, bah_m = main()
