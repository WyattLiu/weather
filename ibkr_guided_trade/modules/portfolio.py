"""
Module 5: Portfolio Optimization — Greek-Matched Paired Positions

Builds iron condor-like structures per underlying (bull put + bear call),
delta-matched via contract ratios, then runs Markowitz optimization across
the paired positions with greek constraints.

Flow:
  1. For each underlying, pair best bull put + best bear call
  2. Compute contract ratio for delta neutrality
  3. Compute combined greeks (delta, gamma, vega, theta)
  4. Run mean-variance optimization with portfolio-level delta constraint
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from ib_insync import Stock


# ============ Paired Position ============

@dataclass
class PairedPosition:
    """An iron condor-like structure: bull put + bear call on the same underlying."""
    symbol: str
    spot: float
    dte: int

    # Bull put leg
    put_short: float
    put_long: float
    put_width: float
    put_credit: float
    put_delta: float       # spread delta per contract (negative for puts, but magnitude)
    put_gamma: float       # spread gamma per contract
    put_vega: float        # spread vega per contract
    put_theta: float       # spread theta per contract
    put_oi: int
    put_score: float

    # Bear call leg
    call_short: float
    call_long: float
    call_width: float
    call_credit: float
    call_delta: float      # spread delta per contract
    call_gamma: float
    call_vega: float
    call_theta: float
    call_oi: int
    call_score: float

    # Contract ratio for delta neutrality
    put_ratio: int = 1     # e.g., 2 put spreads : 1 call spread
    call_ratio: int = 1

    @property
    def total_credit(self):
        """Total credit per unit (put_ratio puts + call_ratio calls)."""
        return self.put_credit * self.put_ratio + self.call_credit * self.call_ratio

    @property
    def total_risk(self):
        """Max loss per unit (worst case: one side hit)."""
        put_risk = (self.put_width - self.put_credit) * self.put_ratio
        call_risk = (self.call_width - self.call_credit) * self.call_ratio
        return max(put_risk, call_risk) * 100

    @property
    def capital_at_risk(self):
        """Capital at risk per unit in dollars."""
        # For an IC, max loss = wider side's risk * 100 - total credit * 100
        put_max = (self.put_width - self.put_credit) * self.put_ratio * 100
        call_max = (self.call_width - self.call_credit) * self.call_ratio * 100
        return max(put_max, call_max)

    @property
    def ror(self):
        """Return on risk if both sides expire worthless."""
        car = self.capital_at_risk
        return (self.total_credit * 100) / car if car > 0 else 0

    @property
    def net_delta(self):
        """Net dollar-delta per unit."""
        # Put spread: positive delta (bullish); call spread: negative delta (bearish)
        return (abs(self.put_delta) * self.put_ratio - abs(self.call_delta) * self.call_ratio) * 100 * self.spot

    @property
    def net_gamma(self):
        """Net gamma per unit. Negated: spread_* are long-spread greeks, we're short."""
        return -(self.put_gamma * self.put_ratio + self.call_gamma * self.call_ratio) * 100

    @property
    def net_vega(self):
        """Net vega per unit. Negated: we're short the spreads (short vol)."""
        return -(self.put_vega * self.put_ratio + self.call_vega * self.call_ratio) * 100

    @property
    def net_theta(self):
        """Net theta per unit. Negated: we collect theta as sellers."""
        return -(self.put_theta * self.put_ratio + self.call_theta * self.call_ratio) * 100

    @property
    def leverage_factor(self):
        """Residual delta-based leverage for covariance matrix."""
        if self.capital_at_risk <= 0:
            return 0
        residual_delta = abs(self.put_delta * self.put_ratio - self.call_delta * self.call_ratio)
        return (residual_delta * 100 * self.spot * 0.01) / self.capital_at_risk

    @property
    def delta_balance(self):
        """How well delta-matched: 0 = perfect, 1 = completely one-sided."""
        put_d = abs(self.put_delta) * self.put_ratio
        call_d = abs(self.call_delta) * self.call_ratio
        total = put_d + call_d
        if total < 1e-6:
            return 0
        return abs(put_d - call_d) / total


# ============ Build Pairs ============

def build_paired_positions(phase3_results):
    """Build delta-matched paired positions from Phase 3 scan results."""
    pairs = []

    for sym, data in phase3_results.items():
        spreads = data.get('spreads', [])
        bear_calls = data.get('bear_call_spreads', [])

        if not spreads or not bear_calls:
            continue

        spot = data['spot']
        dte = data.get('dte', 35)

        # Pick best scoring bull put and bear call
        bp = spreads[0]
        bc = bear_calls[0]

        # Spread deltas (magnitude)
        bp_delta = abs(bp.get('spread_delta') or 0)
        bc_delta = abs(bc.get('spread_delta') or 0)

        # Compute contract ratio for delta neutrality
        # put_ratio * bp_delta ≈ call_ratio * bc_delta
        if bp_delta > 0 and bc_delta > 0:
            ratio = bp_delta / bc_delta
            if ratio >= 1:
                put_ratio = 1
                call_ratio = max(1, round(ratio))
            else:
                call_ratio = 1
                put_ratio = max(1, round(1 / ratio))
            # Cap at 3:1 to avoid extreme ratios
            put_ratio = min(put_ratio, 3)
            call_ratio = min(call_ratio, 3)
        else:
            put_ratio = 1
            call_ratio = 1

        pair = PairedPosition(
            symbol=sym,
            spot=spot,
            dte=dte,
            # Bull put
            put_short=bp['short_strike'],
            put_long=bp['long_strike'],
            put_width=bp['width'],
            put_credit=bp['mid_credit'],
            put_delta=bp.get('spread_delta') or 0,
            put_gamma=bp.get('spread_gamma') or 0,
            put_vega=bp.get('spread_vega') or 0,
            put_theta=bp.get('spread_theta') or 0,
            put_oi=bp.get('min_oi', 0),
            put_score=bp['score'],
            # Bear call
            call_short=bc['short_strike'],
            call_long=bc['long_strike'],
            call_width=bc['width'],
            call_credit=bc['mid_credit'],
            call_delta=bc.get('spread_delta') or 0,
            call_gamma=bc.get('spread_gamma') or 0,
            call_vega=bc.get('spread_vega') or 0,
            call_theta=bc.get('spread_theta') or 0,
            call_oi=bc.get('min_oi', 0),
            call_score=bc['score'],
            # Ratio
            put_ratio=put_ratio,
            call_ratio=call_ratio,
        )
        pairs.append(pair)

    return pairs


# ============ Expected Returns ============

def compute_expected_returns(pairs, oi_beta=0.10):
    """
    Combined E[R] for paired position:
    E[R] = P(both_OTM) * combined_RoR - P(put_ITM) * put_loss_frac - P(call_ITM) * call_loss_frac

    Assumes put/call ITM are mutually exclusive (can't breach both sides).
    """
    n = len(pairs)
    mu = np.zeros(n)

    for i, p in enumerate(pairs):
        # Put side P(OTM)
        p_put_otm = 1.0 - abs(p.put_delta)
        # Call side P(OTM)
        p_call_otm = 1.0 - abs(p.call_delta)

        # P(both OTM) ≈ product (nearly independent for wide IC)
        p_both_otm = p_put_otm * p_call_otm

        # P(put breached) — only the put side loses
        p_put_itm = 1.0 - p_put_otm
        # P(call breached) — only the call side loses
        p_call_itm = 1.0 - p_call_otm

        car = p.capital_at_risk
        if car <= 0:
            continue

        # Max profit: both sides expire worthless
        max_profit = p.total_credit * 100

        # Max loss per side
        put_loss = (p.put_width - p.put_credit) * p.put_ratio * 100
        call_loss = (p.call_width - p.call_credit) * p.call_ratio * 100

        mu[i] = (p_both_otm * max_profit - p_put_itm * put_loss - p_call_itm * call_loss) / car

    return mu


# ============ Covariance ============

def compute_covariance_matrix(pairs, stock_cov, stock_tickers, oi_gamma=0.10):
    """Sigma_pairs = L * M * Sigma_stocks * M^T * L^T using residual delta leverage."""
    n = len(pairs)
    k = len(stock_tickers)
    ticker_to_idx = {t: i for i, t in enumerate(stock_tickers)}

    M = np.zeros((n, k))
    for i, p in enumerate(pairs):
        if p.symbol in ticker_to_idx:
            M[i, ticker_to_idx[p.symbol]] = 1.0

    L = np.zeros((n, n))
    for i, p in enumerate(pairs):
        L[i, i] = p.leverage_factor

    LM = L @ M
    return LM @ stock_cov @ LM.T


# ============ Stock Returns ============

def fetch_stock_returns(ib, tickers):
    """Fetch 1Y daily log returns from IBKR."""
    import pandas as pd

    all_returns = {}
    for sym in tickers:
        ibkr_sym = sym.replace('.', ' ')
        stock = Stock(ibkr_sym, 'SMART', 'USD')
        try:
            ib.qualifyContracts(stock)
            bars = ib.reqHistoricalData(
                stock, endDateTime='', durationStr='1 Y',
                barSizeSetting='1 day', whatToShow='TRADES',
                useRTH=True, formatDate=1, timeout=10,
            )
            if bars and len(bars) > 30:
                closes = [b.close for b in bars]
                dates = [b.date for b in bars]
                log_rets = [math.log(closes[j] / closes[j - 1]) for j in range(1, len(closes))]
                all_returns[sym] = pd.Series(log_rets, index=dates[1:])
        except Exception:
            pass
        ib.sleep(0.5)

    return pd.DataFrame(all_returns).dropna(how='all').ffill().bfill()


# ============ Optimizer ============

def optimize_portfolio(mu, Sigma, pairs, total_capital, max_weight=0.30,
                       risk_free_rate=0.006, delta_tolerance=5000):
    """
    Optimize with:
    - Max Sharpe objective
    - Portfolio net delta within tolerance
    - Max weight per position
    """
    n = len(mu)
    # Ensure feasibility: max_weight * n must be >= 1.0
    effective_max = max(max_weight, 1.0 / n)
    w0 = np.ones(n) / n
    bounds = tuple((0.0, effective_max) for _ in range(n))

    # Dollar deltas per unit of capital
    pair_deltas = np.array([p.net_delta for p in pairs])

    def neg_sharpe(w):
        vol = np.sqrt(w @ Sigma @ w)
        return -(w @ mu - risk_free_rate) / vol if vol > 1e-10 else 0

    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
        # Portfolio net delta constraint: |sum(w_i * delta_i * capital)| <= tolerance
        {'type': 'ineq', 'fun': lambda w: delta_tolerance - abs(np.sum(w * pair_deltas * total_capital / max(p.capital_at_risk for p in pairs)))},
    ]

    result = minimize(neg_sharpe, w0, method='SLSQP', bounds=bounds,
                      constraints=constraints, options={'maxiter': 2000, 'ftol': 1e-12})

    weights = result.x
    port_ret = weights @ mu
    port_vol = np.sqrt(weights @ Sigma @ weights)
    sharpe = (port_ret - risk_free_rate) / port_vol if port_vol > 1e-10 else 0

    # Convert to allocations
    allocations = []
    total_delta = 0
    total_gamma = 0
    total_vega = 0
    total_theta = 0

    for i, (w, p) in enumerate(zip(weights, pairs)):
        if w < 0.01:
            continue

        dollars = w * total_capital
        # Units = how many paired sets we can afford
        units = max(1, int(dollars / p.capital_at_risk)) if p.capital_at_risk > 0 else 0

        put_cts = units * p.put_ratio
        call_cts = units * p.call_ratio
        u_delta = p.net_delta * units
        u_gamma = p.net_gamma * units
        u_vega = p.net_vega * units
        u_theta = p.net_theta * units

        total_delta += u_delta
        total_gamma += u_gamma
        total_vega += u_vega
        total_theta += u_theta

        allocations.append({
            'symbol': p.symbol,
            'put_spread': f'{p.put_short:.0f}/{p.put_long:.0f}P',
            'call_spread': f'{p.call_short:.0f}/{p.call_long:.0f}C',
            'put_cts': put_cts,
            'call_cts': call_cts,
            'ratio': f'{p.put_ratio}:{p.call_ratio}',
            'weight': w,
            'dollars': dollars,
            'units': units,
            'expected_return': mu[i],
            'max_loss': p.capital_at_risk * units,
            'max_profit': p.total_credit * 100 * units,
            'delta': u_delta,
            'gamma': u_gamma,
            'vega': u_vega,
            'theta': u_theta,
            'delta_balance': p.delta_balance,
            'ror': p.ror,
        })

    allocations.sort(key=lambda a: a['weight'], reverse=True)

    return {
        'allocations': allocations,
        'portfolio_return': port_ret,
        'portfolio_vol': port_vol,
        'sharpe_ratio': sharpe,
        'optimizer_success': result.success,
        'total_delta': total_delta,
        'total_gamma': total_gamma,
        'total_vega': total_vega,
        'total_theta': total_theta,
        'total_max_loss': sum(a['max_loss'] for a in allocations),
        'total_max_profit': sum(a['max_profit'] for a in allocations),
    }


# ============ Full Pipeline ============

def run_portfolio_optimization(phase3_results, ib=None, stock_returns_df=None,
                               total_capital=10000, max_weight=0.40,
                               oi_beta=0.10, oi_gamma=0.10, risk_free_rate=0.05):
    """Full pipeline: Phase 3 results -> paired positions -> Markowitz -> allocation."""

    pairs = build_paired_positions(phase3_results)
    if len(pairs) < 2:
        print('  Need at least 2 stocks with both put and call spreads.')
        return None

    # Scale risk-free rate to per-cycle
    avg_dte = sum(p.dte for p in pairs) / len(pairs)
    cycle_rf = risk_free_rate * (avg_dte / 365)

    unique_tickers = list(set(p.symbol for p in pairs))
    print(f'  {len(pairs)} paired positions across {len(unique_tickers)} underlyings '
          f'| avg DTE {avg_dte:.0f} | cycle Rf {cycle_rf*100:.2f}%')

    for p in pairs:
        print(f'\n    {p.symbol} @ ${p.spot:.2f}')
        print(f'      Put:  {p.put_short:.0f}/{p.put_long:.0f}P x{p.put_ratio} '
              f'cr ${p.put_credit:.2f} | d={p.put_delta:+.4f} g={p.put_gamma:+.5f} '
              f'v={p.put_vega:+.3f} th={p.put_theta:+.3f}')
        print(f'      Call: {p.call_short:.0f}/{p.call_long:.0f}C x{p.call_ratio} '
              f'cr ${p.call_credit:.2f} | d={p.call_delta:+.4f} g={p.call_gamma:+.5f} '
              f'v={p.call_vega:+.3f} th={p.call_theta:+.3f}')
        print(f'      Combined: net_d=${p.net_delta:+.0f} gamma={p.net_gamma:+.4f} '
              f'vega={p.net_vega:+.3f} theta={p.net_theta:+.3f} '
              f'| RoR {p.ror*100:.1f}% | bal {p.delta_balance:.2f}')

    # Fetch stock returns
    if stock_returns_df is None:
        if ib is None:
            print('  No IB connection for return data.')
            return None
        print(f'\n  Fetching 1Y returns for {len(unique_tickers)} stocks...')
        stock_returns_df = fetch_stock_returns(ib, unique_tickers)

    available = [t for t in unique_tickers if t in stock_returns_df.columns]
    pairs = [p for p in pairs if p.symbol in available]

    if len(pairs) < 2:
        print('  Not enough pairs with return data.')
        return None

    returns_subset = stock_returns_df[available]
    stock_cov = (returns_subset.cov() * 252).values
    stock_ticker_list = list(returns_subset.columns)

    mu = compute_expected_returns(pairs, oi_beta=oi_beta)
    Sigma = compute_covariance_matrix(pairs, stock_cov, stock_ticker_list, oi_gamma=oi_gamma)
    Sigma += np.eye(len(pairs)) * 1e-8

    result = optimize_portfolio(mu, Sigma, pairs, total_capital,
                                max_weight=max_weight, risk_free_rate=cycle_rf)
    result['pairs'] = pairs
    return result


# ============ Output ============

def print_allocation(result):
    """Pretty-print paired portfolio optimization results."""
    if not result:
        print('  No optimization result.')
        return

    print(f'\n  Portfolio E[Return]: {result["portfolio_return"]*100:+.2f}% per cycle')
    print(f'  Portfolio Vol:       {result["portfolio_vol"]*100:.2f}%')
    print(f'  Sharpe Ratio:        {result["sharpe_ratio"]:.3f}')
    print(f'  Optimizer:           {"CONVERGED" if result["optimizer_success"] else "FAILED"}')

    allocs = result['allocations']
    if not allocs:
        print('  No allocations.')
        return

    print(f'\n  {"Sym":>7} {"Put Sprd":>10} {"Call Sprd":>10} {"Ratio":>5} '
          f'{"Wgt":>6} {"Units":>5} {"RoR":>6} {"E[R]":>7} '
          f'{"Delta":>8} {"Gamma":>7} {"Vega":>7} {"Theta":>7}')
    print(f'  {"-" * 100}')

    for a in allocs:
        print(f'  {a["symbol"]:>7} {a["put_spread"]:>10} {a["call_spread"]:>10} '
              f'{a["ratio"]:>5} {a["weight"]:>5.0%} {a["units"]:>5} '
              f'{a["ror"]*100:>5.1f}% {a["expected_return"]*100:>+6.2f}% '
              f'{a["delta"]:>+8.0f} {a["gamma"]:>+7.2f} '
              f'{a["vega"]:>+7.1f} {a["theta"]:>+7.1f}')

    print(f'\n  PORTFOLIO GREEKS:')
    print(f'    Net Delta:  ${result["total_delta"]:+,.0f}')
    print(f'    Net Gamma:  {result["total_gamma"]:+,.2f}')
    print(f'    Net Vega:   {result["total_vega"]:+,.1f}')
    print(f'    Net Theta:  {result["total_theta"]:+,.1f}')
    print(f'    Max Loss:   ${result["total_max_loss"]:,.0f}')
    print(f'    Max Profit: ${result["total_max_profit"]:,.0f}')
