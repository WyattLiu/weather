"""Generic put-write wheel backtest — parameterized by ticker.

Strategy (intentionally simple, apples-to-apples across UNG/DBA):
  - Every Friday (or N days), sell 1× ~5% OTM put at ~45 DTE
  - Close at 50% of max profit (TP rule)
  - If ITM at expiry: take assignment, then sell covered call ~5% OTM
  - Track NAV, Sharpe, drawdown across the full window

Option pricing: Black-Scholes with realized-vol×1.12 as IV proxy
  (matches existing kernel convention in historical_data_pipeline.py)

Sizing: 1 contract per $X NAV (configurable), so backtests
  are comparable across underlyings at different price points.

Run:
    venv/bin/python research/dba/wheel_backtest.py --ticker DBA
    venv/bin/python research/dba/wheel_backtest.py --ticker UNG
    venv/bin/python research/dba/wheel_backtest.py --both
"""
import os
import math
import argparse
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, 'cache')


def bsm_put(S, K, T, r, sigma):
    """Black-Scholes put price. T in years, sigma annualized."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(K - S, 0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    from scipy.stats import norm
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bsm_call(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(S - K, 0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    from scipy.stats import norm
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


# Real strike increments (verified against live WS chains / user positions):
# DBA $1 (P25/26/27...), CORN $1, SOYB $1, CANE $0.50 (sub-$10), UNG $0.50
STRIKE_STEP = {'DBA': 1.0, 'CORN': 1.0, 'SOYB': 1.0, 'WEAT': 1.0,
               'CANE': 0.5, 'UNG': 0.5}


def third_friday(year, month):
    from datetime import date as _date, timedelta as _td
    d = _date(year, month, 15)
    while d.weekday() != 4:
        d += _td(days=1)
    return d


def nearest_monthly_dte(today, target_dte, lo=25, hi=80):
    """Days to the monthly (3rd-Friday) expiry nearest target_dte within
    [lo, hi]; None if no monthly fits (skip entry)."""
    best = None
    for k in range(0, 4):
        m = today.month + k
        y = today.year + (m - 1) // 12
        m = (m - 1) % 12 + 1
        exp = third_friday(y, m)
        dte = (exp - today.date() if hasattr(today, 'date') else exp - today).days
        if lo <= dte <= hi:
            if best is None or abs(dte - target_dte) < abs(best - target_dte):
                best = dte
    return best


def load_prices(ticker):
    """Load price series from master_panel.csv."""
    panel = pd.read_csv(os.path.join(CACHE, 'master_panel.csv'),
                        index_col=0, parse_dates=True)
    if ticker not in panel.columns:
        raise ValueError(f'{ticker} not in master_panel.csv')
    s = panel[ticker].dropna()
    return s


def realized_vol(prices, window=30):
    """Annualized rolling realized vol."""
    return prices.pct_change().rolling(window).std() * math.sqrt(252)


def run_wheel(ticker, start='2015-01-01',
              dte_target=45, otm_pct=0.05,
              tp_pct=0.50,         # close at 50% of max profit
              entry_cadence_days=7,  # weekly entries
              contract_per_nav=15000,  # 1 contract per $15k NAV
              init_nav=100000, r_free=0.045,
              vol_gate=1.0,        # skip entries if realized vol > vol_gate (annualized)
              signal_fn=None,      # optional date -> {'size_mult': x, 'otm_pct': y}
              realistic=False):    # monthly expiries + real strike grid
    """Run the wheel and return DataFrame indexed by date."""
    prices = load_prices(ticker)
    prices = prices[prices.index >= pd.Timestamp(start)]
    if len(prices) < 100:
        raise ValueError(f'not enough {ticker} data after {start}: {len(prices)}')

    rv = realized_vol(prices, window=30) * 1.12  # IV proxy
    rv = rv.bfill().fillna(0.3)

    # State
    nav = init_nav
    cash = init_nav
    shares = 0  # long shares from put assignment
    open_puts = []  # list of (entry_date, K, T_init, premium_collected, n_contracts)
    open_calls = []  # list of (entry_date, K, T_init, premium_collected, n_contracts)

    last_entry = None
    nav_curve = []
    trades = []

    for date, spot in prices.items():
        sigma = rv.loc[date]

        # --- 1. Mark-to-market open positions, decide close ---
        new_open_puts = []
        for op in open_puts:
            entry, K, T0, premium, nc = op
            days_held = (date - entry).days
            T_remaining = max(0.001, (T0 * 365 - days_held) / 365.25)
            current_premium = bsm_put(spot, K, T_remaining, r_free, sigma)
            pct_decayed = 1 - current_premium / max(premium, 0.01)

            # Expiry assignment
            if T_remaining <= 1.5 / 365:
                if spot < K:
                    # Assigned: buy 100×nc shares at K
                    shares += 100 * nc
                    cash -= 100 * nc * K
                    trades.append({'date': date, 'type': 'put_assign',
                                    'K': K, 'spot': spot, 'shares_added': 100*nc})
                else:
                    trades.append({'date': date, 'type': 'put_expire_otm',
                                    'K': K, 'spot': spot, 'premium_kept': premium*100*nc})
                continue
            # Take profit
            if pct_decayed >= tp_pct:
                cash -= current_premium * 100 * nc  # buy back
                trades.append({'date': date, 'type': 'put_tp',
                                'K': K, 'kept': (premium - current_premium)*100*nc})
                continue
            new_open_puts.append(op)
        open_puts = new_open_puts

        # CC management
        new_open_calls = []
        for oc in open_calls:
            entry, K, T0, premium, nc = oc
            days_held = (date - entry).days
            T_remaining = max(0.001, (T0 * 365 - days_held) / 365.25)
            current_premium = bsm_call(spot, K, T_remaining, r_free, sigma)
            pct_decayed = 1 - current_premium / max(premium, 0.01)

            if T_remaining <= 1.5 / 365:
                if spot > K:
                    # Called away
                    sold = 100 * nc
                    if shares >= sold:
                        shares -= sold
                        cash += sold * K
                        trades.append({'date': date, 'type': 'cc_called_away',
                                        'K': K, 'spot': spot, 'shares_sold': sold})
                else:
                    trades.append({'date': date, 'type': 'cc_expire_otm',
                                    'K': K, 'premium_kept': premium*100*nc})
                continue
            if pct_decayed >= tp_pct:
                cash -= current_premium * 100 * nc
                trades.append({'date': date, 'type': 'cc_tp',
                                'K': K, 'kept': (premium - current_premium)*100*nc})
                continue
            new_open_calls.append(oc)
        open_calls = new_open_calls

        # --- 2. Entry decisions ---
        vol_ok = sigma <= vol_gate
        # factor signal: scale size and shift strike per-day
        size_mult, eff_otm = 1.0, otm_pct
        if signal_fn is not None:
            sig_d = signal_fn(date) or {}
            size_mult = float(sig_d.get('size_mult', 1.0))
            eff_otm = float(sig_d.get('otm_pct', otm_pct))
        if size_mult <= 0:
            vol_ok = False  # signal says stand down entirely
        if vol_ok and (last_entry is None or (date - last_entry).days >= entry_cadence_days):
            # Strike on the REAL grid; expiry on the REAL calendar
            step = STRIKE_STEP.get(ticker, 0.5) if realistic else 0.5
            target_K = round(spot * (1 - eff_otm) / step) * step
            if realistic:
                dte_real = nearest_monthly_dte(date, dte_target)
                if dte_real is None:
                    last_entry = date  # no monthly fits the window — skip
                    T_yr = None
                else:
                    T_yr = dte_real / 365.25
            else:
                T_yr = dte_target / 365.25
            premium = bsm_put(spot, target_K, T_yr, r_free, sigma) if T_yr else 0
            if premium > 0.05:
                # Size: 1 contract per $contract_per_nav
                n_contracts = max(1, int(nav / contract_per_nav * size_mult))
                # Cap by available cash collateral
                max_by_cash = int((cash - 5000) / (target_K * 100))
                n_contracts = max(0, min(n_contracts, max_by_cash))
                if n_contracts > 0:
                    cash += premium * 100 * n_contracts
                    open_puts.append((date, target_K, T_yr, premium, n_contracts))
                    trades.append({'date': date, 'type': 'put_open',
                                    'K': target_K, 'premium': premium,
                                    'n': n_contracts})

            # If holding shares, sell CC — COVERED ONLY: subtract shares
            # already covering open calls (stacking = naked calls; this bug
            # inflated every ag-wheel result until 2026-06-11)
            cc_covered = 100 * sum(oc[4] for oc in open_calls)
            if shares - cc_covered >= 100 and T_yr:
                step = STRIKE_STEP.get(ticker, 0.5) if realistic else 0.5
                cc_K = round(spot * (1 + otm_pct) / step) * step
                cc_premium = bsm_call(spot, cc_K, T_yr, r_free, sigma)
                if cc_premium > 0.05:
                    n_cc = (shares - cc_covered) // 100
                    cash += cc_premium * 100 * n_cc
                    open_calls.append((date, cc_K, T_yr, cc_premium, n_cc))
                    trades.append({'date': date, 'type': 'cc_open',
                                    'K': cc_K, 'premium': cc_premium, 'n': n_cc})

            last_entry = date

        # --- 3. Compute NAV ---
        # Mark open shorts to MtM (subtract their current value as liability)
        liability = 0
        for entry, K, T0, _, nc in open_puts:
            days_held = (date - entry).days
            T_rem = max(0.001, (T0 * 365 - days_held) / 365.25)
            liability += bsm_put(spot, K, T_rem, r_free, sigma) * 100 * nc
        for entry, K, T0, _, nc in open_calls:
            days_held = (date - entry).days
            T_rem = max(0.001, (T0 * 365 - days_held) / 365.25)
            liability += bsm_call(spot, K, T_rem, r_free, sigma) * 100 * nc
        nav = cash + shares * spot - liability
        nav_curve.append({'date': date, 'spot': spot, 'cash': cash,
                          'shares': shares, 'liability': liability, 'nav': nav,
                          'sigma': sigma})

    return pd.DataFrame(nav_curve).set_index('date'), pd.DataFrame(trades)


def summarize(curve, label):
    """Return dict with key metrics."""
    nav = curve['nav']
    ret = nav.pct_change().dropna()
    cum_ret = nav.iloc[-1] / nav.iloc[0] - 1
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    ann_ret = (1 + cum_ret) ** (1 / years) - 1
    sharpe = ret.mean() / ret.std() * math.sqrt(252) if ret.std() > 0 else 0
    mdd = (nav / nav.cummax() - 1).min()
    return {
        'strategy': label,
        'cum_ret': round(cum_ret, 4),
        'ann_ret': round(ann_ret, 4),
        'sharpe': round(sharpe, 3),
        'mdd': round(mdd, 4),
        'years': round(years, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', default='DBA')
    parser.add_argument('--both', action='store_true')
    parser.add_argument('--start', default='2015-01-01')
    parser.add_argument('--otm', type=float, default=0.05)
    parser.add_argument('--dte', type=int, default=45)
    args = parser.parse_args()

    tickers = ['UNG', 'DBA'] if args.both else [args.ticker]
    summaries = []
    curves = {}
    for tk in tickers:
        print(f'\n[wheel] running {tk} ({args.dte}d, {args.otm:.0%} OTM)...')
        curve, trades = run_wheel(tk, start=args.start,
                                   dte_target=args.dte, otm_pct=args.otm)
        curves[tk] = curve
        s = summarize(curve, f'{tk}_wheel')
        summaries.append(s)
        print(f'  bars: {len(curve)}, trades: {len(trades)}')
        # Save artifacts
        curve.to_csv(os.path.join(CACHE, f'wheel_{tk}_curve.csv'))
        trades.to_csv(os.path.join(CACHE, f'wheel_{tk}_trades.csv'), index=False)

    print('\n=== Wheel backtest results ===')
    print(pd.DataFrame(summaries).to_string(index=False))

    # If both, compute correlation of daily returns
    if len(curves) == 2:
        rets = pd.DataFrame({tk: c['nav'].pct_change() for tk, c in curves.items()})
        rets = rets.dropna()
        corr = rets.corr().iloc[0, 1]
        print(f'\nDaily-return correlation UNG×DBA wheels: {corr:+.3f}')
        # 50/50 blend
        blend = 0.5 * rets['UNG'] + 0.5 * rets['DBA']
        blend_sharpe = blend.mean() / blend.std() * math.sqrt(252) if blend.std() > 0 else 0
        blend_ann = (1 + blend).prod() ** (252 / len(blend)) - 1
        print(f'\n50/50 blend: ann={blend_ann:+.2%}  sharpe={blend_sharpe:+.3f}')


if __name__ == '__main__':
    main()
