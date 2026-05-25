#!/usr/bin/env python3
"""
Empirical test of stand-aside hypothesis on UNG wheel income.

Question: does exiting cleanly during 'peak up time' and waiting in cash
improve risk-adjusted income vs always-on wheel?

Approach (no option pricing — just realized share P&L vs theta capture proxy):
1. Compute daily UNG features: price_band (120d range), z-score (60d ret)
2. Test multiple stand-aside rules:
   - Rule A: stand aside when price_band > 0.75 (peak)
   - Rule B: stand aside when z-score < -0.5 (expensive regime)
   - Rule C: combined (either condition)
   - Rule D: combined + price_band > 0.60 (more conservative exit)
3. For each rule: simulate equity curve = theta capture (when deployed) +
   share P&L (when assigned/holding) - friction at transitions
4. Report max drawdown, captured premium, deployment %, Sharpe

This is a coarse first pass — uses IV proxy for premium estimation. Output
informs whether to add stand-aside to the optimizer.
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf

# ── Data ────────────────────────────────────────────────────────────────────
print("Fetching UNG history...")
df = yf.download('UNG', start='2018-01-01', progress=False, auto_adjust=False)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
df = df.reset_index()[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
print(f"  {len(df)} bars, {df['date'].iloc[0]:%Y-%m-%d} to {df['date'].iloc[-1]:%Y-%m-%d}")

# Daily features
df['ret'] = df['close'].pct_change()
df['rvol_60d'] = df['ret'].rolling(60).std() * np.sqrt(252)
df['ma_60'] = df['close'].rolling(60).mean()
df['ret_60d'] = df['close'].pct_change(60)
df['ret_60d_z'] = (df['ret_60d'] - df['ret_60d'].rolling(252, min_periods=60).mean()) \
                  / df['ret_60d'].rolling(252, min_periods=60).std()
df['hi_120d'] = df['high'].rolling(120).max()
df['lo_120d'] = df['low'].rolling(120).min()
df['price_band'] = (df['close'] - df['lo_120d']) / (df['hi_120d'] - df['lo_120d'])

# Premium proxy: ATM 30-day put premium ≈ spot × 0.4 × iv × sqrt(30/365)
df['iv_proxy'] = (df['rvol_60d'] * 1.2).clip(0.20, 1.20)
df['atm_30d_put_pct'] = 0.4 * df['iv_proxy'] * np.sqrt(30/365)  # as % of strike
df['daily_premium_yield'] = df['atm_30d_put_pct'] / 30  # daily theta yield % of capital

df = df.dropna(subset=['price_band', 'ret_60d_z', 'iv_proxy']).reset_index(drop=True)
print(f"  Usable days after warmup: {len(df)}")

# ── Strategy simulation ────────────────────────────────────────────────────
def simulate(df, deploy_mask, capital_0=100_000, share_collateral_frac=0.7,
             switch_friction_bps=20):
    """Coarse strategy simulator.

    When deployed: earn daily_premium_yield × deployed_capital
    When not deployed: earn risk-free (4%/yr) on cash
    Share P&L: when deployed, we're effectively long ~share_collateral_frac × cap
              of UNG via short puts (delta exposure)
    Transitions cost switch_friction_bps in basis points each direction.
    """
    capital = capital_0
    rf_daily = 0.04 / 252
    deployed_prev = False
    equity = [capital]
    premium_captured = 0.0

    for i in range(1, len(df)):
        is_deployed = deploy_mask.iloc[i]
        # Transition friction
        if is_deployed != deployed_prev:
            capital *= (1 - switch_friction_bps / 10000)

        if is_deployed:
            # Premium yield on capital
            premium_today = capital * df['daily_premium_yield'].iloc[i]
            premium_captured += premium_today
            # Share-side exposure: short puts ~ long UNG delta on collateral fraction
            share_pnl_today = capital * share_collateral_frac * df['ret'].iloc[i]
            capital += premium_today + share_pnl_today
        else:
            capital *= (1 + rf_daily)

        equity.append(capital)
        deployed_prev = is_deployed

    eq = pd.Series(equity, index=df['date'])
    return eq, premium_captured

def stats(eq):
    rets = eq.pct_change().dropna()
    ann_ret = (eq.iloc[-1] / eq.iloc[0]) ** (252 / len(eq)) - 1
    ann_vol = rets.std() * np.sqrt(252)
    sharpe = (rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0
    max_dd = (eq / eq.cummax() - 1).min()
    return ann_ret, ann_vol, sharpe, max_dd

# ── Rule definitions ───────────────────────────────────────────────────────
rules = {
    'Always-on (baseline)':  pd.Series(True, index=df.index),
    'A: pb > 0.75':          ~(df['price_band'] > 0.75),
    'B: z < -0.5':           ~(df['ret_60d_z'] < -0.5),
    'C: A OR B':             ~((df['price_band'] > 0.75) | (df['ret_60d_z'] < -0.5)),
    'D: pb > 0.60 (cons.)':  ~(df['price_band'] > 0.60),
    'E: D OR B':             ~((df['price_band'] > 0.60) | (df['ret_60d_z'] < -0.5)),
    'F: pb > 0.50 (aggr.)':  ~(df['price_band'] > 0.50),
}

# ── Run all rules ──────────────────────────────────────────────────────────
print(f"\n{'='*92}")
print(f"{'Rule':28s} {'AnnRet':>8s} {'AnnVol':>8s} {'Sharpe':>7s} {'MaxDD':>8s} {'Deploy%':>8s} {'Prem$':>10s}")
print('='*92)

results = []
for name, mask in rules.items():
    eq, prem = simulate(df, mask)
    ann_ret, ann_vol, sharpe, max_dd = stats(eq)
    deploy_pct = mask.iloc[1:].mean() * 100
    print(f"{name:28s} {ann_ret*100:>7.1f}% {ann_vol*100:>7.1f}% {sharpe:>7.2f} {max_dd*100:>7.1f}% "
          f"{deploy_pct:>7.0f}% ${prem:>9,.0f}")
    results.append({'name': name, 'ann_ret': ann_ret, 'ann_vol': ann_vol, 'sharpe': sharpe,
                    'max_dd': max_dd, 'deploy_pct': deploy_pct, 'premium': prem})

# ── Per-year breakdown for the best rule ───────────────────────────────────
best = max(results, key=lambda r: r['sharpe'])
print(f"\nBest Sharpe: {best['name']}")
best_mask = rules[best['name']]
best_eq, _ = simulate(df, best_mask)
yearly = best_eq.groupby(best_eq.index.year).apply(lambda x: (x.iloc[-1]/x.iloc[0])-1)
print(f"  Yearly returns: {dict((y, f'{r*100:.1f}%') for y, r in yearly.items())}")

# ── Weekly captured-premium statistics for income-target check ─────────────
print(f"\n{'='*60}")
print("WEEKLY CAPTURED PREMIUM (for $1500/wk target check)")
print('='*60)
for name, mask in rules.items():
    eq, prem = simulate(df, mask)
    daily_prem = pd.Series(0.0, index=df['date'])
    for i in range(1, len(df)):
        if mask.iloc[i]:
            daily_prem.iloc[i] = eq.iloc[i-1] * df['daily_premium_yield'].iloc[i]
    weekly = daily_prem.groupby(pd.Grouper(freq='W')).sum()
    pos_weeks = (weekly > 0).sum()
    above_1500 = (weekly >= 1500).sum()
    median_wk = weekly.median()
    print(f"  {name:28s} median=${median_wk:>6.0f}/wk, "
          f"weeks≥$1500: {above_1500}/{len(weekly)} ({above_1500/len(weekly)*100:.0f}%)")
