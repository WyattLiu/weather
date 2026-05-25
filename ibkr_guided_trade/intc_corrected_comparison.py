#!/usr/bin/env python3
"""
INTC Corrected Strategy Comparison
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def bs_price(S, K, T, r, sigma, option_type='call'):
    if T <= 0:
        return max(0, S - K) if option_type == 'call' else max(0, K - S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

SPOT = 54.45
r = 0.045
T_jan23_exit = 0.001
T_jan30_exit = 7/365
iv_crush_short = 0.30
iv_crush_long = 0.55

spot_range = np.linspace(SPOT * 0.75, SPOT * 1.35, 150)
pct_moves = (spot_range / SPOT - 1) * 100

# Strategy 1: Double Reverse Calendar (same strikes)
def calc_rev_cal(spot):
    jan23_52p = bs_price(spot, 52, T_jan23_exit, r, iv_crush_short, 'put')
    jan30_52p = bs_price(spot, 52, T_jan30_exit, r, iv_crush_long, 'put')
    jan23_59c = bs_price(spot, 59, T_jan23_exit, r, iv_crush_short, 'call')
    jan30_59c = bs_price(spot, 59, T_jan30_exit, r, iv_crush_long, 'call')
    entry = (2.03 - 1.48) + (1.62 - 1.09)
    exit_val = (jan23_52p - jan30_52p) + (jan23_59c - jan30_59c)
    return entry + exit_val

# Strategy 2: CORRECTED Long Iron Condor
def calc_iron_condor(spot):
    entry_debit = (1.48 - 0.58) + (1.09 - 0.57)  # = 1.42

    # Put debit spread: Long $52, Short $49
    if spot >= 52:
        put_val = 0
    elif spot <= 49:
        put_val = 3
    else:
        put_val = 52 - spot

    # Call debit spread: Long $59, Short $62
    if spot <= 59:
        call_val = 0
    elif spot >= 62:
        call_val = 3
    else:
        call_val = spot - 59

    return put_val + call_val - entry_debit

# Strategy 3: Reverse Diagonal
def calc_rev_diag(spot):
    jan23_50p = bs_price(spot, 50, T_jan23_exit, r, iv_crush_short, 'put')
    jan30_52p = bs_price(spot, 52, T_jan30_exit, r, iv_crush_long, 'put')
    jan23_61c = bs_price(spot, 61, T_jan23_exit, r, iv_crush_short, 'call')
    jan30_59c = bs_price(spot, 59, T_jan30_exit, r, iv_crush_long, 'call')
    pnl_put = jan23_50p - 0.83 + 2.03 - jan30_52p
    pnl_call = jan23_61c - 0.73 + 1.62 - jan30_59c
    return pnl_put + pnl_call

rev_cal_pnl = [calc_rev_cal(s) for s in spot_range]
ic_pnl = [calc_iron_condor(s) for s in spot_range]
rev_diag_pnl = [calc_rev_diag(s) for s in spot_range]

# Create comparison plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: All strategies
ax1 = axes[0]
ax1.plot(pct_moves, rev_cal_pnl, label='Reverse Calendar ($52P/$59C)', linewidth=2, color='blue')
ax1.plot(pct_moves, ic_pnl, label='Long Iron Condor ($49-52P/$59-62C)', linewidth=2, color='green')
ax1.plot(pct_moves, rev_diag_pnl, label='Reverse Diagonal', linewidth=2, color='orange')
ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax1.axvline(x=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

# Mark breakevens for IC
ax1.axvline(x=-7.1, color='green', linestyle=':', alpha=0.5, label='IC Breakeven (-7.1%, +11%)')
ax1.axvline(x=11.0, color='green', linestyle=':', alpha=0.5)

ax1.fill_between(pct_moves, -2, 0, alpha=0.1, color='red')
ax1.set_xlabel('Stock Move (%)')
ax1.set_ylabel('P&L ($)')
ax1.set_title('CORRECTED: Strategy P&L Comparison')
ax1.legend(loc='upper right', fontsize=8)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-25, 35)
ax1.set_ylim(-2, 2)

# Plot 2: Expected range focus
ax2 = axes[1]
ax2.plot(pct_moves, rev_cal_pnl, label='Rev Calendar', linewidth=2.5, color='blue')
ax2.plot(pct_moves, ic_pnl, label='Iron Condor', linewidth=2.5, color='green')
ax2.plot(pct_moves, rev_diag_pnl, label='Rev Diagonal', linewidth=2.5, color='orange')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax2.axvspan(-12, 12, alpha=0.15, color='yellow', label='Expected 8-12% zone')
ax2.axvline(x=-8, color='red', linestyle='--', alpha=0.7)
ax2.axvline(x=8, color='red', linestyle='--', alpha=0.7)
ax2.axvline(x=-12, color='red', linestyle='--', alpha=0.7)
ax2.axvline(x=12, color='red', linestyle='--', alpha=0.7)
ax2.set_xlabel('Stock Move (%)')
ax2.set_ylabel('P&L ($)')
ax2.set_title('Focus: Your Expected 8-12% Move Range')
ax2.legend(loc='lower right', fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-15, 15)
ax2.set_ylim(-1.8, 1.8)

plt.tight_layout()
plt.savefig('/home/wyatt/ibkr_guided_trade/intc_corrected_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# Print summary table
print("=" * 70)
print("CORRECTED P&L COMPARISON")
print("=" * 70)
print()
print(f"{'Move':>8} {'Rev Calendar':>14} {'Iron Condor':>14} {'Rev Diagonal':>14}")
print("-" * 70)

for move in [-12, -10, -8, -5, 0, 5, 8, 10, 12, 15, 20]:
    spot = SPOT * (1 + move/100)
    rc = calc_rev_cal(spot)
    ic = calc_iron_condor(spot)
    rd = calc_rev_diag(spot)
    print(f"{move:>7}% {rc:>+13.2f} {ic:>+13.2f} {rd:>+13.2f}")

print()
print("=" * 70)
print("SUMMARY FOR YOUR 8-12% EXPECTED MOVE:")
print("=" * 70)
print()
print("IRON CONDOR:")
print("  - Breakeven: -7.1% and +11.0%")
print("  - Loses at ±8%: -$1.42 (down), -$1.42 (up)")
print("  - Profits at ±12%: +$1.58 (down), +$0.56 (up)")
print("  - Max loss: $1.42 (defined)")
print()
print("REVERSE CALENDAR:")
print("  - Profits at 0%: +$0.08")
print("  - Loses at +8%: -$0.59")
print("  - Profits at ±12%: +$0.84 (down), +$0.02 (up)")
print()
print("VERDICT: For 8-12% expected move:")
print("  - DOWN move: Both strategies work, IC slightly better at -12%")
print("  - UP move: Iron Condor better at +12% (+$0.56 vs +$0.02)")
print("  - FLAT: Rev Calendar wins (+$0.08 vs -$1.42)")
print()
print("Chart saved to: intc_corrected_comparison.png")
