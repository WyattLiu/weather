#!/usr/bin/env python3
"""
INTC Reverse Calendar P&L Chart with Calibrated IV Model
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def bs_price(S, K, T, r, sigma, option_type='call'):
    if T <= 0.001:
        return max(0, S - K) if option_type == 'call' else max(0, K - S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# Current market data
SPOT = 54.00
r = 0.045

# Entry prices (from live data)
ENTRY = {
    'buy_jan23_53p': 1.91,   # 0DTE put
    'sell_jan30_53p': 2.51,  # 7DTE put
    'buy_jan23_55c': 2.25,   # 0DTE call
    'sell_jan30_55c': 2.89,  # 7DTE call
}

# Current IV levels
IV_0DTE = 1.59   # 159%
IV_7DTE = 0.94   # 94%

# Post-earnings IV (calibrated from term structure)
IV_POST_7DTE = 0.50  # 50% after crush

net_credit = (ENTRY['sell_jan30_53p'] - ENTRY['buy_jan23_53p']) + \
             (ENTRY['sell_jan30_55c'] - ENTRY['buy_jan23_55c'])

print(f"Net Credit: ${net_credit:.2f}")

# P&L calculation
def calc_rc_pnl(spot_exit, iv_post=IV_POST_7DTE):
    """Calculate Reverse Calendar P&L at exit"""
    T_remaining = 7/365  # 7 days left on Jan 30 options

    # 0DTE options expire at intrinsic
    exit_jan23_53p = max(0, 53 - spot_exit)
    exit_jan23_55c = max(0, spot_exit - 55)

    # 7DTE options valued with post-crush IV
    exit_jan30_53p = bs_price(spot_exit, 53, T_remaining, r, iv_post, 'put')
    exit_jan30_55c = bs_price(spot_exit, 55, T_remaining, r, iv_post, 'call')

    # P&L
    pnl_put = (exit_jan23_53p - ENTRY['buy_jan23_53p']) + (ENTRY['sell_jan30_53p'] - exit_jan30_53p)
    pnl_call = (exit_jan23_55c - ENTRY['buy_jan23_55c']) + (ENTRY['sell_jan30_55c'] - exit_jan30_55c)

    return pnl_put + pnl_call

# Generate P&L curves
spot_range = np.linspace(SPOT * 0.75, SPOT * 1.35, 200)
pct_moves = (spot_range / SPOT - 1) * 100

# Different IV crush scenarios
scenarios = {
    'Aggressive (40%)': 0.40,
    'Expected (50%)': 0.50,
    'Mild (60%)': 0.60,
    'No Crush (94%)': 0.94,
}

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: P&L by IV crush scenario
ax1 = axes[0, 0]
for label, iv in scenarios.items():
    pnl = [calc_rc_pnl(s, iv) for s in spot_range]
    ax1.plot(pct_moves, pnl, label=f'{label}', linewidth=2)

ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax1.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
ax1.axvline(x=-8, color='red', linestyle=':', alpha=0.5)
ax1.axvline(x=8, color='red', linestyle=':', alpha=0.5)
ax1.axvline(x=-12, color='orange', linestyle=':', alpha=0.5)
ax1.axvline(x=12, color='orange', linestyle=':', alpha=0.5)
ax1.fill_between(pct_moves, -3, 0, alpha=0.1, color='red')
ax1.set_xlabel('Stock Move (%)')
ax1.set_ylabel('P&L ($)')
ax1.set_title('P&L by Post-Earnings IV Level\n(Lower IV = More Crush = Better)')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-25, 35)
ax1.set_ylim(-2, 2.5)

# Plot 2: Expected scenario focus
ax2 = axes[0, 1]
pnl_expected = [calc_rc_pnl(s, 0.50) for s in spot_range]
ax2.plot(pct_moves, pnl_expected, linewidth=3, color='blue', label='Expected P&L (50% IV)')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax2.axvspan(-12, 12, alpha=0.15, color='yellow', label='Expected 8-12% zone')
ax2.axvline(x=-8, color='red', linestyle='--', alpha=0.7)
ax2.axvline(x=8, color='red', linestyle='--', alpha=0.7)
ax2.axvline(x=-12, color='orange', linestyle='--', alpha=0.7)
ax2.axvline(x=12, color='orange', linestyle='--', alpha=0.7)

# Mark key points
for move in [-12, -8, 0, 8, 12]:
    spot = SPOT * (1 + move/100)
    pnl = calc_rc_pnl(spot, 0.50)
    ax2.scatter([move], [pnl], s=100, zorder=5)
    ax2.annotate(f'${pnl:+.2f}', (move, pnl), textcoords="offset points",
                 xytext=(0, 10), ha='center', fontsize=9)

ax2.set_xlabel('Stock Move (%)')
ax2.set_ylabel('P&L ($)')
ax2.set_title('Expected P&L with 50% Post-Crush IV\n$53P/$55C Reverse Calendar')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-20, 20)
ax2.set_ylim(-1.5, 2)

# Plot 3: IV levels visualization
ax3 = axes[1, 0]
categories = ['0DTE (Jan 23)\nEarnings Day', '7DTE (Jan 30)\nPre-Earnings', '7DTE (Jan 30)\nPost-Earnings']
ivs = [IV_0DTE * 100, IV_7DTE * 100, IV_POST_7DTE * 100]
colors = ['red', 'orange', 'green']
bars = ax3.bar(categories, ivs, color=colors, alpha=0.7, edgecolor='black')

ax3.set_ylabel('Implied Volatility (%)')
ax3.set_title('IV Term Structure & Crush')
for bar, iv in zip(bars, ivs):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
             f'{iv:.0f}%', ha='center', fontsize=12, fontweight='bold')

# Add arrow showing the crush
ax3.annotate('', xy=(1.5, IV_POST_7DTE * 100), xytext=(1.5, IV_7DTE * 100),
             arrowprops=dict(arrowstyle='->', color='green', lw=3))
ax3.text(1.7, (IV_7DTE + IV_POST_7DTE) / 2 * 100, f'-{(IV_7DTE - IV_POST_7DTE)*100:.0f}%\nCRUSH',
         fontsize=11, color='green', fontweight='bold')

ax3.set_ylim(0, 180)

# Plot 4: Summary table
ax4 = axes[1, 1]
ax4.axis('off')

summary = f"""
INTC REVERSE CALENDAR - CALIBRATED MODEL
{'='*50}

POSITION: $53 Put / $55 Call (0DTE vs 7DTE)
SPOT: ${SPOT:.2f}

ENTRY PRICES:
  BUY  Jan 23 $53 Put  @ $1.91  (IV: 159%)
  SELL Jan 30 $53 Put  @ $2.51  (IV: 94%)
  BUY  Jan 23 $55 Call @ $2.25  (IV: 159%)
  SELL Jan 30 $55 Call @ $2.89  (IV: 94%)

  Net Credit: ${net_credit:.2f} (${net_credit*100:.0f} per combo)

IV CRUSH MODEL:
  0DTE: 159% → Expires at intrinsic
  7DTE: 94% → 50% (44pt crush)

EXPECTED P&L (with 50% post-crush IV):
  Move     P&L
  ─────────────────
  -12%    ${calc_rc_pnl(SPOT*0.88, 0.50):+.2f}
   -8%    ${calc_rc_pnl(SPOT*0.92, 0.50):+.2f}
    0%    ${calc_rc_pnl(SPOT, 0.50):+.2f}   ← Max loss zone
   +8%    ${calc_rc_pnl(SPOT*1.08, 0.50):+.2f}
  +12%    ${calc_rc_pnl(SPOT*1.12, 0.50):+.2f}
  +20%    ${calc_rc_pnl(SPOT*1.20, 0.50):+.2f}

KEY INSIGHT:
  Stock pins between $53-$55 = Max loss
  Big move either direction = Profit
"""

ax4.text(0.02, 0.98, summary, transform=ax4.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace')

plt.tight_layout()
plt.savefig('/home/wyatt/ibkr_guided_trade/intc_rc_calibrated.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nChart saved to: intc_rc_calibrated.png")
print("\nP&L Summary:")
for move in [-12, -8, -5, 0, 5, 8, 12, 20]:
    spot = SPOT * (1 + move/100)
    pnl = calc_rc_pnl(spot, 0.50)
    print(f"  {move:+3d}%: ${pnl:+.2f}")
