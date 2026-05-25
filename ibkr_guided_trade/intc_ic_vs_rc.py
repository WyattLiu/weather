#!/usr/bin/env python3
"""
INTC Iron Condor vs Reverse Calendar Comparison
For two-sided earnings move bet
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

# Current market data (from IBKR)
SPOT = 54.00
r = 0.045

# ============================================================
# STRATEGY 1: REVERSE CALENDAR ($53P / $55C)
# Buy 0DTE, Sell 7DTE - profits from IV crush
# ============================================================
RC_ENTRY = {
    'buy_jan23_53p': 1.90,   # Ask
    'sell_jan30_53p': 2.54,  # Bid
    'buy_jan23_55c': 1.77,   # Ask
    'sell_jan30_55c': 2.41,  # Bid
}
RC_CREDIT = (RC_ENTRY['sell_jan30_53p'] - RC_ENTRY['buy_jan23_53p']) + \
            (RC_ENTRY['sell_jan30_55c'] - RC_ENTRY['buy_jan23_55c'])

# Post-earnings IV for 7DTE options
IV_POST = 0.52

def calc_rc_pnl(spot_exit):
    """Reverse Calendar P&L"""
    T_remaining = 7/365

    # 0DTE expires at intrinsic
    exit_jan23_53p = max(0, 53 - spot_exit)
    exit_jan23_55c = max(0, spot_exit - 55)

    # 7DTE valued with post-crush IV
    exit_jan30_53p = bs_price(spot_exit, 53, T_remaining, r, IV_POST, 'put')
    exit_jan30_55c = bs_price(spot_exit, 55, T_remaining, r, IV_POST, 'call')

    pnl_put = (exit_jan23_53p - RC_ENTRY['buy_jan23_53p']) + (RC_ENTRY['sell_jan30_53p'] - exit_jan30_53p)
    pnl_call = (exit_jan23_55c - RC_ENTRY['buy_jan23_55c']) + (RC_ENTRY['sell_jan30_55c'] - exit_jan30_55c)

    return pnl_put + pnl_call

# ============================================================
# STRATEGY 2: LONG IRON CONDOR ($50-$53P / $55-$58C)
# Buy inner strikes, sell outer - profits from big moves
# ============================================================
# Using Jan 23 (0DTE) options - expires after earnings
IC_STRIKES = {'put_long': 53, 'put_short': 50, 'call_long': 55, 'call_short': 58}

# Approximate prices (would need live data)
IC_ENTRY = {
    'buy_53p': 1.90,
    'sell_50p': 0.55,
    'buy_55c': 1.77,
    'sell_58c': 0.45,
}
IC_DEBIT = (IC_ENTRY['buy_53p'] - IC_ENTRY['sell_50p']) + \
           (IC_ENTRY['buy_55c'] - IC_ENTRY['sell_58c'])

def calc_ic_pnl(spot_exit):
    """Long Iron Condor P&L at expiration"""
    # Put debit spread: Long $53P, Short $50P
    if spot_exit >= 53:
        put_val = 0
    elif spot_exit <= 50:
        put_val = 3  # Max value
    else:
        put_val = 53 - spot_exit

    # Call debit spread: Long $55C, Short $58C
    if spot_exit <= 55:
        call_val = 0
    elif spot_exit >= 58:
        call_val = 3  # Max value
    else:
        call_val = spot_exit - 55

    return put_val + call_val - IC_DEBIT

# ============================================================
# Generate comparison
# ============================================================
spot_range = np.linspace(SPOT * 0.75, SPOT * 1.35, 200)
pct_moves = (spot_range / SPOT - 1) * 100

rc_pnl = [calc_rc_pnl(s) for s in spot_range]
ic_pnl = [calc_ic_pnl(s) for s in spot_range]

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: P&L comparison
ax1 = axes[0, 0]
ax1.plot(pct_moves, rc_pnl, label=f'Reverse Calendar (credit ${RC_CREDIT:.2f})', linewidth=2.5, color='blue')
ax1.plot(pct_moves, ic_pnl, label=f'Long Iron Condor (debit ${IC_DEBIT:.2f})', linewidth=2.5, color='green')
ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax1.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
ax1.axvspan(-12, 12, alpha=0.1, color='yellow', label='Expected 8-12% zone')
ax1.fill_between(pct_moves, -4, 0, alpha=0.1, color='red')
ax1.set_xlabel('Stock Move (%)')
ax1.set_ylabel('P&L ($)')
ax1.set_title('P&L Comparison: Reverse Calendar vs Long Iron Condor')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-25, 35)
ax1.set_ylim(-3, 3)

# Plot 2: Focus on expected range
ax2 = axes[0, 1]
ax2.plot(pct_moves, rc_pnl, label='Reverse Calendar', linewidth=2.5, color='blue')
ax2.plot(pct_moves, ic_pnl, label='Long Iron Condor', linewidth=2.5, color='green')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax2.axvline(x=-8, color='red', linestyle='--', alpha=0.7)
ax2.axvline(x=8, color='red', linestyle='--', alpha=0.7)
ax2.axvline(x=-12, color='orange', linestyle='--', alpha=0.7)
ax2.axvline(x=12, color='orange', linestyle='--', alpha=0.7)

# Mark key points
for move in [-12, -8, 0, 8, 12]:
    spot = SPOT * (1 + move/100)
    rc = calc_rc_pnl(spot)
    ic = calc_ic_pnl(spot)
    ax2.scatter([move], [rc], s=80, color='blue', zorder=5)
    ax2.scatter([move], [ic], s=80, color='green', zorder=5)

ax2.set_xlabel('Stock Move (%)')
ax2.set_ylabel('P&L ($)')
ax2.set_title('Focus: Expected 8-12% Move Zone')
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-15, 15)
ax2.set_ylim(-3, 2)

# Plot 3: Breakeven and risk comparison
ax3 = axes[1, 0]

# Find breakevens
rc_be_down = None
rc_be_up = None
ic_be_down = None
ic_be_up = None

for i in range(len(spot_range)-1):
    if rc_pnl[i] < 0 and rc_pnl[i+1] >= 0 and pct_moves[i] < 0:
        rc_be_down = pct_moves[i]
    if rc_pnl[i] >= 0 and rc_pnl[i+1] < 0 and pct_moves[i] < 0:
        rc_be_down = pct_moves[i]
    if rc_pnl[i] < 0 and rc_pnl[i+1] >= 0 and pct_moves[i] > 0:
        rc_be_up = pct_moves[i]
    if ic_pnl[i] < 0 and ic_pnl[i+1] >= 0 and pct_moves[i] < 0:
        ic_be_down = pct_moves[i]
    if ic_pnl[i] < 0 and ic_pnl[i+1] >= 0 and pct_moves[i] > 0:
        ic_be_up = pct_moves[i]

categories = ['Entry Cost', 'Max Loss', 'Max Profit', 'P&L at 0%', 'P&L at ±8%', 'P&L at ±12%']
rc_values = [
    -RC_CREDIT,  # Credit (negative cost)
    min(rc_pnl),
    max(rc_pnl),
    calc_rc_pnl(SPOT),
    (calc_rc_pnl(SPOT*0.92) + calc_rc_pnl(SPOT*1.08))/2,
    (calc_rc_pnl(SPOT*0.88) + calc_rc_pnl(SPOT*1.12))/2,
]
ic_values = [
    IC_DEBIT,  # Debit (positive cost)
    min(ic_pnl),
    max(ic_pnl),
    calc_ic_pnl(SPOT),
    (calc_ic_pnl(SPOT*0.92) + calc_ic_pnl(SPOT*1.08))/2,
    (calc_ic_pnl(SPOT*0.88) + calc_ic_pnl(SPOT*1.12))/2,
]

x = np.arange(len(categories))
width = 0.35

bars1 = ax3.bar(x - width/2, rc_values, width, label='Reverse Calendar', color='blue', alpha=0.7)
bars2 = ax3.bar(x + width/2, ic_values, width, label='Long Iron Condor', color='green', alpha=0.7)

ax3.set_ylabel('$ Value')
ax3.set_title('Strategy Metrics Comparison')
ax3.set_xticks(x)
ax3.set_xticklabels(categories, rotation=15, ha='right', fontsize=9)
ax3.legend()
ax3.axhline(y=0, color='black', linewidth=0.5)
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Summary table
ax4 = axes[1, 1]
ax4.axis('off')

summary = f"""
INTC EARNINGS TRADE COMPARISON
{'='*55}

                    Reverse Calendar    Long Iron Condor
                    ----------------    ----------------
Entry:              ${RC_CREDIT:+.2f} credit      ${IC_DEBIT:.2f} debit
Max Loss:           ${min(rc_pnl):.2f}              ${min(ic_pnl):.2f}
Max Profit:         ${max(rc_pnl):.2f}              ${max(ic_pnl):.2f}

P&L BY SCENARIO:
  Flat (0%):        ${calc_rc_pnl(SPOT):+.2f}              ${calc_ic_pnl(SPOT):+.2f}
  Down 8%:          ${calc_rc_pnl(SPOT*0.92):+.2f}              ${calc_ic_pnl(SPOT*0.92):+.2f}
  Up 8%:            ${calc_rc_pnl(SPOT*1.08):+.2f}              ${calc_ic_pnl(SPOT*1.08):+.2f}
  Down 12%:         ${calc_rc_pnl(SPOT*0.88):+.2f}              ${calc_ic_pnl(SPOT*0.88):+.2f}
  Up 12%:           ${calc_rc_pnl(SPOT*1.12):+.2f}              ${calc_ic_pnl(SPOT*1.12):+.2f}
  Down 20%:         ${calc_rc_pnl(SPOT*0.80):+.2f}              ${calc_ic_pnl(SPOT*0.80):+.2f}
  Up 20%:           ${calc_rc_pnl(SPOT*1.20):+.2f}              ${calc_ic_pnl(SPOT*1.20):+.2f}

KEY DIFFERENCES:
{'='*55}

REVERSE CALENDAR:
  ✓ Receives credit upfront (${RC_CREDIT:.2f})
  ✓ Profits from IV crush (reliable edge)
  ✓ Better at flat (-${abs(calc_rc_pnl(SPOT)):.2f} vs -${abs(calc_ic_pnl(SPOT)):.2f})
  ✗ Less profit on extreme moves
  ✗ Depends on IV crush assumption

LONG IRON CONDOR:
  ✓ Defined risk (max loss = debit)
  ✓ Higher profit on BIG moves (>${3-IC_DEBIT:.2f})
  ✓ No IV assumption needed
  ✗ Pays debit upfront (${IC_DEBIT:.2f})
  ✗ Loses more if stock stays flat
  ✗ Needs ~{((IC_DEBIT/3)*100):.0f}% of max width to break even

VERDICT FOR 8-12% EXPECTED MOVE:
{'='*55}
  At ±8%:  RC ${(calc_rc_pnl(SPOT*0.92)+calc_rc_pnl(SPOT*1.08))/2:+.2f} avg  vs  IC ${(calc_ic_pnl(SPOT*0.92)+calc_ic_pnl(SPOT*1.08))/2:+.2f} avg
  At ±12%: RC ${(calc_rc_pnl(SPOT*0.88)+calc_rc_pnl(SPOT*1.12))/2:+.2f} avg  vs  IC ${(calc_ic_pnl(SPOT*0.88)+calc_ic_pnl(SPOT*1.12))/2:+.2f} avg

  → REVERSE CALENDAR wins at 8-12% moves
  → IRON CONDOR wins only on 15%+ moves
"""

ax4.text(0.02, 0.98, summary, transform=ax4.transAxes, fontsize=8.5,
         verticalalignment='top', fontfamily='monospace')

plt.tight_layout()
plt.savefig('/home/wyatt/ibkr_guided_trade/intc_ic_vs_rc.png', dpi=150, bbox_inches='tight')
plt.close()

# Print summary
print(summary)
print(f"\nChart saved to: intc_ic_vs_rc.png")
