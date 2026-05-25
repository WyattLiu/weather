#!/usr/bin/env python3
"""
IV Crush Explained - How IV behaves throughout the day after earnings
"""

import numpy as np
import matplotlib.pyplot as plt

print("""
================================================================================
IV CRUSH - HOW IT WORKS THROUGHOUT THE DAY
================================================================================

TIMELINE OF EARNINGS IV CRUSH
==============================

PRE-EARNINGS (Day Before Close):
  0DTE IV: 200-300%  ← Maximum earnings premium
  7DTE IV: 90-120%   ← Elevated but less
  Gap: ~100-150 points

EARNINGS RELEASE:
  Usually after market close (4 PM) or before open (8 AM)
  Stock gaps up/down based on results

POST-EARNINGS DAY (The Crush):

  9:30 AM - MARKET OPEN
  ├─ 0DTE IV: Starts crashing immediately
  │   - If OTM: IV irrelevant (going to $0)
  │   - If ITM: Trades near intrinsic, IV collapses
  │
  ├─ 7DTE IV: Initial shock
  │   - Opens ~20-40% lower than yesterday
  │   - Example: 95% → 60-70%
  │
  └─ This is the FASTEST crush period (9:30-10:00)

  10:00 AM - 10:30 AM
  ├─ 0DTE: Near intrinsic value only
  │   - Far OTM: $0.01-0.05
  │   - ATM: Still has some extrinsic
  │   - ITM: Mostly intrinsic
  │
  ├─ 7DTE: Stabilizing
  │   - IV settles to post-earnings level
  │   - Usually 55-65% for volatile stocks
  │
  └─ OPTIMAL EXIT WINDOW for reverse calendar

  10:30 AM - 12:00 PM
  ├─ 0DTE: Theta decay accelerates
  │   - ATM options lose value rapidly
  │   - Market makers widen spreads
  │
  └─ 7DTE: Mostly stable, small fluctuations

  12:00 PM - 3:00 PM
  ├─ 0DTE: "Dead zone"
  │   - Very low liquidity
  │   - Wide bid-ask spreads
  │   - Quotes become unreliable (negative bids!)
  │
  └─ 7DTE: Normal trading

  3:00 PM - 4:00 PM
  ├─ 0DTE: Final hour
  │   - Extreme theta decay
  │   - Most 0DTE options expire worthless
  │
  └─ 7DTE: End of day adjustments


IV CRUSH BY OPTION TYPE (0DTE)
===============================

FAR OTM (e.g., $55 Call when stock at $47):
  Pre-earnings: High IV, small premium (~$2)
  Post-earnings: IV irrelevant, worth ~$0.01
  Pattern: Straight line down to zero

ATM (at the money):
  Pre-earnings: Highest premium
  Post-earnings: Rapid IV crush + theta decay
  Pattern: Steep decline, especially AM

ITM (in the money, e.g., $52 Put when stock at $47):
  Pre-earnings: Intrinsic + high extrinsic
  Post-earnings: Mostly intrinsic, extrinsic crushed
  Pattern: Converges to intrinsic value ($5)


WHAT WE SAW IN INTC TRADE
==========================

Pre-earnings (Jan 22, ~10 AM):
  Jan 23 $52 Put: IV ~210%, Price ~$1.90
  Jan 23 $55 Call: IV ~210%, Price ~$1.77
  Jan 30 $52 Put: IV ~95%, Price ~$2.50
  Jan 30 $55 Call: IV ~95%, Price ~$2.40

Post-earnings (Jan 23, 9:35 AM):
  INTC: $54 → $47 (-13%)

  Jan 23 $52 Put: Now ITM by $5!
    - Price: ~$5.00 (mostly intrinsic)
    - IV: Not really meaningful (expires today)
    - Extrinsic: ~$0.05

  Jan 23 $55 Call: Far OTM
    - Price: ~$0.01
    - IV: Shows weird values (low liquidity)
    - Worth essentially zero

  Jan 30 $52 Put: IV crushed
    - Pre: 95% → Post: ~60%
    - Price: ~$5.50 (intrinsic $5 + time value $0.50)
    - This is where our profit came from!

  Jan 30 $55 Call: IV crushed, OTM
    - Pre: 95% → Post: ~60%
    - Price: ~$0.25
    - Small time value remaining


KEY LESSONS
============

1. 0DTE IV IS UNRELIABLE AFTER OPEN
   - Low liquidity, wide spreads
   - Quote data often corrupted
   - Use NATURAL price, not MID

2. IV CRUSH HAPPENS FAST (9:30-10:00 AM)
   - 70% of crush in first 30 minutes
   - Don't wait - close early in this window

3. 7DTE OPTIONS STABILIZE BY 10:30 AM
   - Initial shock, then settles
   - Good time to assess and close

4. INTRINSIC VALUE DOMINATES POST-MOVE
   - ITM options: Price ≈ intrinsic
   - OTM options: Price → 0 (0DTE)
   - Extrinsic/IV only matters for longer-dated

5. THE PROFIT COMES FROM 7DTE CRUSH
   - We sold 7DTE at 95% IV
   - Bought back at ~60% IV
   - That 35-point crush = our profit


TYPICAL IV CRUSH PERCENTAGES
=============================

Strong crush (clear result):
  0DTE: IV meaningless (expires)
  7DTE: -40 to -50 points (95% → 50%)

Normal crush:
  0DTE: IV meaningless
  7DTE: -30 to -40 points (95% → 60%)

Weak crush (uncertainty remains):
  0DTE: IV meaningless
  7DTE: -15 to -25 points (95% → 75%)

No crush (rare - ongoing news):
  7DTE: IV stays elevated or rises
  Example: Accounting scandal, CEO resignation
""")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Time axis (market hours)
hours = np.linspace(9.5, 16, 100)  # 9:30 AM to 4:00 PM

# Plot 1: IV throughout the day
ax1 = axes[0, 0]

# 0DTE Put (ITM after move)
iv_0dte_itm = 150 * np.exp(-2 * (hours - 9.5)) + 30  # Rapid decay to near intrinsic

# 0DTE Call (OTM after move)
iv_0dte_otm = 180 * np.exp(-3 * (hours - 9.5)) + 20  # Even faster decay

# 7DTE - crush then stabilize
iv_7dte = 95 - 35 * (1 - np.exp(-1.5 * (hours - 9.5)))

ax1.plot(hours, iv_0dte_itm, label='0DTE ITM Put ($52P)', linewidth=2, color='blue')
ax1.plot(hours, iv_0dte_otm, label='0DTE OTM Call ($55C)', linewidth=2, color='red', linestyle='--')
ax1.plot(hours, iv_7dte, label='7DTE Options', linewidth=2.5, color='green')
ax1.axvline(x=10, color='orange', linestyle=':', alpha=0.7, label='Optimal exit window')
ax1.axvline(x=10.5, color='orange', linestyle=':', alpha=0.7)
ax1.axhspan(55, 65, alpha=0.1, color='green', label='7DTE post-crush zone')
ax1.set_xlabel('Time (Market Hours)')
ax1.set_ylabel('Implied Volatility (%)')
ax1.set_title('IV Crush Throughout the Day (Post-Earnings)')
ax1.legend(loc='upper right', fontsize=8)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(9.5, 16)
ax1.set_ylim(0, 200)
ax1.set_xticks([9.5, 10, 10.5, 11, 12, 13, 14, 15, 16])
ax1.set_xticklabels(['9:30', '10:00', '10:30', '11:00', '12:00', '1:00', '2:00', '3:00', '4:00'])

# Plot 2: Option price throughout the day
ax2 = axes[0, 1]

# 0DTE ITM Put (stock at $47, strike $52)
price_0dte_put = 5.0 + 0.5 * np.exp(-2 * (hours - 9.5))  # Converges to $5 intrinsic

# 0DTE OTM Call (worthless)
price_0dte_call = 0.50 * np.exp(-3 * (hours - 9.5)) + 0.01

# 7DTE ITM Put
price_7dte_put = 5.0 + 1.0 * np.exp(-0.5 * (hours - 9.5))  # Slower decay

# 7DTE OTM Call
price_7dte_call = 0.60 * np.exp(-0.3 * (hours - 9.5)) + 0.15

ax2.plot(hours, price_0dte_put, label='0DTE $52 Put (ITM)', linewidth=2, color='blue')
ax2.plot(hours, price_0dte_call, label='0DTE $55 Call (OTM)', linewidth=2, color='red', linestyle='--')
ax2.plot(hours, price_7dte_put, label='7DTE $52 Put', linewidth=2, color='darkblue', linestyle='-.')
ax2.plot(hours, price_7dte_call, label='7DTE $55 Call', linewidth=2, color='darkred', linestyle='-.')
ax2.axhline(y=5.0, color='gray', linestyle=':', alpha=0.5, label='$52P Intrinsic ($5)')
ax2.set_xlabel('Time (Market Hours)')
ax2.set_ylabel('Option Price ($)')
ax2.set_title('Option Prices Throughout the Day')
ax2.legend(loc='upper right', fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(9.5, 16)
ax2.set_ylim(0, 7)
ax2.set_xticks([9.5, 10, 10.5, 11, 12, 13, 14, 15, 16])
ax2.set_xticklabels(['9:30', '10:00', '10:30', '11:00', '12:00', '1:00', '2:00', '3:00', '4:00'])

# Plot 3: Bid-Ask spread widening
ax3 = axes[1, 0]

# 0DTE spread widens dramatically
spread_0dte = 0.05 + 0.3 * (1 - np.exp(-0.5 * (hours - 9.5)))

# 7DTE spread stays reasonable
spread_7dte = 0.08 + 0.05 * np.sin((hours - 9.5) * 0.5)

ax3.plot(hours, spread_0dte, label='0DTE Bid-Ask Spread', linewidth=2, color='red')
ax3.plot(hours, spread_7dte, label='7DTE Bid-Ask Spread', linewidth=2, color='green')
ax3.axvspan(12, 15, alpha=0.1, color='red', label='0DTE "dead zone"')
ax3.set_xlabel('Time (Market Hours)')
ax3.set_ylabel('Bid-Ask Spread ($)')
ax3.set_title('Bid-Ask Spread Widening (0DTE Liquidity Issue)')
ax3.legend(loc='upper left', fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(9.5, 16)
ax3.set_ylim(0, 0.5)
ax3.set_xticks([9.5, 10, 10.5, 11, 12, 13, 14, 15, 16])
ax3.set_xticklabels(['9:30', '10:00', '10:30', '11:00', '12:00', '1:00', '2:00', '3:00', '4:00'])

# Plot 4: Summary table
ax4 = axes[1, 1]
ax4.axis('off')

summary = """
IV CRUSH TIMELINE SUMMARY
=========================

9:30 AM  MARKET OPEN
         • 70% of IV crush happens here
         • 7DTE: 95% → 65% in minutes
         • 0DTE: Trades at intrinsic ± noise

10:00 AM OPTIMAL EXIT START
         • IV crush mostly complete
         • Prices stabilizing
         • Best liquidity of the day

10:30 AM OPTIMAL EXIT END
         • 7DTE settled at post-crush level
         • 0DTE liquidity declining
         • Close position by now!

12:00 PM 0DTE DEAD ZONE BEGINS
         • Wide spreads on 0DTE
         • Unreliable quotes
         • Bad data (negative bids)

3:00 PM  FINAL HOUR
         • 0DTE theta decay extreme
         • Most expire worthless

ACTION PLAN:
  1. Check prices at 9:30-9:35 AM
  2. Place close order 9:35-10:00 AM
  3. Use NATURAL price for 0DTE legs
  4. Be done by 10:30 AM
"""

ax4.text(0.02, 0.98, summary, transform=ax4.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace')

plt.tight_layout()
plt.savefig('/home/wyatt/ibkr_guided_trade/iv_crush_explained.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nChart saved to: iv_crush_explained.png")
