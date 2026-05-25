#!/usr/bin/env python3
"""
INTC Strategy Comparison: Double Reverse Calendar vs Iron Condor
Comparing bid-ask impact and risk/reward
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def bs_price(S, K, T, r, sigma, option_type='call'):
    """Black-Scholes option pricing"""
    if T <= 0:
        if option_type == 'call':
            return max(0, S - K)
        else:
            return max(0, K - S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# Current INTC price
SPOT = 54.45  # From the option chain (ATM ~$54-55)
r = 0.045

# ============================================================
# ACTUAL BID-ASK DATA FROM WEALTHSIMPLE
# ============================================================

print("=" * 80)
print("INTC BID-ASK SPREADS - ACTUAL MARKET DATA")
print("=" * 80)
print()

# Jan 23 (0DTE) - Options we BUY in Reverse Calendar
jan23_data = {
    '$52 Put': {'bid': 1.46, 'ask': 1.48, 'iv': 1.559},
    '$59 Call': {'bid': 1.02, 'ask': 1.09, 'iv': 1.696},
}

# Jan 30 (7DTE) - Options we SELL in Reverse Calendar
jan30_data = {
    '$52 Put': {'bid': 2.03, 'ask': 2.16, 'iv': 0.941},
    '$59 Call': {'bid': 1.62, 'ask': 1.68, 'iv': 0.994},
}

print("JAN 23 OPTIONS (0DTE - Earnings Day):")
print("-" * 70)
print(f"{'Option':<12} {'Bid':>8} {'Ask':>8} {'Spread':>8} {'Spread%':>10} {'IV':>8}")
print("-" * 70)
for name, data in jan23_data.items():
    spread = data['ask'] - data['bid']
    mid = (data['bid'] + data['ask']) / 2
    spread_pct = spread / mid * 100
    print(f"{name:<12} ${data['bid']:>7.2f} ${data['ask']:>7.2f} ${spread:>7.2f} {spread_pct:>9.1f}% {data['iv']*100:>7.0f}%")

print()
print("JAN 30 OPTIONS (7DTE):")
print("-" * 70)
for name, data in jan30_data.items():
    spread = data['ask'] - data['bid']
    mid = (data['bid'] + data['ask']) / 2
    spread_pct = spread / mid * 100
    print(f"{name:<12} ${data['bid']:>7.2f} ${data['ask']:>7.2f} ${spread:>7.2f} {spread_pct:>9.1f}% {data['iv']*100:>7.0f}%")

# ============================================================
# STRATEGY 1: DOUBLE REVERSE CALENDAR
# ============================================================
print()
print("=" * 80)
print("STRATEGY 1: DOUBLE REVERSE CALENDAR")
print("=" * 80)
print()

# Reverse Calendar: Buy short-dated, Sell long-dated (same strike)
# We BUY at ASK, SELL at BID (worst case entry)

# Put calendar (sell Jan30, buy Jan23) at $52 strike
put_credit = jan30_data['$52 Put']['bid'] - jan23_data['$52 Put']['ask']
# Call calendar (sell Jan30, buy Jan23) at $59 strike
call_credit = jan30_data['$59 Call']['bid'] - jan23_data['$59 Call']['ask']

total_credit = put_credit + call_credit

print("ENTRY (Worst Case - Buy at Ask, Sell at Bid):")
print(f"  Buy  Jan 23 $52 Put  @ ${jan23_data['$52 Put']['ask']:.2f}")
print(f"  Sell Jan 30 $52 Put  @ ${jan30_data['$52 Put']['bid']:.2f}")
print(f"  Put Calendar Credit:   ${put_credit:.2f}")
print()
print(f"  Buy  Jan 23 $59 Call @ ${jan23_data['$59 Call']['ask']:.2f}")
print(f"  Sell Jan 30 $59 Call @ ${jan30_data['$59 Call']['bid']:.2f}")
print(f"  Call Calendar Credit:  ${call_credit:.2f}")
print()
print(f"  TOTAL NET CREDIT:      ${total_credit:.2f} (${total_credit*100:.0f}/contract)")
print()

# Mid-price entry (more realistic)
put_credit_mid = (jan30_data['$52 Put']['bid'] + jan30_data['$52 Put']['ask'])/2 - (jan23_data['$52 Put']['bid'] + jan23_data['$52 Put']['ask'])/2
call_credit_mid = (jan30_data['$59 Call']['bid'] + jan30_data['$59 Call']['ask'])/2 - (jan23_data['$59 Call']['bid'] + jan23_data['$59 Call']['ask'])/2
total_credit_mid = put_credit_mid + call_credit_mid

print("ENTRY (Mid Price - More Realistic):")
print(f"  Put Calendar Credit:   ${put_credit_mid:.2f}")
print(f"  Call Calendar Credit:  ${call_credit_mid:.2f}")
print(f"  TOTAL NET CREDIT:      ${total_credit_mid:.2f} (${total_credit_mid*100:.0f}/contract)")
print()

# Slippage analysis
slippage = total_credit_mid - total_credit
print(f"SLIPPAGE COST (Mid vs Worst): ${abs(slippage):.2f} ({abs(slippage)/total_credit_mid*100:.1f}%)")

# ============================================================
# STRATEGY 2: IRON CONDOR (Long - buying wings)
# ============================================================
print()
print("=" * 80)
print("STRATEGY 2: LONG IRON CONDOR (Buying Wings)")
print("=" * 80)
print()

# Long Iron Condor: Buy OTM put spread + Buy OTM call spread
# This PROFITS from movement, costs premium upfront

# Get additional strikes for iron condor
# Use $49/$52 put spread (buy $52, sell $49) + $59/$62 call spread (buy $59, sell $62)

# From the data we pulled:
# Jan 23 puts: $49 Put bid/ask = 0.58/0.61, $52 Put = 1.46/1.48
# Jan 23 calls: $59 Call = 1.02/1.09, $62 Call = 0.57/0.60

jan23_ic = {
    '$49 Put': {'bid': 0.58, 'ask': 0.61},
    '$52 Put': {'bid': 1.46, 'ask': 1.48},
    '$59 Call': {'bid': 1.02, 'ask': 1.09},
    '$62 Call': {'bid': 0.57, 'ask': 0.60},
}

# Buy $52 put, sell $49 put (debit spread)
put_spread_debit = jan23_ic['$52 Put']['ask'] - jan23_ic['$49 Put']['bid']
# Buy $59 call, sell $62 call (debit spread)
call_spread_debit = jan23_ic['$59 Call']['ask'] - jan23_ic['$62 Call']['bid']
total_debit = put_spread_debit + call_spread_debit

print("LONG IRON CONDOR: $49/$52 Put Spread + $59/$62 Call Spread")
print()
print("ENTRY (Worst Case - Buy at Ask, Sell at Bid):")
print(f"  Buy  Jan 23 $52 Put  @ ${jan23_ic['$52 Put']['ask']:.2f}")
print(f"  Sell Jan 23 $49 Put  @ ${jan23_ic['$49 Put']['bid']:.2f}")
print(f"  Put Spread Debit:      ${put_spread_debit:.2f}")
print()
print(f"  Buy  Jan 23 $59 Call @ ${jan23_ic['$59 Call']['ask']:.2f}")
print(f"  Sell Jan 23 $62 Call @ ${jan23_ic['$62 Call']['bid']:.2f}")
print(f"  Call Spread Debit:     ${call_spread_debit:.2f}")
print()
print(f"  TOTAL NET DEBIT:       ${total_debit:.2f} (${total_debit*100:.0f}/contract)")
print(f"  MAX PROFIT:            ${3.00 - total_debit:.2f} (spread width $3 - debit)")
print(f"  MAX LOSS:              ${total_debit:.2f} (total premium paid)")
print()

# Mid price
put_spread_mid = (jan23_ic['$52 Put']['bid']+jan23_ic['$52 Put']['ask'])/2 - (jan23_ic['$49 Put']['bid']+jan23_ic['$49 Put']['ask'])/2
call_spread_mid = (jan23_ic['$59 Call']['bid']+jan23_ic['$59 Call']['ask'])/2 - (jan23_ic['$62 Call']['bid']+jan23_ic['$62 Call']['ask'])/2
total_debit_mid = put_spread_mid + call_spread_mid

print("ENTRY (Mid Price):")
print(f"  Put Spread Debit:      ${put_spread_mid:.2f}")
print(f"  Call Spread Debit:     ${call_spread_mid:.2f}")
print(f"  TOTAL NET DEBIT:       ${total_debit_mid:.2f}")
print(f"  MAX PROFIT:            ${3.00 - total_debit_mid:.2f}")

# ============================================================
# P&L COMPARISON
# ============================================================
print()
print("=" * 80)
print("RISK/REWARD COMPARISON")
print("=" * 80)
print()

# Parameters for exit calculation
T_exit = 0.001  # Nearly expired
iv_crush_jan23 = 0.30  # Post-earnings IV for 0DTE
iv_crush_jan30 = 0.55  # Post-earnings IV for 7DTE (less crush)

spot_range = np.linspace(SPOT * 0.80, SPOT * 1.20, 100)
pct_moves = (spot_range / SPOT - 1) * 100

# Calculate P&L for Double Reverse Calendar
def calc_rev_cal_pnl(spot_exit, entry_credit):
    # Jan 23 options at exit (nearly worthless after IV crush)
    jan23_put_exit = bs_price(spot_exit, 52, T_exit, r, iv_crush_jan23, 'put')
    jan23_call_exit = bs_price(spot_exit, 59, T_exit, r, iv_crush_jan23, 'call')

    # Jan 30 options at exit (7 days left, IV crushed)
    T_jan30 = 7/365
    jan30_put_exit = bs_price(spot_exit, 52, T_jan30, r, iv_crush_jan30, 'put')
    jan30_call_exit = bs_price(spot_exit, 59, T_jan30, r, iv_crush_jan30, 'call')

    # P&L: we're long Jan23, short Jan30
    # Entry value of Jan23 options (what we paid)
    jan23_put_entry = jan23_data['$52 Put']['ask']
    jan23_call_entry = jan23_data['$59 Call']['ask']
    jan30_put_entry = jan30_data['$52 Put']['bid']
    jan30_call_entry = jan30_data['$59 Call']['bid']

    pnl_jan23_put = jan23_put_exit - jan23_put_entry
    pnl_jan23_call = jan23_call_exit - jan23_call_entry
    pnl_jan30_put = jan30_put_entry - jan30_put_exit  # We sold this
    pnl_jan30_call = jan30_call_entry - jan30_call_exit  # We sold this

    return pnl_jan23_put + pnl_jan23_call + pnl_jan30_put + pnl_jan30_call

# Calculate P&L for Long Iron Condor
def calc_ic_pnl(spot_exit, entry_debit):
    # At expiry, value is intrinsic
    put_spread_value = max(0, min(3, 52 - spot_exit)) - max(0, min(3, 49 - spot_exit))
    call_spread_value = max(0, min(3, spot_exit - 59)) - max(0, min(3, spot_exit - 62))
    total_value = put_spread_value + call_spread_value
    return total_value - entry_debit

rev_cal_pnls = [calc_rev_cal_pnl(s, total_credit) for s in spot_range]
ic_pnls = [calc_ic_pnl(s, total_debit) for s in spot_range]

# Plot comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: P&L curves comparison
ax1 = axes[0, 0]
ax1.plot(pct_moves, rev_cal_pnls, label='Double Reverse Calendar', linewidth=2, color='blue')
ax1.plot(pct_moves, ic_pnls, label='Long Iron Condor', linewidth=2, color='green')
ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax1.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
ax1.axvline(x=-8, color='red', linestyle=':', alpha=0.5)
ax1.axvline(x=8, color='red', linestyle=':', alpha=0.5)
ax1.axvline(x=-12, color='orange', linestyle=':', alpha=0.5)
ax1.axvline(x=12, color='orange', linestyle=':', alpha=0.5)
ax1.set_xlabel('Stock Move (%)')
ax1.set_ylabel('P&L ($)')
ax1.set_title('P&L Comparison: Reverse Calendar vs Iron Condor')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-20, 20)

# Plot 2: Return on Risk comparison
ax2 = axes[0, 1]
# For reverse calendar, risk is the potential loss if IV doesn't crush
rev_cal_risk = abs(min(rev_cal_pnls))  # Approximate max loss
ic_risk = total_debit

rev_cal_returns = [(p / 1.0) * 100 for p in rev_cal_pnls]  # Return on ~$1 margin
ic_returns = [(p / ic_risk) * 100 for p in ic_pnls]

ax2.plot(pct_moves, rev_cal_returns, label='Reverse Calendar (on $1 margin)', linewidth=2, color='blue')
ax2.plot(pct_moves, ic_returns, label='Iron Condor (on premium)', linewidth=2, color='green')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax2.set_xlabel('Stock Move (%)')
ax2.set_ylabel('Return (%)')
ax2.set_title('Return on Capital Comparison')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-20, 20)

# Plot 3: Bid-Ask Impact
ax3 = axes[1, 0]
categories = ['Jan23 $52P', 'Jan23 $59C', 'Jan30 $52P', 'Jan30 $59C']
spreads = [
    jan23_data['$52 Put']['ask'] - jan23_data['$52 Put']['bid'],
    jan23_data['$59 Call']['ask'] - jan23_data['$59 Call']['bid'],
    jan30_data['$52 Put']['ask'] - jan30_data['$52 Put']['bid'],
    jan30_data['$59 Call']['ask'] - jan30_data['$59 Call']['bid'],
]
spread_pcts = [
    (jan23_data['$52 Put']['ask'] - jan23_data['$52 Put']['bid']) / ((jan23_data['$52 Put']['ask'] + jan23_data['$52 Put']['bid'])/2) * 100,
    (jan23_data['$59 Call']['ask'] - jan23_data['$59 Call']['bid']) / ((jan23_data['$59 Call']['ask'] + jan23_data['$59 Call']['bid'])/2) * 100,
    (jan30_data['$52 Put']['ask'] - jan30_data['$52 Put']['bid']) / ((jan30_data['$52 Put']['ask'] + jan30_data['$52 Put']['bid'])/2) * 100,
    (jan30_data['$59 Call']['ask'] - jan30_data['$59 Call']['bid']) / ((jan30_data['$59 Call']['ask'] + jan30_data['$59 Call']['bid'])/2) * 100,
]

x = np.arange(len(categories))
width = 0.35
bars1 = ax3.bar(x - width/2, spreads, width, label='Spread ($)', color='steelblue')
ax3_twin = ax3.twinx()
bars2 = ax3_twin.bar(x + width/2, spread_pcts, width, label='Spread (%)', color='coral')
ax3.set_xticks(x)
ax3.set_xticklabels(categories)
ax3.set_ylabel('Spread ($)', color='steelblue')
ax3_twin.set_ylabel('Spread (%)', color='coral')
ax3.set_title('Bid-Ask Spreads by Option')
ax3.legend(loc='upper left')
ax3_twin.legend(loc='upper right')

# Plot 4: Summary table
ax4 = axes[1, 1]
ax4.axis('off')

summary_text = f"""
STRATEGY COMPARISON SUMMARY
{'='*50}

DOUBLE REVERSE CALENDAR ($52P / $59C):
  Entry Credit (worst):     ${total_credit:.2f}
  Entry Credit (mid):       ${total_credit_mid:.2f}
  Est. Margin Required:     ~$1,000-1,500

  P&L at 0% move:           ${calc_rev_cal_pnl(SPOT, total_credit):.2f}
  P&L at +8% move:          ${calc_rev_cal_pnl(SPOT*1.08, total_credit):.2f}
  P&L at -8% move:          ${calc_rev_cal_pnl(SPOT*0.92, total_credit):.2f}
  P&L at +12% move:         ${calc_rev_cal_pnl(SPOT*1.12, total_credit):.2f}
  P&L at -12% move:         ${calc_rev_cal_pnl(SPOT*0.88, total_credit):.2f}

LONG IRON CONDOR ($49/$52P + $59/$62C):
  Entry Debit (worst):      ${total_debit:.2f}
  Entry Debit (mid):        ${total_debit_mid:.2f}
  Max Loss:                 ${total_debit:.2f}
  Max Profit:               ${3.00 - total_debit:.2f}

  P&L at 0% move:           ${calc_ic_pnl(SPOT, total_debit):.2f}
  P&L at +8% move:          ${calc_ic_pnl(SPOT*1.08, total_debit):.2f}
  P&L at -8% move:          ${calc_ic_pnl(SPOT*0.92, total_debit):.2f}
  P&L at +12% move:         ${calc_ic_pnl(SPOT*1.12, total_debit):.2f}
  P&L at -12% move:         ${calc_ic_pnl(SPOT*0.88, total_debit):.2f}

TOTAL BID-ASK SLIPPAGE:
  Reverse Calendar:         ${abs(total_credit_mid - total_credit):.2f}
  Iron Condor:              ${abs(total_debit_mid - total_debit):.2f}
"""

ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace')

plt.tight_layout()
plt.savefig('/home/wyatt/ibkr_guided_trade/intc_strategy_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

print(summary_text)
print("\nChart saved to: intc_strategy_comparison.png")
