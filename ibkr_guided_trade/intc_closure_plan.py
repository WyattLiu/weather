#!/usr/bin/env python3
"""
INTC Reverse Calendar - P&L Expectations & Closure Plan
Position: 2x ($52P/$55C) 0DTE/7DTE Reverse Calendar
"""

import numpy as np
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

# Position details
SPOT = 54.00
NUM_CONTRACTS = 2
r = 0.045

# Entry costs (from IBKR)
ENTRY = {
    'buy_jan23_52p': 1.5255,  # Avg cost
    'sell_jan30_52p': 2.1695,
    'buy_jan23_55c': 1.6605,
    'sell_jan30_55c': 2.3295,
}

TOTAL_CREDIT = ((ENTRY['sell_jan30_52p'] - ENTRY['buy_jan23_52p']) +
                (ENTRY['sell_jan30_55c'] - ENTRY['buy_jan23_55c'])) * NUM_CONTRACTS

# Post-earnings IV scenarios
IV_SCENARIOS = {
    'Aggressive Crush (45%)': 0.45,
    'Expected Crush (52%)': 0.52,
    'Mild Crush (60%)': 0.60,
    'No Crush (90%)': 0.90,
}

def calc_exit_value(spot_exit, iv_post):
    """Calculate exit value of position"""
    T_remaining = 7/365  # 7 days left on Jan 30

    # Jan 23 options expire at intrinsic (0DTE)
    jan23_52p = max(0, 52 - spot_exit)
    jan23_55c = max(0, spot_exit - 55)

    # Jan 30 options valued with post-earnings IV
    jan30_52p = bs_price(spot_exit, 52, T_remaining, r, iv_post, 'put')
    jan30_55c = bs_price(spot_exit, 55, T_remaining, r, iv_post, 'call')

    # P&L per contract
    pnl_put = (jan23_52p - ENTRY['buy_jan23_52p']) + (ENTRY['sell_jan30_52p'] - jan30_52p)
    pnl_call = (jan23_55c - ENTRY['buy_jan23_55c']) + (ENTRY['sell_jan30_55c'] - jan30_55c)

    return (pnl_put + pnl_call) * NUM_CONTRACTS

print("=" * 80)
print("INTC REVERSE CALENDAR - CLOSURE PLAN")
print("=" * 80)
print(f"\nPosition: {NUM_CONTRACTS}x ($52P/$55C) Jan 23/Jan 30 Reverse Calendar")
print(f"Entry Credit: ${TOTAL_CREDIT:.2f} received")
print(f"Current Spot: ${SPOT:.2f}")

print("\n" + "=" * 80)
print("P&L EXPECTATIONS BY INTC PRICE (Tomorrow Morning)")
print("=" * 80)

# Price scenarios
prices = [43, 45, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 62, 65]
moves = [(p / SPOT - 1) * 100 for p in prices]

print(f"\n{'Price':>7} {'Move':>7} {'IV 45%':>10} {'IV 52%':>10} {'IV 60%':>10} {'IV 90%':>10}")
print("-" * 60)

for price, move in zip(prices, moves):
    pnls = [calc_exit_value(price, iv) for iv in IV_SCENARIOS.values()]
    print(f"${price:>6.2f} {move:>+6.1f}% " + " ".join(f"${p:>+8.2f}" for p in pnls))

print("\n" + "=" * 80)
print("KEY PRICE LEVELS")
print("=" * 80)

# Calculate key levels
for iv_name, iv in [('Expected (52%)', 0.52)]:
    print(f"\nWith {iv_name} IV crush:")

    # Find breakevens
    for price in np.linspace(40, 70, 300):
        pnl = calc_exit_value(price, iv)
        if abs(pnl) < 5:  # Near breakeven
            move = (price / SPOT - 1) * 100
            print(f"  Breakeven: ~${price:.2f} ({move:+.1f}%)")
            break

    # Max loss zone
    max_loss = float('inf')
    max_loss_price = SPOT
    for price in np.linspace(50, 57, 100):
        pnl = calc_exit_value(price, iv)
        if pnl < max_loss:
            max_loss = pnl
            max_loss_price = price
    print(f"  Max Loss: ${max_loss:.2f} at ${max_loss_price:.2f}")

    # Key P&L points
    for label, price in [('-20%', SPOT*0.80), ('-12%', SPOT*0.88), ('-8%', SPOT*0.92),
                         ('Flat', SPOT), ('+8%', SPOT*1.08), ('+12%', SPOT*1.12), ('+20%', SPOT*1.20)]:
        pnl = calc_exit_value(price, iv)
        print(f"  {label}: ${price:.2f} → P&L ${pnl:+.2f}")

print("\n" + "=" * 80)
print("CLOSURE PLAN")
print("=" * 80)

print("""
TIMELINE:
  - Earnings: Tonight after close (Jan 22)
  - Exit Window: Tomorrow (Jan 23) 9:30 AM - 10:30 AM ET
  - Jan 23 options expire: End of day Jan 23

ACTION PLAN BY SCENARIO:

1. BIG MOVE (±10% or more) - BEST CASE
   Action: Close Jan 30 legs immediately at open
   Expected P&L: +$150 to +$300
   How:
     - Buy back Jan 30 $52 Put
     - Buy back Jan 30 $55 Call
     - Jan 23 options will be deep ITM (one side) or worthless

2. MODERATE MOVE (±5-10%) - GOOD CASE
   Action: Close Jan 30 legs within first hour
   Expected P&L: +$50 to +$150
   How: Same as above, monitor IV crush

3. SMALL MOVE (±0-5%) - WORST CASE
   Action: Close everything ASAP to limit losses
   Expected P&L: -$50 to -$150
   How: Close all 4 legs as combo or individually
   Priority: Get out before theta decay accelerates

4. STOCK PINS AT $52-$55 - MAX LOSS
   Action: Close immediately at any price
   Expected P&L: -$100 to -$200
   Note: This is unlikely but be prepared

CLOSING COMMANDS:
================

Option A: Close Jan 30 legs only (if Jan 23 expire worthless)
  python ibkr_trading.py spread INTC 20260130 P 52 52 --close --qty 2
  python ibkr_trading.py spread INTC 20260130 C 55 55 --close --qty 2

Option B: Close as vertical spreads
  # Close put calendar
  python ibkr_trading.py spread INTC 20260130 P 52 XX --close --qty 2
  # Close call calendar
  python ibkr_trading.py spread INTC 20260130 C 55 XX --close --qty 2

Option C: Manual close each leg
  Check prices and close individually through IBKR TWS

PROFIT TARGETS:
  - Take profit at +$150 or more
  - Cut loss at -$150 (don't let it get to max loss)
  - If unsure, close by 10:30 AM to avoid theta decay
""")

print("=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"""
Position: {NUM_CONTRACTS}x Reverse Calendar ($52P/$55C)
Credit Received: ${TOTAL_CREDIT:.2f}
Max Expected Loss: ~${calc_exit_value(53.5, 0.52):.2f} (if pins at $53.50)
Max Expected Profit: ~${calc_exit_value(SPOT*0.80, 0.52):.2f} (20% move)

Expected P&L at likely scenarios (52% IV crush):
  ±8% move:  ${(calc_exit_value(SPOT*0.92, 0.52) + calc_exit_value(SPOT*1.08, 0.52))/2:+.2f} avg
  ±12% move: ${(calc_exit_value(SPOT*0.88, 0.52) + calc_exit_value(SPOT*1.12, 0.52))/2:+.2f} avg
  ±20% move: ${(calc_exit_value(SPOT*0.80, 0.52) + calc_exit_value(SPOT*1.20, 0.52))/2:+.2f} avg

REMEMBER: Close by 10:30 AM tomorrow to capture IV crush!
""")
