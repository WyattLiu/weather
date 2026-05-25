#!/usr/bin/env python3
"""
INTC Double Reverse Calendar P&L Analysis
Analyzes different IV crush scenarios to find the "catch"
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from datetime import datetime, timedelta

# Black-Scholes pricing
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

# Current parameters
SPOT = 56.57  # Current INTC price
r = 0.045  # Risk-free rate

# Entry parameters (at market close Thursday before earnings)
# Jan 23 = 0DTE (earnings day), Jan 30 = 7DTE
T_short_entry = 1/365  # ~1 day to expiry at entry
T_long_entry = 8/365   # ~8 days to expiry at entry

# Exit parameters (Friday morning after earnings)
T_short_exit = 0.001   # Nearly expired
T_long_exit = 7/365    # 7 days left

# IV assumptions at entry
IV_short_entry = 1.60  # 160% IV for 0DTE (earnings premium)
IV_long_entry = 1.00   # 100% IV for 7DTE

# Strikes for the double reverse calendar
PUT_STRIKE = 52
CALL_STRIKE = 59

def calculate_entry_prices():
    """Calculate entry prices for the strategy"""
    # Short-dated options (we BUY these)
    buy_put_jan23 = bs_price(SPOT, PUT_STRIKE, T_short_entry, r, IV_short_entry, 'put')
    buy_call_jan23 = bs_price(SPOT, CALL_STRIKE, T_short_entry, r, IV_short_entry, 'call')

    # Long-dated options (we SELL these)
    sell_put_jan30 = bs_price(SPOT, PUT_STRIKE, T_long_entry, r, IV_long_entry, 'put')
    sell_call_jan30 = bs_price(SPOT, CALL_STRIKE, T_long_entry, r, IV_long_entry, 'call')

    net_credit = (sell_put_jan30 - buy_put_jan23) + (sell_call_jan30 - buy_call_jan23)

    return {
        'buy_put_jan23': buy_put_jan23,
        'buy_call_jan23': buy_call_jan23,
        'sell_put_jan30': sell_put_jan30,
        'sell_call_jan30': sell_call_jan30,
        'net_credit': net_credit
    }

def calculate_exit_pnl(spot_exit, iv_short_exit, iv_long_exit, entry_prices):
    """Calculate P&L at exit for given spot and IV levels"""

    # Short-dated options at exit (nearly worthless due to time + IV crush)
    exit_put_jan23 = bs_price(spot_exit, PUT_STRIKE, T_short_exit, r, iv_short_exit, 'put')
    exit_call_jan23 = bs_price(spot_exit, CALL_STRIKE, T_short_exit, r, iv_short_exit, 'call')

    # Long-dated options at exit
    exit_put_jan30 = bs_price(spot_exit, PUT_STRIKE, T_long_exit, r, iv_long_exit, 'put')
    exit_call_jan30 = bs_price(spot_exit, CALL_STRIKE, T_long_exit, r, iv_long_exit, 'call')

    # P&L on each leg
    pnl_buy_put = exit_put_jan23 - entry_prices['buy_put_jan23']
    pnl_buy_call = exit_call_jan23 - entry_prices['buy_call_jan23']
    pnl_sell_put = entry_prices['sell_put_jan30'] - exit_put_jan30
    pnl_sell_call = entry_prices['sell_call_jan30'] - exit_call_jan30

    total_pnl = pnl_buy_put + pnl_buy_call + pnl_sell_put + pnl_sell_call

    return {
        'total_pnl': total_pnl,
        'pnl_buy_put': pnl_buy_put,
        'pnl_buy_call': pnl_buy_call,
        'pnl_sell_put': pnl_sell_put,
        'pnl_sell_call': pnl_sell_call,
        'exit_put_jan23': exit_put_jan23,
        'exit_call_jan23': exit_call_jan23,
        'exit_put_jan30': exit_put_jan30,
        'exit_call_jan30': exit_call_jan30
    }

def plot_pnl_scenarios():
    """Plot P&L across different IV crush scenarios"""

    entry = calculate_entry_prices()
    print("=" * 60)
    print("ENTRY PRICES (Theoretical)")
    print("=" * 60)
    print(f"Buy  Jan 23 $52 Put:  ${entry['buy_put_jan23']:.3f}")
    print(f"Sell Jan 30 $52 Put:  ${entry['sell_put_jan30']:.3f}")
    print(f"Buy  Jan 23 $59 Call: ${entry['buy_call_jan23']:.3f}")
    print(f"Sell Jan 30 $59 Call: ${entry['sell_call_jan30']:.3f}")
    print(f"\nNet Credit: ${entry['net_credit']:.3f}")
    print()

    # Price range: -20% to +20%
    spot_range = np.linspace(SPOT * 0.80, SPOT * 1.20, 100)
    pct_moves = (spot_range / SPOT - 1) * 100

    # Different IV crush scenarios for the 7DTE options
    # The KEY assumption: how much does the Jan 30 IV drop?
    iv_scenarios = {
        'Aggressive Crush (45%)': (0.30, 0.45),   # Jan23→30%, Jan30→45%
        'Normal Crush (55%)': (0.30, 0.55),       # Jan23→30%, Jan30→55%
        'Mild Crush (65%)': (0.30, 0.65),         # Jan23→30%, Jan30→65%
        'Minimal Crush (75%)': (0.30, 0.75),      # Jan23→30%, Jan30→75%
        'No Crush (100%)': (0.30, 1.00),          # Jan23→30%, Jan30 stays 100%
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: P&L curves for different IV scenarios
    ax1 = axes[0, 0]
    for label, (iv_short, iv_long) in iv_scenarios.items():
        pnls = [calculate_exit_pnl(s, iv_short, iv_long, entry)['total_pnl'] for s in spot_range]
        ax1.plot(pct_moves, pnls, label=f'{label}', linewidth=2)

    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
    ax1.axvline(x=-8, color='red', linestyle=':', alpha=0.5, label='±8% move')
    ax1.axvline(x=8, color='red', linestyle=':', alpha=0.5)
    ax1.axvline(x=-12, color='orange', linestyle=':', alpha=0.5, label='±12% move')
    ax1.axvline(x=12, color='orange', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Stock Move (%)')
    ax1.set_ylabel('P&L ($)')
    ax1.set_title('P&L vs Stock Move - Different IV Crush Scenarios\n(Jan 30 IV levels shown)')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-20, 20)

    # Plot 2: THE CATCH - What if IV doesn't crush on Jan 30?
    ax2 = axes[0, 1]

    # Worst case: Jan 30 IV stays high or increases
    worst_scenarios = {
        'Normal (55%)': (0.30, 0.55),
        'Mild (65%)': (0.30, 0.65),
        'No Crush (100%)': (0.30, 1.00),
        'IV SPIKE (120%)': (0.30, 1.20),  # THE CATCH!
        'IV SPIKE (140%)': (0.30, 1.40),  # Even worse!
    }

    for label, (iv_short, iv_long) in worst_scenarios.items():
        pnls = [calculate_exit_pnl(s, iv_short, iv_long, entry)['total_pnl'] for s in spot_range]
        ax2.plot(pct_moves, pnls, label=f'{label}', linewidth=2)

    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
    ax2.fill_between(pct_moves, -5, 0, alpha=0.1, color='red')
    ax2.set_xlabel('Stock Move (%)')
    ax2.set_ylabel('P&L ($)')
    ax2.set_title('THE CATCH: What if Jan 30 IV Stays High or Spikes?\n(Short options lose value = we lose money)')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-20, 20)
    ax2.set_ylim(-3, 2)

    # Plot 3: P&L breakdown by leg (Normal crush scenario)
    ax3 = axes[1, 0]
    iv_short, iv_long = 0.30, 0.55  # Normal crush

    pnl_details = []
    for s in spot_range:
        result = calculate_exit_pnl(s, iv_short, iv_long, entry)
        pnl_details.append(result)

    ax3.plot(pct_moves, [p['pnl_buy_put'] for p in pnl_details], label='Long Jan23 Put', linestyle='--')
    ax3.plot(pct_moves, [p['pnl_buy_call'] for p in pnl_details], label='Long Jan23 Call', linestyle='--')
    ax3.plot(pct_moves, [p['pnl_sell_put'] for p in pnl_details], label='Short Jan30 Put', linestyle='-.')
    ax3.plot(pct_moves, [p['pnl_sell_call'] for p in pnl_details], label='Short Jan30 Call', linestyle='-.')
    ax3.plot(pct_moves, [p['total_pnl'] for p in pnl_details], label='Total P&L', linewidth=3, color='black')

    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
    ax3.set_xlabel('Stock Move (%)')
    ax3.set_ylabel('P&L ($)')
    ax3.set_title('P&L Breakdown by Leg (Normal 55% Crush)')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-20, 20)

    # Plot 4: Sensitivity to Jan 30 exit IV
    ax4 = axes[1, 1]

    # At different stock moves, how does Jan 30 IV affect P&L?
    jan30_ivs = np.linspace(0.40, 1.40, 50)

    for move_pct, color in [(0, 'blue'), (8, 'green'), (12, 'orange'), (-12, 'red')]:
        spot_at_move = SPOT * (1 + move_pct/100)
        pnls_by_iv = [calculate_exit_pnl(spot_at_move, 0.30, iv, entry)['total_pnl'] for iv in jan30_ivs]
        ax4.plot(jan30_ivs * 100, pnls_by_iv, label=f'{move_pct:+d}% move', color=color, linewidth=2)

    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax4.axvline(x=55, color='gray', linestyle='--', linewidth=0.5, label='Expected crush (55%)')
    ax4.axvline(x=100, color='red', linestyle='--', linewidth=0.5, label='No crush (100%)')
    ax4.fill_between(jan30_ivs * 100, -5, 0, alpha=0.1, color='red')
    ax4.set_xlabel('Jan 30 Exit IV (%)')
    ax4.set_ylabel('P&L ($)')
    ax4.set_title('P&L Sensitivity to Jan 30 IV at Exit\n(Lower IV = Better for us)')
    ax4.legend(loc='upper right', fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(40, 140)
    ax4.set_ylim(-3, 2)

    plt.tight_layout()
    plt.savefig('/home/wyatt/ibkr_guided_trade/intc_reverse_cal_pnl.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Print detailed analysis
    print("=" * 60)
    print("THE CATCHES / RISKS")
    print("=" * 60)
    print()
    print("1. IV CRUSH ON JAN 30 MAY NOT HAPPEN AS EXPECTED")
    print("   - We are SHORT the Jan 30 options")
    print("   - If Jan 30 IV stays high (doesn't crush), we LOSE money")
    print("   - If Jan 30 IV INCREASES, we lose even more")
    print()

    # Calculate breakeven IV
    print("2. BREAKEVEN JAN 30 IV (at 0% stock move):")
    for iv in np.linspace(0.50, 1.20, 15):
        pnl = calculate_exit_pnl(SPOT, 0.30, iv, entry)['total_pnl']
        if abs(pnl) < 0.05:
            print(f"   Breakeven Jan 30 IV: ~{iv*100:.0f}%")
            break

    print()
    print("3. P&L AT KEY SCENARIOS:")
    print("-" * 60)
    print(f"{'Scenario':<25} {'Jan30 IV':<12} {'P&L':<10} {'Return':<10}")
    print("-" * 60)

    scenarios = [
        ("0% move, Normal crush", 0, 0.55),
        ("0% move, Mild crush", 0, 0.65),
        ("0% move, No crush", 0, 1.00),
        ("0% move, IV spike 120%", 0, 1.20),
        ("+10% move, Normal crush", 10, 0.55),
        ("+10% move, No crush", 10, 1.00),
        ("-10% move, Normal crush", -10, 0.55),
        ("-10% move, No crush", -10, 1.00),
        ("+12% move, Normal crush", 12, 0.55),
        ("-12% move, Normal crush", -12, 0.55),
    ]

    for label, move_pct, iv_long in scenarios:
        spot_exit = SPOT * (1 + move_pct/100)
        result = calculate_exit_pnl(spot_exit, 0.30, iv_long, entry)
        pnl = result['total_pnl']
        ret = (pnl / entry['net_credit']) * 100 if entry['net_credit'] > 0 else 0
        print(f"{label:<25} {iv_long*100:>6.0f}%      ${pnl:>+6.2f}    {ret:>+6.1f}%")

    print()
    print("=" * 60)
    print("SUMMARY: THE REAL RISK")
    print("=" * 60)
    print("""
The strategy profits IF the Jan 30 options lose more value than the Jan 23 options.

WHEN WE WIN:
- IV crushes significantly on Jan 30 (from 100% to 55% or less)
- This is the typical post-earnings behavior

WHEN WE LOSE:
- Jan 30 IV stays elevated (uncertainty continues)
- Jan 30 IV spikes (bad news, ongoing investigation, etc.)
- Stock pins exactly at one of our strikes (pin risk)

THE CATCH: This strategy is essentially a BET that Jan 30 IV will crush.
If there's ongoing news/uncertainty after earnings, Jan 30 IV may not drop.

HISTORICAL CONTEXT:
- Typically IV does crush after earnings
- But occasionally, ongoing drama keeps IV elevated
- Examples: accounting scandals, guidance revisions, CEO changes
""")

    print("\nChart saved to: intc_reverse_cal_pnl.png")

if __name__ == "__main__":
    plot_pnl_scenarios()
