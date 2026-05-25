#!/usr/bin/env python3
"""
INTC Reverse Calendar Trade - Post-Mortem Analysis
Trade Date: Jan 22-23, 2026
"""

print("""
================================================================================
INTC REVERSE CALENDAR - POST-MORTEM ANALYSIS
================================================================================

TRADE SUMMARY
=============
Position:     2x Reverse Calendar ($52P/$55C) 0DTE/7DTE
Entry:        Jan 22, 2026 (day before earnings)
Exit:         Jan 23, 2026 (morning after earnings)
Result:       +$154.68 profit

ENTRY EXECUTION
===============
  RC #1: Filled at $1.30 credit (target was $1.30)     ✓ Good
  RC #2: Started at $1.45, trimmed to $1.37 (filled)  ✓ Good

  Average credit: $1.335 per combo
  Total credit: ~$267 for 2 combos

  WHAT WORKED:
  - Trim algo (reduce $0.01 every 10s) worked well for entry
  - Got better than mid on RC #2 ($1.37 vs $1.35 mid)

  IMPROVEMENT:
  - Could have been more aggressive starting point
  - RC #1 at $1.30 was conservative (mid was ~$1.35)

EXIT EXECUTION
==============
  Natural debit: $0.52
  Mid debit:     $0.89 (WRONG - bad quote data)
  Our fill:      $0.69

  WHAT WENT WRONG:
  1. Started algo at mid - $0.20 = $0.69 (too high!)
  2. Max was set to natural + $0.15 = $0.67 (LOWER than start!)
  3. Algo filled immediately at $0.69 without working lower
  4. Bad quote data: Jan 23 $55 Call showed Bid = -$1.00

  MONEY LEFT ON TABLE:
  - Fill: $0.69 × 2 = $1.38 debit
  - Could have been: ~$0.55 × 2 = $1.10 debit
  - Lost: ~$28

  ROOT CAUSES:
  1. Used "mid" as reference when mid was corrupted by bad data
  2. Algo started ABOVE the max, so it filled immediately
  3. Didn't validate quote data before calculating prices
  4. 0DTE options have unreliable quotes near expiry

MARKET ANALYSIS
===============
  INTC moved: $54 → ~$47 (-13%)
  This was in our "big move" scenario

  Expected P&L at -13%: ~$240 (from our model)
  Actual P&L:           $155

  WHY THE DIFFERENCE?
  1. Exit execution cost us ~$28
  2. IV crush was less than modeled (actual ~60% vs expected 52%)
  3. Bid-ask slippage on 4-leg combo
  4. Model used theoretical prices, reality had wider spreads

IV CRUSH ANALYSIS
=================
  Pre-earnings (Jan 22):
    0DTE IV: ~210%
    7DTE IV: ~95%

  Post-earnings (Jan 23):
    0DTE: Expired at intrinsic (INTC ~$47, so $52P worth $5)
    7DTE: IV crushed to ~60-65% (we modeled 52%)

  LESSON: Our 52% post-crush estimate was slightly aggressive.
  Real crush was to ~60%, which reduced our profit.

================================================================================
IMPROVEMENTS FOR NEXT TIME
================================================================================

1. EXIT ALGO - START FROM NATURAL, NOT MID
   ----------------------------------------
   Current:  start = mid - $0.20
   Better:   start = natural - $0.05 (aggressive)

   ```python
   # Better starting point for close
   start_debit = round(natural_debit - 0.05, 2)  # Start below natural
   max_debit = round(mid_debit + 0.10, 2)        # Don't exceed mid + buffer
   ```

2. VALIDATE QUOTE DATA
   --------------------
   Before calculating mid/natural, check for bad data:

   ```python
   def is_valid_quote(ticker):
       if ticker.bid is None or ticker.ask is None:
           return False
       if ticker.bid < 0 or ticker.ask < 0:
           return False
       if ticker.bid > ticker.ask:  # Crossed market
           return False
       if ticker.ask - ticker.bid > 1.0:  # Very wide spread
           print("WARNING: Wide spread detected")
       return True
   ```

3. USE NATURAL AS PRIMARY REFERENCE FOR 0DTE
   ------------------------------------------
   0DTE options have unreliable mid prices. Use natural as baseline:

   ```python
   if is_0dte:
       reference = natural_debit
   else:
       reference = mid_debit
   ```

4. BETTER IV CRUSH MODEL
   ----------------------
   Our model: 95% → 52% (43 point crush)
   Reality:   95% → 60% (35 point crush)

   More conservative estimate:
   ```python
   # Post-earnings IV by remaining DTE
   post_earnings_iv = {
       0: 0.50,   # Expires at intrinsic
       7: 0.58,   # More conservative (was 0.52)
       14: 0.55,
       21: 0.52,
   }
   ```

5. POSITION SIZING
   ----------------
   We used ~$4,000 margin for 2 combos = ~$2,000 per combo
   Profit: $155 = 3.9% return on margin

   With better exit: ~$183 = 4.6% return

   CONSIDER: Could have done 3-4 combos with available margin
   But 2 was prudent for first trade.

6. ENTRY TIMING
   -------------
   We entered mid-day on earnings day.
   BETTER: Enter at close (3:50-4:00 PM) when IV is highest
   This maximizes the credit received.

7. EXIT TIMING
   ------------
   We exited at 9:35 AM.
   IV crush is usually complete by 10:00-10:30 AM.
   Our timing was good - captured the crush.

================================================================================
UPDATED CLOSE ALGO
================================================================================

```python
def close_rc_with_algo(ib, combo, qty, natural_debit, mid_debit):
    '''
    Close reverse calendar with ascending price algo.
    Starts below natural and works up to mid.
    '''

    # Validate - natural should be <= mid
    if natural_debit > mid_debit:
        print(f"WARNING: natural ({natural_debit}) > mid ({mid_debit})")
        print("Quote data may be corrupted. Using natural as reference.")
        mid_debit = natural_debit + 0.30

    # Start 5 cents below natural (aggressive)
    start_debit = max(0.01, round(natural_debit - 0.05, 2))

    # Max is mid + 10 cents (don't overpay)
    max_debit = round(mid_debit + 0.10, 2)

    print(f"Natural: ${natural_debit:.2f}")
    print(f"Mid: ${mid_debit:.2f}")
    print(f"Starting at ${start_debit:.2f}, max ${max_debit:.2f}")

    current_debit = start_debit
    order = LimitOrder('BUY', qty, current_debit)
    order.account = DEFAULT_ACCOUNT
    order.tif = 'DAY'

    trade = ib.placeOrder(combo, order)
    ib.sleep(3)

    while current_debit < max_debit:
        if trade.orderStatus.status == 'Filled':
            print(f"FILLED at ${current_debit:.2f}!")
            return current_debit

        current_debit += 0.01
        current_debit = round(current_debit, 2)

        print(f"Raising to ${current_debit:.2f}...")
        trade.order.lmtPrice = current_debit
        ib.placeOrder(combo, trade.order)
        ib.sleep(10)  # Wait 10 seconds between raises

    return None  # Not filled at max
```

================================================================================
SCORE CARD
================================================================================

ENTRY:                 8/10 (good fills, trim algo worked)
EXIT:                  5/10 (filled too high, bad data handling)
POSITION SIZING:       7/10 (conservative but appropriate)
IV MODEL:              6/10 (slightly aggressive crush estimate)
TIMING:                8/10 (good entry/exit timing)
RISK MANAGEMENT:       9/10 (defined risk, profit target hit)

OVERALL:               7/10

PROFIT:                +$154.68
COULD HAVE BEEN:       ~$180-200 with better exit
IMPROVEMENT POTENTIAL: +$25-45

================================================================================
KEY TAKEAWAYS
================================================================================

1. REVERSE CALENDAR WORKS for earnings IV crush plays
2. 0DTE/7DTE combo is better than 7DTE/14DTE (more IV differential)
3. EXIT ALGO needs to start from NATURAL, not mid
4. VALIDATE QUOTE DATA especially for 0DTE options
5. IV CRUSH model should be slightly more conservative
6. TRIM/RAISE ALGO is effective - use it for both entry and exit

The strategy is sound. Execution can be improved.
Next earnings trade: Apply these lessons!
""")
