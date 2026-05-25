# SPY Straddle Vega Scalping — Playbook & Trade Log

**Strategy:** Buy delta-neutral straddles, profit from VIX spikes and large SPY moves, close into vol events.
**Platform:** Wealthsimple (multi-leg API)
**Expiry Target:** Apr 30, 2026 (was ~45 DTE at entry)
**Account NLV:** ~$95,000

---

## Round 1 — Mar 10 (Tue): The Big Scale-In

### Setup
- SPY ~$683, forward ATM at $683 (fwd $683.14)
- Apr 30 straddle at $683: mid ~$38.50, IV ~20%
- GEX regime: deeply negative (-706M total), 0DTE contributing -489M
- Thesis: negative gamma = amplified moves = straddles print

### Execution
1. Started with $683 straddles, penny ladders below mid ($38.45, $38.46, $38.47...)
2. Scaled to 7 straddles, hit margin — cancelled bottom-of-ladder orders to free capital
3. Forward ATM shifted to $682 as SPY dipped, then $681
4. Placed mix: 3x $682 ($36.58-$36.60) + 2x $681 ($36.89-$36.90) — $681 filled instantly
5. Added more in same pattern — kept filling
6. SPY surged to $683 — all resting orders filled. Forward ATM shifted to $685
7. Added 1x $685 straddle at $35.45

### Peak Position
- **20 straddles total**: 12x $683, 6x $682, 1x $681, 1x $685
- ~$75,000 deployed (79% of account)

### Close — Mar 10 (same day)
- SPY dropped, user decided to close $683 and $685
- First tried bulk 11x $683 close — didn't fill
- **Lesson: scatter close orders** — placed 6 at mid, 5 at mid+$0.01 → all filled
- 1x $685 closed separately at mid

### Result
| Position | P&L |
|----------|-----|
| 11x $683 straddle | **+$1,210** |
| 1x $685 straddle | **+$143** |
| **Round 1 Total** | **+$1,353** |

---

## Round 2 — Mar 11 (Wed): Clean Up Remainders

### Position
- 9 straddles remaining: 6x $682 + 3x $681

### Close
- Placed close orders at mid — didn't fill
- Cancelled, re-placed at mid — still didn't fill
- **Lesson: for fast exit, price at bid + $0.02** → all filled quickly

### Result
| Position | P&L |
|----------|-----|
| 6x $682 straddle | **+$288** |
| 3x $681 straddle | **+$126** |
| **Round 2 Total** | **+$414** |

---

## Round 3 — Mar 13 (Fri): GEX-Informed Entry

### Setup
- SPY back to $675 neutral area
- Ran comprehensive GEX analysis:
  - Total GEX: -706M (deeply negative)
  - 0DTE contributing -489M of the total
  - Massive put walls at $655-$660 (97-99K OI)
  - Call walls at $680-$700
  - GEX flip at ~$670
- Negative GEX = amplified moves = good for straddles

### Execution
- Forward ATM at $672-$673
- Placed straddles at $673 first, then $672 as forward shifted
- Scaled to 9 straddles: 6x $672 + 3x $673

### Intraday Move
- SPY dropped to $669.20 — position +$360

### GEX Timeline Analysis (Key Insight)
Built GEX expiry timeline showing:
- 947M of negative GEX expires week of Mar 16 (OpEx week)
- 0DTE alone: -489M expires daily
- After OpEx: only -11M remains near spot
- **Conclusion: close before OpEx clears the gamma fuel**

### Close — Mar 13
- Closed all 9 at bid + $0.02 for fast fills
- 11 buy orders left resting (DAY orders, expired at close)

### Result
| Position | P&L |
|----------|-----|
| 9 straddles (6x $672 + 3x $673) | **+$249** |

---

## Round 3b — Mar 13: Stray Fill

- 1x $672 straddle filled from resting buy order
- Closed immediately at $40.55

| Position | P&L |
|----------|-----|
| 1x $672 straddle | **+$147** |

---

## IV-Targeted Orders (Experiment — Mar 13)

### Concept
Instead of pricing straddle orders at current mid, price them at a TARGET IV level. This way orders only fill when IV compresses (e.g., SPY rallies with calm vol).

### Execution
- Cancelled all 11 resting orders priced at current mid (~20% IV)
- Placed IV-targeted orders:
  - 4x $676 at $35.78-$35.85 → fills at ~18.5% IV
  - 4x $678 at $34.97-$35.05 → fills at ~18.0% IV
  - 3x $680 at $34.10-$34.20 → fills at ~17.5% IV
- These would only fill if SPY rallied $5-10 AND IV dropped 1.5-2.5%

### Result
- None filled (SPY didn't rally enough)
- Cancelled all before close — good experiment for future use

---

## Round 4 — Mar 13-18: The Big Hold + FOMC Exit

### Entry (Mar 13-15)
- Accumulated via `spy_scalp.py` automated loop over multiple sessions
- Script dynamically found forward ATM per expiry, placed penny ladders
- Hit margin multiple times — loop had no position cap initially
- **Lesson: added MAX_STRADDLES = 10 hard cap after over-accumulating**

### Peak Position
- **20 straddles**: 12x May15 $675 + 8x May29 $676
- ~$84,000 deployed (88% of account)
- Reached 20 straddles (should have been 10 max) — scalp loop kept filling

### The Bleed (Mar 15-17)
- SPY ranged $669-$676, VIX dropped from 23 → 17.7
- Near-spot GEX was **positive** (+28.8M) = pinning regime
- Theta ~$1,044/day with 20 straddles
- P&L bottomed at **-$3,012 (-3.4%)**
- **Lesson: near-spot GEX positive = straddles bleed. Should not have entered.**

### The Roll (Mar 17)
- SPY rallied to $676, positions at $669-671 strikes were $5-6 from ATM
- Rolled all 20 straddles using **4-leg combo orders** (sell old C+P, buy new C+P)
- Priced at natural credit + $0.02 → all 19 combos filled instantly
- Total roll credit: $26.17
- New positions: 12x May15 $675 + 8x May29 $676
- **Lesson: 4-leg combo rolls work great on WS. Price at natural + $0.02 for instant fill.**

### GEX Regime Flip (Mar 18 — FOMC day)
- GEX flipped from +34.6M near-spot (pinning) to **-999.7M** (deeply negative)
- Massive put walls at $660 (-328M), $665 (-146M), $670 (-140M)
- P&L recovered from -$3K to **+$726 (+0.9%)**

### Exit (Mar 18 — pre-FOMC)
- Decided to size down to half before FOMC binary event
- Sold May15 first (shorter DTE = more theta), then May29
- **Execution: one at a time, mid+$0.01, wait for fill, repeat**
- 10 filled quickly at mid+$0.01 (tight $0.12 spreads)
- Remaining 10: laddered sells mid+$0.01 through mid+$0.06
- 8 more filled from ladder
- Last 2 struggled — our sells WERE the ask (lesson below)
- Repriced progressively lower, eventually hit bid+$0.02
- **FOMC at 2:00 PM was a nothingburger** — no move, mild IV crush
- Dumped last straddle at bid+$0.02 post-FOMC

### Result
| Position | P&L |
|----------|-----|
| 19x straddles (sold pre-FOMC at profit) | **~+$563** |
| 1x last straddle (dumped post-FOMC) | **~-$29** |
| **Round 4 Total** | **~+$534** |

### Key Lessons from Round 4

15. **Position cap is mandatory.** Automated loops MUST have a hard cap (MAX_STRADDLES). We went from planned 4 to 20 straddles without one.

16. **Near-spot GEX matters more than total GEX.** Total GEX was -138M (negative = good) but near-spot was +28.8M (positive = pinning). The near-spot number determines your straddle's fate.

17. **Your sell orders ARE the ask.** When selling straddles on WS, your limit order becomes the best ask. The "mid" you see includes your own order. Cancel first, read the natural spread, then reprice.

18. **Size down before binary events.** Cutting from 20 to 10 before FOMC locked in profit while keeping exposure. Even though FOMC was flat, the discipline was correct.

19. **Sell one at a time for best fills.** Scatter sells at mid+$0.01, wait for fill, repeat with fresh quote. Better than placing all at once (which can stack the ask).

20. **4-leg combo rolls are cheap and instant.** Sell old C+P, buy new C+P as one combo. Price at natural credit + $0.02. All 19 rolled in <5 min.

21. **Don't fight the pin.** If near-spot GEX is positive, your straddles will bleed theta without enough realized vol. Wait for negative near-spot GEX before entering.

---

## Total P&L Summary

| Round | Date | Action | P&L |
|-------|------|--------|-----|
| 1 | Mar 10 | 12 straddles closed | +$1,353 |
| 2 | Mar 11 | 9 straddles closed | +$414 |
| 3 | Mar 13 | 9 straddles closed | +$249 |
| 3b | Mar 13 | 1 stray fill closed | +$147 |
| 4 | Mar 13-18 | 20 straddles (held 5 days, rolled, closed pre-FOMC) | +$534 |
| **Total** | | | **+$2,697** |

**Return on deployed capital:** ~1-3% per round
**Win rate:** 5/5 rounds profitable

---

## Key Lessons Learned

### Execution

1. **Penny ladder below mid for entry.** Place orders at mid-$0.01, mid-$0.02, mid-$0.03. Creep toward mid one at a time. Never exceed mid.

2. **Scatter close orders.** Don't place 1 bulk order — split into individual orders at mid and mid+$0.01. Better fill probability across market makers.

3. **For fast exit: bid + $0.02.** When you want out NOW (regime change, profit target), don't wait at mid. Hit bid+$0.02 and they fill instantly.

4. **DAY orders on WS.** All multi-leg orders are DAY only — expire at close. Must re-place next day if unfilled.

5. **Margin management.** When scaling to 15-20 straddles, margin can block new orders. Cancel bottom-of-ladder unfilled orders to free margin for higher-priority fills.

6. **Track forward ATM, not spot.** Forward ATM = strike where |call_mid - put_mid| is smallest. Uses put-call parity: F = K + C - P. This is where delta is naturally zero and gamma is maximized.

### Strategy

7. **GEX is the primary signal.** Negative GEX = amplified moves = enter straddles. Positive GEX = pinning = avoid straddles.

8. **0DTE gamma dominates intraday** but expires at close. It affects market behavior, not your longer-dated position directly. Your Apr 30 straddles are fine to hold through 0DTE expiry.

9. **OpEx clears the regime.** ~95% of negative GEX near spot can expire during monthly OpEx week. The vol regime can flip from "amplified" to "pinned" in days. Close before the gamma fuel disappears.

10. **IV-targeted orders for rallies.** Price straddle buy orders at target IV (e.g., 18%) not current mid (e.g., 20%). They only fill when SPY rallies with compressed vol — buying cheap IV.

11. **Don't hold through regime change.** The straddle makes money when realized vol > implied vol. If GEX flips positive, realized vol drops and you bleed theta. Close before the flip.

### Greeks

12. **Vega is the big lever.** VIX +3 with $8 SPY move = +$3,000-$5,000 on 10 straddles. Pure gamma from a $5 move only covers ~$200. You need vol spikes, not just moves.

13. **Theta budget.** Apr 30 straddle at ~$37: theta is ~$41/day per straddle. 10 straddles = $410/day. Budget 3-5 days of theta as your "admission cost" (~$1,200-$2,000).

14. **Delta drift is ok if small.** Forward ATM straddle starts at +0.01 to +0.04 delta. If SPY moves $3, you pick up ~$0.05 delta per straddle per dollar × 3 = +15 per straddle. 10 straddles × 15 = 150 delta — still manageable. Only rebalance if >200.

---

## GEX Analysis Method

### How to Compute
1. Pull OI for PUT and CALL chains across 6-8 expiries (0DTE through monthly)
2. Compute BSM gamma: `g = exp(-d1^2/2) / (S * sigma * sqrt(2*pi*T))`
3. Dealer GEX per strike: `(-put_OI + call_OI) * 100 * gamma * spot`
   - Dealers are SHORT puts → negative gamma contribution
   - Dealers are LONG calls → positive gamma contribution
4. Sum across all strikes and expiries

### Key Levels to Watch
- **Put walls** (high put OI at round strikes): magnetic support, but if breached → waterfall
- **Call walls** (high call OI at round strikes): resistance
- **GEX flip zone**: price where cumulative GEX changes sign
- **Near-spot GEX**: sum within $10 of spot — most relevant for daily moves

### Regime Rules
| Total GEX | Near-Spot GEX | Regime | Straddle? |
|-----------|---------------|--------|-----------|
| Negative | Negative | Amplified moves | YES |
| Negative | Positive | Mixed — pinned near spot, explosive on breakout | MAYBE |
| Positive | Positive | Pinning — dealers dampen all moves | NO |
| Positive | Negative | Unusual — check what's causing it | CHECK |

### GEX Timeline
Track how GEX changes as expiries roll off:
- 0DTE: massive gamma, resets daily
- Weekly: builds through the week, clears Friday
- Monthly OpEx: the big event — clears months of accumulated gamma
- After OpEx: totally different regime. Re-scan before trading.

---

## Execution Playbook (Step by Step)

### Pre-Trade
1. Run `straddle-scan SPY --expiry YYYY-MM-DD` → get forward ATM, straddle mid, IV, greeks
2. Run GEX analysis across 6-8 expiries → confirm negative gamma regime
3. Check for upcoming events (FOMC, CPI, OpEx) → align with catalysts
4. Size position: 10-15 straddles typical, never >80% of NLV

### Entry
1. Start with 3-4 straddles at forward ATM strike
2. Penny ladder: mid-$0.03, mid-$0.02, mid-$0.01
3. As fills come, add more at same ladder pattern
4. If forward ATM shifts (SPY moves $2+), switch to new ATM strike
5. Scale to 10-15 over 1-2 hours, mixing strikes near ATM for diversification

### Monitoring
1. Check position delta — rebalance if >200 shares equivalent
2. Check P&L vs profit targets
3. Re-run GEX if SPY moves $5+ (walls shift)
4. Watch VIX — spike = close opportunity

### Exit
| Trigger | Action | Method |
|---------|--------|--------|
| VIX spike 3+ pts | Close all | Scatter at mid & mid+$0.01 |
| SPY moves $8+ from entry | Close all | Scatter at mid |
| P&L hits +$2,000+ (10 straddles) | Close all | Scatter at mid |
| Regime flips (GEX turns positive) | Close all | Bid + $0.02 (urgent) |
| Theta budget exhausted (3-5 days flat) | Close all | Bid + $0.02 |
| Stop loss -$4,000 | Close all | Bid + $0.02 (urgent) |

### Post-Trade
1. Log P&L per strike
2. Note what worked / what didn't
3. Wait for next negative GEX regime before re-entering
