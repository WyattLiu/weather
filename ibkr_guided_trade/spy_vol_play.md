# SPY Volatility Play - Feb 27 Expiry

**Opened:** 2026-02-02
**Expiry:** 2026-02-27 (25 DTE at open)
**Thesis:** Long volatility / long VIX via delta-neutral straddles + strangles

---

## Position

| Leg | Qty | Avg Cost | Book Value | Strike | Type |
|-----|-----|----------|------------|--------|------|
| SPY $697 Call | 27 | $9.69 | $26,158 | 697 | Long |
| SPY $697 Put | 25 | $9.62 | $24,054 | 697 | Long |
| SPY $696 Put | 10 | $9.58 | $9,580 | 696 | Long |
| SPY $701 Call | 8 | $7.49 | $5,993 | 701 | Long |

**Total cost basis:** ~$65,785
**Account NLV:** ~$92,000
**Deployed:** ~71%
**Cash buffer:** ~$27,000

### Structure Breakdown

1. **15x $697 straddle** (15 calls + 15 puts at $697) - core ATM position
2. **10x $697C/$696P strangle** (10 calls at $697 + 10 puts at $696) - pre-existing, slight downside tilt
3. **2x $697C/$701C** - extra upside calls from order flow
4. **8x $697P/$701C strangle** - delta-balancing strangles added to neutralize bullish bias

---

## Portfolio Greeks (at open)

| Greek | Value | Meaning |
|-------|-------|---------|
| **Delta** | +5 shares | Essentially neutral - no directional bias |
| **Gamma** | +230 | Gains +230 shares of delta per $1 SPY move |
| **Vega** | +$5,064 | Gains $5,064 per 1% rise in implied volatility |
| **Theta** | -$670/day | Daily time decay cost |

---

## The Logic

### Why This Trade

- **Long gamma** = profit from large moves in either direction. Don't need to predict direction.
- **Long vega** = profit from VIX spikes. Even if SPY doesn't move much, a fear event raises IV and the position profits.
- **Delta neutral** = no directional bet. Pure volatility play.

### Why These Strikes

- **$697 = forward ATM.** SPY spot was ~$695 but the option-implied forward (put-call parity) pointed to ~$697. This is where gamma is maximized and delta is naturally near zero.
- **$696 put** = accounts for IV skew. Puts carry higher IV than calls at the same strike. Using $696P instead of $697P for some legs captures the skew edge (cheaper per unit of delta).
- **$701 call** = delta-balancing leg. Further OTM call is cheap, contributes vega/gamma at lower cost, and offsets the excess put delta from the $696 puts.

### Why Feb 27 (25 DTE)

- **Theta/Vega sweet spot.** Shorter-dated options have more gamma but faster theta decay. Longer-dated have more vega but less gamma. 25 DTE balances both.
- **Good vega for VIX plays.** If VIX spikes 5 points in a week, the position gains ~$15k.
- **Manageable theta.** -$670/day means we need SPY to move ~$2.40/day OR IV to tick up ~0.13%/day to break even. SPY average daily range is ~$5-7, so gamma alone should cover theta on most days.

### Execution Method

- Orders placed via Wealthsimple multi-leg API (discovered by reverse-engineering frontend JS bundle).
- Placed as limit orders $0.01-$0.10 below mid to avoid overpaying.
- One at a time, creeping toward mid until fill, then dropping back $0.01 below last fill.
- Never exceeded mid price.

---

## Scenario P&L Table

| Scenario | Approx P&L | Notes |
|----------|-----------|-------|
| SPY +$5 (1 day) | +$2,875 | Gamma profit |
| SPY -$5 (1 day) | +$2,875 | Gamma profit (symmetric) |
| SPY +$10 (1 day) | +$11,500 | Big move = big gamma |
| SPY -$10 (1 day) | +$11,500 | Works both directions |
| VIX +5 pts (~3% IV) | +$15,191 | Fear event |
| VIX +10 pts (~6% IV) | +$30,382 | Major fear event |
| 1 day flat | -$670 | Theta bleed |
| 5 days flat | -$3,348 | Worst case - pinned |
| 10 days flat | -$6,700 | Extended pin pain |

---

## Exit Plan

### Profit Targets

| Trigger | Action | Expected P&L |
|---------|--------|-------------|
| **SPY moves $7+ intraday** | Close entire position at market | +$5,000 to +$8,000 |
| **SPY moves $10+ intraday** | Close entire position | +$10,000+ |
| **VIX spikes 5+ pts** | Close entire position | +$12,000 to +$18,000 |
| **VIX spikes 10+ pts** | Close 50%, let rest ride | +$15,000 on closed half |
| **Accumulated +$5,000 P&L** | Close all | Lock in ~8% return on capital |

### Theta Management (No Move)

| DTE Remaining | Action |
|---------------|--------|
| **20+ DTE** | Hold. Theta is manageable. Wait for move/VIX event. |
| **15-20 DTE** | Evaluate. If no move and down >$3k, consider rolling to next month. |
| **10-15 DTE** | Theta accelerates. Close or roll unless a move is imminent (event, FOMC, etc). |
| **<10 DTE** | Close. Theta decay is exponential. Gamma needs massive moves to compensate. |
| **<5 DTE** | Emergency close. Even at a loss. Time value is evaporating. |

### Stop Loss

| Trigger | Action |
|---------|--------|
| **P&L hits -$6,000 (~9% of NLV)** | Close all. Reassess. |
| **P&L hits -$4,000 AND no catalyst ahead** | Close all. Capital preservation. |
| **IV crushes 2%+ without SPY moving** | Close. The vol-selling thesis is broken. |

### Delta Drift Management

The position starts delta-neutral but will drift as SPY moves:
- **If delta exceeds +/-200 shares:** Rebalance by selling the overweight leg or adding a mini strangle.
- **If SPY moves $5+ and stays:** The drift is your gamma profit. Close the profitable side, keep the other or close all.
- **Check delta daily.** If it's within +/-100 shares, leave it.

### Rolling (if needed)

If at 15 DTE the position is flat or slightly down:
1. Close all current positions.
2. Re-open at the new forward ATM strike with ~25 DTE expiry.
3. Collect any remaining time value from the close.
4. Cost of the roll: typically $1-2k in theta lost + bid-ask slippage.

---

## Key Risks

1. **Pin risk:** SPY trades in a narrow range for days. Theta eats $670/day. 5 days flat = -$3,348.
2. **IV crush:** If VIX drops (calm market), vega works against you. A 2% IV drop = -$10,128.
3. **Bid-ask slippage on exit:** Multi-leg orders have wider spreads. Budget $0.05-$0.10/contract on close.
4. **Concentration:** 71% of NLV in one trade. A black swan in the wrong direction (pin + IV crush) could lose $10k+.

---

## Daily Monitoring Checklist

- [ ] Check SPY price vs $697 (how far from center)
- [ ] Check portfolio delta (rebalance if >200 shares)
- [ ] Check P&L vs exit triggers
- [ ] Check VIX level and direction
- [ ] Check DTE remaining vs theta schedule
- [ ] Check for upcoming events (FOMC, CPI, earnings) that could trigger a move
