# Bull Put Spread Portfolio Plan — Feb 2026

## Scan Summary
- Scanned all 503 S&P 500 stocks via IBKR
- Phase 1: Historical RV from 1-year daily bars
- Phase 2: ATM IV at ~35 DTE, computed IV-RV selling edge
- Phase 3: Full put chain with OI, bid/ask, spread analysis
- All spreads target **Mar 20 (35 DTE)** for unified expiry management

## Key Metric: IV-RV Edge
Positive edge = options are overpriced vs realized movement = selling has positive expected value.

---

## Tier 1 — High Edge + Good Liquidity (Execute Confidently)

### 1. DASH 140/135P — Edge +34.7% (BEST EDGE IN UNIVERSE)
- **Spot**: $160.42 | **IV**: 75.4% | **RV30**: 40.7%
- **Spread**: Sell 140P / Buy 135P ($5 wide)
- **Credit**: $1.58 mid / $0.70 natural
- **Risk**: $3.42 | **RoR**: 46.0% | **Ann**: 480%/y
- **OTM**: 12.7% | **Delta**: -0.228
- **OI**: 254/232 (low but nonzero — fillable at mid with patience)
- **Supports**: Recent IPO-era stock, $130 area as support
- **Drop 20d**: -25.1% (already crashed — IV elevated post-drop)
- **Verdict**: MASSIVE edge. OI is the concern but $1.58 mid on a $5 spread is extraordinary. Place at mid, may take time to fill. The 34.7% edge compensates for any slippage.
- **Alternative**: 145/140P for similar RoR but 9.6% OTM

### 2. AVGO 290/280P — Edge +17.7%
- **Spot**: $326.81 | **IV**: 62.9% | **RV30**: 41.8%
- **Spread**: Sell 290P / Buy 280P ($10 wide)
- **Credit**: $2.33 mid / $2.00 natural
- **Risk**: $7.67 | **RoR**: 30.4% | **Ann**: 316%/y
- **OTM**: 11.3% | **Delta**: -0.234
- **OI**: 5,611 / 8,331 (Grade A liquidity)
- **Supports**: $295 (9.6%), $287 (12.1%), $273 (16.4%)
- **RV declining**: 60d 51% → 30d 42% (bullish for selling)
- **Warning**: 50% breach rate at -12% historically. AVGO swings.
- **Verdict**: Second-best edge with excellent liquidity. Short leg sits below $295 support. Natural credit of $2.00 is immediately fillable.

### 3. ORCL 140/135P — Edge +12.3%
- **Spot**: $161.11 | **IV**: 71.9% | **RV30**: 56.2%
- **Spread**: Sell 140P / Buy 135P ($5 wide)
- **Credit**: $1.15 mid / $0.90 natural
- **Risk**: $3.85 | **RoR**: 29.9% | **Ann**: 312%/y
- **OTM**: 13.1% | **Delta**: -0.224
- **OI**: 17,818 / 3,046 (EXCEPTIONAL liquidity on short leg)
- **Supports**: $154 (4.2%), $137 (14.5%)
- **Post-crash**: ORCL dropped 33% in 20d, now stabilizing. Post-crash IV is classic selling opportunity.
- **Verdict**: Safest trade. 13% OTM, below two support levels, monster OI. Natural fill at $0.90.
- **Alternative**: 145/140P for 10% OTM, $1.48 cr, 42% RoR, 4.6k/17.8k OI

### 4. WBD 27/24P — Edge +18.1%
- **Spot**: $28.11 | **IV**: 35.0% | **RV30**: 17.0%
- **Spread**: Sell 27P / Buy 24P ($3 wide)
- **Credit**: $0.58 mid / $0.43 natural
- **Risk**: $2.42 | **RoR**: 23.7% | **Ann**: 247%/y
- **OTM**: 3.9% | **Delta**: -0.329
- **OI**: 160,958 / 38,802 (INSANE liquidity — best in universe)
- **Warning**: Only 3.9% OTM. WBD is a cheap stock so $3 wide is all you get.
- **Verdict**: Incredible liquidity and edge, but too close to the money. Only for aggressive allocations. Could get assigned easily on a bad day.

---

## Tier 2 — Good Edge, Moderate Liquidity (Place at Mid, Patient Fill)

### 5. TGT 105/100P — Edge +13.8%
- **Spot**: $116.05 | **IV**: 49.3% | **RV30**: 33.6%
- **Spread**: Sell 105P / Buy 100P ($5 wide)
- **Credit**: $1.05 mid / $0.82 natural
- **Risk**: $3.96 | **RoR**: 26.5% | **Ann**: 276%/y
- **OTM**: 9.5% | **Delta**: -0.225
- **OI**: 2,031 / 3,093 (adequate)
- **Supports**: $102 (12.3%), $98 (15.2%), $100 round number
- **Warning**: 100% breach rate at -5% to -12% historically. Long downtrend stock.
- **Verdict**: Good edge but TGT has been weak. The 9.5% OTM buffer helps, and $100/$102 are real support. RV is declining (bullish).

### 6. DG 145/140P — Edge +17.9%
- **Spot**: $151.07 | **IV**: 48.4% | **RV30**: 29.4%
- **Spread**: Sell 145P / Buy 140P ($5 wide)
- **Credit**: $1.52 mid / $1.10 natural
- **Risk**: $3.48 | **RoR**: 43.9% | **Ann**: 458%/y
- **OTM**: 4.0% | **Delta**: -0.333
- **OI**: 6,325 / 542 (short leg excellent, long leg thin)
- **Warning**: Only 4% OTM. Similar to WBD problem.
- **Alternative**: 135/130P for 10.6% OTM, $0.87 cr, 21% RoR, 426/808 OI
- **Verdict**: Good edge + OI on short leg, but long leg is thin. Use 140/135P (7.3% OTM) or 135/130P (10.6%) for more safety.

### 7. ADBE 250/240P — Edge +13.0%
- **Spot**: $264.21 | **IV**: 52.6% | **RV30**: 37.9%
- **Spread**: Sell 250P / Buy 240P ($10 wide)
- **Credit**: $3.18 mid / $2.70 natural
- **Risk**: $6.82 | **RoR**: 46.5% | **Ann**: 485%/y
- **OTM**: 5.4% | **Delta**: ~-0.27
- **OI**: 2,086 / 1,358 (decent)
- **Verdict**: Highest RoR (46.5%) with good edge. Only 5.4% OTM though. Post-earnings drop (-14%).

### 8. NFLX 74/71P — Edge +11.2%  *(note: NFLX had 10:1 split)*
- **Spot**: $76.55 | **IV**: 37.5% | **RV30**: 26.3%
- **Spread**: Sell 74P / Buy 71P ($3 wide)
- **Credit**: $0.94 mid / $0.86 natural
- **Risk**: $2.06 | **RoR**: 45.3% | **Ann**: 472%/y
- **OTM**: 3.1% | **Delta**: -0.357
- **OI**: 6,071 / 16,428 (excellent)
- **Warning**: Only 3.1% OTM — too close.
- **Alternative**: 70/65P for 8.4% OTM, $0.73 cr, 17% RoR
- **Verdict**: Great liquidity and edge but needs wider strikes. Post-split makes $3 spreads tight.

---

## Recommended Portfolio (Mar 20 Expiry)

### Conservative (4 positions, ~$1000 max risk each):

| # | Trade | Qty | Credit | Max Risk | RoR | Edge | Priority |
|---|-------|-----|--------|----------|-----|------|----------|
| 1 | ORCL 140/135P | 2x | $2.30 | $7.70 | 29.9% | +12.3% | EXECUTE FIRST |
| 2 | AVGO 290/280P | 1x | $2.33 | $7.67 | 30.4% | +17.7% | EXECUTE |
| 3 | DASH 140/135P | 2x | $3.16 | $6.84 | 46.0% | +34.7% | PLACE AT MID |
| 4 | TGT 105/100P | 2x | $2.10 | $7.90 | 26.5% | +13.8% | EXECUTE |

**Total portfolio**: ~$30 credit, ~$30 max risk, ~100% return on risk if all expire OTM.

### Execution Notes:
- **ORCL, AVGO**: Natural fills available immediately (tight spreads)
- **DASH**: Place at mid ($1.58), may need patience. If no fill in 1hr, lower to $1.30
- **TGT**: Place at mid ($1.05), tight enough for quick fill
- All are **GTC** orders — no need for active management until close to expiry
- Single expiry (Mar 20) means one monitoring date

### Risk Management:
- If any stock drops >5% in a day: evaluate, but don't panic — the IV edge protects you
- Close at 50% profit if reached early (e.g., credit drops to $0.50-$0.60)
- Close at 14 DTE if not at 50% profit yet (avoid gamma risk)
- Max loss on entire portfolio: ~$30 (unlikely — requires ALL 4 to breach)

---

## Tier 3 — Moderate Edge (8-12%), Not Deep-Scanned (Phase 2 Only)

These stocks had positive IV-RV edge in Phase 2 but were not deep-scanned (no OI/spread data yet). Run `python scan_spreads.py --ticker SYMBOL` for full spread analysis before placing.

### Worth Deep-Scanning (Good Edge + Reasonable Stock)

| # | Ticker | Spot | IV | RV30 | Edge | Drop 20d | Why Interesting |
|---|--------|------|----|------|------|----------|-----------------|
| 16 | GPN | $68.93 | 51.2% | 39.4% | +11.8% | -9.9% | Mid-cap payments stock, good edge, accessible price |
| 17 | DPZ | $380.01 | 36.1% | 24.5% | +11.6% | -9.5% | Blue-chip pizza, low RV, post-drop IV elevated |
| 20 | LYV | $158.65 | 43.2% | 32.6% | +10.6% | -0.8% | Live Nation, stable, good IV/RV spread |
| 21 | CRL | $162.55 | 61.6% | 51.5% | +10.1% | -28.8% | Crashed hard — classic post-crash IV selling |
| 22 | POOL | $271.21 | 36.7% | 26.6% | +10.1% | -2.5% | Pool Corp, steady business, 10% edge |
| 24 | GRMN | $214.15 | 41.8% | 31.8% | +10.0% | -0.4% | Stable stock, low drop, good edge |
| 32 | CRM | $191.10 | 53.5% | 45.0% | +8.5% | -18.9% | Salesforce post-crash, high IV, liquid name |
| 34 | TAP | $53.79 | 36.0% | 27.5% | +8.4% | -1.9% | Molson Coors, defensive stock |
| 36 | VRSK | $181.00 | 47.8% | 39.4% | +8.4% | -19.0% | Verisk, post-drop, data analytics |

### Probably Skip (Low Price = Thin Spreads, or Low Edge)

| # | Ticker | Spot | Edge | Why Skip |
|---|--------|------|------|----------|
| 19 | EME | $803.10 | +10.7% | Very high stock price — $10 wide spread = small % OTM |
| 23 | FIX | $1346.66 | +10.0% | Ultra-high price, thin options likely |
| 25 | VTRS | $15.88 | +9.5% | Too cheap — spreads will be $1-2 wide max |
| 26 | ALLE | $179.24 | +9.5% | Low IV (28.5%), credits will be tiny |
| 27 | KVUE | $18.76 | +9.4% | Too cheap — Kenvue |
| 28 | NDSN | $297.53 | +9.4% | Low IV (28.5%), niche industrial |
| 29 | LNT | $71.07 | +9.1% | Utility, very low IV (24.9%) |
| 30 | L | $109.37 | +9.1% | Insurance, low IV (22.4%) |
| 31 | HSIC | $79.22 | +9.0% | Henry Schein, likely thin OI |
| 33 | AZO | $3880.47 | +8.5% | Auto Zone ultra-high price, illiquid options |
| 35 | KDP | $29.82 | +8.4% | Cheap stock, thin spreads |
| 37 | NI | $46.22 | +8.0% | Utility, low IV (22.0%) |
| 38 | RSG | $222.97 | +8.0% | Waste management, low IV (24.0%) |
| 39 | HPE | $22.90 | +8.0% | Too cheap, $2-3 wide max |
| 40 | CTAS | $194.73 | +7.9% | Low IV (24.4%), credits tiny |
| 41 | TJX | $155.55 | +7.9% | Low IV (26.8%) |
| 42 | RL | $371.36 | +7.7% | Niche luxury, likely thin OI |
| 43 | FE | $49.80 | +7.3% | Utility, low IV (22.6%) |
| 44 | A | $126.41 | +7.2% | Agilent, moderate but -12.7% drop |
| 45 | SJM | $108.97 | +7.2% | Smucker's, defensive, low IV |
| 46 | MOS | $29.70 | +7.2% | Fertilizer, cheap, high RV |
| 47 | WEC | $115.63 | +7.1% | Utility |
| 48 | FDS | $202.67 | +7.0% | FactSet, -30.3% crash but only 7% edge |
| 49 | TKO | $204.67 | +6.7% | WWE/UFC, likely thin OI |
| 50 | HST | $20.00 | +6.4% | Host Hotels, too cheap |

---

## Stocks Skipped (Low OI / Poor Risk-Reward from Deep Scan)

| Stock | Edge | Why Skipped |
|-------|------|-------------|
| AXON +16.7% | OI 79/70 — untradeable |
| PODD +17.5% | OI 27/21 — untradeable |
| COO +21.8% | OI 133/273 — thin, only $5 spread at 4.8% OTM |
| WSM +14.1% | OI 37/118 — untradeable |
| ULTA +14.7% | OI 17/67 — untradeable |
| KEYS +12.9% | OI 12/43 — untradeable |
| MNST +15.9% | OI 1400/647 — borderline, only $0.30 credit on $2 spread |

---

## Existing Position
- **GOOGL**: Already placed (from earlier session)

## Data Source
- IBKR API (ib_insync), scanned Feb 13 2026 during market hours
- Scanner script: `scan_spreads.py`
- Phase 2 covered 50 stocks with positive edge out of 503 scanned
